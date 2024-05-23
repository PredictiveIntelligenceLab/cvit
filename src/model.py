from typing import Callable

import einops
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jax.nn.initializers import normal, xavier_uniform


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
            ),
        0
        )


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing='ij')  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return jnp.expand_dims(pos_embed, 0)


class PatchEmbed(nn.Module):
    patch_size: tuple = (1, 16, 16)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape

        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            kernel_init=self.kernel_init,
            name="proj",
            )(x)

        num_patches = (t // self.patch_size[0], h // self.patch_size[1], w // self.patch_size[2])

        x = jnp.reshape(x, (b, num_patches[0], num_patches[1] * num_patches[2], self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=self.layer_norm_eps)(x)
        return x


class MlpBlock(nn.Module):
    dim: int = 256
    out_dim: int = 256
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(
            self.dim, kernel_init=self.kernel_init
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dense(
            self.out_dim, kernel_init=self.kernel_init
        )(x)
        return x


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                            qkv_features=self.emb_dim)(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                            qkv_features=self.emb_dim)(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class TimeAggregation(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, T, S, D) --> (B, T', S, D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (T', D)
                             )

        latents = repeat(latents, 't d -> b s t d', b=x.shape[0], s=x.shape[2])  # (B, T', S, D)
        x = rearrange(x, 'b t s d -> b s t d')  # (B, S, T, D)

        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)
        latents = rearrange(latents, 'b s t d -> b t s d')  # (B, T', S, D)
        return latents


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.num_layers):
            y = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            y = nn.gelu(y)
            x = x + y
            x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        x = nn.Dense(features=self.out_dim)(x)
        return x


t_emb_init = get_1d_sincos_pos_embed
s_emb_init = get_2d_sincos_pos_embed


class Encoder(nn.Module):
    patch_size: int = (1, 16, 16)
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: int = 1
    out_dim: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):

        b, t, h, w, c = x.shape

        x = PatchEmbed(self.patch_size, self.emb_dim)(x)

        t_emb = self.variable(
            "pos_emb",
            "enc_t_emb",
            t_emb_init,
            self.emb_dim,
            t // self.patch_size[0],
            )

        s_emb = self.variable(
            "pos_emb",
            "enc_s_emb",
            s_emb_init,
            self.emb_dim,
            (h // self.patch_size[1], w // self.patch_size[2])
            )

        x = x + t_emb.value[:, :, jnp.newaxis, :] + s_emb.value[:, jnp.newaxis, :, :]

        x = TimeAggregation(num_latents=1,
                            emb_dim=self.emb_dim,
                            depth=2,
                            num_heads=self.num_heads,
                            mlp_ratio=self.mlp_ratio,
                            layer_norm_eps=self.layer_norm_eps
                            )(x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = rearrange(x, 'b t s d -> b (t s) d')

        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads,
                self.emb_dim,
                self.mlp_ratio,
                self.layer_norm_eps
                )(x)

        return x


class Vit(nn.Module):
    patch_size: tuple = (1, 16, 16)
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: int = 1
    num_mlp_layers: int = 1
    out_dim: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        x = Encoder(
                self.patch_size,
                self.emb_dim,
                self.depth,
                self.num_heads,
                self.mlp_ratio,
                self.layer_norm_eps
                )(x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        x = Mlp(num_layers=self.num_mlp_layers,
                hidden_dim=self.emb_dim,
                out_dim=self.patch_size[1] * self.patch_size[2] * self.out_dim,
                layer_norm_eps=self.layer_norm_eps)(x)
        return x


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y

class CVit(nn.Module):
    patch_size: tuple = (1, 16, 16)
    grid_size: tuple = (128, 128)
    latent_dim: int = 256
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    dec_emb_dim: int = 256
    dec_num_heads: int = 8
    dec_depth: int = 1
    num_mlp_layers: int = 1
    mlp_ratio: int = 1
    out_dim: int = 1
    eps: float = 1e5
    layer_norm_eps: float = 1e-5
    embedding_type: str = 'grid'

    def setup(self):

        if self.embedding_type == 'grid':
            # Create grid and latents
            n_x, n_y = self.grid_size[0], self.grid_size[1]

            x = jnp.linspace(0, 1, n_x)
            y = jnp.linspace(0, 1, n_y)
            xx, yy = jnp.meshgrid(x, y, indexing='ij')

            self.grid = jnp.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
            self.latents = self.param(
                "latents", normal(), (n_x * n_y, self.latent_dim)
                )

    @nn.compact
    def __call__(self, x, coords):

        b, t, h, w, c = x.shape

        if self.embedding_type == 'grid':
            #
            d2 = ((coords[:, jnp.newaxis, :] - self.grid[jnp.newaxis, :, :]) ** 2).sum(axis=2)
            w = jnp.exp(-self.eps * d2) / jnp.exp(-self.eps * d2).sum(axis=1, keepdims=True)

            coords = jnp.einsum('ic,pi->pc', self.latents, w)
            coords = nn.Dense(self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

        elif self.embedding_type == 'fourier':
            coords = FourierEmbs(embed_scale=2 * jnp.pi, embed_dim=self.dec_emb_dim)(coords)

        elif self.embedding_type == 'mlp':
            coords = MlpBlock(self.dec_emb_dim, self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

        coords = einops.repeat(coords, 'n d -> b n d', b=b)

        x = Encoder(
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps
            )(x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.Dense(self.dec_emb_dim)(x)

        for _ in range(self.dec_depth):
            x = CrossAttnBlock(num_heads=self.dec_num_heads,
                               emb_dim=self.dec_emb_dim,
                               mlp_ratio=self.mlp_ratio,
                               layer_norm_eps=self.layer_norm_eps)(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = Mlp(num_layers=self.num_mlp_layers,
                hidden_dim=self.dec_emb_dim,
                out_dim=self.out_dim,
                layer_norm_eps=self.layer_norm_eps)(x)

        return x