import jax
import optax
import itertools
import einops
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import torch.utils.data as data
import pickle

from typing import Any, Callable, Sequence, Optional, Union, Dict
from functools import partial
from tqdm.auto import trange


def fourier_encoding(x, freqs):
    x = jnp.einsum("i,ij->j", x, freqs)
    x = jnp.hstack([jnp.cos(2.0 * jnp.pi * x), jnp.sin(2.0 * jnp.pi * x)])
    return x


def periodic_encoding(x, k=1, L=1.0):
    dim = x.shape[0]
    freqs = jnp.array(jnp.meshgrid(*(jnp.arange(k) for _ in range(dim)))).reshape(
        dim, -1
    )
    x = fourier_encoding(x, freqs / L)
    return x


class MLP(nn.Module):
    num_layers: int = 2
    hidden_dim: int = 64
    output_dim: int = 1
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class SplitDecoder(nn.Module):
    num_layers: int = 8
    hidden_dim: int = 256
    output_dim: int = 2
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, beta, y):
        beta = jnp.split(beta, self.num_layers)
        y = periodic_encoding(y, k=8, L=2.0)
        for i in range(self.num_layers):
            y = nn.Dense(self.hidden_dim)(jnp.concatenate([y, beta[i]]))
            y = self.activation(y)
        outputs = nn.Dense(self.output_dim)(y)
        return outputs


class CNF(nn.Module):
    @nn.compact
    def __call__(self, u, y):
        z = self.conditioner(u, y)
        s = self.basefield(z, y)
        return s


class Nomad(CNF):
    encoder_num_layers: int = 2
    encoder_hidden_dim: int = 64
    latent_dim: int = 64
    decoder_num_layers: int = 2
    decoder_hidden_dim: int = 64
    output_channels: int = 2
    activation: Callable = nn.gelu

    def setup(self):
        self.encoder = MLP(
            self.encoder_num_layers,
            self.encoder_hidden_dim,
            self.latent_dim,
            self.activation,
        )

        self.decoder = SplitDecoder(
            self.decoder_num_layers,
            self.decoder_hidden_dim,
            self.output_channels,
            self.activation,
        )

    def conditioner(self, u, y):
        z_global = self.encoder(u)
        return z_global

    def basefield(self, z, y):
        s = self.decoder(z, y)
        return s.flatten()


class DeepONet(CNF):
    encoder_num_layers: int = 8
    decoder_num_layers: int = 8
    latent_dim: int = 64
    activation: Callable = nn.gelu

    def setup(self):
        self.encoder = MLP(
            self.encoder_num_layers,
            self.latent_dim,
            self.latent_dim,
            self.activation,
        )

        self.decoder = MLP(
            self.decoder_num_layers,
            self.latent_dim,
            self.latent_dim,
            self.activation,
        )

    def conditioner(self, u, y):
        z_global = self.encoder(u)
        return z_global

    def basefield(self, z, y):
        b = z
        t = self.decoder(y)
        s = jnp.sum(b * t, axis=-1)
        return s


class FNO1dBlock(CNF):
    in_channels: int = 1
    out_channels: int = 1
    num_modes: int = 8

    def setup(self):
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.K = self.param(
            "global_kernel",
            jax.nn.initializers.normal(scale, jnp.float32),
            (2, self.in_channels, self.out_channels, self.num_modes),
        )
        self.W = self.param(
            "local_kernel",
            jax.nn.initializers.glorot_normal(),
            (self.out_channels, self.out_channels),
        )

    def complex_mul1d(self, inputs, weights):
        # (modes, in_channels), (in_channels, out_channels, modes) -> (modes, out_channels)
        return jnp.einsum("xi,iox->xo", inputs, weights)

    def conditioner(self, u, y):
        # Global conditioning
        u_hat = jnp.fft.rfft(u, axis=0)
        z_global = jnp.zeros_like(u_hat)
        z_global = z_global.at[: self.num_modes, :].set(
            self.complex_mul1d(
                u_hat[: self.num_modes, :], self.K[0, ...] + 1j * self.K[1, ...]
            )
        )
        # Local conditioning
        length = u.shape[0]
        idx = jnp.int32(y * length)[0][0]  # (assumes regular input domain)
        z_local = jnp.dot(self.W, u[idx, : self.out_channels])
        return z_global, z_local

    def basefield(self, z, y):
        z_global, z_local = z
        length = z_global.shape[0]
        s = jnp.fft.irfft(z_global, n=2 * (length - 1), axis=0)
        s = s + z_local
        return s


class FNO1d(nn.Module):
    num_blocks: int = 3
    num_modes: int = 8
    lift_dim: int = 16
    project_dim: int = 32
    output_dim: int = 1
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, u, y):
        # Lift to feature space
        u = jnp.concatenate([u, y], axis=-1)
        u = nn.Dense(self.lift_dim)(u)
        # FNO blocks
        for _ in range(self.num_blocks - 1):
            u = FNO1dBlock(self.lift_dim, self.lift_dim, self.num_modes)(u, y)
            u = self.activation(u)
        # Last block without activation
        u = FNO1dBlock(self.lift_dim, self.lift_dim, self.num_modes)(u, y)
        # Project to output space
        u = nn.Dense(self.project_dim)(u)
        u = self.activation(u)
        u = nn.Dense(self.output_dim)(u)
        return u


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
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0,
    )


class PatchEmbed1D(nn.Module):
    patch_size: tuple = (4,)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, h, c = x.shape

        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0],),
            (self.patch_size[0],),
            kernel_init=self.kernel_init,
            name="proj",
        )(x)

        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=self.layer_norm_eps)(x)
        return x


class MlpBlock(nn.Module):
    dim: int = 256
    out_dim: int = 256
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
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

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


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


x_emb_init = get_1d_sincos_pos_embed


class Encoder1D(nn.Module):
    patch_size: int = (4,)
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: int = 1
    out_dim: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, n, c = x.shape

        x = PatchEmbed1D(self.patch_size, self.emb_dim)(x)

        x_emb = self.variable(
            "pos_emb",
            "enc_emb",
            x_emb_init,
            self.emb_dim,
            n // self.patch_size[0],
        )

        x = x + x_emb.value

        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)

        return x


class CVit1D(nn.Module):
    patch_size: tuple = (4,)
    grid_size: tuple = (200,)
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
    layer_norm_eps: float = 1e-5
    embedding_type: str = "grid"

    def setup(self):
        if self.embedding_type == "grid":
            # Create grid and latents
            n_x = self.grid_size[0]
            self.grid = jnp.linspace(0, 1, n_x)
            self.latents = self.param("latents", normal(), (n_x, self.latent_dim))

    @nn.compact
    def __call__(self, x, coords):
        b, h, c = x.shape

        if self.embedding_type == "grid":
            d2 = (coords - self.grid[None, :]) ** 2
            w = jnp.exp(-1e5 * d2) / jnp.exp(-1e5 * d2).sum(axis=1, keepdims=True)

            coords = jnp.einsum("ic,pi->pc", self.latents, w)
            coords = nn.Dense(self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

        elif self.embedding_type == "mlp":
            coords = MlpBlock(self.dec_emb_dim, self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

        coords = einops.repeat(coords, "n d -> b n d", b=b)

        x = Encoder1D(
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps,
        )(x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.Dense(self.dec_emb_dim)(x)

        for _ in range(self.dec_depth):
            coords = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        x = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )(x)

        return x


# Define the model
class OperatorModel:
    def __init__(self, key, model_name):
        self.model_name = model_name
        if model_name == "nomad":
            model = Nomad(
                encoder_num_layers=4,
                encoder_hidden_dim=512,
                latent_dim=512,
                decoder_num_layers=4,
                decoder_hidden_dim=512,
                output_channels=1,
                activation=nn.gelu,
            )

        elif model_name == "deeponet":
            model = DeepONet(
                encoder_num_layers=4,
                decoder_num_layers=4,
                latent_dim=512,
                activation=nn.gelu,
            )

        elif model_name == "fno1d":
            model = FNO1d(
                num_blocks=3,
                num_modes=12,
                lift_dim=256,
                project_dim=256,
                output_dim=1,
                activation=nn.gelu,
            )
        elif model_name == "cvit":
            model = CVit1D(
                patch_size=(4,),
                grid_size=(200,),
                latent_dim=256,
                emb_dim=256,
                depth=6,
                num_heads=16,
                dec_emb_dim=256,
                dec_num_heads=16,
                dec_depth=1,
                num_mlp_layers=1,
                mlp_ratio=1,
                out_dim=1,
                layer_norm_eps=1e-5,
                embedding_type="grid",
            )
        else:
            raise ValueError("Model not implemented")

        self.init, self.apply = model.init, model.apply
        if model_name == "fno1d":
            self.params = self.init(
                key,
                jax.random.normal(key, (200, 1)),
                jax.random.normal(key, (200, 1)),
            )

        elif model_name == "deeponet" or model_name == "nomad":
            self.params = self.init(
                key, jax.random.normal(key, (200,)), jax.random.normal(key, (1,))
            )

        else:
            x = jnp.ones((32, 200, 1))
            coords = jnp.ones((128, 1))
            self.params = self.init(key, x, coords)

        # getting model size in MB:
        self.model_size = (
            sum([p.size for p in jax.tree.leaves(self.params)]) * 4 / 1024 / 1024
        )
        self.model_count = sum([p.size for p in jax.tree_leaves(self.params)])

        print(f"Model size: {self.model_size:.2f} MB")
        print(f"Model count: {self.model_count}")

        self.avg_params = self.params
        scheduler = optax.exponential_decay(1e-4, 1000, 0.95)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(scheduler, weight_decay=1e-5),
        )

        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.l2_error_log = []
        self.loss_log = []

    def s_net(self, params, u, y):
        s = self.apply(params, u, y)
        return s

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, batch):
        u, y, s = batch
        if self.model_name == "deeponet" or self.model_name == "nomad":
            u = jnp.squeeze(u)
            s_pred = jax.vmap(jax.vmap(self.s_net, (None, None, 0)), (None, 0, 0))(
                params, u, y
            )

        elif self.model_name == "fno1d":
            s_pred = jax.vmap(self.s_net, (None, 0, 0))(params, u, y)

        else:
            s_pred = self.s_net(params, u, y[0])

        s_pred = jnp.squeeze(s_pred)
        return jnp.mean((s - s_pred) ** 2)

    @partial(jax.jit, static_argnums=(0,))
    def compute_l2_error(self, params, batch):
        u, y, s = batch
        if self.model_name == "deeponet" or self.model_name == "nomad":
            u = jnp.squeeze(u)
            s_pred = jax.vmap(jax.vmap(self.s_net, (None, None, 0)), (None, 0, 0))(
                params, u, y
            )

        elif self.model_name == "fno1d":
            s_pred = jax.vmap(self.s_net, (None, 0, 0))(params, u, y)

        else:
            s_pred = self.s_net(params, u, y[0])

        s_pred = jnp.squeeze(s_pred)

        return jnp.mean(
            jnp.linalg.norm(s - s_pred, 2, axis=-1) / jnp.linalg.norm(s, 2, axis=-1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = jax.grad(self.loss)(params, batch)
        updates, self.opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def ema_update(self, params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    def save_model(self):
        to_dump = (
            self.params,
            self.avg_params,
            self.opt_state,
            self.loss_log,
            self.l2_error_log,
        )
        with open("./model_states/" + self.model_name + ".pickle", "wb") as handle:
            pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open("./model_states/" + self.model_name + ".pickle", "rb") as handle:
            loaded = pickle.load(handle)
        (
            self.params,
            self.avg_params,
            self.opt_state,
            self.loss_log,
            self.l2_error_log,
        ) = loaded

    # Optimize parameters in a loop
    def train(self, train_dataloader, val_dataloader, n_iter):
        pbar = trange(n_iter)
        # Main training loop
        best_error = 1000
        for it in pbar:
            batch = next(train_dataloader)
            self.params, self.opt_state = self.step(self.params, self.opt_state, batch)
            self.avg_params = self.ema_update(self.params, self.avg_params)
            if it % 100 == 0:
                params = self.avg_params
                error = 0
                loss_value = 0
                for _ in range(10000 // 256):
                    batch = next(val_dataloader)
                    error += self.compute_l2_error(params, batch)
                    loss_value += self.loss(params, batch)

                error /= 10000 // 256
                loss_value /= 10000 // 256
                if error < best_error:
                    best_error = error
                    self.save_model()

                self.l2_error_log.append(error)
                self.loss_log.append(loss_value)
                pbar.set_postfix(
                    {
                        "l2 error": error,
                        "Loss": loss_value,
                    }
                )

        print(f"Best error: {best_error}")

    def setup_ensemble(self, key, n_ensemble):
        keys = jax.random.split(key, n_ensemble)
        self.params = jax.vmap(self.init, in_axes=(0, None, None))(
            keys, jax.random.normal(key, (200, 1)), jax.random.normal(key, (200, 1))
        )
        self.avg_params = self.params
        self.opt_state = jax.vmap(self.optimizer.init)(self.params)
        return

    def ensemble_train(self, dataloader, n_iter):
        pbar = trange(n_iter)
        for it in pbar:
            batch = next(dataloader)
            self.params, self.opt_state = jax.vmap(self.step, (0, 0, None))(
                self.params, self.opt_state, batch
            )
            self.avg_params = self.ema_update(self.params, self.avg_params)
            if it % 100 == 0:
                params = self.avg_params
                error = jax.vmap(self.compute_l2_error, (0, None))(params, batch)
                loss_value = jax.vmap(self.loss, (0, None))(params, batch)

                self.l2_error_log.append(error)
                self.loss_log.append(loss_value)
                pbar.set_postfix(
                    {
                        "l2 error": error.mean(),
                        "Loss": loss_value.mean(),
                    }
                )
