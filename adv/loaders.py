import jax
import jax.numpy as jnp
import numpy as np
import einops
import torch.utils.data as data
from functools import partial


# defining the dataloader:
class GridSampling(data.Dataset):
    def __init__(self, key, u, y, s, batch_size, grid_size):
        self.key = key
        self.u = u
        self.y = y
        self.s = s
        self.batch_size = batch_size
        self.grid_size = grid_size

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = jax.random.split(self.key)
        batch = self.__data_generation(subkey)
        return batch

    @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key):
        batch_idx = jax.random.randint(key, (self.batch_size,), 0, self.u.shape[0])
        grid_idx = jnp.sort(
            jax.random.randint(key, (self.grid_size,), 0, self.u.shape[1])
        )

        return (
            self.u[batch_idx, :],
            self.y[batch_idx, :][:, grid_idx],
            self.s[batch_idx, :][:, grid_idx],
        )


def get_train_val_test_loaders(batch_size, grid_size):
    data_dir = "/scratch/PDEDatasets/advection_1d"
    inputs = np.load(f"{data_dir}/adv_a0.npy")
    outputs = np.load(f"{data_dir}/adv_aT.npy")
    grid = np.linspace(0, 1, inputs.shape[0])
    grid = einops.repeat(grid, "i -> i b", b=inputs.shape[1])

    # swapping the first two axes:
    inputs = einops.rearrange(inputs, "i j -> j i 1")
    outputs = einops.rearrange(outputs, "i j -> j i")
    grid = einops.rearrange(grid, "i j -> j i 1")

    idx = jax.random.permutation(jax.random.PRNGKey(441), inputs.shape[0])
    n_train = 20000
    n_val = 10000
    n_test = 10000
    inputs_train, outputs_train, grid_train = (
        inputs[idx[:n_train]],
        outputs[idx[:n_train]],
        grid[idx[:n_train]],
    )
    inputs_val, outputs_val, grid_val = (
        inputs[idx[n_train : n_train + n_val]],
        outputs[idx[n_train : n_train + n_val]],
        grid[idx[n_train : n_train + n_val]],
    )
    inputs_test, outputs_test, grid_test = (
        inputs[idx[-n_test:]],
        outputs[idx[-n_test:]],
        grid[idx[-n_test:]],
    )

    train_dataloader = GridSampling(
        jax.random.PRNGKey(42),
        jnp.array(inputs_train),
        jnp.array(grid_train),
        jnp.array(outputs_train),
        batch_size,
        grid_size,
    )

    val_loader = GridSampling(
        jax.random.PRNGKey(42),
        jnp.array(inputs_val),
        jnp.array(grid_val),
        jnp.array(outputs_val),
        batch_size,
        200,
    )

    test_dataloader = GridSampling(
        jax.random.PRNGKey(42),
        jnp.array(inputs_test),
        jnp.array(grid_test),
        jnp.array(outputs_test),
        batch_size,
        200,
    )

    return train_dataloader, val_loader, test_dataloader
