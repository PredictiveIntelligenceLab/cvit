import jax
import jax.numpy as jnp
import numpy as np
import einops


def grid_sampling_iter(key, u, y, s, batch_size, grid_size):
    @jax.jit
    def _data_generation(sample_key):
        batch_idx = jax.random.randint(sample_key, (batch_size,), 0, u.shape[0])
        grid_idx = jnp.sort(
            jax.random.randint(sample_key, (grid_size,), 0, u.shape[1])
        )

        return (
            u[batch_idx, :],
            y[batch_idx, :][:, grid_idx],
            s[batch_idx, :][:, grid_idx],
        )

    while True:
        key, subkey = jax.random.split(key)
        yield _data_generation(subkey)


def get_train_val_test_data():
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

    train_data = (
        jnp.array(inputs_train),
        jnp.array(grid_train),
        jnp.array(outputs_train),
    )
    val_data = (
        jnp.array(inputs_val),
        jnp.array(grid_val),
        jnp.array(outputs_val),
    )
    test_data = (
        jnp.array(inputs_test),
        jnp.array(grid_test),
        jnp.array(outputs_test),
    )

    return train_data, val_data, test_data


def get_train_val_test_loaders(batch_size, grid_size):
    train_data, val_data, test_data = get_train_val_test_data()
    inputs_train, grid_train, outputs_train = train_data
    inputs_val, grid_val, outputs_val = val_data
    inputs_test, grid_test, outputs_test = test_data

    train_dataloader = grid_sampling_iter(
        jax.random.PRNGKey(42),
        inputs_train,
        grid_train,
        outputs_train,
        batch_size,
        grid_size,
    )

    val_loader = grid_sampling_iter(
        jax.random.PRNGKey(42),
        inputs_val,
        grid_val,
        outputs_val,
        batch_size,
        200,
    )

    test_dataloader = grid_sampling_iter(
        jax.random.PRNGKey(42),
        inputs_test,
        grid_test,
        outputs_test,
        batch_size,
        200,
    )

    return train_dataloader, val_loader, test_dataloader
