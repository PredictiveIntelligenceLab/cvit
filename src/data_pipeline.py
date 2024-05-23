import numpy as np

from einops import rearrange

import jax
from jax import random

import jax.numpy as jnp


import flax

import tensorflow as tf

from torch.utils.data import Dataset, DataLoader, Subset


class BaseDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        # Ensure all datasets have the same length
        assert all(
            len(datasets[0]) == len(dataset) for dataset in datasets
        ), "All datasets must have the same length"
        self.datasets = datasets

    def __len__(self):
        # Assuming all datasets have the same length, use the first one to determine the length
        return len(self.datasets[0])

    def __getitem__(self, index):
        # Retrieve the corresponding item from each dataset
        return tuple(dataset[index] for dataset in self.datasets)


def prepare_tf_data(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        x = x._numpy()
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def prefetch(dataset, n_prefetch=None):
    """Prefetches data to device and converts to numpy array."""
    ds_iter = iter(dataset)
    ds_iter = map(
        lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x), ds_iter
    )
    if n_prefetch:
        ds_iter = map(prepare_tf_data, ds_iter)
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)

    return ds_iter


def batch_parser(batch, rng=None, num_query_points=None):
    inputs, outputs = batch
    # TODO: Support multi-GPU training
    batch_inputs = jnp.squeeze(inputs)
    batch_outputs = jnp.squeeze(outputs)

    b, t, h, w, c = batch_inputs.shape
    x_star = jnp.linspace(0, 1, h)
    y_star = jnp.linspace(0, 1, w)

    x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")
    batch_coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    batch_outputs = rearrange(batch_outputs, "b h w c -> b (h w) c")

    if num_query_points is not None:
        query_index = random.choice(
            rng, batch_outputs.shape[1], (num_query_points,), replace=False
        )
        batch_coords = batch_coords[query_index]
        batch_outputs = batch_outputs[:, query_index]

    return batch_coords, batch_inputs, batch_outputs


def create_dataloaders(config, train_dataset, test_dataset):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

    shuffle_buffer_size = config.batch_size * 100

    train_dataset = (
        train_dataset.cache()
        .shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        .repeat()
        .batch(config.batch_size)
        .prefetch(8)
    )
    test_dataset = (
        test_dataset.cache().repeat().batch(config.batch_size // 4).prefetch(8)
    )

    train_iter = map(prepare_tf_data, train_dataset)
    test_iter = map(prepare_tf_data, test_dataset)

    train_iter = flax.jax_utils.prefetch_to_device(train_iter, 2)
    test_iter = flax.jax_utils.prefetch_to_device(test_iter, 2)

    return train_iter, test_iter
