import os
import re

import h5py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from einops import rearrange


# Construct the full dataset
def prepare_ns_dataset(directory, mode, keys, prev_steps, pred_steps, num_samples, downsample=1):
    # Use list comprehension for efficiency
    file_names = [directory + f for f in os.listdir(directory) if re.match(f"^NavierStokes2D_{mode}", f)]

    # Initialize dictionaries to hold the inputs and outputs
    data_dict = {key: [] for key in keys}

    num_files = len(file_names)

    f = h5py.File(file_names[0], "r")
    s = f[mode][keys[0]].shape[0]

    for i in range(min(num_files, num_samples // s + 1)):
        with h5py.File(file_names[i], "r") as f:
            data_group = f[mode]

            for key in keys:
                # Use memory-mapping to reduce memory usage
                data_dict[key].append(np.array(data_group[key], dtype=np.float32))

    for key in keys:
        data_dict[key] = np.vstack(data_dict[key])

    data = np.concatenate([np.expand_dims(arr, axis=-1) for arr in data_dict.values()], axis=-1)
    data = data[:num_samples, :, ::downsample, ::downsample, :]

    # Use sliding window to generate inputs and outputs
    sliding_data = sliding_window_view(data, window_shape=prev_steps + pred_steps, axis=1)
    sliding_data = rearrange(sliding_data, "n m h w c s -> (n m) s h w c")

    inputs = sliding_data[:, :prev_steps, ...]
    outputs = sliding_data[:, prev_steps:prev_steps + pred_steps, ...]

    return inputs, outputs


def create_ns_datasets(config):
    train_inputs, train_outputs = prepare_ns_dataset(
        directory=config.path,
        keys=config.components,
        prev_steps=config.prev_steps,
        pred_steps=config.pred_steps,
        num_samples=config.train_samples,
        downsample=config.downsample,
        mode="train",
        )

    test_inputs, test_outputs = prepare_ns_dataset(
        directory=config.path,
        keys=config.components,
        prev_steps=config.prev_steps,
        pred_steps=config.pred_steps,
        num_samples=config.test_samples,
        downsample=config.downsample,
        mode="test",
        )

    train_dataset = (train_inputs, train_outputs)
    test_dataset = (test_inputs, test_outputs)

    return train_dataset, test_dataset





# Pytorch dataloader
# class SWEDataset(Dataset):
#     def __init__(self, directory, mode, keys, prev_steps, pred_steps):
#         self.keys = keys
#         self.filename = directory + mode + ".zarr"
#         # normalization constants
#         self.normstats = torch.load(directory + "normstats.pt")
#
#         self.prev_steps = prev_steps
#         self.pred_steps = pred_steps
#         self.file = zarr.open(self.filename, mode="r")
#
#         N, T = self.file[self.keys[0]].shape[:2]
#         # subsample time step
#         self.T = T // 8
#
#         self.data_shape = self.file[self.keys[0]].shape
#
#     def __len__(self):
#         return self.data_shape[0]
#
#     def __getitem__(self, index):
#         start_time = np.random.randint(0, self.T - self.pred_steps - self.prev_steps)
#
#         end_time = start_time + self.prev_steps
#         pred_time = end_time + self.pred_steps
#
#         inputs = dict()
#         outputs = dict()
#         for k in self.keys:
#             mean = np.array(self.normstats[k]["mean"])
#             std = np.array(self.normstats[k]["std"])
#             # subsample time step
#             i = np.array(self.file[k][index, ::8][start_time:end_time])
#             i = (i - mean[None,]) / std[None,]
#             # subsample time step
#             o = np.array(self.file[k][index, ::8][end_time:pred_time])
#             o = (o - mean[None,]) / std[None,]
#             if i.ndim == 3:
#                 i = i[..., None]
#                 o = o[..., None]
#             elif i.ndim == 4:
#                 # move the feature axis to the last to comply with the flax conv convention
#                 i = np.moveaxis(i, 1, -1)
#                 o = np.moveaxis(o, 1, -1)
#             inputs[k] = i
#             outputs[k] = o
#
#         return inputs, outputs
#
#
# def batch_parser(batch, components, num_query_points=None):
#     inputs, outputs = batch
#
#     batch_inputs = [inputs[k] for k in components]
#     batch_outputs = [outputs[k] for k in components]
#
#     # Concatenation
#     batch_inputs = torch.cat(batch_inputs, -1)
#     batch_outputs = torch.cat(batch_outputs, -1)
#
#     # Create grid
#     b, t, h, w, c = batch_inputs.shape
#     x_star = np.linspace(0, 1, h)
#     y_star = np.linspace(0, 1, w)
#
#     x_star, y_star = np.meshgrid(x_star, y_star, indexing="ij")
#     batch_coords = np.hstack([x_star.flatten()[:,None], y_star.flatten()[:,None]])
#
#     batch_outputs = rearrange(batch_outputs, 'b t h w c -> b (t h w) c')
#
#     if num_query_points is not None:
#         query_index = np.random.choice(batch_coords.shape[0], num_query_points, replace=False)
#         batch_coords = batch_coords[query_index]
#         batch_outputs = batch_outputs[:, query_index]
#
#     batch = batch_coords, batch_inputs, batch_outputs
#     batch = jax.tree_map(lambda x: jnp.array(x), batch)
#
#     return batch


