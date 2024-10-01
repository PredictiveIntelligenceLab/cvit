import os

import h5py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from einops import rearrange


# Construct the full dataset
def create_dr_datasets(filename, prev_steps, pred_steps, train_samples, test_samples, downsample_factor=1, normalize=True):
    data_list = []
    with h5py.File(filename, 'r') as fp:
        num_samples = train_samples + test_samples
        for i in range(0, num_samples):
            data = fp["{0:0=4d}/data".format(i)][::downsample_factor]
            data_list.append(data)
        data = np.stack(data_list, axis=0)

    train_data = data[:train_samples]
    test_data = data[train_samples:train_samples + test_samples]

    # Normalize data
    mean = np.mean(train_data, axis=(0, 2, 3), keepdims=True)
    std = np.std(train_data, axis=(0, 2, 3), keepdims=True)

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Use sliding window to generate inputs and outputs
    train_data = sliding_window_view(train_data, window_shape=prev_steps + pred_steps, axis=1)
    test_data = sliding_window_view(test_data, window_shape=prev_steps + pred_steps, axis=1)

    train_data = rearrange(train_data, "n m h w c s -> (n m) s h w c")
    test_data = rearrange(test_data, "n m h w c s -> (n m) s h w c")

    train_inputs = train_data[:, :prev_steps, ...]
    train_outputs = train_data[:, prev_steps:prev_steps + pred_steps, ...]

    test_inputs = test_data[:, :prev_steps, ...]
    test_outputs = test_data[:, prev_steps:prev_steps + pred_steps, ...]

    train_dataset = (train_inputs, train_outputs)
    test_dataset = (test_inputs, test_outputs)

    return train_dataset, test_dataset, mean, std


