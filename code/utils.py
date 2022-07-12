import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import torch.nn.functional as F
from operator import add

import kitti_utils
from dataset import load_dataset
import train


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / \
        torch.sqrt(2 * torch.tensor(math.pi)) / sigma


def RBF_encode(x, device, num_neurons=5, eta=1.):
    if num_neurons < 2:
        print('neurons number should be larger than 1')
        assert Exception
        return 0

    res = torch.zeros(tuple(x.shape) + (num_neurons, ), device=device)
    scale = 1. / (num_neurons - 2)
    mus = [(2 * i - 2) / 2 * scale for i in range(num_neurons)]
    sigmas = scale / eta
    for i in range(num_neurons):
        res[:, :, :, i] = gaussian(x, mu=mus[i], sigma=sigmas)
    res = res.reshape(tuple(x.shape[:-1]) + (x.shape[-1] * num_neurons, ))
    return res


def get_temporal_encoding(windows, time_per_window, device):
    (batch_size, sqeuence_length, input_size) = windows.shape
    # Bin values in time_per_window bins; initial values are in the interval
    # [0, 1]; round leads to int values between 0 and time_per_window.
    # Subtract 1 in order to ignore 0 values.
    windows[windows == 0.0] = -1.0
    windows = torch.floor(windows * time_per_window)

    new_windows = torch.zeros(
        (batch_size, sqeuence_length * time_per_window, input_size), device=device)
    for i in range(sqeuence_length):
        input_pixel_intensity = torch.clamp(
            windows[:, i, :], -1, time_per_window - 1)
        for time_step in range(time_per_window):
            new_windows[:, i * time_per_window + time_step,
                        :] = input_pixel_intensity == time_step
    del windows
    return new_windows


def encode_data(input_data, kernel_size, stride, device, encoding="rate", time_per_window=0, neurons_per_pixel=0):
    windows = F.unfold(input_data, kernel_size=kernel_size,
                       stride=stride).permute(0, 2, 1).to(device)
    if encoding == "rate":
        return windows
    if encoding == "temporal":
        return get_temporal_encoding(windows, time_per_window, device)
    if encoding == "RBF":
        return RBF_encode(windows, device, num_neurons=neurons_per_pixel)
    if encoding == "RBF_TC":
        new_windows = RBF_encode(
            windows, device, num_neurons=neurons_per_pixel)
        new_windows = get_temporal_encoding(
            new_windows, time_per_window, device)
        return new_windows
    return windows
