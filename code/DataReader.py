#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def load_data(filename):
    data = np.load(filename)
    x = data['x']
    y = data['y']
    return (x, y)


def train_valid_split(raw_data, labels, split_index):
    return (raw_data[:split_index], raw_data[split_index:],
            labels[:split_index], labels[split_index:])


def prepare_X(raw_X):
    raw_image = raw_X.reshape((-1, 16, 16))

    symmetry = np.zeros(int(np.size(raw_image) / 256))
    intensity = np.zeros(int(np.size(raw_image) / 256))
    bias = np.zeros(int(np.size(raw_image) / 256))

    for (i, row) in enumerate(raw_image):

      # Feature 1: Measure of Symmetry
        # ## YOUR CODE HERE

        symmetry[i] = -1 * np.sum(np.abs(np.subtract(np.fliplr(row),
                                  row))) / 256

      # ## END YOUR CODE

      # Feature 2: Measure of Intensity
        # ## YOUR CODE HERE

        intensity[i] = np.sum(row) / 256

      # ## END YOUR CODE

      # Feature 3: Bias Term. Always 1.
        # ## YOUR CODE HERE

        bias[i] = 1

        # ## END YOUR CODE

    # Stack features together in the following order.
    # [Feature 3, Feature 1, Feature 2]
    # ## YOUR CODE HERE

    X = np.dstack([bias, symmetry, intensity])[0]

    # ## END YOUR CODE

    return X


def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """

    y = raw_y
    idx = np.where((raw_y == 1) | (raw_y == 2))
    y[np.where(raw_y == 0)] = 0
    y[np.where(raw_y == 1)] = 1
    y[np.where(raw_y == 2)] = 2

    return (y, idx)
