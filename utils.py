import random
import torch


def to_one_hot(size, excitation):
    y = torch.zeros(size)
    y[excitation] = 1
    return y


def to_one_hot_multiple(size, excitation):
    batch_size = len(excitation)
    y = torch.zeros(size * batch_size)
    for index in range(batch_size):
        if isinstance(index * size + excitation[index], torch.Tensor):
            y[(index * size + excitation[index]).long()] = 1
        else:
            y[index * size + excitation[index]] = 1
    return y.view(batch_size, size)


def generate_randomized_one_hot_encoding(size, probabilities):
    epsilon = torch.rand(1)
    print(epsilon)
    start = 0
    for index in range(probabilities.shape[0]):
        start += probabilities[index]
        if start > epsilon:
            return to_one_hot(size, index)
