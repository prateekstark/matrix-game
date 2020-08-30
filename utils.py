import random
import torch


def to_one_hot(size, excitation):
    y = torch.zeros(size)
    y[excitation] = 1
    return y


def generate_randomized_one_hot_encoding(size, probabilities):
    epsilon = torch.rand(1)
    print(epsilon)
    start = 0
    for index in range(probabilities.shape[0]):
        start += probabilities[index]
        if start > epsilon:
            return to_one_hot(size, index)
