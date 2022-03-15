import torch


def spatial_broadcast(x, resolution):
    x = x.reshape(-1, x.shape[-1], 1, 1)
    x = x.expand(-1, -1, *resolution)
    return x


def spatial_flatten(x):
    x = torch.swapaxes(x, 1, -1)
    return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])