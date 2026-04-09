import math

import torch
from torch import nn


def _canonical_axis(x, axis=-1):
    return axis if axis >= 0 else x.ndim + axis


def _split(x, axis=-1):
    axis = _canonical_axis(x, axis)
    size = x.size(axis)
    if size % 2 != 0:
        raise ValueError(f"Complex tensor axis must be even, got {size}.")
    return torch.split(x, size // 2, dim=axis)


def get_realpart(x, axis=-1):
    if torch.is_complex(x):
        return x.real
    real, _ = _split(x, axis=axis)
    return real


def get_imagpart(x, axis=-1):
    if torch.is_complex(x):
        return x.imag
    _, imag = _split(x, axis=axis)
    return imag


def get_abs(x, axis=-1):
    if torch.is_complex(x):
        return x.abs()
    real, imag = _split(x, axis=axis)
    return torch.sqrt(real.square() + imag.square() + 1e-12)


def getpart_output_shape(input_shape, axis=-1):
    output_shape = list(input_shape)
    axis = axis if axis >= 0 else len(output_shape) + axis
    if output_shape[axis] is not None:
        output_shape[axis] = output_shape[axis] // 2
    return tuple(output_shape)


def split_complex(x, axis=-1):
    return _split(x, axis=axis)


def combine_complex(real, imag, axis=-1):
    return torch.cat([real, imag], dim=axis)


def to_complex(x, axis=-1):
    if torch.is_complex(x):
        return x
    real, imag = _split(x, axis=axis)
    return torch.complex(real, imag)


def from_complex(x, axis=-1):
    return torch.cat([x.real, x.imag], dim=axis)


class GetReal(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return get_realpart(inputs, axis=self.axis)


class GetImag(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return get_imagpart(inputs, axis=self.axis)


class GetAbs(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return get_abs(inputs, axis=self.axis)

