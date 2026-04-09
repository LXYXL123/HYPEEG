import math

import torch
from torch import nn


def _compute_fans(shape):
    if len(shape) < 2:
        return 1.0, 1.0
    receptive_field = 1
    if len(shape) > 2:
        receptive_field = math.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field
    fan_out = shape[-1] * receptive_field
    return float(fan_in), float(fan_out)


class IndependentFilters:
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion="glorot", seed=None):
        self.kernel_size = tuple(kernel_size)
        self.input_dim = int(input_dim)
        self.weight_dim = int(weight_dim)
        self.nb_filters = None if nb_filters is None else int(nb_filters)
        self.criterion = criterion
        self.seed = 1337 if seed is None else int(seed)

    def __call__(self, shape, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)

        shape = tuple(shape)
        if self.nb_filters is None:
            rows, cols = shape[-2], shape[-1]
        else:
            rows = self.nb_filters * self.input_dim
            cols = math.prod(self.kernel_size)
        base = torch.randn(rows, cols, generator=generator, dtype=dtype, device=device)
        q = torch.linalg.qr(base, mode="reduced").Q
        q = q[:rows, :cols].reshape(shape)
        fan_in, fan_out = _compute_fans(shape)
        scale = math.sqrt(2.0 / (fan_in + fan_out)) if self.criterion == "glorot" else math.sqrt(2.0 / fan_in)
        return q * scale


class ComplexIndependentFilters:
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion="glorot", seed=None):
        self.real_initializer = IndependentFilters(kernel_size, input_dim, weight_dim, nb_filters, criterion, seed)
        imag_seed = None if seed is None else seed + 1
        self.imag_initializer = IndependentFilters(kernel_size, input_dim, weight_dim, nb_filters, criterion, imag_seed)

    def __call__(self, shape, dtype=None, device=None):
        real = self.real_initializer(shape, dtype=dtype, device=device)
        imag = self.imag_initializer(shape, dtype=dtype, device=device)
        return torch.cat([real, imag], dim=-1)


class ComplexInit:
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion="glorot", seed=None):
        self.kernel_size = tuple(kernel_size)
        self.input_dim = int(input_dim)
        self.weight_dim = int(weight_dim)
        self.nb_filters = None if nb_filters is None else int(nb_filters)
        self.criterion = criterion
        self.seed = 1337 if seed is None else int(seed)

    def __call__(self, shape, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        fan_in, fan_out = _compute_fans(tuple(shape))
        scale = math.sqrt(1.0 / (fan_in + fan_out)) if self.criterion == "glorot" else math.sqrt(1.0 / fan_in)
        real = torch.randn(*shape, generator=generator, dtype=dtype, device=device) * scale
        imag = torch.randn(*shape, generator=generator, dtype=dtype, device=device) * scale
        return torch.cat([real, imag], dim=-1)


class SqrtInit:
    def __call__(self, shape, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        return torch.full(tuple(shape), 1.0 / math.sqrt(2.0), dtype=dtype, device=device)

