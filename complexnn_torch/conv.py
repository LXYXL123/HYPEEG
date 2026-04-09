import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair, _single, _triple

from .utils import combine_complex, split_complex


def sanitizedInitGet(init):
    return init


def sanitizedInitSer(init):
    return str(init)


def _normalize_tuple(rank, value):
    return {1: _single, 2: _pair, 3: _triple}[rank](value)


def _same_padding(kernel_size, dilation):
    return tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))


class ComplexConv(nn.Module):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        in_channels=None,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        normalize_weight=False,
        kernel_initializer="complex",
        bias_initializer="zeros",
        gamma_diag_initializer="sqrt_init",
        gamma_off_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        gamma_diag_regularizer=None,
        gamma_off_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        gamma_diag_constraint=None,
        gamma_off_constraint=None,
        init_criterion="he",
        spectral_parametrization=False,
        epsilon=1e-7,
        seed=None,
    ):
        super().__init__()
        self.rank = rank
        self.filters = filters
        self.in_channels = in_channels
        self.kernel_size = _normalize_tuple(rank, kernel_size)
        self.strides = _normalize_tuple(rank, strides)
        self.padding = padding
        self.data_format = "channels_last" if data_format is None and rank == 1 else (data_format or "channels_first")
        self.dilation_rate = _normalize_tuple(rank, dilation_rate)
        self.activation = activation
        self.use_bias = use_bias
        self.normalize_weight = normalize_weight
        self.spectral_parametrization = spectral_parametrization
        self.epsilon = epsilon
        self.seed = seed
        self._initialized = False
        if self.in_channels is not None:
            self._initialize_layers(self.in_channels)

    def _conv_cls(self):
        return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[self.rank]

    def _padding_value(self):
        if self.padding == "same":
            return _same_padding(self.kernel_size, self.dilation_rate)
        if self.padding == "valid":
            return 0 if self.rank == 1 else tuple(0 for _ in range(self.rank))
        raise ValueError(f"Unsupported padding mode: {self.padding}")

    def _to_channels_first(self, x):
        if self.data_format == "channels_first":
            return x
        if self.rank == 1:
            return x.transpose(1, 2)
        order = [0, x.ndim - 1] + list(range(1, x.ndim - 1))
        return x.permute(*order)

    def _from_channels_first(self, x):
        if self.data_format == "channels_first":
            return x
        if self.rank == 1:
            return x.transpose(1, 2)
        order = [0] + list(range(2, x.ndim)) + [1]
        return x.permute(*order)

    def _initialize_layers(self, in_channels, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        conv_cls = self._conv_cls()
        kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._padding_value(),
            dilation=self.dilation_rate,
            bias=self.use_bias,
        )
        self.real_conv = conv_cls(in_channels, self.filters, **kwargs).to(dtype=dtype, device=device)
        self.imag_conv = conv_cls(in_channels, self.filters, **kwargs).to(dtype=dtype, device=device)
        self._initialized = True

    def _ensure_init(self, inputs):
        if self._initialized:
            return
        x = self._to_channels_first(inputs)
        in_channels = x.size(1) // 2
        self._initialize_layers(in_channels, dtype=x.dtype, device=x.device)

    def _apply_conv(self, conv, x):
        weight = conv.weight
        if self.normalize_weight:
            flat = weight.view(weight.size(0), -1)
            normed = F.normalize(flat, dim=1, eps=self.epsilon).view_as(weight)
            return F.conv1d(x, normed, conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation) if self.rank == 1 else (
                F.conv2d(x, normed, conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation) if self.rank == 2 else
                F.conv3d(x, normed, conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
            )
        return conv(x)

    def forward(self, inputs):
        self._ensure_init(inputs)
        x = self._to_channels_first(inputs)
        real, imag = split_complex(x, axis=1)
        rr = self._apply_conv(self.real_conv, real)
        ii = self._apply_conv(self.imag_conv, imag)
        ri = self._apply_conv(self.imag_conv, real)
        ir = self._apply_conv(self.real_conv, imag)
        output = combine_complex(rr - ii, ri + ir, axis=1)
        if self.activation is not None:
            output = self.activation(output)
        return self._from_channels_first(output)


class ComplexConv1D(ComplexConv):
    def __init__(self, filters, kernel_size, in_channels=None, strides=1, padding="valid", dilation_rate=1, activation=None, use_bias=True, kernel_initializer="complex", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, init_criterion="he", spectral_parametrization=False, **kwargs):
        super().__init__(1, filters, kernel_size, in_channels=in_channels, strides=strides, padding=padding, data_format="channels_last", dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, seed=seed, init_criterion=init_criterion, spectral_parametrization=spectral_parametrization, **kwargs)


class ComplexConv2D(ComplexConv):
    def __init__(self, filters, kernel_size, in_channels=None, strides=(1, 1), padding="valid", data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer="complex", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, init_criterion="he", spectral_parametrization=False, **kwargs):
        super().__init__(2, filters, kernel_size, in_channels=in_channels, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, seed=seed, init_criterion=init_criterion, spectral_parametrization=spectral_parametrization, **kwargs)


class ComplexConv3D(ComplexConv):
    def __init__(self, filters, kernel_size, in_channels=None, strides=(1, 1, 1), padding="valid", data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer="complex", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=None, init_criterion="he", spectral_parametrization=False, **kwargs):
        super().__init__(3, filters, kernel_size, in_channels=in_channels, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, seed=seed, init_criterion=init_criterion, spectral_parametrization=spectral_parametrization, **kwargs)


class WeightNorm_Conv(nn.Module):
    def __init__(
        self,
        rank=1,
        in_channels=None,
        out_channels=None,
        kernel_size=1,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        gamma_initializer="ones",
        gamma_regularizer=None,
        gamma_constraint=None,
        epsilon=1e-7,
    ):
        super().__init__()
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(rank, kernel_size)
        self.strides = _normalize_tuple(rank, strides)
        self.padding = padding
        self.data_format = "channels_last" if data_format is None and rank == 1 else (data_format or "channels_first")
        self.dilation_rate = _normalize_tuple(rank, dilation_rate)
        self.activation = activation
        self.use_bias = use_bias
        self.epsilon = epsilon
        self._initialized = False

    def _conv_cls(self):
        return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[self.rank]

    def _padding_value(self):
        if self.padding == "same":
            return _same_padding(self.kernel_size, self.dilation_rate)
        return 0 if self.rank == 1 else tuple(0 for _ in range(self.rank))

    def _to_channels_first(self, x):
        if self.data_format == "channels_first":
            return x
        if self.rank == 1:
            return x.transpose(1, 2)
        return x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))

    def _from_channels_first(self, x):
        if self.data_format == "channels_first":
            return x
        if self.rank == 1:
            return x.transpose(1, 2)
        return x.permute(0, *range(2, x.ndim), 1)

    def _ensure_init(self, x):
        if self._initialized:
            return
        x_cf = self._to_channels_first(x)
        in_channels = self.in_channels or x_cf.size(1)
        out_channels = self.out_channels
        if out_channels is None:
            raise ValueError("out_channels must be provided for WeightNorm_Conv.")
        conv_cls = self._conv_cls()
        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._padding_value(),
            dilation=self.dilation_rate,
            bias=self.use_bias,
        ).to(dtype=x_cf.dtype, device=x_cf.device)
        self.gamma = nn.Parameter(torch.ones(out_channels, dtype=x_cf.dtype, device=x_cf.device))
        self._initialized = True

    def forward(self, x):
        self._ensure_init(x)
        x_cf = self._to_channels_first(x)
        weight = self.conv.weight
        flat = weight.view(weight.size(0), -1)
        normed = F.normalize(flat, dim=1, eps=self.epsilon) * self.gamma[:, None]
        normed = normed.view_as(weight)
        if self.rank == 1:
            out = F.conv1d(x_cf, normed, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        elif self.rank == 2:
            out = F.conv2d(x_cf, normed, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        else:
            out = F.conv3d(x_cf, normed, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        if self.activation is not None:
            out = self.activation(out)
        return self._from_channels_first(out)


ComplexConvolution1D = ComplexConv1D
ComplexConvolution2D = ComplexConv2D
ComplexConvolution3D = ComplexConv3D
