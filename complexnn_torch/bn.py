import math

import torch
from torch import nn

from .utils import combine_complex, split_complex


def sqrt_init(shape, dtype=None, device=None):
    dtype = dtype or torch.float32
    device = device or "cpu"
    return torch.full(tuple(shape), 1.0 / math.sqrt(2.0), dtype=dtype, device=device)


def sanitizedInitGet(init):
    if init in ["sqrt_init", sqrt_init]:
        return sqrt_init
    if callable(init):
        return init
    raise ValueError(f"Unsupported initializer: {init}")


def sanitizedInitSer(init):
    if init in [sqrt_init, "sqrt_init"]:
        return "sqrt_init"
    return str(init)


def _canonical_axis(x, axis):
    return axis if axis >= 0 else x.ndim + axis


def _expand_param(param, x, axis, layernorm=False):
    if param is None:
        return None
    axis = _canonical_axis(x, axis)
    shape = [1] * x.ndim
    if layernorm and param.ndim == 2:
        shape[0] = param.shape[0]
        shape[axis] = param.shape[1]
    else:
        shape[axis] = param.shape[0]
    return param.reshape(shape)


def complex_standardization(input_centred, Vrr, Vii, Vri, layernorm=False, axis=-1):
    axis = _canonical_axis(input_centred, axis)
    real, imag = split_complex(input_centred, axis=axis)
    s = torch.sqrt(torch.clamp(Vrr * Vii - Vri.square(), min=1e-12))
    t = torch.sqrt(torch.clamp(Vrr + Vii + 2.0 * s, min=1e-12))
    inverse_st = 1.0 / torch.clamp(s * t, min=1e-12)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st
    Wrr = _expand_param(Wrr, real, axis, layernorm)
    Wii = _expand_param(Wii, imag, axis, layernorm)
    Wri = _expand_param(Wri, real, axis, layernorm)
    return combine_complex(Wrr * real + Wri * imag, Wri * real + Wii * imag, axis=axis)


def ComplexBN(input_centred, Vrr, Vii, Vri, beta, gamma_rr, gamma_ri, gamma_ii, scale=True, center=True, layernorm=False, axis=-1):
    axis = _canonical_axis(input_centred, axis)
    output = input_centred
    if scale:
        output = complex_standardization(input_centred, Vrr, Vii, Vri, layernorm=layernorm, axis=axis)
        real, imag = split_complex(output, axis=axis)
        gamma_rr = _expand_param(gamma_rr, real, axis, layernorm)
        gamma_ii = _expand_param(gamma_ii, imag, axis, layernorm)
        gamma_ri = _expand_param(gamma_ri, real, axis, layernorm)
        output = combine_complex(
            gamma_rr * real + gamma_ri * imag,
            gamma_ri * real + gamma_ii * imag,
            axis=axis,
        )
    if center and beta is not None:
        beta = _expand_param(beta, output, axis, layernorm)
        output = output + beta
    return output


class ComplexBatchNormalization(nn.Module):
    def __init__(
        self,
        num_features=None,
        axis=-1,
        momentum=0.9,
        epsilon=1e-4,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_diag_initializer="sqrt_init",
        gamma_off_initializer="zeros",
        moving_mean_initializer="zeros",
        moving_variance_initializer="sqrt_init",
        moving_covariance_initializer="zeros",
        beta_regularizer=None,
        gamma_diag_regularizer=None,
        gamma_off_regularizer=None,
        beta_constraint=None,
        gamma_diag_constraint=None,
        gamma_off_constraint=None,
    ):
        super().__init__()
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.num_features = num_features
        self._initialized = False
        if self.num_features is not None:
            self._initialize_parameters(self.num_features)

    def _initialize_parameters(self, features, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        if self.scale:
            self.gamma_rr = nn.Parameter(sqrt_init((features,), dtype=dtype, device=device))
            self.gamma_ii = nn.Parameter(sqrt_init((features,), dtype=dtype, device=device))
            self.gamma_ri = nn.Parameter(torch.zeros(features, dtype=dtype, device=device))
            self.register_buffer("moving_Vrr", torch.ones(features, dtype=dtype, device=device))
            self.register_buffer("moving_Vii", torch.ones(features, dtype=dtype, device=device))
            self.register_buffer("moving_Vri", torch.zeros(features, dtype=dtype, device=device))
        else:
            self.gamma_rr = self.gamma_ii = self.gamma_ri = None
            self.register_buffer("moving_Vrr", None)
            self.register_buffer("moving_Vii", None)
            self.register_buffer("moving_Vri", None)
        if self.center:
            channels = features * 2
            self.beta = nn.Parameter(torch.zeros(channels, dtype=dtype, device=device))
            self.register_buffer("moving_mean", torch.zeros(channels, dtype=dtype, device=device))
        else:
            self.beta = None
            self.register_buffer("moving_mean", None)
        self._initialized = True

    def _ensure_init(self, inputs):
        axis = _canonical_axis(inputs, self.axis)
        channels = inputs.size(axis)
        if channels % 2 != 0:
            raise ValueError("Complex batch norm expects concatenated real/imag channels.")
        features = channels // 2
        if self._initialized:
            return
        self._initialize_parameters(features, dtype=inputs.dtype, device=inputs.device)

    def forward(self, inputs):
        self._ensure_init(inputs)
        axis = _canonical_axis(inputs, self.axis)
        reduction_axes = tuple(i for i in range(inputs.ndim) if i != axis)

        if self.training:
            mu = inputs.mean(dim=reduction_axes)
            centred = inputs - _expand_param(mu, inputs, axis)
            real, imag = split_complex(centred, axis=axis)
            Vrr = real.square().mean(dim=reduction_axes) + self.epsilon
            Vii = imag.square().mean(dim=reduction_axes) + self.epsilon
            Vri = (real * imag).mean(dim=reduction_axes)
            if self.center:
                self.moving_mean.mul_(self.momentum).add_(mu.detach() * (1.0 - self.momentum))
            if self.scale:
                self.moving_Vrr.mul_(self.momentum).add_(Vrr.detach() * (1.0 - self.momentum))
                self.moving_Vii.mul_(self.momentum).add_(Vii.detach() * (1.0 - self.momentum))
                self.moving_Vri.mul_(self.momentum).add_(Vri.detach() * (1.0 - self.momentum))
        else:
            centred = inputs if not self.center else inputs - _expand_param(self.moving_mean, inputs, axis)
            Vrr = self.moving_Vrr
            Vii = self.moving_Vii
            Vri = self.moving_Vri

        return ComplexBN(
            centred,
            Vrr,
            Vii,
            Vri,
            self.beta,
            self.gamma_rr,
            self.gamma_ri,
            self.gamma_ii,
            scale=self.scale,
            center=self.center,
            axis=axis,
        )
