import math

import torch
from torch import nn

from .bn import ComplexBN, sqrt_init
from .utils import split_complex


def _canonical_axis(x, axis):
    return axis if axis >= 0 else x.ndim + axis


def _expand_param(param, x, axis):
    axis = _canonical_axis(x, axis)
    shape = [1] * x.ndim
    shape[axis] = param.shape[0]
    return param.reshape(shape)


def layernorm(x, axis, epsilon, gamma, beta):
    axis = _canonical_axis(x, axis)
    reduction_axes = tuple(i for i in range(1, x.ndim) if i != axis)
    mean = x.mean(dim=reduction_axes, keepdim=True)
    var = (x - mean).square().mean(dim=reduction_axes, keepdim=True)
    out = (x - mean) / torch.sqrt(var + epsilon)
    return out * _expand_param(gamma, x, axis) + _expand_param(beta, x, axis)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-4, axis=-1, beta_init="zeros", gamma_init="ones", gamma_regularizer=None, beta_regularizer=None):
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis
        self._initialized = False

    def _ensure_init(self, x):
        if self._initialized:
            return
        axis = _canonical_axis(x, self.axis)
        dim = x.size(axis)
        self.gamma = nn.Parameter(torch.ones(dim, dtype=x.dtype, device=x.device))
        self.beta = nn.Parameter(torch.zeros(dim, dtype=x.dtype, device=x.device))
        self._initialized = True

    def forward(self, x):
        self._ensure_init(x)
        return layernorm(x, self.axis, self.epsilon, self.gamma, self.beta)


class ComplexLayerNorm(nn.Module):
    def __init__(
        self,
        epsilon=1e-4,
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_diag_initializer=sqrt_init,
        gamma_off_initializer="zeros",
        beta_regularizer=None,
        gamma_diag_regularizer=None,
        gamma_off_regularizer=None,
        beta_constraint=None,
        gamma_diag_constraint=None,
        gamma_off_constraint=None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis
        self.center = center
        self.scale = scale
        self._initialized = False

    def _ensure_init(self, inputs):
        if self._initialized:
            return
        axis = _canonical_axis(inputs, self.axis)
        channels = inputs.size(axis)
        features = channels // 2
        if self.scale:
            self.gamma_rr = nn.Parameter(sqrt_init((features,), dtype=inputs.dtype, device=inputs.device))
            self.gamma_ii = nn.Parameter(sqrt_init((features,), dtype=inputs.dtype, device=inputs.device))
            self.gamma_ri = nn.Parameter(torch.zeros(features, dtype=inputs.dtype, device=inputs.device))
        else:
            self.gamma_rr = self.gamma_ii = self.gamma_ri = None
        if self.center:
            self.beta = nn.Parameter(torch.zeros(channels, dtype=inputs.dtype, device=inputs.device))
        else:
            self.beta = None
        self._initialized = True

    def forward(self, inputs):
        self._ensure_init(inputs)
        axis = _canonical_axis(inputs, self.axis)
        reduction_axes = tuple(i for i in range(1, inputs.ndim) if i != axis)
        mean = inputs.mean(dim=reduction_axes, keepdim=True)
        centred = inputs - mean if self.center else inputs
        real, imag = split_complex(centred, axis=axis)
        Vrr = real.square().mean(dim=reduction_axes) + self.epsilon
        Vii = imag.square().mean(dim=reduction_axes) + self.epsilon
        Vri = (real * imag).mean(dim=reduction_axes)
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
            layernorm=True,
            axis=axis,
        )

