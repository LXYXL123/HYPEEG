import math

import torch
from torch import nn

from .utils import combine_complex, split_complex


class ComplexDense(nn.Module):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        init_criterion="he",
        kernel_initializer="complex",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        input_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.seed = seed
        self.input_dim = input_dim
        self._initialized = False
        if self.input_dim is not None:
            self._initialize_parameters(self.input_dim)

    def _initialize_parameters(self, input_dim, dtype=None, device=None):
        dtype = dtype or torch.float32
        device = device or "cpu"
        self.real_kernel = nn.Parameter(torch.empty(input_dim, self.units, dtype=dtype, device=device))
        self.imag_kernel = nn.Parameter(torch.empty(input_dim, self.units, dtype=dtype, device=device))
        bound = math.sqrt(1.0 / input_dim)
        nn.init.uniform_(self.real_kernel, -bound, bound)
        nn.init.uniform_(self.imag_kernel, -bound, bound)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units * 2, dtype=dtype, device=device))
        else:
            self.bias = None
        self._initialized = True

    def _ensure_init(self, inputs):
        if self._initialized:
            return
        input_dim = inputs.shape[-1] // 2
        self._initialize_parameters(input_dim, dtype=inputs.dtype, device=inputs.device)

    def forward(self, inputs):
        self._ensure_init(inputs)
        real_input, imag_input = split_complex(inputs, axis=-1)
        real_output = real_input @ self.real_kernel - imag_input @ self.imag_kernel
        imag_output = real_input @ self.imag_kernel + imag_input @ self.real_kernel
        output = combine_complex(real_output, imag_output, axis=-1)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
