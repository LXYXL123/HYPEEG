import numpy as np
import torch
from torch import nn

from complexnn_torch import ComplexDense
from .dilated_conv_complex_geom import DilatedConvEncoderComplexGeom


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t : t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def analytic_signal_concat(x, dim=1):
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape={tuple(x.shape)}.")

    seq_len = x.size(dim)
    spectrum = torch.fft.fft(x, dim=dim)

    h = torch.zeros(seq_len, dtype=x.dtype, device=x.device)
    if seq_len % 2 == 0:
        h[0] = 1
        h[seq_len // 2] = 1
        h[1 : seq_len // 2] = 2
    else:
        h[0] = 1
        h[1 : (seq_len + 1) // 2] = 2

    shape = [1] * x.ndim
    shape[dim] = seq_len
    analytic = torch.fft.ifft(spectrum * h.view(*shape), dim=dim)
    return torch.cat([analytic.real, analytic.imag], dim=-1)


class TSEncoderComplexGeom(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        self.input_complex_fc = ComplexDense(hidden_dims, input_dim=input_dims)
        self.feature_extractor = DilatedConvEncoderComplexGeom(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x = x.clone()
        x[~nan_mask] = 0

        x = analytic_signal_concat(x)
        x = self.input_complex_fc(x)

        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        x = self.repr_dropout(self.feature_extractor(x))
        return x

