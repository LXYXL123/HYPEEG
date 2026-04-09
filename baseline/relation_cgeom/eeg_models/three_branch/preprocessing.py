from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn


def _prepare_real_fft_input(x: torch.Tensor) -> torch.Tensor:
    # cuFFT is noticeably more fragile on non-contiguous CUDA views.
    return x.to(dtype=torch.float32).contiguous()


def _run_fft_with_cpu_fallback(op, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    try:
        return op(x, *args, **kwargs)
    except RuntimeError as err:
        err_msg = str(err)
        if x.device.type != 'cuda' or ('cuFFT' not in err_msg and 'CUFFT' not in err_msg):
            raise
        x_cpu = x.to('cpu').contiguous()
        return op(x_cpu, *args, **kwargs).to(x.device)


def fft_bandpass(x: torch.Tensor, low_hz: float, high_hz: float, sampling_rate: float, dim: int = 1) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f'Expected [B, T, C] input, got shape={tuple(x.shape)}')
    x = _prepare_real_fft_input(x)
    seq_len = x.size(dim)
    spectrum = _run_fft_with_cpu_fallback(torch.fft.rfft, x, dim=dim)
    freqs = torch.fft.rfftfreq(seq_len, d=1.0 / sampling_rate).to(device=x.device, dtype=x.dtype)
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).to(spectrum.dtype)
    shape = [1] * x.ndim
    shape[dim] = mask.numel()
    filtered = spectrum * mask.view(*shape)
    return _run_fft_with_cpu_fallback(torch.fft.irfft, filtered.contiguous(), n=seq_len, dim=dim)


def analytic_signal_concat(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f'Expected [B, T, C] input, got shape={tuple(x.shape)}')
    x = _prepare_real_fft_input(x)
    seq_len = x.size(dim)
    spectrum = _run_fft_with_cpu_fallback(torch.fft.fft, x, dim=dim)

    h = torch.zeros(seq_len, dtype=x.dtype, device=x.device)
    if seq_len % 2 == 0:
        h[0] = 1
        h[seq_len // 2] = 1
        h[1: seq_len // 2] = 2
    else:
        h[0] = 1
        h[1: (seq_len + 1) // 2] = 2

    shape = [1] * x.ndim
    shape[dim] = seq_len
    analytic = _run_fft_with_cpu_fallback(torch.fft.ifft, (spectrum * h.view(*shape)).contiguous(), dim=dim)
    return torch.cat([analytic.real, analytic.imag], dim=-1)


class FilterBankHilbert(nn.Module):
    def __init__(self, bands: Iterable[Tuple[float, float]], sampling_rate: float):
        super().__init__()
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.sampling_rate = float(sampling_rate)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for low_hz, high_hz in self.bands:
            narrowband = fft_bandpass(x, low_hz=low_hz, high_hz=high_hz, sampling_rate=self.sampling_rate, dim=1)
            outputs.append(analytic_signal_concat(narrowband, dim=1))
        return outputs
