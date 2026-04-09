import torch
from torch import nn


def _spectral_mask(length, topf, device, dtype):
    mask = torch.zeros(length, device=device, dtype=dtype)
    if topf <= 0 or length < 2 * topf:
        return torch.ones(length, device=device, dtype=dtype)
    mask[:topf] = 1
    mask[-topf:] = 1
    return mask


class SpectralPooling1D(nn.Module):
    def __init__(self, topf=None, gamma=None):
        super().__init__()
        if topf is None and gamma is None:
            raise RuntimeError("Must provide either topf= or gamma=.")
        self.topf = None if topf is None else int(topf[0] if isinstance(topf, (tuple, list)) else topf) // 2
        self.gamma = None if gamma is None else float(gamma[0] if isinstance(gamma, (tuple, list)) else gamma) / 2.0

    def forward(self, x):
        length = x.size(-1)
        topf = self.topf if self.topf is not None else int(self.gamma * length)
        return x * _spectral_mask(length, topf, x.device, x.dtype).view(*([1] * (x.ndim - 1)), length)


class SpectralPooling2D(nn.Module):
    def __init__(self, topf=None, gamma=None):
        super().__init__()
        if topf is None and gamma is None:
            raise RuntimeError("Must provide either topf= or gamma=.")
        if topf is not None:
            self.topf = (int(topf[0]) // 2, int(topf[1]) // 2)
            self.gamma = None
        else:
            self.topf = None
            self.gamma = (float(gamma[0]) / 2.0, float(gamma[1]) / 2.0)

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        if self.topf is not None:
            topf_h, topf_w = self.topf
        else:
            topf_h, topf_w = int(self.gamma[0] * h), int(self.gamma[1] * w)
        mask_h = _spectral_mask(h, topf_h, x.device, x.dtype).view(*([1] * (x.ndim - 2)), h, 1)
        mask_w = _spectral_mask(w, topf_w, x.device, x.dtype).view(*([1] * (x.ndim - 2)), 1, w)
        return x * mask_h * mask_w
