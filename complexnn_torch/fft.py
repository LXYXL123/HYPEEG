import torch
from torch import nn

from .utils import from_complex, to_complex


def fft(z, axis=-1, norm="ortho"):
    return from_complex(torch.fft.fft(to_complex(z, axis=0), dim=axis, norm=norm), axis=0)


def ifft(z, axis=-1, norm="ortho"):
    return from_complex(torch.fft.ifft(to_complex(z, axis=0), dim=axis, norm=norm), axis=0)


def fft2(x, axes=(-2, -1), norm="ortho"):
    return from_complex(torch.fft.fft2(to_complex(x, axis=0), dim=axes, norm=norm), axis=0)


def ifft2(x, axes=(-2, -1), norm="ortho"):
    return from_complex(torch.fft.ifft2(to_complex(x, axis=0), dim=axes, norm=norm), axis=0)


class FFT(nn.Module):
    def __init__(self, axis=-1, norm="ortho"):
        super().__init__()
        self.axis = axis
        self.norm = norm

    def forward(self, x):
        return fft(x, axis=self.axis, norm=self.norm)


class IFFT(nn.Module):
    def __init__(self, axis=-1, norm="ortho"):
        super().__init__()
        self.axis = axis
        self.norm = norm

    def forward(self, x):
        return ifft(x, axis=self.axis, norm=self.norm)


class FFT2(nn.Module):
    def __init__(self, axes=(-2, -1), norm="ortho"):
        super().__init__()
        self.axes = axes
        self.norm = norm

    def forward(self, x):
        return fft2(x, axes=self.axes, norm=self.norm)


class IFFT2(nn.Module):
    def __init__(self, axes=(-2, -1), norm="ortho"):
        super().__init__()
        self.axes = axes
        self.norm = norm

    def forward(self, x):
        return ifft2(x, axes=self.axes, norm=self.norm)

