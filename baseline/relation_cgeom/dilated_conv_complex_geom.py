import torch
from torch import nn
import torch.nn.functional as F

from complexnn_torch import ComplexBatchNormalization, ComplexConv1D
from complexnn_torch.utils import combine_complex, split_complex


def crelu(x):
    real, imag = split_complex(x, axis=-1)
    return combine_complex(F.relu(real), F.relu(imag), axis=-1)


class SamePadConvComplexGeom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = ComplexConv1D(
            out_channels,
            kernel_size,
            in_channels=in_channels,
            padding='same',
            dilation_rate=dilation,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, : -self.remove]
        return out


class ConvBlockComplexGeom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConvComplexGeom(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = ComplexBatchNormalization(num_features=out_channels, axis=-1)
        self.conv2 = SamePadConvComplexGeom(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = ComplexBatchNormalization(num_features=out_channels, axis=-1)
        self.projector = (
            ComplexConv1D(out_channels, 1, in_channels=in_channels, padding='same')
            if in_channels != out_channels or final else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = crelu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = crelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class DilatedConvEncoderComplexGeom(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlockComplexGeom(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)
