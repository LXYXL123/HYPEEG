from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from baseline.relation_cgeom.eeg_models.three_branch.setup_conditioned_relation import masked_mean_pool
from baseline.relation_cgeom.eeg_models.three_branch.variable_channel_frontend import (
    SetupConditionedChannelAggregator,
)


class SpectralTokenizer(nn.Module):
    """Convert variable-channel EEG into lightweight spectral tokens.

    x: [B, T, C]
    spec_tokens: [B, C, Nw, Ds]
    spec_mask: [B, Nw]
    """

    def __init__(
        self,
        spec_dims: int = 64,
        win_len: int = 64,
        stride: int = 32,
        sampling_rate: float = 256.0,
        freq_low: float = 4.0,
        freq_high: float = 40.0,
    ):
        super().__init__()
        self.spec_dims = int(spec_dims)
        self.win_len = int(win_len)
        self.stride = int(stride)
        self.sampling_rate = float(sampling_rate)
        self.freq_low = float(freq_low)
        self.freq_high = float(freq_high)

        if self.win_len <= 0 or self.stride <= 0:
            raise ValueError("SpectralTokenizer requires positive win_len and stride.")

        freqs = torch.fft.rfftfreq(self.win_len, d=1.0 / self.sampling_rate)
        freq_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        if not bool(freq_mask.any()):
            raise ValueError(
                f"No FFT bins selected for freq range [{self.freq_low}, {self.freq_high}] "
                f"with win_len={self.win_len}, fs={self.sampling_rate}."
            )
        self.register_buffer("freq_mask", freq_mask, persistent=False)
        self.register_buffer("window", torch.hann_window(self.win_len), persistent=False)

        in_dims = int(freq_mask.sum().item()) * 3
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dims),
            nn.Linear(in_dims, self.spec_dims),
            nn.GELU(),
            nn.Linear(self.spec_dims, self.spec_dims),
        )

    def _pad_time(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len = x.size(1)
        if seq_len >= self.win_len:
            return x, mask

        pad_len = self.win_len - seq_len
        x = F.pad(x, (0, 0, 0, pad_len))
        if mask is not None:
            mask = F.pad(mask, (0, pad_len), value=False)
        return x, mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.shape
        x, mask = self._pad_time(x, mask)

        # [B, T, C] -> [B, C, Nw, win_len]
        x_ch = x.permute(0, 2, 1).contiguous()
        x_win = x_ch.unfold(dimension=-1, size=self.win_len, step=self.stride)
        n_windows = x_win.size(2)
        x_win = x_win * self.window.to(device=x.device, dtype=x.dtype)

        # Keep amplitude and phase information without a heavy complex network.
        spectrum = torch.fft.rfft(x_win.to(torch.float32).contiguous(), dim=-1)
        spectrum = spectrum[..., self.freq_mask]
        amplitude = torch.log1p(spectrum.abs())
        phase = torch.angle(spectrum)
        spec_feat = torch.cat([amplitude, torch.sin(phase), torch.cos(phase)], dim=-1)
        spec_tokens = self.proj(spec_feat)  # [B, C, Nw, Ds]

        if mask is None:
            spec_mask = torch.ones(batch_size, n_windows, device=x.device, dtype=torch.bool)
        else:
            mask_win = mask.unfold(dimension=1, size=self.win_len, step=self.stride)
            spec_mask = mask_win.to(torch.float32).mean(dim=-1) > 0.5
        return spec_tokens, spec_mask


class SpectralMixerBlock(nn.Module):
    """Small per-channel spectral token mixer over the window axis."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*C, Nw, Ds]
        y = self.norm1(x).transpose(1, 2)
        y = self.dwconv(y).transpose(1, 2)
        x = x + y
        return x + self.ffn(self.norm2(x))


class SpectralPerChannelMixer(nn.Module):
    """Shared lightweight spectral backbone for each channel.

    spec_tokens: [B, C, Nw, Ds]
    h_spec_ch: [B, C, Nw, Ds]
    """

    def __init__(self, spec_dims: int = 64, depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpectralMixerBlock(spec_dims, dropout=dropout)
            for _ in range(max(0, int(depth)))
        ])
        self.out_norm = nn.LayerNorm(spec_dims)

    def forward(self, spec_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_windows, spec_dims = spec_tokens.shape
        x = spec_tokens.reshape(batch_size * n_channels, n_windows, spec_dims)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x.reshape(batch_size, n_channels, n_windows, spec_dims)


class SpectralChannelAggregator(SetupConditionedChannelAggregator):
    """Setup-conditioned channel attention pooling for spectral tokens."""


class SetupConditionedSpectralSelector(nn.Module):
    """Select important spectral window tokens under the current setup.

    h_spec: [B, Nw, Ds]
    z_setup: [B, S]
    h_spec_sel: [B, Nw, Ds]
    spec_global: [B, D]
    """

    def __init__(self, spec_dims: int, repr_dims: int, setup_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(spec_dims + setup_dim),
            nn.Linear(spec_dims + setup_dim, spec_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spec_dims, spec_dims),
        )
        self.global_norm = nn.LayerNorm(spec_dims)
        self.to_relation = nn.Linear(spec_dims, repr_dims)

    def forward(
        self,
        h_spec: torch.Tensor,
        z_setup: torch.Tensor,
        spec_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        z = z_setup.unsqueeze(1).expand(-1, h_spec.size(1), -1)
        spec_token_gate = torch.sigmoid(self.gate(torch.cat([h_spec, z], dim=-1)))
        h_spec_sel = h_spec * spec_token_gate
        spec_pooled = masked_mean_pool(self.global_norm(h_spec_sel), spec_mask)
        spec_global = self.to_relation(spec_pooled)
        return {
            "h_spec": h_spec_sel,
            "spec_global": spec_global,
            "spec_token_gate": spec_token_gate.detach(),
        }


class SimpleFinalFusionGate(nn.Module):
    """Light raw + global-relation fusion.

    main_repr = raw_residual + alpha * rel_residual
    """

    def __init__(self, repr_dims: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(repr_dims * 3),
            nn.Linear(repr_dims * 3, repr_dims),
            nn.GELU(),
            nn.Linear(repr_dims, 1),
        )
        # Start close to raw-only so the random relation path does not dominate
        # early supervised fine-tuning.
        nn.init.constant_(self.gate[-1].bias, -2.1972246)

    def forward(
        self,
        raw_residual: torch.Tensor,
        rel_residual: torch.Tensor,
        raw_global: torch.Tensor,
        spec_global: torch.Tensor,
        relation_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.gate(torch.cat([raw_global, spec_global, relation_global], dim=-1)))  # [B, 1]
        main_repr = raw_residual + alpha.unsqueeze(1) * rel_residual
        return main_repr, alpha
