from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from complexnn_torch import ComplexDense
from complexnn_torch.utils import combine_complex, split_complex
from baseline.relation_cgeom.dilated_conv import DilatedConvEncoder
from baseline.relation_cgeom.dilated_conv_complex_geom import DilatedConvEncoderComplexGeom
from baseline.relation_cgeom.eeg_models.three_branch.preprocessing import FilterBankHilbert


def _sanitize_num_heads(feature_dim: int, num_heads: int) -> int:
    num_heads = max(1, int(num_heads))
    return num_heads if feature_dim % num_heads == 0 else 1


def _default_channel_ids(batch_size: int, n_channels: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n_channels, device=device).unsqueeze(0).expand(batch_size, -1)


class RawPerChannelStem(nn.Module):
    """Shared per-channel temporal encoder.

    x: [B, T, C]
    h_raw_ch: [B, C, T, D]
    """

    def __init__(self, repr_dims: int, hidden_dims: int, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_fc = nn.Linear(1, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * max(0, int(depth)) + [repr_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_channels = x.shape
        x_ch = x.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len, 1)
        h = self.input_fc(x_ch)  # [B*C, T, H]
        h = h.transpose(1, 2)  # [B*C, H, T]
        h = self.repr_dropout(self.feature_extractor(h))  # [B*C, D, T]
        h = h.transpose(1, 2)  # [B*C, T, D]
        return h.reshape(batch_size, n_channels, seq_len, -1)


class SetupConditionedChannelAggregator(nn.Module):
    """Attention-pool variable channel sets into a fixed feature sequence.

    h_ch: [B, C, T, D]
    channel_ids: [B, C]
    z_setup: [B, S]
    pooled: [B, T, D]
    attn: [B, H, T, C]
    """

    def __init__(
        self,
        feature_dim: int,
        setup_dim: int,
        channel_vocab_size: int = 128,
        channel_emb_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.setup_dim = int(setup_dim)
        self.channel_emb_dim = int(channel_emb_dim or feature_dim)
        self.num_heads = _sanitize_num_heads(self.feature_dim, num_heads)
        self.head_dim = self.feature_dim // self.num_heads

        self.channel_embed = nn.Embedding(int(channel_vocab_size), self.channel_emb_dim)
        self.value_proj = nn.Linear(self.feature_dim, self.feature_dim)
        # Factorized scoring avoids materializing a large [B, C, T, D+E+S]
        # tensor while still conditioning channel attention on signal, channel
        # identity, and sample setup.
        self.feature_score = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_heads),
        )
        self.channel_score = nn.Linear(self.channel_emb_dim, self.num_heads)
        self.setup_score = nn.Sequential(
            nn.LayerNorm(self.setup_dim),
            nn.Linear(self.setup_dim, self.num_heads),
        )
        self.score_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)

    def forward(
        self,
        h_ch: torch.Tensor,
        channel_ids: Optional[torch.Tensor],
        z_setup: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_channels, seq_len, _ = h_ch.shape
        if channel_ids is None:
            channel_ids = _default_channel_ids(batch_size, n_channels, h_ch.device)
        channel_ids = channel_ids.to(device=h_ch.device, dtype=torch.long)

        ch_emb = self.channel_embed(channel_ids)  # [B, C, E]
        scores = (
            self.feature_score(h_ch)
            + self.channel_score(ch_emb).unsqueeze(2)
            + self.setup_score(z_setup).unsqueeze(1).unsqueeze(2)
        )  # [B, C, T, H]
        scores = self.score_dropout(scores)
        attn = torch.softmax(scores, dim=1)

        values = self.value_proj(h_ch).reshape(
            batch_size,
            n_channels,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        pooled = (attn.unsqueeze(-1) * values).sum(dim=1).reshape(batch_size, seq_len, self.feature_dim)
        pooled = self.out_norm(self.out_proj(pooled))
        return pooled, attn.permute(0, 3, 2, 1).contiguous().detach(), ch_emb.mean(dim=1).detach()


class RawChannelAggregator(SetupConditionedChannelAggregator):
    """Raw-side setup-conditioned channel attention pooling."""


class ComplexPerChannelStem(nn.Module):
    """Shared per-channel complex temporal encoder for one frequency band.

    x_complex: [B, T, 2C] with real/imag concatenated on the last dim.
    h_band_ch: [B, C, T, 2D]
    """

    def __init__(self, repr_dims: int, hidden_dims: int, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_complex_fc = ComplexDense(hidden_dims, input_dim=1)
        self.feature_extractor = DilatedConvEncoderComplexGeom(
            hidden_dims,
            [hidden_dims] * max(0, int(depth)) + [repr_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=dropout)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        real, imag = split_complex(x_complex, axis=-1)  # each [B, T, C]
        batch_size, seq_len, n_channels = real.shape
        real_ch = real.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len, 1)
        imag_ch = imag.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len, 1)
        x_ch = combine_complex(real_ch, imag_ch, axis=-1)  # [B*C, T, 2]
        h = self.input_complex_fc(x_ch)  # [B*C, T, 2H]
        h = self.repr_dropout(self.feature_extractor(h))  # [B*C, T, 2D]
        return h.reshape(batch_size, n_channels, seq_len, -1)


class ComplexChannelAggregator(SetupConditionedChannelAggregator):
    """Complex-side setup-conditioned channel attention pooling."""


class VariableChannelRawBranch(nn.Module):
    """Raw branch with parameters independent of the input channel count."""

    def __init__(
        self,
        repr_dims: int,
        hidden_dims: int,
        setup_dim: int,
        channel_vocab_size: int = 128,
        channel_attn_heads: int = 4,
        per_channel_stem_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = RawPerChannelStem(
            repr_dims=repr_dims,
            hidden_dims=hidden_dims,
            depth=per_channel_stem_depth,
            dropout=dropout,
        )
        self.aggregator = RawChannelAggregator(
            feature_dim=repr_dims,
            setup_dim=setup_dim,
            channel_vocab_size=channel_vocab_size,
            num_heads=channel_attn_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        channel_ids: Optional[torch.Tensor],
        z_setup: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_raw_ch = self.stem(x)  # [B, C, T, D]
        h_raw, attn, ch_emb_summary = self.aggregator(h_raw_ch, channel_ids, z_setup)
        return h_raw, {
            'raw_channel_attn_weights': attn,
            'channel_name_emb_summary': ch_emb_summary,
        }


class VariableChannelFilterBankComplexGeomBranch(nn.Module):
    """Filter-bank complex branch with variable-channel per-band frontends."""

    def __init__(
        self,
        repr_dims: int,
        hidden_dims: int,
        setup_dim: int,
        bands: Iterable[Tuple[float, float]],
        sampling_rate: float,
        channel_vocab_size: int = 128,
        channel_attn_heads: int = 4,
        per_channel_stem_depth: int = 2,
        share_band_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.share_band_encoder = bool(share_band_encoder)
        self.filter_bank = FilterBankHilbert(self.bands, sampling_rate=sampling_rate)
        self.complex_repr_dims = repr_dims * 2

        if self.share_band_encoder:
            self.shared_stem = ComplexPerChannelStem(
                repr_dims=repr_dims,
                hidden_dims=hidden_dims,
                depth=per_channel_stem_depth,
                dropout=dropout,
            )
            self.band_stems = None
        else:
            self.shared_stem = None
            self.band_stems = nn.ModuleList([
                ComplexPerChannelStem(
                    repr_dims=repr_dims,
                    hidden_dims=hidden_dims,
                    depth=per_channel_stem_depth,
                    dropout=dropout,
                )
                for _ in self.bands
            ])

        self.band_aggregators = nn.ModuleList([
            ComplexChannelAggregator(
                feature_dim=self.complex_repr_dims,
                setup_dim=setup_dim,
                channel_vocab_size=channel_vocab_size,
                num_heads=channel_attn_heads,
                dropout=dropout,
            )
            for _ in self.bands
        ])

    def _stem(self, idx: int) -> ComplexPerChannelStem:
        if self.shared_stem is not None:
            return self.shared_stem
        return self.band_stems[idx]

    def forward(
        self,
        x: torch.Tensor,
        channel_ids: Optional[torch.Tensor],
        z_setup: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        band_inputs = self.filter_bank(x)
        band_features = []
        aux = {}
        for idx, band_input in enumerate(band_inputs):
            h_band_ch = self._stem(idx)(band_input)  # [B, C, T, 2D]
            h_band, attn, _ = self.band_aggregators[idx](h_band_ch, channel_ids, z_setup)
            band_features.append(h_band)  # [B, T, 2D]
            aux[f'complex_channel_attn_weights_band{idx + 1}'] = attn
        return band_features, aux
