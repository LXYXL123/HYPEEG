from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_relation_branch import HyperbolicRelationBranch


def masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool sequence features with optional valid-time mask.

    x: [B, T, D]
    mask: [B, T]
    pooled: [B, D]
    """
    if mask is None:
        return x.mean(dim=1)
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class TinySelfAttentionBlock(nn.Module):
    """One-layer token mixer used only for setup/relation summary tokens."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)
        return self.norm2(tokens + self.ffn(tokens))


class SetupEncoder(nn.Module):
    """Encode static setup metadata plus shallow per-sample EEG statistics.

    The channel-name path is implemented as channel-id embeddings. The ids are
    the standardized electrode ids emitted by the dataset adapter.
    """

    def __init__(
        self,
        input_dims: int,
        setup_dim: int = 128,
        meta_dim: int = 128,
        ctx_dim: int = 128,
        channel_vocab_size: int = 128,
        sampling_rate: float = 256.0,
        bands: tuple[tuple[float, float], ...] = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0)),
    ):
        super().__init__()
        self.input_dims = int(input_dims)
        self.setup_dim = int(setup_dim)
        self.meta_dim = int(meta_dim)
        self.ctx_dim = int(ctx_dim)
        self.channel_vocab_size = int(channel_vocab_size)
        self.sampling_rate = float(sampling_rate)
        self.bands = tuple((float(low), float(high)) for low, high in bands)

        self.channel_embed = nn.Embedding(self.channel_vocab_size, self.meta_dim)
        self.count_mlp = self._scalar_mlp(self.meta_dim)
        self.fs_mlp = self._scalar_mlp(self.meta_dim)
        self.window_mlp = self._scalar_mlp(self.meta_dim)
        self.meta_mlp = nn.Sequential(
            nn.LayerNorm(self.meta_dim * 4),
            nn.Linear(self.meta_dim * 4, self.meta_dim),
            nn.GELU(),
            nn.Linear(self.meta_dim, self.meta_dim),
        )

        # global_energy(1) + channel_var_summary(3) + coarse_band_power(3)
        # + channel_corr_summary(2) = 9 shallow context features.
        self.ctx_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.ctx_dim),
            nn.GELU(),
            nn.Linear(self.ctx_dim, self.ctx_dim),
        )
        self.setup_mlp = nn.Sequential(
            nn.LayerNorm(self.meta_dim + self.ctx_dim),
            nn.Linear(self.meta_dim + self.ctx_dim, self.setup_dim),
            nn.GELU(),
            nn.Linear(self.setup_dim, self.setup_dim),
        )

    @staticmethod
    def _scalar_mlp(out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _resolve_channel_ids(self, x: torch.Tensor, setup: Optional[dict]) -> torch.Tensor:
        batch_size, _, n_channels = x.shape
        channel_ids = None if setup is None else setup.get('channel_ids')
        if channel_ids is None:
            channel_ids = torch.arange(n_channels, device=x.device).unsqueeze(0).expand(batch_size, -1)
        else:
            channel_ids = channel_ids.to(device=x.device, dtype=torch.long)
            if channel_ids.ndim == 1:
                channel_ids = channel_ids.unsqueeze(0).expand(batch_size, -1)
        return channel_ids.clamp(min=0, max=self.channel_vocab_size - 1)

    def _scalar_features(self, x: torch.Tensor, setup: Optional[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_channels = x.shape
        device = x.device
        dtype = x.dtype
        channel_count = torch.full((batch_size, 1), float(n_channels) / 100.0, device=device, dtype=dtype)
        sampling_rate = self.sampling_rate if setup is None or setup.get('sampling_rate') is None else float(setup['sampling_rate'])
        fs = torch.full((batch_size, 1), sampling_rate / 256.0, device=device, dtype=dtype)
        if setup is not None and setup.get('window_len') is not None:
            window_len_samples = float(setup['window_len'])
        else:
            window_len_samples = float(seq_len)
        window_len = torch.full((batch_size, 1), window_len_samples / max(sampling_rate, 1.0), device=device, dtype=dtype)
        return channel_count, fs, window_len

    def _coarse_band_power(self, x: torch.Tensor, sampling_rate: float) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.to(torch.float32).contiguous(), dim=1)
        power = spectrum.abs().pow(2)  # [B, F, C]
        freqs = torch.fft.rfftfreq(x.size(1), d=1.0 / sampling_rate).to(device=x.device)
        band_powers = []
        for low_hz, high_hz in self.bands[:3]:
            freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
            if freq_mask.any():
                band_power = power[:, freq_mask, :].mean(dim=(1, 2))
            else:
                band_power = power.new_zeros(x.size(0))
            band_powers.append(torch.log1p(band_power).to(dtype=x.dtype))
        while len(band_powers) < 3:
            band_powers.append(x.new_zeros(x.size(0)))
        return torch.stack(band_powers[:3], dim=-1)  # [B, 3]

    def _context_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor], setup: Optional[dict]) -> torch.Tensor:
        batch_size, seq_len, n_channels = x.shape
        weights = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
        if mask is not None:
            weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)

        mean = (x * weights).sum(dim=1) / denom  # [B, C]
        centered = (x - mean.unsqueeze(1)) * weights
        var = centered.pow(2).sum(dim=1) / denom  # [B, C]
        var_summary = torch.stack(
            [
                torch.log1p(var.mean(dim=1)),
                torch.log1p(var.std(dim=1, unbiased=False)),
                torch.log1p(var.max(dim=1).values),
            ],
            dim=-1,
        )  # [B, 3]

        global_energy = torch.log1p((x.pow(2) * weights).sum(dim=(1, 2)) / (denom.squeeze(-1) * n_channels).clamp_min(1.0))
        global_energy = global_energy.unsqueeze(-1)  # [B, 1]

        sampling_rate = self.sampling_rate if setup is None or setup.get('sampling_rate') is None else float(setup['sampling_rate'])
        band_power = self._coarse_band_power(x * weights, sampling_rate=sampling_rate)  # [B, 3]

        std = var.clamp_min(1e-6).sqrt()
        normed = centered / std.unsqueeze(1)
        corr = torch.einsum('btc,btd->bcd', normed, normed) / denom.clamp_min(1.0).unsqueeze(-1)
        if n_channels > 1:
            offdiag_mask = ~torch.eye(n_channels, device=x.device, dtype=torch.bool)
            offdiag = corr[:, offdiag_mask]  # [B, C * (C - 1)]
            corr_summary = torch.stack(
                [offdiag.abs().mean(dim=1), offdiag.std(dim=1, unbiased=False)],
                dim=-1,
            )  # [B, 2]
        else:
            corr_summary = x.new_zeros(batch_size, 2)

        return torch.cat([global_energy, var_summary, band_power, corr_summary], dim=-1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, setup: Optional[dict] = None) -> dict[str, torch.Tensor]:
        """Return setup representations.

        x: [B, T, C]
        z_meta/z_ctx/z_setup: [B, 128] by default
        """
        channel_ids = self._resolve_channel_ids(x, setup)
        u_ch = self.channel_embed(channel_ids).mean(dim=1)  # [B, meta_dim]
        channel_count_value = x.new_full((x.size(0), 1), float(x.size(-1)))
        channel_count, fs, window_len = self._scalar_features(x, setup)
        u_cnt = self.count_mlp(channel_count)
        u_fs = self.fs_mlp(fs)
        u_win = self.window_mlp(window_len)
        z_meta = self.meta_mlp(torch.cat([u_ch, u_cnt, u_fs, u_win], dim=-1))

        ctx_features = self._context_stats(x, mask, setup)
        z_ctx = self.ctx_mlp(ctx_features)
        z_setup = self.setup_mlp(torch.cat([z_meta, z_ctx], dim=-1))
        return {
            'z_meta': z_meta,
            'z_ctx': z_ctx,
            'z_setup': z_setup,
            'channel_name_emb_summary': u_ch,
            'channel_count': channel_count_value,
        }


class RawConditioner(nn.Module):
    """Setup-conditioned FiLM over the last feature dimension."""

    def __init__(self, input_dims: int = None, setup_dim: int = 128, scale: float = 0.1, feature_dim: int = None):
        super().__init__()
        feature_dim = feature_dim if feature_dim is not None else input_dims
        if feature_dim is None:
            raise ValueError('RawConditioner requires feature_dim or input_dims.')
        self.feature_dim = int(feature_dim)
        self.scale = float(scale)
        self.to_film = nn.Sequential(
            nn.LayerNorm(setup_dim),
            nn.Linear(setup_dim, self.feature_dim * 2),
        )

    def forward(self, x: torch.Tensor, z_setup: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_film(z_setup).chunk(2, dim=-1)  # each [B, D]
        gamma = self.scale * torch.tanh(gamma).unsqueeze(1)
        beta = self.scale * beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class SetupConditionedBandFusion(nn.Module):
    """Band fusion conditioned on sample setup/context tokens."""

    def __init__(self, repr_dims: int, setup_dim: int, num_bands: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.repr_dims = int(repr_dims)
        self.complex_dims = self.repr_dims * 2
        self.num_bands = int(num_bands)
        self.summary_norm = nn.LayerNorm(self.complex_dims)
        self.setup_proj = nn.Linear(setup_dim, self.complex_dims)
        self.band_attn = TinySelfAttentionBlock(self.complex_dims, num_heads=num_heads, dropout=dropout)
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(self.complex_dims + setup_dim),
            nn.Linear(self.complex_dims + setup_dim, self.complex_dims // 2),
            nn.GELU(),
            nn.Linear(self.complex_dims // 2, 1),
        )
        self.band_to_rel = nn.Linear(self.complex_dims, self.repr_dims)
        self.complex_norm = nn.LayerNorm(self.complex_dims)
        self.complex_to_rel = nn.Linear(self.complex_dims, self.repr_dims)

    def forward(self, band_features: list[torch.Tensor], z_setup: torch.Tensor, mask: Optional[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if len(band_features) != self.num_bands:
            raise ValueError(f'Expected {self.num_bands} band features, got {len(band_features)}')

        band_summaries = [self.summary_norm(masked_mean_pool(feature, mask)) for feature in band_features]  # each [B, 2D]
        setup_summary = self.setup_proj(z_setup)  # [B, 2D]
        tokens = torch.stack([*band_summaries, setup_summary], dim=1)  # [B, 4, 2D]
        refined = self.band_attn(tokens)
        refined_bands = [refined[:, idx, :] for idx in range(self.num_bands)]  # list[[B, 2D]]

        scores = [
            self.score_mlp(torch.cat([refined_band, z_setup], dim=-1)).squeeze(-1)
            for refined_band in refined_bands
        ]
        band_weights = torch.softmax(torch.stack(scores, dim=-1), dim=-1)  # [B, 3]

        stacked = torch.stack(band_features, dim=2)  # [B, T, 3, 2D]
        h_complex = (stacked * band_weights[:, None, :, None]).sum(dim=2)  # [B, T, 2D]

        band_rel = [self.band_to_rel(refined_band) for refined_band in refined_bands]  # list[[B, D]]
        complex_token = self.complex_to_rel(self.complex_norm(h_complex))  # [B, T, D]
        complex_global = masked_mean_pool(complex_token, mask)  # [B, D]

        return {
            'h_complex': h_complex,
            'complex_global': complex_global,
            'band_summaries': band_rel,
            'band_weights': band_weights,
        }


class GlobalRelationPath(nn.Module):
    """High-level raw/complex/setup relation path."""

    def __init__(
        self,
        repr_dims: int,
        depth: int = 1,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_attn = TinySelfAttentionBlock(repr_dims, num_heads=num_heads, dropout=dropout)
        self.hyperbolic = HyperbolicRelationBranch(
            repr_dims=repr_dims,
            depth=depth,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            num_heads=num_heads,
        )

    def forward(self, raw_global: torch.Tensor, complex_global: torch.Tensor, setup_token_global: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([raw_global, complex_global, setup_token_global], dim=1)  # [B, 3, D]
        tokens = self.token_attn(tokens)
        return self.hyperbolic.forward_global([tokens[:, 0], tokens[:, 1], tokens[:, 2]])  # [B, D]


class FineRelationPath(nn.Module):
    """Band-level setup-conditioned relation path."""

    def __init__(
        self,
        repr_dims: int,
        depth: int = 1,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight_embed = nn.Sequential(
            nn.Linear(1, repr_dims),
            nn.GELU(),
            nn.Linear(repr_dims, repr_dims),
        )
        self.token_attn = TinySelfAttentionBlock(repr_dims, num_heads=num_heads, dropout=dropout)
        self.hyperbolic = HyperbolicRelationBranch(
            repr_dims=repr_dims,
            depth=depth,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            num_heads=num_heads,
        )

    def forward(self, band_summaries: list[torch.Tensor], setup_token_fine: torch.Tensor, band_weights: torch.Tensor) -> torch.Tensor:
        if len(band_summaries) != 3:
            raise ValueError(f'FineRelationPath expects 3 band summaries, got {len(band_summaries)}')
        weighted_bands = [
            band + self.weight_embed(band_weights[:, idx:idx + 1])
            for idx, band in enumerate(band_summaries)
        ]
        tokens = torch.stack([*weighted_bands, setup_token_fine], dim=1)  # [B, 4, D]
        tokens = self.token_attn(tokens)
        return self.hyperbolic.forward_global([tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]])  # [B, D]


class RelationFusionGate(nn.Module):
    """Sample-level global/fine relation fusion."""

    def __init__(self, repr_dims: int, setup_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(repr_dims * 2 + setup_dim),
            nn.Linear(repr_dims * 2 + setup_dim, repr_dims),
            nn.GELU(),
            nn.Linear(repr_dims, 2),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(repr_dims * 2),
            nn.Linear(repr_dims * 2, repr_dims),
        )

    def forward(self, global_repr: torch.Tensor, fine_repr: torch.Tensor, z_setup: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(torch.cat([global_repr, fine_repr, z_setup], dim=-1)), dim=-1)  # [B, 2]
        residual = self.residual(torch.cat([global_repr, fine_repr], dim=-1))  # [B, D]
        relation_global = (
            weights[:, 0:1] * global_repr
            + weights[:, 1:2] * fine_repr
            + residual
        )
        return relation_global, weights


class FinalFusionGate(nn.Module):
    """Sample-level raw/relation residual fusion."""

    def __init__(self, repr_dims: int, setup_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(repr_dims * 3 + setup_dim),
            nn.Linear(repr_dims * 3 + setup_dim, repr_dims),
            nn.GELU(),
            nn.Linear(repr_dims, 2),
        )

    def forward(
        self,
        raw_residual: torch.Tensor,
        rel_residual: torch.Tensor,
        raw_global: torch.Tensor,
        complex_global: torch.Tensor,
        relation_global: torch.Tensor,
        z_setup: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(
            self.gate(torch.cat([raw_global, complex_global, relation_global, z_setup], dim=-1)),
            dim=-1,
        )  # [B, 2]
        main_repr = weights[:, 0:1].unsqueeze(1) * raw_residual + weights[:, 1:2].unsqueeze(1) * rel_residual
        return main_repr, weights
