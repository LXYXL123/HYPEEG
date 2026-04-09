from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn

from complexnn_torch import ComplexDense
from baseline.relation_cgeom.dilated_conv_complex_geom import DilatedConvEncoderComplexGeom
from baseline.relation_cgeom.eeg_models.three_branch.preprocessing import FilterBankHilbert


class SingleBandComplexGeomEncoder(nn.Module):
    def __init__(self, input_dims: int, repr_dims: int, hidden_dims: int, depth: int):
        super().__init__()
        self.input_complex_fc = ComplexDense(hidden_dims, input_dim=input_dims)
        self.feature_extractor = DilatedConvEncoderComplexGeom(
            hidden_dims,
            [hidden_dims] * depth + [repr_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        x_complex = self.input_complex_fc(x_complex)
        return self.repr_dropout(self.feature_extractor(x_complex))


class BandAttentionFusion(nn.Module):
    def __init__(
        self,
        repr_dims: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 4096,
    ):
        super().__init__()
        if repr_dims % num_heads != 0:
            raise ValueError(f'repr_dims={repr_dims} must be divisible by num_heads={num_heads}.')
        self.chunk_size = chunk_size
        self.attn = nn.MultiheadAttention(repr_dims, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(repr_dims)
        self.ffn = nn.Sequential(
            nn.Linear(repr_dims, repr_dims * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dims * 2, repr_dims),
        )
        self.norm2 = nn.LayerNorm(repr_dims)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_bands, repr_dims = stacked.shape
        tokens = stacked.reshape(batch_size * seq_len, num_bands, repr_dims)
        outputs = []
        for start in range(0, tokens.size(0), self.chunk_size):
            tokens_chunk = tokens[start:start + self.chunk_size]
            attn_out, _ = self.attn(tokens_chunk, tokens_chunk, tokens_chunk, need_weights=False)
            tokens_chunk = self.norm1(tokens_chunk + attn_out)
            ffn_out = self.ffn(tokens_chunk)
            tokens_chunk = self.norm2(tokens_chunk + ffn_out)
            outputs.append(tokens_chunk.mean(dim=1))
        return torch.cat(outputs, dim=0).reshape(batch_size, seq_len, repr_dims)


class FilterBankComplexGeomBranch(nn.Module):
    def __init__(
        self,
        input_dims: int,
        repr_dims: int,
        hidden_dims: int,
        depth: int,
        bands: Iterable[Tuple[float, float]],
        sampling_rate: float,
        share_band_encoder: bool = False,
        fusion_type: str = 'concat_linear',
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        attention_chunk_size: int = 4096,
    ):
        super().__init__()
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.share_band_encoder = share_band_encoder
        self.fusion_type = fusion_type
        self.filter_bank = FilterBankHilbert(self.bands, sampling_rate=sampling_rate)
        self.complex_repr_dims = repr_dims * 2

        if share_band_encoder:
            self.shared_encoder = SingleBandComplexGeomEncoder(
                input_dims=input_dims,
                repr_dims=repr_dims,
                hidden_dims=hidden_dims,
                depth=depth,
            )
            self.band_encoders = None
        else:
            self.shared_encoder = None
            self.band_encoders = nn.ModuleList([
                SingleBandComplexGeomEncoder(
                    input_dims=input_dims,
                    repr_dims=repr_dims,
                    hidden_dims=hidden_dims,
                    depth=depth,
                )
                for _ in self.bands
            ])

        if len(self.bands) > 1:
            if fusion_type == 'concat_linear':
                self.fusion_proj = nn.Linear(self.complex_repr_dims * len(self.bands), self.complex_repr_dims)
                self.band_gate = None
                self.band_attn = None
            elif fusion_type == 'gated_sum':
                self.fusion_proj = None
                self.band_gate = nn.Linear(self.complex_repr_dims * len(self.bands), len(self.bands))
                self.band_attn = None
            elif fusion_type == 'band_attention':
                self.fusion_proj = None
                self.band_gate = None
                self.band_attn = BandAttentionFusion(
                    repr_dims=self.complex_repr_dims,
                    num_heads=attention_heads,
                    dropout=attention_dropout,
                    chunk_size=attention_chunk_size,
                )
            else:
                raise ValueError(f'Unsupported fusion_type={fusion_type}')
        else:
            self.fusion_proj = None
            self.band_gate = None
            self.band_attn = None

    def _encode_single_band(self, idx: int, x_band: torch.Tensor) -> torch.Tensor:
        if self.shared_encoder is not None:
            return self.shared_encoder(x_band)
        return self.band_encoders[idx](x_band)

    def _fuse_bands(self, band_features: List[torch.Tensor]) -> torch.Tensor:
        if len(band_features) == 1:
            return band_features[0]

        if self.fusion_type == 'concat_linear':
            return self.fusion_proj(torch.cat(band_features, dim=-1))
        if self.fusion_type == 'band_attention':
            return self.band_attn(torch.stack(band_features, dim=2))

        stacked = torch.stack(band_features, dim=-2)
        fused = torch.cat(band_features, dim=-1)
        weights = torch.softmax(self.band_gate(fused), dim=-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=-2)

    def forward(self, x: torch.Tensor, return_band_features: bool = False):
        band_inputs = self.filter_bank(x)
        band_features = [
            self._encode_single_band(idx, band_input)
            for idx, band_input in enumerate(band_inputs)
        ]
        fused = self._fuse_bands(band_features)
        if return_band_features:
            return fused, band_features
        return fused
