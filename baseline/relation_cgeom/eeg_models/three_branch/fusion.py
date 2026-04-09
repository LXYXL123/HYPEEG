from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemporalFrequencyFusion(nn.Module):
    def __init__(self, repr_dims: int, fusion_type: str = 'concat_linear'):
        super().__init__()
        self.repr_dims = repr_dims
        self.fusion_type = fusion_type
        if fusion_type == 'concat_linear':
            self.proj = nn.Linear(repr_dims * 2, repr_dims)
        elif fusion_type == 'gated':
            self.gate = nn.Linear(repr_dims * 2, repr_dims)
        else:
            raise ValueError(f'Unsupported fusion_type={fusion_type}')

    def forward(self, h_raw: Optional[torch.Tensor], h_complex: Optional[torch.Tensor]) -> torch.Tensor:
        if h_raw is None and h_complex is None:
            raise ValueError('At least one branch output must be provided.')
        if h_raw is None:
            return h_complex
        if h_complex is None:
            return h_raw

        fused = torch.cat([h_raw, h_complex], dim=-1)
        if self.fusion_type == 'concat_linear':
            return self.proj(fused)

        gate = torch.sigmoid(self.gate(fused))
        return gate * h_raw + (1.0 - gate) * h_complex


class HeadFusion(nn.Module):
    def __init__(self, repr_dims: int, fusion_type: str = 'concat_linear'):
        super().__init__()
        self.repr_dims = repr_dims
        self.fusion_type = fusion_type
        if fusion_type == 'concat_linear':
            self.proj = nn.Linear(repr_dims * 2, repr_dims)
        elif fusion_type == 'gated':
            self.gate = nn.Linear(repr_dims * 2, repr_dims)
        else:
            raise ValueError(f'Unsupported fusion_type={fusion_type}')

    def forward(self, pooled_tf: torch.Tensor, h_hyp: Optional[torch.Tensor]) -> torch.Tensor:
        if h_hyp is None:
            return pooled_tf

        fused = torch.cat([pooled_tf, h_hyp], dim=-1)
        if self.fusion_type == 'concat_linear':
            return self.proj(fused)

        gate = torch.sigmoid(self.gate(fused))
        return gate * pooled_tf + (1.0 - gate) * h_hyp


class SequenceGlobalFusion(nn.Module):
    def __init__(self, repr_dims: int, fusion_type: str = 'concat_linear'):
        super().__init__()
        self.repr_dims = repr_dims
        self.fusion_type = fusion_type
        if fusion_type == 'concat_linear':
            self.proj = nn.Linear(repr_dims * 2, repr_dims)
        elif fusion_type == 'gated':
            self.gate = nn.Linear(repr_dims * 2, repr_dims)
        else:
            raise ValueError(f'Unsupported fusion_type={fusion_type}')

    def forward(self, seq_features: torch.Tensor, global_features: Optional[torch.Tensor]) -> torch.Tensor:
        if global_features is None:
            return seq_features

        global_features = global_features.unsqueeze(1).expand(-1, seq_features.size(1), -1)
        fused = torch.cat([seq_features, global_features], dim=-1)
        if self.fusion_type == 'concat_linear':
            return self.proj(fused)

        gate = torch.sigmoid(self.gate(fused))
        return gate * seq_features + (1.0 - gate) * global_features
