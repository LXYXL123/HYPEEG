from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_branch import expmap0, logmap0


class HyperbolicRelationLayer(nn.Module):
    def __init__(self, repr_dims: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(repr_dims, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(repr_dims)
        self.ffn = nn.Sequential(
            nn.Linear(repr_dims, repr_dims * 2),
            nn.GELU(),
            nn.Linear(repr_dims * 2, repr_dims),
        )
        self.norm2 = nn.LayerNorm(repr_dims)

    def forward(self, x_hyp: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_tan = logmap0(x_hyp, c)
        attn_out, _ = self.attn(x_tan, x_tan, x_tan, need_weights=False)
        x_tan = self.norm1(x_tan + attn_out)
        ffn_out = self.ffn(x_tan)
        x_tan = self.norm2(x_tan + ffn_out)
        return expmap0(x_tan, c)


class HyperbolicRelationBranch(nn.Module):
    def __init__(
        self,
        repr_dims: int,
        depth: int = 1,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(repr_dims, repr_dims)
        self.layers = nn.ModuleList([HyperbolicRelationLayer(repr_dims, num_heads=num_heads) for _ in range(depth)])
        if learnable_curvature:
            init = torch.log(torch.expm1(torch.tensor(float(curvature))))
            self.raw_curvature = nn.Parameter(init.clone().detach())
            self.fixed_curvature = None
        else:
            self.raw_curvature = None
            self.register_buffer('fixed_curvature', torch.tensor(float(curvature)))

    def curvature(self) -> torch.Tensor:
        if self.raw_curvature is not None:
            return F.softplus(self.raw_curvature).to(dtype=self.input_proj.weight.dtype) + 1e-5
        return self.fixed_curvature.to(dtype=self.input_proj.weight.dtype)

    def forward(self, h_raw: Optional[torch.Tensor], band_features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        if h_raw is not None:
            tokens.append(h_raw.mean(dim=1))
        for band_feature in band_features:
            tokens.append(band_feature.mean(dim=1))
        if not tokens:
            raise ValueError('Relation branch requires at least one token source.')

        token_set = torch.stack(tokens, dim=1)
        token_set = self.input_proj(token_set)
        c = self.curvature()
        x_hyp = expmap0(token_set, c)
        for layer in self.layers:
            x_hyp = layer(x_hyp, c)
        tokens_out = logmap0(x_hyp, c)
        return tokens_out.mean(dim=1)
