from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_branch import expmap0, logmap0


class HyperbolicTokenLayer(nn.Module):
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


class HyperbolicTokenBranch(nn.Module):
    def __init__(
        self,
        repr_dims: int,
        num_tokens: int = 16,
        depth: int = 1,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_proj = nn.Linear(repr_dims, repr_dims)
        self.layers = nn.ModuleList([HyperbolicTokenLayer(repr_dims, num_heads=num_heads) for _ in range(depth)])
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

    def forward(self, h_tf: torch.Tensor) -> torch.Tensor:
        tokens = F.adaptive_avg_pool1d(h_tf.transpose(1, 2), self.num_tokens).transpose(1, 2)
        tokens = self.input_proj(tokens)
        c = self.curvature()
        x_hyp = expmap0(tokens, c)
        for layer in self.layers:
            x_hyp = layer(x_hyp, c)
        tokens_out = logmap0(x_hyp, c)
        return F.interpolate(
            tokens_out.transpose(1, 2),
            size=h_tf.size(1),
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)
