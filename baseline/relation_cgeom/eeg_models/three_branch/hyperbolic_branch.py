from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _project(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    max_norm = (1.0 - eps) / torch.sqrt(c)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    projected = x / norm * max_norm
    return torch.where(norm > max_norm, projected, x)


def expmap0(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sqrt_c = torch.sqrt(c)
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
    gamma = torch.tanh(sqrt_c * u_norm) / (sqrt_c * u_norm)
    return _project(gamma * u, c)


def logmap0(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sqrt_c = torch.sqrt(c)
    y = _project(y, c)
    y_norm = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.atanh((sqrt_c * y_norm).clamp(max=1 - 1e-5)) / (sqrt_c * y_norm)
    return scale * y


class TangentLinear(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.linear = nn.Linear(dims, dims)

    def forward(self, x_hyp: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_tan = logmap0(x_hyp, c)
        x_tan = F.gelu(self.linear(x_tan))
        return expmap0(x_tan, c)


class HyperbolicBranch(nn.Module):
    def __init__(
        self,
        repr_dims: int,
        depth: int = 1,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(repr_dims, repr_dims)
        self.layers = nn.ModuleList([TangentLinear(repr_dims) for _ in range(depth)])
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
        pooled = h_tf.mean(dim=1)
        pooled = self.input_proj(pooled)
        c = self.curvature()
        x_hyp = expmap0(pooled, c)
        for layer in self.layers:
            x_hyp = layer(x_hyp, c)
        return logmap0(x_hyp, c)
