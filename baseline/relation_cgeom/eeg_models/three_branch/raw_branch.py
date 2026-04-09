from __future__ import annotations

from torch import nn

from baseline.relation_cgeom.encoder import TSEncoder


class RawTemporalBranch(nn.Module):
    def __init__(self, input_dims: int, repr_dims: int, hidden_dims: int, depth: int):
        super().__init__()
        self.encoder = TSEncoder(
            input_dims=input_dims,
            output_dims=repr_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        )

    def forward(self, x, mask='all_true'):
        return self.encoder(x, mask=mask)
