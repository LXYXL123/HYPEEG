import numpy as np
import torch
from torch import nn

from baseline.relation_cgeom.eeg_models.three_branch.complex_branch import FilterBankComplexBranch
from baseline.relation_cgeom.eeg_models.three_branch.config import ThreeBranchConfig
from baseline.relation_cgeom.eeg_models.three_branch.fusion import SequenceGlobalFusion, TemporalFrequencyFusion
from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_branch import HyperbolicBranch
from baseline.relation_cgeom.eeg_models.three_branch.raw_branch import RawTemporalBranch


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoderThreeBranch(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        depth=10,
        sampling_rate=256.0,
        bands=((8.0, 13.0), (13.0, 20.0), (20.0, 30.0)),
        use_raw_branch=True,
        use_complex_branch=True,
        use_hyperbolic_branch=True,
        share_band_encoder=False,
        band_fusion_type='concat_linear',
        tf_fusion_type='concat_linear',
        global_fusion_type='concat_linear',
        hyperbolic_depth=1,
        hyperbolic_curvature=1.0,
        learnable_curvature=True,
        mask_mode='binomial',
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        config = ThreeBranchConfig(
            input_channels=input_dims,
            num_classes=0,
            sampling_rate=sampling_rate,
            repr_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            use_raw_branch=use_raw_branch,
            use_complex_branch=use_complex_branch,
            use_hyperbolic_branch=use_hyperbolic_branch,
            bands=bands,
            share_band_encoder=share_band_encoder,
            band_fusion_type=band_fusion_type,
            tf_fusion_type=tf_fusion_type,
            global_fusion_type=global_fusion_type,
            hyperbolic_depth=hyperbolic_depth,
            hyperbolic_curvature=hyperbolic_curvature,
            learnable_curvature=learnable_curvature,
            dropout=0.1,
        )

        if not config.use_raw_branch and not config.use_complex_branch:
            raise ValueError('At least one of raw or complex branch must be enabled.')

        self.raw_branch = (
            RawTemporalBranch(
                input_dims=input_dims,
                repr_dims=output_dims,
                hidden_dims=hidden_dims,
                depth=depth,
            )
            if config.use_raw_branch else None
        )
        self.complex_branch = (
            FilterBankComplexBranch(
                input_dims=input_dims,
                repr_dims=output_dims,
                hidden_dims=hidden_dims,
                depth=depth,
                bands=config.bands,
                sampling_rate=config.sampling_rate,
                share_band_encoder=config.share_band_encoder,
                fusion_type=config.band_fusion_type,
            )
            if config.use_complex_branch else None
        )
        self.tf_fusion = (
            TemporalFrequencyFusion(output_dims, fusion_type=config.tf_fusion_type)
            if config.use_raw_branch and config.use_complex_branch else None
        )
        self.hyperbolic_branch = (
            HyperbolicBranch(
                repr_dims=output_dims,
                depth=config.hyperbolic_depth,
                curvature=config.hyperbolic_curvature,
                learnable_curvature=config.learnable_curvature,
            )
            if config.use_hyperbolic_branch else None
        )
        self.global_fusion = (
            SequenceGlobalFusion(output_dims, fusion_type=config.global_fusion_type)
            if config.use_hyperbolic_branch else None
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def _resolve_mask(self, x, mask):
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        return mask

    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x = x.clone()
        x[~nan_mask] = 0

        mask = self._resolve_mask(x, mask)
        mask &= nan_mask
        x[~mask] = 0

        h_raw = self.raw_branch(x) if self.raw_branch is not None else None
        h_complex = self.complex_branch(x) if self.complex_branch is not None else None
        if self.tf_fusion is not None:
            h_tf = self.tf_fusion(h_raw, h_complex)
        else:
            h_tf = h_raw if h_raw is not None else h_complex
        h_hyp = self.hyperbolic_branch(h_tf) if self.hyperbolic_branch is not None else None
        if self.global_fusion is not None:
            h_out = self.global_fusion(h_tf, h_hyp)
        else:
            h_out = h_tf
        h_out = self.repr_dropout(h_out)
        h_out[~nan_mask] = 0
        return h_out
