from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from baseline.relation_cgeom.eeg_models.three_branch.complex_branch import FilterBankComplexBranch
from baseline.relation_cgeom.eeg_models.three_branch.config import ThreeBranchConfig
from baseline.relation_cgeom.eeg_models.three_branch.fusion import HeadFusion, TemporalFrequencyFusion
from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_branch import HyperbolicBranch
from baseline.relation_cgeom.eeg_models.three_branch.raw_branch import RawTemporalBranch


class EEGThreeBranchClassifier(nn.Module):
    def __init__(self, config: ThreeBranchConfig):
        super().__init__()
        if not config.use_raw_branch and not config.use_complex_branch:
            raise ValueError('At least one of raw or complex branch must be enabled.')

        self.config = config
        self.use_raw_branch = config.use_raw_branch
        self.use_complex_branch = config.use_complex_branch
        self.use_hyperbolic_branch = config.use_hyperbolic_branch

        self.raw_branch = (
            RawTemporalBranch(
                input_dims=config.input_channels,
                repr_dims=config.repr_dims,
                hidden_dims=config.hidden_dims,
                depth=config.depth,
            )
            if config.use_raw_branch else None
        )
        self.complex_branch = (
            FilterBankComplexBranch(
                input_dims=config.input_channels,
                repr_dims=config.repr_dims,
                hidden_dims=config.hidden_dims,
                depth=config.depth,
                bands=config.bands,
                sampling_rate=config.sampling_rate,
                share_band_encoder=config.share_band_encoder,
                fusion_type=config.band_fusion_type,
            )
            if config.use_complex_branch else None
        )
        self.tf_fusion = (
            TemporalFrequencyFusion(config.repr_dims, fusion_type=config.tf_fusion_type)
            if config.use_raw_branch and config.use_complex_branch else None
        )
        self.hyperbolic_branch = (
            HyperbolicBranch(
                repr_dims=config.repr_dims,
                depth=config.hyperbolic_depth,
                curvature=config.hyperbolic_curvature,
                learnable_curvature=config.learnable_curvature,
            )
            if config.use_hyperbolic_branch else None
        )
        self.head_fusion = (
            HeadFusion(config.repr_dims, fusion_type=config.head_fusion_type)
            if config.use_hyperbolic_branch else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.repr_dims, config.num_classes)

    def freeze_feature_extractors(self) -> None:
        for module in (self.raw_branch, self.complex_branch):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        h_raw = self.raw_branch(x) if self.raw_branch is not None else None
        h_complex = self.complex_branch(x) if self.complex_branch is not None else None
        if self.tf_fusion is not None:
            h_tf = self.tf_fusion(h_raw, h_complex)
        else:
            h_tf = h_raw if h_raw is not None else h_complex
        pooled_tf = h_tf.mean(dim=1)
        h_hyp = self.hyperbolic_branch(h_tf) if self.hyperbolic_branch is not None else None
        if self.head_fusion is not None:
            h_out = self.head_fusion(pooled_tf, h_hyp)
        else:
            h_out = pooled_tf
        return {
            'h_raw': h_raw,
            'h_complex': h_complex,
            'h_tf': h_tf,
            'pooled_tf': pooled_tf,
            'h_hyp': h_hyp,
            'h_out': h_out,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        h_out = self.dropout(features['h_out'])
        return self.classifier(h_out)
