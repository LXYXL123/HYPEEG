import torch
from torch import nn

from baseline.relation_cgeom.eeg_models.three_branch.complex_branch_cgeom import FilterBankComplexGeomBranch
from baseline.relation_cgeom.eeg_models.three_branch.config import ThreeBranchConfig
from baseline.relation_cgeom.eeg_models.three_branch.hyperbolic_relation_branch import HyperbolicRelationBranch
from baseline.relation_cgeom.eeg_models.three_branch.raw_branch import RawTemporalBranch
from baseline.relation_cgeom.encoder_three_branch import generate_binomial_mask, generate_continuous_mask
from baseline.relation_cgeom.losses_complex_geom import complex_avg_pool1d


class TSEncoderThreeBranchHypRelationCGeom(nn.Module):
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
        use_raw_mask_loss=False,
        mask_mode='binomial',
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.use_raw_mask_loss = use_raw_mask_loss

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

        if not config.use_complex_branch:
            raise ValueError('The cgeom-aligned hyp_relation encoder requires the complex branch.')

        self.raw_branch = (
            RawTemporalBranch(
                input_dims=input_dims,
                repr_dims=output_dims,
                hidden_dims=hidden_dims,
                depth=depth,
            )
            if config.use_raw_branch else None
        )
        self.complex_branch = FilterBankComplexGeomBranch(
            input_dims=input_dims,
            repr_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            bands=config.bands,
            sampling_rate=config.sampling_rate,
            share_band_encoder=config.share_band_encoder,
            fusion_type=config.band_fusion_type,
        )
        self.hyperbolic_branch = (
            HyperbolicRelationBranch(
                repr_dims=output_dims,
                depth=config.hyperbolic_depth,
                curvature=config.hyperbolic_curvature,
                learnable_curvature=config.learnable_curvature,
            )
            if config.use_hyperbolic_branch else None
        )

        self.band_relation_projectors = nn.ModuleList([
            nn.Linear(output_dims * 2, output_dims)
            for _ in config.bands
        ])
        # Auxiliary pooled projections kept for diagnostics / optional alignment losses.
        self.raw_align_proj = nn.Linear(output_dims, output_dims) if config.use_raw_branch else None
        self.complex_align_proj = nn.Linear(output_dims * 2, output_dims)

        # Lightweight residual fusion:
        # - raw branch directly contributes to the final representation via
        #   [B, T, D] -> LayerNorm -> Linear -> [B, T, 2D]
        # - relation branch stays as a global auxiliary modulator via
        #   [B, D] -> LayerNorm -> Linear -> [B, 1, 2D]
        self.raw_fusion_norm = nn.LayerNorm(output_dims) if config.use_raw_branch else None
        self.proj_raw = nn.Linear(output_dims, output_dims * 2) if config.use_raw_branch else None

        self.rel_fusion_norm = nn.LayerNorm(output_dims) if config.use_hyperbolic_branch else None
        self.proj_rel = nn.Linear(output_dims, output_dims * 2) if config.use_hyperbolic_branch else None

        # Keep complex features as the dominant path at initialization. The model
        # can then gradually learn how much direct raw and relation residual it needs.
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
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

    def forward(self, x, mask=None, return_aux=False):
        nan_mask = ~x.isnan().any(axis=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x_full = x.clone()

        mask = self._resolve_mask(x, mask)
        mask &= nan_mask
        x[~mask] = 0

        h_raw = self.raw_branch(x) if self.raw_branch is not None else None
        raw_full_repr = None
        if self.raw_branch is not None and self.use_raw_mask_loss and return_aux:
            raw_branch_training = self.raw_branch.training
            self.raw_branch.eval()
            with torch.no_grad():
                raw_full_repr = self.raw_branch(x_full, mask='all_true').mean(dim=1)
            self.raw_branch.train(raw_branch_training)
        h_complex, band_complex = self.complex_branch(x, return_band_features=True)

        relation_global = None
        if self.hyperbolic_branch is not None:
            relation_bands = [
                projector(band_feature)
                for projector, band_feature in zip(self.band_relation_projectors, band_complex)
            ]
            relation_global = self.hyperbolic_branch(h_raw, relation_bands)

        raw_residual = None
        if h_raw is not None:
            # h_raw: [B, T, D] -> raw_residual: [B, T, 2D]
            raw_residual = self.proj_raw(self.raw_fusion_norm(h_raw))

        rel_residual = None
        if relation_global is not None:
            # relation_global: [B, D] -> rel_residual: [B, 1, 2D]
            rel_residual = self.proj_rel(self.rel_fusion_norm(relation_global)).unsqueeze(1)

        # Final fused representation:
        # - h_complex:      [B, T, 2D]  (main spectral-complex path)
        # - raw_residual:   [B, T, 2D]  (direct temporal residual path)
        # - rel_residual:   [B, 1, 2D]  (global relation residual, broadcast on T)
        # - main_repr:      [B, T, 2D]
        #
        # This keeps the hyperbolic relation branch, while making the raw branch
        # directly visible to the classifier instead of only affecting the output
        # through relation pooling.
        main_repr = h_complex
        if raw_residual is not None:
            main_repr = main_repr + self.alpha * raw_residual
        if rel_residual is not None:
            main_repr = main_repr + self.beta * rel_residual

        raw_align = None
        if h_raw is not None:
            raw_align = self.raw_align_proj(h_raw.mean(dim=1))
        complex_align = self.complex_align_proj(
            complex_avg_pool1d(h_complex, kernel_size=h_complex.size(1)).squeeze(1)
        )

        main_repr = self.repr_dropout(main_repr)
        main_repr[~nan_mask] = 0

        if return_aux:
            aux = {
                'main_repr': main_repr,
                'raw_align': raw_align,
                'complex_align': complex_align,
                'relation_global': relation_global,
            }
            if h_raw is not None and raw_full_repr is not None:
                aux['raw_masked_repr'] = h_raw.mean(dim=1)
                aux['raw_full_repr'] = raw_full_repr
            return aux
        return main_repr
