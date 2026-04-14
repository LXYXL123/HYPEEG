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
        use_raw_residual=True,
        use_complex_residual=False,
        use_hyper_relation=True,
        use_gated_raw_residual=True,
        use_gated_complex_residual=False,
        local_fusion_type='add',
        relation_condition_type='residual',
        relation_gate_scale=0.1,
        relation_film_scale=0.1,
        mask_mode='binomial',
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.use_raw_mask_loss = use_raw_mask_loss
        self.use_raw_residual = use_raw_residual and use_raw_branch
        self.use_complex_residual = use_complex_residual and use_complex_branch
        self.use_hyper_relation = use_hyper_relation and use_hyperbolic_branch
        self.use_gated_raw_residual = use_gated_raw_residual
        self.use_gated_complex_residual = use_gated_complex_residual
        self.local_fusion_type = local_fusion_type
        self.relation_condition_type = relation_condition_type
        self.relation_gate_scale = float(relation_gate_scale)
        self.relation_film_scale = float(relation_film_scale)

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
            if self.use_hyper_relation else None
        )

        # Auxiliary pooled projections kept for diagnostics / optional alignment losses.
        self.raw_align_proj = nn.Linear(output_dims, output_dims) if config.use_raw_branch else None
        self.complex_align_proj = nn.Linear(output_dims * 2, output_dims)

        # Lightweight local + relation fusion:
        # - raw branch contributes token-level temporal details:
        #   [B, T, D] -> LayerNorm -> Linear/Gate -> [B, T, 2D]
        # - optional complex branch contributes token-level spectral/phase details:
        #   [B, T, 2D] -> LayerNorm -> [B, T, 2D]
        # - default v1 relation branch sees raw_global + complex_global and
        #   contributes a weak global residual: [B, D] -> [B, 1, 2D].
        self.raw_fusion_norm = nn.LayerNorm(output_dims) if self.use_raw_residual else None
        self.proj_raw = nn.Linear(output_dims, output_dims * 2) if self.use_raw_residual else None
        self.raw_gate = nn.Linear(output_dims, output_dims * 2) if self.use_raw_residual and self.use_gated_raw_residual else None

        self.complex_global_norm = (
            nn.LayerNorm(output_dims * 2)
            if self.use_hyper_relation and self.relation_condition_type == 'residual'
            else None
        )
        self.complex_global_proj = (
            nn.Linear(output_dims * 2, output_dims)
            if self.use_hyper_relation and self.relation_condition_type == 'residual'
            else None
        )
        self.rel_fusion_norm = (
            nn.LayerNorm(output_dims)
            if self.use_hyper_relation and self.relation_condition_type == 'residual'
            else None
        )
        self.proj_rel = (
            nn.Linear(output_dims, output_dims * 2)
            if self.use_hyper_relation and self.relation_condition_type == 'residual'
            else None
        )

        self.complex_fusion_norm = nn.LayerNorm(output_dims * 2) if self.use_complex_residual else None
        self.proj_complex = nn.Linear(output_dims * 2, output_dims * 2) if self.use_complex_residual and self.use_gated_complex_residual else None
        self.complex_gate = nn.Linear(output_dims * 2, output_dims * 2) if self.use_complex_residual and self.use_gated_complex_residual else None
        self.local_fusion_norm = nn.LayerNorm(output_dims * 4) if self.local_fusion_type == 'concat_linear' else None
        self.local_fusion_proj = nn.Linear(output_dims * 4, output_dims * 2) if self.local_fusion_type == 'concat_linear' else None

        self.band_global_norm = nn.LayerNorm(output_dims * 2) if self.use_hyper_relation else None
        self.band_global_proj = nn.Linear(output_dims * 2, output_dims) if self.use_hyper_relation else None
        self.relation_condition_norm = nn.LayerNorm(output_dims) if self.use_hyper_relation else None
        self.relation_gate = (
            nn.Linear(output_dims, output_dims * 2)
            if self.use_hyper_relation and self.relation_condition_type == 'gate'
            else None
        )
        self.relation_film = (
            nn.Linear(output_dims, output_dims * 4)
            if self.use_hyper_relation and self.relation_condition_type == 'film'
            else None
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
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

    @staticmethod
    def masked_mean_pool(x, mask):
        """Masked mean pooling over time.

        x: [B, T, D]
        mask: [B, T]
        pooled: [B, D]
        """
        if mask is None:
            return x.mean(dim=1)

        weights = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = weights.sum(dim=1).clamp_min(1.0)  # [B, 1]
        return (x * weights).sum(dim=1) / denom

    def build_raw_residual(self, h_raw):
        """Build token-level raw residual.

        h_raw: [B, T, D]
        raw_residual: [B, T, 2D]
        """
        if not self.use_raw_residual or h_raw is None:
            return None

        raw_norm = self.raw_fusion_norm(h_raw)  # [B, T, D]
        raw_proj = self.proj_raw(raw_norm)  # [B, T, 2D]
        if self.use_gated_raw_residual:
            raw_gate = torch.sigmoid(self.raw_gate(raw_norm))  # [B, T, 2D]
            if self.relation_condition_type == 'residual':
                return raw_proj * raw_gate
            return raw_proj * (1.0 + raw_gate)
        return raw_proj

    def build_complex_residual(self, h_complex):
        """Build token-level complex residual.

        h_complex: [B, T, 2D]
        complex_residual: [B, T, 2D]
        """
        if not self.use_complex_residual or h_complex is None:
            return None

        complex_norm = self.complex_fusion_norm(h_complex)  # [B, T, 2D]
        if self.use_gated_complex_residual:
            complex_proj = self.proj_complex(complex_norm)  # [B, T, 2D]
            complex_gate = torch.sigmoid(self.complex_gate(complex_norm))  # [B, T, 2D]
            return complex_proj * (1.0 + complex_gate)
        return complex_norm  # [B, T, 2D]

    def build_complex_global(self, h_complex, mask):
        """Build fused-complex global token for v1 residual relation.

        h_complex: [B, T, 2D]
        complex_global: [B, D]
        """
        if not self.use_hyper_relation or self.relation_condition_type != 'residual':
            return None

        complex_norm = self.complex_global_norm(h_complex)  # [B, T, 2D]
        complex_token = self.complex_global_proj(complex_norm)  # [B, T, D]
        return self.masked_mean_pool(complex_token, mask)  # [B, D]

    def build_band_globals(self, band_features, mask):
        """Build per-band global tokens for hyperbolic relation.

        band_features: list of [B, T, 2D]
        band_globals: list of [B, D]
        """
        if not self.use_hyper_relation:
            return []

        band_globals = []
        for band_feature in band_features:
            band_norm = self.band_global_norm(band_feature)  # [B, T, 2D]
            band_token = self.band_global_proj(band_norm)  # [B, T, D]
            band_globals.append(self.masked_mean_pool(band_token, mask))  # [B, D]
        return band_globals

    def build_relation_condition(self, raw_global, band_globals):
        """Build relation condition from raw_global and per-band globals.

        raw_global: [B, D]
        band_globals: list of [B, D]
        relation_global: [B, D]
        relation_gate: [B, 1, 2D] when relation_condition_type == 'gate'
        gamma / film_beta: [B, 1, 2D] when relation_condition_type == 'film'
        """
        if not self.use_hyper_relation or self.hyperbolic_branch is None:
            return None, None, None

        relation_tokens = []
        if raw_global is not None:
            relation_tokens.append(raw_global)
        relation_tokens.extend(band_globals)

        relation_global = self.hyperbolic_branch.forward_global(relation_tokens)  # [B, D]
        relation_norm = self.relation_condition_norm(relation_global)  # [B, D]

        if self.relation_condition_type == 'film':
            film = self.relation_film(relation_norm)  # [B, 4D]
            gamma, film_beta = film.chunk(2, dim=-1)  # each [B, 2D]
            gamma = self.relation_film_scale * torch.tanh(gamma).unsqueeze(1)  # [B, 1, 2D]
            film_beta = self.relation_film_scale * film_beta.unsqueeze(1)  # [B, 1, 2D]
            return relation_global, gamma, film_beta

        relation_gate = torch.sigmoid(self.relation_gate(relation_norm)).unsqueeze(1)  # [B, 1, 2D]
        return relation_global, relation_gate, None

    def build_relation_residual(self, raw_global, complex_global):
        """Build weak global residual from raw_global and fused complex_global.

        raw_global: [B, D]
        complex_global: [B, D]
        relation_global: [B, D]
        rel_residual: [B, 1, 2D]
        """
        if not self.use_hyper_relation or self.hyperbolic_branch is None:
            return None, None

        relation_tokens = []
        if raw_global is not None:
            relation_tokens.append(raw_global)
        if complex_global is not None:
            relation_tokens.append(complex_global)

        relation_global = self.hyperbolic_branch.forward_global(relation_tokens)  # [B, D]
        rel_norm = self.rel_fusion_norm(relation_global)  # [B, D]
        rel_residual = self.proj_rel(rel_norm).unsqueeze(1)  # [B, 1, 2D]
        return relation_global, rel_residual

    def build_local_repr(self, raw_residual, complex_residual):
        """Fuse local raw and complex residuals.

        raw_residual: [B, T, 2D] or None
        complex_residual: [B, T, 2D] or None
        local_repr: [B, T, 2D]
        """
        if raw_residual is not None and complex_residual is not None:
            if self.local_fusion_type == 'concat_linear':
                local_cat = torch.cat([raw_residual, complex_residual], dim=-1)  # [B, T, 4D]
                return self.local_fusion_proj(self.local_fusion_norm(local_cat))  # [B, T, 2D]
            return raw_residual + complex_residual
        if raw_residual is not None:
            return raw_residual
        return complex_residual

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
        return_band_features = self.relation_condition_type != 'residual'
        complex_out = self.complex_branch(x, return_band_features=return_band_features)
        if return_band_features:
            h_complex, band_features = complex_out  # [B, T, 2D], list[[B, T, 2D]]
        else:
            h_complex, band_features = complex_out, []

        # raw_global and raw_residual serve different roles:
        # - raw_global: [B, D] for global relation modeling
        # - raw_residual: [B, T, 2D] for token-level temporal information
        raw_global = self.masked_mean_pool(h_raw, mask) if h_raw is not None else None  # [B, D]
        raw_residual = self.build_raw_residual(h_raw)  # [B, T, 2D] or None

        if self.relation_condition_type == 'residual':
            # v1 default: complex branch affects the classifier only through
            # fused complex_global -> HyperbolicRelationBranch. It is not added
            # directly to the token-level classifier representation.
            complex_global = self.build_complex_global(h_complex, mask)  # [B, D]
            relation_global, rel_residual = self.build_relation_residual(raw_global, complex_global)
            local_repr = raw_residual
            main_repr = h_complex.new_zeros(h_complex.shape)
            if raw_residual is not None:
                main_repr = main_repr + self.alpha * raw_residual
            if self.use_hyper_relation and rel_residual is not None:
                main_repr = main_repr + self.beta * rel_residual
            band_globals = []
            relation_cond = rel_residual
            film_beta = None
        else:
            # v2/v3 experimental paths.
            complex_residual = self.build_complex_residual(h_complex)  # [B, T, 2D]
            band_globals = self.build_band_globals(band_features, mask)  # list[[B, D]]
            relation_global, relation_cond, film_beta = self.build_relation_condition(raw_global, band_globals)
            local_repr = self.build_local_repr(raw_residual, complex_residual)

            if self.use_hyper_relation and relation_cond is not None and self.relation_condition_type == 'film':
                main_repr = local_repr * (1.0 + relation_cond) + film_beta
            elif self.use_hyper_relation and relation_cond is not None:
                main_repr = local_repr * (1.0 + self.relation_gate_scale * relation_cond)
            else:
                main_repr = local_repr

        raw_align = None
        if h_raw is not None:
            raw_align = self.raw_align_proj(self.masked_mean_pool(h_raw, mask))
        complex_align = self.complex_align_proj(
            complex_avg_pool1d(h_complex, kernel_size=h_complex.size(1)).squeeze(1)
        )

        main_repr = self.repr_dropout(main_repr)
        main_repr[~nan_mask] = 0

        if return_aux:
            aux = {
                'main_repr': main_repr,
                'local_repr': local_repr,
                'raw_align': raw_align,
                'complex_align': complex_align,
                'raw_global': raw_global,
                'complex_global': self.build_complex_global(h_complex, mask)
                if self.relation_condition_type == 'residual' else None,
                'band_globals': band_globals,
                'relation_global': relation_global,
                'relation_condition': relation_cond,
                'film_beta': film_beta,
            }
            if h_raw is not None and raw_full_repr is not None:
                aux['raw_masked_repr'] = h_raw.mean(dim=1)
                aux['raw_full_repr'] = raw_full_repr
            return aux
        return main_repr
