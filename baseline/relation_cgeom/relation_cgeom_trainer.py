"""
Relation-CGeom trainer for downstream EEG classification.
"""

import logging
import os
from typing import Optional

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.relation_cgeom.encoder_three_branch_hyp_relation_cgeom import TSEncoderThreeBranchHypRelationCGeom
from baseline.relation_cgeom.heegnet_lorentz import MultiHeadLorentzClassifier
from baseline.relation_cgeom.relation_cgeom_adapter import RelationCGeomDataLoaderFactory, RelationCGeomDatasetAdapter
from baseline.relation_cgeom.relation_cgeom_config import RelationCGeomConfig, RelationCGeomModelArgs
from data.processor.wrapper import get_dataset_montage


logger = logging.getLogger("baseline")


class RelationCGeomUnifiedModel(nn.Module):
    """Wrapper that adapts relation-CGeom encoder outputs to the benchmark classifier API."""

    def __init__(
        self,
        encoder: TSEncoderThreeBranchHypRelationCGeom,
        classifier: nn.Module,
        sampling_rate: float = 256.0,
        downstream_mask: Optional[str] = "all_true",
        grad_cam: bool = False,
        use_lorentz_classifier: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.sampling_rate = float(sampling_rate)
        self.downstream_mask = downstream_mask
        self.grad_cam = grad_cam
        self.use_lorentz_classifier = bool(use_lorentz_classifier)
        self.grad_cam_activation = None

    def _build_setup_context(self, batch, x):
        """Build static setup metadata for the setup-conditioned encoder.

        x: [B, C, T]. The encoder receives channel ids and scalar setup values;
        shallow signal statistics are computed inside SetupEncoder from x itself.
        """
        # chs contains the global ElectrodeSet ids emitted by the processor.
        # This is the channel-name signal used by SetupEncoder; chans_id is only
        # a fallback for older cached datasets.
        channel_ids = batch.get("chs", batch.get("chans_id"))
        return {
            "channel_ids": channel_ids,
            "sampling_rate": self.sampling_rate,
            "window_len": x.size(-1),
        }

    def forward(self, batch):
        x = batch["data"]  # [B, C, T]
        montage = batch["montage"][0]
        setup = self._build_setup_context(batch, x)

        # relation-CGeom encoder consumes [B, T, C].
        x = x.transpose(1, 2)
        if self.use_lorentz_classifier:
            out = self.encoder(x, mask=self.downstream_mask, setup=setup, return_aux=True)
            if self.grad_cam:
                self.grad_cam_activation = out["main_repr"].unsqueeze(2)
            logits = self.classifier(out["main_global"], montage)
            return logits

        features = self.encoder(x, mask=self.downstream_mask, setup=setup)  # [B, T, 2 * output_dims]
        features = features.unsqueeze(2)  # [B, T, 1, D]

        if self.grad_cam:
            self.grad_cam_activation = features

        logits = self.classifier(features, montage)
        return logits


class RelationCGeomTrainer(AbstractTrainer):
    """Trainer for relation-CGeom downstream classification."""

    def __init__(self, cfg: RelationCGeomConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.dataloader_factory = RelationCGeomDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
        )

        self.encoder = None
        self.classifier = None

        if self.cfg.training.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def _resolve_input_dims(self, allow_variable: bool = False) -> int:
        channel_counts = set()
        supported_channels = set(RelationCGeomDatasetAdapter.get_relation_cgeom_channels())

        for ds_name, info in self.ds_info.items():
            ds_conf = info["config"]
            montages = get_dataset_montage(ds_name, ds_conf)
            for _, channel_names in montages.items():
                effective_channels = [ch for ch in channel_names if ch in supported_channels]
                channel_counts.add(len(effective_channels))

        if allow_variable:
            input_dims = max(channel_counts)
            logger.info(
                "Relation-CGeom variable-channel frontend enabled; active effective channel counts: "
                f"{sorted(channel_counts)}"
            )
            return input_dims

        if len(channel_counts) != 1:
            raise ValueError(
                "Relation-CGeom requires a consistent channel count across the active dataset setup. "
                f"Found channel counts: {sorted(channel_counts)}"
            )

        input_dims = next(iter(channel_counts))
        logger.info(f"Relation-CGeom effective input_dims resolved from adapter channels: {input_dims}")
        return input_dims

    def setup_model(self):
        logger.info("Setting up relation-CGeom model architecture...")
        model_cfg: RelationCGeomModelArgs = self.cfg.model

        input_dims = self._resolve_input_dims(
            allow_variable=model_cfg.use_variable_channel_frontend or model_cfg.architecture == "spectral_lite"
        )
        self.encoder = TSEncoderThreeBranchHypRelationCGeom(
            input_dims=input_dims,
            output_dims=model_cfg.output_dims,
            hidden_dims=model_cfg.hidden_dims,
            depth=model_cfg.depth,
            sampling_rate=float(self.cfg.fs),
            bands=model_cfg.bands,
            use_raw_branch=model_cfg.use_raw_branch,
            use_complex_branch=model_cfg.use_complex_branch,
            use_hyperbolic_branch=model_cfg.use_hyperbolic_branch,
            share_band_encoder=model_cfg.share_band_encoder,
            band_fusion_type=model_cfg.band_fusion_type,
            hyperbolic_depth=model_cfg.hyperbolic_depth,
            hyperbolic_curvature=model_cfg.hyperbolic_curvature,
            learnable_curvature=model_cfg.learnable_curvature,
            use_raw_residual=model_cfg.use_raw_residual,
            use_complex_residual=model_cfg.use_complex_residual,
            use_hyper_relation=model_cfg.use_hyper_relation,
            use_gated_raw_residual=model_cfg.use_gated_raw_residual,
            use_gated_complex_residual=model_cfg.use_gated_complex_residual,
            local_fusion_type=model_cfg.local_fusion_type,
            relation_condition_type=model_cfg.relation_condition_type,
            relation_gate_scale=model_cfg.relation_gate_scale,
            relation_film_scale=model_cfg.relation_film_scale,
            use_setup_conditioned=model_cfg.use_setup_conditioned,
            setup_meta_dim=model_cfg.setup_meta_dim,
            setup_ctx_dim=model_cfg.setup_ctx_dim,
            setup_dim=model_cfg.setup_dim,
            setup_channel_vocab_size=model_cfg.setup_channel_vocab_size,
            setup_condition_scale=model_cfg.setup_condition_scale,
            use_variable_channel_frontend=model_cfg.use_variable_channel_frontend,
            per_channel_stem_depth=model_cfg.per_channel_stem_depth,
            channel_attn_heads=model_cfg.channel_attn_heads,
            architecture=model_cfg.architecture,
            spectral_dims=model_cfg.spectral_dims,
            spectral_win_len=model_cfg.spectral_win_len,
            spectral_stride=model_cfg.spectral_stride,
            spectral_freq_low=model_cfg.spectral_freq_low,
            spectral_freq_high=model_cfg.spectral_freq_high,
            spectral_mixer_depth=model_cfg.spectral_mixer_depth,
            use_heegnet_lorentz=model_cfg.use_heegnet_lorentz,
            heegnet_lorentz_dim=model_cfg.heegnet_lorentz_dim,
            mask_mode=model_cfg.mask_mode,
        )

        embed_dim = model_cfg.output_dims * 2
        head_configs = {ds_name: info["n_class"] for ds_name, info in self.ds_info.items()}
        head_cfg = model_cfg.classifier_head

        ds_shape_out_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, _) in info["shape_info"].items():
                ds_shape_out_info[montage_key] = (n_timepoints, 1, embed_dim)

        if model_cfg.use_heegnet_lorentz_classifier:
            self.classifier = MultiHeadLorentzClassifier(
                input_dim=embed_dim,
                lorentz_dim=model_cfg.heegnet_lorentz_dim,
                head_configs=head_configs,
                curvature=model_cfg.hyperbolic_curvature,
                learnable_curvature=model_cfg.learnable_curvature,
                t_sne=model_cfg.t_sne,
            )
        else:
            self.classifier = MultiHeadClassifier(
                embed_dim=embed_dim,
                head_configs=head_configs,
                head_cfg=head_cfg,
                ds_shape_info=ds_shape_out_info,
                t_sne=model_cfg.t_sne,
            )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        self.load_checkpoint(model_cfg.pretrained_path)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        model = RelationCGeomUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            sampling_rate=float(self.cfg.fs),
            downstream_mask=model_cfg.downstream_mask,
            grad_cam=model_cfg.grad_cam,
            use_lorentz_classifier=model_cfg.use_heegnet_lorentz_classifier,
        )

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)

        self.model = model
        return model

    def load_checkpoint(self, checkpoint_path: Optional[str]):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for key, value in state_dict.items():
            if key == "n_averaged":
                continue

            new_key = key
            for prefix in ("module.", "encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            cleaned_state[new_key] = value

        # Cross-dataset pretraining may change the input channel count
        # (e.g. PhysioMI 64ch -> BCIC-2a 22ch). strict=False does not ignore
        # same-key shape mismatches, so filter them explicitly and load the
        # compatible encoder layers.
        target_state = self.encoder.state_dict()
        compatible_state = {}
        skipped_mismatch = []
        for key, value in cleaned_state.items():
            if key in target_state and target_state[key].shape != value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(target_state[key].shape)))
                continue
            compatible_state[key] = value

        missing, unexpected = self.encoder.load_state_dict(compatible_state, strict=False)
        if skipped_mismatch:
            logger.warning("Skipped pretrained keys with incompatible shapes:")
            for key, ckpt_shape, model_shape in skipped_mismatch:
                logger.warning(f"  - {key}: checkpoint={ckpt_shape}, model={model_shape}")
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        logger.info("Successfully loaded pretrained encoder weights")
        return checkpoint
