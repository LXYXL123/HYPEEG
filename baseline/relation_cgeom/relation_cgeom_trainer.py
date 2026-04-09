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
from baseline.relation_cgeom.relation_cgeom_adapter import RelationCGeomDataLoaderFactory, RelationCGeomDatasetAdapter
from baseline.relation_cgeom.relation_cgeom_config import RelationCGeomConfig, RelationCGeomModelArgs
from data.processor.wrapper import get_dataset_montage


logger = logging.getLogger("baseline")


class RelationCGeomUnifiedModel(nn.Module):
    """Wrapper that adapts relation-CGeom encoder outputs to the benchmark classifier API."""

    def __init__(
        self,
        encoder: TSEncoderThreeBranchHypRelationCGeom,
        classifier: MultiHeadClassifier,
        downstream_mask: Optional[str] = "all_true",
        grad_cam: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.downstream_mask = downstream_mask
        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        x = batch["data"]  # [B, C, T]
        montage = batch["montage"][0]

        # relation-CGeom encoder consumes [B, T, C].
        x = x.transpose(1, 2)
        features = self.encoder(x, mask=self.downstream_mask)  # [B, T, 2 * output_dims]
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

    def _resolve_input_dims(self) -> int:
        channel_counts = set()
        supported_channels = set(RelationCGeomDatasetAdapter.get_standard_eeg_channels())

        for ds_name, info in self.ds_info.items():
            ds_conf = info["config"]
            montages = get_dataset_montage(ds_name, ds_conf)
            for _, channel_names in montages.items():
                effective_channels = [ch for ch in channel_names if ch in supported_channels]
                channel_counts.add(len(effective_channels))

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

        input_dims = self._resolve_input_dims()
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
            mask_mode=model_cfg.mask_mode,
        )

        embed_dim = model_cfg.output_dims * 2
        head_configs = {ds_name: info["n_class"] for ds_name, info in self.ds_info.items()}
        head_cfg = model_cfg.classifier_head

        ds_shape_out_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, _) in info["shape_info"].items():
                ds_shape_out_info[montage_key] = (n_timepoints, 1, embed_dim)

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
            downstream_mask=model_cfg.downstream_mask,
            grad_cam=model_cfg.grad_cam,
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

        missing, unexpected = self.encoder.load_state_dict(cleaned_state, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        logger.info("Successfully loaded pretrained encoder weights")
        return checkpoint
