#!/usr/bin/env python3
"""Overfit a tiny downstream batch to sanity-check relation_cgeom training.

This is not an evaluation script. It intentionally trains on the same small
set repeatedly. If train accuracy cannot rise far above chance here, the issue
is in the model/data/loss/training path rather than pretraining quality.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import datasets
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.abstract.factory import ModelRegistry  # noqa: E402
from common.path import get_conf_file_path  # noqa: E402
from common.utils import setup_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny relation_cgeom downstream batch.")
    parser.add_argument("--conf-file", default="baseline/relation_cgeom/relation_cgeom_bcic2a_spectral_lite.yaml")
    parser.add_argument("--dataset", default="bcic_2a")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--balanced", action="store_true", help="Build a class-balanced tiny set if possible.")
    parser.add_argument("--scan-batches", type=int, default=100, help="Max train batches scanned for balanced sampling.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--disable-lorentz", action="store_true")
    parser.add_argument("--disable-lorentz-classifier", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use bf16 autocast. Default is FP32 for debugging.")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    setup_yaml()
    file_cfg = OmegaConf.load(get_conf_file_path(args.conf_file))
    model_type = file_cfg.get("model_type", "relation_cgeom")
    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())

    dataset_config = args.dataset_config or f"finetune_loso_fold{args.fold}"
    cli_cfg = OmegaConf.create(
        {
            "seed": args.seed,
            "model_type": model_type,
            "data": {
                "batch_size": args.batch_size,
                "num_workers": 0,
                "datasets": {args.dataset: dataset_config},
            },
            "model": {
                "pretrained_path": None,
            },
            "training": {
                "freeze_encoder": False,
                "use_amp": bool(args.amp),
                "weight_decay": args.weight_decay,
            },
            "logging": {
                "use_cloud": False,
                "save_best_only": False,
                "save_last_checkpoint": False,
            },
        }
    )

    if args.disable_lorentz:
        cli_cfg.model.use_heegnet_lorentz = False
    if args.disable_lorentz_classifier:
        cli_cfg.model.use_heegnet_lorentz_classifier = False

    merged = OmegaConf.merge(code_cfg, file_cfg, cli_cfg)
    cfg_dict = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    return config_class.model_validate(cfg_dict)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def concat_batches(batches: list[dict[str, Any]], max_samples: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batches[0].keys()
    for key in keys:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.cat(values, dim=0)[:max_samples]
        elif isinstance(values[0], (list, tuple)):
            merged = []
            for value in values:
                merged.extend(list(value))
            out[key] = merged[:max_samples]
        else:
            out[key] = values[0]
    return out


def subset_batch(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    index_list = indices.tolist()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[indices]
        elif isinstance(value, (list, tuple)):
            out[key] = [value[i] for i in index_list]
        else:
            out[key] = value
    return out


def make_balanced_subset(batch: dict[str, Any], n_class: int, max_samples: int) -> dict[str, Any]:
    labels = batch["label"].long()
    per_class = max(1, max_samples // n_class)
    selected = []
    for cls in range(n_class):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        if cls_idx.numel() == 0:
            raise RuntimeError(f"Cannot build balanced set: class {cls} has no samples in scanned data.")
        selected.append(cls_idx[:per_class])
    indices = torch.cat(selected, dim=0)
    return subset_batch(batch, indices)


def collect_tiny_batch(trainer, ds_name: str, ds_config: str, num_samples: int):
    loader, _ = trainer.create_single_dataloader(ds_name, ds_config, datasets.Split.TRAIN)
    batches = []
    total = 0
    for batch in loader:
        batches.append(batch)
        total += int(batch["label"].shape[0])
        if total >= num_samples:
            break
    if not batches:
        raise RuntimeError("No training batch loaded.")
    return concat_batches(batches, num_samples)


def collect_balanced_tiny_batch(
    trainer,
    ds_name: str,
    ds_config: str,
    num_samples: int,
    n_class: int,
    scan_batches: int,
):
    loader, _ = trainer.create_single_dataloader(ds_name, ds_config, datasets.Split.TRAIN)
    batches = []
    for idx, batch in enumerate(loader):
        batches.append(batch)
        if idx + 1 >= scan_batches:
            break
    if not batches:
        raise RuntimeError("No training batch loaded.")
    scanned = concat_batches(batches, max_samples=sum(int(batch["label"].shape[0]) for batch in batches))
    return make_balanced_subset(scanned, n_class=n_class, max_samples=num_samples)


def trainable_param_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_feature_stats(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, float]:
    """Measure whether the encoder produces sample-varying global features."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = batch["data"]
        setup = model._build_setup_context(batch, x)
        out = model.encoder(
            x.transpose(1, 2),
            mask=model.downstream_mask,
            setup=setup,
            return_aux=True,
        )
        main_global = out["main_global"].detach().float()
        relation_global = out["relation_global"].detach().float()
        logits = model(batch).detach().float()
    if was_training:
        model.train()

    return {
        "std_main": main_global.std(dim=0).mean().item(),
        "std_relation": relation_global.std(dim=0).mean().item(),
        "std_logits": logits.std(dim=0).mean().item(),
        "mean_abs_main": main_global.abs().mean().item(),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = build_cfg(args)
    ds_name = args.dataset
    ds_config = cfg.data.datasets[ds_name]

    trainer = ModelRegistry.create_trainer(cfg)
    trainer.setup_device(args.device if torch.cuda.is_available() else "cpu")
    trainer.collect_dataset_info(mixed=False, ds_name=ds_name)
    model = trainer.setup_model()
    model.train()

    n_class = trainer.ds_info[ds_name]["n_class"]
    if args.balanced:
        batch = collect_balanced_tiny_batch(
            trainer,
            ds_name,
            ds_config,
            num_samples=args.num_samples,
            n_class=n_class,
            scan_batches=args.scan_batches,
        )
    else:
        batch = collect_tiny_batch(trainer, ds_name, ds_config, args.num_samples)
    batch = move_batch(batch, trainer.device)
    labels = batch["label"].long()
    label_counts = torch.bincount(labels.detach().cpu(), minlength=n_class)

    total, trainable = trainable_param_count(model)
    print(f"dataset={ds_name}/{ds_config}")
    print(f"device={trainer.device}")
    print(f"samples={labels.numel()} balanced={args.balanced} label_bincount={label_counts.tolist()}")
    print(f"params total={total:,} trainable={trainable:,}")
    print(
        "lorentz_relation="
        f"{getattr(cfg.model, 'use_heegnet_lorentz', None)} "
        "lorentz_classifier="
        f"{getattr(cfg.model, 'use_heegnet_lorentz_classifier', None)}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=bool(args.amp))

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
            logits = model(batch)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == args.steps:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean()
                pred_counts = torch.bincount(preds.detach().cpu(), minlength=n_class)
                logit_std = logits.detach().float().std().item()
            feature_stats = compute_feature_stats(model, batch)
            print(
                f"step={step:04d} loss={loss.item():.6f} acc={acc.item():.4f} "
                f"logits_std={logit_std:.6f} pred_bincount={pred_counts.tolist()} "
                f"std_main={feature_stats['std_main']:.6f} "
                f"std_rel={feature_stats['std_relation']:.6f} "
                f"std_logits_eval={feature_stats['std_logits']:.6f}"
            )

    print("Expected sanity result: acc should approach 0.90-1.00 on this tiny fixed set.")


if __name__ == "__main__":
    main()
