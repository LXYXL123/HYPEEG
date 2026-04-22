#!/usr/bin/env python3
"""Inspect relation_cgeom prediction collapse on a processed dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
import torch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score

from baseline.relation_cgeom.relation_cgeom_config import RelationCGeomConfig
from baseline.relation_cgeom.relation_cgeom_trainer import RelationCGeomTrainer
from common.path import get_conf_file_path
from common.utils import setup_yaml


SPLIT_MAP = {
    "train": datasets.Split.TRAIN,
    "validation": datasets.Split.VALIDATION,
    "eval": datasets.Split.VALIDATION,
    "test": datasets.Split.TEST,
}


def load_cfg(conf_file: str, cli_overrides: list[str]) -> RelationCGeomConfig:
    setup_yaml()
    code_cfg = OmegaConf.create(RelationCGeomConfig().model_dump())
    file_cfg = OmegaConf.load(get_conf_file_path(conf_file))
    cli_cfg = OmegaConf.from_dotlist(cli_overrides)
    merged = OmegaConf.merge(code_cfg, file_cfg, cli_cfg)
    cfg_dict = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    return RelationCGeomConfig.model_validate(cfg_dict)


def load_full_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()

    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module.") and not any(k.startswith("module.") for k in target):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value

    compatible = {}
    skipped = []
    for key, value in cleaned.items():
        if key in target and target[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"loaded downstream checkpoint: {ckpt_path}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)} skipped_shape={len(skipped)}")
    if missing[:10]:
        print("missing_keys_sample=", missing[:10])
    if unexpected[:10]:
        print("unexpected_keys_sample=", unexpected[:10])
    if skipped[:10]:
        print("skipped_shape_sample=", skipped[:10])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file", default="baseline/relation_cgeom/relation_cgeom_bcic2a_spectral_lite.yaml")
    parser.add_argument("--ckpt", default=None, help="Optional downstream full-model checkpoint to inspect.")
    parser.add_argument("--split", choices=sorted(SPLIT_MAP), default="test")
    parser.add_argument("--dataset", default="bcic_2a")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_cfg(args.conf_file, args.overrides)
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = 0

    device = torch.device(args.device)
    trainer = RelationCGeomTrainer(cfg)
    trainer.setup_device(str(device))
    trainer.collect_dataset_info(mixed=False, ds_name=args.dataset)
    model = trainer.setup_model()
    if args.ckpt:
        load_full_model_checkpoint(model, args.ckpt, device)
    model.eval()

    loader, _ = trainer.dataloader_factory.loading_dataset(
        datasets_config={args.dataset: cfg.data.datasets[args.dataset]},
        split=SPLIT_MAP[args.split],
        fs=cfg.fs,
        num_replicas=1,
        rank=0,
    )

    n_class = trainer.ds_info[args.dataset]["n_class"]
    labels_all = []
    preds_all = []
    logits_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=cfg.training.use_amp and device.type == "cuda", dtype=torch.bfloat16):
                logits = model(batch).float()
            labels = batch["label"].long()
            preds = torch.argmax(logits, dim=1)
            labels_all.append(labels.cpu())
            preds_all.append(preds.cpu())
            logits_all.append(logits.cpu())

    labels = torch.cat(labels_all)
    preds = torch.cat(preds_all)
    logits = torch.cat(logits_all)

    cm = torch.zeros((n_class, n_class), dtype=torch.int64)
    for y, p in zip(labels, preds):
        cm[int(y), int(p)] += 1

    acc = (labels == preds).float().mean().item()
    bacc = balanced_accuracy_score(labels.numpy(), preds.numpy())

    print(f"split={args.split} n={len(labels)}")
    print("label_bincount=", torch.bincount(labels, minlength=n_class).tolist())
    print("pred_bincount =", torch.bincount(preds, minlength=n_class).tolist())
    print("confusion_matrix rows=true cols=pred")
    print(cm)
    print(f"acc={acc:.6f} balanced_acc={bacc:.6f}")
    print(f"logits_mean={logits.mean().item():.6f} logits_std={logits.std().item():.6f}")
    print("logits_class_mean=", logits.mean(dim=0).tolist())
    print("logits_class_std =", logits.std(dim=0).tolist())


if __name__ == "__main__":
    main()
