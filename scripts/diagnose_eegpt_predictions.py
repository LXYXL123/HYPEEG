#!/usr/bin/env python3
"""Inspect EEGPT prediction collapse on a processed dataset split."""

import argparse
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
import torch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.eegpt.eegpt_adapter import EegptDataLoaderFactory
from baseline.eegpt.eegpt_config import EegptConfig
from baseline.eegpt.eegpt_trainer import EEGPTUnifiedModel
from baseline.eegpt.model import EEGTransformer
from common.path import get_conf_file_path
from common.utils import setup_yaml
from data.processor.wrapper import get_dataset_n_class, get_dataset_shape_info


SPLIT_MAP = {
    "train": datasets.Split.TRAIN,
    "validation": datasets.Split.VALIDATION,
    "eval": datasets.Split.VALIDATION,
    "test": datasets.Split.TEST,
}


def load_cfg(conf_file: str, cli_overrides: list[str]):
    setup_yaml()
    code_cfg = OmegaConf.create(EegptConfig().model_dump())
    file_cfg = OmegaConf.load(get_conf_file_path(conf_file))
    cli_cfg = OmegaConf.from_dotlist(cli_overrides)
    return OmegaConf.merge(code_cfg, file_cfg, cli_cfg)


def build_ds_info(cfg) -> Dict[str, dict]:
    ds_info = {}
    for ds_name, ds_config in cfg.data.datasets.items():
        ds_info[ds_name] = {
            "n_class": get_dataset_n_class(ds_name, ds_config),
            "shape_info": get_dataset_shape_info(ds_name, ds_config, cfg.fs),
        }
    return ds_info


def build_model(cfg, ds_info: Dict[str, dict], device: torch.device) -> EEGPTUnifiedModel:
    model_conf = cfg.model
    encoder = EEGTransformer(
        img_size=[64, 60 * 256],
        patch_size=model_conf.patch_size,
        patch_stride=model_conf.patch_stride,
        embed_num=model_conf.embed_num,
        embed_dim=model_conf.embed_dim,
        depth=model_conf.depth,
        num_heads=model_conf.num_heads,
        mlp_ratio=model_conf.mlp_ratio,
        drop_rate=model_conf.dropout_rate,
        attn_drop_rate=model_conf.attn_dropout_rate,
        drop_path_rate=model_conf.drop_path_rate,
        init_std=model_conf.init_std,
        qkv_bias=model_conf.qkv_bias,
        norm_layer=nn.LayerNorm,
    )

    head_configs = {ds_name: info["n_class"] for ds_name, info in ds_info.items()}
    ds_shape_info = {}
    for ds_name, info in ds_info.items():
        for montage_key, (n_timepoints, _n_channels) in info["shape_info"].items():
            if model_conf.patch_stride is None:
                seq_len = n_timepoints // model_conf.patch_size
            else:
                seq_len = (n_timepoints - model_conf.patch_size) // model_conf.patch_stride + 1
            ds_shape_info[montage_key] = (seq_len, model_conf.embed_num, model_conf.embed_dim)

    classifier = MultiHeadClassifier(
        embed_dim=model_conf.embed_dim,
        head_configs=head_configs,
        head_cfg=model_conf.classifier_head,
        ds_shape_info=ds_shape_info,
        t_sne=model_conf.t_sne,
    )

    model = EEGPTUnifiedModel(
        encoder=encoder,
        classifier=classifier,
        chan_conv=None,
        grad_cam=model_conf.grad_cam,
    )
    return model.to(device)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing[:10]:
        print("missing_keys_sample=", missing[:10])
    if unexpected[:10]:
        print("unexpected_keys_sample=", unexpected[:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file", default="baseline/eegpt/eegpt_bcic2a.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_MAP), default="test")
    parser.add_argument("--dataset", default="bcic_2a")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_cfg(args.conf_file, args.overrides)
    device = torch.device(args.device)
    ds_info = build_ds_info(cfg)
    model = build_model(cfg, ds_info, device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    factory = EegptDataLoaderFactory(
        batch_size=cfg.data.batch_size,
        num_workers=0,
        seed=cfg.seed,
    )
    loader, _ = factory.loading_dataset(
        datasets_config={args.dataset: cfg.data.datasets[args.dataset]},
        split=SPLIT_MAP[args.split],
        fs=cfg.fs,
        num_replicas=1,
        rank=0,
    )

    n_class = ds_info[args.dataset]["n_class"]
    labels_all = []
    preds_all = []
    logits_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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


if __name__ == "__main__":
    main()
