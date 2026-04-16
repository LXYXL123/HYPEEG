#!/usr/bin/env python3
"""Setup-gap invariance plus light batch discrimination for relation_cgeom.

This script is the single pretraining path for the current
Setup-Conditioned Hierarchical Relation-CGeom model:

  teacher weak view -> EMA encoder targets
  student corrupted views -> predictor heads -> invariance + discrimination

It intentionally does not use the old TS2Vec objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.relation_cgeom.encoder_three_branch_hyp_relation_cgeom import (  # noqa: E402
    TSEncoderThreeBranchHypRelationCGeom,
)
from scripts.pretrain_relation_cgeom_hf import load_pretrain_array, parse_bands  # noqa: E402


@dataclass
class PretrainDataset:
    dataset: str
    config: str
    train_data: np.ndarray
    train_channel_ids: np.ndarray
    eval_data: np.ndarray
    eval_channel_ids: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.config}".replace("-", "_").replace("/", "_")

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.config}"


class PredictorHead(nn.Module):
    """Small student-only predictor used for distillation targets."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureQueue:
    """Detached FIFO queue of teacher main_global features for extra negatives."""

    def __init__(self, dim: int, size: int, device: torch.device):
        self.size = int(size)
        self.ptr = 0
        self.full = False
        self.buffer = torch.empty(self.size, dim, dtype=torch.float32, device=device) if self.size > 0 else None

    def get(self) -> torch.Tensor | None:
        if self.buffer is None:
            return None
        n_valid = self.size if self.full else self.ptr
        if n_valid <= 0:
            return None
        return self.buffer[:n_valid].detach()

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor) -> None:
        if self.buffer is None:
            return
        features = features.detach().to(device=self.buffer.device, dtype=self.buffer.dtype)
        if features.ndim != 2 or features.numel() == 0:
            return
        if features.size(0) >= self.size:
            self.buffer.copy_(features[-self.size:])
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + features.size(0)
        if end <= self.size:
            self.buffer[self.ptr:end].copy_(features)
        else:
            first = self.size - self.ptr
            self.buffer[self.ptr:].copy_(features[:first])
            self.buffer[:end - self.size].copy_(features[first:])
        self.ptr = end % self.size
        self.full = self.full or end >= self.size


def parse_dataset_configs(spec: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            dataset, config = chunk.split(":", 1)
        else:
            dataset, config = chunk, "pretrain_bci"
        dataset = dataset.strip()
        config = config.strip()
        if not dataset or not config:
            raise ValueError(f"Invalid dataset config entry: {chunk}")
        pairs.append((dataset, config))
    if not pairs:
        raise ValueError("At least one dataset config is required.")
    return pairs


def resolve_data_root(data_root: str) -> str:
    if data_root.startswith("s3://") or os.path.isabs(data_root):
        return data_root
    return str((PROJECT_ROOT / data_root).resolve())


def resolve_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith("s3://") or os.path.isabs(path):
        return path
    return str((PROJECT_ROOT / path).resolve())


def load_datasets(args: argparse.Namespace) -> list[PretrainDataset]:
    data_root = resolve_data_root(args.data_root)
    array_cache_dir = resolve_optional_path(args.array_cache_dir)
    datasets = []
    for dataset_name, config_name in parse_dataset_configs(args.dataset_configs):
        train_data, train_channel_ids = load_pretrain_array(
            dataset_name=dataset_name,
            config_name=config_name,
            fs=args.fs,
            data_root=data_root,
            split=args.split,
            scale=args.scale,
            max_samples=args.max_samples,
            array_cache_dir=array_cache_dir,
            overwrite_array_cache=args.overwrite_array_cache,
            mmap_array_cache=args.mmap_array_cache,
        )
        eval_data, eval_channel_ids = load_pretrain_array(
            dataset_name=dataset_name,
            config_name=config_name,
            fs=args.fs,
            data_root=data_root,
            split=args.eval_split,
            scale=args.scale,
            max_samples=args.max_eval_samples if args.max_eval_samples is not None else args.max_samples,
            array_cache_dir=array_cache_dir,
            overwrite_array_cache=args.overwrite_array_cache,
            mmap_array_cache=args.mmap_array_cache,
        )

        train_valid = ~np.isnan(train_data).all(axis=2).all(axis=1)
        eval_valid = ~np.isnan(eval_data).all(axis=2).all(axis=1)
        train_data = train_data[train_valid]
        train_channel_ids = train_channel_ids[train_valid]
        eval_data = eval_data[eval_valid]
        eval_channel_ids = eval_channel_ids[eval_valid]

        datasets.append(
            PretrainDataset(
                dataset=dataset_name,
                config=config_name,
                train_data=train_data,
                train_channel_ids=train_channel_ids,
                eval_data=eval_data,
                eval_channel_ids=eval_channel_ids,
            )
        )
        print(
            f"Loaded {dataset_name}/{config_name}: "
            f"train={train_data.shape}, eval={eval_data.shape}",
            flush=True,
        )
    return datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup-gap invariance pretraining for relation_cgeom.")
    parser.add_argument(
        "--dataset-configs",
        default="motor_mv_img:pretrain_bci,cho2017:pretrain_bci",
        help="Comma-separated dataset:config entries.",
    )
    parser.add_argument("--data-root", default="assets/data/pretrain")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.001, help="Convert saved uV data to model input scale.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--array-cache-dir",
        default="assets/data/pretrain/array_cache",
        help="Decoded .npy cache directory. Reused before falling back to Arrow.",
    )
    parser.add_argument("--overwrite-array-cache", action="store_true")
    parser.add_argument("--mmap-array-cache", action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--sampling-alpha", type=float, default=1.0)
    parser.add_argument("--channel-subsample-ratio", type=float, default=1.0)
    parser.add_argument("--min-channel-subsample", type=int, default=1)
    parser.add_argument("--max-train-length", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-momentum", type=float, default=0.995)
    parser.add_argument("--lambda-rel", type=float, default=1.0)
    parser.add_argument("--lambda-disc-max", type=float, default=0.05)
    parser.add_argument("--disc-tau", type=float, default=0.2)
    parser.add_argument(
        "--disc-queue-size",
        type=int,
        default=512,
        help="Per-dataset teacher main_global queue size used as extra InfoNCE negatives.",
    )

    parser.add_argument("--repr-dims", type=int, default=320)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--bands", default="8-13,13-20,20-30")
    parser.add_argument("--share-band-encoder", action="store_true")
    parser.add_argument("--disable-raw-branch", action="store_true")
    parser.add_argument("--disable-hyperbolic-branch", action="store_true")
    parser.add_argument("--band-fusion-type", default="concat_linear", choices=["concat_linear", "gated_sum"])
    parser.add_argument("--hyperbolic-depth", type=int, default=1)
    parser.add_argument("--hyperbolic-curvature", type=float, default=1.0)
    parser.add_argument("--fixed-curvature", action="store_true")
    parser.add_argument("--use-setup-conditioned", action="store_true", default=True)
    parser.add_argument("--disable-setup-conditioned", action="store_true")
    parser.add_argument("--setup-meta-dim", type=int, default=128)
    parser.add_argument("--setup-ctx-dim", type=int, default=128)
    parser.add_argument("--setup-dim", type=int, default=128)
    parser.add_argument("--setup-channel-vocab-size", type=int, default=128)
    parser.add_argument("--setup-condition-scale", type=float, default=0.1)
    parser.add_argument("--use-variable-channel-frontend", action="store_true", default=True)
    parser.add_argument("--disable-variable-channel-frontend", action="store_true")
    parser.add_argument("--per-channel-stem-depth", type=int, default=2)
    parser.add_argument("--channel-attn-heads", type=int, default=4)
    parser.add_argument("--mask-mode", default="binomial")

    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "output_dims": args.repr_dims,
        "hidden_dims": args.hidden_dims,
        "depth": args.depth,
        "sampling_rate": float(args.fs),
        "bands": parse_bands(args.bands),
        "use_raw_branch": not args.disable_raw_branch,
        "use_complex_branch": True,
        "use_hyperbolic_branch": not args.disable_hyperbolic_branch,
        "share_band_encoder": args.share_band_encoder,
        "band_fusion_type": args.band_fusion_type,
        "tf_fusion_type": "concat_linear",
        "global_fusion_type": "concat_linear",
        "hyperbolic_depth": args.hyperbolic_depth,
        "hyperbolic_curvature": args.hyperbolic_curvature,
        "learnable_curvature": not args.fixed_curvature,
        "use_setup_conditioned": bool(args.use_setup_conditioned and not args.disable_setup_conditioned),
        "setup_meta_dim": args.setup_meta_dim,
        "setup_ctx_dim": args.setup_ctx_dim,
        "setup_dim": args.setup_dim,
        "setup_channel_vocab_size": args.setup_channel_vocab_size,
        "setup_condition_scale": args.setup_condition_scale,
        "use_variable_channel_frontend": bool(
            args.use_variable_channel_frontend and not args.disable_variable_channel_frontend
        ),
        "per_channel_stem_depth": args.per_channel_stem_depth,
        "channel_attn_heads": args.channel_attn_heads,
        "mask_mode": args.mask_mode,
    }


def make_output_dir(args: argparse.Namespace, datasets: list[PretrainDataset]) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    dataset_part = "_".join(f"{ds.dataset}-{ds.config}" for ds in datasets)
    output_dir = (
        PROJECT_ROOT
        / "assets"
        / "run"
        / "pretrain"
        / "relation_cgeom"
        / f"setup_gap_disc_{dataset_part}"
        / f"local_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_loaders(
    datasets: list[PretrainDataset],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[list[DataLoader], list[DataLoader]]:
    train_loaders = []
    eval_loaders = []
    for ds in datasets:
        train_dataset = TensorDataset(
            torch.from_numpy(ds.train_data).to(torch.float),
            torch.from_numpy(ds.train_channel_ids).to(torch.long),
        )
        if len(train_dataset) < batch_size:
            raise RuntimeError(f"{ds.label} has fewer train samples than batch_size={batch_size}.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        eval_dataset = TensorDataset(
            torch.from_numpy(ds.eval_data).to(torch.float),
            torch.from_numpy(ds.eval_channel_ids).to(torch.long),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)
        print(
            f"Loaders {ds.label}: train_batches={len(train_loader)} eval_batches={len(eval_loader)}",
            flush=True,
        )
    return train_loaders, eval_loaders


def next_batch(loaders: list[DataLoader], iterators: list, dataset_idx: int):
    try:
        return next(iterators[dataset_idx])
    except StopIteration:
        iterators[dataset_idx] = iter(loaders[dataset_idx])
        return next(iterators[dataset_idx])


def build_epoch_schedule(
    loader_lengths: np.ndarray,
    rng: np.random.Generator,
    steps_per_epoch: int | None,
    sampling_alpha: float,
) -> np.ndarray:
    if steps_per_epoch is None:
        schedule = np.concatenate([np.full(int(length), i, dtype=np.int64) for i, length in enumerate(loader_lengths)])
        rng.shuffle(schedule)
        return schedule

    weights = np.power(loader_lengths.astype(np.float64), sampling_alpha)
    weights = weights / weights.sum()
    return rng.choice(len(loader_lengths), size=steps_per_epoch, replace=True, p=weights)


def maybe_crop_time(x: torch.Tensor, max_length: int | None, rng: np.random.Generator) -> torch.Tensor:
    if max_length is None or x.size(1) <= max_length:
        return x
    offset = int(rng.integers(x.size(1) - max_length + 1))
    return x[:, offset: offset + max_length]


def maybe_subsample_channels(
    x: torch.Tensor,
    channel_ids: torch.Tensor,
    ratio: float,
    min_channels: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Randomly keep a batch-shared channel subset to reduce memory.

    x: [B, T, C]
    channel_ids: [B, C]
    """
    n_channels = x.size(-1)
    if ratio >= 1.0 or n_channels <= 1:
        return x, channel_ids, n_channels

    n_keep = int(round(n_channels * max(0.0, ratio)))
    n_keep = max(int(min_channels), n_keep)
    n_keep = min(n_channels, max(1, n_keep))
    if n_keep >= n_channels:
        return x, channel_ids, n_channels

    selected = np.sort(rng.choice(n_channels, size=n_keep, replace=False))
    selected = torch.as_tensor(selected, dtype=torch.long, device=x.device)
    return x.index_select(-1, selected), channel_ids.index_select(-1, selected), n_keep


def apply_very_light_band_perturbation(
    x: torch.Tensor,
    sampling_rate: float,
    bands: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Apply tiny band-wise gain jitter without changing EEG semantics."""
    dtype = x.dtype
    x_float = x.to(torch.float32)
    spectrum = torch.fft.rfft(x_float.contiguous(), dim=1)
    freqs = torch.fft.rfftfreq(x.size(1), d=1.0 / sampling_rate).to(x.device)
    for low_hz, high_hz in bands[:3]:
        freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not freq_mask.any():
            continue
        gain = torch.empty(x.size(0), 1, 1, device=x.device, dtype=x_float.dtype).uniform_(0.98, 1.02)
        spectrum[:, freq_mask, :] = spectrum[:, freq_mask, :] * gain
    return torch.fft.irfft(spectrum, n=x.size(1), dim=1).to(dtype=dtype)


def augment_view(
    x: torch.Tensor,
    *,
    time_mask_ratio: float,
    channel_dropout_ratio: float,
    noise_std: float,
    amplitude_low: float,
    amplitude_high: float,
    band_perturb: bool,
    sampling_rate: float,
    bands: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Create one setup-corrupted view. Shape stays [B, T, C]."""
    out = x.clone()
    scale = torch.empty(out.size(0), 1, 1, device=out.device, dtype=out.dtype).uniform_(amplitude_low, amplitude_high)
    out = out * scale
    if band_perturb:
        out = apply_very_light_band_perturbation(out, sampling_rate=sampling_rate, bands=bands)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    if time_mask_ratio > 0:
        time_mask = torch.rand(out.size(0), out.size(1), 1, device=out.device) < time_mask_ratio
        out = out.masked_fill(time_mask, 0)
    if channel_dropout_ratio > 0:
        channel_mask = torch.rand(out.size(0), 1, out.size(2), device=out.device) < channel_dropout_ratio
        out = out.masked_fill(channel_mask, 0)
    return out


def make_teacher_view(x: torch.Tensor, args: argparse.Namespace, bands) -> torch.Tensor:
    return augment_view(
        x,
        time_mask_ratio=0.05,
        channel_dropout_ratio=0.0,
        noise_std=0.005,
        amplitude_low=0.98,
        amplitude_high=1.02,
        band_perturb=False,
        sampling_rate=float(args.fs),
        bands=bands,
    )


def make_student_view(x: torch.Tensor, args: argparse.Namespace, bands) -> torch.Tensor:
    return augment_view(
        x,
        time_mask_ratio=0.15,
        channel_dropout_ratio=0.15,
        noise_std=0.01,
        amplitude_low=0.95,
        amplitude_high=1.05,
        band_perturb=True,
        sampling_rate=float(args.fs),
        bands=bands,
    )


def forward_globals(
    encoder: nn.Module,
    x: torch.Tensor,
    channel_ids: torch.Tensor,
    sampling_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    setup = {
        "channel_ids": channel_ids,
        "sampling_rate": float(sampling_rate),
        "window_len": int(x.size(1)),
    }
    out = encoder(x, mask="all_true", return_aux=True, setup=setup)
    if "main_global" not in out or "relation_global" not in out:
        raise RuntimeError("Encoder forward must return main_global and relation_global for setup-gap pretraining.")
    return out["main_global"], out["relation_global"], out


def cos_sim(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    return (u * v).sum(dim=-1).mean()


def cos_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return 1.0 - cos_sim(u, v)


def info_nce(anchor: torch.Tensor, target: torch.Tensor, tau: float, queue: torch.Tensor | None = None) -> torch.Tensor:
    """Batch InfoNCE over current teacher targets plus optional queued negatives.

    anchor: [B, D], student predictor output
    target: [B, D], matching EMA teacher target
    queue: [Q, D], detached extra negatives
    """
    anchor = F.normalize(anchor, dim=-1)
    target = F.normalize(target, dim=-1)
    candidates = target if queue is None else torch.cat([target, F.normalize(queue.to(target.device), dim=-1)], dim=0)
    logits = anchor @ candidates.T / tau
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def disc_accuracy_in_batch(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    target = F.normalize(target, dim=-1)
    pred = torch.argmax(anchor @ target.T, dim=-1)
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return (pred == labels).to(torch.float32).mean()


def get_lambda_disc(progress: float, lambda_disc_max: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.10:
        return 0.0
    if progress < 0.30:
        return float(lambda_disc_max) * (progress - 0.10) / 0.20
    return float(lambda_disc_max)


def batch_std(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=0, unbiased=False).mean()


def mean_debug_scalar(aux: dict, key: str, index: int | None = None) -> float | None:
    value = aux.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return None
    if index is not None:
        value = value[:, index]
    return float(value.detach().float().mean().cpu().item())


def compute_step_loss(
    student: nn.Module,
    teacher: nn.Module,
    predictor_main: nn.Module,
    predictor_rel: nn.Module,
    x: torch.Tensor,
    channel_ids: torch.Tensor,
    args: argparse.Namespace,
    bands: tuple[tuple[float, float], ...],
    lambda_disc: float,
    disc_queue: FeatureQueue | None = None,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    x_teacher = make_teacher_view(x, args, bands)
    x_student_a = make_student_view(x, args, bands)
    x_student_b = make_student_view(x, args, bands)

    with torch.no_grad():
        main_t, rel_t, _ = forward_globals(teacher, x_teacher, channel_ids, args.fs)
        main_t = main_t.detach()
        rel_t = rel_t.detach()

    main_a, rel_a, aux_a = forward_globals(student, x_student_a, channel_ids, args.fs)
    main_b, rel_b, _ = forward_globals(student, x_student_b, channel_ids, args.fs)

    pred_main_a = predictor_main(main_a)
    pred_main_b = predictor_main(main_b)
    pred_rel_a = predictor_rel(rel_a)
    pred_rel_b = predictor_rel(rel_b)

    loss_main_inv_a = cos_loss(pred_main_a, main_t)
    loss_relation_inv_a = cos_loss(pred_rel_a, rel_t)
    loss_main_inv_b = cos_loss(pred_main_b, main_t)
    loss_relation_inv_b = cos_loss(pred_rel_b, rel_t)
    loss_inv = (
        loss_main_inv_a
        + args.lambda_rel * loss_relation_inv_a
        + loss_main_inv_b
        + args.lambda_rel * loss_relation_inv_b
    )

    queued_targets = None if disc_queue is None else disc_queue.get()
    loss_disc_a = info_nce(pred_main_a, main_t, tau=args.disc_tau, queue=queued_targets)
    loss_disc_b = info_nce(pred_main_b, main_t, tau=args.disc_tau, queue=queued_targets)
    loss_disc = 0.5 * (loss_disc_a + loss_disc_b)
    loss_total = loss_inv + float(lambda_disc) * loss_disc

    metrics = {
        "loss_total": loss_total.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_disc": loss_disc.detach(),
        "loss_main_inv_a": loss_main_inv_a.detach(),
        "loss_relation_inv_a": loss_relation_inv_a.detach(),
        "loss_main_inv_b": loss_main_inv_b.detach(),
        "loss_relation_inv_b": loss_relation_inv_b.detach(),
        "lambda_disc": float(lambda_disc),
        "disc_queue_size": 0 if queued_targets is None else int(queued_targets.size(0)),
        "disc_acc_in_batch": disc_accuracy_in_batch(pred_main_a.detach(), main_t).detach(),
        "cos_sim_main_at": cos_sim(pred_main_a.detach(), main_t).detach(),
        "cos_sim_rel_at": cos_sim(pred_rel_a.detach(), rel_t).detach(),
        "cos_sim_main_bt": cos_sim(pred_main_b.detach(), main_t).detach(),
        "cos_sim_rel_bt": cos_sim(pred_rel_b.detach(), rel_t).detach(),
        "std_main_global_batch": batch_std(main_a.detach()).detach(),
        "std_relation_global_batch": batch_std(rel_a.detach()).detach(),
    }
    for idx in range(3):
        value = mean_debug_scalar(aux_a, "band_weights", idx)
        if value is not None:
            metrics[f"band_weight_{idx}"] = value
    for key in ("alpha", "beta", "w_g", "w_f"):
        value = mean_debug_scalar(aux_a, key)
        if value is not None:
            metrics[key] = value
    return loss_total, metrics, main_t.detach()


def to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def accumulate_metrics(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + to_float(value)


def average_metrics(sums: dict, count: int, prefix: str = "") -> dict:
    if count <= 0:
        return {}
    return {f"{prefix}{key}": float(value / count) for key, value in sums.items()}


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def optimizer_update(
    student: nn.Module,
    teacher: nn.Module,
    predictor_main: nn.Module,
    predictor_rel: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    params = list(student.parameters()) + list(predictor_main.parameters()) + list(predictor_rel.parameters())
    if scaler.is_enabled():
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    update_ema_teacher(student, teacher, momentum=args.ema_momentum)


@torch.no_grad()
def evaluate_loss(
    student: nn.Module,
    teacher: nn.Module,
    predictor_main: nn.Module,
    predictor_rel: nn.Module,
    eval_loaders: list[DataLoader],
    datasets: list[PretrainDataset],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    rng: np.random.Generator,
    bands: tuple[tuple[float, float], ...],
    lambda_disc: float,
) -> tuple[float, list[dict]]:
    student.eval()
    teacher.eval()
    predictor_main.eval()
    predictor_rel.eval()

    per_dataset = []
    for ds, loader in zip(datasets, eval_loaders):
        metric_sums = {}
        n_batches = 0
        iterator = loader
        if not args.no_progress:
            iterator = tqdm(loader, desc=f"Setup-gap eval {ds.dataset}", leave=False, dynamic_ncols=True)
        for x, channel_ids in iterator:
            x = maybe_crop_time(x, args.max_train_length, rng)
            x, channel_ids, _ = maybe_subsample_channels(
                x,
                channel_ids,
                ratio=args.channel_subsample_ratio,
                min_channels=args.min_channel_subsample,
                rng=rng,
            )
            x = x.to(device, non_blocking=True)
            channel_ids = channel_ids.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                _, metrics, _ = compute_step_loss(
                    student,
                    teacher,
                    predictor_main,
                    predictor_rel,
                    x,
                    channel_ids,
                    args,
                    bands,
                    lambda_disc=lambda_disc,
                    disc_queue=None,
                )
            accumulate_metrics(metric_sums, metrics)
            n_batches += 1
        record = average_metrics(metric_sums, n_batches, prefix="eval_")
        record["dataset"] = ds.key
        per_dataset.append(record)

    eval_loss = float(np.mean([record["eval_loss_total"] for record in per_dataset]))
    return eval_loss, per_dataset


def save_checkpoint(
    student: nn.Module,
    output_dir: Path,
    epoch_record: dict,
    best_eval_loss: dict,
) -> None:
    epoch = int(epoch_record["epoch"])
    epoch_path = output_dir / f"relation_cgeom_encoder_epoch_{epoch:03d}.pt"
    last_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"

    torch.save(student.state_dict(), epoch_path)
    shutil.copyfile(epoch_path, last_path)
    if epoch_record["eval_loss_total"] < best_eval_loss["value"]:
        best_eval_loss["value"] = float(epoch_record["eval_loss_total"])
        shutil.copyfile(epoch_path, best_path)
        epoch_record["is_best"] = True
    else:
        epoch_record["is_best"] = False

    print(
        "Epoch #{epoch}: train_loss={train_loss:.6f} eval_loss={eval_loss:.6f} "
        "train_time={train_time:.2f}s eval_time={eval_time:.2f}s total_time={total_time:.2f}s "
        "best={is_best}".format(
            epoch=epoch,
            train_loss=epoch_record["loss_total"],
            eval_loss=epoch_record["eval_loss_total"],
            train_time=epoch_record["epoch_time_sec"],
            eval_time=epoch_record["eval_time_sec"],
            total_time=epoch_record["total_epoch_time_sec"],
            is_best=epoch_record["is_best"],
        ),
        flush=True,
    )
    print(f"Saved epoch checkpoint: {epoch_path}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.disable_setup_conditioned or args.disable_variable_channel_frontend:
        raise ValueError("Setup-gap pretraining requires the setup-conditioned variable-channel frontend.")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    datasets = load_datasets(args)
    output_dir = make_output_dir(args, datasets)
    metrics_path = output_dir / "metrics.jsonl"
    model_config = build_model_config(args)
    bands = tuple(model_config["bands"])
    input_dims = max(ds.train_data.shape[-1] for ds in datasets)

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Model input_dims placeholder: {input_dims}", flush=True)
    print(
        "Dataset schedule: "
        + ", ".join(f"{ds.label} train={ds.train_data.shape} eval={ds.eval_data.shape}" for ds in datasets),
        flush=True,
    )

    device = torch.device(args.device)
    student = TSEncoderThreeBranchHypRelationCGeom(
        input_dims=input_dims,
        **model_config,
    ).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    predictor_main = PredictorHead(args.repr_dims * 2).to(device)
    predictor_rel = PredictorHead(args.repr_dims).to(device)
    disc_queues = [
        FeatureQueue(dim=args.repr_dims * 2, size=args.disc_queue_size, device=device)
        for _ in datasets
    ]

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(predictor_main.parameters()) + list(predictor_rel.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    pin_memory = device.type == "cuda"
    train_loaders, eval_loaders = make_loaders(datasets, args.batch_size, eval_batch_size, args.num_workers, pin_memory)
    loader_lengths = np.asarray([len(loader) for loader in train_loaders], dtype=np.int64)
    epoch_steps = int(args.steps_per_epoch) if args.steps_per_epoch is not None else int(loader_lengths.sum())
    total_updates = math.ceil(epoch_steps / max(1, args.grad_accum_steps)) * max(1, args.epochs)
    scheduler = build_scheduler(optimizer, total_steps=total_updates, warmup_ratio=args.warmup_ratio)

    amp_enabled = bool(args.use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    loss_log = []
    train_history = []
    best_eval_loss = {"value": float("inf")}
    global_update_steps = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        student.train()
        teacher.eval()
        predictor_main.train()
        predictor_rel.train()
        iterators = [iter(loader) for loader in train_loaders]
        schedule = build_epoch_schedule(loader_lengths, rng, args.steps_per_epoch, args.sampling_alpha)

        metric_sums = {}
        dataset_step_counts = {ds.key: 0 for ds in datasets}
        n_micro_steps = 0
        n_optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        iterator = schedule
        if not args.no_progress:
            iterator = tqdm(schedule, desc=f"Setup-gap pretrain epoch {epoch}", leave=True, dynamic_ncols=True)

        for dataset_idx in iterator:
            dataset_idx = int(dataset_idx)
            x, channel_ids = next_batch(train_loaders, iterators, dataset_idx)
            x = maybe_crop_time(x, args.max_train_length, rng)
            x, channel_ids, n_channels_kept = maybe_subsample_channels(
                x,
                channel_ids,
                ratio=args.channel_subsample_ratio,
                min_channels=args.min_channel_subsample,
                rng=rng,
            )
            x = x.to(device, non_blocking=True)
            channel_ids = channel_ids.to(device, non_blocking=True)
            progress = float(global_update_steps) / float(max(1, total_updates))
            lambda_disc = get_lambda_disc(progress, args.lambda_disc_max)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                loss, metrics, queue_features = compute_step_loss(
                    student,
                    teacher,
                    predictor_main,
                    predictor_rel,
                    x,
                    channel_ids,
                    args,
                    bands,
                    lambda_disc=lambda_disc,
                    disc_queue=disc_queues[dataset_idx],
                )
                loss_for_backward = loss / max(1, args.grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            n_micro_steps += 1
            dataset_step_counts[datasets[dataset_idx].key] += 1
            accumulate_metrics(metric_sums, metrics)
            disc_queues[dataset_idx].enqueue(queue_features)

            should_step = n_micro_steps % max(1, args.grad_accum_steps) == 0
            if should_step:
                optimizer_update(
                    student,
                    teacher,
                    predictor_main,
                    predictor_rel,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                )
                global_update_steps += 1
                n_optimizer_steps += 1

            if not args.no_progress:
                iterator.set_postfix(
                    {
                        "dataset": datasets[dataset_idx].dataset,
                        "chs": n_channels_kept,
                        "loss": f"{to_float(metrics['loss_total']):.5f}",
                        "inv": f"{to_float(metrics['loss_inv']):.5f}",
                        "disc": f"{to_float(metrics['loss_disc']):.5f}",
                        "ld": f"{to_float(metrics['lambda_disc']):.3f}",
                        "dacc": f"{to_float(metrics['disc_acc_in_batch']):.3f}",
                    }
                )

        if n_micro_steps % max(1, args.grad_accum_steps) != 0:
            optimizer_update(
                student,
                teacher,
                predictor_main,
                predictor_rel,
                optimizer,
                scheduler,
                scaler,
                args,
            )
            global_update_steps += 1
            n_optimizer_steps += 1

        train_time_sec = time.time() - epoch_start
        eval_start = time.time()
        eval_loss, per_dataset_eval = evaluate_loss(
            student,
            teacher,
            predictor_main,
            predictor_rel,
            eval_loaders,
            datasets,
            args,
            device,
            amp_enabled,
            amp_dtype,
            rng,
            bands,
            lambda_disc=get_lambda_disc(float(global_update_steps) / float(max(1, total_updates)), args.lambda_disc_max),
        )
        eval_time_sec = time.time() - eval_start

        epoch_record = {
            "epoch": int(epoch),
            "epoch_time_sec": float(train_time_sec),
            "eval_time_sec": float(eval_time_sec),
            "total_epoch_time_sec": float(train_time_sec + eval_time_sec),
            "eval_loss_total": float(eval_loss),
            "optimizer_steps": int(n_optimizer_steps),
            "global_update_steps": int(global_update_steps),
            "grad_accum_steps": int(args.grad_accum_steps),
            "lr": float(scheduler.get_last_lr()[0]),
            "channel_subsample_ratio": float(args.channel_subsample_ratio),
            "min_channel_subsample": int(args.min_channel_subsample),
        }
        epoch_record.update(average_metrics(metric_sums, n_micro_steps))
        for key, value in dataset_step_counts.items():
            epoch_record[f"train_steps_{key}"] = int(value)
        for record in per_dataset_eval:
            ds_key = record.pop("dataset")
            for key, value in record.items():
                epoch_record[f"{ds_key}_{key}"] = float(value)

        save_checkpoint(student, output_dir, epoch_record, best_eval_loss)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_record, sort_keys=True) + "\n")

        train_history.append(epoch_record)
        loss_log.append(epoch_record["loss_total"])

    last_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"
    if last_path.exists() and not best_path.exists():
        shutil.copyfile(last_path, best_path)

    with open(output_dir / "train_history.pkl", "wb") as f:
        pickle.dump(train_history, f)
    with open(output_dir / "loss_log.pkl", "wb") as f:
        pickle.dump(loss_log, f)

    print(f"Saved last encoder checkpoint: {last_path}", flush=True)
    print(f"Saved best encoder checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
