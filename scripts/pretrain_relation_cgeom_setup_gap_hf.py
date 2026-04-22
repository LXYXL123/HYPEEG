#!/usr/bin/env python3
"""Prototype self-distillation pretraining for relation_cgeom.

This script is the single self-supervised pretraining path for the current
Setup-Conditioned Relation-CGeom model:

  teacher weak view -> EMA encoder targets -> per-dataset prototype targets
  student corrupted views -> shared main projector -> prototype matching

The main supervision is on main_global, with a light relation_global
consistency term. It intentionally does not use TS2Vec or instance InfoNCE.
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.distributed.env import (  # noqa: E402
    clean_torch_distributed,
    get_global_rank,
    get_is_master,
    get_local_rank,
    get_master_addr,
    get_master_port,
    get_world_size,
)

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


class MainProjector(nn.Module):
    """Shared projector for main_global before prototype assignment."""

    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiDatasetPrototypeHeads(nn.Module):
    """Per-dataset prototype heads over projected main_global."""

    def __init__(self, dataset_keys: list[str], input_dim: int, num_prototypes: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            dataset_key: nn.Linear(input_dim, num_prototypes, bias=False)
            for dataset_key in dataset_keys
        })

    def forward(self, x: torch.Tensor, dataset_key: str) -> torch.Tensor:
        return self.heads[dataset_key](x)


def print0(*args, **kwargs) -> None:
    if get_is_master():
        print(*args, **kwargs)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def broadcast_path(path: Path | None) -> Path:
    if not is_distributed():
        assert path is not None
        return path
    obj_list = [str(path) if path is not None else ""]
    dist.broadcast_object_list(obj_list, src=0)
    return Path(obj_list[0])


def setup_runtime(args: argparse.Namespace) -> tuple[torch.device, bool]:
    world_size = get_world_size()
    distributed = world_size > 1
    if distributed:
        rank = get_global_rank()
        local_rank = get_local_rank()
        master_addr = get_master_addr()
        master_port = get_master_port(
            job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
            port=int(os.environ.get("MASTER_PORT", 29500)),
            is_port_random=False,
        )
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    return device, distributed


def reduce_metric_sums(metric_sums: dict[str, float], count: int, device: torch.device) -> tuple[dict[str, float], int]:
    if not is_distributed():
        return metric_sums, count
    keys = sorted(metric_sums.keys())
    values = [float(metric_sums[key]) for key in keys]
    payload = torch.tensor(values + [float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    reduced = {key: float(payload[idx].item()) for idx, key in enumerate(keys)}
    reduced_count = int(payload[-1].item())
    return reduced, reduced_count


def reduce_count_dict(count_dict: dict[str, int], device: torch.device) -> dict[str, int]:
    if not is_distributed():
        return count_dict
    keys = sorted(count_dict.keys())
    payload = torch.tensor([float(count_dict[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return {key: int(payload[idx].item()) for idx, key in enumerate(keys)}


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
        print0(
            f"Loaded {dataset_name}/{config_name}: "
            f"train={train_data.shape}, eval={eval_data.shape}",
            flush=True,
        )
    return datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype self-distillation pretraining for relation_cgeom.")
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
    parser.add_argument("--lambda-rel", type=float, default=0.1)
    parser.add_argument("--proto-dim", type=int, default=256)
    parser.add_argument("--num-prototypes", type=int, default=64)
    parser.add_argument("--proto-teacher-temp", type=float, default=0.10)
    parser.add_argument("--proto-teacher-temp-start", type=float, default=0.20)
    parser.add_argument("--proto-teacher-temp-warmup-ratio", type=float, default=0.30)
    parser.add_argument("--proto-student-temp", type=float, default=0.10)
    parser.add_argument("--proto-center-momentum", type=float, default=0.9)
    parser.add_argument("--lambda-proto-balance", type=float, default=0.1)

    parser.add_argument("--repr-dims", type=int, default=320)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--architecture", default="default", choices=["default", "spectral_lite"])
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
    parser.add_argument("--spectral-dims", type=int, default=64)
    parser.add_argument("--spectral-win-len", type=int, default=64)
    parser.add_argument("--spectral-stride", type=int, default=32)
    parser.add_argument("--spectral-freq-low", type=float, default=4.0)
    parser.add_argument("--spectral-freq-high", type=float, default=40.0)
    parser.add_argument("--spectral-mixer-depth", type=int, default=1)
    parser.add_argument("--use-heegnet-lorentz", action="store_true")
    parser.add_argument("--heegnet-lorentz-dim", type=int, default=65)
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
        "architecture": args.architecture,
        "sampling_rate": float(args.fs),
        "bands": parse_bands(args.bands),
        "use_raw_branch": not args.disable_raw_branch,
        "use_complex_branch": args.architecture != "spectral_lite",
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
        "spectral_dims": args.spectral_dims,
        "spectral_win_len": args.spectral_win_len,
        "spectral_stride": args.spectral_stride,
        "spectral_freq_low": args.spectral_freq_low,
        "spectral_freq_high": args.spectral_freq_high,
        "spectral_mixer_depth": args.spectral_mixer_depth,
        "use_heegnet_lorentz": args.use_heegnet_lorentz,
        "heegnet_lorentz_dim": args.heegnet_lorentz_dim,
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
    prefix = "proto_balanced_distill" if args.architecture == "default" else f"proto_balanced_distill_{args.architecture}"
    output_dir = (
        PROJECT_ROOT
        / "assets"
        / "run"
        / "pretrain"
        / "relation_cgeom"
        / f"{prefix}_{dataset_part}"
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
    distributed: bool,
) -> tuple[list[DataLoader], list[DataLoader], list[DistributedSampler | None]]:
    train_loaders = []
    eval_loaders = []
    train_samplers = []
    world_size = get_world_size()
    rank = get_global_rank()
    for ds in datasets:
        train_dataset = TensorDataset(
            torch.from_numpy(ds.train_data).to(torch.float),
            torch.from_numpy(ds.train_channel_ids).to(torch.long),
        )
        if distributed:
            per_rank_train = len(train_dataset) // max(1, world_size)
            if per_rank_train < batch_size:
                raise RuntimeError(
                    f"{ds.label} has only {per_rank_train} train samples per rank for world_size={world_size}, "
                    f"smaller than batch_size={batch_size}."
                )
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_sampler = None
            if len(train_dataset) < batch_size:
                raise RuntimeError(f"{ds.label} has fewer train samples than batch_size={batch_size}.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        eval_dataset = TensorDataset(
            torch.from_numpy(ds.eval_data).to(torch.float),
            torch.from_numpy(ds.eval_channel_ids).to(torch.long),
        )
        eval_sampler = None
        if distributed:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)
        train_samplers.append(train_sampler)
        print0(
            f"Loaders {ds.label}: train_batches={len(train_loader)} eval_batches={len(eval_loader)}",
            flush=True,
        )
    return train_loaders, eval_loaders, train_samplers


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
        time_mask_ratio=0.10,
        channel_dropout_ratio=0.10,
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


def prototype_target_distribution(
    teacher_logits: torch.Tensor,
    center: torch.Tensor,
    teacher_temp: float,
) -> torch.Tensor:
    centered_logits = (teacher_logits - center.unsqueeze(0)) / max(teacher_temp, 1e-6)
    return F.softmax(centered_logits, dim=-1)


def soft_cross_entropy(student_logits: torch.Tensor, teacher_probs: torch.Tensor, student_temp: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / max(student_temp, 1e-6), dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def prototype_balance_loss(student_probs_a: torch.Tensor, student_probs_b: torch.Tensor) -> torch.Tensor:
    """Encourage each batch to use prototypes broadly without using labels."""
    probs = 0.5 * (student_probs_a.mean(dim=0) + student_probs_b.mean(dim=0))
    uniform = torch.full_like(probs, 1.0 / probs.numel())
    return F.kl_div(probs.clamp_min(1e-8).log(), uniform, reduction="batchmean")


def distribution_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1).mean()


@torch.no_grad()
def update_proto_center(center: torch.Tensor, teacher_logits: torch.Tensor, momentum: float) -> None:
    batch_center = teacher_logits.detach().float().mean(dim=0)
    if is_distributed():
        dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
        batch_center /= float(get_world_size())
    center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)


def active_proto_count(teacher_probs: torch.Tensor) -> torch.Tensor:
    assignments = teacher_probs.argmax(dim=-1)
    return torch.tensor(float(assignments.unique().numel()), device=teacher_probs.device)


def prototype_usage_perplexity(probs: torch.Tensor) -> torch.Tensor:
    mean_probs = probs.mean(dim=0).clamp_min(1e-8)
    entropy = -(mean_probs * mean_probs.log()).sum()
    return entropy.exp()


def get_teacher_temperature(args: argparse.Namespace, progress: float) -> float:
    warmup_ratio = max(float(args.proto_teacher_temp_warmup_ratio), 1e-8)
    if progress >= warmup_ratio:
        return float(args.proto_teacher_temp)
    alpha = max(0.0, min(1.0, progress / warmup_ratio))
    return float(args.proto_teacher_temp_start + alpha * (args.proto_teacher_temp - args.proto_teacher_temp_start))


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
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    student_proto_heads: MultiDatasetPrototypeHeads,
    teacher_proto_heads: MultiDatasetPrototypeHeads,
    proto_centers: dict[str, torch.Tensor],
    dataset_key: str,
    x: torch.Tensor,
    channel_ids: torch.Tensor,
    args: argparse.Namespace,
    bands: tuple[tuple[float, float], ...],
    progress: float,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    x_teacher = make_teacher_view(x, args, bands)
    x_student_a = make_student_view(x, args, bands)
    x_student_b = make_student_view(x, args, bands)

    with torch.no_grad():
        main_t, rel_t, _ = forward_globals(teacher, x_teacher, channel_ids, args.fs)
        main_t = main_t.detach()
        rel_t = rel_t.detach()
        proj_t = F.normalize(teacher_projector(main_t), dim=-1)
        logits_t = teacher_proto_heads(proj_t, dataset_key)
        teacher_temp = get_teacher_temperature(args, progress)
        proto_targets = prototype_target_distribution(
            logits_t,
            center=proto_centers[dataset_key],
            teacher_temp=teacher_temp,
        ).detach()

    main_a, rel_a, aux_a = forward_globals(student, x_student_a, channel_ids, args.fs)
    main_b, rel_b, _ = forward_globals(student, x_student_b, channel_ids, args.fs)

    proj_a = F.normalize(student_projector(main_a), dim=-1)
    proj_b = F.normalize(student_projector(main_b), dim=-1)
    logits_a = student_proto_heads(proj_a, dataset_key)
    logits_b = student_proto_heads(proj_b, dataset_key)

    loss_proto_a = soft_cross_entropy(logits_a, proto_targets, student_temp=args.proto_student_temp)
    loss_proto_b = soft_cross_entropy(logits_b, proto_targets, student_temp=args.proto_student_temp)
    loss_proto_main = 0.5 * (loss_proto_a + loss_proto_b)

    loss_rel_a = cos_loss(rel_a, rel_t)
    loss_rel_b = cos_loss(rel_b, rel_t)
    loss_rel_cons = 0.5 * (loss_rel_a + loss_rel_b)

    student_probs_a = F.softmax(logits_a.detach() / max(args.proto_student_temp, 1e-6), dim=-1)
    student_probs_b = F.softmax(logits_b.detach() / max(args.proto_student_temp, 1e-6), dim=-1)
    balance_probs_a = F.softmax(logits_a / max(args.proto_student_temp, 1e-6), dim=-1)
    balance_probs_b = F.softmax(logits_b / max(args.proto_student_temp, 1e-6), dim=-1)
    loss_proto_balance = prototype_balance_loss(balance_probs_a, balance_probs_b)
    loss_total = (
        loss_proto_main
        + args.lambda_rel * loss_rel_cons
        + args.lambda_proto_balance * loss_proto_balance
    )

    metrics = {
        "loss_total": loss_total.detach(),
        "loss_proto_main": loss_proto_main.detach(),
        "loss_proto_a": loss_proto_a.detach(),
        "loss_proto_b": loss_proto_b.detach(),
        "loss_proto_balance": loss_proto_balance.detach(),
        "loss_rel_cons": loss_rel_cons.detach(),
        "loss_rel_a": loss_rel_a.detach(),
        "loss_rel_b": loss_rel_b.detach(),
        "cos_sim_main_at": cos_sim(main_a.detach(), main_t).detach(),
        "cos_sim_rel_at": cos_sim(rel_a.detach(), rel_t).detach(),
        "cos_sim_main_bt": cos_sim(main_b.detach(), main_t).detach(),
        "cos_sim_rel_bt": cos_sim(rel_b.detach(), rel_t).detach(),
        "std_main_global_batch": batch_std(main_a.detach()).detach(),
        "std_relation_global_batch": batch_std(rel_a.detach()).detach(),
        "teacher_entropy": distribution_entropy(proto_targets).detach(),
        "student_entropy_a": distribution_entropy(student_probs_a).detach(),
        "student_entropy_b": distribution_entropy(student_probs_b).detach(),
        "active_proto_count": active_proto_count(proto_targets).detach(),
        "proto_usage_perplexity": prototype_usage_perplexity(proto_targets).detach(),
        "teacher_temp": float(teacher_temp),
        "lambda_proto_balance": float(args.lambda_proto_balance),
    }
    for idx in range(3):
        value = mean_debug_scalar(aux_a, "band_weights", idx)
        if value is not None:
            metrics[f"band_weight_{idx}"] = value
    for key in ("alpha", "beta", "w_g", "w_f"):
        value = mean_debug_scalar(aux_a, key)
        if value is not None:
            metrics[key] = value
    return loss_total, metrics, logits_t.detach()


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
def update_ema_module(student_module: nn.Module, teacher_module: nn.Module, momentum: float) -> None:
    student_module = unwrap_model(student_module)
    teacher_module = unwrap_model(teacher_module)
    for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher_module.buffers(), student_module.buffers()):
        teacher_buffer.copy_(student_buffer)


def optimizer_update(
    student: nn.Module,
    teacher: nn.Module,
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    student_proto_heads: MultiDatasetPrototypeHeads,
    teacher_proto_heads: MultiDatasetPrototypeHeads,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    params = list(student.parameters()) + list(student_projector.parameters()) + list(student_proto_heads.parameters())
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
    update_ema_module(student, teacher, momentum=args.ema_momentum)
    update_ema_module(student_projector, teacher_projector, momentum=args.ema_momentum)
    update_ema_module(student_proto_heads, teacher_proto_heads, momentum=args.ema_momentum)


@torch.no_grad()
def evaluate_loss(
    student: nn.Module,
    teacher: nn.Module,
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    student_proto_heads: MultiDatasetPrototypeHeads,
    teacher_proto_heads: MultiDatasetPrototypeHeads,
    proto_centers: dict[str, torch.Tensor],
    eval_loaders: list[DataLoader],
    datasets: list[PretrainDataset],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    bands: tuple[tuple[float, float], ...],
    progress: float,
) -> tuple[float, list[dict]]:
    student.eval()
    teacher.eval()
    student_projector.eval()
    teacher_projector.eval()
    student_proto_heads.eval()
    teacher_proto_heads.eval()

    per_dataset = []
    for ds, loader in zip(datasets, eval_loaders):
        metric_sums = {}
        n_batches = 0
        iterator = loader
        if not args.no_progress and get_is_master():
            iterator = tqdm(loader, desc=f"Setup-gap eval {ds.dataset}", leave=False, dynamic_ncols=True)
        for x, channel_ids in iterator:
            x = x.to(device, non_blocking=True)
            channel_ids = channel_ids.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                _, metrics, _ = compute_step_loss(
                    student,
                    teacher,
                    student_projector,
                    teacher_projector,
                    student_proto_heads,
                    teacher_proto_heads,
                    proto_centers,
                    ds.key,
                    x,
                    channel_ids,
                    args,
                    bands,
                    progress,
                )
            accumulate_metrics(metric_sums, metrics)
            n_batches += 1
        metric_sums, n_batches = reduce_metric_sums(metric_sums, n_batches, device)
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
    if not get_is_master():
        return
    epoch = int(epoch_record["epoch"])
    epoch_path = output_dir / f"relation_cgeom_encoder_epoch_{epoch:03d}.pt"
    last_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"

    torch.save(unwrap_model(student).state_dict(), epoch_path)
    shutil.copyfile(epoch_path, last_path)
    if epoch_record["eval_loss_total"] < best_eval_loss["value"]:
        best_eval_loss["value"] = float(epoch_record["eval_loss_total"])
        shutil.copyfile(epoch_path, best_path)
        epoch_record["is_best"] = True
    else:
        epoch_record["is_best"] = False

    print0(
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
    print0(f"Saved epoch checkpoint: {epoch_path}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.disable_setup_conditioned or args.disable_variable_channel_frontend:
        raise ValueError("Setup-gap pretraining requires the setup-conditioned variable-channel frontend.")
    device, distributed = setup_runtime(args)
    args.no_progress = args.no_progress or not get_is_master()

    try:
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = True

        datasets = load_datasets(args)
        output_dir = broadcast_path(make_output_dir(args, datasets) if get_is_master() else None)
        metrics_path = output_dir / "metrics.jsonl"
        model_config = build_model_config(args)
        bands = tuple(model_config["bands"])
        input_dims = max(ds.train_data.shape[-1] for ds in datasets)

        print0(f"Output directory: {output_dir}", flush=True)
        print0(f"Model input_dims placeholder: {input_dims}", flush=True)
        print0(
            "Dataset schedule: "
            + ", ".join(f"{ds.label} train={ds.train_data.shape} eval={ds.eval_data.shape}" for ds in datasets),
            flush=True,
        )

        student_model = TSEncoderThreeBranchHypRelationCGeom(
            input_dims=input_dims,
            **model_config,
        ).to(device)
        teacher = copy.deepcopy(student_model).to(device)
        teacher.requires_grad_(False)
        teacher.eval()

        dataset_keys = [ds.key for ds in datasets]
        student_projector_model = MainProjector(args.repr_dims * 2, args.proto_dim).to(device)
        teacher_projector = copy.deepcopy(student_projector_model).to(device)
        teacher_projector.requires_grad_(False)
        teacher_projector.eval()

        student_proto_heads_model = MultiDatasetPrototypeHeads(dataset_keys, args.proto_dim, args.num_prototypes).to(device)
        teacher_proto_heads = copy.deepcopy(student_proto_heads_model).to(device)
        teacher_proto_heads.requires_grad_(False)
        teacher_proto_heads.eval()

        if distributed:
            student = DDP(
                student_model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
            student_projector = DDP(
                student_projector_model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )
            student_proto_heads = DDP(
                student_proto_heads_model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
        else:
            student = student_model
            student_projector = student_projector_model
            student_proto_heads = student_proto_heads_model

        proto_centers = {
            dataset_key: torch.zeros(args.num_prototypes, device=device, dtype=torch.float32)
            for dataset_key in dataset_keys
        }

        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(student_projector.parameters()) + list(student_proto_heads.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        eval_batch_size = args.eval_batch_size or args.batch_size
        pin_memory = device.type == "cuda"
        train_loaders, eval_loaders, train_samplers = make_loaders(
            datasets,
            args.batch_size,
            eval_batch_size,
            args.num_workers,
            pin_memory,
            distributed=distributed,
        )
        loader_lengths = np.asarray([len(loader) for loader in train_loaders], dtype=np.int64)
        epoch_steps = int(args.steps_per_epoch) if args.steps_per_epoch is not None else int(loader_lengths.sum())
        total_updates = math.ceil(epoch_steps / max(1, args.grad_accum_steps)) * max(1, args.epochs)
        scheduler = build_scheduler(optimizer, total_steps=total_updates, warmup_ratio=args.warmup_ratio)

        amp_enabled = bool(args.use_amp and device.type == "cuda")
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

        if get_is_master():
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
            student_projector.train()
            teacher_projector.eval()
            student_proto_heads.train()
            teacher_proto_heads.eval()
            for sampler in train_samplers:
                if sampler is not None:
                    sampler.set_epoch(epoch)
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

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    loss, metrics, teacher_logits = compute_step_loss(
                        student,
                        teacher,
                        student_projector,
                        teacher_projector,
                        student_proto_heads,
                        teacher_proto_heads,
                        proto_centers,
                        datasets[dataset_idx].key,
                        x,
                        channel_ids,
                        args,
                        bands,
                        progress,
                    )
                    loss_for_backward = loss / max(1, args.grad_accum_steps)

                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                n_micro_steps += 1
                dataset_step_counts[datasets[dataset_idx].key] += 1
                accumulate_metrics(metric_sums, metrics)
                update_proto_center(
                    proto_centers[datasets[dataset_idx].key],
                    teacher_logits,
                    momentum=args.proto_center_momentum,
                )

                should_step = n_micro_steps % max(1, args.grad_accum_steps) == 0
                if should_step:
                    optimizer_update(
                        student,
                        teacher,
                        student_projector,
                        teacher_projector,
                        student_proto_heads,
                        teacher_proto_heads,
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
                            "proto": f"{to_float(metrics['loss_proto_main']):.5f}",
                            "bal": f"{to_float(metrics['loss_proto_balance']):.4f}",
                            "rel": f"{to_float(metrics['loss_rel_cons']):.5f}",
                            "ent": f"{to_float(metrics['teacher_entropy']):.3f}",
                            "active": f"{to_float(metrics['active_proto_count']):.1f}",
                            "ppl": f"{to_float(metrics['proto_usage_perplexity']):.1f}",
                        }
                    )

            if n_micro_steps % max(1, args.grad_accum_steps) != 0:
                optimizer_update(
                    student,
                    teacher,
                    student_projector,
                    teacher_projector,
                    student_proto_heads,
                    teacher_proto_heads,
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
                student_projector,
                teacher_projector,
                student_proto_heads,
                teacher_proto_heads,
                proto_centers,
                eval_loaders,
                datasets,
                args,
                device,
                amp_enabled,
                amp_dtype,
                bands,
                progress=float(global_update_steps) / float(max(1, total_updates)),
            )
            eval_time_sec = time.time() - eval_start

            metric_sums, n_micro_steps = reduce_metric_sums(metric_sums, n_micro_steps, device)
            dataset_step_counts = reduce_count_dict(dataset_step_counts, device)

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
                "world_size": int(get_world_size()),
                "per_device_batch_size": int(args.batch_size),
                "global_batch_size": int(args.batch_size * max(1, args.grad_accum_steps) * get_world_size()),
                "distributed": bool(distributed),
            }
            epoch_record.update(average_metrics(metric_sums, n_micro_steps))
            for key, value in dataset_step_counts.items():
                epoch_record[f"train_steps_{key}"] = int(value)
            for record in per_dataset_eval:
                ds_key = record.pop("dataset")
                for key, value in record.items():
                    epoch_record[f"{ds_key}_{key}"] = float(value)

            save_checkpoint(student, output_dir, epoch_record, best_eval_loss)
            if get_is_master():
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(epoch_record, sort_keys=True) + "\n")
                train_history.append(epoch_record)
                loss_log.append(epoch_record["loss_total"])

        if get_is_master():
            last_path = output_dir / "relation_cgeom_encoder_last.pt"
            best_path = output_dir / "relation_cgeom_encoder_best.pt"
            if last_path.exists() and not best_path.exists():
                shutil.copyfile(last_path, best_path)

            with open(output_dir / "train_history.pkl", "wb") as f:
                pickle.dump(train_history, f)
            with open(output_dir / "loss_log.pkl", "wb") as f:
                pickle.dump(loss_log, f)

            print0(f"Saved last encoder checkpoint: {last_path}", flush=True)
            print0(f"Saved best encoder checkpoint: {best_path}", flush=True)
    finally:
        if is_distributed():
            clean_torch_distributed(get_local_rank())


if __name__ == "__main__":
    main()
