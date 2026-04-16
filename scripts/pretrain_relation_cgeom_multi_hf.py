#!/usr/bin/env python3
"""Joint pretraining for relation_cgeom on multiple HF EEG datasets.

The model frontend is variable-channel, but each batch still needs one
consistent setup. This script keeps one DataLoader per dataset and shuffles the
dataset-level batch schedule every epoch, so a batch never mixes datasets while
the epoch can cover multiple datasets.
"""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.relation_cgeom.ts2vec_three_branch_hyp_relation_cgeom import (  # noqa: E402
    TS2VecThreeBranchHypRelationCGeom,
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
    parser = argparse.ArgumentParser(description="Joint pretrain relation_cgeom on multiple HF EEG datasets.")
    parser.add_argument(
        "--dataset-configs",
        default="motor_mv_img:pretrain_bci,cho2017:pretrain_bci",
        help="Comma-separated dataset:config entries, e.g. motor_mv_img:pretrain_bci,cho2017:pretrain_bci.",
    )
    parser.add_argument("--data-root", default="assets/data/pretrain")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.001, help="Convert saved uV data to model input scale.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional train subset size per dataset.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional eval subset size per dataset.")
    parser.add_argument(
        "--array-cache-dir",
        default="assets/data/pretrain/array_cache",
        help="Decoded .npy cache directory. Reusing it avoids slow Arrow reads on later runs.",
    )
    parser.add_argument("--overwrite-array-cache", action="store_true")
    parser.add_argument("--mmap-array-cache", action="store_true", help="Memory-map cached arrays instead of loading into RAM.")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--channel-subsample-ratio",
        type=float,
        default=1.0,
        help="Randomly keep this ratio of channels per train batch. 1.0 disables channel subsampling.",
    )
    parser.add_argument(
        "--min-channel-subsample",
        type=int,
        default=1,
        help="Minimum number of channels to keep when channel subsampling is enabled.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=1.0,
        help="Used only when steps-per-epoch is set. 1.0=size-proportional, 0.0=uniform datasets.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--repr-dims", type=int, default=320)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-train-length", type=int, default=None)
    parser.add_argument("--temporal-unit", type=int, default=0)
    parser.add_argument("--bands", default="8-13,13-20,20-30")
    parser.add_argument("--share-band-encoder", action="store_true")
    parser.add_argument("--disable-raw-branch", action="store_true")
    parser.add_argument("--disable-hyperbolic-branch", action="store_true")
    parser.add_argument("--band-fusion-type", default="concat_linear", choices=["concat_linear", "gated_sum"])
    parser.add_argument("--hyperbolic-depth", type=int, default=1)
    parser.add_argument("--hyperbolic-curvature", type=float, default=1.0)
    parser.add_argument("--fixed-curvature", action="store_true")
    parser.add_argument("--use-setup-conditioned", action="store_true")
    parser.add_argument("--setup-meta-dim", type=int, default=128)
    parser.add_argument("--setup-ctx-dim", type=int, default=128)
    parser.add_argument("--setup-dim", type=int, default=128)
    parser.add_argument("--setup-channel-vocab-size", type=int, default=128)
    parser.add_argument("--setup-condition-scale", type=float, default=0.1)
    parser.add_argument("--use-variable-channel-frontend", action="store_true")
    parser.add_argument("--per-channel-stem-depth", type=int, default=2)
    parser.add_argument("--channel-attn-heads", type=int, default=4)
    parser.add_argument("--tf-align-weight", type=float, default=0.3)
    parser.add_argument("--tf-align-type", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--raw-mask-weight", type=float, default=0.0)
    parser.add_argument("--raw-mask-type", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--use-amp", action="store_true", help="Use CUDA autocast during pretraining.")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "output_dims": args.repr_dims,
        "hidden_dims": args.hidden_dims,
        "depth": args.depth,
        "max_train_length": args.max_train_length,
        "temporal_unit": args.temporal_unit,
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
        "use_setup_conditioned": args.use_setup_conditioned,
        "setup_meta_dim": args.setup_meta_dim,
        "setup_ctx_dim": args.setup_ctx_dim,
        "setup_dim": args.setup_dim,
        "setup_channel_vocab_size": args.setup_channel_vocab_size,
        "setup_condition_scale": args.setup_condition_scale,
        "use_variable_channel_frontend": args.use_variable_channel_frontend,
        "per_channel_stem_depth": args.per_channel_stem_depth,
        "channel_attn_heads": args.channel_attn_heads,
        "tf_align_weight": args.tf_align_weight,
        "tf_align_type": args.tf_align_type,
        "raw_mask_weight": args.raw_mask_weight,
        "raw_mask_type": args.raw_mask_type,
        "show_progress": not args.no_progress,
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
        / f"multi_{dataset_part}"
        / f"local_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_train_loaders(
    datasets: list[PretrainDataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> list[DataLoader]:
    loaders = []
    for ds in datasets:
        tensor_dataset = TensorDataset(
            torch.from_numpy(ds.train_data).to(torch.float),
            torch.from_numpy(ds.train_channel_ids).to(torch.long),
        )
        if len(tensor_dataset) < batch_size:
            raise RuntimeError(f"{ds.label} has fewer train samples than batch_size={batch_size}.")
        loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders.append(loader)
        print(f"Train loader {ds.label}: batches_per_epoch={len(loader)}", flush=True)
    return loaders


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


def to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def maybe_subsample_channels(
    x: torch.Tensor,
    channel_ids: torch.Tensor,
    ratio: float,
    min_channels: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Randomly keep a batch-shared channel subset.

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

    # Keep original montage order after random selection. One channel subset is
    # shared by the whole batch so tensor shapes stay consistent.
    selected = np.sort(rng.choice(n_channels, size=n_keep, replace=False))
    selected = torch.as_tensor(selected, dtype=torch.long)
    return x.index_select(-1, selected), channel_ids.index_select(-1, selected), n_keep


def save_checkpoint(model, output_dir: Path, epoch_record: dict, best_eval_loss: dict) -> None:
    epoch = int(epoch_record["epoch"])
    epoch_path = output_dir / f"relation_cgeom_encoder_epoch_{epoch:03d}.pt"
    last_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"

    model.save(str(epoch_path))
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
    if args.use_variable_channel_frontend and not args.use_setup_conditioned:
        raise ValueError("--use-variable-channel-frontend requires --use-setup-conditioned.")

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

    input_dims = max(ds.train_data.shape[-1] for ds in datasets)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Model input_dims placeholder: {input_dims}", flush=True)
    print(
        "Dataset schedule: "
        + ", ".join(f"{ds.label} train={ds.train_data.shape} eval={ds.eval_data.shape}" for ds in datasets),
        flush=True,
    )

    device = torch.device(args.device)
    model = TS2VecThreeBranchHypRelationCGeom(
        input_dims=input_dims,
        device=device,
        distributed=False,
        **model_config,
    )
    optimizer = torch.optim.AdamW(model._train_net.parameters(), lr=args.lr)
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    pin_memory = device.type == "cuda"
    loaders = make_train_loaders(datasets, args.batch_size, args.num_workers, pin_memory)
    loader_lengths = np.asarray([len(loader) for loader in loaders], dtype=np.int64)
    iterators = [iter(loader) for loader in loaders]

    loss_log = []
    best_eval_loss = {"value": float("inf")}
    eval_batch_size = args.eval_batch_size or args.batch_size

    for _ in range(args.epochs):
        epoch = model.n_epochs
        epoch_start = time.time()
        model._train_net.train()

        schedule = build_epoch_schedule(loader_lengths, rng, args.steps_per_epoch, args.sampling_alpha)
        cum_loss = 0.0
        n_epoch_iters = 0
        n_optimizer_steps = 0
        epoch_term_sums = {}
        dataset_step_counts = {ds.key: 0 for ds in datasets}
        optimizer.zero_grad(set_to_none=True)

        iterator = schedule
        if not args.no_progress:
            iterator = tqdm(schedule, desc=f"Pretrain epoch {epoch}", leave=True, dynamic_ncols=True)

        for dataset_idx in iterator:
            dataset_idx = int(dataset_idx)
            batch = next_batch(loaders, iterators, dataset_idx)
            x, channel_ids = batch
            if args.max_train_length is not None and x.size(1) > args.max_train_length:
                window_offset = rng.integers(x.size(1) - args.max_train_length + 1)
                x = x[:, window_offset: window_offset + args.max_train_length]
            x, channel_ids, n_channels_kept = maybe_subsample_channels(
                x,
                channel_ids,
                ratio=args.channel_subsample_ratio,
                min_channels=args.min_channel_subsample,
                rng=rng,
            )

            x = x.to(device, non_blocking=True)
            channel_ids = channel_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                loss, loss_terms = model._compute_pair_loss(x, channel_ids=channel_ids)

            loss_for_backward = loss / max(1, args.grad_accum_steps)
            should_step = (n_epoch_iters + 1) % max(1, args.grad_accum_steps) == 0

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
                if should_step:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model._train_net.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    model.net.update_parameters(model._model)
                    n_optimizer_steps += 1
            else:
                loss_for_backward.backward()
                if should_step:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model._train_net.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    model.net.update_parameters(model._model)
                    n_optimizer_steps += 1

            loss_value = float(loss.detach().cpu().item())
            cum_loss += loss_value
            n_epoch_iters += 1
            model.n_iters += 1
            dataset_step_counts[datasets[dataset_idx].key] += 1
            for key, value in loss_terms.items():
                epoch_term_sums[key] = epoch_term_sums.get(key, 0.0) + to_float(value)

            if not args.no_progress:
                postfix = {
                    "dataset": datasets[dataset_idx].dataset,
                    "chs": n_channels_kept,
                    "loss": f"{loss_value:.6f}",
                }
                if "loss_cgeom" in loss_terms:
                    postfix["cgeom"] = f"{to_float(loss_terms['loss_cgeom']):.6f}"
                if "loss_tf_align" in loss_terms:
                    postfix["tf_align"] = f"{to_float(loss_terms['loss_tf_align']):.6f}"
                iterator.set_postfix(postfix)

        if n_epoch_iters % max(1, args.grad_accum_steps) != 0:
            if scaler.is_enabled():
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model._train_net.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model._train_net.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.net.update_parameters(model._model)
            n_optimizer_steps += 1

        if n_epoch_iters < 1:
            raise RuntimeError("No train batches were processed.")

        train_time_sec = time.time() - epoch_start
        eval_start = time.time()
        per_dataset_eval = []
        for ds in datasets:
            record = model.evaluate_loss(
                ds.eval_data,
                eval_channel_ids=ds.eval_channel_ids,
                batch_size=eval_batch_size,
                verbose=not args.no_progress,
                desc=f"Pretrain eval {ds.dataset} epoch {epoch}",
            )
            per_dataset_eval.append(record)

        eval_time_sec = time.time() - eval_start
        epoch_record = {
            "epoch": epoch,
            "loss_total": float(cum_loss / n_epoch_iters),
            "epoch_time_sec": float(train_time_sec),
            "eval_time_sec": float(eval_time_sec),
            "total_epoch_time_sec": float(train_time_sec + eval_time_sec),
            "eval_loss_total": float(np.mean([rec["eval_loss_total"] for rec in per_dataset_eval])),
            "eval_loss_weighted": float(
                np.average(
                    [rec["eval_loss_total"] for rec in per_dataset_eval],
                    weights=[len(ds.eval_data) for ds in datasets],
                )
            ),
            "optimizer_steps": int(n_optimizer_steps),
            "grad_accum_steps": int(args.grad_accum_steps),
            "channel_subsample_ratio": float(args.channel_subsample_ratio),
            "min_channel_subsample": int(args.min_channel_subsample),
        }
        for key, value in epoch_term_sums.items():
            epoch_record[key] = float(value / n_epoch_iters)
        for key, value in dataset_step_counts.items():
            epoch_record[f"train_steps_{key}"] = int(value)
        for ds, record in zip(datasets, per_dataset_eval):
            for key, value in record.items():
                clean_key = key.removeprefix("eval_")
                epoch_record[f"eval_{ds.key}_{clean_key}"] = float(value)

        save_checkpoint(model, output_dir, epoch_record, best_eval_loss)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_record, sort_keys=True) + "\n")

        model.train_history.append(epoch_record)
        loss_log.append(epoch_record["loss_total"])
        model.n_epochs += 1

    last_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"
    if last_path.exists() and not best_path.exists():
        shutil.copyfile(last_path, best_path)

    with open(output_dir / "train_history.pkl", "wb") as f:
        pickle.dump(model.train_history, f)
    with open(output_dir / "loss_log.pkl", "wb") as f:
        pickle.dump(loss_log, f)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    print(f"Saved last encoder checkpoint: {last_path}", flush=True)
    print(f"Saved best encoder checkpoint: {best_path}", flush=True)
    print(f"Use for downstream with: model.pretrained_path={best_path}", flush=True)


if __name__ == "__main__":
    main()
