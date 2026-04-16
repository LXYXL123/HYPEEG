#!/usr/bin/env python3
"""Pretrain relation_cgeom on preprocessed HuggingFace EEG datasets.

This is a thin bridge script for smoke-testing self-supervised pretraining:
it reads datasets generated under assets/data/pretrain, converts windows from
[C, T] to [T, C], and reuses the existing TS2Vec-style relation_cgeom trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.processor.wrapper import DATASET_SELECTOR  # noqa: E402
from baseline.relation_cgeom.ts2vec_three_branch_hyp_relation_cgeom import (  # noqa: E402
    TS2VecThreeBranchHypRelationCGeom,
)


def parse_bands(spec: str) -> tuple[tuple[float, float], ...]:
    bands = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        low, high = chunk.split("-", 1)
        bands.append((float(low), float(high)))
    if not bands:
        raise ValueError(f"Invalid band specification: {spec}")
    return tuple(bands)


def _safe_cache_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _array_cache_paths(
    cache_dir: str | os.PathLike | None,
    dataset_name: str,
    config_name: str,
    fs: int,
    split: str,
    scale: float,
    max_samples: int | None,
) -> tuple[Path, Path, Path] | None:
    if cache_dir is None:
        return None
    max_part = "all" if max_samples is None else str(max_samples)
    key = "__".join(
        _safe_cache_token(item)
        for item in (dataset_name, config_name, f"fs{int(fs)}", split, f"scale{scale:g}", f"max{max_part}")
    )
    root = Path(cache_dir)
    return root / f"{key}.data.npy", root / f"{key}.chs.npy", root / f"{key}.meta.json"


def _load_array_cache(
    paths: tuple[Path, Path, Path] | None,
    mmap_cache: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    if paths is None:
        return None
    data_path, chs_path, meta_path = paths
    if not (data_path.exists() and chs_path.exists() and meta_path.exists()):
        return None
    mmap_mode = "r" if mmap_cache else None
    data = np.load(data_path, mmap_mode=mmap_mode)
    channel_ids = np.load(chs_path, mmap_mode=mmap_mode)
    print(f"Loaded array cache: {data_path}", flush=True)
    print(f"Loaded channel cache: {chs_path}", flush=True)
    return data, channel_ids


def _save_array_cache(
    paths: tuple[Path, Path, Path] | None,
    data: np.ndarray,
    channel_ids: np.ndarray,
    meta: dict,
) -> None:
    if paths is None:
        return
    data_path, chs_path, meta_path = paths
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_data_path = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp_chs_path = chs_path.with_suffix(chs_path.suffix + ".tmp")
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_data_path, "wb") as f:
        np.save(f, data)
    with open(tmp_chs_path, "wb") as f:
        np.save(f, channel_ids)
    with open(tmp_meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp_data_path, data_path)
    os.replace(tmp_chs_path, chs_path)
    os.replace(tmp_meta_path, meta_path)
    print(f"Saved array cache: {data_path}", flush=True)
    print(f"Saved channel cache: {chs_path}", flush=True)


def load_pretrain_array(
    dataset_name: str,
    config_name: str,
    fs: int,
    data_root: str,
    split: str,
    scale: float,
    max_samples: int | None,
    load_chunk_size: int = 256,
    array_cache_dir: str | os.PathLike | None = None,
    overwrite_array_cache: bool = False,
    mmap_array_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_paths = _array_cache_paths(array_cache_dir, dataset_name, config_name, fs, split, scale, max_samples)
    if not overwrite_array_cache:
        cached = _load_array_cache(cache_paths, mmap_cache=mmap_array_cache)
        if cached is not None:
            return cached

    print(
        f"Opening dataset: {dataset_name}/{config_name}, split={split}, "
        f"fs={fs}, data_root={data_root}",
        flush=True,
    )
    builder_cls = DATASET_SELECTOR[dataset_name]
    builder = builder_cls(config_name=config_name, fs=fs, database_proc_root=data_root)
    print(f"Builder cache_dir: {builder.cache_dir}", flush=True)
    ds = builder.as_dataset(split=split)
    print(f"Dataset opened: split={split}, rows={len(ds)}, columns={ds.column_names}", flush=True)

    n_samples = len(ds) if max_samples is None else min(len(ds), max_samples)
    if n_samples <= 0:
        raise RuntimeError(f"No samples found for {dataset_name}/{config_name} split={split}")

    first = np.asarray(ds[0]["data"], dtype=np.float32)
    if first.ndim != 2:
        raise ValueError(f"Expected data shape [C, T], got {first.shape}")
    first_chs = np.asarray(ds[0]["chs"], dtype=np.int64)

    n_channels, n_timepoints = first.shape
    data = np.empty((n_samples, n_timepoints, n_channels), dtype=np.float32)
    channel_ids = np.empty((n_samples, n_channels), dtype=np.int64)
    # Read Arrow rows in chunks instead of row-by-row. Per-row HF access is
    # correct but very slow for pretraining-scale window counts.
    for start in tqdm(range(0, n_samples, load_chunk_size), desc="Loading windows into RAM"):
        end = min(start + load_chunk_size, n_samples)
        batch = ds[start:end]
        samples = np.asarray(batch["data"], dtype=np.float32)
        sample_chs = np.asarray(batch["chs"], dtype=np.int64)
        if samples.shape != (end - start, n_channels, n_timepoints):
            raise ValueError(
                f"Inconsistent batch data shape at {start}:{end}: "
                f"{samples.shape} vs {(end - start, n_channels, n_timepoints)}"
            )
        if sample_chs.shape != (end - start, n_channels):
            raise ValueError(
                f"Inconsistent batch channel-id shape at {start}:{end}: "
                f"{sample_chs.shape} vs {(end - start, n_channels)}"
            )
        data[start:end] = np.transpose(samples, (0, 2, 1)) * scale
        channel_ids[start:end] = sample_chs

    _save_array_cache(
        cache_paths,
        data,
        channel_ids,
        {
            "dataset": dataset_name,
            "config": config_name,
            "fs": int(fs),
            "split": split,
            "scale": float(scale),
            "max_samples": max_samples,
            "shape": list(data.shape),
            "channel_shape": list(channel_ids.shape),
        },
    )
    return data, channel_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain relation_cgeom on HF EEG windows.")
    parser.add_argument("--dataset", default="motor_mv_img")
    parser.add_argument("--config", default="pretrain_bci")
    parser.add_argument("--data-root", default="assets/data/pretrain")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.001, help="Convert saved uV data to model input scale, default uV->mV.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset size for smoke tests.")
    parser.add_argument("--array-cache-dir", default=None, help="Optional .npy cache directory for decoded HF windows.")
    parser.add_argument("--overwrite-array-cache", action="store_true")
    parser.add_argument("--mmap-array-cache", action="store_true", help="Memory-map cached .npy arrays instead of loading into RAM.")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars during training.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root
    if not data_root.startswith("s3://") and not os.path.isabs(data_root):
        data_root = str((PROJECT_ROOT / data_root).resolve())

    train_data, train_channel_ids = load_pretrain_array(
        dataset_name=args.dataset,
        config_name=args.config,
        fs=args.fs,
        data_root=data_root,
        split=args.split,
        scale=args.scale,
        max_samples=args.max_samples,
        array_cache_dir=args.array_cache_dir,
        overwrite_array_cache=args.overwrite_array_cache,
        mmap_array_cache=args.mmap_array_cache,
    )
    eval_data, eval_channel_ids = load_pretrain_array(
        dataset_name=args.dataset,
        config_name=args.config,
        fs=args.fs,
        data_root=data_root,
        split=args.eval_split,
        scale=args.scale,
        max_samples=args.max_samples,
        array_cache_dir=args.array_cache_dir,
        overwrite_array_cache=args.overwrite_array_cache,
        mmap_array_cache=args.mmap_array_cache,
    )

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "assets" / "run" / "pretrain" / "relation_cgeom" / f"{args.dataset}_{args.config}" / f"local_{timestamp}"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
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

    print(f"Loaded {args.dataset}/{args.config} split={args.split}: {train_data.shape}")
    print(f"Loaded {args.dataset}/{args.config} split={args.eval_split}: {eval_data.shape}")
    print(f"Output directory: {output_dir}")
    metrics_path = output_dir / "metrics.jsonl"
    best_eval_loss = {"value": float("inf")}

    def save_epoch_checkpoint(model, epoch_record):
        eval_start = datetime.now()
        eval_record = model.evaluate_loss(
            eval_data,
            eval_channel_ids=eval_channel_ids,
            batch_size=args.batch_size,
            verbose=not args.no_progress,
            desc=f"Pretrain eval epoch {int(epoch_record['epoch'])}",
        )
        eval_time_sec = (datetime.now() - eval_start).total_seconds()
        epoch_record.update(eval_record)
        epoch_record["eval_time_sec"] = float(eval_time_sec)
        epoch_record["total_epoch_time_sec"] = float(epoch_record["epoch_time_sec"] + eval_time_sec)

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

        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_record, sort_keys=True) + "\n")

        print(
            "Epoch #{epoch}: "
            "train_loss={train_loss:.6f} eval_loss={eval_loss:.6f} "
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

    model = TS2VecThreeBranchHypRelationCGeom(
        input_dims=train_data.shape[-1],
        device=torch.device(args.device),
        distributed=False,
        after_epoch_callback=save_epoch_checkpoint,
        **model_config,
    )
    # Epoch summaries are printed by the callback after validation, so they
    # include train/eval/total time in one line.
    loss_log = model.fit(
        train_data,
        train_channel_ids=train_channel_ids,
        n_epochs=args.epochs,
        verbose=False,
    )

    encoder_path = output_dir / "relation_cgeom_encoder_last.pt"
    best_path = output_dir / "relation_cgeom_encoder_best.pt"
    if encoder_path.exists() and not best_path.exists():
        shutil.copyfile(encoder_path, best_path)
    with open(output_dir / "train_history.pkl", "wb") as f:
        pickle.dump(model.train_history, f)
    with open(output_dir / "loss_log.pkl", "wb") as f:
        pickle.dump(loss_log, f)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    print(f"Saved last encoder checkpoint: {encoder_path}")
    print(f"Saved best encoder checkpoint: {best_path}")
    print(f"Use for downstream with: model.pretrained_path={best_path}")


if __name__ == "__main__":
    main()
