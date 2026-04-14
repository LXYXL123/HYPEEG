#!/usr/bin/env python3
"""Summarize benchmark logs by selecting the eval-best epoch per run."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, Optional


METRIC_LINE_RE = re.compile(
    r"(?P<dataset>[^/\s]+)/(?P<split>eval|test)\s+epoch:\s+(?P<epoch>\d+),\s+(?P<metrics>.*)$"
)
METRIC_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<value>[-+0-9.eE]+)")


def parse_seed_list(seed_arg: Optional[str]) -> Optional[list[int]]:
    if not seed_arg:
        return None
    return [int(item.strip()) for item in seed_arg.split(",") if item.strip()]


def parse_metric_blob(blob: str) -> Dict[str, float]:
    metrics = {}
    for match in METRIC_RE.finditer(blob):
        metrics[match.group("key")] = float(match.group("value"))
    return metrics


def parse_log(log_path: Path) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = METRIC_LINE_RE.search(line)
            if not match:
                continue
            dataset = match.group("dataset")
            split = match.group("split")
            epoch = int(match.group("epoch"))
            metrics = parse_metric_blob(match.group("metrics"))

            results.setdefault(dataset, {}).setdefault(split, {})[epoch] = metrics
    return results


def find_logs(log_roots: Iterable[str], latest: Optional[int]) -> list[Path]:
    logs: list[Path] = []
    for root_str in log_roots:
        root = Path(root_str)
        if root.is_file():
            logs.append(root)
        elif root.exists():
            found = sorted(root.rglob("*_trainer.log"), key=lambda p: str(p.parent))
            if latest is not None:
                found = found[-latest:]
            logs.extend(found)
        else:
            print(f"[WARN] log root not found: {root}")
    return sorted(logs, key=lambda p: str(p.parent))


def pick_best_epoch(
    parsed: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    dataset: Optional[str],
    monitor_metric: str,
    monitor_mode: str,
    target_split: str,
):
    rows = []
    dataset_names = [dataset] if dataset else sorted(parsed.keys())
    for ds_name in dataset_names:
        ds_result = parsed.get(ds_name, {})
        eval_result = ds_result.get("eval", {})
        target_result = ds_result.get(target_split, {})
        candidates = [
            (epoch, metrics)
            for epoch, metrics in eval_result.items()
            if monitor_metric in metrics and epoch in target_result
        ]
        if not candidates:
            continue

        reverse = monitor_mode == "max"
        best_epoch, best_eval_metrics = sorted(
            candidates,
            key=lambda item: (item[1][monitor_metric], -item[0] if reverse else item[0]),
            reverse=reverse,
        )[0]
        rows.append((ds_name, best_epoch, best_eval_metrics, target_result[best_epoch]))
    return rows


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def print_table(rows: list[dict], metrics: list[str], monitor_metric: str):
    if not rows:
        print("No valid runs found.")
        return

    headers = ["idx", "seed", "dataset", "best_epoch", f"eval_{monitor_metric}"]
    headers += [f"test_{metric}" for metric in metrics]
    headers += ["log"]
    print("\t".join(headers))

    for idx, row in enumerate(rows, start=1):
        items = [
            str(idx),
            str(row.get("seed", "")),
            row["dataset"],
            str(row["best_epoch"]),
            fmt(row["eval"].get(monitor_metric)),
        ]
        items += [fmt(row["test"].get(metric)) for metric in metrics]
        items += [row["log"]]
        print("\t".join(items))


def print_summary(rows: list[dict], metrics: list[str]):
    if not rows:
        return

    print("\nSummary over runs:")
    for metric in metrics:
        values = [row["test"].get(metric) for row in rows if metric in row["test"]]
        if not values:
            continue
        std = stdev(values) if len(values) > 1 else 0.0
        print(
            f"test_{metric}: "
            f"mean={mean(values):.6f}, std={std:.6f}, "
            f"median={median(values):.6f}, min={min(values):.6f}, max={max(values):.6f}, n={len(values)}"
        )


def write_csv(rows: list[dict], metrics: list[str], monitor_metric: str, csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "dataset", "best_epoch", f"eval_{monitor_metric}"]
    fieldnames += [f"test_{metric}" for metric in metrics]
    fieldnames += ["log"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "seed": row.get("seed", ""),
                "dataset": row["dataset"],
                "best_epoch": row["best_epoch"],
                f"eval_{monitor_metric}": row["eval"].get(monitor_metric),
                "log": row["log"],
            }
            for metric in metrics:
                out[f"test_{metric}"] = row["test"].get(metric)
            writer.writerow(out)
    print(f"\nCSV written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best eval epoch per run and report the corresponding test metrics. "
            "Use this for mean/std over seeds without selecting by test performance."
        )
    )
    parser.add_argument(
        "--log-root",
        nargs="+",
        required=True,
        help="One or more log root directories or trainer.log files.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. bcic_2a or motor_mv_img.")
    parser.add_argument("--monitor-metric", default="balanced_acc")
    parser.add_argument("--monitor-mode", choices=["max", "min"], default="max")
    parser.add_argument("--target-split", choices=["test", "eval"], default="test")
    parser.add_argument(
        "--metrics",
        default="acc,balanced_acc,cohen_kappa,f1",
        help="Comma-separated target split metrics to summarize.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seeds assigned in sorted log order, e.g. 42,43,44,45,46.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Keep only the latest N logs per log root, useful when a root contains older runs.",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    seeds = parse_seed_list(args.seeds)
    logs = find_logs(args.log_root, args.latest)

    rows = []
    for log_path in logs:
        parsed = parse_log(log_path)
        picked = pick_best_epoch(
            parsed=parsed,
            dataset=args.dataset,
            monitor_metric=args.monitor_metric,
            monitor_mode=args.monitor_mode,
            target_split=args.target_split,
        )
        for ds_name, best_epoch, eval_metrics, test_metrics in picked:
            rows.append(
                {
                    "dataset": ds_name,
                    "best_epoch": best_epoch,
                    "eval": eval_metrics,
                    "test": test_metrics,
                    "log": str(log_path),
                }
            )

    if seeds is not None:
        if len(seeds) != len(rows):
            print(f"[WARN] seed count ({len(seeds)}) != run count ({len(rows)}); seed column may be incomplete.")
        for row, seed in zip(rows, seeds):
            row["seed"] = seed

    print_table(rows, metrics, args.monitor_metric)
    print_summary(rows, metrics)
    if args.csv:
        write_csv(rows, metrics, args.monitor_metric, args.csv)


if __name__ == "__main__":
    main()
