#!/usr/bin/env python3
"""
Unified Baseline Model Training Script

This script provides a unified interface for training different baseline models
(EEGPT, LABRAM, etc.) using the abstract class architecture.

Usage:
    python baseline_main.py conf_file=assets/conf/eegpt/eegpt_unified.yaml model_type=eegpt
    python baseline_main.py conf_file=assets/conf/labram/labram_config.yaml model_type=labram

The config file should contain all necessary parameters for training.
The model_type parameter specifies which model architecture to use.
"""

import sys
import os
import subprocess

from omegaconf import OmegaConf
import torch

from baseline.abstract.factory import ModelRegistry
from common.path import get_conf_file_path
from common.utils import setup_yaml


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _get_cli_int(cli_args, key: str, default: int) -> int:
    value = cli_args.get(key, default)
    if value is None:
        return default
    return int(value)


def _strip_launcher_args(raw_args: list[str]) -> list[str]:
    launcher_keys = {
        "multi_gpu",
        "nproc_per_node",
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
    }
    keep_args = []
    for arg in raw_args:
        if "=" not in arg:
            keep_args.append(arg)
            continue
        key = arg.split("=", 1)[0]
        if key in launcher_keys:
            continue
        keep_args.append(arg)
    return keep_args


def _maybe_launch_torchrun(cli_args, raw_args: list[str]) -> None:
    # Already in torchrun child process.
    if os.environ.get("LOCAL_RANK") is not None:
        return

    multi_gpu = _parse_bool(cli_args.get("multi_gpu", os.environ.get("MULTI_GPU", "false")))
    requested_nproc = int(cli_args.get("nproc_per_node", os.environ.get("NPROC_PER_NODE", 0) or 0))

    if requested_nproc <= 1 and not multi_gpu:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Requested multi-GPU run but CUDA is not available.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"Requested multi-GPU run but only detected {gpu_count} GPU.")

    if requested_nproc <= 1:
        requested_nproc = gpu_count

    if requested_nproc > gpu_count:
        raise ValueError(
            f"Requested nproc_per_node={requested_nproc}, but only {gpu_count} GPU(s) are visible."
        )

    nnodes = _get_cli_int(cli_args, "nnodes", int(os.environ.get("NNODES", 1)))
    node_rank = _get_cli_int(cli_args, "node_rank", int(os.environ.get("NODE_RANK", 0)))
    master_addr = str(cli_args.get("master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1")))
    master_port = _get_cli_int(cli_args, "master_port", int(os.environ.get("MASTER_PORT", 29500)))

    launch_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(requested_nproc),
        "--nnodes",
        str(nnodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        sys.argv[0],
        *_strip_launcher_args(raw_args),
    ]

    print(
        f"[baseline_main] Launching multi-GPU run via torchrun: "
        f"nproc_per_node={requested_nproc}, nnodes={nnodes}, node_rank={node_rank}"
    )
    completed = subprocess.run(launch_cmd, check=False)
    sys.exit(completed.returncode)


def main():
    """Main training function that can handle any registered baseline model."""
    setup_yaml()
    
    # Parse CLI arguments
    raw_args = sys.argv[1:]
    cli_args = OmegaConf.from_cli(raw_args)

    _maybe_launch_torchrun(cli_args, raw_args)

    if 'conf_file' not in cli_args:
        raise ValueError("Please provide a config file: conf_file=path/to/config.yaml")
    
    # Get model type from CLI args or config
    model_type: str = cli_args.get('model_type', None)

    # Load config file
    conf_file_path = get_conf_file_path(cli_args.conf_file)
    file_cfg = OmegaConf.load(conf_file_path)

    if model_type is None:
        model_type = file_cfg.get('model_type')

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    # Create base config for the specified model type
    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())
    
    # Merge configurations: code defaults < file config < CLI args
    merged_config = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    
    # Ensure model_type is set correctly
    merged_config.model_type = model_type
    
    # Convert to config object
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)
    
    # Validate configuration
    if not cfg.validate_config():
        raise ValueError(f"Invalid configuration for model type: {model_type}")

    # Create and run trainer
    trainer = ModelRegistry.create_trainer(cfg)
    trainer.run()


def list_available_models():
    """List all available model types."""
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main() 