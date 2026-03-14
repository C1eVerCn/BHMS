#!/usr/bin/env python3
"""兼容入口：按来源训练 BHMS MVP 模型。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = {
    ("nasa", "bilstm"): "configs/nasa_bilstm.yaml",
    ("nasa", "hybrid"): "configs/nasa_hybrid.yaml",
    ("calce", "bilstm"): "configs/calce_bilstm.yaml",
    ("calce", "hybrid"): "configs/calce_hybrid.yaml",
    ("kaggle", "bilstm"): "configs/kaggle_bilstm.yaml",
    ("kaggle", "hybrid"): "configs/kaggle_hybrid.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BHMS MVP source-specific models")
    parser.add_argument("--source", choices=["nasa", "calce", "kaggle"], default="nasa")
    parser.add_argument("--model", choices=["bilstm", "hybrid"], default="hybrid")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = args.config or DEFAULT_CONFIGS[(args.source, args.model)]
    cmd = [sys.executable, str(Path(__file__).with_name("train_models.py")), "--source", args.source, "--model", args.model, "--config", config]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
