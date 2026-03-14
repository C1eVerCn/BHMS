#!/usr/bin/env python3
"""运行或汇总单来源 Bi-LSTM 与 Hybrid 对比实验。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

CONFIGS = {
    ("nasa", "bilstm"): "configs/nasa_bilstm.yaml",
    ("nasa", "hybrid"): "configs/nasa_hybrid.yaml",
    ("calce", "bilstm"): "configs/calce_bilstm.yaml",
    ("calce", "hybrid"): "configs/calce_hybrid.yaml",
    ("kaggle", "bilstm"): "configs/kaggle_bilstm.yaml",
    ("kaggle", "hybrid"): "configs/kaggle_hybrid.yaml",
}


def load_or_train(source: str, model_type: str, force: bool) -> dict:
    summary_path = Path("data/models") / source / model_type / f"{model_type}_experiment_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_models.py")),
        "--source",
        source,
        "--model",
        model_type,
        "--config",
        CONFIGS[(source, model_type)],
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    json_blob = "\n".join(output_lines[:-1]) if len(output_lines) > 1 else output_lines[0]
    return json.loads(json_blob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS source comparison experiment")
    parser.add_argument("--source", choices=["nasa", "calce", "kaggle"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = {model_type: load_or_train(args.source, model_type, force=args.force) for model_type in ("bilstm", "hybrid")}
    comparison = {
        "source": args.source,
        "models": {
            model: {
                "best_val_loss": result.get("best_val_loss"),
                "test_metrics": result.get("test_metrics", {}),
                "best_checkpoint": result.get("best_checkpoint"),
                "final_checkpoint": result.get("final_checkpoint"),
            }
            for model, result in results.items()
        },
    }
    output_path = Path("data/models") / args.source / "comparison_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
