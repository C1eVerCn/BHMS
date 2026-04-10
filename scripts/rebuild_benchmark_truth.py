#!/usr/bin/env python3
"""Rebuild benchmark truth-source assets, comparison summaries, and paper evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.benchmark_truth import CORE_WITHIN_SOURCE_SOURCES, rebuild_benchmark_truth_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild BHMS benchmark truth-source assets")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--sources", nargs="+", default=list(CORE_WITHIN_SOURCE_SOURCES))
    parser.add_argument("--paper-json", default="Doc/BHMS论文证据包.json")
    parser.add_argument("--paper-markdown", default="Doc/BHMS论文证据包.md")
    args = parser.parse_args()

    report = rebuild_benchmark_truth_assets(
        model_dir=args.model_dir,
        sources=[item.lower() for item in args.sources],
        paper_json_path=args.paper_json,
        paper_markdown_path=args.paper_markdown,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
