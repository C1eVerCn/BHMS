"""Experiment pipeline and case export integration tests."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.database import DatabaseManager  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402
from backend.app.services.insight_service import InsightService  # noqa: E402
from backend.app.services.repository import BHMSRepository  # noqa: E402
from ml.data.schema import finalize_cycle_frame  # noqa: E402
from ml.training.experiment_constants import ABLATION_VARIANTS  # noqa: E402
from ml.training.experiment_runner import create_ablation_summary, create_multi_seed_summary, generate_source_plot_bundle  # noqa: E402
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment  # noqa: E402
from scripts.run_ablation_study import build_variant_summary  # noqa: E402


def _make_settings(tmp_path: Path):
    base = get_settings()
    data_dir = tmp_path / "data"
    return replace(
        base,
        project_root=tmp_path,
        data_dir=data_dir,
        raw_nasa_dir=data_dir / "raw" / "nasa",
        raw_calce_dir=data_dir / "raw" / "calce",
        raw_kaggle_dir=data_dir / "raw" / "kaggle",
        processed_dir=data_dir / "processed",
        knowledge_path=PROJECT_ROOT / "data" / "knowledge" / "battery_fault_knowledge.json",
        model_dir=data_dir / "models",
        upload_dir=data_dir / "uploads",
        demo_upload_dir=data_dir / "demo_uploads",
        database_path=data_dir / "bhms.db",
        graph_backend="memory",
    )


def _build_cycle_frame(source: str, battery_prefix: str, battery_count: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_index in range(battery_count):
        for cycle in range(1, 22):
            rows.append(
                {
                    "source_battery_id": f"{battery_prefix}_{battery_index:03d}",
                    "cycle_number": cycle,
                    "voltage_mean": 3.72 - cycle * 0.002,
                    "voltage_std": 0.02 + battery_index * 0.001,
                    "voltage_min": 3.4 - cycle * 0.001,
                    "voltage_max": 4.1 - cycle * 0.001,
                    "current_mean": -1.75,
                    "current_std": 0.03,
                    "temperature_mean": 24.0 + cycle * 0.08,
                    "temperature_std": 0.12,
                    "temperature_rise_rate": 0.45 + cycle * 0.02,
                    "capacity": 2.1 - cycle * 0.03 - battery_index * 0.01,
                }
            )
    return finalize_cycle_frame(pd.DataFrame(rows), source=source, dataset_name=f"{source}_experiment_test", eol_capacity_ratio=0.8)


def _write_config(tmp_path: Path, source: str, model_type: str, csv_path: Path) -> Path:
    config = {
        "data": {
            "csv_path": str(csv_path),
            "sources": [source],
            "feature_columns": [
                "voltage_mean",
                "voltage_std",
                "voltage_min",
                "voltage_max",
                "current_mean",
                "current_std",
                "temperature_mean",
                "temperature_std",
                "temperature_rise_rate",
                "capacity",
                "cycle_number",
            ],
            "target_config": {
                "observation_ratios": [0.2, 0.3, 0.4],
                "default_observation_ratio": 0.3,
                "encoder_len": 24,
                "future_len": 32,
                "target_column": "capacity_ratio",
            },
        },
        "model": {
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.05,
            "use_domain_embeddings": True,
            "use_trajectory_head": True,
        }
        if model_type == "bilstm"
        else {
            "d_model": 16,
            "xlstm_layers": 1,
            "transformer_layers": 1,
            "fusion_dim": 16,
            "dropout": 0.05,
            "use_domain_embeddings": True,
            "use_trajectory_head": True,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 2,
            "model_version": f"{source}-{model_type}-test",
            "checkpoint_dir": str(tmp_path / "data" / "models"),
            "log_dir": str(tmp_path / "data" / "models" / "logs"),
        },
    }
    config_path = tmp_path / f"{source}_{model_type}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_multi_seed_summary_contains_mean_std_runs_and_plots(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("calce", "CALCE_MULTI")
    csv_path = settings.processed_dir / "calce" / "calce_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    config_path = _write_config(tmp_path, "calce", "bilstm", csv_path)
    summaries = []
    for seed in (7, 21, 42):
        summaries.append(
            run_lifecycle_experiment(
                "calce",
                "bilstm",
                config_path=config_path,
                training_overrides={"seed": seed, "model_version": f"calce-bilstm-seed-{seed}"},
                artifact_subdir=f"runs/seed-{seed}",
                suite_kind="multi_seed",
                variant_key="bilstm",
                persist_training_run=False,
            )
        )

    summary = create_multi_seed_summary(
        "calce",
        "bilstm",
        seeds=[7, 21, 42],
        per_seed_summaries=summaries,
        config_path=config_path,
        model_dir=settings.model_dir,
    )
    source_plots = generate_source_plot_bundle("calce", model_dir=settings.model_dir, processed_dir=settings.processed_dir)

    assert summary["aggregate_metrics"]["mean"]["rmse"] is not None
    assert summary["aggregate_metrics"]["std"]["rmse"] is not None
    assert len(summary["per_seed_runs"]) == 3
    assert len(summary["plots"]) >= 3
    assert Path(summary["artifact_paths"]["summary"]).exists()
    assert Path(summary["plots"][0]["path"]).exists()
    assert any(item["key"] == "dataset_split" for item in source_plots)


def test_ablation_summary_contains_real_variants_and_delta(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("nasa", "NASA_ABLATE")
    csv_path = settings.processed_dir / "nasa" / "nasa_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    config_path = _write_config(tmp_path, "nasa", "hybrid", csv_path)
    full_runs = []
    for seed in (7, 21, 42):
        full_runs.append(
            run_lifecycle_experiment(
                "nasa",
                "hybrid",
                config_path=config_path,
                training_overrides={"seed": seed, "model_version": f"nasa-hybrid-seed-{seed}"},
                artifact_subdir=f"runs/seed-{seed}",
                suite_kind="multi_seed",
                variant_key="hybrid",
                persist_training_run=False,
            )
        )
    full_summary = create_multi_seed_summary(
        "nasa",
        "hybrid",
        seeds=[7, 21, 42],
        per_seed_summaries=full_runs,
        config_path=config_path,
        model_dir=settings.model_dir,
    )

    variants: list[dict[str, object]] = [
        {
            "key": "full_hybrid",
            "label": "完整 Hybrid",
            "description": "复用主实验的 Hybrid 多随机种子结果。",
            "status": "available",
            "seeds": full_summary["seeds"],
            "per_seed_runs": full_summary["per_seed_runs"],
            "aggregate_metrics": full_summary["aggregate_metrics"],
            "best_checkpoint": full_summary["best_checkpoint"],
            "artifact_paths": full_summary["artifact_paths"],
            "plots": full_summary["plots"],
        }
    ]

    for variant in ABLATION_VARIANTS:
        if variant["key"] == "full_hybrid":
            continue
        per_seed = []
        for seed in (7, 21, 42):
            config_overrides = {"model": dict(variant.get("model_overrides", {}))}
            if variant.get("feature_columns"):
                config_overrides["data"] = {"feature_columns": list(variant["feature_columns"])}
            per_seed.append(
                run_lifecycle_experiment(
                    "nasa",
                    "hybrid",
                    config_path=config_path,
                    config_overrides=config_overrides,
                    training_overrides={"seed": seed, "model_version": f"nasa-{variant['key']}-seed-{seed}"},
                    artifact_subdir=f"ablation/{variant['key']}/seed-{seed}",
                    suite_kind="ablation",
                    variant_key=str(variant["key"]),
                    persist_training_run=False,
                )
            )
        variants.append(build_variant_summary("nasa", variant, per_seed, model_dir=settings.model_dir))

    summary = create_ablation_summary("nasa", variants=variants, model_dir=settings.model_dir)
    feature_config = json.loads((settings.processed_dir / "nasa" / "nasa_feature_config.json").read_text(encoding="utf-8"))

    assert summary["available"] is True
    variant_keys = {item["key"] for item in summary["variants"]}
    assert {"full_hybrid", "no_xlstm", "no_transformer", "no_domain_embedding", "no_trajectory_head"} <= variant_keys
    assert all("aggregate_metrics" in item for item in summary["variants"])
    assert all("delta_vs_full" in item for item in summary["variants"])
    assert Path(summary["artifact_paths"]["summary"]).exists()
    assert "voltage_mean" in feature_config["feature_columns"]
    assert Path(settings.model_dir / "nasa" / "hybrid" / "ablation" / "no_domain_embedding" / "seed-7" / "data_profile" / "no_domain_embedding_feature_config.json").exists()


def test_case_bundle_export_auto_generates_missing_reports_and_directory(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    battery_service = BatteryService(repository=repo, settings=settings)
    insight_service = InsightService(repository=repo, settings=settings)

    frame = _build_cycle_frame("calce", "CALCE_EXPORT", battery_count=4)
    csv_path = tmp_path / "case.csv"
    frame.to_csv(csv_path, index=False)
    summary = battery_service.import_frame(frame, source="calce", dataset_path=csv_path, include_in_training=True)
    battery_service.prepare_training_dataset("calce", seq_len=10, batch_size=4)

    comparison_path = settings.model_dir / "calce" / "comparison_summary.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(
            {
                "source": "calce",
                "models": {
                    "bilstm": {"test_metrics": {"rmse": 3.2, "mae": 2.8, "mape": 12.0, "r2": 0.42}},
                    "hybrid": {"test_metrics": {"rmse": 2.7, "mae": 2.4, "mape": 10.2, "r2": 0.55}},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = insight_service.export_case_bundle(summary["battery_ids"][0], ensure_artifacts=True)

    assert result["generated_artifacts"]["prediction_generated"] is True
    assert result["generated_artifacts"]["diagnosis_generated"] is True
    export_dir = Path(result["export_dir"])
    assert export_dir.exists()
    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "case_bundle.md").exists()
    assert (export_dir / "lifecycle_prediction_report.md").exists()
    assert (export_dir / "mechanism_report.md").exists()
    assert (export_dir / "sample_profile.json").exists()
    assert (export_dir / "dataset_profile.json").exists()
    assert (export_dir / "experiment_context.json").exists()
    assert (export_dir / "charts" / "lifecycle_trajectory.png").exists()
    assert (export_dir / "charts" / "graphrag_evidence.png").exists()
    assert (export_dir / "charts" / "benchmark_summary.png").exists()
    assert result["bundle_snapshot"]["last_export"] is not None
