"""Experiment pipeline and case export integration tests."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
import scripts.run_ablation_study as run_ablation_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.database import DatabaseManager  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402
from backend.app.services.insight_service import InsightService  # noqa: E402
from backend.app.services.repository import BHMSRepository  # noqa: E402
from ml.data import create_synthetic_data  # noqa: E402
from ml.models.lifecycle import DomainConditioning, DominanceSafeFusion, LifecycleDecoder, LifecycleLoss  # noqa: E402
from ml.data.schema import finalize_cycle_frame  # noqa: E402
from ml.training.experiment_constants import ABLATION_VARIANTS  # noqa: E402
from ml.training.experiment_runner import create_ablation_summary, create_multi_seed_summary, generate_source_plot_bundle  # noqa: E402
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment  # noqa: E402
from ml.training.lifecycle_trainer import LifecycleTrainer, LifecycleTrainingConfig, build_lifecycle_model  # noqa: E402
from ml.training.lifecycle_transfer_runner import run_transfer_benchmark  # noqa: E402
from scripts.run_ablation_study import build_variant_summary  # noqa: E402
from scripts.run_multi_seed_experiment import reusable_root_summary  # noqa: E402


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


def _write_config(
    tmp_path: Path,
    source: str,
    model_type: str,
    csv_path: Path | list[Path],
    *,
    sources: list[str] | None = None,
) -> Path:
    data_config = {
        "sources": sources or [source],
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
    }
    if isinstance(csv_path, list):
        data_config["csv_paths"] = [str(path) for path in csv_path]
    else:
        data_config["csv_path"] = str(csv_path)
    config = {
        "data": data_config,
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
    assert "history" not in summary["per_seed_runs"][0]
    assert "test_details" not in summary["per_seed_runs"][0]
    assert len(summary["plots"]) >= 3
    assert Path(summary["artifact_paths"]["summary"]).exists()
    assert Path(summary["plots"][0]["path"]).exists()
    assert any(item["key"] == "dataset_split" for item in source_plots)


def test_lifecycle_experiment_supports_resume_and_early_stopping(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("calce", "CALCE_RESUME")
    csv_path = settings.processed_dir / "calce" / "calce_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    config_path = _write_config(tmp_path, "calce", "hybrid", csv_path)
    initial = run_lifecycle_experiment(
        "calce",
        "hybrid",
        config_path=config_path,
        training_overrides={"num_epochs": 1, "model_version": "calce-hybrid-initial"},
        artifact_subdir="resume-check",
        persist_training_run=False,
    )
    resumed = run_lifecycle_experiment(
        "calce",
        "hybrid",
        config_path=config_path,
        training_overrides={
            "num_epochs": 3,
            "resume_from": initial["final_checkpoint"],
            "model_version": "calce-hybrid-resumed",
        },
        artifact_subdir="resume-check",
        persist_training_run=False,
    )
    early_stop = run_lifecycle_experiment(
        "calce",
        "hybrid",
        config_path=config_path,
        training_overrides={
            "num_epochs": 6,
            "early_stopping": True,
            "patience": 1,
            "min_delta": 1e9,
            "model_version": "calce-hybrid-early-stop",
        },
        artifact_subdir="early-stop-check",
        persist_training_run=False,
    )

    assert resumed["epochs_completed"] == 3
    assert resumed["resume_from"] == initial["final_checkpoint"]
    assert resumed["best_epoch"] is not None
    assert resumed["status"] == "completed"
    assert resumed["history_summary"]["epochs_ran"] == 3
    assert Path(resumed["summary_path"]).exists()
    assert Path(resumed["test_details_path"]).exists()

    assert early_stop["stopped_early"] is True
    assert early_stop["epochs_completed"] < 6
    assert early_stop["best_checkpoint"] is not None


def test_lifecycle_experiment_rejects_unknown_config_keys(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("calce", "CALCE_UNKNOWN")
    csv_path = settings.processed_dir / "calce" / "calce_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    config_path = _write_config(tmp_path, "calce", "hybrid", csv_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["data"]["seq_len"] = 24
    payload["model"]["transformer_parallel"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported lifecycle config keys"):
        run_lifecycle_experiment(
            "calce",
            "hybrid",
            config_path=config_path,
            persist_training_run=False,
        )


def test_hybrid_domain_conditioning_zeroes_single_source_vocab():
    torch.manual_seed(0)
    conditioning = DomainConditioning(
        hidden_dim=8,
        source_vocab_size=2,
        chemistry_vocab_size=1,
        protocol_vocab_size=1,
        embedding_dim=4,
        dropout=0.0,
        enabled=True,
        zero_for_degenerate_vocab=True,
    )
    zero_sequence, zero_context = conditioning(
        batch_size=3,
        seq_len=5,
        source_id=torch.tensor([1, 1, 1], dtype=torch.long),
        chemistry_id=torch.tensor([0, 0, 0], dtype=torch.long),
        protocol_id=torch.tensor([0, 0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert torch.allclose(zero_sequence, torch.zeros_like(zero_sequence))
    assert torch.allclose(zero_context, torch.zeros_like(zero_context))

    conditioning_multi = DomainConditioning(
        hidden_dim=8,
        source_vocab_size=3,
        chemistry_vocab_size=1,
        protocol_vocab_size=1,
        embedding_dim=4,
        dropout=0.0,
        enabled=True,
        zero_for_degenerate_vocab=True,
    )
    live_sequence, live_context = conditioning_multi(
        batch_size=3,
        seq_len=5,
        source_id=torch.tensor([1, 2, 1], dtype=torch.long),
        chemistry_id=torch.tensor([0, 0, 0], dtype=torch.long),
        protocol_id=torch.tensor([0, 0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert not torch.allclose(live_sequence, torch.zeros_like(live_sequence))
    assert not torch.allclose(live_context, torch.zeros_like(live_context))


def test_hybrid_dominance_safe_fusion_prefers_fused_branch_and_can_fallback():
    torch.manual_seed(0)
    fusion = DominanceSafeFusion(4, 4, 4, dropout=0.0)
    xlstm_feat = torch.randn(2, 3, 4)
    transformer_feat = torch.randn(2, 3, 4)

    _, details = fusion(xlstm_feat, transformer_feat)
    selector_weights = details["selector_weights"]

    assert torch.all(selector_weights[..., 0] > selector_weights[..., 1])
    assert torch.all(selector_weights[..., 0] > selector_weights[..., 2])

    with torch.no_grad():
        fusion.selector.weight.zero_()
        fusion.selector.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    fused_output, fused_details = fusion(xlstm_feat, transformer_feat)
    assert torch.allclose(fused_output, fused_details["fused_candidate"], atol=1e-6)

    with torch.no_grad():
        fusion.selector.weight.zero_()
        fusion.selector.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
    xlstm_output, xlstm_details = fusion(xlstm_feat, transformer_feat)
    assert torch.allclose(xlstm_output, xlstm_details["xlstm_only_candidate"], atol=1e-6)

    with torch.no_grad():
        fusion.selector.weight.zero_()
        fusion.selector.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))
    transformer_output, transformer_details = fusion(xlstm_feat, transformer_feat)
    assert torch.allclose(transformer_output, transformer_details["transformer_only_candidate"], atol=1e-6)


def test_lifecycle_decoder_uses_dual_trajectory_fusion_and_fallback_only_modes():
    decoder = LifecycleDecoder(
        hidden_dim=4,
        future_len=3,
        decoder_heads=2,
        dropout=0.0,
        trajectory_scale=10.0,
        source_vocab_size=2,
        use_trajectory_head=True,
        use_uncertainty_head=False,
        use_fluctuation_head=False,
        fluctuation_scale=0.0,
        dual_trajectory_fusion=True,
    )
    for parameter in decoder.parameters():
        parameter.data.zero_()
    decoder.trajectory_head.bias.data.fill_(1.0)
    decoder.trajectory_fallback.bias.data.fill_(3.0)

    outputs = decoder(
        torch.zeros(1, 3, 4),
        torch.zeros(1, 4),
        last_capacity_ratio=torch.tensor([1.0]),
        observed_cycle=torch.tensor([0.0]),
    )
    expected_gate = 1.0 - (1.0 - 0.5) ** 2
    expected_fade = expected_gate * F.softplus(torch.tensor(1.0)) + (1.0 - expected_gate) * F.softplus(torch.tensor(3.0))
    expected_trajectory = torch.tensor(
        [[1.0 - expected_fade / 10.0, 1.0 - 2.0 * expected_fade / 10.0, 1.0 - 3.0 * expected_fade / 10.0]]
    ).clamp(min=0.0, max=1.2)

    assert outputs["trajectory_gate"] is not None
    assert torch.allclose(outputs["trajectory_gate"], torch.full((1, 3), expected_gate))
    assert torch.allclose(outputs["trajectory"], expected_trajectory, atol=1e-6)

    fallback_decoder = LifecycleDecoder(
        hidden_dim=4,
        future_len=3,
        decoder_heads=2,
        dropout=0.0,
        trajectory_scale=10.0,
        source_vocab_size=2,
        use_trajectory_head=False,
        use_uncertainty_head=False,
        use_fluctuation_head=False,
        fluctuation_scale=0.0,
        dual_trajectory_fusion=True,
    )
    for parameter in fallback_decoder.parameters():
        parameter.data.zero_()
    fallback_decoder.trajectory_head.bias.data.fill_(9.0)
    fallback_decoder.trajectory_fallback.bias.data.fill_(2.0)

    fallback_outputs = fallback_decoder(
        torch.zeros(1, 3, 4),
        torch.zeros(1, 4),
        last_capacity_ratio=torch.tensor([1.0]),
        observed_cycle=torch.tensor([0.0]),
    )
    expected_fallback = F.softplus(torch.tensor(2.0))
    expected_fallback_trajectory = torch.tensor(
        [[1.0 - expected_fallback / 10.0, 1.0 - 2.0 * expected_fallback / 10.0, 1.0 - 3.0 * expected_fallback / 10.0]]
    ).clamp(min=0.0, max=1.2)

    assert fallback_outputs["trajectory_gate"] is None
    assert torch.allclose(fallback_outputs["trajectory"], expected_fallback_trajectory, atol=1e-6)


def test_lifecycle_decoder_initializes_step_preferred_gate():
    decoder = LifecycleDecoder(
        hidden_dim=4,
        future_len=3,
        decoder_heads=2,
        dropout=0.0,
        trajectory_scale=10.0,
        source_vocab_size=2,
        use_trajectory_head=True,
        use_uncertainty_head=False,
        use_fluctuation_head=False,
        fluctuation_scale=0.0,
        dual_trajectory_fusion=True,
    )

    assert decoder.trajectory_gate is not None
    assert torch.allclose(decoder.trajectory_gate.weight, torch.zeros_like(decoder.trajectory_gate.weight))
    assert torch.allclose(decoder.trajectory_gate.bias, torch.ones_like(decoder.trajectory_gate.bias))


def test_lifecycle_loss_zeroes_domain_supervision_when_disabled():
    criterion = LifecycleLoss(
        traj_weight=0.0,
        rul_weight=0.0,
        eol_weight=0.0,
        knee_weight=0.0,
        mono_weight=0.0,
        smooth_weight=0.0,
        domain_weight=1.0,
    )
    batch = {
        "trajectory_target": torch.ones(2, 3),
        "rul_target": torch.zeros(2, 1),
        "eol_target": torch.zeros(2, 1),
        "knee_target": torch.zeros(2, 1),
        "knee_mask": torch.ones(2, 1),
        "source_id": torch.tensor([[1], [0]], dtype=torch.long),
    }
    shared_outputs = {
        "trajectory": torch.ones(2, 3),
        "rul": torch.zeros(2, 1),
        "eol_cycle": torch.zeros(2, 1),
        "knee_cycle": torch.zeros(2, 1),
        "domain_logits": torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32),
    }

    disabled_losses = criterion({**shared_outputs, "domain_loss_enabled": False}, batch)
    enabled_losses = criterion({**shared_outputs, "domain_loss_enabled": True}, batch)

    assert disabled_losses["domain_loss"].item() == 0.0
    assert disabled_losses["loss"].item() == 0.0
    assert enabled_losses["domain_loss"].item() > 0.0
    assert torch.allclose(enabled_losses["loss"], enabled_losses["domain_loss"])


def test_ensure_full_hybrid_summary_refreshes_matching_truth_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "nasa_hybrid.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    matching_summary = {
        "task_kind": "lifecycle",
        "config_path": str(config_path),
        "seeds": [7, 21, 42],
        "aggregate_metrics": {"mean": {"rmse": 0.1, "r2": 0.2}},
        "per_seed_runs": [],
        "best_checkpoint": {"seed": 7, "path": "dummy.pt"},
        "artifact_paths": {"summary": "dummy.json"},
        "plots": [],
    }
    runner_calls: list[int] = []

    monkeypatch.setattr(run_ablation_module, "load_json", lambda path: matching_summary)

    def fake_runner(source, model_type, **kwargs):
        seed = int(kwargs["training_overrides"]["seed"])
        runner_calls.append(seed)
        return {
            "seed": seed,
            "test_metrics": {"rmse": 0.1 + seed * 0.001, "r2": 0.2},
            "best_checkpoint": f"checkpoint-{seed}.pt",
            "final_checkpoint": f"final-{seed}.pt",
            "artifact_dir": f"artifact-{seed}",
            "summary_path": f"summary-{seed}.json",
            "history": {},
            "history_summary": {},
            "test_details": {},
            "test_details_path": f"details-{seed}.json",
            "task_kind": "lifecycle",
            "config_snapshot": {},
            "split_snapshot": {},
            "feature_columns": [],
        }

    monkeypatch.setattr(run_ablation_module, "run_lifecycle_experiment", fake_runner)
    monkeypatch.setattr(
        run_ablation_module,
        "create_multi_seed_summary",
        lambda source, model_type, **kwargs: {
            "source": source,
            "model_type": model_type,
            "task_kind": "lifecycle",
            "seeds": kwargs["seeds"],
            "per_seed_runs": kwargs["per_seed_summaries"],
            "aggregate_metrics": {"mean": {"rmse": 0.123, "r2": 0.456}},
            "best_checkpoint": {"seed": 7, "path": "checkpoint-7.pt"},
            "artifact_paths": {"summary": "summary.json"},
            "plots": [],
        },
    )

    cached_summary, refreshed = run_ablation_module.ensure_full_hybrid_summary(
        "nasa",
        task="lifecycle",
        config_path=config_path,
        seeds=[7, 21, 42],
        refresh_full_hybrid=False,
    )
    assert refreshed is False
    assert cached_summary is matching_summary
    assert runner_calls == []

    refreshed_summary, refreshed = run_ablation_module.ensure_full_hybrid_summary(
        "nasa",
        task="lifecycle",
        config_path=config_path,
        seeds=[7, 21, 42],
        refresh_full_hybrid=True,
    )
    assert refreshed is True
    assert runner_calls == [7, 21, 42]
    assert refreshed_summary["seeds"] == [7, 21, 42]


def test_lifecycle_experiment_resolves_gzipped_cycle_summary_from_legacy_config_path(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("hust", "HUST_GZIP", battery_count=4)
    gzip_path = settings.processed_dir / "hust" / "hust_cycle_summary.csv.gz"
    legacy_path = settings.processed_dir / "hust" / "hust_cycle_summary.csv"
    gzip_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(gzip_path, index=False)

    config_path = _write_config(tmp_path, "hust", "hybrid", legacy_path)
    result = run_lifecycle_experiment(
        "hust",
        "hybrid",
        config_path=config_path,
        training_overrides={"num_epochs": 1, "model_version": "hust-hybrid-gzip"},
        persist_training_run=False,
    )

    assert result["config_snapshot"]["data"]["csv_path"].endswith(".csv.gz")
    assert Path(result["best_checkpoint"]).exists()


def test_transfer_benchmark_runs_multisource_pretrain_then_finetune(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    csv_paths: dict[str, Path] = {}
    for source in ("nasa", "calce", "hust", "matr"):
        csv_path = settings.processed_dir / source / f"{source}_cycle_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        create_synthetic_data(
            csv_path,
            num_batteries=5,
            num_cycles=72,
            source=source,
            dataset_name=f"{source}_transfer_test",
        )
        csv_paths[source] = csv_path

    pretrain_config = _write_config(
        tmp_path,
        "calce",
        "bilstm",
        [csv_paths["nasa"], csv_paths["calce"], csv_paths["hust"], csv_paths["matr"]],
        sources=["nasa", "calce", "hust", "matr"],
    )
    fine_tune_config = _write_config(tmp_path, "calce", "bilstm", csv_paths["calce"], sources=["calce"])

    summary = run_transfer_benchmark(
        "calce",
        "bilstm",
        pretrain_config_path=pretrain_config,
        fine_tune_config_path=fine_tune_config,
        seeds=[7],
        model_dir=settings.model_dir,
        persist_training_run=False,
        pretrain_training_overrides={"num_epochs": 1, "batch_size": 4},
        fine_tune_training_overrides={"num_epochs": 1, "batch_size": 4},
    )

    assert summary["suite_kind"] == "transfer"
    assert summary["task_kind"] == "lifecycle"
    assert len(summary["pretrain_runs"]) == 1
    assert len(summary["fine_tune_runs"]) == 1
    assert Path(summary["artifact_paths"]["summary"]).exists()
    assert Path(summary["best_checkpoint"]["path"]).exists()
    assert summary["aggregate_metrics"]["mean"]["trajectory_rmse"] is not None
    assert "summary_path" not in summary["fine_tune_runs"][0]
    assert summary["pretrain_runs"][0]["best_checkpoint"]

    fine_tune_summary_path = settings.model_dir / "calce" / "bilstm" / "transfer" / "multisource_to_calce" / "fine_tune" / "seed-7" / "bilstm_experiment_summary.json"
    fine_tune_summary = json.loads(fine_tune_summary_path.read_text(encoding="utf-8"))
    assert fine_tune_summary["stage_kind"] == "fine_tune"
    assert fine_tune_summary["init_from_checkpoint"] == summary["pretrain_runs"][0]["best_checkpoint"]
    assert Path(fine_tune_summary["best_checkpoint"]).exists()


def test_lifecycle_trainer_partial_init_skips_shape_mismatches(tmp_path: Path):
    model, model_config = build_lifecycle_model(
        "hybrid",
        input_dim=11,
        vocab_sizes={"source": 2, "chemistry": 1, "protocol": 1},
        overrides={"d_model": 16, "fusion_dim": 16, "transformer_layers": 1, "xlstm_layers": 1},
    )
    state_dict = model.state_dict()
    matching_key = "decoder.rul_head.3.bias"
    mismatched_key = "input_proj.weight"
    checkpoint_path = tmp_path / "partial_init.pt"
    torch.save(
        {
            "model_state_dict": {
                matching_key: torch.full_like(state_dict[matching_key], 3.14),
                mismatched_key: torch.zeros(state_dict[mismatched_key].shape[0] + 1, state_dict[mismatched_key].shape[1]),
            }
        },
        checkpoint_path,
    )

    reloaded_model, reloaded_config = build_lifecycle_model(
        "hybrid",
        input_dim=11,
        vocab_sizes={"source": 2, "chemistry": 1, "protocol": 1},
        overrides={"d_model": 16, "fusion_dim": 16, "transformer_layers": 1, "xlstm_layers": 1},
    )
    trainer = LifecycleTrainer(
        model=reloaded_model,
        model_config=reloaded_config,
        training_config=LifecycleTrainingConfig(
            source="calce",
            model_type="hybrid",
            checkpoint_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            init_from_checkpoint=str(checkpoint_path),
        ),
        train_loader=[],
        val_loader=[],
        test_loader=[],
        data_summary={"feature_columns": ["f"] * 11},
    )

    assert trainer.init_from_report is not None
    assert trainer.init_from_report["loaded_key_count"] == 1
    assert trainer.init_from_report["skipped_shape_mismatch_count"] == 1
    assert torch.allclose(trainer.model.state_dict()[matching_key], torch.full_like(trainer.model.state_dict()[matching_key], 3.14))


def test_lifecycle_experiment_reads_extended_loss_weights(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_cycle_frame("calce", "CALCE_WEIGHT")
    csv_path = settings.processed_dir / "calce" / "calce_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    config_path = _write_config(tmp_path, "calce", "hybrid", csv_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"].update(
        {
            "num_epochs": 1,
            "artifact_subdir": "experiments/weights-check",
            "traj_weight": 1.23,
            "rul_weight": 0.34,
            "eol_weight": 0.45,
            "knee_weight": 0.56,
            "mono_weight": 0.067,
            "smooth_weight": 0.078,
            "domain_weight": 0.089,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = run_lifecycle_experiment("calce", "hybrid", config_path=config_path, persist_training_run=False)

    training_config = summary["training_config"]
    assert training_config["traj_weight"] == pytest.approx(1.23)
    assert training_config["rul_weight"] == pytest.approx(0.34)
    assert training_config["eol_weight"] == pytest.approx(0.45)
    assert training_config["knee_weight"] == pytest.approx(0.56)
    assert training_config["mono_weight"] == pytest.approx(0.067)
    assert training_config["smooth_weight"] == pytest.approx(0.078)
    assert training_config["domain_weight"] == pytest.approx(0.089)
    assert summary["artifact_dir"].endswith("experiments/weights-check")


def test_reusable_root_summary_and_plot_bundle_use_single_run_fallback(tmp_path: Path):
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(settings.database_path)
    database.initialize()

    frame = _build_cycle_frame("nasa", "NASA_ROOT")
    csv_path = settings.processed_dir / "nasa" / "nasa_cycle_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    bilstm_config = _write_config(tmp_path, "nasa", "bilstm", csv_path)
    hybrid_config = _write_config(tmp_path, "nasa", "hybrid", csv_path)
    run_lifecycle_experiment("nasa", "bilstm", config_path=bilstm_config, persist_training_run=False)
    run_lifecycle_experiment("nasa", "hybrid", config_path=hybrid_config, persist_training_run=False)

    cached = reusable_root_summary("nasa", "hybrid", 42, "lifecycle", model_dir=settings.model_dir)
    plots = generate_source_plot_bundle("nasa", model_dir=settings.model_dir, processed_dir=settings.processed_dir)

    assert cached is not None
    assert cached["seed"] == 42
    assert any(item["key"] == "experiment_summary" for item in plots)


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
            "aggregate_metrics": full_summary["aggregate_metrics"],
            "best_checkpoint": full_summary["best_checkpoint"],
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
    assert summary["task_kind"] == "lifecycle"
    variant_keys = {item["key"] for item in summary["variants"]}
    assert {"full_hybrid", "no_xlstm", "no_transformer", "no_domain_embedding", "no_trajectory_head"} <= variant_keys
    assert all("aggregate_metrics" in item for item in summary["variants"])
    assert all("delta_vs_full" in item for item in summary["variants"])
    assert summary["guardrail"]["checked_variants"] == ["no_xlstm", "no_transformer"]
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
    assert not (export_dir / "prediction_report.md").exists()
    assert not (export_dir / "diagnosis_report.md").exists()
    assert (export_dir / "sample_profile.json").exists()
    assert (export_dir / "dataset_profile.json").exists()
    assert (export_dir / "experiment_context.json").exists()
    assert (export_dir / "charts" / "lifecycle_trajectory.png").exists()
    assert (export_dir / "charts" / "graphrag_evidence.png").exists()
    assert (export_dir / "charts" / "benchmark_summary.png").exists()
    assert not (export_dir / "charts" / "rul_projection.png").exists()
    assert not (export_dir / "charts" / "diagnosis_graph.png").exists()
    assert not (export_dir / "charts" / "experiment_summary.png").exists()
    assert result["bundle_snapshot"]["last_export"] is not None
