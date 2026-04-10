"""本地训练任务编排服务。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.repository import BHMSRepository
from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training.benchmark_truth import build_source_comparison_summary, normalize_legacy_comparison_summary
from ml.training.experiment_constants import ABLATION_VARIANTS


class TrainingService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()

    def get_comparison(self, source: str) -> dict[str, Any]:
        source = source.lower()
        card = get_dataset_card(source)
        if not card.training_ready:
            return {
                "source": source,
                "previous": None,
                "current": None,
                "latest_job": None,
            }
        latest_job = self.repository.latest_completed_training_job(source)
        current = self._load_comparison_file(source)
        previous = latest_job.get("baseline") if latest_job else None
        job_result = latest_job.get("result") if latest_job else None
        if isinstance(job_result, dict) and isinstance(job_result.get("comparison"), dict):
            current_payload = job_result.get("comparison")
        else:
            current_payload = job_result
        return {
            "source": source,
            "previous": normalize_legacy_comparison_summary(previous, source=source) if isinstance(previous, dict) else previous,
            "current": normalize_legacy_comparison_summary(current_payload, source=source) if isinstance(current_payload, dict) else current,
            "latest_job": latest_job,
        }

    def get_overview(self) -> dict[str, Any]:
        sources = [self.get_experiment_detail(source) for source in list_supported_sources()]
        warnings = [warning for detail in sources for warning in detail.get("warnings", [])]
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "sources": [
                {
                    "source": detail["source"],
                    "best_model": detail.get("best_model"),
                    "best_models": detail.get("best_models", {}),
                    "dataset_batteries": detail.get("dataset_summary", {}).get("num_batteries", 0),
                    "academic_status": detail.get("academic_status"),
                    "headline": detail.get("headline"),
                    "warnings": detail.get("warnings", []),
                    "plot_count": len(detail.get("plots", [])),
                    "paper_gate_passed": bool((detail.get("paper_gate") or {}).get("passed")),
                    "training_ready": detail.get("training_ready", True),
                    "ingestion_mode": detail.get("ingestion_mode"),
                    "source_role": detail.get("source_role", "training_ready"),
                }
                for detail in sources
            ],
            "summary_notes": [
                "该概览优先显示 lifecycle within-source、多随机种子、消融与 benchmark 资产是否齐备。",
                "若某个来源仍缺少 multi-seed、ablation 或 comparison 产物，详细页会保留可直接执行的命令提示。",
                "当前优先指标默认兼容为 trajectory RMSE/R2；若 R2 为负或误差过高，系统会显式标记实验风险。",
            ],
            "warnings": list(dict.fromkeys(warnings)),
        }

    def get_experiment_detail(self, source: str) -> dict[str, Any]:
        source = source.lower()
        card = get_dataset_card(source)
        dataset_summary = self._load_json_file(self.settings.processed_dir / source / f"{source}_dataset_summary.json") or {}
        comparison = self.get_comparison(source)
        if not card.training_ready:
            source_role = "enhancement_only" if card.ingestion_mode == "enhancement_assets" else "auxiliary"
            headline = "增强资产源，仅供机制增强" if source_role == "enhancement_only" else "辅助轨迹源，不进入主训练基准"
            key_findings = (
                [
                    "该来源仅生成 enhancement 资产索引，供 GraphRAG/机理侧检索使用。",
                    "不会进入 prepare_training_dataset、baseline comparison 或 benchmark 排名。",
                ]
                if source_role == "enhancement_only"
                else [
                    "该来源已转换为标准 cycle summary，可供 trajectory 辅助分析与案例展示。",
                    "不会自动加入 lifecycle 主训练候选池。",
                ]
            )
            return {
                "source": source,
                "dataset_summary": dataset_summary,
                "comparison": comparison,
                "models": {},
                "best_model": None,
                "best_models": {},
                "benchmark_units": [],
                "paper_gate": {"passed": False, "required_units": [], "available_units": [], "passing_units": [], "failing_units": []},
                "ablation_gate": {"available": False, "checked_variants": [], "blocking_variants": [], "passed": False},
                "headline": headline,
                "academic_status": "增强资产源" if source_role == "enhancement_only" else "辅助轨迹源",
                "warnings": [],
                "key_findings": key_findings,
                "plots": [],
                "artifact_paths": {
                    "source_dir": str(self.settings.model_dir / source),
                    "plots_dir": str(self.settings.model_dir / source / "plots"),
                },
                "recommended_commands": {},
                "training_ready": False,
                "ingestion_mode": card.ingestion_mode,
                "source_group": card.group,
                "source_role": source_role,
            }
        model_details = {
            model_type: self._build_model_detail(source, model_type)
            for model_type in ("bilstm", "hybrid")
        }
        comparison_current = comparison.get("current") or {}
        best_models = comparison_current.get("best_models") or {}
        paper_gate = comparison_current.get("paper_gate") or {}
        ablation_gate = comparison_current.get("ablation_gate") or {}
        benchmark_units = comparison_current.get("benchmark_units") or []
        best_model = best_models.get("within_source") or self._best_model_name_from_details(model_details)
        warnings = self._experiment_warnings(model_details, comparison_current)
        improvement = self._hybrid_improvement(model_details)
        headline = self._detail_headline(comparison_current, warnings)
        plots = self._source_plots(source)
        hybrid_config_name = f"{source}_hybrid.yaml"
        paper_config_path = self.settings.project_root / "configs" / f"{source}_hybrid_paper.yaml"
        if paper_config_path.exists() and paper_gate.get("passed"):
            hybrid_config_name = paper_config_path.name
        recommended_commands = {
            "multi_seed_hybrid": f"python scripts/run_multi_seed_experiment.py --source {source} --model hybrid --task lifecycle --config configs/{hybrid_config_name} --force",
            "multi_seed_bilstm": f"python scripts/run_multi_seed_experiment.py --source {source} --model bilstm --task lifecycle --config configs/{source}_bilstm.yaml --force",
            "ablation_study": f"python scripts/run_ablation_study.py --source {source} --task lifecycle --config configs/{hybrid_config_name} --force",
            "rebuild_benchmark_truth": f"python scripts/rebuild_benchmark_truth.py --sources {source}",
        }
        if source in {"calce", "nasa"}:
            recommended_commands["transfer_hybrid"] = (
                f"python scripts/run_transfer_benchmark.py --target {source} --model hybrid "
                f"--pretrain-config configs/multisource_pretrain_hybrid.yaml --finetune-config configs/transfer_{source}_hybrid.yaml --seeds 7,21,42"
            )
            recommended_commands["transfer_bilstm"] = (
                f"python scripts/run_transfer_benchmark.py --target {source} --model bilstm "
                f"--pretrain-config configs/multisource_pretrain_bilstm.yaml --finetune-config configs/transfer_{source}_bilstm.yaml --seeds 7,21,42"
            )
        return {
            "source": source,
            "dataset_summary": dataset_summary,
            "comparison": comparison,
            "models": model_details,
            "best_model": best_model,
            "best_models": best_models,
            "benchmark_units": benchmark_units,
            "paper_gate": paper_gate,
            "ablation_gate": ablation_gate,
            "headline": headline,
            "academic_status": "具备论文展示基础" if paper_gate.get("passed") and not warnings else "需要补实验",
            "warnings": warnings,
            "key_findings": self._key_findings(model_details, comparison_current, best_model, improvement),
            "plots": plots,
            "artifact_paths": {
                "source_dir": str(self.settings.model_dir / source),
                "plots_dir": str(self.settings.model_dir / source / "plots"),
            },
            "recommended_commands": recommended_commands,
            "training_ready": card.training_ready,
            "ingestion_mode": card.ingestion_mode,
            "source_group": card.group,
            "source_role": "training_ready",
        }

    def get_ablation_summary(self, source: str) -> dict[str, Any]:
        source = source.lower()
        card = get_dataset_card(source)
        if not card.training_ready:
            return {
                "source": source,
                "available": False,
                "notes": [
                    f"{source.upper()} 当前被标记为 {card.ingestion_mode}，不会进入 lifecycle hybrid 消融。",
                    "如需使用该来源，请优先查看 processed 资产或 enhancement 索引，而非 benchmark 训练结果。",
                ],
                "recommended_command": None,
                "variants": [],
            }
        path = self.settings.model_dir / source / "ablation_summary.json"
        payload = self._load_json_file(path)
        if payload is not None:
            payload.setdefault("source", source)
            payload.setdefault("available", True)
            return payload
        return {
            "source": source,
            "available": False,
            "notes": [
                "当前来源尚未生成 ablation_summary.json。",
                "建议先确保 lifecycle hybrid 多随机种子结果齐备，再运行 lifecycle hybrid 消融。",
            ],
            "recommended_command": f"python scripts/run_ablation_study.py --source {source} --task lifecycle --config configs/{source}_hybrid.yaml --force",
            "variants": [
                {
                    "key": item["key"],
                    "label": item["label"],
                    "description": item["description"],
                    "status": "planned",
                    "config_overrides": item.get("model_overrides", {}),
                    "feature_columns": item.get("feature_columns"),
                }
                for item in ABLATION_VARIANTS
            ],
        }

    def _config_for(self, source: str, model_type: str) -> str:
        return str(self.settings.project_root / "configs" / f"{source}_{model_type}.yaml")

    def _load_comparison_file(self, source: str) -> Optional[dict[str, Any]]:
        path = self.settings.model_dir / source / "comparison_summary.json"
        if not path.exists():
            return build_source_comparison_summary(source, model_dir=self.settings.model_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return normalize_legacy_comparison_summary(payload, source=source)

    def _build_model_detail(self, source: str, model_type: str) -> dict[str, Any]:
        experiment_summary = self._load_json_file(
            self.settings.model_dir / source / model_type / f"{model_type}_experiment_summary.json"
        ) or {}
        multi_seed_summary = self._load_json_file(
            self.settings.model_dir / source / model_type / f"{model_type}_multi_seed_summary.json"
        )
        metrics = experiment_summary.get("test_metrics", {})
        aggregate_metrics = (multi_seed_summary or {}).get("aggregate_metrics") or {"mean": {}, "std": {}}
        preferred_metrics = self._preferred_metrics(experiment_summary, multi_seed_summary)
        return {
            "model_type": model_type,
            "best_val_loss": experiment_summary.get("best_val_loss"),
            "test_metrics": metrics,
            "multi_seed_available": bool(multi_seed_summary),
            "aggregate_metrics": aggregate_metrics,
            "plots_available": bool((multi_seed_summary or {}).get("plots")),
            "preferred_metrics": preferred_metrics,
            "assessment": self._model_assessment(model_type, preferred_metrics, multi_seed_summary),
        }

    @staticmethod
    def _load_json_file(path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _model_assessment(model_type: str, metrics: dict[str, Any], multi_seed_summary: Optional[dict[str, Any]]) -> str:
        rmse = metrics.get("rmse")
        r2 = metrics.get("r2")
        model_label = "Lifecycle Hybrid" if model_type == "hybrid" else "Lifecycle Bi-LSTM"
        if not metrics:
            return f"{model_label} 还没有形成可解读的生命周期测试指标。"
        if isinstance(r2, (int, float)) and float(r2) < 0:
            if multi_seed_summary:
                return f"{model_label} 已形成多随机种子资产，但聚合 trajectory R2 仍为负，说明泛化结论仍偏弱。"
            return f"{model_label} 已能跑通，但当前 trajectory R2 为负，仍偏演示级结果。"
        if multi_seed_summary:
            return f"{model_label} 已补充多随机种子摘要，可直接支撑生命周期实验分析。"
        if isinstance(rmse, (int, float)):
            return f"{model_label} 已形成单次实验结果，建议继续补充多随机种子、消融与 transfer 实验。"
        return f"{model_label} 指标仍不完整，建议重新生成生命周期实验摘要。"

    def _source_plots(self, source: str) -> list[dict[str, Any]]:
        manifest_path = self.settings.model_dir / source / "plots" / "plot_manifest.json"
        payload = self._load_json_file(manifest_path) or {}
        return payload.get("plots", [])

    @staticmethod
    def _preferred_metrics(experiment_summary: dict[str, Any], multi_seed_summary: Optional[dict[str, Any]]) -> dict[str, Any]:
        aggregate_mean = ((multi_seed_summary or {}).get("aggregate_metrics") or {}).get("mean") or {}
        return aggregate_mean or experiment_summary.get("test_metrics", {}) or {}

    @staticmethod
    def _best_model_name_from_details(model_details: dict[str, dict[str, Any]]) -> Optional[str]:
        best_name: Optional[str] = None
        best_rmse: Optional[float] = None
        for model_name, payload in model_details.items():
            metrics = payload.get("preferred_metrics") or {}
            rmse = metrics.get("rmse")
            if isinstance(rmse, (int, float)) and (best_rmse is None or float(rmse) < best_rmse):
                best_rmse = float(rmse)
                best_name = model_name
        return best_name

    @staticmethod
    def _best_model_name(comparison: dict[str, Any]) -> Optional[str]:
        best_models = (comparison or {}).get("best_models") or {}
        winner = best_models.get("within_source") or best_models.get("transfer")
        if winner:
            return winner
        models = (comparison or {}).get("models") or {}
        best_name: Optional[str] = None
        best_rmse: Optional[float] = None
        for model_name, payload in models.items():
            metrics = payload.get("test_metrics") or {}
            rmse = metrics.get("rmse")
            if isinstance(rmse, (int, float)) and (best_rmse is None or float(rmse) < best_rmse):
                best_rmse = float(rmse)
                best_name = model_name
        return best_name

    @staticmethod
    def _hybrid_improvement(model_details: dict[str, dict[str, Any]]) -> Optional[float]:
        bilstm_metrics = model_details.get("bilstm", {}).get("preferred_metrics") or {}
        hybrid_metrics = model_details.get("hybrid", {}).get("preferred_metrics") or {}
        bilstm_rmse = bilstm_metrics.get("rmse")
        hybrid_rmse = hybrid_metrics.get("rmse")
        if not isinstance(bilstm_rmse, (int, float)) or not isinstance(hybrid_rmse, (int, float)):
            return None
        if float(bilstm_rmse) == 0:
            return None
        return round((float(bilstm_rmse) - float(hybrid_rmse)) / float(bilstm_rmse) * 100.0, 3)

    @staticmethod
    def _detail_headline(comparison: dict[str, Any], warnings: list[str]) -> str:
        benchmark_units = comparison.get("benchmark_units") or []
        unit_parts: list[str] = []
        for unit in benchmark_units:
            key = unit.get("key")
            winner = unit.get("winner_model")
            label = "transfer" if key == "transfer" else "within-source"
            if winner:
                winner_label = "Lifecycle Hybrid" if winner == "hybrid" else "Lifecycle Bi-LSTM"
                unit_parts.append(f"{label} 由 {winner_label} 占优")
            else:
                unit_parts.append(f"{label} 暂无单一占优模型")
        if not unit_parts:
            return "当前来源尚未形成稳定的生命周期实验对比摘要。"
        suffix = "论文门槛已通过。" if (comparison.get("paper_gate") or {}).get("passed") else "论文门槛未通过，仍需补实验。"
        if warnings:
            return f"{'；'.join(unit_parts)}；{suffix}"
        return f"{'；'.join(unit_parts)}；{suffix}"

    @staticmethod
    def _experiment_warnings(model_details: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        for model_name, payload in model_details.items():
            metrics = payload.get("preferred_metrics") or {}
            r2 = metrics.get("r2")
            mape = metrics.get("mape")
            if isinstance(r2, (int, float)) and float(r2) < 0:
                warnings.append(f"{model_name} 的 R2 为负，说明当前泛化效果仍不足。")
            if isinstance(mape, (int, float)) and float(mape) > 100:
                warnings.append(f"{model_name} 的 MAPE 偏高，建议补充误差分布图和多随机种子实验。")
            if not payload.get("multi_seed_available"):
                warnings.append(f"{model_name} 尚未生成多随机种子汇总。")
        if not (comparison or {}).get("benchmark_units"):
            warnings.append("当前来源还未生成 comparison_summary.json。")
        if not any(payload.get("plots_available") for payload in model_details.values()):
            warnings.append("当前来源尚未生成模型级论文图表。")
        warnings.extend((comparison or {}).get("warnings") or [])
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _key_findings(
        model_details: dict[str, dict[str, Any]],
        comparison: dict[str, Any],
        best_model: Optional[str],
        improvement: Optional[float],
    ) -> list[str]:
        findings: list[str] = []
        if best_model:
            findings.append(f"当前按优先指标表现更优的模型是 {best_model}。")
        if improvement is not None:
            direction = "降低" if improvement >= 0 else "升高"
            findings.append(f"Lifecycle Hybrid 相对 Lifecycle Bi-LSTM 的 trajectory RMSE {direction}了 {abs(improvement):.3f}%。")
        for unit in comparison.get("benchmark_units", []):
            winner = unit.get("winner_model") or "mixed"
            findings.append(
                f"{unit.get('label', unit.get('key', 'benchmark'))}：winner={winner}，paper_gate={'pass' if unit.get('paper_gate_passed') else 'fail'}。"
            )
        paper_gate = comparison.get("paper_gate") or {}
        if paper_gate:
            findings.append(
                f"论文门槛：required={paper_gate.get('required_units', [])}，passing={paper_gate.get('passing_units', [])}，failing={paper_gate.get('failing_units', [])}。"
            )
        ablation_gate = comparison.get("ablation_gate") or {}
        if ablation_gate.get("available"):
            findings.append(
                f"Hybrid 消融门槛：{'通过' if ablation_gate.get('passed') else '未通过'}，blocking={len(ablation_gate.get('blocking_variants', []))}。"
            )
        for model_name, payload in model_details.items():
            findings.append(payload["assessment"])
            if payload.get("multi_seed_available"):
                aggregate = payload.get("aggregate_metrics", {})
                mean = aggregate.get("mean", {})
                std = aggregate.get("std", {})
                findings.append(
                    f"{model_name} 多随机种子聚合：trajectory RMSE {mean.get('rmse')} ± {std.get('rmse')}，R2 {mean.get('r2')} ± {std.get('r2')}。"
                )
        return findings

__all__ = ["TrainingService"]
