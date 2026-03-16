"""本地训练任务编排服务。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.battery_service import BatteryService
from backend.app.services.repository import BHMSRepository
from ml.data.source_registry import list_supported_sources
from ml.training.experiment_constants import ABLATION_VARIANTS


class TrainingService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.battery_service = BatteryService(repository=self.repository, settings=self.settings)

    def create_job(
        self,
        source: str,
        model_scope: str = "all",
        force_run: bool = False,
        job_kind: str = "baseline",
        seed_count: int = 3,
    ) -> dict[str, Any]:
        source = source.lower()
        baseline = self._load_comparison_file(source)
        job_id = self.repository.insert_training_job(
            {
                "source": source,
                "model_scope": model_scope,
                "status": "queued",
                "current_stage": "queued",
                "force_run": force_run,
                "baseline": baseline,
                "metadata": {
                    "source": source,
                    "model_scope": model_scope,
                    "job_kind": job_kind,
                    "seed_count": seed_count,
                    "suite_progress": [],
                    "artifact_paths": [],
                },
            }
        )
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source, model_scope, force_run, job_kind, seed_count),
            daemon=True,
        )
        thread.start()
        job = self.repository.get_training_job(job_id)
        if job is None:
            raise BHMSException("训练任务创建失败", status_code=500, code="training_job_create_failed")
        return job

    def list_jobs(self, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self.repository.list_training_jobs(source=source)

    def get_job(self, job_id: int) -> dict[str, Any]:
        job = self.repository.get_training_job(job_id)
        if job is None:
            raise BHMSException(f"未找到训练任务 {job_id}", status_code=404, code="training_job_not_found")
        return job

    def list_runs(self, source: Optional[str] = None, model_type: Optional[str] = None) -> list[dict[str, Any]]:
        return self.repository.list_training_runs(source=source, model_type=model_type)

    def get_comparison(self, source: str) -> dict[str, Any]:
        source = source.lower()
        latest_job = self.repository.latest_completed_training_job(source)
        current = self._load_comparison_file(source)
        previous = latest_job.get("baseline") if latest_job else None
        job_result = latest_job.get("result") if latest_job else None
        if isinstance(job_result, dict) and isinstance(job_result.get("comparison"), dict):
            current_payload = job_result.get("comparison")
        else:
            current_payload = job_result
        runs = self.repository.list_training_runs(source=source, limit=20)
        return {
            "source": source,
            "previous": previous,
            "current": current_payload or current,
            "latest_job": latest_job,
            "runs": runs,
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
                    "dataset_batteries": detail.get("dataset_summary", {}).get("num_batteries", 0),
                    "academic_status": detail.get("academic_status"),
                    "headline": detail.get("headline"),
                    "warnings": detail.get("warnings", []),
                    "plot_count": len(detail.get("plots", [])),
                }
                for detail in sources
            ],
            "summary_notes": [
                "该概览优先显示多随机种子与消融资产是否齐备，以及论文图表是否已经生成。",
                "若某个来源仍缺少 multi-seed 或 ablation 产物，详细页会保留可直接执行的命令提示。",
                "R2 为负值或 MAPE 过高时，系统会在 headline 与 warnings 中显式标记实验风险。",
            ],
            "warnings": list(dict.fromkeys(warnings)),
        }

    def get_experiment_detail(self, source: str) -> dict[str, Any]:
        source = source.lower()
        dataset_summary = self._load_json_file(self.settings.processed_dir / source / f"{source}_dataset_summary.json") or {}
        comparison = self.get_comparison(source)
        model_details = {
            model_type: self._build_model_detail(source, model_type)
            for model_type in ("bilstm", "hybrid")
        }
        best_model = self._best_model_name_from_details(model_details)
        warnings = self._experiment_warnings(model_details, comparison.get("current") or {})
        improvement = self._hybrid_improvement(model_details)
        headline = self._detail_headline(best_model, improvement, warnings)
        plots = self._source_plots(source)
        return {
            "source": source,
            "dataset_summary": dataset_summary,
            "comparison": comparison,
            "models": model_details,
            "best_model": best_model,
            "headline": headline,
            "academic_status": "需要补实验" if warnings else "具备论文展示基础",
            "warnings": warnings,
            "key_findings": self._key_findings(model_details, best_model, improvement),
            "plots": plots,
            "artifact_paths": {
                "source_dir": str(self.settings.model_dir / source),
                "plots_dir": str(self.settings.model_dir / source / "plots"),
            },
            "recommended_commands": {
                "multi_seed_hybrid": f"python scripts/run_multi_seed_experiment.py --source {source} --model hybrid --config configs/{source}_hybrid.yaml --force",
                "multi_seed_bilstm": f"python scripts/run_multi_seed_experiment.py --source {source} --model bilstm --config configs/{source}_bilstm.yaml --force",
                "ablation_study": f"python scripts/run_ablation_study.py --source {source} --config configs/{source}_hybrid.yaml --force",
            },
        }

    def get_ablation_summary(self, source: str) -> dict[str, Any]:
        source = source.lower()
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
                "建议先确保 hybrid 多随机种子结果齐备，再运行三组 Hybrid 消融。",
            ],
            "recommended_command": f"python scripts/run_ablation_study.py --source {source} --config configs/{source}_hybrid.yaml --force",
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

    def _run_job(self, job_id: int, source: str, model_scope: str, force_run: bool, job_kind: str, seed_count: int) -> None:
        effective_scope = "all" if job_kind == "full_suite" else model_scope
        self.repository.update_training_job(
            job_id,
            status="running",
            current_stage="prepare_dataset",
            started_at=self.repository._now(),
        )
        logs: list[str] = []
        suite_progress: list[str] = []
        artifact_paths: list[str] = []
        try:
            prepared = self.battery_service.prepare_training_dataset(source)
            logs.append(json.dumps(prepared["data_summary"], ensure_ascii=False, indent=2))
            suite_progress.append("prepare_dataset")
            self._update_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=artifact_paths)

            if job_kind in {"baseline", "full_suite"}:
                if effective_scope in {"all", "bilstm"}:
                    self.repository.update_training_job(job_id, current_stage="baseline_bilstm", log_excerpt=self._collapse_logs(logs))
                    logs.append(self._run_command([sys.executable, "scripts/train_models.py", "--source", source, "--model", "bilstm", "--config", self._config_for(source, "bilstm")]))
                    suite_progress.append("baseline_bilstm")
                if effective_scope in {"all", "hybrid"}:
                    self.repository.update_training_job(job_id, current_stage="baseline_hybrid", log_excerpt=self._collapse_logs(logs))
                    logs.append(self._run_command([sys.executable, "scripts/train_models.py", "--source", source, "--model", "hybrid", "--config", self._config_for(source, "hybrid")]))
                    suite_progress.append("baseline_hybrid")
                if effective_scope == "all":
                    self.repository.update_training_job(job_id, current_stage="compare_models", log_excerpt=self._collapse_logs(logs))
                    compare_cmd = [sys.executable, "scripts/run_comparison.py", "--source", source]
                    if force_run:
                        compare_cmd.append("--force")
                    logs.append(self._run_command(compare_cmd))
                    suite_progress.append("compare_models")
                self._update_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=artifact_paths)

            if job_kind in {"multi_seed", "full_suite"}:
                for model_type in ("bilstm", "hybrid"):
                    if effective_scope not in {"all", model_type}:
                        continue
                    self.repository.update_training_job(job_id, current_stage=f"multi_seed_{model_type}", log_excerpt=self._collapse_logs(logs))
                    multi_seed_cmd = [
                        sys.executable,
                        "scripts/run_multi_seed_experiment.py",
                        "--source",
                        source,
                        "--model",
                        model_type,
                        "--config",
                        self._config_for(source, model_type),
                        "--seeds",
                        self._seeds_arg(seed_count),
                    ]
                    if force_run:
                        multi_seed_cmd.append("--force")
                    logs.append(self._run_command(multi_seed_cmd))
                    suite_progress.append(f"multi_seed_{model_type}")
                    artifact_paths.append(str(self.settings.model_dir / source / model_type / f"{model_type}_multi_seed_summary.json"))
                self._update_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=artifact_paths)

            if job_kind in {"ablation", "full_suite"} and effective_scope in {"all", "hybrid"}:
                self.repository.update_training_job(job_id, current_stage="ablation_hybrid", log_excerpt=self._collapse_logs(logs))
                ablation_cmd = [
                    sys.executable,
                    "scripts/run_ablation_study.py",
                    "--source",
                    source,
                    "--config",
                    self._config_for(source, "hybrid"),
                    "--seeds",
                    self._seeds_arg(seed_count),
                ]
                if force_run:
                    ablation_cmd.append("--force")
                logs.append(self._run_command(ablation_cmd))
                suite_progress.append("ablation_hybrid")
                artifact_paths.append(str(self.settings.model_dir / source / "ablation_summary.json"))
                self._update_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=artifact_paths)

            artifact_paths.extend(self._existing_artifact_paths(source))
            self.repository.update_training_job(job_id, current_stage="generate_plots", log_excerpt=self._collapse_logs(logs))
            result = {
                "job_kind": job_kind,
                "comparison": self._load_comparison_file(source),
                "detail": self.get_experiment_detail(source),
                "ablation": self.get_ablation_summary(source),
                "artifact_paths": list(dict.fromkeys(artifact_paths)),
            }

            self.repository.update_training_job(
                job_id,
                status="completed",
                current_stage="completed",
                result=result,
                log_excerpt=self._collapse_logs(logs),
                metadata=self._merged_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=list(dict.fromkeys(artifact_paths))),
                finished_at=self.repository._now(),
            )
        except Exception as exc:
            logs.append(str(exc))
            self.repository.update_training_job(
                job_id,
                status="failed",
                current_stage="failed",
                error_message=str(exc),
                log_excerpt=self._collapse_logs(logs),
                metadata=self._merged_job_metadata(job_id, suite_progress=suite_progress, artifact_paths=list(dict.fromkeys(artifact_paths))),
                finished_at=self.repository._now(),
            )

    def _run_command(self, cmd: list[str]) -> str:
        completed = subprocess.run(
            cmd,
            cwd=self.settings.project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        output = "\n".join([chunk for chunk in [completed.stdout.strip(), completed.stderr.strip()] if chunk]).strip()
        return output or f"{' '.join(cmd)} finished"

    def _config_for(self, source: str, model_type: str) -> str:
        return str(self.settings.project_root / "configs" / f"{source}_{model_type}.yaml")

    def _load_comparison_file(self, source: str) -> Optional[dict[str, Any]]:
        path = self.settings.model_dir / source / "comparison_summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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
            "best_checkpoint": experiment_summary.get("best_checkpoint"),
            "final_checkpoint": experiment_summary.get("final_checkpoint"),
            "history_available": bool(experiment_summary.get("history")),
            "multi_seed_summary": multi_seed_summary,
            "single_run_summary": experiment_summary,
            "multi_seed_available": bool(multi_seed_summary),
            "aggregate_metrics": aggregate_metrics,
            "per_seed_runs": (multi_seed_summary or {}).get("per_seed_runs", []),
            "plots": (multi_seed_summary or {}).get("plots", []),
            "artifact_paths": (multi_seed_summary or {}).get("artifact_paths", {}),
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
        model_label = "Hybrid" if model_type == "hybrid" else "Bi-LSTM"
        if not metrics:
            return f"{model_label} 还没有形成可解读的测试指标。"
        if isinstance(r2, (int, float)) and float(r2) < 0:
            if multi_seed_summary:
                return f"{model_label} 已形成多随机种子资产，但聚合 R2 仍为负，说明泛化结论仍偏弱。"
            return f"{model_label} 已能跑通，但当前 R2 为负，仍偏演示级结果。"
        if multi_seed_summary:
            return f"{model_label} 已补充多随机种子摘要，可直接支撑论文实验分析。"
        if isinstance(rmse, (int, float)):
            return f"{model_label} 已形成单次实验结果，建议继续补充多随机种子与消融实验。"
        return f"{model_label} 指标仍不完整，建议重新生成实验摘要。"

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
    def _seeds_arg(seed_count: int) -> str:
        base = [7, 21, 42, 84, 126]
        return ",".join(str(item) for item in base[: max(1, seed_count)])

    def _existing_artifact_paths(self, source: str) -> list[str]:
        candidates = [
            self.settings.model_dir / source / "comparison_summary.json",
            self.settings.model_dir / source / "ablation_summary.json",
            self.settings.model_dir / source / "plots" / "plot_manifest.json",
            self.settings.model_dir / source / "bilstm" / "bilstm_multi_seed_summary.json",
            self.settings.model_dir / source / "hybrid" / "hybrid_multi_seed_summary.json",
        ]
        return [str(path) for path in candidates if path.exists()]

    def _merged_job_metadata(self, job_id: int, **updates: Any) -> dict[str, Any]:
        job = self.repository.get_training_job(job_id) or {}
        metadata = dict(job.get("metadata", {}))
        metadata.update(updates)
        return metadata

    def _update_job_metadata(self, job_id: int, **updates: Any) -> None:
        self.repository.update_training_job(job_id, metadata=self._merged_job_metadata(job_id, **updates))

    @staticmethod
    def _best_model_name(comparison: dict[str, Any]) -> Optional[str]:
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
    def _detail_headline(best_model: Optional[str], improvement: Optional[float], warnings: list[str]) -> str:
        if not best_model:
            return "当前来源尚未形成稳定的实验对比摘要。"
        label = "Hybrid" if best_model == "hybrid" else "Bi-LSTM"
        if improvement is not None:
            direction = "提升" if improvement >= 0 else "下降"
            return f"{label} 当前在该来源上相对基线 {direction} {abs(improvement):.3f}%（按 RMSE 口径）。"
        if warnings:
            return f"{label} 当前暂列最优，但仍有实验风险需要补强。"
        return f"{label} 当前在该来源上表现最优。"

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
            if not payload.get("multi_seed_summary"):
                warnings.append(f"{model_name} 尚未生成多随机种子汇总。")
        if not (comparison or {}).get("models"):
            warnings.append("当前来源还未生成 comparison_summary.json。")
        if not any(payload.get("plots") for payload in model_details.values()):
            warnings.append("当前来源尚未生成模型级论文图表。")
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _key_findings(model_details: dict[str, dict[str, Any]], best_model: Optional[str], improvement: Optional[float]) -> list[str]:
        findings: list[str] = []
        if best_model:
            findings.append(f"当前按优先指标表现更优的模型是 {best_model}。")
        if improvement is not None:
            direction = "降低" if improvement >= 0 else "升高"
            findings.append(f"Hybrid 相对 Bi-LSTM 的 RMSE {direction}了 {abs(improvement):.3f}%。")
        for model_name, payload in model_details.items():
            findings.append(payload["assessment"])
            if payload.get("multi_seed_available"):
                aggregate = payload.get("aggregate_metrics", {})
                mean = aggregate.get("mean", {})
                std = aggregate.get("std", {})
                findings.append(
                    f"{model_name} 多随机种子聚合：RMSE {mean.get('rmse')} ± {std.get('rmse')}，R2 {mean.get('r2')} ± {std.get('r2')}。"
                )
        return findings

    @staticmethod
    def _collapse_logs(chunks: list[str], limit: int = 8000) -> str:
        joined = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        if len(joined) <= limit:
            return joined
        return joined[-limit:]


__all__ = ["TrainingService"]
