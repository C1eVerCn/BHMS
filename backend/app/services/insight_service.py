"""Graduation-design insight and export services."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.model_service import PredictionService
from backend.app.services.repository import BHMSRepository
from backend.app.services.training_service import TrainingService
from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training.experiment_artifacts import write_placeholder_png

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

SUPPORTED_SOURCES = tuple(list_supported_sources())
PROFILE_FEATURES = (
    "capacity",
    "voltage_mean",
    "current_mean",
    "temperature_mean",
    "internal_resistance",
)
CJK_FONT_CANDIDATES = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
]


def _configure_matplotlib_fonts() -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in CJK_FONT_CANDIDATES if font in available), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


class InsightService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.prediction_service = PredictionService(repository=self.repository, settings=self.settings)
        self.training_service = TrainingService(repository=self.repository, settings=self.settings)

    def get_dataset_profile(self, source: str) -> dict[str, Any]:
        source = self._validate_source(source)
        card = get_dataset_card(source)
        processed_dir = self.settings.processed_dir / source
        dataset_summary = self._load_json(processed_dir / f"{source}_dataset_summary.json") or {}
        feature_config = self._load_json(processed_dir / f"{source}_feature_config.json") or {}
        split = dataset_summary.get("split", {})
        num_samples = dataset_summary.get("num_samples", {})

        with self.repository.database.connection() as connection:
            battery_row = connection.execute(
                """
                SELECT COUNT(*) AS battery_count,
                       SUM(CASE WHEN include_in_training = 1 THEN 1 ELSE 0 END) AS training_candidate_count
                FROM batteries
                WHERE LOWER(source) = ?
                """,
                (source,),
            ).fetchone()
            dataset_rows = connection.execute(
                """
                SELECT dataset_name, COUNT(*) AS battery_count
                FROM batteries
                WHERE LOWER(source) = ?
                GROUP BY dataset_name
                ORDER BY battery_count DESC, dataset_name ASC
                """,
                (source,),
            ).fetchall()
            cycle_row = connection.execute(
                """
                SELECT COUNT(*) AS cycle_point_count,
                       MIN(cycle_number) AS min_cycle,
                       MAX(cycle_number) AS max_cycle,
                       AVG(capacity) AS avg_capacity,
                       MIN(capacity) AS min_capacity,
                       MAX(capacity) AS max_capacity,
                       AVG(voltage_mean) AS avg_voltage_mean,
                       MIN(voltage_mean) AS min_voltage_mean,
                       MAX(voltage_mean) AS max_voltage_mean,
                       AVG(current_mean) AS avg_current_mean,
                       MIN(current_mean) AS min_current_mean,
                       MAX(current_mean) AS max_current_mean,
                       AVG(temperature_mean) AS avg_temperature_mean,
                       MIN(temperature_mean) AS min_temperature_mean,
                       MAX(temperature_mean) AS max_temperature_mean,
                       AVG(internal_resistance) AS avg_internal_resistance,
                       MIN(internal_resistance) AS min_internal_resistance,
                       MAX(internal_resistance) AS max_internal_resistance,
                       SUM(CASE WHEN timestamp IS NULL OR timestamp = '' THEN 1 ELSE 0 END) AS missing_timestamp_count,
                       SUM(CASE WHEN internal_resistance IS NULL OR internal_resistance = 0 THEN 1 ELSE 0 END) AS missing_internal_resistance_count
                FROM cycle_points
                WHERE LOWER(source) = ?
                """,
                (source,),
            ).fetchone()
            top_batteries = connection.execute(
                """
                SELECT battery_id, MIN(cycle_number) AS first_cycle, MAX(cycle_number) AS last_cycle, COUNT(*) AS cycle_points
                FROM cycle_points
                WHERE LOWER(source) = ?
                GROUP BY battery_id
                ORDER BY cycle_points DESC, battery_id ASC
                LIMIT 5
                """,
                (source,),
            ).fetchall()
            dataset_file_rows = connection.execute(
                """
                SELECT file_name, file_path, row_count, include_in_training, created_at
                FROM dataset_files
                WHERE LOWER(source) = ?
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (source,),
            ).fetchall()

        feature_ranges = {
            feature: {
                "min": self._as_float(cycle_row[f"min_{feature}"]) if f"min_{feature}" in cycle_row.keys() else None,
                "max": self._as_float(cycle_row[f"max_{feature}"]) if f"max_{feature}" in cycle_row.keys() else None,
                "avg": self._as_float(cycle_row[f"avg_{feature}"]) if f"avg_{feature}" in cycle_row.keys() else None,
            }
            for feature in PROFILE_FEATURES
        }
        demo_files = [item for item in self.get_demo_presets() if item["source"] == source]
        comparison = self._comparison_summary(source)

        return {
            "source": source,
            "battery_count": int(battery_row["battery_count"] or 0),
            "training_candidate_count": int(battery_row["training_candidate_count"] or 0),
            "cycle_point_count": int(cycle_row["cycle_point_count"] or 0),
            "dataset_names": [row["dataset_name"] for row in dataset_rows if row["dataset_name"]],
            "dataset_breakdown": [dict(row) for row in dataset_rows],
            "cycle_window": {
                "min_cycle": int(cycle_row["min_cycle"] or 0),
                "max_cycle": int(cycle_row["max_cycle"] or 0),
            },
            "feature_ranges": feature_ranges,
            "missing_stats": {
                "timestamp": int(cycle_row["missing_timestamp_count"] or 0),
                "internal_resistance": int(cycle_row["missing_internal_resistance_count"] or 0),
            },
            "top_batteries_by_cycles": [dict(row) for row in top_batteries],
            "available_feature_columns": feature_config.get("feature_columns", dataset_summary.get("feature_columns", [])),
            "split": split,
            "num_samples": num_samples,
            "processed_summary_path": str(processed_dir / f"{source}_dataset_summary.json"),
            "comparison_summary_path": str(self.settings.model_dir / source / "comparison_summary.json"),
            "comparison_available": bool(comparison),
            "dataset_files": [
                {
                    **dict(row),
                    "include_in_training": bool(row["include_in_training"]),
                }
                for row in dataset_file_rows
            ],
            "demo_files": demo_files,
            "training_ready": card.training_ready,
            "ingestion_mode": card.ingestion_mode,
            "source_group": card.group,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_demo_presets(self) -> list[dict[str, Any]]:
        presets: list[dict[str, Any]] = []
        root = self.settings.demo_upload_dir
        if not root.exists():
            return presets
        for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
            source = file_path.parent.name.lower() if file_path.parent.name.lower() in SUPPORTED_SOURCES else self._infer_source(file_path)
            scenario = "fault_case" if "fault" in file_path.stem.lower() else "unseen_sample"
            presets.append(
                {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "source": source,
                    "scenario": scenario,
                    "recommended": bool(scenario == "fault_case"),
                    "description": self._describe_demo_preset(file_path.name, scenario, source),
                }
            )
        return presets

    def get_knowledge_summary(self) -> dict[str, Any]:
        payload = self._load_json(self.settings.knowledge_path) or {"symptom_aliases": {}, "faults": []}
        faults = payload.get("faults", [])
        categories = Counter(str(item.get("category", "未分类")) for item in faults)
        severity_distribution = Counter(str(item.get("severity", "unknown")) for item in faults)
        source_coverage = Counter(
            source
            for item in faults
            for source in (item.get("source_scope") or ["generic"])
        )
        evidence_sources = Counter(
            source
            for item in faults
            for source in (item.get("evidence_source") or [])
        )
        symptom_counter = Counter(
            symptom
            for item in faults
            for symptom in item.get("symptoms", [])
        )
        return {
            "fault_count": len(faults),
            "symptom_alias_count": len(payload.get("symptom_aliases", {})),
            "categories": dict(categories),
            "severity_distribution": dict(severity_distribution),
            "source_coverage": dict(source_coverage),
            "evidence_sources": evidence_sources.most_common(8),
            "rule_count": sum(1 for item in faults if item.get("rule_id")),
            "threshold_rule_count": sum(1 for item in faults if item.get("threshold_hints")),
            "fault_names": [item.get("name") for item in faults],
            "top_symptoms": symptom_counter.most_common(8),
            "graph_backend": self.settings.graph_backend,
            "knowledge_path": str(self.settings.knowledge_path),
            "coverage_notes": [
                "知识库用于 GraphRAG 候选故障排序、根因链组织与处理建议生成。",
                "当前候选故障会显式展示规则编号、证据来源、适用数据源和排序依据。",
                "当前结果展示的是可审计证据链，而不是模型隐式思维过程。",
                "如需扩展论文深度，可继续补充故障类型、症状别名、阈值与规则来源。",
            ],
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_system_status(self) -> dict[str, Any]:
        warnings: list[str] = []
        source_statuses: list[dict[str, Any]] = []
        demo_presets = self.get_demo_presets()
        for source in SUPPORTED_SOURCES:
            card = get_dataset_card(source)
            profile = self.get_dataset_profile(source)
            comparison = self._comparison_summary(source)
            raw_dir = getattr(self.settings, f"raw_{source}_dir")
            raw_files = sorted(item for item in raw_dir.glob("**/*") if item.is_file()) if raw_dir.exists() else []
            if card.training_ready:
                model_risks = self._comparison_risks(comparison)
            else:
                model_risks = []
            warnings.extend(f"{source.upper()}: {item}" for item in model_risks)
            if card.ingestion_mode == "enhancement_assets":
                processed_ready = bool(self._load_json(self.settings.processed_dir / source / f"{source}_dataset_summary.json"))
            else:
                processed_ready = bool(profile["split"]) or bool(profile["battery_count"])
            source_statuses.append(
                {
                    "source": source,
                    "raw_file_count": len(raw_files),
                    "battery_count": profile["battery_count"],
                    "training_candidate_count": profile["training_candidate_count"],
                    "processed_ready": processed_ready,
                    "comparison_ready": profile["comparison_available"] if card.training_ready else False,
                    "demo_preset_count": len([item for item in demo_presets if item["source"] == source]),
                    "best_model": self._best_model_name(comparison) if card.training_ready else None,
                    "training_ready": card.training_ready,
                    "ingestion_mode": card.ingestion_mode,
                    "source_group": card.group,
                }
            )
        if not self.settings.knowledge_path.exists():
            warnings.append("知识库文件缺失，GraphRAG 将无法构建完整候选故障链。")
        if self.settings.graph_backend == "neo4j":
            warnings.append("当前默认图谱后端为 Neo4j，答辩时请确保数据库已启动并执行初始化脚本。")
        return {
            "app_name": self.settings.app_name,
            "api_prefix": self.settings.api_prefix,
            "graph_backend": self.settings.graph_backend,
            "database_path": str(self.settings.database_path),
            "database_ready": self.settings.database_path.exists(),
            "knowledge_ready": self.settings.knowledge_path.exists(),
            "demo_preset_count": len(demo_presets),
            "source_statuses": source_statuses,
            "demo_acceptance_flow": [
                "上传未见样本",
                "立即执行全生命周期预测",
                "触发异常检测与机理解释",
                "导出 lifecycle / mechanism 报告",
                "进入分析中心查看 benchmark 与案例闭环",
            ],
            "warnings": list(dict.fromkeys(warnings)),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_case_bundle(self, battery_id: str) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        prediction = next(iter(self.repository.list_predictions(battery_id, limit=1)), None)
        diagnosis = next(iter(self.repository.list_diagnoses(battery_id, limit=1)), None)
        anomalies = self.repository.list_anomalies(battery_id, limit=10)
        cycles = self.repository.get_cycle_points(battery_id, limit=240)
        dataset_profile = self.get_dataset_profile(str(battery.get("source", "unknown")))
        comparison = self._comparison_summary(str(battery.get("source", "unknown")))
        experiment_detail = self.training_service.get_experiment_detail(str(battery.get("source", "unknown")))
        dataset_position = self._dataset_position(battery, dataset_profile)
        last_export = self._latest_case_export(battery_id)
        chart_artifacts = ((last_export or {}).get("manifest") or {}).get("charts", [])
        bundle_markdown = self._build_case_bundle_markdown(
            battery=battery,
            cycles=cycles,
            prediction=prediction,
            diagnosis=diagnosis,
            anomalies=anomalies,
            dataset_profile=dataset_profile,
            comparison=comparison,
            dataset_position=dataset_position,
        )
        return {
            "battery_id": battery_id,
            "source": battery.get("source"),
            "dataset_name": battery.get("dataset_name"),
            "health_score": battery.get("health_score"),
            "cycle_count": battery.get("cycle_count"),
            "prediction": prediction,
            "diagnosis": diagnosis,
            "anomalies": anomalies,
            "dataset_profile": dataset_profile,
            "dataset_position": dataset_position,
            "export_ready": bool(prediction and diagnosis and comparison),
            "last_export": last_export,
            "chart_artifacts": chart_artifacts,
            "artifacts": [
                {
                    "key": "sample-profile",
                    "title": "样本画像",
                    "available": True,
                    "description": f"{battery.get('source', 'unknown').upper()} 来源样本，当前共 {battery.get('cycle_count', 0)} 个循环点。",
                },
                {
                    "key": "lifecycle-prediction",
                    "title": "生命周期预测证据链",
                    "available": prediction is not None,
                    "description": "包含 trajectory、knee/EOL/RUL、风险窗口与模型证据。",
                },
                {
                    "key": "mechanism-report",
                    "title": "机理解释与 GraphRAG 证据链",
                    "available": diagnosis is not None,
                    "description": "包含候选机理、根因链、建议、未来风险窗口与图谱子图说明。",
                },
                {
                    "key": "benchmark-context",
                    "title": "benchmark 背景",
                    "available": bool(comparison),
                    "description": "记录该来源当前可用的 lifecycle 模型对比结果与学术风险提示。",
                },
                {
                    "key": "case-export",
                    "title": "目录化 lifecycle 案例导出",
                    "available": last_export is not None,
                    "description": "生成论文附录可复用的 Markdown、JSON 与生命周期 / GraphRAG / benchmark 图表目录。",
                },
            ],
            "recommended_story": [
                f"先说明样本来自 {str(battery.get('source', 'unknown')).upper()} 的 {battery.get('dataset_name') or '--'}，当前位于 {dataset_position['split_name']} 划分。",
                "再展示观测轨迹、未来 trajectory 与 knee/EOL/RUL 关键节点，并指向导出的生命周期图表。",
                "随后说明未来风险窗口如何映射为候选机理、根因链与 GraphRAG 证据子图。",
                "最后补充该来源的 benchmark 结果、论文图表与当前模型局限。",
            ],
            "bundle_markdown": bundle_markdown,
            "experiment_context": {
                "comparison": comparison,
                "detail_headline": experiment_detail.get("headline"),
                "warnings": experiment_detail.get("warnings", []),
                "plots": experiment_detail.get("plots", []),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    def export_case_bundle(self, battery_id: str, ensure_artifacts: bool = True) -> dict[str, Any]:
        generated_artifacts = self._ensure_case_artifacts(battery_id, ensure_artifacts=ensure_artifacts)
        bundle = self.get_case_bundle(battery_id)
        export_dir = self._case_export_root(battery_id) / datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        charts_dir = export_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        battery = self.repository.get_battery(battery_id)
        cycles = self.repository.get_cycle_points(battery_id, limit=240)
        prediction = bundle.get("prediction")
        diagnosis = bundle.get("diagnosis")
        experiment_context = bundle.get("experiment_context", {})

        prediction_report = prediction.get("report_markdown") if prediction else "# 全生命周期预测报告\n\n- 暂无生命周期预测记录"
        diagnosis_report = diagnosis.get("report_markdown") if diagnosis else "# 机理解释报告\n\n- 暂无机理解释记录"
        manifest = {
            "battery_id": battery_id,
            "source": battery.get("source") if battery else None,
            "generated_at": datetime.utcnow().isoformat(),
            "generated_artifacts": generated_artifacts,
            "export_dir": str(export_dir),
            "asset_status": {
                "lifecycle_prediction_ready": prediction is not None,
                "mechanism_explanation_ready": diagnosis is not None,
                "benchmark_context_ready": bool(experiment_context.get("comparison")),
            },
        }

        file_map = {
            "case_bundle.md": bundle["bundle_markdown"],
            "lifecycle_prediction_report.md": prediction_report,
            "mechanism_report.md": diagnosis_report,
        }
        for file_name, content in file_map.items():
            (export_dir / file_name).write_text(str(content), encoding="utf-8")

        sample_profile = {
            "battery": battery,
            "cycles": cycles,
            "prediction": prediction,
            "diagnosis": diagnosis,
            "anomalies": bundle.get("anomalies", []),
            "dataset_position": bundle.get("dataset_position", {}),
        }
        (export_dir / "sample_profile.json").write_text(json.dumps(sample_profile, ensure_ascii=False, indent=2), encoding="utf-8")
        (export_dir / "dataset_profile.json").write_text(json.dumps(bundle["dataset_profile"], ensure_ascii=False, indent=2), encoding="utf-8")
        (export_dir / "experiment_context.json").write_text(json.dumps(experiment_context, ensure_ascii=False, indent=2), encoding="utf-8")

        chart_entries = [
            self._write_lifecycle_trajectory_chart(charts_dir / "lifecycle_trajectory.png", prediction, cycles),
            self._write_diagnosis_graph_chart(charts_dir / "graphrag_evidence.png", diagnosis),
            self._write_experiment_summary_chart(charts_dir / "benchmark_summary.png", battery.get("source") if battery else None, experiment_context),
        ]
        manifest["charts"] = chart_entries
        manifest["files"] = [
            {"path": str(export_dir / "manifest.json"), "kind": "manifest"},
            {"path": str(export_dir / "case_bundle.md"), "kind": "case_bundle"},
            {"path": str(export_dir / "lifecycle_prediction_report.md"), "kind": "lifecycle_prediction_report"},
            {"path": str(export_dir / "mechanism_report.md"), "kind": "mechanism_report"},
            {"path": str(export_dir / "sample_profile.json"), "kind": "sample_profile"},
            {"path": str(export_dir / "dataset_profile.json"), "kind": "dataset_profile"},
            {"path": str(export_dir / "experiment_context.json"), "kind": "experiment_context"},
            *chart_entries,
        ]
        (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "export_dir": str(export_dir),
            "files": manifest["files"],
            "generated_artifacts": generated_artifacts,
            "bundle_snapshot": self.get_case_bundle(battery_id),
        }

    def _build_case_bundle_markdown(
        self,
        *,
        battery: dict[str, Any],
        cycles: list[dict[str, Any]],
        prediction: Optional[dict[str, Any]],
        diagnosis: Optional[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        dataset_profile: dict[str, Any],
        comparison: dict[str, Any],
        dataset_position: dict[str, Any],
    ) -> str:
        latest_capacity = battery.get("latest_capacity")
        prediction_payload = (prediction or {}).get("payload") or {}
        future_risks = prediction_payload.get("future_risks") or {}
        risk_windows = prediction_payload.get("risk_windows") or []
        explanation = (prediction or {}).get("explanation") or {}
        lines = [
            "# BHMS 案例包",
            "",
            "## 一、样本概况",
            f"- 电池 ID：{battery['battery_id']}",
            f"- 数据来源：{str(battery.get('source', 'unknown')).upper()}",
            f"- 数据集：{battery.get('dataset_name') or '--'}",
            f"- 循环次数：{battery.get('cycle_count', 0)}",
            f"- 当前健康分：{float(battery.get('health_score', 0.0)):.2f}",
            f"- 当前容量：{float(latest_capacity):.4f}Ah" if latest_capacity is not None else "- 当前容量：--",
            "",
            "## 二、数据画像",
            f"- 当前来源样本数：{dataset_profile.get('battery_count', 0)}",
            f"- 当前来源周期点：{dataset_profile.get('cycle_point_count', 0)}",
            f"- 当前来源训练候选：{dataset_profile.get('training_candidate_count', 0)}",
            f"- 当前来源数据集：{', '.join(dataset_profile.get('dataset_names', [])) or '--'}",
            f"- 当前样本划分：{dataset_position.get('split_name', '--')}",
            f"- 当前样本训练池状态：{'已加入' if dataset_position.get('include_in_training') else '未加入'}",
            "",
            "## 三、历史轨迹摘要",
            f"- 已读取 cycle 点：{len(cycles)}",
        ]
        if cycles:
            first_point = cycles[0]
            last_point = cycles[-1]
            lines.extend(
                [
                    f"- 首个 cycle：{first_point.get('cycle_number')}，容量 {float(first_point.get('capacity', 0.0)):.4f}Ah",
                    f"- 最新 cycle：{last_point.get('cycle_number')}，容量 {float(last_point.get('capacity', 0.0)):.4f}Ah",
                ]
            )
        lines.extend(["", "## 四、未来 trajectory 摘要"])
        if prediction:
            lines.extend(
                [
                    f"- 模型：{prediction.get('model_name', '--')}",
                    f"- 预测 RUL：{float(prediction.get('predicted_rul', 0.0)):.2f} cycles",
                    f"- 预测 knee 周期：{prediction.get('predicted_knee_cycle', '--')}",
                    f"- 预测 EOL 周期：{prediction.get('predicted_eol_cycle', '--')}",
                    f"- 置信度：{float(prediction.get('confidence', 0.0)) * 100:.1f}%",
                    f"- Checkpoint：{prediction.get('checkpoint_id', '--')}",
                ]
            )
            trajectory = prediction.get("trajectory") or []
            if trajectory:
                lines.append(
                    f"- 未来 trajectory 覆盖 {trajectory[0].get('cycle', '--')} -> {trajectory[-1].get('cycle', '--')} cycles，共 {len(trajectory)} 个点。"
                )
        else:
            lines.append("- 尚未生成生命周期预测记录")
        lines.extend(["", "## 五、风险窗口与机理线索"])
        if future_risks:
            lines.extend(
                [
                    f"- 衰退模式：{future_risks.get('future_capacity_fade_pattern', '--')}",
                    f"- 温度风险：{future_risks.get('temperature_risk', '--')}",
                    f"- 内阻风险：{future_risks.get('resistance_risk', '--')}",
                    f"- 电压风险：{future_risks.get('voltage_risk', '--')}",
                ]
            )
        else:
            lines.append("- 当前尚无未来风险摘要")
        if risk_windows:
            for item in risk_windows[:4]:
                lines.append(
                    f"- 风险窗口 {item.get('label', '--')}: {item.get('start_cycle', '--')} -> {item.get('end_cycle', '--')} ({item.get('severity', '--')})"
                )
        if explanation.get("feature_contributions"):
            for item in list(explanation.get("feature_contributions") or [])[:4]:
                lines.append(f"- 关键特征 {item.get('feature')}: {item.get('description')}")
        lines.extend(["", "## 六、异常与机理解释"])
        if anomalies:
            lines.extend(f"- 异常症状：{item.get('symptom')} / {item.get('description')}" for item in anomalies[:6])
        else:
            lines.append("- 当前无异常事件记录")
        if diagnosis:
            lines.extend(
                [
                    f"- 诊断结论：{diagnosis.get('fault_type', '--')}",
                    f"- 诊断置信度：{float(diagnosis.get('confidence', 0.0)) * 100:.1f}%",
                ]
            )
            for item in list(diagnosis.get("root_causes") or [])[:4]:
                lines.append(f"- 根因：{item}")
            for item in list(diagnosis.get("recommendations") or [])[:4]:
                lines.append(f"- 建议：{item}")
            for item in list(diagnosis.get("decision_basis") or [])[:3]:
                lines.append(f"- 排序依据：{item}")
        else:
            lines.append("- 尚未生成机理解释 / GraphRAG 诊断记录")
        lines.extend(["", "## 七、benchmark 背景"])
        best_models = (comparison or {}).get("best_models") or {}
        if best_models.get("within_source"):
            lines.append(f"- within-source 当前占优模型：{best_models.get('within_source')}")
        if best_models.get("transfer"):
            lines.append(f"- transfer 当前占优模型：{best_models.get('transfer')}")
        if not best_models:
            lines.append("- 当前来源尚未生成可用的模型对比摘要")
        paper_gate = (comparison or {}).get("paper_gate") or {}
        if paper_gate:
            lines.append(
                f"- 论文门槛：{'通过' if paper_gate.get('passed') else '未通过'}；failing_units={paper_gate.get('failing_units', [])}"
            )
        for risk in self._comparison_risks(comparison)[:4]:
            lines.append(f"- 实验风险：{risk}")
        lines.extend(
            [
                "",
                "## 八、建议与答辩讲解",
                "- 先用样本概况回答“数据来自哪里”。",
                "- 再用 trajectory + knee/EOL/RUL 回答“未来如何衰退、还能用多久”。",
                "- 用风险窗口与机理解释回答“为什么会变差、应该怎么处理”。",
                "- 最后用 benchmark 背景说明“为什么当前模型设计值得做，但还有哪些局限”。",
            ]
        )
        return "\n".join(lines)

    def _ensure_case_artifacts(self, battery_id: str, *, ensure_artifacts: bool) -> dict[str, Any]:
        generated = {"prediction_generated": False, "diagnosis_generated": False, "anomaly_generated": False}
        if not ensure_artifacts:
            return generated
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        prediction = next(iter(self.repository.list_predictions(battery_id, limit=1)), None)
        if prediction is None:
            self.prediction_service.predict_lifecycle(
                battery_id=battery_id,
                model_name="hybrid",
                seq_len=self.settings.default_seq_len,
            )
            generated["prediction_generated"] = True
        diagnosis = next(iter(self.repository.list_diagnoses(battery_id, limit=1)), None)
        if diagnosis is None:
            anomaly_result = self.prediction_service.detect_anomaly(battery_id=battery_id)
            generated["anomaly_generated"] = True
            self.prediction_service.explain_mechanism(
                battery_id=battery_id,
                anomalies=anomaly_result.get("events", []),
                battery_info=battery,
            )
            generated["diagnosis_generated"] = True
        return generated

    def _case_export_root(self, battery_id: str) -> Path:
        safe_battery_id = battery_id.replace("/", "_")
        return self.settings.data_dir / "exports" / "cases" / safe_battery_id

    def _dataset_position(self, battery: dict[str, Any], dataset_profile: dict[str, Any]) -> dict[str, Any]:
        battery_key = str(battery.get("canonical_battery_id") or battery.get("battery_id"))
        split = dataset_profile.get("split", {})
        split_name = "unassigned"
        for key, items in (
            ("train", split.get("train_batteries", [])),
            ("val", split.get("val_batteries", [])),
            ("test", split.get("test_batteries", [])),
        ):
            if battery_key in items:
                split_name = key
                break
        return {
            "canonical_battery_id": battery_key,
            "split_name": split_name,
            "include_in_training": bool(battery.get("include_in_training")),
            "dataset_name": battery.get("dataset_name"),
            "source": battery.get("source"),
        }

    def _latest_case_export(self, battery_id: str) -> Optional[dict[str, Any]]:
        root = self._case_export_root(battery_id)
        if not root.exists():
            return None
        export_dirs = sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)
        for export_dir in export_dirs:
            manifest_path = export_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = self._load_json(manifest_path)
            if manifest is None:
                continue
            return {
                "export_dir": str(export_dir),
                "manifest": manifest,
                "generated_at": manifest.get("generated_at"),
                "files": manifest.get("files", []),
            }
        return None

    def _write_lifecycle_trajectory_chart(
        self,
        output_path: Path,
        prediction: Optional[dict[str, Any]],
        cycles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not MATPLOTLIB_AVAILABLE:
            write_placeholder_png(output_path)
            return {"path": str(output_path), "kind": "chart", "key": "lifecycle_trajectory"}
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if prediction and prediction.get("projection"):
            projection = prediction["projection"]
            actual = projection.get("actual_points", [])
            forecast = projection.get("forecast_points", [])
            if actual:
                ax.plot([item["cycle"] for item in actual], [item["capacity"] for item in actual], label="Observed", color="#1f77b4")
            if forecast:
                ax.plot([item["cycle"] for item in forecast], [item["capacity"] for item in forecast], label="Future trajectory", color="#ff7f0e")
            band = projection.get("confidence_band", [])
            if band:
                x = [item["cycle"] for item in band]
                lower = [item["lower"] for item in band]
                upper = [item["upper"] for item in band]
                ax.fill_between(x, lower, upper, color="#ff7f0e", alpha=0.15, label="Confidence band")
            if prediction.get("predicted_knee_cycle") is not None:
                ax.axvline(float(prediction["predicted_knee_cycle"]), color="#34c759", linestyle="--", linewidth=1.6, label="knee")
            if prediction.get("predicted_eol_cycle") is not None:
                ax.axvline(float(prediction["predicted_eol_cycle"]), color="#d62728", linestyle=":", linewidth=1.8, label="EOL")
        elif cycles:
            ax.plot([item["cycle_number"] for item in cycles], [item["capacity"] for item in cycles], label="Capacity", color="#1f77b4")
        else:
            ax.text(0.5, 0.5, "No lifecycle data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Lifecycle trajectory")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity")
        ax.grid(linestyle="--", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return {"path": str(output_path), "kind": "chart", "key": "lifecycle_trajectory"}

    def _write_diagnosis_graph_chart(self, output_path: Path, diagnosis: Optional[dict[str, Any]]) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not MATPLOTLIB_AVAILABLE:
            write_placeholder_png(output_path)
            return {"path": str(output_path), "kind": "chart", "key": "graphrag_evidence"}
        fig, ax = plt.subplots(figsize=(8, 5))
        trace = (diagnosis or {}).get("graph_trace") or {}
        nodes = trace.get("nodes", [])
        edges = trace.get("edges", [])
        if not nodes:
            ax.text(0.5, 0.5, "No GraphRAG evidence", ha="center", va="center", transform=ax.transAxes)
        else:
            angles = np.linspace(0, 2 * np.pi, num=len(nodes), endpoint=False)
            positions = {
                node["id"]: (float(np.cos(angle)), float(np.sin(angle)))
                for node, angle in zip(nodes, angles)
            }
            for edge in edges:
                source = positions.get(edge.get("source"))
                target = positions.get(edge.get("target"))
                if source and target:
                    ax.plot([source[0], target[0]], [source[1], target[1]], color="#9ecae1", alpha=0.8)
            for node in nodes:
                x, y = positions[node["id"]]
                ax.scatter([x], [y], s=900, color="#4c78a8", alpha=0.85)
                ax.text(x, y, str(node.get("label", "--")), ha="center", va="center", color="white", fontsize=8)
        ax.set_title("GraphRAG evidence")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return {"path": str(output_path), "kind": "chart", "key": "graphrag_evidence"}

    def _write_experiment_summary_chart(self, output_path: Path, source: Optional[str], experiment_context: dict[str, Any]) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if source:
            source_plot = self.settings.model_dir / source / "plots" / "experiment_summary.png"
            if source_plot.exists():
                shutil.copy2(source_plot, output_path)
                return {"path": str(output_path), "kind": "chart", "key": "benchmark_summary"}
        if not MATPLOTLIB_AVAILABLE:
            write_placeholder_png(output_path)
            return {"path": str(output_path), "kind": "chart", "key": "benchmark_summary"}
        comparison = experiment_context.get("comparison", {}) or {}
        benchmark_units = comparison.get("benchmark_units", [])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        units = [
            item
            for item in benchmark_units
            if isinstance(item, dict) and isinstance(item.get("models"), dict) and item.get("models")
        ]
        model_order = ("hybrid", "bilstm")
        palette = {"hybrid": "#1f77b4", "bilstm": "#ff7f0e"}
        if units:
            labels = [str(item.get("label") or item.get("key") or "benchmark") for item in units]
            x = np.arange(len(labels))
            width = 0.34
            for index, model_name in enumerate(model_order):
                offsets = x + (index - 0.5) * width
                values: list[float] = []
                for item in units:
                    metrics = ((item.get("models") or {}).get(model_name) or {}).get("metrics") or {}
                    rmse = metrics.get("rmse")
                    values.append(float(rmse) if isinstance(rmse, (int, float)) else 0.0)
                ax.bar(offsets, values, width=width, label=model_name.upper(), color=palette.get(model_name, "#9ca3af"))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No benchmark summary", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Benchmark summary")
        ax.set_ylabel("Trajectory RMSE")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return {"path": str(output_path), "kind": "chart", "key": "benchmark_summary"}

    def _validate_source(self, source: str) -> str:
        normalized = source.lower()
        if normalized not in SUPPORTED_SOURCES:
            raise BHMSException(f"暂不支持的数据源: {source}", status_code=400, code="unsupported_source")
        return normalized

    @staticmethod
    def _load_json(path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _comparison_summary(self, source: str) -> dict[str, Any]:
        comparison = self.training_service.get_comparison(source)
        current = comparison.get("current")
        return current if isinstance(current, dict) else {}

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_source(path: Path) -> str:
        name = path.name.lower()
        for source in SUPPORTED_SOURCES:
            if source in name:
                return source
        return "kaggle"

    @staticmethod
    def _describe_demo_preset(file_name: str, scenario: str, source: str) -> str:
        if scenario == "fault_case":
            return f"推荐用于展示 {source.upper()} 新样本的异常检测、机理解释、GraphRAG 检索与完整案例导出。"
        return f"推荐用于展示 {source.upper()} 未见样本的导入、生命周期预测与分析闭环。"

    @staticmethod
    def _best_model_name(comparison: dict[str, Any]) -> Optional[str]:
        best_models = (comparison or {}).get("best_models") or {}
        return best_models.get("within_source") or best_models.get("transfer")

    @staticmethod
    def _comparison_risks(comparison: dict[str, Any]) -> list[str]:
        risks = list((comparison or {}).get("warnings") or [])
        paper_gate = (comparison or {}).get("paper_gate") or {}
        if paper_gate.get("failing_units"):
            risks.append(f"论文门槛未通过：{paper_gate.get('failing_units')}。")
        ablation_gate = (comparison or {}).get("ablation_gate") or {}
        if ablation_gate.get("available") and not ablation_gate.get("passed"):
            risks.append("Hybrid 消融门槛未通过，说明结构优势证据仍不稳定。")
        return list(dict.fromkeys(risks))


__all__ = ["InsightService"]
