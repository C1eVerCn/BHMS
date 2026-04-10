"""RUL 推理服务，支持来源级训练权重、可解释性输出与启发式 fallback。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ml.data.nasa_preprocessor import DEFAULT_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AttentionHeatmap:
    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    disclaimer: str = "注意力热力图仅作为辅助参考，不直接等于因果解释。"


@dataclass
class FeatureContribution:
    feature: str
    impact: float
    direction: str
    description: str


@dataclass
class WindowContribution:
    window_label: str
    start_cycle: float
    end_cycle: float
    impact: float
    description: str


@dataclass
class PredictionExplanation:
    input_summary: dict[str, object]
    model_info: dict[str, object]
    feature_contributions: list[FeatureContribution]
    window_contributions: list[WindowContribution]
    confidence_summary: dict[str, object]
    attention_heatmap: Optional[AttentionHeatmap] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_summary": self.input_summary,
            "model_info": self.model_info,
            "feature_contributions": [asdict(item) for item in self.feature_contributions],
            "window_contributions": [asdict(item) for item in self.window_contributions],
            "confidence_summary": self.confidence_summary,
            "attention_heatmap": asdict(self.attention_heatmap) if self.attention_heatmap else None,
        }


@dataclass
class PredictionOutput:
    predicted_rul: float
    confidence: float
    model_name: str
    model_version: str
    model_source: str
    checkpoint_id: Optional[str] = None
    fallback_used: bool = False
    explanation: Optional[PredictionExplanation] = None


@dataclass
class LifecyclePredictionOutput:
    predicted_rul: float
    predicted_knee_cycle: Optional[float]
    predicted_eol_cycle: Optional[float]
    confidence: float
    model_name: str
    model_version: str
    model_source: str
    checkpoint_id: Optional[str] = None
    fallback_used: bool = False
    trajectory: list[dict[str, float]] = field(default_factory=list)
    projection: dict[str, object] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    explanation: Optional[PredictionExplanation] = None


class RULInferenceService:
    def __init__(self, model_dir: str | Path = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, object] = {}

    def predict(
        self,
        sequence: Optional[np.ndarray],
        source: str,
        model_name: str = "hybrid",
        feature_cols: Optional[Sequence[str]] = None,
        points: Optional[Sequence[dict[str, float]]] = None,
    ) -> PredictionOutput:
        source = source.lower()
        requested_model = model_name
        for candidate in self._preferred_models(source, requested_model):
            try:
                output = self._predict_with_checkpoint(
                    sequence=sequence,
                    points=points,
                    source=source,
                    model_name=candidate,
                    feature_cols=feature_cols,
                )
                output.fallback_used = candidate != requested_model
                return output
            except Exception:
                continue
        return self._heuristic_predict(sequence=sequence, points=points, source=source, feature_cols=feature_cols)

    def _preferred_models(self, source: str, requested: str) -> list[str]:
        _ = source
        if requested == "auto":
            return ["hybrid", "bilstm"]
        ordered = [requested]
        if requested != "hybrid":
            ordered.append("hybrid")
        if "bilstm" not in ordered:
            ordered.append("bilstm")
        return ordered

    def _predict_with_checkpoint(
        self,
        sequence: Optional[np.ndarray],
        points: Optional[Sequence[dict[str, float]]],
        source: str,
        model_name: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> PredictionOutput:
        import torch
        from ml.models.baseline import BiLSTMConfig, BiLSTMRULPredictor
        from ml.models.hybrid import RULPredictor, RULPredictorConfig

        checkpoint_path = self._resolve_checkpoint(source=source, model_name=model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runtime_key = f"{source}:{model_name}:{checkpoint_path}"
        runtime = self._cache.get(runtime_key)
        if runtime is None:
            model_type = checkpoint["model_type"]
            config = checkpoint["model_config"]
            if model_type == "hybrid":
                model = RULPredictor(RULPredictorConfig(**config))
            elif model_type == "bilstm":
                model = BiLSTMRULPredictor(BiLSTMConfig(**config))
            else:
                raise ValueError(f"未知模型类型: {model_type}")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            runtime = model
            self._cache[runtime_key] = runtime
        model = runtime

        feature_names = list(checkpoint.get("feature_columns") or feature_cols or DEFAULT_FEATURE_COLUMNS)
        raw_sequence = self._build_sequence_from_inputs(points=points, sequence=sequence, feature_cols=feature_names)
        normalized_sequence = self._normalize_sequence(raw_sequence, checkpoint.get("normalization", {}), feature_names)
        tensor = torch.tensor(normalized_sequence[None, :, :], dtype=torch.float32)

        with torch.no_grad():
            prediction, features = model.forward(tensor, return_features=True)
            raw_prediction = max(0.0, float(prediction.squeeze().item()))

        raw_cycles = self._extract_series(points, feature_names, raw_sequence, "cycle_number")
        raw_capacities = self._extract_series(points, feature_names, raw_sequence, "capacity")
        trend_proxy = self._heuristic_rul_from_sequence(raw_sequence, feature_names)
        calibration = self._calibrate_rul_prediction(
            raw_prediction=raw_prediction,
            trend_proxy_rul=trend_proxy["predicted_rul"],
            latest_capacity=trend_proxy["latest_capacity"],
            eol_capacity=trend_proxy["eol_capacity"],
            initial_capacity=trend_proxy["initial_capacity"],
        )
        predicted_rul = calibration["predicted_rul"]
        confidence = self._estimate_confidence(
            predicted_rul=predicted_rul,
            model_name=model_name,
            seq_len=raw_sequence.shape[0],
            capacities=raw_capacities,
            fallback_used=False,
        )
        explanation = self._build_explanation(
            model=model,
            tensor=tensor,
            raw_sequence=raw_sequence,
            feature_names=feature_names,
            prediction=predicted_rul,
            confidence=confidence,
            source=source,
            model_name=model_name,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            features=features,
            raw_cycles=raw_cycles,
            raw_capacities=raw_capacities,
            raw_model_prediction=raw_prediction,
            trend_proxy=trend_proxy,
            calibration=calibration,
        )
        return PredictionOutput(
            predicted_rul=predicted_rul,
            confidence=confidence,
            model_name=model_name,
            model_version=str(checkpoint.get("model_version", checkpoint.get("epoch", "trained"))),
            model_source=source.lower(),
            checkpoint_id=checkpoint_path.name,
            fallback_used=False,
            explanation=explanation,
        )

    def _heuristic_predict(
        self,
        sequence: Optional[np.ndarray],
        points: Optional[Sequence[dict[str, float]]],
        source: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> PredictionOutput:
        columns = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        raw_sequence = self._build_sequence_from_inputs(points=points, sequence=sequence, feature_cols=columns)
        trend_proxy = self._heuristic_rul_from_sequence(raw_sequence, columns)
        predicted_rul = trend_proxy["predicted_rul"]
        raw_cycles = self._extract_series(points, columns, raw_sequence, "cycle_number")
        raw_capacities = self._extract_series(points, columns, raw_sequence, "capacity")
        confidence = self._estimate_confidence(
            predicted_rul=float(predicted_rul),
            model_name="heuristic",
            seq_len=raw_sequence.shape[0],
            capacities=raw_capacities,
            fallback_used=True,
        )
        explanation = PredictionExplanation(
            input_summary=self._build_input_summary(raw_sequence, columns, raw_cycles, raw_capacities),
            model_info={
                "model_name": "heuristic",
                "model_source": source.lower(),
                "checkpoint_id": None,
                "fallback_used": True,
                "trend_proxy_rul": round(float(predicted_rul), 2),
                "note": "当前未找到可用训练权重，系统退回到基于容量衰减趋势的启发式估计。",
            },
            feature_contributions=[
                FeatureContribution(
                    feature="capacity",
                    impact=0.82,
                    direction="decrease",
                    description="启发式预测主要依赖容量衰减趋势与当前容量水平。",
                )
            ],
            window_contributions=[
                WindowContribution(
                    window_label="最近窗口",
                    start_cycle=float(raw_cycles[max(0, len(raw_cycles) - 5)]) if len(raw_cycles) else 0.0,
                    end_cycle=float(raw_cycles[-1]) if len(raw_cycles) else 0.0,
                    impact=0.78,
                    description="最近几个 cycle 的容量变化决定了启发式投影斜率。",
                )
            ]
            if len(raw_cycles)
            else [],
            confidence_summary={
                "score": confidence,
                "factors": [
                    "输入窗口越长，启发式估计越稳定",
                    "当前结果未使用训练模型，因此置信度低于 Hybrid/Bi-LSTM",
                    "容量曲线若波动较大，可信度会下降",
                ],
            },
            attention_heatmap=None,
        )
        return PredictionOutput(
            predicted_rul=float(predicted_rul),
            confidence=confidence,
            model_name="heuristic",
            model_version="heuristic-v1",
            model_source=source.lower(),
            checkpoint_id=None,
            fallback_used=True,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        *,
        model,
        tensor,
        raw_sequence: np.ndarray,
        feature_names: list[str],
        prediction: float,
        confidence: float,
        source: str,
        model_name: str,
        checkpoint: dict,
        checkpoint_path: Path,
        features: Optional[dict],
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
        raw_model_prediction: float,
        trend_proxy: dict[str, float | str],
        calibration: dict[str, object],
    ) -> PredictionExplanation:
        feature_contributions = self._feature_importance(model, tensor, feature_names, prediction)
        window_contributions = self._window_importance(model, tensor, raw_cycles, prediction)
        attention_heatmap = self._attention_heatmap(features, raw_cycles)
        confidence_factors = [
            f"输入窗口长度为 {tensor.shape[1]} 个 cycle",
            f"当前模型为 {model_name}，来源 {source}",
            "本次预测直接使用训练权重推理",
            f"轨迹稳定度评分为 {self._trajectory_stability(raw_capacities):.3f}",
        ]
        if calibration.get("applied"):
            confidence_factors.append(
                f"模型原始 RUL 为 {raw_model_prediction:.2f}，趋势代理为 {float(trend_proxy['predicted_rul']):.2f}，已执行一致性校准。"
            )
        confidence_summary = {
            "score": confidence,
            "factors": confidence_factors,
        }
        return PredictionExplanation(
            input_summary=self._build_input_summary(raw_sequence, feature_names, raw_cycles, raw_capacities),
            model_info={
                "model_name": model_name,
                "model_source": source.lower(),
                "checkpoint_id": checkpoint_path.name,
                "model_version": str(checkpoint.get("model_version", checkpoint.get("epoch", "trained"))),
                "fallback_used": False,
                "raw_model_rul": round(float(raw_model_prediction), 2),
                "trend_proxy_rul": round(float(trend_proxy["predicted_rul"]), 2),
                "trend_proxy_method": str(trend_proxy["method"]),
                "calibration_applied": bool(calibration.get("applied")),
                "calibrated_rul": round(float(prediction), 2),
            },
            feature_contributions=feature_contributions,
            window_contributions=window_contributions,
            confidence_summary=confidence_summary,
            attention_heatmap=attention_heatmap,
        )

    def _build_input_summary(
        self,
        raw_sequence: np.ndarray,
        feature_names: list[str],
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
    ) -> dict[str, object]:
        voltage_series = self._safe_column(raw_sequence, feature_names, "voltage_mean")
        temp_series = self._safe_column(raw_sequence, feature_names, "temperature_mean")
        resistance_series = self._safe_column(raw_sequence, feature_names, "internal_resistance")
        recent_delta = 0.0
        if raw_capacities.size >= 2:
            recent_delta = float(raw_capacities[-1] - raw_capacities[max(0, raw_capacities.size - 5)])
        return {
            "seq_len": int(raw_sequence.shape[0]),
            "cycle_range": [float(raw_cycles[0]) if raw_cycles.size else 0.0, float(raw_cycles[-1]) if raw_cycles.size else 0.0],
            "capacity_range": [float(np.min(raw_capacities)) if raw_capacities.size else 0.0, float(np.max(raw_capacities)) if raw_capacities.size else 0.0],
            "recent_capacity_delta": recent_delta,
            "temperature_mean": float(np.mean(temp_series)) if temp_series.size else 0.0,
            "voltage_mean": float(np.mean(voltage_series)) if voltage_series.size else 0.0,
            "internal_resistance_latest": float(resistance_series[-1]) if resistance_series.size else None,
        }

    def _feature_importance(self, model, tensor, feature_names: list[str], base_prediction: float) -> list[FeatureContribution]:
        import torch

        impacts: list[FeatureContribution] = []
        baseline = tensor.clone()
        feature_means = baseline.mean(dim=1, keepdim=True)
        with torch.no_grad():
            for index, feature_name in enumerate(feature_names):
                perturbed = baseline.clone()
                perturbed[:, :, index] = feature_means[:, :, index]
                prediction = float(model.predict(perturbed).item())
                delta = base_prediction - prediction
                impacts.append(
                    FeatureContribution(
                        feature=feature_name,
                        impact=round(abs(delta), 4),
                        direction="increase" if delta < 0 else "decrease",
                        description=self._feature_description(feature_name, delta),
                    )
                )
        impacts.sort(key=lambda item: item.impact, reverse=True)
        return impacts[:5]

    def _window_importance(self, model, tensor, raw_cycles: np.ndarray, base_prediction: float) -> list[WindowContribution]:
        import torch

        seq_len = tensor.shape[1]
        window_count = min(4, max(1, seq_len // 4))
        edges = np.linspace(0, seq_len, num=window_count + 1, dtype=int)
        baseline = tensor.clone()
        fill_value = baseline.mean(dim=1, keepdim=True)
        contributions: list[WindowContribution] = []
        with torch.no_grad():
            for start, end in zip(edges[:-1], edges[1:]):
                if start == end:
                    continue
                perturbed = baseline.clone()
                perturbed[:, start:end, :] = fill_value.expand(-1, end - start, -1)
                prediction = float(model.predict(perturbed).item())
                impact = abs(base_prediction - prediction)
                start_cycle = float(raw_cycles[start]) if raw_cycles.size else float(start)
                end_cycle = float(raw_cycles[min(end - 1, raw_cycles.size - 1)]) if raw_cycles.size else float(end)
                contributions.append(
                    WindowContribution(
                        window_label=f"cycles {int(start_cycle)}-{int(end_cycle)}",
                        start_cycle=start_cycle,
                        end_cycle=end_cycle,
                        impact=round(float(impact), 4),
                        description=f"遮挡该时间窗口后，预测 RUL 变化 {impact:.3f}。",
                    )
                )
        contributions.sort(key=lambda item: item.impact, reverse=True)
        return contributions[:4]

    def _attention_heatmap(self, features: Optional[dict], raw_cycles: np.ndarray) -> Optional[AttentionHeatmap]:
        if not features:
            return None
        attn_weights = features.get("attn_weights")
        if not attn_weights:
            return None
        last_layer = attn_weights[-1]
        if last_layer is None:
            return None
        weights = last_layer.detach().cpu().numpy()
        if weights.ndim != 4:
            return None
        matrix = weights.mean(axis=1)[0]
        labels = [str(int(cycle)) for cycle in raw_cycles.tolist()] if raw_cycles.size else [str(index + 1) for index in range(matrix.shape[0])]
        return AttentionHeatmap(
            x_labels=labels,
            y_labels=labels,
            values=np.round(matrix, 4).tolist(),
        )

    def _estimate_confidence(
        self,
        *,
        predicted_rul: float,
        model_name: str,
        seq_len: int,
        capacities: np.ndarray,
        fallback_used: bool,
    ) -> float:
        normalized_name = model_name.replace("lifecycle_", "")
        base = 0.88 if normalized_name == "hybrid" else 0.8 if normalized_name == "bilstm" else 0.68
        if fallback_used:
            base -= 0.08
        length_bonus = min(0.08, max(0.0, (seq_len - 10) * 0.003))
        stability = self._trajectory_stability(capacities)
        stability_bonus = min(0.08, stability * 0.12)
        long_horizon_penalty = min(0.1, predicted_rul / 4000.0)
        score = base + length_bonus + stability_bonus - long_horizon_penalty
        return round(float(np.clip(score, 0.35, 0.97)), 3)

    @staticmethod
    def _heuristic_rul_from_sequence(sequence: np.ndarray, feature_names: Sequence[str]) -> dict[str, float | str]:
        columns = list(feature_names)
        capacity_index = columns.index("capacity") if "capacity" in columns else -1
        cycle_index = columns.index("cycle_number") if "cycle_number" in columns else -1
        if capacity_index < 0 or cycle_index < 0:
            return {
                "predicted_rul": 120.0,
                "method": "fallback_default",
                "initial_capacity": 0.0,
                "latest_capacity": 0.0,
                "eol_capacity": 0.0,
            }

        capacities = sequence[:, capacity_index].astype(float)
        cycles = sequence[:, cycle_index].astype(float)
        initial_capacity = float(np.max(capacities[: max(3, len(capacities) // 4)]))
        latest_capacity = float(capacities[-1])
        eol_capacity = initial_capacity * 0.8
        if latest_capacity <= eol_capacity:
            return {
                "predicted_rul": 0.0,
                "method": "already_below_eol",
                "initial_capacity": initial_capacity,
                "latest_capacity": latest_capacity,
                "eol_capacity": eol_capacity,
            }

        slope, intercept = np.polyfit(cycles, capacities, deg=1)
        if slope >= -1e-6:
            predicted_rul = max(20.0, (latest_capacity - eol_capacity) / max(0.002, initial_capacity * 0.002))
            method = "flat_trend_proxy"
        else:
            eol_cycle = (eol_capacity - intercept) / slope
            predicted_rul = max(0.0, eol_cycle - cycles[-1])
            method = "linear_trend_proxy"
        return {
            "predicted_rul": float(predicted_rul),
            "method": method,
            "initial_capacity": initial_capacity,
            "latest_capacity": latest_capacity,
            "eol_capacity": eol_capacity,
        }

    @staticmethod
    def _calibrate_rul_prediction(
        *,
        raw_prediction: float,
        trend_proxy_rul: float,
        latest_capacity: float,
        eol_capacity: float,
        initial_capacity: float,
    ) -> dict[str, object]:
        margin = max(0.0, latest_capacity - eol_capacity)
        healthy_margin = margin >= max(initial_capacity * 0.03, 0.05)
        strong_gap = trend_proxy_rul >= max(12.0, raw_prediction * 3.0)
        if not healthy_margin or not strong_gap:
            return {"predicted_rul": round(float(raw_prediction), 2), "applied": False}

        calibrated = max(raw_prediction, min(trend_proxy_rul, max(12.0, trend_proxy_rul * 0.45)))
        return {
            "predicted_rul": round(float(calibrated), 2),
            "applied": True,
            "raw_model_rul": round(float(raw_prediction), 2),
            "trend_proxy_rul": round(float(trend_proxy_rul), 2),
        }

    @staticmethod
    def _trajectory_stability(capacities: np.ndarray) -> float:
        if capacities.size < 3:
            return 0.25
        deltas = np.diff(capacities)
        spread = float(np.std(deltas))
        mean_magnitude = float(np.mean(np.abs(deltas))) or 1e-6
        score = 1.0 / (1.0 + spread / mean_magnitude)
        return float(np.clip(score, 0.0, 1.0))

    def _normalize_sequence(self, sequence: np.ndarray, normalization: dict, feature_names: list[str]) -> np.ndarray:
        means = normalization.get("means", {}) if isinstance(normalization, dict) else {}
        stds = normalization.get("stds", {}) if isinstance(normalization, dict) else {}
        normalized = sequence.astype(float).copy()
        for index, feature_name in enumerate(feature_names):
            mean = float(means.get(feature_name, 0.0))
            std = float(stds.get(feature_name, 1.0)) or 1.0
            normalized[:, index] = (normalized[:, index] - mean) / max(std, 1e-6)
        return normalized

    @staticmethod
    def _build_sequence_from_inputs(
        *,
        points: Optional[Sequence[dict[str, float]]],
        sequence: Optional[np.ndarray],
        feature_cols: Sequence[str],
    ) -> np.ndarray:
        if points is not None:
            return RULInferenceService.sequence_from_cycle_points(points, feature_cols=feature_cols)
        if sequence is None:
            raise ValueError("缺少可用于预测的输入序列")
        return np.asarray(sequence, dtype=float)

    @staticmethod
    def _safe_column(sequence: np.ndarray, feature_names: list[str], feature_name: str) -> np.ndarray:
        if feature_name not in feature_names:
            return np.asarray([], dtype=float)
        index = feature_names.index(feature_name)
        return sequence[:, index].astype(float)

    def _extract_series(
        self,
        points: Optional[Sequence[dict[str, float]]],
        feature_names: list[str],
        sequence: np.ndarray,
        feature_name: str,
    ) -> np.ndarray:
        if points is not None:
            return np.asarray([float(item.get(feature_name, 0.0) or 0.0) for item in points], dtype=float)
        return self._safe_column(sequence, feature_names, feature_name)

    @staticmethod
    def _feature_description(feature_name: str, delta: float) -> str:
        effect = "提升" if delta < 0 else "压低"
        return f"遮挡 {feature_name} 后，模型输出相对基线{effect}了 RUL 估计。"

    @classmethod
    def _summary_checkpoint_candidates(
        cls,
        summary_path: Path,
        payload: dict[str, object],
        *,
        run_keys: Sequence[str] = ("per_seed_runs",),
        metric_keys: Sequence[str] = ("rmse",),
    ) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def maybe_add(raw_path: object) -> None:
            if not isinstance(raw_path, (str, Path)):
                return
            candidate = cls._resolve_reference_path(summary_path, raw_path)
            if candidate is None or candidate in seen:
                return
            seen.add(candidate)
            candidates.append(candidate)

        best_checkpoint = payload.get("best_checkpoint")
        if isinstance(best_checkpoint, dict):
            maybe_add(best_checkpoint.get("path"))
        else:
            maybe_add(best_checkpoint)
        maybe_add(payload.get("final_checkpoint"))

        ranked_runs: list[tuple[float, object]] = []
        for run_key in run_keys:
            runs = payload.get(run_key) or []
            if not isinstance(runs, list):
                continue
            for run in runs:
                if not isinstance(run, dict):
                    continue
                metrics = run.get("metrics") or run.get("test_metrics") or {}
                checkpoint_path = run.get("best_checkpoint") or run.get("final_checkpoint")
                if checkpoint_path is None:
                    continue
                metric_value = float("inf")
                if isinstance(metrics, dict):
                    for metric_key in metric_keys:
                        value = metrics.get(metric_key)
                        if isinstance(value, (int, float)):
                            metric_value = float(value)
                            break
                ranked_runs.append((metric_value, checkpoint_path))
        ranked_runs.sort(key=lambda item: item[0])
        for _, checkpoint_path in ranked_runs:
            maybe_add(checkpoint_path)
        return candidates

    @classmethod
    def _resolve_checkpoint_from_release_manifest(cls, release_path: Path) -> Optional[Path]:
        if not release_path.exists():
            return None
        try:
            payload = json.loads(release_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        explicit_checkpoint = payload.get("checkpoint_path")
        if explicit_checkpoint is None:
            best_checkpoint = payload.get("best_checkpoint")
            if isinstance(best_checkpoint, dict):
                explicit_checkpoint = best_checkpoint.get("path")
            else:
                explicit_checkpoint = best_checkpoint
        if explicit_checkpoint:
            candidate = cls._resolve_reference_path(release_path, explicit_checkpoint)
            if candidate is not None:
                return candidate

        summary_reference = payload.get("summary_path")
        if not summary_reference:
            return None
        summary_path = cls._resolve_reference_path(release_path, summary_reference)
        if summary_path is None:
            return None
        return cls._resolve_checkpoint_from_summary(summary_path)

    def _resolve_checkpoint(self, source: str, model_name: str) -> Path:
        release_manifest = self.model_dir / source / model_name / "release" / "final_release.json"
        resolved = self._resolve_checkpoint_from_release_manifest(release_manifest)
        if resolved is not None:
            return resolved
        summary_candidates = [
            self.model_dir / source / model_name / f"{model_name}_multi_seed_summary.json",
            self.model_dir / source / model_name / "optimized-config" / "optimized_multi_seed_summary.json",
            self.model_dir / source / model_name / f"{model_name}_experiment_summary.json",
        ]
        for summary_path in summary_candidates:
            resolved = self._resolve_checkpoint_from_summary(summary_path)
            if resolved is not None:
                return resolved
        candidates = [
            self.model_dir / source / model_name / f"{model_name}_best.pt",
            self.model_dir / source / model_name / f"{model_name}_final.pt",
            self.model_dir / f"{model_name}_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"未找到来源 {source} 的模型权重: {model_name}")

    @staticmethod
    def _resolve_checkpoint_from_summary(summary_path: Path) -> Optional[Path]:
        if not summary_path.exists():
            return None
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        for candidate in RULInferenceService._summary_checkpoint_candidates(
            summary_path,
            payload,
            run_keys=("per_seed_runs",),
            metric_keys=("rmse",),
        ):
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _resolve_reference_path(summary_path: Path, raw_path: str | Path) -> Optional[Path]:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate if candidate.exists() else None

        # Prefer paths relative to the artifact tree first so copied repos or relocated
        # workspaces do not accidentally resolve back to the original project root.
        search_roots = [summary_path.parent, *summary_path.parents, PROJECT_ROOT]
        seen: set[Path] = set()
        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        return None

    @staticmethod
    def sequence_from_cycle_points(points: Iterable[dict[str, float]], feature_cols: Optional[Sequence[str]] = None) -> np.ndarray:
        columns = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        rows = []
        for point in points:
            rows.append([float(point.get(column, 0.0) or 0.0) for column in columns])
        if not rows:
            raise ValueError("历史序列为空，无法执行预测")
        return np.asarray(rows, dtype=float)

    def save_metadata(self, output_path: str | Path, metadata: dict[str, object]) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class LifecycleInferenceService(RULInferenceService):
    _LEGACY_LIFECYCLE_KEY_PREFIXES: dict[str, str] = {
        "fusion.xlstm_proj.": "fusion.fused_path.xlstm_proj.",
        "fusion.trans_proj.": "fusion.fused_path.trans_proj.",
        "fusion.gate.": "fusion.fused_path.gate.",
        "fusion.fusion_norm.": "fusion.fused_path.fusion_norm.",
    }
    _LEGACY_LIFECYCLE_ALLOWED_MISSING_PREFIXES: tuple[str, ...] = (
        "fusion.xlstm_only.",
        "fusion.transformer_only.",
        "fusion.selector.",
        "decoder.trajectory_gate.",
        "decoder.fluctuation_head.",
    )

    @staticmethod
    def _checkpoint_is_lifecycle(checkpoint_path: Path) -> bool:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
        return payload.get("task_kind") == "lifecycle"

    @classmethod
    def _resolve_lifecycle_checkpoint_from_summary(cls, summary_path: Path) -> Optional[Path]:
        if not summary_path.exists():
            return None
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        for candidate in cls._summary_checkpoint_candidates(
            summary_path,
            payload,
            run_keys=("fine_tune_runs", "per_seed_runs"),
            metric_keys=("trajectory_rmse", "rmse"),
        ):
            if candidate is not None and cls._checkpoint_is_lifecycle(candidate):
                return candidate
        return None

    @classmethod
    def _resolve_lifecycle_checkpoint_from_release_manifest(cls, release_path: Path) -> Optional[Path]:
        candidate = cls._resolve_checkpoint_from_release_manifest(release_path)
        if candidate is None or not cls._checkpoint_is_lifecycle(candidate):
            return None
        return candidate

    def _resolve_checkpoint(self, source: str, model_name: str) -> Path:
        release_manifest = self.model_dir / source / model_name / "release" / "final_release.json"
        resolved = self._resolve_lifecycle_checkpoint_from_release_manifest(release_manifest)
        if resolved is not None:
            return resolved
        summary_candidates = [
            self.model_dir / source / model_name / "transfer" / f"multisource_to_{source}" / f"{model_name}_transfer_summary.json",
            self.model_dir / source / model_name / f"{model_name}_multi_seed_summary.json",
            self.model_dir / source / model_name / "optimized-config" / "optimized_multi_seed_summary.json",
            self.model_dir / source / model_name / f"{model_name}_experiment_summary.json",
        ]
        for summary_path in summary_candidates:
            resolved = self._resolve_lifecycle_checkpoint_from_summary(summary_path)
            if resolved is not None:
                return resolved
        candidates = [
            self.model_dir / source / model_name / f"{model_name}_best.pt",
            self.model_dir / source / model_name / f"{model_name}_final.pt",
            self.model_dir / f"{model_name}_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists() and self._checkpoint_is_lifecycle(candidate):
                return candidate
        raise FileNotFoundError(f"未找到来源 {source} 的生命周期模型权重: {model_name}")

    def predict(
        self,
        sequence: Optional[np.ndarray],
        source: str,
        model_name: str = "hybrid",
        feature_cols: Optional[Sequence[str]] = None,
        points: Optional[Sequence[dict[str, float]]] = None,
        battery_info: Optional[dict[str, object]] = None,
    ) -> LifecyclePredictionOutput:
        source = source.lower()
        requested_model = model_name
        for candidate in self._preferred_models(source, requested_model):
            try:
                output = self._predict_with_checkpoint(
                    sequence=sequence,
                    points=points,
                    source=source,
                    model_name=candidate,
                    feature_cols=feature_cols,
                    battery_info=battery_info or {},
                )
                output.fallback_used = candidate != requested_model
                return output
            except Exception:
                continue
        return self._heuristic_predict(
            sequence=sequence,
            points=points,
            source=source,
            feature_cols=feature_cols,
            battery_info=battery_info or {},
        )

    @classmethod
    def _remap_legacy_lifecycle_state_dict(cls, model_state: dict[str, object]) -> dict[str, object]:
        remapped: dict[str, object] = {}
        for name, value in model_state.items():
            mapped_name = name
            for legacy_prefix, current_prefix in cls._LEGACY_LIFECYCLE_KEY_PREFIXES.items():
                if name.startswith(legacy_prefix):
                    mapped_name = current_prefix + name[len(legacy_prefix):]
                    break
            remapped[mapped_name] = value
        return remapped

    @classmethod
    def _is_allowed_legacy_lifecycle_missing_key(cls, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in cls._LEGACY_LIFECYCLE_ALLOWED_MISSING_PREFIXES)

    @classmethod
    def _load_lifecycle_model_state(cls, model, checkpoint: dict[str, object], checkpoint_path: Path) -> None:
        model_state = checkpoint.get("model_state_dict")
        if not model_state:
            raise ValueError(f"checkpoint {checkpoint_path.name} 缺少 model_state_dict")
        try:
            model.load_state_dict(model_state)
            return
        except RuntimeError as exc:
            remapped_state = cls._remap_legacy_lifecycle_state_dict(model_state)
            current_state = model.state_dict()
            compatible_state: dict[str, object] = {}
            skipped_shape_mismatches: list[str] = []
            for name, value in remapped_state.items():
                target = current_state.get(name)
                if target is None:
                    continue
                if getattr(value, "shape", None) != target.shape:
                    skipped_shape_mismatches.append(name)
                    continue
                compatible_state[name] = value

            if not compatible_state:
                raise RuntimeError(f"checkpoint {checkpoint_path.name} 与当前生命周期模型结构不兼容") from exc

            missing_keys = [name for name in current_state.keys() if name not in compatible_state]
            unexpected_keys = [name for name in remapped_state.keys() if name not in current_state]
            unsupported_missing = [name for name in missing_keys if not cls._is_allowed_legacy_lifecycle_missing_key(name)]
            if unsupported_missing or unexpected_keys or skipped_shape_mismatches:
                details: list[str] = []
                if unsupported_missing:
                    details.append(f"missing={unsupported_missing[:8]}")
                if unexpected_keys:
                    details.append(f"unexpected={unexpected_keys[:8]}")
                if skipped_shape_mismatches:
                    details.append(f"shape_mismatch={skipped_shape_mismatches[:8]}")
                detail_text = ", ".join(details) if details else "legacy compatibility load failed"
                raise RuntimeError(f"checkpoint {checkpoint_path.name} 无法兼容当前生命周期模型: {detail_text}") from exc

            model.load_state_dict(compatible_state, strict=False)

    def _predict_with_checkpoint(
        self,
        sequence: Optional[np.ndarray],
        points: Optional[Sequence[dict[str, float]]],
        source: str,
        model_name: str,
        feature_cols: Optional[Sequence[str]] = None,
        battery_info: Optional[dict[str, object]] = None,
    ) -> LifecyclePredictionOutput:
        import torch

        from ml.models import (
            LifecycleBiLSTMConfig,
            LifecycleBiLSTMPredictor,
            LifecycleHybridConfig,
            LifecycleHybridPredictor,
        )

        checkpoint_path = self._resolve_checkpoint(source=source, model_name=model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("task_kind") != "lifecycle":
            raise ValueError(f"checkpoint {checkpoint_path.name} is not a lifecycle model")
        runtime_key = f"lifecycle:{source}:{model_name}:{checkpoint_path}"
        runtime = self._cache.get(runtime_key)
        if runtime is None:
            model_type = str(checkpoint.get("model_type", model_name)).lower()
            config = checkpoint["model_config"]
            if model_type in {"hybrid", "lifecycle_hybrid"}:
                model = LifecycleHybridPredictor(LifecycleHybridConfig(**config))
            else:
                model = LifecycleBiLSTMPredictor(LifecycleBiLSTMConfig(**config))
            self._load_lifecycle_model_state(model, checkpoint, checkpoint_path)
            model.eval()
            runtime = model
            self._cache[runtime_key] = runtime
        model = runtime

        feature_names = list(checkpoint.get("feature_columns") or feature_cols or DEFAULT_FEATURE_COLUMNS)
        raw_sequence = self._build_lifecycle_sequence(
            points=points,
            sequence=sequence,
            feature_cols=feature_names,
            battery_info=battery_info or {},
        )
        normalization = checkpoint.get("normalization", {})
        normalized_sequence = self._normalize_sequence(raw_sequence, normalization, feature_names)
        tensor = torch.tensor(normalized_sequence[None, :, :], dtype=torch.float32)
        raw_cycles = self._extract_series(points, feature_names, raw_sequence, "cycle_number")
        raw_capacities = self._extract_series(points, feature_names, raw_sequence, "capacity")
        initial_capacity = self._initial_capacity(raw_capacities, battery_info or {})
        eol_ratio = float((battery_info or {}).get("eol_ratio") or 0.8)
        observed_cycle = float(raw_cycles[-1]) if raw_cycles.size else float(raw_sequence.shape[0])
        last_capacity_ratio = self._last_capacity_ratio(raw_sequence, feature_names, initial_capacity)
        domain_vocab = checkpoint.get("domain_vocab", {})
        domain_ids = self._domain_ids(domain_vocab, source=source, battery_info=battery_info or {})

        with torch.no_grad():
            outputs = model(
                tensor,
                source_id=torch.tensor([domain_ids["source_id"]], dtype=torch.long),
                chemistry_id=torch.tensor([domain_ids["chemistry_id"]], dtype=torch.long),
                protocol_id=torch.tensor([domain_ids["protocol_id"]], dtype=torch.long),
                last_capacity_ratio=torch.tensor([last_capacity_ratio], dtype=torch.float32),
                observed_cycle=torch.tensor([observed_cycle], dtype=torch.float32),
                return_features=True,
            )
        trajectory = outputs["trajectory"].detach().cpu().numpy()[0]
        raw_rul = float(outputs["rul"].detach().cpu().view(-1)[0].item())
        raw_eol_cycle = float(outputs["eol_cycle"].detach().cpu().view(-1)[0].item())
        raw_knee_cycle = float(outputs["knee_cycle"].detach().cpu().view(-1)[0].item())
        uncertainty = float(outputs["uncertainty"].detach().cpu().view(-1)[0].item()) if "uncertainty" in outputs else None
        predicted_eol_cycle = self._infer_eol_cycle(
            trajectory,
            observed_cycle=observed_cycle,
            eol_ratio=eol_ratio,
            fallback_eol=raw_eol_cycle,
        )
        predicted_rul = max(0.0, float(predicted_eol_cycle - observed_cycle))
        predicted_knee_cycle = self._infer_knee_cycle(
            trajectory,
            observed_cycle=observed_cycle,
            fallback_knee=raw_knee_cycle,
            predicted_eol_cycle=predicted_eol_cycle,
            eol_ratio=eol_ratio,
        )
        projection = self._build_lifecycle_projection(
            raw_cycles=raw_cycles,
            raw_capacities=raw_capacities,
            trajectory=trajectory,
            initial_capacity=initial_capacity,
            observed_cycle=observed_cycle,
            predicted_eol_cycle=predicted_eol_cycle,
            uncertainty=uncertainty,
            eol_ratio=eol_ratio,
        )
        confidence = self._estimate_confidence(
            predicted_rul=predicted_rul,
            model_name=model_name,
            seq_len=raw_sequence.shape[0],
            capacities=raw_capacities,
            fallback_used=False,
        )
        explanation = self._build_lifecycle_explanation(
            model=model,
            tensor=tensor,
            raw_sequence=raw_sequence,
            feature_names=feature_names,
            source=source,
            model_name=model_name,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            outputs=outputs,
            raw_cycles=raw_cycles,
            raw_capacities=raw_capacities,
            predicted_rul=predicted_rul,
            predicted_eol_cycle=predicted_eol_cycle,
            predicted_knee_cycle=predicted_knee_cycle,
            confidence=confidence,
            domain_ids=domain_ids,
            last_capacity_ratio=last_capacity_ratio,
            observed_cycle=observed_cycle,
        )
        return LifecyclePredictionOutput(
            predicted_rul=predicted_rul,
            predicted_knee_cycle=predicted_knee_cycle,
            predicted_eol_cycle=predicted_eol_cycle,
            confidence=confidence,
            model_name=model_name,
            model_version=str(checkpoint.get("model_version", "lifecycle-trained")),
            model_source=source,
            checkpoint_id=checkpoint_path.name,
            fallback_used=False,
            trajectory=self._serialize_trajectory(
                trajectory,
                observed_cycle=observed_cycle,
                predicted_eol_cycle=predicted_eol_cycle,
                eol_ratio=eol_ratio,
            ),
            projection=projection,
            uncertainty=uncertainty,
            explanation=explanation,
        )

    def _heuristic_predict(
        self,
        sequence: Optional[np.ndarray],
        points: Optional[Sequence[dict[str, float]]],
        source: str,
        feature_cols: Optional[Sequence[str]] = None,
        battery_info: Optional[dict[str, object]] = None,
    ) -> LifecyclePredictionOutput:
        columns = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        raw_sequence = self._build_lifecycle_sequence(
            points=points,
            sequence=sequence,
            feature_cols=columns,
            battery_info=battery_info or {},
        )
        raw_cycles = self._extract_series(points, columns, raw_sequence, "cycle_number")
        raw_capacities = self._extract_series(points, columns, raw_sequence, "capacity")
        trend_proxy = self._heuristic_rul_from_sequence(raw_sequence, columns)
        initial_capacity = self._initial_capacity(raw_capacities, battery_info or {})
        eol_ratio = float((battery_info or {}).get("eol_ratio") or 0.8)
        observed_cycle = float(raw_cycles[-1]) if raw_cycles.size else float(raw_sequence.shape[0])
        predicted_rul = float(trend_proxy["predicted_rul"])
        predicted_eol_cycle = observed_cycle + predicted_rul
        predicted_knee_cycle = observed_cycle + max(5.0, predicted_rul * 0.35)
        last_ratio = self._last_capacity_ratio(raw_sequence, columns, initial_capacity)
        horizon_grid = np.linspace(0.0, 1.0, num=64, dtype=float)
        target_ratio = min(max(eol_ratio, 0.0), last_ratio)
        trajectory = last_ratio - (last_ratio - target_ratio) * np.power(horizon_grid, 1.2)
        trajectory = np.minimum.accumulate(trajectory)
        projection = self._build_lifecycle_projection(
            raw_cycles=raw_cycles,
            raw_capacities=raw_capacities,
            trajectory=trajectory,
            initial_capacity=initial_capacity,
            observed_cycle=observed_cycle,
            predicted_eol_cycle=predicted_eol_cycle,
            uncertainty=None,
            eol_ratio=eol_ratio,
        )
        confidence = self._estimate_confidence(
            predicted_rul=predicted_rul,
            model_name="heuristic",
            seq_len=raw_sequence.shape[0],
            capacities=raw_capacities,
            fallback_used=True,
        )
        explanation = PredictionExplanation(
            input_summary=self._build_input_summary(raw_sequence, columns, raw_cycles, raw_capacities),
            model_info={
                "model_name": "heuristic",
                "model_source": source,
                "checkpoint_id": None,
                "task_kind": "lifecycle",
                "fallback_used": True,
                "predicted_eol_cycle": round(predicted_eol_cycle, 2),
                "predicted_knee_cycle": round(predicted_knee_cycle, 2),
                "note": "当前未找到可用生命周期权重，系统退回到基于容量趋势的生命周期启发式估计。",
            },
            feature_contributions=[
                FeatureContribution(
                    feature="capacity_ratio",
                    impact=0.82,
                    direction="decrease",
                    description="启发式轨迹主要依赖当前容量比例和近端衰减速度。",
                )
            ],
            window_contributions=[
                WindowContribution(
                    window_label="observed_tail",
                    start_cycle=float(raw_cycles[max(0, len(raw_cycles) - 5)]) if len(raw_cycles) else 0.0,
                    end_cycle=float(raw_cycles[-1]) if len(raw_cycles) else 0.0,
                    impact=0.78,
                    description="最近若干 cycle 的容量斜率决定了未来 trajectory 投影。",
                )
            ]
            if len(raw_cycles)
            else [],
            confidence_summary={
                "score": confidence,
                "factors": [
                    "当前结果未使用生命周期训练权重",
                    "轨迹由容量衰减趋势直接投影得到",
                    "输入窗口越长且越平滑，启发式稳定度越高",
                ],
            },
            attention_heatmap=None,
        )
        return LifecyclePredictionOutput(
            predicted_rul=predicted_rul,
            predicted_knee_cycle=round(predicted_knee_cycle, 2),
            predicted_eol_cycle=round(predicted_eol_cycle, 2),
            confidence=confidence,
            model_name="heuristic",
            model_version="lifecycle-heuristic-v1",
            model_source=source,
            checkpoint_id=None,
            fallback_used=True,
            trajectory=self._serialize_trajectory(
                trajectory,
                observed_cycle=observed_cycle,
                predicted_eol_cycle=predicted_eol_cycle,
                eol_ratio=eol_ratio,
            ),
            projection=projection,
            uncertainty=None,
            explanation=explanation,
        )

    @staticmethod
    def _initial_capacity(raw_capacities: np.ndarray, battery_info: dict[str, object]) -> float:
        return float(
            battery_info.get("initial_capacity")
            or battery_info.get("nominal_capacity")
            or (float(raw_capacities[0]) if raw_capacities.size else 1.0)
            or 1.0
        )

    @staticmethod
    def _last_capacity_ratio(sequence: np.ndarray, feature_names: list[str], initial_capacity: float) -> float:
        if "capacity_ratio" in feature_names:
            return float(sequence[-1, feature_names.index("capacity_ratio")])
        if "capacity" in feature_names:
            capacity = float(sequence[-1, feature_names.index("capacity")])
            return capacity / max(initial_capacity, 1e-6)
        return 1.0

    @staticmethod
    def _domain_ids(domain_vocab: dict[str, dict[str, int]], *, source: str, battery_info: dict[str, object]) -> dict[str, int]:
        source_map = domain_vocab.get("source_to_id", {}) if isinstance(domain_vocab, dict) else {}
        chemistry_map = domain_vocab.get("chemistry_to_id", {}) if isinstance(domain_vocab, dict) else {}
        protocol_map = domain_vocab.get("protocol_to_id", {}) if isinstance(domain_vocab, dict) else {}
        chemistry = str(battery_info.get("chemistry") or "unknown")
        protocol = str(battery_info.get("protocol_id") or "unknown")
        return {
            "source_id": source_map.get(source, source_map.get("unknown", 0)),
            "chemistry_id": chemistry_map.get(chemistry, chemistry_map.get("unknown", 0)),
            "protocol_id": protocol_map.get(protocol, protocol_map.get("unknown", 0)),
        }

    @staticmethod
    def _trajectory_progress(point_count: int) -> np.ndarray:
        if point_count <= 0:
            return np.asarray([], dtype=float)
        return np.linspace(1.0 / point_count, 1.0, num=point_count, dtype=float)

    @classmethod
    def _estimate_eol_progress(cls, trajectory: np.ndarray, *, eol_ratio: float) -> float:
        values = np.asarray(trajectory, dtype=float).reshape(-1)
        if values.size == 0:
            return 1.0
        progress = cls._trajectory_progress(values.size)
        below = np.where(values <= eol_ratio)[0]
        if not below.size:
            return 1.0
        index = int(below[0])
        if index <= 0:
            return float(progress[index])
        previous_index = index - 1
        upper = float(values[previous_index])
        lower = float(values[index])
        if upper <= eol_ratio:
            return float(progress[index])
        ratio = (upper - eol_ratio) / max(upper - lower, 1e-6)
        interpolated = progress[previous_index] + (progress[index] - progress[previous_index]) * np.clip(ratio, 0.0, 1.0)
        return float(np.clip(interpolated, progress[0], progress[-1]))

    @classmethod
    def _map_trajectory_cycles(
        cls,
        point_count: int,
        *,
        observed_cycle: float,
        predicted_eol_cycle: float,
        eol_progress: float,
    ) -> np.ndarray:
        progress = cls._trajectory_progress(point_count)
        if progress.size == 0:
            return progress
        horizon = max(float(predicted_eol_cycle - observed_cycle), 1.0)
        safe_eol_progress = max(float(eol_progress), float(progress[0]))
        return observed_cycle + (progress / safe_eol_progress) * horizon

    @classmethod
    def _build_variation_pattern(
        cls,
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
        *,
        initial_capacity: float,
    ) -> tuple[np.ndarray, float]:
        recent_window = min(24, raw_capacities.size)
        if recent_window < 6:
            return np.asarray([], dtype=float), 0.0
        recent_cycles = np.asarray(raw_cycles[-recent_window:], dtype=float)
        recent_capacities = np.asarray(raw_capacities[-recent_window:], dtype=float)
        if not np.all(np.isfinite(recent_capacities)):
            return np.asarray([], dtype=float), 0.0
        trend = np.polyval(np.polyfit(recent_cycles, recent_capacities, deg=1), recent_cycles)
        residuals = recent_capacities - trend
        residuals = residuals - float(np.mean(residuals))
        scale = float(np.percentile(np.abs(residuals), 75))
        minimum_scale = max(initial_capacity * 0.002, 0.0025)
        if scale < minimum_scale:
            return np.asarray([], dtype=float), 0.0
        pattern = np.clip(residuals / max(scale, 1e-6), -1.25, 1.25)
        return pattern.astype(float), scale

    @classmethod
    def _build_display_projection_points(
        cls,
        *,
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
        forecast_cycles: np.ndarray,
        forecast_capacities: np.ndarray,
        display_seed_capacities: np.ndarray,
        band_widths: np.ndarray,
        initial_capacity: float,
        predicted_eol_cycle: float,
    ) -> list[dict[str, float]]:
        if forecast_cycles.size == 0:
            return []
        if display_seed_capacities.size == forecast_capacities.size and display_seed_capacities.size:
            positive_rebounds = np.diff(display_seed_capacities)
            rebound_threshold = max(initial_capacity * 0.0012, 0.0015)
            if np.any(positive_rebounds > rebound_threshold):
                lower_bound = np.maximum(0.0, forecast_capacities - band_widths * 0.9)
                upper_bound = forecast_capacities + band_widths * 0.9
                display_capacities = np.clip(display_seed_capacities, lower_bound, upper_bound)
                display_capacities[0] = forecast_capacities[0]
                display_capacities[-1] = forecast_capacities[-1]
                return [
                    {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
                    for cycle, capacity in zip(forecast_cycles, display_capacities)
                ]
        pattern, pattern_scale = cls._build_variation_pattern(
            raw_cycles,
            raw_capacities,
            initial_capacity=initial_capacity,
        )
        if pattern.size == 0 or pattern_scale <= 0.0:
            return [
                {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
                for cycle, capacity in zip(forecast_cycles, forecast_capacities)
            ]

        repeated_pattern = np.resize(pattern, forecast_capacities.size)
        repeated_pattern[0] = 0.0
        repeated_pattern[-1] = 0.0

        start_cycle = float(forecast_cycles[0])
        end_cycle = float(forecast_cycles[-1])
        total_horizon = max(end_cycle - start_cycle, 1e-6)
        eol_anchor = float(np.clip((predicted_eol_cycle - start_cycle) / total_horizon, 0.0, 1.0))
        progress = (forecast_cycles - start_cycle) / total_horizon
        if eol_anchor <= 1e-6:
            decay = 0.18 * (1.0 - progress)
        else:
            decay = np.where(
                progress <= eol_anchor,
                1.0 - 0.45 * (progress / max(eol_anchor, 1e-6)),
                0.55 * (1.0 - (progress - eol_anchor) / max(1.0 - eol_anchor, 1e-6)) + 0.12 * ((progress - eol_anchor) / max(1.0 - eol_anchor, 1e-6)),
            )
        decay = np.clip(decay, 0.08, 1.0)

        amplitude_cap = np.minimum(pattern_scale * decay * 0.85, band_widths * 0.48)
        modulation = repeated_pattern * amplitude_cap
        lower_bound = np.maximum(0.0, forecast_capacities - band_widths * 0.72)
        upper_bound = forecast_capacities + band_widths * 0.72
        display_capacities = np.clip(forecast_capacities + modulation, lower_bound, upper_bound)
        display_capacities[0] = forecast_capacities[0]
        display_capacities[-1] = forecast_capacities[-1]

        return [
            {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
            for cycle, capacity in zip(forecast_cycles, display_capacities)
        ]

    @classmethod
    def _infer_eol_cycle(
        cls,
        trajectory: np.ndarray,
        *,
        observed_cycle: float,
        eol_ratio: float,
        fallback_eol: float,
    ) -> float:
        candidate = float(fallback_eol)
        if np.isfinite(candidate) and candidate > observed_cycle:
            return round(candidate, 2)
        progress = cls._estimate_eol_progress(trajectory, eol_ratio=eol_ratio)
        fallback_horizon = max(8.0, trajectory.size * max(progress, 0.3))
        return round(observed_cycle + fallback_horizon, 2)

    @classmethod
    def _infer_knee_cycle(
        cls,
        trajectory: np.ndarray,
        *,
        observed_cycle: float,
        fallback_knee: float,
        predicted_eol_cycle: float,
        eol_ratio: float,
    ) -> Optional[float]:
        if trajectory.size >= 6:
            slopes = np.diff(trajectory)
            curvature = np.diff(slopes)
            if curvature.size:
                candidate = int(np.argmin(curvature)) + 2
                progress = cls._trajectory_progress(trajectory.size)
                eol_progress = cls._estimate_eol_progress(trajectory, eol_ratio=eol_ratio)
                candidate_progress = float(progress[min(candidate, progress.size - 1)])
                if candidate_progress <= eol_progress:
                    estimated = observed_cycle + (candidate_progress / max(eol_progress, progress[0])) * max(predicted_eol_cycle - observed_cycle, 1.0)
                    if observed_cycle <= estimated <= predicted_eol_cycle:
                        return round(float(estimated), 2)
        if fallback_knee >= observed_cycle:
            return round(min(float(fallback_knee), float(predicted_eol_cycle)), 2)
        return None

    def _build_lifecycle_sequence(
        self,
        *,
        points: Optional[Sequence[dict[str, float]]],
        sequence: Optional[np.ndarray],
        feature_cols: Sequence[str],
        battery_info: dict[str, object],
    ) -> np.ndarray:
        if points is None:
            if sequence is None:
                raise ValueError("缺少可用于生命周期预测的输入序列")
            return np.asarray(sequence, dtype=float)
        initial_capacity = float(
            battery_info.get("initial_capacity")
            or battery_info.get("nominal_capacity")
            or (float(points[0].get("capacity", 1.0) or 1.0) if points else 1.0)
        )
        rows = []
        for point in points:
            row: list[float] = []
            for column in feature_cols:
                if column == "capacity_ratio":
                    capacity = float(point.get("capacity", 0.0) or 0.0)
                    row.append(capacity / max(initial_capacity, 1e-6))
                else:
                    row.append(float(point.get(column, 0.0) or 0.0))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    @classmethod
    def _serialize_trajectory(
        cls,
        trajectory: np.ndarray,
        *,
        observed_cycle: float,
        predicted_eol_cycle: Optional[float] = None,
        eol_ratio: float = 0.8,
    ) -> list[dict[str, float]]:
        if trajectory.size == 0:
            return []
        values = np.asarray(trajectory, dtype=float)
        if predicted_eol_cycle is not None and predicted_eol_cycle > observed_cycle:
            eol_progress = cls._estimate_eol_progress(values, eol_ratio=eol_ratio)
            cycles = cls._map_trajectory_cycles(
                len(values),
                observed_cycle=observed_cycle,
                predicted_eol_cycle=predicted_eol_cycle,
                eol_progress=eol_progress,
            )
        else:
            cycles = observed_cycle + np.arange(1, len(values) + 1, dtype=float)
        return [
            {
                "cycle": round(float(cycle), 2),
                "capacity_ratio": round(float(value), 4),
                "soh": round(float(value), 4),
            }
            for cycle, value in zip(cycles, values)
        ]

    @classmethod
    def _build_lifecycle_projection(
        cls,
        *,
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
        trajectory: np.ndarray,
        initial_capacity: float,
        observed_cycle: float,
        predicted_eol_cycle: float,
        uncertainty: Optional[float],
        eol_ratio: float,
    ) -> dict[str, object]:
        trajectory = np.asarray(trajectory, dtype=float).reshape(-1)
        forecast_cycles = np.asarray([], dtype=float)
        raw_forecast_capacities = np.clip(trajectory, 0.0, None) * max(initial_capacity, 1e-6)
        forecast_capacities = np.maximum(raw_forecast_capacities.copy(), 0.0)
        eol_capacity = initial_capacity * eol_ratio
        last_observed_capacity = float(raw_capacities[-1]) if raw_capacities.size else eol_capacity
        display_seed_capacities = np.asarray([], dtype=float)
        if forecast_capacities.size:
            upper_cap = max(last_observed_capacity + initial_capacity * 0.03, last_observed_capacity)
            raw_forecast_capacities = np.clip(raw_forecast_capacities, 0.0, upper_cap)
            forecast_capacities = np.minimum(np.maximum(raw_forecast_capacities.copy(), 0.0), last_observed_capacity)
            display_seed_capacities = raw_forecast_capacities.copy()
            eol_progress = cls._estimate_eol_progress(trajectory, eol_ratio=eol_ratio)
            forecast_cycles = cls._map_trajectory_cycles(
                forecast_capacities.size,
                observed_cycle=observed_cycle,
                predicted_eol_cycle=predicted_eol_cycle,
                eol_progress=eol_progress,
            )
        if forecast_cycles.size == 0:
            forecast_cycles = np.asarray([observed_cycle + 1.0], dtype=float)
            fallback_capacity = float(raw_capacities[-1]) if raw_capacities.size else eol_capacity
            forecast_capacities = np.asarray([fallback_capacity], dtype=float)
            display_seed_capacities = forecast_capacities.copy()

        forecast_capacities = np.maximum(forecast_capacities, 0.0)
        reference_horizon = max(float(forecast_cycles[-1] - observed_cycle), float(predicted_eol_cycle - observed_cycle), 1.0)
        predicted_zero_cycle = round(float(forecast_cycles[-1]), 2)
        tail_points: list[dict[str, float]] = []

        zero_index = np.where(forecast_capacities <= 1e-6)[0]
        if zero_index.size:
            stop = int(zero_index[0]) + 1
            forecast_cycles = forecast_cycles[:stop]
            forecast_capacities = forecast_capacities[:stop]
            forecast_capacities[-1] = 0.0
            predicted_zero_cycle = round(float(forecast_cycles[-1]), 2)
        elif forecast_cycles.size >= 2 and forecast_capacities[-1] > 1e-6:
            recent_window = min(6, forecast_cycles.size)
            recent_cycles = forecast_cycles[-recent_window:]
            recent_capacities = forecast_capacities[-recent_window:]
            slope_steps = np.diff(recent_cycles)
            slope_drop = -np.diff(recent_capacities)
            tail_slopes = slope_drop / np.maximum(slope_steps, 1e-6)
            tail_slopes = tail_slopes[tail_slopes > 1e-6]
            tail_slope = float(np.median(tail_slopes)) if tail_slopes.size else initial_capacity * 0.002
            raw_extra_horizon = forecast_capacities[-1] / max(tail_slope, 1e-6)
            extra_horizon = float(np.clip(raw_extra_horizon, reference_horizon * 0.35, reference_horizon * 2.0))
            predicted_zero_cycle = round(float(forecast_cycles[-1] + extra_horizon), 2)
            tail_cycles = np.linspace(forecast_cycles[-1], predicted_zero_cycle, num=max(10, min(64, int(extra_horizon) + 1)))
            tail_progress = (tail_cycles - tail_cycles[0]) / max(extra_horizon, 1e-6)
            tail_shape = float(np.clip((tail_slope * extra_horizon) / max(forecast_capacities[-1], 1e-6), 0.9, 2.4))
            tail_capacities = np.maximum(0.0, forecast_capacities[-1] * np.power(np.clip(1.0 - tail_progress, 0.0, 1.0), tail_shape))
            tail_points = [
                {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
                for cycle, capacity in zip(tail_cycles[1:], tail_capacities[1:])
            ]
            if tail_points:
                tail_points[-1]["capacity"] = 0.0
                forecast_cycles = np.concatenate([forecast_cycles, tail_cycles[1:]])
                forecast_capacities = np.concatenate([forecast_capacities, tail_capacities[1:]])
                forecast_capacities[-1] = 0.0
                display_seed_capacities = np.concatenate([display_seed_capacities, tail_capacities[1:]])
                display_seed_capacities[-1] = 0.0
        band_progress = np.linspace(0.0, 1.0, num=forecast_cycles.size, dtype=float)
        base_band = max(initial_capacity * 0.012, float(uncertainty or 0.0) * initial_capacity * 0.06)
        growth_band = max(initial_capacity * 0.01, base_band * 0.75)
        band_widths = base_band + growth_band * np.power(band_progress, 1.15)
        display_points = cls._build_display_projection_points(
            raw_cycles=raw_cycles,
            raw_capacities=raw_capacities,
            forecast_cycles=forecast_cycles,
            forecast_capacities=forecast_capacities,
            display_seed_capacities=display_seed_capacities,
            band_widths=band_widths,
            initial_capacity=initial_capacity,
            predicted_eol_cycle=predicted_eol_cycle,
        )
        confidence_band = [
            {
                "cycle": round(float(cycle), 2),
                "lower": round(float(max(0.0, capacity - band_width)), 4),
                "upper": round(float(capacity + band_width), 4),
            }
            for cycle, capacity, band_width in zip(forecast_cycles, forecast_capacities, band_widths)
        ]
        return {
            "actual_points": [
                {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
                for cycle, capacity in zip(raw_cycles, raw_capacities)
            ],
            "forecast_points": [
                {"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)}
                for cycle, capacity in zip(forecast_cycles, forecast_capacities)
            ],
            "eol_capacity": round(float(eol_capacity), 4),
            "predicted_eol_cycle": round(float(predicted_eol_cycle), 2),
            "confidence_band": confidence_band,
            "display_points": display_points,
            "tail_points": tail_points,
            "predicted_zero_cycle": predicted_zero_cycle,
            "projection_method": "lifecycle_decoder",
        }

    def _build_lifecycle_explanation(
        self,
        *,
        model,
        tensor,
        raw_sequence: np.ndarray,
        feature_names: list[str],
        source: str,
        model_name: str,
        checkpoint: dict[str, object],
        checkpoint_path: Path,
        outputs: dict[str, torch.Tensor],
        raw_cycles: np.ndarray,
        raw_capacities: np.ndarray,
        predicted_rul: float,
        predicted_eol_cycle: float,
        predicted_knee_cycle: Optional[float],
        confidence: float,
        domain_ids: dict[str, int],
        last_capacity_ratio: float,
        observed_cycle: float,
    ) -> PredictionExplanation:
        feature_contributions = self._lifecycle_feature_importance(
            model=model,
            tensor=tensor,
            feature_names=feature_names,
            predicted_rul=predicted_rul,
            domain_ids=domain_ids,
            last_capacity_ratio=last_capacity_ratio,
            observed_cycle=observed_cycle,
        )
        window_contributions = self._lifecycle_window_importance(
            model=model,
            tensor=tensor,
            raw_cycles=raw_cycles,
            predicted_rul=predicted_rul,
            domain_ids=domain_ids,
            last_capacity_ratio=last_capacity_ratio,
            observed_cycle=observed_cycle,
        )
        attention_heatmap = self._lifecycle_attention_heatmap(outputs, raw_cycles)
        confidence_factors = [
            f"生命周期输入窗口长度为 {tensor.shape[1]} 个 cycle",
            f"当前模型为 {model_name}，来源 {source}",
            f"预测 EOL 周期为 {predicted_eol_cycle:.2f}",
            f"预测 knee 周期为 {predicted_knee_cycle if predicted_knee_cycle is not None else '--'}",
            f"轨迹稳定度评分为 {self._trajectory_stability(raw_capacities):.3f}",
        ]
        return PredictionExplanation(
            input_summary=self._build_input_summary(raw_sequence, feature_names, raw_cycles, raw_capacities),
            model_info={
                "task_kind": "lifecycle",
                "model_name": model_name,
                "model_source": source,
                "checkpoint_id": checkpoint_path.name,
                "model_version": str(checkpoint.get("model_version", "lifecycle-trained")),
                "fallback_used": False,
                "predicted_rul": round(float(predicted_rul), 2),
                "predicted_eol_cycle": round(float(predicted_eol_cycle), 2),
                "predicted_knee_cycle": round(float(predicted_knee_cycle), 2) if predicted_knee_cycle is not None else None,
            },
            feature_contributions=feature_contributions,
            window_contributions=window_contributions,
            confidence_summary={"score": confidence, "factors": confidence_factors},
            attention_heatmap=attention_heatmap,
        )

    def _lifecycle_feature_importance(
        self,
        *,
        model,
        tensor,
        feature_names: list[str],
        predicted_rul: float,
        domain_ids: dict[str, int],
        last_capacity_ratio: float,
        observed_cycle: float,
    ) -> list[FeatureContribution]:
        import torch

        baseline = tensor.clone()
        feature_means = baseline.mean(dim=1, keepdim=True)
        impacts: list[FeatureContribution] = []
        with torch.no_grad():
            for index, feature_name in enumerate(feature_names):
                perturbed = baseline.clone()
                perturbed[:, :, index] = feature_means[:, :, index]
                output = model(
                    perturbed,
                    source_id=torch.tensor([domain_ids["source_id"]], dtype=torch.long),
                    chemistry_id=torch.tensor([domain_ids["chemistry_id"]], dtype=torch.long),
                    protocol_id=torch.tensor([domain_ids["protocol_id"]], dtype=torch.long),
                    last_capacity_ratio=torch.tensor([last_capacity_ratio], dtype=torch.float32),
                    observed_cycle=torch.tensor([observed_cycle], dtype=torch.float32),
                )
                new_rul = float(output["rul"].detach().cpu().view(-1)[0].item())
                delta = predicted_rul - new_rul
                impacts.append(
                    FeatureContribution(
                        feature=feature_name,
                        impact=round(abs(delta), 4),
                        direction="increase" if delta < 0 else "decrease",
                        description=self._feature_description(feature_name, delta),
                    )
                )
        impacts.sort(key=lambda item: item.impact, reverse=True)
        return impacts[:5]

    def _lifecycle_window_importance(
        self,
        *,
        model,
        tensor,
        raw_cycles: np.ndarray,
        predicted_rul: float,
        domain_ids: dict[str, int],
        last_capacity_ratio: float,
        observed_cycle: float,
    ) -> list[WindowContribution]:
        import torch

        seq_len = tensor.shape[1]
        edges = np.linspace(0, seq_len, num=min(4, max(2, seq_len // 4)) + 1, dtype=int)
        baseline = tensor.clone()
        fill_value = baseline.mean(dim=1, keepdim=True)
        contributions: list[WindowContribution] = []
        with torch.no_grad():
            for start, end in zip(edges[:-1], edges[1:]):
                if start == end:
                    continue
                perturbed = baseline.clone()
                perturbed[:, start:end, :] = fill_value.expand(-1, end - start, -1)
                output = model(
                    perturbed,
                    source_id=torch.tensor([domain_ids["source_id"]], dtype=torch.long),
                    chemistry_id=torch.tensor([domain_ids["chemistry_id"]], dtype=torch.long),
                    protocol_id=torch.tensor([domain_ids["protocol_id"]], dtype=torch.long),
                    last_capacity_ratio=torch.tensor([last_capacity_ratio], dtype=torch.float32),
                    observed_cycle=torch.tensor([observed_cycle], dtype=torch.float32),
                )
                new_rul = float(output["rul"].detach().cpu().view(-1)[0].item())
                impact = abs(predicted_rul - new_rul)
                start_cycle = float(raw_cycles[start]) if raw_cycles.size else float(start)
                end_cycle = float(raw_cycles[min(end - 1, raw_cycles.size - 1)]) if raw_cycles.size else float(end)
                contributions.append(
                    WindowContribution(
                        window_label=f"cycles {int(start_cycle)}-{int(end_cycle)}",
                        start_cycle=start_cycle,
                        end_cycle=end_cycle,
                        impact=round(float(impact), 4),
                        description=f"遮挡该时间窗口后，生命周期 RUL 变化 {impact:.3f}。",
                    )
                )
        contributions.sort(key=lambda item: item.impact, reverse=True)
        return contributions[:4]

    @staticmethod
    def _lifecycle_attention_heatmap(outputs: dict[str, object], raw_cycles: np.ndarray) -> Optional[AttentionHeatmap]:
        decoder_attention = outputs.get("decoder_attention")
        if decoder_attention is None:
            return None
        weights = decoder_attention.detach().cpu().numpy()
        if weights.ndim != 3:
            return None
        matrix = weights[0]
        matrix = matrix[: min(12, matrix.shape[0]), :]
        x_labels = [str(int(cycle)) for cycle in raw_cycles.tolist()] if raw_cycles.size else [str(index + 1) for index in range(matrix.shape[1])]
        y_labels = [f"h{index + 1}" for index in range(matrix.shape[0])]
        return AttentionHeatmap(
            x_labels=x_labels,
            y_labels=y_labels,
            values=np.round(matrix, 4).tolist(),
            disclaimer="生命周期 heatmap 表示未来 horizon query 对观测窗口的关注强度，不直接等于因果关系。",
        )


__all__ = [
    "AttentionHeatmap",
    "FeatureContribution",
    "LifecycleInferenceService",
    "LifecyclePredictionOutput",
    "PredictionExplanation",
    "PredictionOutput",
    "RULInferenceService",
    "WindowContribution",
]
