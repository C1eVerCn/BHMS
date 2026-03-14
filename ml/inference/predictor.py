"""RUL 推理服务，支持来源级训练权重与启发式 fallback。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ml.data.nasa_preprocessor import DEFAULT_FEATURE_COLUMNS


@dataclass
class PredictionOutput:
    predicted_rul: float
    confidence: float
    model_name: str
    model_version: str
    model_source: str
    checkpoint_id: Optional[str] = None
    fallback_used: bool = False


class RULInferenceService:
    def __init__(self, model_dir: str | Path = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, object] = {}

    def predict(
        self,
        sequence: np.ndarray,
        source: str,
        model_name: str = "hybrid",
        feature_cols: Optional[Sequence[str]] = None,
    ) -> PredictionOutput:
        source = source.lower()
        for candidate in self._preferred_models(source, model_name):
            try:
                return self._predict_with_checkpoint(sequence, source=source, model_name=candidate)
            except Exception:
                continue
        return self._heuristic_predict(sequence, source=source, feature_cols=feature_cols)

    def _preferred_models(self, source: str, requested: str) -> list[str]:
        if requested == "auto":
            return ["hybrid", "bilstm"]
        ordered = [requested]
        if requested != "hybrid":
            ordered.append("hybrid")
        if "bilstm" not in ordered:
            ordered.append("bilstm")
        return ordered

    def _predict_with_checkpoint(self, sequence: np.ndarray, source: str, model_name: str) -> PredictionOutput:
        import torch
        from ml.models.baseline import BiLSTMConfig, BiLSTMRULPredictor
        from ml.models.hybrid import RULPredictor, RULPredictorConfig

        checkpoint_path = self._resolve_checkpoint(source=source, model_name=model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runtime_key = f"{source}:{model_name}:{checkpoint_path}"
        model = self._cache.get(runtime_key)
        if model is None:
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
            self._cache[runtime_key] = model

        tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            prediction = model.predict(tensor).item()
        return PredictionOutput(
            predicted_rul=max(0.0, float(prediction)),
            confidence=0.9 if model_name == "hybrid" else 0.82,
            model_name=model_name,
            model_version=str(checkpoint.get("model_version", checkpoint.get("epoch", "trained"))),
            model_source=source.lower(),
            checkpoint_id=checkpoint_path.name,
            fallback_used=False,
        )

    def _heuristic_predict(
        self,
        sequence: np.ndarray,
        source: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> PredictionOutput:
        columns = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        capacity_index = columns.index("capacity") if "capacity" in columns else -1
        cycle_index = columns.index("cycle_number") if "cycle_number" in columns else -1
        if capacity_index < 0 or cycle_index < 0:
            return PredictionOutput(120.0, 0.55, "heuristic", "heuristic-v1", source, checkpoint_id=None, fallback_used=True)

        capacities = sequence[:, capacity_index].astype(float)
        cycles = sequence[:, cycle_index].astype(float)
        initial_capacity = float(np.max(capacities[: max(3, len(capacities) // 4)]))
        latest_capacity = float(capacities[-1])
        eol_capacity = initial_capacity * 0.8
        if latest_capacity <= eol_capacity:
            predicted_rul = 0.0
        else:
            slope, intercept = np.polyfit(cycles, capacities, deg=1)
            if slope >= -1e-6:
                predicted_rul = max(20.0, (latest_capacity - eol_capacity) / max(0.002, initial_capacity * 0.002))
            else:
                eol_cycle = (eol_capacity - intercept) / slope
                predicted_rul = max(0.0, eol_cycle - cycles[-1])
        confidence = 0.72 if len(sequence) >= 20 else 0.6
        return PredictionOutput(
            predicted_rul=float(predicted_rul),
            confidence=confidence,
            model_name="heuristic",
            model_version="heuristic-v1",
            model_source=source.lower(),
            checkpoint_id=None,
            fallback_used=True,
        )

    def _resolve_checkpoint(self, source: str, model_name: str) -> Path:
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


__all__ = ["PredictionOutput", "RULInferenceService"]
