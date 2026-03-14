from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.schemas import AnomalyDetectionRequest, RULPredictionRequest
from backend.app.services import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/predict/rul")
def predict_rul(request: RULPredictionRequest):
    historical = [item.model_dump() for item in request.historical_data] if request.historical_data else None
    data = service.predict_rul(
        battery_id=request.battery_id,
        model_name=request.model_name,
        seq_len=request.seq_len,
        historical_data=historical,
    )
    return success_response(data, message="RUL 预测完成")


@router.post("/detect/anomaly")
def detect_anomaly(request: AnomalyDetectionRequest):
    current_data = request.current_data.model_dump() if request.current_data else None
    data = service.detect_anomaly(
        battery_id=request.battery_id,
        current_data=current_data,
        baseline_capacity=request.baseline_capacity,
    )
    return success_response(data, message="异常检测完成")
