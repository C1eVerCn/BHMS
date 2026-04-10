from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.schemas import AnomalyDetectionRequest
from backend.app.services import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/detect/anomaly")
def detect_anomaly(request: AnomalyDetectionRequest):
    current_data = request.current_data.model_dump() if request.current_data else None
    data = service.detect_anomaly(
        battery_id=request.battery_id,
        current_data=current_data,
        baseline_capacity=request.baseline_capacity,
    )
    return success_response(data, message="异常检测完成")
