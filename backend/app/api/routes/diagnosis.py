from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.schemas import DiagnosisRequest
from backend.app.services import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/diagnose")
def diagnose(request: DiagnosisRequest):
    data = service.diagnose(
        battery_id=request.battery_id,
        anomalies=[item.model_dump() for item in request.anomalies],
        battery_info=request.battery_info,
    )
    return success_response(data, message="故障诊断完成")
