from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.schemas import LifecyclePredictionRequest, MechanismExplanationRequest
from backend.app.services import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/predict/lifecycle")
def predict_lifecycle(request: LifecyclePredictionRequest):
    historical = [item.model_dump() for item in request.historical_data] if request.historical_data else None
    data = service.predict_lifecycle(
        battery_id=request.battery_id,
        model_name=request.model_name,
        seq_len=request.seq_len,
        historical_data=historical,
    )
    return success_response(data, message="Lifecycle 预测完成")


@router.post("/explain/mechanism")
def explain_mechanism(request: MechanismExplanationRequest):
    anomalies = [item.model_dump() for item in request.anomalies] if request.anomalies else None
    data = service.explain_mechanism(
        battery_id=request.battery_id,
        anomalies=anomalies,
        battery_info=request.battery_info,
        model_name=request.model_name,
        seq_len=request.seq_len,
    )
    return success_response(data, message="机理解释完成")
