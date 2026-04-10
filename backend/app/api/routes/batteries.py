from fastapi import APIRouter, Query

from backend.app.core.responses import success_response
from backend.app.schemas import UpdateTrainingCandidateRequest
from backend.app.services import BatteryService

router = APIRouter()
service = BatteryService()


@router.get("/batteries")
def list_batteries(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    return success_response(service.list_batteries(page=page, page_size=page_size))


@router.get("/batteries/options")
def list_battery_options():
    return success_response(service.list_battery_options())


@router.get("/battery/{battery_id}")
def get_battery(battery_id: str):
    return success_response(service.get_battery(battery_id))


@router.get("/battery/{battery_id}/cycles")
def get_battery_cycles(battery_id: str, limit: int = Query(120, ge=10, le=1000)):
    return success_response({"battery_id": battery_id, "items": service.get_cycles(battery_id, limit=limit)})


@router.get("/battery/{battery_id}/history")
def get_battery_history(battery_id: str):
    return success_response(service.get_history(battery_id))


@router.get("/battery/{battery_id}/health")
def get_battery_health(battery_id: str):
    return success_response(service.get_health(battery_id))


@router.post("/battery/{battery_id}/training-candidate")
def update_training_candidate(battery_id: str, request: UpdateTrainingCandidateRequest):
    return success_response(
        service.update_training_candidate(battery_id=battery_id, include_in_training=request.include_in_training),
        message="训练候选状态已更新",
    )
