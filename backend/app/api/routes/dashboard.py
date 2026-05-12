from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.services import BatteryService

router = APIRouter()
service = BatteryService()


@router.get("/dashboard/summary")
def get_dashboard_summary():
    return success_response(service.get_dashboard())
