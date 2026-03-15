from fastapi import APIRouter

from backend.app.core.responses import success_response
from backend.app.services.insight_service import InsightService

router = APIRouter()
service = InsightService()


@router.get('/system/status')
def get_system_status():
    return success_response(service.get_system_status())


@router.get('/demo/presets')
def get_demo_presets():
    return success_response(service.get_demo_presets())


@router.get('/data/profile/{source}')
def get_data_profile(source: str):
    return success_response(service.get_dataset_profile(source))


@router.get('/diagnosis/knowledge/summary')
def get_knowledge_summary():
    return success_response(service.get_knowledge_summary())
