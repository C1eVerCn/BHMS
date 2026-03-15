from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.app.core.responses import success_response
from backend.app.services.insight_service import InsightService
from backend.app.services import PredictionService

router = APIRouter()
service = PredictionService()
insight_service = InsightService()


@router.get("/reports/prediction/{prediction_id}")
def get_prediction_report(prediction_id: int):
    return PlainTextResponse(service.get_prediction_report(prediction_id), media_type="text/markdown; charset=utf-8")


@router.get("/reports/diagnosis/{diagnosis_id}")
def get_diagnosis_report(diagnosis_id: int):
    return PlainTextResponse(service.get_diagnosis_report(diagnosis_id), media_type="text/markdown; charset=utf-8")


@router.get("/reports/case-bundle/{battery_id}")
def get_case_bundle(battery_id: str):
    return success_response(insight_service.get_case_bundle(battery_id), message="案例包已生成")


@router.post("/reports/case-bundle/{battery_id}/export")
def export_case_bundle(battery_id: str, ensure_artifacts: bool = True):
    return success_response(
        insight_service.export_case_bundle(battery_id, ensure_artifacts=ensure_artifacts),
        message="案例目录已导出",
    )
