from fastapi import APIRouter, Query

from backend.app.core.responses import success_response
from backend.app.services import TrainingService

router = APIRouter()
service = TrainingService()


@router.get("/training/comparison")
def get_training_comparison(source: str = Query(...)):
    return success_response(service.get_comparison(source=source))


@router.get("/training/overview")
def get_training_overview():
    return success_response(service.get_overview())


@router.get("/training/experiments/{source}")
def get_training_experiment_detail(source: str):
    return success_response(service.get_experiment_detail(source=source))


@router.get("/training/ablations/{source}")
def get_training_ablation_summary(source: str):
    return success_response(service.get_ablation_summary(source=source))
