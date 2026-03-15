from fastapi import APIRouter, Query

from backend.app.core.responses import success_response
from backend.app.schemas import CreateTrainingJobRequest
from backend.app.services import TrainingService

router = APIRouter()
service = TrainingService()


@router.post("/training/jobs")
def create_training_job(request: CreateTrainingJobRequest):
    data = service.create_job(
        source=request.source,
        model_scope=request.model_scope,
        force_run=request.force_run,
        job_kind=request.job_kind,
        seed_count=request.seed_count,
    )
    return success_response(data, message="训练任务已创建")


@router.get("/training/jobs")
def list_training_jobs(source: str | None = Query(default=None)):
    return success_response(service.list_jobs(source=source))


@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: int):
    return success_response(service.get_job(job_id))


@router.get("/training/runs")
def list_training_runs(source: str | None = Query(default=None), model_type: str | None = Query(default=None)):
    return success_response(service.list_runs(source=source, model_type=model_type))


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
