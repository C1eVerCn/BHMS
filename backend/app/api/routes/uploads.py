from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.core.config import get_settings
from backend.app.core.responses import success_response
from backend.app.schemas import DataImportRequest, DemoPresetImportRequest
from backend.app.services import BatteryService

router = APIRouter()
service = BatteryService()
settings = get_settings()


@router.post("/data/upload")
async def upload_battery_data(
    file: UploadFile = File(...),
    battery_id: str | None = Form(default=None),
    source: str | None = Form(default="auto"),
    include_in_training: bool = Form(default=False),
):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    saved_path = settings.upload_dir / saved_name
    content = await file.read()
    saved_path.write_bytes(content)
    summary = service.import_uploaded_file(
        saved_path,
        source=source,
        battery_id_hint=battery_id,
        include_in_training=include_in_training,
    )
    return success_response(summary, message="文件上传并导入成功")


@router.post("/data/import-source")
def import_source_dataset(request: DataImportRequest):
    summary = service.import_builtin_source(
        source=request.source,
        battery_ids=request.battery_ids,
        include_in_training=request.include_in_training,
    )
    return success_response(summary, message=f"{request.source.upper()} 数据导入成功")


@router.post("/data/import-demo-preset")
def import_demo_preset(request: DemoPresetImportRequest):
    summary = service.import_demo_preset(request.preset_name, include_in_training=request.include_in_training)
    return success_response(summary, message=f"演示样本 {request.preset_name} 导入成功")
