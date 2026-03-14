from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.core.config import get_settings
from backend.app.core.responses import success_response
from backend.app.schemas import DataImportRequest
from backend.app.services import BatteryService

router = APIRouter()
service = BatteryService()
settings = get_settings()


@router.post("/data/upload")
async def upload_battery_data(file: UploadFile = File(...), battery_id: str | None = Form(default=None)):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    saved_path = settings.upload_dir / saved_name
    content = await file.read()
    saved_path.write_bytes(content)
    summary = service.import_csv_file(saved_path, battery_id_hint=battery_id)
    return success_response(summary, message="文件上传并导入成功")


@router.post("/data/import-nasa")
def import_nasa_dataset(request: DataImportRequest):
    summary = service.import_nasa(battery_ids=request.battery_ids)
    return success_response(summary, message="NASA 数据导入成功")
