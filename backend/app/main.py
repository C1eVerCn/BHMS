"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.router import router as api_router
from backend.app.core.config import get_settings
from backend.app.core.database import get_database
from backend.app.core.exceptions import BHMSException
from backend.app.core.responses import error_response, success_response
from backend.app.services import BatteryService

settings = get_settings()
database = get_database()
battery_service = BatteryService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize()
    battery_service.bootstrap_demo_data()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="毕业设计 MVP：真实数据导入、RUL 预测、异常检测与诊断闭环。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.exception_handler(BHMSException)
async def handle_bhms_exception(_: Request, exc: BHMSException):
    return JSONResponse(status_code=exc.status_code, content=error_response(exc.message, exc.code))


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content=error_response(f"服务器内部错误: {exc}", "internal_error"))


@app.get("/")
def root():
    return success_response(
        {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "api_prefix": settings.api_prefix,
        },
        message="BHMS backend is ready",
    )


@app.get("/health")
def health():
    return success_response({"status": "healthy"})
