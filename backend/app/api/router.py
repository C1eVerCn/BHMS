"""聚合路由。"""

from fastapi import APIRouter

from backend.app.api.routes.batteries import router as batteries_router
from backend.app.api.routes.dashboard import router as dashboard_router
from backend.app.api.routes.diagnosis import router as diagnosis_router
from backend.app.api.routes.predictions import router as predictions_router
from backend.app.api.routes.uploads import router as uploads_router

router = APIRouter()
router.include_router(dashboard_router, tags=["dashboard"])
router.include_router(uploads_router, tags=["data"])
router.include_router(batteries_router, tags=["batteries"])
router.include_router(predictions_router, tags=["prediction"])
router.include_router(diagnosis_router, tags=["diagnosis"])
