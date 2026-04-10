"""聚合路由。"""

from fastapi import APIRouter

from backend.app.api.routes.anomalies import router as anomalies_router
from backend.app.api.routes.batteries import router as batteries_router
from backend.app.api.routes.dashboard import router as dashboard_router
from backend.app.api.routes.insights import router as insights_router
from backend.app.api.routes.reports import router as reports_router
from backend.app.api.routes.training import router as training_router
from backend.app.api.routes.uploads import router as uploads_router

router = APIRouter()
router.include_router(dashboard_router, tags=["dashboard"])
router.include_router(insights_router, tags=["insights"])
router.include_router(uploads_router, tags=["data"])
router.include_router(batteries_router, tags=["batteries"])
router.include_router(anomalies_router, tags=["anomaly"])
router.include_router(training_router, tags=["training"])
router.include_router(reports_router, tags=["reports"])
