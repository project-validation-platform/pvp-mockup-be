from fastapi import APIRouter

from app.api.v1 import datasets, health, auth, sandbox

# Create main v1 router
api_router = APIRouter()

# Include all v1 sub-routers
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"]
)

api_router.include_router(
    health.router,
    prefix="/health", 
    tags=["health"]
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    sandbox.router,
    prefix="/sandbox",
    tags=["sandbox"]
)