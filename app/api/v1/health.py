from fastapi import APIRouter
from typing import Dict, Any

import torch
from datetime import datetime

from app.core.config import settings
from pvp_core_lib.storage import storage_service

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION
    }