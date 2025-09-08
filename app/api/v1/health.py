from fastapi import APIRouter
from typing import Dict, Any
import psutil
import torch
from datetime import datetime

from app.core.config import settings
from app.services.storage_service import storage_service

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION
    }

@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """Detailed health check with system information"""
    
    # System metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    # Storage info
    try:
        datasets = storage_service.list_datasets()
        dataset_count = len(datasets)
        storage_accessible = True
    except Exception:
        dataset_count = 0
        storage_accessible = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "system": {
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "gpu": {
                "available": gpu_available,
                "count": gpu_count
            }
        },
        "storage": {
            "accessible": storage_accessible,
            "dataset_count": dataset_count
        },
        "config": {
            "data_dir": settings.DATA_DIR,
            "model_dir": settings.MODEL_DIR,
            "debug_mode": settings.DEBUG
        }
    }