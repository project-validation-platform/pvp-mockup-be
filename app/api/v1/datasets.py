import logging
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse
import json
import pandas as pd
import io

from app.core.config import settings
from app.services.model_service import model_service

from pvp_core_lib.models import (
    DatasetInfo, ModelInfo, ModelTrainingConfig, SampleRequest,
    UploadResponse, TrainingResponse, SamplingResponse, ModelStatus
)
from pvp_core_lib.exceptions import (
    DatasetNotFoundError, ModelNotFoundError, FileProcessingError, 
    ModelTrainingError, InvalidDatasetError
)
from pvp_core_lib.storage import storage_service

router = APIRouter()
logger = logging.getLogger(__name__)

# --- NEW Response Model for Async Triggers ---
class WorkflowTriggerResponse(BaseModel):
    """
    Standard response for triggering an asynchronous workflow in Airflow.
    """
    dataset_id: str
    run_id: str
    message: str
    status: ModelStatus


### FILE UPLOAD AND MANAGEMENT (Largely Unchanged) ###
# These endpoints are lightweight and interact with storage, so they remain in the API.

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}")
):
    """
    Upload a CSV dataset with optional metadata
    [CHANGED] Now uses storage_service imported from pvp_core_lib
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=f"File name is not set. Please name your file."
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    contents = await file.read()
    if len(contents) > settings.UPLOAD_MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.UPLOAD_MAX_SIZE / 1024 / 1024:.1f}MB"
        )
    
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    try:
        dataset_info = storage_service.save_dataset(contents, file.filename, meta)
        logger.info(f"Dataset uploaded successfully: {dataset_info.dataset_id}")
        
        return UploadResponse(
            dataset_id=dataset_info.dataset_id,
            message="Dataset uploaded successfully",
            dataset_info=dataset_info
        )
        
    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.get("/", response_model=List[DatasetInfo])
async def list_datasets():
    """List all uploaded datasets"""
    try:
        return storage_service.list_datasets()
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve datasets")

@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get information about a specific dataset"""
    try:
        return storage_service.get_dataset_info(dataset_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset and its associated model
    [CHANGED] Removed call to model_service.delete_model, as
    storage_service.delete_dataset (from core lib) handles all file cleanup.
    """
    try:
        storage_service.delete_dataset(dataset_id)
        logger.info(f"Dataset deleted: {dataset_id}")
        return {"message": "Dataset deleted successfully"}
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

# [REMOVED] The `train_model_background` function is deleted.
# Airflow is now our background task executor.

@router.get("/{dataset_id}/stats")
async def get_dataset_statistics(dataset_id: str):
    """Get statistical information about the stored dataset"""
    try:
        df = storage_service.get_dataset(dataset_id)
        
        # This is a lightweight, read-only operation.
        # It's perfect to keep in the API.
        stats = {
            "dataset_id": dataset_id,
            "total_rows": len(df),
            # ... (rest of stats logic) ...
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        if stats["numeric_columns"]:
            numeric_stats = df[stats["numeric_columns"]].describe().to_dict()
            stats["numeric_statistics"] = numeric_stats
        
        return stats
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=44, detail=str(e))
    except Exception as e:
        logger.error(f"Stats generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate dataset statistics")

@router.post("/{dataset_id}/validate")
async def validate_dataset_for_training(dataset_id: str):
    """
    Validate that a dataset is suitable for training.
    This is also a fast, synchronous check.
    """
    try:
        df = storage_service.get_dataset(dataset_id)
        # ... (rest of validation logic) ...
        
        validation_results = {
            "dataset_id": dataset_id,
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        if len(df) < 100:
            validation_results["errors"].append("Dataset has fewer than 100 rows - insufficient for training")
            validation_results["valid"] = False
        
        # ... (etc)
        
        return validation_results
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Validation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Dataset validation failed")


### TRAINING (HEAVILY REFACTORED) ###

@router.get("/models/status")
async def list_all_model_status():
    """Get training status for all models"""
    try:
        # Calls the refactored model_service
        models = model_service.list_models()
        
        summary = {
            "total_datasets": len(models),
            "models_trained": len([m for m in models if m.status == ModelStatus.TRAINED]),
            "models_training": len([m for m in models if m.status == ModelStatus.TRAINING]),
            "models_not_trained": len([m for m in models if m.status == ModelStatus.NOT_TRAINED]),
            "models_error": len([m for m in models if m.status == ModelStatus.ERROR]),
            "models": models
        }
        return summary
    except Exception as e:
        logger.error(f"Failed to list model status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model status")

@router.post("/{dataset_id}/train", response_model=WorkflowTriggerResponse)
async def train_model(
    dataset_id: str,
    config: Optional[ModelTrainingConfig] = None
):
    """
    [REFACTORED] Triggers an Airflow DAG to train a PATE-GAN model.
    This endpoint no longer uses BackgroundTasks and returns immediately.
    """
    try:
        # Validate dataset exists
        storage_service.get_dataset_info(dataset_id)
        
        if config is None:
            config = ModelTrainingConfig()
        
        # Calls the refactored model_service, which makes an API call to Airflow
        run_id = model_service.trigger_training_workflow(dataset_id, config)
            
        return WorkflowTriggerResponse(
            dataset_id=dataset_id,
            run_id=run_id,
            message=f"Training job queued successfully. Run ID: {run_id}",
            status=ModelStatus.TRAINING
        )
            
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Training request failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/model", response_model=ModelInfo)
async def get_model_info(dataset_id: str):
    """Get information about the trained model"""
    try:
        return model_service.get_model_info(dataset_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


### SAMPLING (HEAVILY REFACTORED) ###

@router.post("/{dataset_id}/sample", response_model=WorkflowTriggerResponse)
async def generate_samples(
    dataset_id: str,
    num_rows: int = Query(..., ge=1, le=10000),
    conditions: Optional[Dict[str, Any]] = Body(None)
):
    """
    [REFACTORED] Triggers an Airflow DAG to generate synthetic samples.
    This endpoint is now asynchronous and returns a job ID.
    The frontend will need to be updated to poll for results.
    """
    try:
        # Validate that model exists and is trained
        model_info = model_service.get_model_info(dataset_id)
        if model_info.status != ModelStatus.TRAINED:
            raise HTTPException(status_code=400, detail=f"Model is in {model_info.status} state.")
        
        # This is a new function you will add to your refactored model_service
        run_id = model_service.trigger_sampling_workflow(
            dataset_id, 
            num_rows, 
            conditions
        )
        
        return WorkflowTriggerResponse(
            dataset_id=dataset_id,
            run_id=run_id,
            message=f"Sampling job queued successfully. Run ID: {run_id}",
            status=ModelStatus.TRAINED # Model status is still trained
        )
        
    except (DatasetNotFoundError, ModelNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Sample generation request failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Sample generation request failed")

@router.get("/{dataset_id}/sample/batch")
async def generate_batch_n_samples(
    dataset_id: str,
    batch_size: int = Query(1000, ge=100, le=5000),
    num_batches: int = Query(1, ge=1, le=10)
):
    """
    [REFACTORED] Triggers a large batch generation job via Airflow.
    """
    total_samples = batch_size * num_batches
    
    try:
        model_info = model_service.get_model_info(dataset_id)
        if model_info.status != ModelStatus.TRAINED:
            raise HTTPException(status_code=400, detail="Model must be trained.")
        
        # This is a new function you will add to your refactored model_service
        run_id = model_service.trigger_sampling_workflow(
            dataset_id, 
            total_samples, 
            conditions=None
        )
        
        return {
            "dataset_id": dataset_id,
            "run_id": run_id,
            "total_samples_requested": total_samples,
            "status": "queued",
            "message": "Large batch generation job queued."
        }
            
    except Exception as e:
        logger.error(f"Batch generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch generation failed")

### PRIVACY CHECK (Unchanged) ###

@router.get("/{dataset_id}/privacy")
async def get_privacy_info(dataset_id: str):
    """Get privacy budget information for the model"""
    try:
        model_info = model_service.get_model_info(dataset_id)
        privacy_spent = model_service.get_privacy_spent(dataset_id)
        
        return {
            "dataset_id": dataset_id,
            "privacy_spent": privacy_spent,
            "epsilon": model_info.config.epsilon if model_info.config else None,
            "delta": model_info.config.delta if model_info.config else None,
            "privacy_remaining": max(0, (model_info.config.epsilon - privacy_spent)) if model_info.config else None
        }
        
    except (DatasetNotFoundError, ModelNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))