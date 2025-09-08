import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import json

from app.core.config import settings
from app.core.exceptions import (
    DatasetNotFoundError, ModelNotFoundError, FileProcessingError, 
    ModelTrainingError, InvalidDatasetError
)
from app.models.dataset import (
    DatasetInfo, ModelInfo, ModelTrainingConfig, SampleRequest,
    UploadResponse, TrainingResponse, SamplingResponse
)
from app.services.storage_service import storage_service
from app.services.model_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}")
):
    """
    Upload a CSV dataset with optional metadata
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Only CSV files are supported"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.UPLOAD_MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.UPLOAD_MAX_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Parse metadata
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    try:
        # Save dataset
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
    """Delete a dataset and its associated model"""
    try:
        storage_service.delete_dataset(dataset_id)
        model_service.delete_model(dataset_id)
        
        logger.info(f"Dataset deleted: {dataset_id}")
        return {"message": "Dataset deleted successfully"}
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

def train_model_background(dataset_id: str, config: ModelTrainingConfig):
    """Background task for model training"""
    try:
        model_service.train_model(dataset_id, config)
        logger.info(f"Background training completed for dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Background training failed for dataset {dataset_id}: {str(e)}")

@router.post("/{dataset_id}/train", response_model=TrainingResponse)
async def train_model(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    config: Optional[ModelTrainingConfig] = None
):
    """
    Train a PATE-GAN model on the dataset
    Training runs in the background for large datasets
    """
    try:
        # Validate dataset exists
        dataset_info = storage_service.get_dataset_info(dataset_id)
        
        # Use default config if none provided
        if config is None:
            config = ModelTrainingConfig()
        
        # For small datasets, train synchronously
        if dataset_info.rows < 1000:
            model_info = model_service.train_model(dataset_id, config)
            
            return TrainingResponse(
                dataset_id=dataset_id,
                message="Model trained successfully",
                model_info=model_info
            )
        else:
            # For large datasets, train in background
            background_tasks.add_task(train_model_background, dataset_id, config)
            
            # Update status to training
            storage_service.update_model_status(dataset_id, "training")
            
            return TrainingResponse(
                dataset_id=dataset_id,
                message="Model training started in background",
                model_info=ModelInfo(
                    dataset_id=dataset_id,
                    status="training",
                    config=config
                )
            )
            
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelTrainingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Training request failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Training request failed")

@router.get("/{dataset_id}/model", response_model=ModelInfo)
async def get_model_info(dataset_id: str):
    """Get information about the trained model"""
    try:
        return model_service.get_model_info(dataset_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{dataset_id}/sample", response_model=SamplingResponse)
async def generate_samples(dataset_id: str, request: SampleRequest):
    """Generate synthetic samples from trained model"""
    try:
        # Validate request
        if request.dataset_id != dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID mismatch")
        
        # Generate samples
        samples = model_service.generate_samples(
            dataset_id, 
            request.num_rows, 
            request.conditions
        )
        
        # Get current privacy spent
        privacy_spent = model_service.get_privacy_spent(dataset_id)
        
        logger.info(f"Generated {len(samples)} samples for dataset {dataset_id}")
        
        return SamplingResponse(
            dataset_id=dataset_id,
            num_rows=len(samples),
            synthetic_data=samples,
            privacy_spent=privacy_spent
        )
        
    except (DatasetNotFoundError, ModelNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelTrainingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Sample generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Sample generation failed")

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