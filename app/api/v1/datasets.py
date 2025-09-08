import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
import json

from app.core.config import settings
from app.core.exceptions import (
    DatasetNotFoundError, ModelNotFoundError, FileProcessingError, 
    ModelTrainingError, InvalidDatasetError
)
from app.models.dataset import (
    DatasetInfo, ModelInfo, ModelTrainingConfig, SampleRequest,
    UploadResponse, TrainingResponse, SamplingResponse, ModelStatus
)
from app.services.storage_service import storage_service
from app.services.model_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)

### FILE UPLOAD AND MANAGEMENT ###

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

@router.get("/{dataset_id}/stats")
async def get_dataset_statistics(dataset_id: str):
    """Get statistical information about the stored dataset"""
    try:
        df = storage_service.get_dataset(dataset_id)
        
        # Basic statistics
        stats = {
            "dataset_id": dataset_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Add basic descriptive statistics for numeric columns
        if stats["numeric_columns"]:
            numeric_stats = df[stats["numeric_columns"]].describe().to_dict()
            stats["numeric_statistics"] = numeric_stats
        
        return stats
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Stats generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate dataset statistics")

@router.post("/{dataset_id}/validate")
async def validate_dataset_for_training(dataset_id: str):
    """
    Validate that a dataset is suitable for PATE-GAN training.
    Checks data quality, size, and format requirements.
    """
    try:
        df = storage_service.get_dataset(dataset_id)
        dataset_info = storage_service.get_dataset_info(dataset_id)
        
        validation_results = {
            "dataset_id": dataset_id,
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check minimum rows
        if len(df) < 100:
            validation_results["errors"].append("Dataset has fewer than 100 rows - insufficient for training")
            validation_results["valid"] = False
        elif len(df) < 1000:
            validation_results["warnings"].append("Dataset has fewer than 1000 rows - may affect model quality")
        
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50].to_dict()
        if high_missing:
            validation_results["warnings"].append(f"Columns with >50% missing values: {list(high_missing.keys())}")
        
        # Check column diversity
        if len(df.columns) < 3:
            validation_results["warnings"].append("Dataset has very few columns - may limit synthetic data utility")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation_results["warnings"].append(f"Constant columns detected: {constant_cols}")
        
        # Recommendations
        if len(df) >= 10000:
            validation_results["recommendations"].append("Large dataset detected - consider using background training")
        
        if len(df.select_dtypes(include=['object']).columns) > len(df.columns) * 0.8:
            validation_results["recommendations"].append("Many categorical columns - consider feature engineering")
        
        return validation_results
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Validation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Dataset validation failed")


### TRAINING ###

@router.get("/models/status")
async def list_all_model_status():
    """Get training status for all models"""
    try:
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


### SAMPLING ###
@router.post("/{dataset_id}/sample", response_model=SamplingResponse)
async def generate_samples(
    dataset_id: str,
    num_rows: int = Query(..., ge=1, le=10000),
    conditions: Optional[Dict[str, Any]] = Body(None),
    format: str = Query("json", description="Output format: json or csv")
):
    """
    Generate synthetic samples using the locally stored trained model.
    Model instances are not transferred - only generated data is returned.
    """
    try:
        # Create SampleRequest object from individual parameters
        request = SampleRequest(
            dataset_id=dataset_id,
            num_rows=num_rows,
            conditions=conditions
        )
        
        # Rest of the function remains the same...
        # Validate that model exists and is trained
        model_info = model_service.get_model_info(dataset_id)
        if model_info.status != ModelStatus.TRAINED:
            if model_info.status == ModelStatus.TRAINING:
                raise HTTPException(
                    status_code=400, 
                    detail="Model is still training. Please wait for training to complete."
                )
            elif model_info.status == ModelStatus.NOT_TRAINED:
                raise HTTPException(
                    status_code=400,
                    detail="No trained model found. Please train a model first using /train endpoint."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model is in {model_info.status} state and cannot generate samples."
                )
        
        # Generate samples using local model
        samples = model_service.generate_samples(
            dataset_id, 
            request.num_rows, 
            request.conditions
        )
        
        # Get current privacy spent
        privacy_spent = model_service.get_privacy_spent(dataset_id)
        
        logger.info(f"Generated {len(samples)} samples for dataset {dataset_id}")
        
        response_data = SamplingResponse(
            dataset_id=dataset_id,
            num_rows=len(samples),
            synthetic_data=samples,
            privacy_spent=privacy_spent
        )
        
        # Return different formats based on request
        if format.lower() == "csv":
            # Convert to CSV format for download
            import pandas as pd
            from fastapi.responses import StreamingResponse
            import io
            
            df = pd.DataFrame(samples)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            
            return StreamingResponse(
                io.BytesIO(stream.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=synthetic_data_{dataset_id}.csv"}
            )
        else:
            return response_data
        
    except (DatasetNotFoundError, ModelNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelTrainingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Sample generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Sample generation failed")

@router.get("/{dataset_id}/sample/batch")
async def generate_batch_n_samples(
    dataset_id: str,
    batch_size: int = Query(1000, ge=100, le=5000, description="Samples per batch"),
    num_batches: int = Query(1, ge=1, le=10, description="Number of batches"),
    background_tasks: BackgroundTasks = None
):
    """
    Generate large batches of synthetic data.
    For very large requests, generation happens in background.
    """
    total_samples = batch_size * num_batches
    
    try:
        # Validate model exists and is trained
        model_info = model_service.get_model_info(dataset_id)
        if model_info.status != ModelStatus.TRAINED:
            raise HTTPException(
                status_code=400,
                detail="Model must be trained before generating samples"
            )
        
        # For smaller requests, generate immediately
        if total_samples <= 2000:
            samples = model_service.generate_samples(dataset_id, total_samples)
            return {
                "dataset_id": dataset_id,
                "total_samples": len(samples),
                "batches_completed": num_batches,
                "status": "completed",
                "data": samples[:100],  # Return first 100 for preview
                "message": f"Generated {len(samples)} samples immediately"
            }
        else:
            # For large requests, use background processing
            return {
                "dataset_id": dataset_id,
                "total_samples_requested": total_samples,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "status": "queued",
                "message": "Large batch generation not fully implemented - use regular sample endpoint"
            }
            
    except Exception as e:
        logger.error(f"Batch generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch generation failed")

### PRIVACY CHECK ###

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