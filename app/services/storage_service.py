import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, ModelNotFoundError, FileProcessingError
from app.models.dataset import DatasetInfo, ModelInfo, DatasetStatus, ModelStatus

class StorageService:
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)
        self.model_dir = Path(settings.MODEL_DIR)
        self.metadata_file = self.data_dir / "metadata.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"datasets": {}, "models": {}}
        return {"datasets": {}, "models": {}}
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except IOError as e:
            raise FileProcessingError(f"Failed to save metadata: {str(e)}")
    
    def save_dataset(self, file_contents: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> DatasetInfo:
        """Save uploaded dataset and return dataset info"""
        dataset_id = str(uuid.uuid4())
        file_path = self.data_dir / f"{dataset_id}.csv"
        
        try:
            # Save CSV file
            with open(file_path, 'wb') as f:
                f.write(file_contents)
            
            # Load and validate CSV
            df = pd.read_csv(file_path)
            if df.empty:
                raise FileProcessingError("Uploaded CSV file is empty")
            
            # Create dataset info
            dataset_info = DatasetInfo(
                dataset_id=dataset_id,
                filename=filename,
                columns=df.columns.tolist(),
                rows=len(df),
                status=DatasetStatus.READY,
                uploaded_at=datetime.utcnow().isoformat(),
                metadata=metadata
            )
            
            # Store in metadata
            self._metadata["datasets"][dataset_id] = dataset_info.dict()
            self._save_metadata()
            
            return dataset_info
            
        except pd.errors.EmptyDataError:
            raise FileProcessingError("Invalid CSV file - no data found")
        except pd.errors.ParserError as e:
            raise FileProcessingError(f"Invalid CSV format: {str(e)}")
        except Exception as e:
            # Cleanup file if something went wrong
            if file_path.exists():
                file_path.unlink()
            raise FileProcessingError(f"Failed to process dataset: {str(e)}")
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset as DataFrame"""
        if dataset_id not in self._metadata["datasets"]:
            raise DatasetNotFoundError(dataset_id)
        
        file_path = self.data_dir / f"{dataset_id}.csv"
        if not file_path.exists():
            raise DatasetNotFoundError(dataset_id)
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FileProcessingError(f"Failed to load dataset: {str(e)}")
    
    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get dataset metadata"""
        if dataset_id not in self._metadata["datasets"]:
            raise DatasetNotFoundError(dataset_id)
        
        return DatasetInfo(**self._metadata["datasets"][dataset_id])
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all datasets"""
        return [DatasetInfo(**info) for info in self._metadata["datasets"].values()]
    
    def delete_dataset(self, dataset_id: str):
        """Delete dataset and associated model"""
        if dataset_id not in self._metadata["datasets"]:
            raise DatasetNotFoundError(dataset_id)
        
        # Delete files
        dataset_file = self.data_dir / f"{dataset_id}.csv"
        model_file = self.model_dir / f"{dataset_id}.pkl"
        
        if dataset_file.exists():
            dataset_file.unlink()
        if model_file.exists():
            model_file.unlink()
        
        # Remove from metadata
        del self._metadata["datasets"][dataset_id]
        if dataset_id in self._metadata["models"]:
            del self._metadata["models"][dataset_id]
        
        self._save_metadata()
    
    def save_model(self, dataset_id: str, model, model_info: ModelInfo):
        """Save trained model"""
        if dataset_id not in self._metadata["datasets"]:
            raise DatasetNotFoundError(dataset_id)
        
        model_path = self.model_dir / f"{dataset_id}.pkl"
        
        try:
            # Save model using pickle (since SDV models may not have save method)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Update metadata
            self._metadata["models"][dataset_id] = model_info.dict()
            self._save_metadata()
            
        except Exception as e:
            raise FileProcessingError(f"Failed to save model: {str(e)}")
    
    def load_model(self, dataset_id: str):
        """Load trained model"""
        if dataset_id not in self._metadata["models"]:
            raise ModelNotFoundError(dataset_id)
        
        model_path = self.model_dir / f"{dataset_id}.pkl"
        if not model_path.exists():
            raise ModelNotFoundError(dataset_id)
        
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise FileProcessingError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self, dataset_id: str) -> ModelInfo:
        """Get model metadata"""
        if dataset_id not in self._metadata["models"]:
            raise ModelNotFoundError(dataset_id)
        
        return ModelInfo(**self._metadata["models"][dataset_id])
    
    def update_model_status(self, dataset_id: str, status: ModelStatus, **kwargs):
        """Update model status and other info"""
        if dataset_id not in self._metadata["models"]:
            # Create new model entry
            self._metadata["models"][dataset_id] = {
                "dataset_id": dataset_id,
                "status": status.value
            }
        
        # Update status and other fields
        model_data = self._metadata["models"][dataset_id]
        model_data["status"] = status.value
        
        for key, value in kwargs.items():
            if hasattr(ModelInfo, key):
                model_data[key] = value
        
        self._save_metadata()
    
    def model_exists(self, dataset_id: str) -> bool:
        """Check if model exists for dataset"""
        return (dataset_id in self._metadata["models"] and 
                self._metadata["models"][dataset_id].get("status") == ModelStatus.TRAINED.value)

# Global storage service instance
storage_service = StorageService()