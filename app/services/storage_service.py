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
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
    
    def _get_metadata_path(self, dataset_id: str) -> Path:
        """Get the metadata file path for a dataset"""
        return self.data_dir / f"{dataset_id}_metadata.json"
    
    def _load_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Load metadata from JSON file for a specific dataset"""
        metadata_path = self._get_metadata_path(dataset_id)
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"dataset": None, "model": None}
        return {"dataset": None, "model": None}
    
    def _save_metadata(self, dataset_id: str, metadata: Dict[str, Any]):
        """Save metadata to JSON file for a specific dataset"""
        try:
            metadata_path = self._get_metadata_path(dataset_id)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except IOError as e:
            raise FileProcessingError(f"Failed to save metadata: {str(e)}")
    
    def _get_dataset_csv_path(self, dataset_id: str) -> Path:
        """Get the CSV file path for a dataset"""
        return self.data_dir / f"{dataset_id}.csv"
    
    def _get_model_pkl_path(self, dataset_id: str) -> Path:
        """Get the model pickle file path for a dataset"""
        return self.model_dir / f"{dataset_id}.pkl"
    
    def save_dataset(self, file_contents: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> DatasetInfo:
        """Save uploaded dataset and return dataset info"""
        dataset_id = str(uuid.uuid4())
        csv_path = self._get_dataset_csv_path(dataset_id)
        
        try:
            # Save CSV file
            with open(csv_path, 'wb') as f:
                f.write(file_contents)
            
            # Load and validate CSV
            df = pd.read_csv(csv_path)
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
            
            # Load existing metadata (in case model info exists)
            existing_metadata = self._load_metadata(dataset_id)
            
            # Update with dataset info
            existing_metadata["dataset"] = dataset_info.model_dump()
            
            # Save metadata
            self._save_metadata(dataset_id, existing_metadata)
            
            return dataset_info
            
        except pd.errors.EmptyDataError:
            # Cleanup on error
            if csv_path.exists():
                csv_path.unlink()
            raise FileProcessingError("Invalid CSV file - no data found")
        except pd.errors.ParserError as e:
            # Cleanup on error
            if csv_path.exists():
                csv_path.unlink()
            raise FileProcessingError(f"Invalid CSV format: {str(e)}")
        except Exception as e:
            # Cleanup file if something went wrong
            if csv_path.exists():
                csv_path.unlink()
            raise FileProcessingError(f"Failed to process dataset: {str(e)}")
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset as DataFrame"""
        # Check if dataset exists in metadata
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("dataset"):
            raise DatasetNotFoundError(dataset_id)
        
        csv_path = self._get_dataset_csv_path(dataset_id)
        if not csv_path.exists():
            raise DatasetNotFoundError(dataset_id)
        
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            raise FileProcessingError(f"Failed to load dataset: {str(e)}")
    
    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get dataset metadata"""
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("dataset"):
            raise DatasetNotFoundError(dataset_id)
        
        return DatasetInfo(**metadata["dataset"])
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all datasets"""
        datasets = []
        
        # Find all metadata files
        for metadata_file in self.data_dir.glob("*_metadata.json"):
            try:
                dataset_id = metadata_file.stem.replace("_metadata", "")
                metadata = self._load_metadata(dataset_id)
                if metadata.get("dataset"):
                    datasets.append(DatasetInfo(**metadata["dataset"]))
            except Exception:
                # Skip corrupted metadata files
                continue
        
        return datasets
    
    def delete_dataset(self, dataset_id: str):
        """Delete dataset and associated model"""
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("dataset"):
            raise DatasetNotFoundError(dataset_id)
        
        # Delete files
        csv_path = self._get_dataset_csv_path(dataset_id)
        model_path = self._get_model_pkl_path(dataset_id)
        metadata_path = self._get_metadata_path(dataset_id)
        
        if csv_path.exists():
            csv_path.unlink()
        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    
    def save_model(self, dataset_id: str, model, model_info: ModelInfo):
        """Save trained model"""
        # Check if dataset exists
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("dataset"):
            raise DatasetNotFoundError(dataset_id)
        
        model_path = self._get_model_pkl_path(dataset_id)
        
        try:
            # Save model using pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Update metadata with model info
            metadata["model"] = model_info.model_dump()
            self._save_metadata(dataset_id, metadata)
            
        except Exception as e:
            raise FileProcessingError(f"Failed to save model: {str(e)}")
    
    def load_model(self, dataset_id: str):
        """Load trained model"""
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("model"):
            raise ModelNotFoundError(dataset_id)
        
        model_path = self._get_model_pkl_path(dataset_id)
        if not model_path.exists():
            raise ModelNotFoundError(dataset_id)
        
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise FileProcessingError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self, dataset_id: str) -> ModelInfo:
        """Get model metadata"""
        metadata = self._load_metadata(dataset_id)
        if not metadata.get("model"):
            raise ModelNotFoundError(dataset_id)
        
        return ModelInfo(**metadata["model"])
    
    def update_model_status(self, dataset_id: str, status: ModelStatus, **kwargs):
        """Update model status and other info"""
        metadata = self._load_metadata(dataset_id)
        
        # Initialize model metadata if it doesn't exist
        if not metadata.get("model"):
            metadata["model"] = {
                "dataset_id": dataset_id,
                "status": status.value
            }
        
        # Update status and other fields
        model_data = metadata["model"]
        model_data["status"] = status.value
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(ModelInfo, key):
                if key.endswith('_at') and isinstance(value, str):
                    # Ensure datetime fields are properly formatted
                    model_data[key] = value
                else:
                    model_data[key] = value
        
        # Save updated metadata
        self._save_metadata(dataset_id, metadata)
    
    def model_exists(self, dataset_id: str) -> bool:
        """Check if model exists for dataset"""
        try:
            metadata = self._load_metadata(dataset_id)
            model_info = metadata.get("model")
            return (model_info is not None and 
                    model_info.get("status") == ModelStatus.TRAINED.value and
                    self._get_model_pkl_path(dataset_id).exists())
        except Exception:
            return False
    
    def get_all_dataset_ids(self) -> List[str]:
        """Get all dataset IDs that have metadata files"""
        dataset_ids = []
        for metadata_file in self.data_dir.glob("*_metadata.json"):
            dataset_id = metadata_file.stem.replace("_metadata", "")
            dataset_ids.append(dataset_id)
        return dataset_ids
    
    def cleanup_orphaned_files(self):
        """Clean up files that don't have corresponding metadata"""
        valid_dataset_ids = set(self.get_all_dataset_ids())
        
        # Clean up orphaned CSV files
        for csv_file in self.data_dir.glob("*.csv"):
            dataset_id = csv_file.stem
            if dataset_id not in valid_dataset_ids:
                csv_file.unlink()
        
        # Clean up orphaned model files
        for pkl_file in self.model_dir.glob("*.pkl"):
            dataset_id = pkl_file.stem
            if dataset_id not in valid_dataset_ids:
                pkl_file.unlink()

# Global storage service instance
storage_service = StorageService()