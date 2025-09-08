# app/models/dataset.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing" 
    READY = "ready"
    ERROR = "error"

class ModelStatus(str, Enum):
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"

class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    columns: List[str]
    rows: int
    status: DatasetStatus
    uploaded_at: str
    metadata: Optional[Dict[str, Any]] = None

class ModelTrainingConfig(BaseModel):
    latent_dim: int = Field(default=128, ge=16, le=1024)
    teacher_epochs: int = Field(default=100, ge=10, le=1000)
    student_epochs: int = Field(default=100, ge=10, le=1000) 
    generator_epochs: int = Field(default=100, ge=10, le=1000)
    num_teachers: int = Field(default=10, ge=2, le=50)
    epsilon: float = Field(default=1.0, gt=0, le=10.0)
    delta: float = Field(default=1e-5, gt=0, lt=1.0)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.1)
    batch_size: int = Field(default=64, ge=8, le=512)
    teacher_batch_size: int = Field(default=32, ge=8, le=256)
    lambda_gradient_penalty: float = Field(default=10.0, ge=0)
    noise_multiplier: float = Field(default=1.0, ge=0)

class ModelInfo(BaseModel):
    dataset_id: str
    status: ModelStatus
    config: Optional[ModelTrainingConfig] = None
    trained_at: Optional[str] = None
    training_time: Optional[float] = None  # seconds
    privacy_spent: Optional[float] = None
    
class SampleRequest(BaseModel):
    dataset_id: str
    num_rows: int = Field(ge=1, le=10000)
    conditions: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    dataset_id: str
    message: str
    dataset_info: DatasetInfo

class TrainingResponse(BaseModel):
    dataset_id: str
    message: str
    model_info: ModelInfo

class SamplingResponse(BaseModel):
    dataset_id: str
    num_rows: int
    synthetic_data: List[Dict[str, Any]]
    privacy_spent: Optional[float] = None

class ErrorResponse(BaseModel):
    message: str
    details: Optional[Dict[str, Any]] = None