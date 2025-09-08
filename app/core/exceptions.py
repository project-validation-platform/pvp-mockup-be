from fastapi import HTTPException
from typing import Any, Dict, Optional

class AppException(Exception):
    """Base exception for application errors"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class DatasetNotFoundError(AppException):
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Dataset with ID '{dataset_id}' not found",
            status_code=404,
            details={"dataset_id": dataset_id}
        )

class ModelNotFoundError(AppException):
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Model for dataset ID '{dataset_id}' not found",
            status_code=404,
            details={"dataset_id": dataset_id}
        )

class ModelNotFittedError(AppException):
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Model for dataset ID '{dataset_id}' has not been trained yet",
            status_code=400,
            details={"dataset_id": dataset_id}
        )

class InvalidDatasetError(AppException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Invalid dataset: {message}",
            status_code=400,
            details=details
        )

class FileProcessingError(AppException):
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=f"File processing error: {message}",
            status_code=400,
            details={"filename": filename} if filename else {}
        )

class ModelTrainingError(AppException):
    def __init__(self, message: str, dataset_id: Optional[str] = None):
        super().__init__(
            message=f"Model training error: {message}",
            status_code=500,
            details={"dataset_id": dataset_id} if dataset_id else {}
        )

def convert_to_http_exception(exc: AppException) -> HTTPException:
    """Convert AppException to HTTPException for FastAPI"""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "message": exc.message,
            "details": exc.details
        }
    )