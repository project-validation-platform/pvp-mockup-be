from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import io

from app.utils import preprocess_data


router = APIRouter()

# Pydantic models for sandbox endpoints
class DataPreviewRequest(BaseModel):
    dataset_id: str
    num_rows: int = 5

class DataPreviewResponse(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    sample_data: List[Dict[str, Any]]
    total_rows: int

class ModelComparisonRequest(BaseModel):
    dataset_id: str
    model_configs: List[Dict[str, Any]]

class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

#### --- Helper Functions --- ####
def get_df_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract dataframe information from a DataFrame
    """
    df_info = {}
    for col in df.columns:
        df_info[col] = {
            "dtype": str(df[col].dtype),
            "num_missing": int(df[col].isnull().sum()),
            "num_unique": int(df[col].nunique())
        }
    return df_info

#### --- Sandbox API Endpoints --- ####
@router.get("/")
async def sandbox_info():
    """
    Get information about available sandbox features
    """
    return {
        "message": "Sandbox API for testing and experimentation",
        "features": [
            "data_preview",
            "model_testing", 
            "parameter_tuning",
            "performance_comparison",
            "data_validation"
        ],
        "status": "development"
    }

@router.post("/data/preview", response_model=DataPreviewResponse)
async def preview_dataset(request: DataPreviewRequest):
    """
    Preview dataset structure and sample data
    TODO: Implement dataset preview functionality
    """

    try:
        df = pd.read_csv(request.dataset_id)
        preprocessed_df, feature_names, metadata = preprocess_data(df)
        preprocessed_df_head = preprocessed_df.head(request.num_rows)

        return DataPreviewResponse(
            columns=preprocessed_df_head.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in preprocessed_df_head.dtypes.items()},
            sample_data=preprocessed_df_head.to_dict(orient="records"),
            total_rows=len(pd.read_csv(request.dataset_id))
        )
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail="Dataset preview not implemented yet"
        )

@router.post("/data/validate")
async def validate_csv(file: UploadFile = File(...)):
    """
    Validate CSV file structure without saving
    TODO: Implement CSV validation logic
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        df = pd.read_csv(io.StringIO(raw.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    
    columns = df.columns.tolist()
    if not columns:
        raise HTTPException(status_code=400, detail="CSV has no columns")
    
    has_header = pd.read_csv(io.StringIO(raw.decode('utf-8')), nrows=0).columns.tolist() == columns
    
    cols_count = df.columns.value_counts().to_dict()
    dup_cols_count = {col: count for col, count in cols_count.itmes() if count > 1}
    
    empty_cols = [col for col in df.columns if df[col].isnull().all()]        

    return {
        "valid": empty_cols == [],
        "message": "OK" if len(dup_cols) == 0 and len(empty_cols) == 0 and has_header else "Potential issues found",
        "filename": file.filename,
        "size": file.size,
        "has_header": has_header,
        "duplicate_columns": dup_cols_count,
        "empty_columns": empty_cols
    }

@router.post("/models/test")
async def test_model_config(config: ExperimentConfig):
    """
    Test model configuration without full training
    TODO: Implement quick model validation
    """

    params = config.parameters or {}
    dataset_id = params.get("dataset_id", "")
    if not dataset_id:
        raise HTTPException(status_code = 400, detail = "parameters.dataset_id is missing")
    
    try:
        df = pd.read_csv(dataset_id, nrows = 1000)
        preprocessed_df, feature_names, metadata = preprocess_data(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail= f"Invalid dataset_id provided : {e}")
    
    issues: List[str] = []

    # check if target (output for ML) column exists and is valid
    target = params.get("target", "")
    
    if target:
        if target not in df.columns:
            issues.append(f"Target column '{target}' not found in dataset")
        
        if df[target].isnull().all():
            issues.append(f"Target column '{target}' contains only null values")
        elif df[target].isnull() > len(df) * 0.5:
            issues.append(f"Target column '{target}' has more than 50% missing values")
        
        if df[target].nunique() < 2:
            issues.append(f"Target column '{target}' must have at least 2 unique values")
    else:
        issues.append("parameters.target is missing")

    # check if features (inputs for ML) are valid and available
    features = params.get("features", [])
    if features:
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            issues.append(f"Features not found in dataset: {', '.join(missing_features)}")
    else:
        issues.append("parameters.features is missing")
    
    return {
        "name": config.name,
        "description": config.description,
        "valid": len(issues) == 0,
        "issues": issues,
        "dataframe info": get_df_info(df),
        "message": "Model configuration test completed"
    }


@router.post("/models/compare")
async def compare_models(request: ModelComparisonRequest):
    """
    Compare different model configurations
    TODO: Implement model comparison functionality
    """
    try:
        df = pd.read_csv(request.dataset_id)
    except Exception as e:  
        raise HTTPException(
            status_code=501,
            detail="Model comparison not implemented yet"
        )

@router.get("/experiments")
async def list_experiments():
    """
    List all sandbox experiments
    TODO: Implement experiment tracking
    """
    return {
        "experiments": [],
        "message": "Experiment tracking not implemented yet"
    }

@router.post("/experiments")
async def create_experiment(config: ExperimentConfig):
    """
    Create new sandbox experiment
    TODO: Implement experiment creation
    """
    raise HTTPException(
        status_code=501,
        detail="Experiment creation not implemented yet"
    )

@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """
    Delete sandbox experiment
    TODO: Implement experiment deletion
    """
    raise HTTPException(
        status_code=501,
        detail="Experiment deletion not implemented yet"
    )

@router.post("/data/synthetic/quality")
async def evaluate_synthetic_quality(dataset_id: str):
    """
    Evaluate quality of synthetic data
    TODO: Implement quality metrics calculation
    """
    raise HTTPException(
        status_code=501,
        detail="Quality evaluation not implemented yet"
    )

@router.get("/utils/system")
async def get_system_resources():
    """
    Get available system resources for sandbox testing
    TODO: Implement system resource monitoring for sandbox
    """
    return {
        "cpu_available": True,
        "gpu_available": False,
        "memory_available": "Unknown",
        "message": "System resource monitoring not fully implemented"
    }