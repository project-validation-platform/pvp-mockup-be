import logging
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime

from pvp_core_lib.models import ModelTrainingConfig, ModelInfo, ModelStatus
from pvp_core_lib.storage import storage_service
from pvp_core_lib.exceptions import ModelNotFoundError

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        logger.info("ModelService initialized (Control Plane)")
    
    def get_model_info(self, dataset_id: str) -> ModelInfo:
        """Get model information directly from storage/metadata"""
        try:
            return storage_service.get_model_info(dataset_id)
        except ModelNotFoundError:
            return ModelInfo(
                dataset_id=dataset_id,
                status=ModelStatus.NOT_TRAINED
            )

    def list_models(self) -> List[ModelInfo]:
        """List all models with their status from storage"""
        return storage_service.list_models()
    
    def get_privacy_spent(self, dataset_id: str) -> float:
        """Get privacy spent from storage"""
        return storage_service.get_privacy_spent(dataset_id)

    def trigger_training_workflow(self, dataset_id: str, config: ModelTrainingConfig) -> str:
        """
        Triggers an Airflow DAG to run the training.
        Returns the Airflow run_id.
        """
        AIRFLOW_URL = settings.AIRFLOW_API_URL # e.g., "http://localhost:8080/api/v2"
        AIRFLOW_USER = settings.AIRFLOW_API_USER
        AIRFLOW_PASS = settings.AIRFLOW_API_PASS
        
        dag_id = "pvp_workflow_runner" # The name of the DAG you will create in Airflow
        endpoint = f"{AIRFLOW_URL}/dags/{dag_id}/dagRuns"
        
        # This is the configuration payload sent to Airflow
        payload = {
            "conf": {
                "dataset_id": dataset_id,
                "workflow_type": "train",
                "config": config.model_dump()
            }
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                auth=(AIRFLOW_USER, AIRFLOW_PASS)
            )
            response.raise_for_status() # Raise an error for bad status codes
            
            run_data = response.json()
            run_id = run_data.get('dag_run_id')
            
            if not run_id:
                raise Exception("Airflow API did not return a dag_run_id")
                
            logger.info(f"Triggered Airflow DAG run {run_id} for dataset {dataset_id}")

            # Update your local DB status to "TRAINING"
            storage_service.update_model_status(
                dataset_id, 
                ModelStatus.TRAINING,
                config=config.model_dump(),
                training_started_at=datetime.now().isoformat(),
                airflow_run_id=run_id # Store the run_id!
            )
            
            return run_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger Airflow DAG: {e}")
            raise Exception(f"Failed to queue training job: {e}")

# Global model service instance
model_service = ModelService()