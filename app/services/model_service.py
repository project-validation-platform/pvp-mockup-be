import logging
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime


from pvp_core_lib.models import ModelTrainingConfig, ModelInfo, ModelStatus
from pvp_core_lib.storage import storage_service
from pvp_core_lib.exceptions import ModelNotFoundError

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Airflow Configuration ---
# REMINDER: You must add these settings to your `app/core/config.py`
AIRFLOW_API_URL = getattr(settings, "AIRFLOW_API_URL", "http://localhost:8080/api/v2")
AIRFLOW_API_USER = getattr(settings, "AIRFLOW_API_USER", "admin")
AIRFLOW_API_PASS = getattr(settings, "AIRFLOW_API_PASS", "admin")
WORKFLOW_DAG_ID = "pvp_workflow_runner"

class ModelService:
    """
    Refactored Model Service for the Control Plane.
    
    This service is stateless. It does not hold models in memory.
    It reads/writes metadata from the shared storage_service and
    triggers Airflow workflows for all heavy computation.
    """
    
    def __init__(self):
        logger.info("ModelService initialized (Control Plane Mode)")
    
    # --- READ-ONLY METHODS ---
    # These are thin wrappers around the pvp_core_lib.storage_service

    def get_model_info(self, dataset_id: str) -> ModelInfo:
        """Get model information directly from storage/metadata"""
        try:
            return storage_service.get_model_info(dataset_id)
        except ModelNotFoundError:
            # If no model metadata exists, return a "not_trained" status
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
    
    # --- WORKFLOW TRIGGER METHODS ---

    def _trigger_airflow_dag(self, conf: dict) -> str:
        """
        Private helper to make the API call to trigger an Airflow DAG.
        
        Args:
            conf (dict): The 'conf' payload to pass to the Airflow DAG.
        
        Returns:
            str: The `dag_run_id` for the triggered workflow.
        """
        endpoint = f"{AIRFLOW_API_URL}/dags/{WORKFLOW_DAG_ID}/dagRuns"
        payload = {"conf": conf}
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                auth=(AIRFLOW_API_USER, AIRFLOW_API_PASS)
            )
            response.raise_for_status() # Raise an error for bad status codes
            
            run_data = response.json()
            run_id = run_data.get('dag_run_id')
            
            if not run_id:
                raise Exception("Airflow API did not return a dag_run_id")
                
            logger.info(f"Triggered Airflow DAG run {run_id} with conf: {conf}")
            return run_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger Airflow DAG {WORKFLOW_DAG_ID}: {e}")
            raise Exception(f"Failed to queue job via Airflow: {e}")


    def trigger_training_workflow(self, dataset_id: str, config: ModelTrainingConfig) -> str:
        """
        Triggers an Airflow DAG to run the training.
        """
        # 1. Define the "conf" payload for this job
        conf_payload = {
            "workflow_type": "train",
            "dataset_id": dataset_id,
            "config": config.model_dump() # Pass the Pydantic model as a dict
        }
        
        # 2. Trigger the DAG
        run_id = self._trigger_airflow_dag(conf_payload)
        
        # 3. Update our local database to show the job is queued
        storage_service.update_model_status(
            dataset_id, 
            ModelStatus.TRAINING,
            config=config.model_dump(),
            training_started_at=datetime.now().isoformat(),
            airflow_run_id=run_id # Store the run_id for polling
        )
        
        return run_id
    
    def trigger_sampling_workflow(self, dataset_id: str, num_rows: int, conditions: Optional[Dict[str, Any]]) -> str:
        """
        Triggers an Airflow DAG to run sampling.
        """
        # 1. Define the "conf" payload for this job
        conf_payload = {
            "workflow_type": "sample",
            "dataset_id": dataset_id,
            "num_rows": num_rows,
            "conditions": conditions
        }
        
        # 2. Trigger the DAG
        run_id = self._trigger_airflow_dag(conf_payload)
        
        # 3. (Optional) Store a record of this sampling job.
        # We don't change the model status (it's still "TRAINED"),
        # but you could write to a new "sampling_jobs" table here
        # or just log it and return the run_id.
        logger.info(f"Queued sampling job for dataset {dataset_id} with run_id {run_id}")
        
        return run_id

# Global model service instance
model_service = ModelService()