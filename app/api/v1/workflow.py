import requests
import logging

from app.core.config import settings

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/runs/{run_id}/status")
async def get_workflow_run_status(run_id: str):
    """
    Gets the status of a specific Airflow DAG run.
    This is what your frontend will poll.
    """

    AIRFLOW_URL = settings.AIRFLOW_API_URL
    AIRFLOW_USER = settings.AIRFLOW_API_USER
    AIRFLOW_PASS = settings.AIRFLOW_API_PASS
    
    dag_id = "pvp_workflow_runner"
    endpoint = f"{AIRFLOW_URL}/dags/{dag_id}/dagRuns/{run_id}"

    try:
        response = requests.get(endpoint, auth=(AIRFLOW_USER, AIRFLOW_PASS))
        response.raise_for_status()
        
        run_data = response.json()
        return {
            "run_id": run_data.get("dag_run_id"),
            "status": run_data.get("state"),
            "start_date": run_data.get("start_date"),
            "end_date": run_data.get("end_date")
        }
        
    except requests.exceptions.RequestException as e:
        logger.warn(f"Failed to get Airflow run status: {e}")

        raise HTTPException(status_code=404, detail=f"Job run {run_id} not found or Airflow API is down.")