import requests
import time
import sys
import json

# --- Configuration ---
AIRFLOW_URL = "http://localhost:8080/api/v2"
AUTH = ("admin", "admin")
DAG_ID = "debug_library_check"
TASK_ID = "verify_imports"

def log(msg, status="INFO"):
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "ERROR": "\033[91m", "RESET": "\033[0m"}
    print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")

def trigger_dag():
    """Triggers the DAG and returns the DAG Run ID."""
    log(f"Triggering DAG: {DAG_ID}...")
    endpoint = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns"
    
    # We use a unique ID to avoid collision with previous manual runs
    run_id_suffix = int(time.time())
    payload = {"dag_run_id": f"integration_test_{run_id_suffix}", "conf": {}}
    
    try:
        response = requests.post(endpoint, json=payload, auth=AUTH)
        if response.status_code == 200:
            run_data = response.json()
            run_id = run_data['dag_run_id']
            log(f"DAG Triggered! Run ID: {run_id}", "SUCCESS")
            return run_id
        elif response.status_code == 404:
            log(f"DAG '{DAG_ID}' not found. Did you add the file to 'dags/' and wait 30s?", "ERROR")
            return None
        else:
            log(f"Failed to trigger DAG: {response.text}", "ERROR")
            return None
    except requests.exceptions.ConnectionError:
        log("Could not connect to Airflow. Is 'airflow-api-server' running on port 8080?", "ERROR")
        return None

def wait_for_completion(run_id):
    """Polls the DAG Run status until Success or Failed."""
    endpoint = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns/{run_id}"
    
    log("Waiting for execution...")
    for _ in range(30): # Wait up to 30 seconds
        response = requests.get(endpoint, auth=AUTH)
        state = response.json().get('state')
        
        if state == 'success':
            log("DAG Run Success!", "SUCCESS")
            return True
        elif state == 'failed':
            log("DAG Run Failed!", "ERROR")
            return True # Return True to proceed to log analysis (to see WHY it failed)
        
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(1)
        
    log("\nTimeout waiting for DAG.", "ERROR")
    return False

def check_task_logs(run_id):
    """
    Fetches the logs for the specific task to verify library output.
    Note: Airflow API returns logs wrapped in metadata.
    """
    # Attempt 1 is usually the first execution
    try_number = 1
    endpoint = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances/{TASK_ID}/logs/{try_number}"
    
    log(f"Fetching logs for task: {TASK_ID}...")
    response = requests.get(endpoint, auth=AUTH)
    
    if response.status_code != 200:
        log(f"Could not fetch logs: {response.text}", "ERROR")
        return False

    # The API returns raw text content
    logs = response.text
    
    # 1. Check for Critical Success Message
    if "SUCCESS: pvp_core_lib imported" in logs:
        log("Verified: pvp_core_lib was imported successfully inside the container.", "SUCCESS")
    else:
        log("FAILED: The library import success message was NOT found in logs.", "ERROR")
        print("--- PARTIAL LOGS ---")
        print(logs[-500:]) # Print last 500 chars for debugging
        return False

    # 2. Check for Preprocessor Function
    if "SUCCESS: Imported preprocess_data" in logs:
        log("Verified: preprocess_data function is available.", "SUCCESS")
    else:
        log("FAILED: preprocess_data import failed.", "ERROR")
        return False
        
    return True

if __name__ == "__main__":
    log("--- Starting Airflow Integration Test ---")
    
    # Prerequisite Check
    try:
        requests.get(f"{AIRFLOW_URL}")
    except:
        log("Airflow is not reachable. Start Docker first!", "ERROR")
        sys.exit(1)

    run_id = trigger_dag()
    if run_id:
        if wait_for_completion(run_id):
            time.sleep(2) # Give logs a moment to sync to API
            check_task_logs(run_id)