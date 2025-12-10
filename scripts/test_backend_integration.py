import requests
import pandas as pd
import io
import sys

# Configuration
API_URL = "http://localhost:8000/api/v1"
UPLOAD_URL = f"{API_URL}/datasets/upload"
PREVIEW_URL = f"{API_URL}/sandbox/data/preview"

def log(msg, status="INFO"):
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "ERROR": "\033[91m", "RESET": "\033[0m"}
    print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")

def run_test():
    # 1. Create Dummy Data
    csv_content = "age,salary,city\n25,50000,New York\n30,60000,Paris\n35,75000,London"
    files = {"file": ("integration_test.csv", csv_content, "text/csv")}

    # 2. Upload (Tests Database + Storage)
    log("Step 1: Uploading test dataset...")
    try:
        resp = requests.post(UPLOAD_URL, files=files)
        if resp.status_code != 200:
            log(f"Upload Failed: {resp.text}", "ERROR")
            return
        
        dataset_id = resp.json()['dataset_id']
        log(f"Upload Success! ID: {dataset_id}", "SUCCESS")

    except Exception as e:
        log(f"Connection Error: {e}", "ERROR")
        return

    # 3. Preview (Tests pvp-core-lib Integration)
    #    This endpoint calls pvp_core_lib.utils.preprocessor.preprocess_data
    log("Step 2: Requesting Data Preview (Testing pvp-core-lib)...")
    try:
        payload = {"dataset_id": dataset_id, "num_rows": 5}
        resp = requests.post(PREVIEW_URL, json=payload)
        
        if resp.status_code == 200:
            data = resp.json()
            # If we see transformed columns, the lib is working!
            columns = data.get("columns", [])
            if "city_Paris" in columns or "salary" in columns:
                log(f"Backend Integration Perfect! Lib returned columns: {columns}", "SUCCESS")
            else:
                log(f"Response received but unexpected format: {data}", "ERROR")
        else:
            log(f"Preview Failed (Is pvp-core-lib installed?): {resp.text}", "ERROR")

    except Exception as e:
        log(f"Request Error: {e}", "ERROR")

if __name__ == "__main__":
    run_test()