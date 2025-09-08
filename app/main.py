from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import uuid
import os

from sdv.metadata import SingleTableMetadata
from synthesizers.synthetic_data_generation_model import PATEGANSynthesizer

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_all_models():
    model_dict = {}
    for f in os.listdir(MODEL_DIR):
        if os.path.isfile(os.path.join(MODEL_DIR, f)):
            filename_sections = f.split(".")
            if len(filename_sections) == 0:
                return model_dict
            dataset_id = filename_sections[0]
            model = PATEGANSynthesizer.load(f)
            model_dict[dataset_id] = model

    return model_dict
            

# In-memory stores for model names
MODELS = get_all_models()
METADATA = {}

@app.post("/upload/")
async def upload_dataset(
    file: UploadFile = File(...),
    metadata: str = Form(...)
):
    """
    Accepts a CSV file and a JSON metadata payload via FormData.
    Saves the file, stores metadata, and returns a dataset_id.
    """
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    # Save CSV
    with open(file_path, "wb") as out_file:
        contents = await file.read()
        out_file.write(contents)

    # Optionally, you could load and inspect the CSV here
    df = pd.read_csv(file_path)

    # Store metadata and columns
    METADATA[dataset_id] = meta

    return {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "received_metadata": meta
    }

@app.post("/fit/")
async def fit_model(dataset_id: str = Form(...)):
    file_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load data
    df = pd.read_csv(file_path)
    # Build or load metadata (SDV)
    sdv_meta = SingleTableMetadata()
    sdv_meta.detect_from_dataframe(data=df)

    # Fit model
    synthesizer = PATEGANSynthesizer(metadata=sdv_meta)
    synthesizer.fit(df)

    # Store model for sampling
    MODELS[dataset_id] = synthesizer

    model_filename = f"{dataset_id}.pkl"
    target_dir = os.path.join(MODEL_DIR, "PATE_GAN", model_filename)
    synthesizer.save(target_dir)

    return {"message": "Model trained and saved successfully", "dataset_id": dataset_id}

class SampleRequest(BaseModel):
    dataset_id: str
    num_rows: int

@app.post("/sample/")
def generate_samples(request: SampleRequest):
    synthesizer = MODELS.get(request.dataset_id)
    if synthesizer is None:
        raise HTTPException(status_code=404, detail="Model not found for this dataset ID")
    synthetic_df = synthesizer._sample(request.num_rows)

    # Return synthetic records as a json
    records = synthetic_df.to_json(orient="records")
    return JSONResponse({"synthetic_data": records})


@app.get("/")
def read_root():
    return {"status": "running"}

# Run with: uvicorn backend_main:app --reload --port 8000
