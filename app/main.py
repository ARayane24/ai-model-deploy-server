from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.model_loader import load_model
from app.inference_router import run_inference
import shutil, os, json
from fastapi import HTTPException
import uuid
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()
model_store = {}
MODELS_DIR = "storage"                  # Where models are stored
METADATA_DIR = "metadata"              # Folder to hold metadata
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.json")


            
# LOAD ALL MODELS FROM DISK INTO MEMORY
with open(METADATA_FILE) as f:
    model_metadata = json.load(f)

for fname, meta in model_metadata.items():
    path = os.path.join(MODELS_DIR, fname)
    model_store[fname] = (
        load_model(path, meta["framework"]),
        meta["framework"]
    )


class InferenceRequest(BaseModel):
    model_name: str
    inputs: list
    params: dict = {}




@app.post("/upload_model")
async def upload_model(model_file: UploadFile = File(...), metadata_json: str = Form(...)):
    # Generate a unique filename using uuid
    ext = os.path.splitext(model_file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"

    # Save model to storage/
    model_path = os.path.join(MODELS_DIR, unique_filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    # Save metadata to memory and metadata file
    metadata = json.loads(metadata_json)
    model_metadata[unique_filename] = metadata

    with open(METADATA_FILE, "w") as f:
        json.dump(model_metadata, f)

    # Load model
    model_store[unique_filename] = (
        load_model(model_path, metadata["framework"]),
        metadata["framework"]
    )
    
    return {"status": "uploaded", "model": unique_filename}





@app.post("/predict")
def predict(req: InferenceRequest):
    model, framework = model_store[req.model_name]
    result = run_inference(model, framework, req.inputs, req.params)
    return {"result": result}





@app.delete("/delete_model/{model_name}")
def delete_model(model_name: str):
    if model_name not in model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from in-memory store and metadata
    del model_store[model_name]
    if model_name in model_metadata:
        del model_metadata[model_name]
        with open(METADATA_FILE, "w") as f:
            json.dump(model_metadata, f)

    # Remove model file from storage/
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    return {"status": "deleted", "model": model_name}

