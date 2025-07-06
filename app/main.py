from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.model_loader import load_model
from app.inference_router import run_inference
import shutil, os, json, uuid, requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# Constants
MODELS_DIR = "storage"
METADATA_DIR = "metadata"
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.json")

# In-memory model store
model_store = {}

# Load metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE) as f:
        model_metadata = json.load(f)
else:
    model_metadata = {}

# Load models into memory
for fname, meta in model_metadata.items():
    path = os.path.join(MODELS_DIR, fname)
    model_store[fname] = (load_model(path, meta["framework"]), meta["framework"])


class InferenceRequest(BaseModel):
    model_name: str
    inputs: list
    params: dict = {}  # Includes options like threshold, activation, device, etc.


def get_global_ip():
    try:
        return requests.get('https://api.ipify.org').text
    except Exception:
        return "Unavailable"


@app.get("/")
def root():
    return {
        "message": "AI Model Deploy Server is running.",
        "global_ip": get_global_ip(),
        "port": 8000,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Server status and info"},
            {"path": "/upload_model", "method": "POST", "description": "Upload a new model"},
            {"path": "/predict", "method": "POST", "description": "Run inference on a model"},
            {"path": "/delete_model/{model_name}", "method": "DELETE", "description": "Delete a model"},
        ]
    }


@app.post("/upload_model")
async def upload_model(model_file: UploadFile = File(...), metadata_json: str = Form(...)):
    ext = os.path.splitext(model_file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"

    # Save model file
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, unique_filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    # Save metadata
    os.makedirs(METADATA_DIR, exist_ok=True)
    metadata = json.loads(metadata_json)
    model_metadata[unique_filename] = metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(model_metadata, f, indent=2)

    # Load model to memory
    model_store[unique_filename] = (load_model(model_path, metadata["framework"]), metadata["framework"])

    return {"status": "uploaded", "model": unique_filename}


@app.post("/predict")
def predict(req: InferenceRequest):
    if req.model_name not in model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model, framework = model_store[req.model_name]
    try:
        result = run_inference(model, framework, req.inputs, req.params)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_model/{model_name}")
def delete_model(model_name: str):
    if model_name not in model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from memory
    del model_store[model_name]

    # Remove from metadata
    if model_name in model_metadata:
        del model_metadata[model_name]
        with open(METADATA_FILE, "w") as f:
            json.dump(model_metadata, f, indent=2)

    # Delete file
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    return {"status": "deleted", "model": model_name}
