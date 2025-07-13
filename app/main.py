from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from pydantic import BaseModel
from app.model_loader import load_model
from app.inference_router import run_inference
from app.db.session import SessionLocal
import shutil, os, json, uuid, requests
from sqlalchemy.orm import Session
from sqlalchemy import UUID, func
from typing import List, Optional
from app.db.models import AIModel, AIModelResult
from app.schemas import AIModelOut, AIModelResultOut
from datetime import datetime
from app.db.models import Base
from app.db.session import engine
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Done.")


app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    if os.path.exists(path):
        try:
            model_store[fname] = (load_model(path, meta["framework"]), meta["framework"])
        except Exception as e:
            print(f"[ERROR] Failed to load model '{fname}': {e}")
    else:
        print(f"[WARNING] Model file not found: {path}, skipping.")



class InferenceRequest(BaseModel):
    model_name: str
    inputs: list
    params: dict = {}  # Includes options like threshold, activation, device, etc.
    save_result: Optional[bool] = False 


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
            {"path": "/delete_model/{model_name}", "method": "DELETE", "description": "Delete a model and its results"},
            {"path": "/models/", "method": "GET", "description": "List or search uploaded models"},
            {"path": "/results", "method": "GET", "description": "Get all saved inference results"}
        ]
    }



@app.post("/upload_model", response_model=AIModelOut)
async def upload_model(
    model_file: UploadFile = File(...),
    metadata_json: str = Form(...),
    db: Session = Depends(get_db)
):
    print(f"[UPLOAD] Upload triggered for model: {model_file.filename}")
    
    ext = os.path.splitext(model_file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"

    # Save model file
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, unique_filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    file_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)

    # Parse metadata
    try:
        metadata = json.loads(metadata_json)
        name = metadata["name"]
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])
        framework = metadata["framework"]
    except (KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

    # === Extract model info ===
    num_inputs = num_outputs = num_parameters = 0
    if framework == "onnx":
        import onnx
        model = onnx.load(model_path)
        num_inputs = len(model.graph.input)
        num_outputs = len(model.graph.output)
        num_parameters = sum([tensor.dims[0] if tensor.dims else 0 for tensor in model.graph.initializer])
    elif framework == "pytorch":
        import torch
        torch_model = torch.load(model_path, map_location="cpu")
        if hasattr(torch_model, 'parameters'):
            num_parameters = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

    def get_supported_params(framework: str) -> dict:
        return {
            "sklearn": {"probabilistic": True, "threshold": 0.5},
            "pickle": {"probabilistic": True, "threshold": 0.5},
            "pytorch": {"device": "cpu", "activation": "sigmoid"},
            "tensorflow": {"threshold": 0.5},
            "onnx": {"device": "cpu", "threshold": 0.5},
        }.get(framework, {})

    supported_params = get_supported_params(framework)

    # Save to metadata JSON file
    os.makedirs(METADATA_DIR, exist_ok=True)
    model_metadata[unique_filename] = metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(model_metadata, f, indent=2)

    # Load model to memory
    model_store[unique_filename] = (load_model(model_path, framework), framework)

    # Save to database
    new_model = AIModel(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        tags=tags,
        file_name=unique_filename,
        file_size=file_size_mb,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_parameters=num_parameters,
        params=supported_params,
        created_at=datetime.now()
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    print(f"[UPLOAD] Saved model as: {unique_filename} ({file_size_mb} MB)")

    return new_model


@app.get("/models/", response_model=List[AIModelOut])
def get_models(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tag: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(AIModel)

    if name:
        query = query.filter(func.lower(AIModel.name).like(f"%{name.lower()}%"))
    if description:
        query = query.filter(func.lower(AIModel.description).like(f"%{description.lower()}%"))
    if tag:
        query = query.filter(func.lower(tag) == func.any(func.lower(AIModel.tags)))

    return query.all()

@app.get("/results", response_model=List[AIModelResultOut])
def list_all_results(db: Session = Depends(get_db)):
    try:
        results = db.query(AIModelResult).all()
        return results
    except Exception as e:
        # Optionally log the error here with logging module
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")
    
    
@app.post("/predict")
def predict(req: InferenceRequest, db: Session = Depends(get_db)):
    if req.model_name not in model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model, framework = model_store[req.model_name]

    try:
        start_time = time.time()
        result = run_inference(model, framework, req.inputs, req.params)
        duration = round(time.time() - start_time, 4)

        # Prepare base response
        response = {
            "status": "success",
            "model_name": req.model_name,
            "framework": framework,
            "params_used": req.params,
            "inference_time_sec": duration,
            "result": result
        }

        # Optional: Save to DB
        if req.save_result:
            db_model = db.query(AIModel).filter(AIModel.file_name == req.model_name).first()
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found in database")

            result_entry = AIModelResult(
                model_id=db_model.id,
                output_vector=result
            )
            db.add(result_entry)
            db.commit()
            db.refresh(result_entry)

            response["saved_result_id"] = str(result_entry.id)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.delete("/delete_model/{model_name}")
def delete_model(model_name: str, db: Session = Depends(get_db)):
    if model_name not in model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from in-memory store
    del model_store[model_name]

    # Remove from metadata JSON
    if model_name in model_metadata:
        del model_metadata[model_name]
        with open(METADATA_FILE, "w") as f:
            json.dump(model_metadata, f, indent=2)

    # Delete file
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    # Delete from DB (by file_name, which is unique)
    db_model = db.query(AIModel).filter(AIModel.file_name == model_name).first()
    if db_model:
        db.delete(db_model)  # This cascades to AIModelResult
        db.commit()

    return {"status": "deleted", "model": model_name}








