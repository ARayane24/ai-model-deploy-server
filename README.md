# 🧠 AI Model API Backend

A **FastAPI**-powered backend that allows users to:

* 🔼 Upload machine learning models (`.pkl`, `.onnx`, `.pt`, etc.)
* 🤖 Run inference with optional result persistence to a PostgreSQL database
* ❌ Delete models and all related metadata and results
* 🧾 Manage and query saved inference results
* 📦 Manage models via file system and database

---

## 🚀 Features

* 🔧 Multi-framework support: `scikit-learn`, `TensorFlow`, `PyTorch`, `ONNX`, and generic `pickle`
* 📤 Upload models and save their metadata
* 🔎 Query and filter saved models by name, description, or tag
* 🧠 Perform inference on uploaded models
* 💾 Optionally save inference results to the database
* 📥 Retrieve all saved results via an API
* 🧼 Clean deletion of models and associated results from disk and database

---

## 🧰 Requirements

* Python 3.11
* PostgreSQL (for persistent result storage)
* Dependencies in `requirements.txt`

Optional:

* Docker

---

## 🔧 Setup & Run

### 💻 Local Setup

```bash
# Clone and enter the repo
git clone https://github.com/ARayane24/ai-model-deploy-server
cd ai_model_api

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

---

### 🐳 Docker Setup

```bash
# Build Docker image
docker build -t ai-model-api .

# Run the container
docker run -p 8000:8000 ai-model-api
```

---

## 🔍 API Endpoints

### `GET /`

Returns a welcome message, IP address, and list of all available endpoints.

---

### `POST /upload_model`

Upload a machine learning model and its metadata.

**Form fields**:

* `model_file`: Model file (`.onnx`, `.pkl`, `.pt`, etc.)
* `metadata_json`: JSON string, example:

```json
{
  "name": "MOE System Final",
  "description": "Multi-Output Encoder system in ONNX format.",
  "tags": ["onnx", "segmentation"],
  "framework": "onnx"
}
```

---

### `POST /predict`

Run inference on a specific model.

**Request body**:

```json
{
  "model_name": "cbe8bf6e9a814c4888056c3342ad6c9a.onnx",
  "inputs": [[...input vector...]],
  "params": {
    "device": "cpu",
    "threshold": 0.5
  },
  "save_result": true
}
```

**If `save_result` is true**, the result will be saved in the database and an ID will be returned.

---

### `GET /models/`

Retrieve a list of saved models with optional filters:

* `?name=...`
* `?description=...`
* `?tag=...`

---

### `GET /results`

Retrieve all saved inference results. Each result includes:

* ID
* Model ID
* Output vector (as JSON)
* Timestamp

---

### `DELETE /delete_model/{model_name}`

Deletes the specified model:

* From memory
* From file storage
* From metadata.json
* From PostgreSQL (including all related results)

---

## 🧪 Example Usage

Run [`user_example.py`](./user_example.py) to:

1. Load and train a model
2. Upload the model
3. Run predictions
4. Optionally save and view inference results

---

## 🗃️ Model Storage

* Model files: `storage/`
* Metadata JSON: `metadata/metadata.json`
* Results and metadata: stored in PostgreSQL via SQLAlchemy

---

## ✅ Supported Frameworks

| Framework            | File Format       | Required Libraries       | Notes                                                      |
| -------------------- | ----------------- | ------------------------ | ---------------------------------------------------------- |
| **Scikit-learn**     | `.pkl`            | `scikit-learn`, `joblib` | Basic classification or regression                         |
| **PyTorch**          | `.pt`, `.pth`     | `torch`                  | Uses `.eval()` and device params                           |
| **TensorFlow/Keras** | `.h5`, SavedModel | `tensorflow`             | Uses `.predict` for inference                              |
| **ONNX**             | `.onnx`           | `onnxruntime`            | 6-band satellite imagery supported, thresholding available |
| **Pickled Models**   | `.pkl`            | Custom                   | Specify `"framework": "pickle"` in metadata                |

---

## 📦 Database Tables

### `ai_models`

Stores metadata for each uploaded model.

### `ai_model_results`

Stores JSON-encoded inference results with timestamps.

---

## 📄 License

MIT License © 2025

---
