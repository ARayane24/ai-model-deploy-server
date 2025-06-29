# 🧠 AI Model API Backend

A **FastAPI**-powered backend that allows users to:

* 🔼 Upload machine learning models (e.g., `scikit-learn` `.pkl`, `TensorFlow`, `PyTorch`, `ONNX`)
* 🤖 Run inference on uploaded models
* ❌ Delete stored models and associated metadata
* 🧾 Manage and persist model metadata

---

## 🚀 Features

* Supports multiple ML frameworks: `scikit-learn`, `TensorFlow`, `PyTorch`, `ONNX`, and generic pickled models
* Perform predictions via JSON API
* Dynamically load and unload models at runtime
* Metadata handling for framework, input/output types
* Clean separation of models and metadata in the file system

---

## 🧰 Requirements

* Python 3.11
* Dependencies from `requirements.txt`

Or, for containerized usage:

* Docker

---

## 🔧 Setup & Run

### 💻 Local Setup

```bash
# Clone the repo and navigate to the project
git clone <your-repo-url>
cd ai_model_api

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

### 🐳 Docker Setup

```bash
# Build Docker image
docker build -t ai-model-api .

# Run the container
docker run -p 8000:8000 ai-model-api
```

---

## 🔍 API Endpoints

### `POST /upload_model`

Upload a model file and its metadata.

**Form fields**:

* `model_file`: binary file (`.pkl`, `.onnx`, etc.)
* `metadata_json`: JSON string

```json
{
  "framework": "sklearn",
  "input_type": "list",
  "output_type": "class"
}
```

---

### `POST /predict`

Run inference using a previously uploaded model.

**Body** (JSON):

```json
{
  "model_name": "user_model.pkl",
  "inputs": [[5.1, 3.5, 1.4, 0.2]],
  "params": {}
}
```

---

### `DELETE /delete_model/{model_name}`

Deletes the specified model and its metadata.

---

## 🧪 Example Usage

Run [`user_example.py`](./user_example.py) to:

1. Load and train a model (e.g., on Iris or Digits dataset)
2. Upload the model to the backend
3. Run prediction requests
4. Visualize inputs and model predictions

---

## 🗃️ Model Storage

* Models are saved in: `storage/`
* Metadata is saved in: `metadata/metadata.json`

---

## ✅ Supported Frameworks

| Framework            | File Format        | Required Libraries       | Notes                                                              |
| -------------------- | ------------------ | ------------------------ | ------------------------------------------------------------------ |
| **Scikit-learn**     | `.pkl` (pickle)    | `scikit-learn`, `joblib` | Supports `predict`, `predict_proba`. Optional threshold support.   |
| **PyTorch**          | `.pt` / `.pth`     | `torch`                  | Applies `sigmoid` or `softmax` if specified in params.             |
| **TensorFlow/Keras** | `.h5` / SavedModel | `tensorflow`             | Uses `.predict`. Thresholding supported for binary classification. |
| **ONNX**             | `.onnx`            | `onnxruntime`            | Accepts NumPy input, returns raw or thresholded predictions.       |
| **Pickled Models**   | `.pkl`             | Varies                   | Generic support using `framework: "pickle"` in metadata.           |

---

## 📄 License

MIT License © 2025

---
