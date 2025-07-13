# train_and_upload.py

import requests
import joblib
import io
import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# === 🧠 Load & Prepare Dataset ===
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 🤖 Train Model ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === 📦 Dump model into memory (no file write) ===
model_bytes = io.BytesIO()
joblib.dump(clf, model_bytes)
model_bytes.seek(0)

# === 📄 Define Metadata ===
metadata = {
    "name": "Digits Classifier",
    "description": "Logistic Regression model trained on sklearn digits dataset.",
    "tags": ["sklearn", "digits", "classifier"],
    "framework": "sklearn"
}

# === ⬆️ Upload Model and Metadata ===
response = requests.post(
    "http://127.0.0.1:8000/upload_model",
    files={"model_file": ("user_model.pkl", model_bytes, "application/octet-stream")},
    data={"metadata_json": json.dumps(metadata)}
)

# Handle response
upload_json = response.json()
print("✅ Upload Response:", upload_json)

# Save model file name for later use
with open("test_user/model_name.txt", "w") as f:
    f.write(upload_json["file_name"])


np.savez("test_data.npz", X_test=X_test, y_test=y_test, images=digits.images)
