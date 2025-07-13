# test_api_predict.py

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# === 📂 Load model name ===
with open("./test_user/model_name.txt", "r") as f:
    model_name = f.read().strip()

# === 🧠 Load digits data again and split ===
digits = load_digits()
X, y, images = digits.data, digits.target, digits.images

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42
)

# === 🧪 Pick a Sample ===
idx = np.random.randint(len(X_test))
sample_input = [X_test[idx].tolist()]
real_label = y_test[idx]
sample_image = images_test[idx]

# === 🔮 Send Prediction Request ===
req_payload = {
            "model_name": model_name,
            "inputs": sample_input,
            "params": {
                "device": "cpu"
            },
            "save_result": True 
        }

response = requests.post("http://127.0.0.1:8000/predict", json=req_payload)

# === 📊 Display Result ===
print("\n🎯 Real Label:", real_label)
print("📦 Response Code:", response.status_code)

if response.status_code == 200:
    pred = response.json()["result"][0]
    print("🔮 Prediction:", pred)

    # === 🎨 Show Image ===
    plt.figure(figsize=(2, 2))
    plt.imshow(sample_image, cmap="gray")
    plt.title(f"Pred: {pred}\nReal: {real_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("test_user/prediction_visualization.png")
    print("🖼️ Visualization saved to 'prediction_visualization.png'")
else:
    print("❌ Failed:", response.text)
