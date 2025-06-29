import pickle
import joblib
import torch
import onnxruntime as ort
import tensorflow as tf

def load_model(path, framework):
    if framework == "sklearn":
        return joblib.load(path)
    elif framework == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif framework == "pytorch":
        model = torch.load(path)
        model.eval()
        return model
    elif framework == "onnx":
        return ort.InferenceSession(path)
    elif framework == "tensorflow":
        return tf.keras.models.load_model(path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
