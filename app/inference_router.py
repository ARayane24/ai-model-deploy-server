import torch
import numpy as np
from typing import Any, List, Optional

def run_inference(model: Any, framework: str, inputs: List, params: Optional[dict] = None) -> List:
    """
    Runs inference on a given model using the specified framework and device.

    Args:
        model (Any): Loaded ML model object.
        framework (str): The name of the framework ('sklearn', 'tensorflow', 'pytorch', 'onnx', 'pickle').
        inputs (List): Input data to run inference on.
        params (dict, optional): Additional parameters (e.g., threshold, activation, device).

    Returns:
        List: Prediction results.
    """
    params = params or {}
    device = params.get("device", "cpu")  # e.g., 'cpu', 'cuda', or 'cuda:1'

    # Ensure batch shape
    if isinstance(inputs, (list, np.ndarray)) and not isinstance(inputs[0], (list, np.ndarray)):
        inputs = [inputs]

    if framework in ["sklearn", "pickle"]:
        result = model.predict_proba(inputs) if params.get("probabilistic") else model.predict(inputs)
        if "threshold" in params:
            return (result[:, 1] > params["threshold"]).astype(int).tolist()
        return result.tolist()

    elif framework == "pytorch":
        model.eval()
        device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        model.to(device)
        with torch.no_grad():
            tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            output = model(tensor)

            if "activation" in params:
                if params["activation"] == "sigmoid":
                    output = torch.sigmoid(output)
                elif params["activation"] == "softmax":
                    output = torch.nn.functional.softmax(output, dim=1)

            return output.cpu().numpy().tolist()

    elif framework == "tensorflow":
        # TensorFlow uses all available GPUs automatically unless configured otherwise.
        # For multi-GPU control, use tf.device context in the actual model definition/training.
        output = model.predict(np.array(inputs))
        if "threshold" in params:
            return (output > params["threshold"]).astype(int).tolist()
        return output.tolist()

    elif framework == "onnx":
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        if "InferenceSession" in str(type(model)):
            session = model
        else:
            session = ort.InferenceSession(model, sess_options)

        providers = ["CPUExecutionProvider"]
        if "cuda" in device and ort.get_device() == "GPU":
            providers = [("CUDAExecutionProvider", {"device_id": int(device.split(":")[-1])})]

        session.set_providers(providers)
        input_name = session.get_inputs()[0].name
        input_array = np.array(inputs, dtype=np.float32)
        result = session.run(None, {input_name: input_array})[0]
        if "threshold" in params:
            return (result > params["threshold"]).astype(int).tolist()
        return result.tolist()

    else:
        raise ValueError(f"Unsupported framework: {framework}")
