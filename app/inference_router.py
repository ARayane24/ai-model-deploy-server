import torch
import numpy as np
from typing import Any, List, Optional

def run_inference(model: Any, framework: str, inputs: List, params: Optional[dict] = None) -> List:
    """
    Runs inference on a given model using the specified framework.

    Args:
        model (Any): Loaded ML model object.
        framework (str): The name of the framework ('sklearn', 'tensorflow', 'pytorch', 'onnx', 'pickle').
        inputs (List): Input data to run inference on.
        params (dict, optional): Additional parameters (e.g., threshold, activation).

    Returns:
        List: Prediction results.
    """
    params = params or {}

    if framework in ["sklearn", "pickle"]:
        # Scikit-learn or generic pickle model
        result = model.predict_proba(inputs) if params.get("probabilistic") else model.predict(inputs)
        if "threshold" in params:
            # Apply binary threshold on positive class
            return (result[:, 1] > params["threshold"]).astype(int).tolist()
        return result.tolist()

    elif framework == "pytorch":
        # PyTorch model
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(inputs, dtype=torch.float32)
            output = model(tensor)

            if "activation" in params:
                if params["activation"] == "sigmoid":
                    output = torch.sigmoid(output)
                elif params["activation"] == "softmax":
                    output = torch.nn.functional.softmax(output, dim=1)

            return output.numpy().tolist()

    elif framework == "tensorflow":
        # TensorFlow/Keras model
        output = model.predict(np.array(inputs))
        if "threshold" in params:
            return (output > params["threshold"]).astype(int).tolist()
        return output.tolist()

    elif framework == "onnx":
        # ONNX Runtime model
        input_name = model.get_inputs()[0].name
        input_array = np.array(inputs, dtype=np.float32)
        result = model.run(None, {input_name: input_array})[0]
        if "threshold" in params:
            return (result > params["threshold"]).astype(int).tolist()
        return result.tolist()

    else:
        raise ValueError(f"Unsupported framework: {framework}")
