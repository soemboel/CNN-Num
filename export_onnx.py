"""
Exports the trained CNN (model/mnist_cnn.pt) to ONNX so it can run
entirely in the browser via onnxruntime-web - no backend server needed.

Run after training:
    python export_onnx.py
"""
import os

import onnx
import torch

from train_model import DigitCNN, MODEL_PATH

ONNX_PATH = "model.onnx"


def main():
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # the exporter defaults to writing weights to a separate model.onnx.data
    # file, which onnxruntime-web can't fetch in the browser - fold them
    # back into a single self-contained .onnx file.
    onnx_model = onnx.load(ONNX_PATH, load_external_data=True)
    onnx.save_model(onnx_model, ONNX_PATH, save_as_external_data=False)
    data_file = ONNX_PATH + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"Exported to {ONNX_PATH} ({os.path.getsize(ONNX_PATH)} bytes, single file)")


if __name__ == "__main__":
    main()
