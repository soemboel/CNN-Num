"""
Flask API that serves the digit-recognizer web page and a /predict
endpoint. The frontend sends a drawn image (white digits on black,
like MNIST) as a base64 PNG; this splits it into individual digits,
runs each through the trained CNN, and returns the combined number.

Run:
    python app.py
Then open http://localhost:5000 in a browser.
"""
import base64
import io
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from train_model import DigitCNN

MODEL_PATH = "model/mnist_cnn.pt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
CORS(app)  # allow the frontend to call this API even if hosted elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
if os.path.exists(MODEL_PATH):
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print(f"WARNING: {MODEL_PATH} not found. Run `python train_model.py` first.")


def decode_image(data_url: str) -> np.ndarray:
    """Turns a 'data:image/png;base64,...' string into a grayscale numpy array."""
    _header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    return np.array(img)


def extract_digits(gray: np.ndarray):
    """
    Finds each separate digit blob in the image (assumes white strokes
    on a black background, same convention as MNIST) and returns a list
    of 28x28 normalized arrays ready for the model, ordered left to right.
    """
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 30:  # skip tiny noise specks
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])  # left to right

    digit_images = []
    for (x, y, w, h) in boxes:
        crop = thresh[y:y + h, x:x + w]

        # pad to a square, keeping the digit centered, then leave a border
        # like the real MNIST images have (digit isn't touching the edges)
        side = max(w, h)
        pad = side // 4 + 2
        square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
        y_off = pad + (side - h) // 2
        x_off = pad + (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop

        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        digit_images.append(normalized.reshape(1, 28, 28))  # (C, H, W) for torch

    return digit_images


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    body = request.get_json(silent=True)
    if not body or "image" not in body:
        return jsonify({"error": "Missing 'image' field."}), 400

    try:
        gray = decode_image(body["image"])
    except Exception:
        return jsonify({"error": "Could not decode image."}), 400

    digit_images = extract_digits(gray)
    if not digit_images:
        return jsonify({"digits": [], "number": "", "message": "No digits found."})

    batch = torch.from_numpy(np.stack(digit_images)).to(device)  # (N, 1, 28, 28)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    digits = preds.cpu().tolist()
    confidences = confidences.cpu().tolist()
    probabilities = probs.cpu().tolist()  # per digit: [p(0), p(1), ..., p(9)]
    number = "".join(str(d) for d in digits)

    return jsonify({
        "digits": digits,
        "confidences": confidences,
        "probabilities": probabilities,
        "number": number,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
