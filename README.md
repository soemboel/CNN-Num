# CNN Digit Recognizer

Draw a number on the canvas (e.g. `100`) and a CNN trained on MNIST reads
each digit and returns it as text.

## How it works
- `train_model.py` trains a small, efficient CNN (PyTorch) on MNIST and saves
  its weights to `model/mnist_cnn.pt`. It uses BatchNorm + a 4x4 adaptive
  average pool instead of a big flatten+dense layer — **33,770 parameters**
  (vs. ~225k for the naive version), with **99.2%+ test accuracy**, trained
  with a OneCycle LR schedule and automatic mixed precision on GPU.
- `app.py` is a Flask API: it serves the website (`static/`) and a `/predict`
  endpoint that takes the drawn image, splits it into individual digit blobs
  with OpenCV, runs each through the CNN, and returns the combined number.
- `static/index.html` + `style.css` + `script.js` is the drawing UI: an organic,
  warm-toned board. On desktop it's a two-column layout — draw on the left,
  live confidence charts on the right; it stacks into one column on narrow/
  mobile screens. It draws white strokes on a black canvas (same format as
  MNIST), sends a PNG snapshot to `/predict`, and renders a bar chart per
  detected digit showing the CNN's full 0-9 probability distribution (the
  winning digit is highlighted and labeled).

## Run locally
```bash
pip install -r requirements.txt
python train_model.py     # trains the CNN, saves model/mnist_cnn.pt
python app.py              # starts the API + website at http://localhost:5000
```
Open http://localhost:5000, draw a number, click **Predict**.

### GPU or CPU training
`train_model.py` auto-detects a CUDA GPU and uses it automatically, falling
back to CPU with no changes needed:
```bash
python train_model.py                  # auto: GPU if available, else CPU (~2-5 min on CPU, <1 min on GPU)
python train_model.py --device cpu     # force CPU
python train_model.py --device cuda    # force GPU (errors if none found)
python train_model.py --epochs 15      # train longer
```
`app.py` picks the same device automatically for inference.

**If you have an NVIDIA GPU but it's not being detected:** plain
`pip install torch` (what `requirements.txt` installs) gives you the
**CPU-only** build — that's deliberate, since it keeps the app light for
hosting where there's no GPU anyway. To let PyTorch use your GPU for local
training, install the CUDA build on top of it:
```bash
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```
(swap `cu126` for whatever CUDA branch `pip index versions torch --index-url
https://download.pytorch.org/whl/cu126` etc. shows as available for your
Python version — check with `nvidia-smi` first to confirm your driver/GPU are
visible at all). Verify it worked with:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Hosting it
`app.py` already serves the frontend, so deploying it anywhere that runs
Python (Render, Railway, Fly.io, an EC2/VPS, etc.) hosts both the API and
the site together:
1. Make sure `model/mnist_cnn.pt` is trained and committed/uploaded (or
   train it as part of your build step).
2. Start command: `gunicorn app:app` (already in requirements.txt) or
   `python app.py` for a quick test.
3. The frontend calls `/predict` on the same origin by default. If you ever
   split the frontend onto a different host (e.g. GitHub Pages), set
   `API_URL` at the top of `static/script.js` to your API's full URL —
   CORS is already enabled in `app.py` via `flask-cors`.

## Notes
- Draw digits reasonably large and spaced apart so they don't touch —
  touching digits get segmented as one blob.
- Retraining with more epochs or data augmentation in `train_model.py` will
  improve accuracy further.
