---
title: CNN Number Recognizer
emoji: 🔢
colorFrom: yellow
colorTo: orange
sdk: static
pinned: false
---

# CNN Digit Recognizer

Draw a number on the canvas (e.g. `100`) and a CNN trained on MNIST reads
each digit and returns it as text — **entirely in your browser**, no
backend server required.

## How it works
- `train_model.py` trains a small, efficient CNN (PyTorch) on MNIST and saves
  its weights to `model/mnist_cnn.pt`. It uses BatchNorm + a 4x4 adaptive
  average pool instead of a big flatten+dense layer — **33,770 parameters**
  (vs. ~225k for the naive version), with **99.2%+ test accuracy**, trained
  with a OneCycle LR schedule and automatic mixed precision on GPU.
- `export_onnx.py` converts the trained model to `model.onnx` (~145KB), a
  self-contained format that runs via [onnxruntime-web](https://github.com/microsoft/onnxruntime)
  — no Python needed to serve predictions.
- `index.html` + `style.css` + `script.js` (project root) is the whole app:
  an organic, warm-toned board where you draw. On desktop it's a two-column
  layout — draw on the left, live confidence charts on the right; it stacks
  into one column on narrow/mobile screens. When you click Predict, it:
  1. reads the canvas pixels and finds each digit blob with a
     connected-component algorithm (a JS port of what `cv2.findContours`
     used to do server-side),
  2. pads and downscales each blob to a 28x28 tensor the same way MNIST
     images are shaped,
  3. runs `model.onnx` locally via onnxruntime-web (WebAssembly) to get a
     prediction, and
  4. renders a bar chart per digit showing the CNN's full 0-9 probability
     distribution (the winning digit is highlighted and labeled).

  Because all of this runs client-side, these three files (plus
  `model.onnx`) are the *entire* deployable app — drop them on any static
  host and it works.

### Optional: Flask API (`app.py`)
This repo also keeps a Flask + PyTorch server (`app.py`, `Dockerfile`) that
does the same thing server-side via a `/predict` endpoint, from back when
this was backend-hosted. It's no longer needed for the default setup above,
but it's kept in case you'd rather run inference server-side (e.g. swapping
in a bigger model later that's too heavy for a browser). See
[Optional: run it as a server](#optional-run-it-as-a-server) below.

## Run it
Just open `index.html` through a local web server (opening the file
directly via `file://` will block the WASM/model loading in most browsers):
```bash
python -m http.server 8000
```
Then open http://localhost:8000, draw a number, click **Predict**.

## Hosting it
Since the whole app is static files, deploy it to any static host —
literally drag-and-drop `index.html`, `style.css`, `script.js`, and
`model.onnx` into any of these, no build step, no card, no server:

- **Cloudflare Pages** — connect this GitHub repo, leave the build command
  empty and build output directory as `/` (default).
- **Hugging Face Spaces** — the `Static` SDK (free for everyone, no PRO
  plan needed) works great here too.
- **GitHub Pages**, **Netlify**, or literally any static file host.

(Pages/Spaces will also upload `app.py`, `Dockerfile`, `requirements.txt`,
etc. as inert static files since they're in the repo — harmless, just
unused, since nothing executes them.)

## Retraining / changing the model
```bash
pip install -r requirements.txt
python train_model.py     # trains the CNN, saves model/mnist_cnn.pt
python export_onnx.py     # re-exports model.onnx for the browser to use
```
Redeploy the static site afterward so it picks up the new `model.onnx`.

### GPU or CPU training
`train_model.py` auto-detects a CUDA GPU and uses it automatically, falling
back to CPU with no changes needed:
```bash
python train_model.py                  # auto: GPU if available, else CPU (~2-5 min on CPU, <1 min on GPU)
python train_model.py --device cpu     # force CPU
python train_model.py --device cuda    # force GPU (errors if none found)
python train_model.py --epochs 15      # train longer
```

**If you have an NVIDIA GPU but it's not being detected:** plain
`pip install torch` (what `requirements.txt` installs) gives you the
**CPU-only** build — that's deliberate, since it keeps the install light.
To let PyTorch use your GPU for local training, install the CUDA build on
top of it:
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

## Optional: run it as a server
If you'd rather do inference server-side instead of in the browser:
```bash
python app.py              # starts the API + website at http://localhost:5000
```
`app.py` serves the same frontend plus a `/predict` endpoint (image in,
prediction out, via OpenCV + PyTorch). To host it, you need a real Python
runtime — note that free tiers on Render/Railway/Fly.io now require a card
for verification, and Hugging Face Spaces now requires a paid PRO plan to
create a Docker Space (Static Spaces remain free, which is what the
client-side version above uses instead). A `Dockerfile` is included if you
do have somewhere to run it. Set `API_URL` at the top of `script.js` to
point back at `/predict` if you go this route and want the frontend to call
it instead of running the model locally.

## Notes
- Draw digits reasonably large and spaced apart so they don't touch —
  touching digits get segmented as one blob.
- Retraining with more epochs or data augmentation in `train_model.py` will
  improve accuracy further — just re-run `export_onnx.py` afterward.
