---
title: CNN Number Recognizer
emoji: 🔢
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# CNN Digit Recognizer

Draw a number on the canvas (e.g. `100`) and a CNN trained on MNIST reads
each digit and returns it as text.

> The YAML block above is metadata Hugging Face Spaces reads to build this
> as a Docker Space — harmless to keep even if you only use GitHub/Render/etc.

## How it works
- `train_model.py` trains a small, efficient CNN (PyTorch) on MNIST and saves
  its weights to `model/mnist_cnn.pt`. It uses BatchNorm + a 4x4 adaptive
  average pool instead of a big flatten+dense layer — **33,770 parameters**
  (vs. ~225k for the naive version), with **99.2%+ test accuracy**, trained
  with a OneCycle LR schedule and automatic mixed precision on GPU.
- `app.py` is a Flask API: it serves the frontend (`index.html`, `style.css`,
  `script.js`, each via its own explicit route — nothing else in the repo is
  reachable over HTTP) and a `/predict` endpoint that takes the drawn image,
  splits it into individual digit blobs with OpenCV, runs each through the
  CNN, and returns the combined number.
- `index.html` + `style.css` + `script.js` (project root) is the drawing UI:
  an organic, warm-toned board. On desktop it's a two-column layout — draw on
  the left, live confidence charts on the right; it stacks into one column on
  narrow/mobile screens. It draws white strokes on a black canvas (same
  format as MNIST), sends a PNG snapshot to `/predict`, and renders a bar
  chart per detected digit showing the CNN's full 0-9 probability
  distribution (the winning digit is highlighted and labeled). These three
  files live at the repo root (not in a subfolder) so they can be deployed
  as-is to a static host like Cloudflare Pages.

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

**Cloudflare (Pages/Workers) cannot run `app.py`** — there's no Python/PyTorch
runtime there, and the model + its dependencies are far past what a Worker
allows. Cloudflare is a great fit for the *frontend only*; the Flask + CNN
backend needs a real Python host. Two ways to deploy:

### Option A — one host runs everything (simplest)
`app.py` already serves the frontend itself, so deploying the whole repo
to any Python host serves both the site and the API from one URL. Start
command: `gunicorn app:app` (already in requirements.txt), or `python app.py`
for a quick test. Nothing else to configure — `script.js`'s `API_URL` can
stay empty/relative since frontend and API share an origin.

Where to run it: **Render, Railway, and Fly.io now require a credit card on
file** even for their free tiers (for verification — you're not charged
unless you upgrade, but not everyone wants to hand over a card for a hobby
project). If you'd rather avoid that:

- **Hugging Face Spaces** — free, **no credit card**, and literally built
  for hosting ML demos like this one. This repo already includes a
  `Dockerfile` + `.dockerignore` for it, and the YAML block at the top of
  this README is the Space config it reads (`sdk: docker`, `app_port: 7860`).
  To deploy:
  1. Create a free account at huggingface.co, then **New Space** → pick
     **Docker** as the SDK (not Gradio/Streamlit).
  2. Push this repo to the Space's git remote (shown on the Space's page,
     looks like `https://huggingface.co/spaces/<you>/<space-name>`):
     ```bash
     git remote add space https://huggingface.co/spaces/<you>/<space-name>
     git push space main
     ```
  3. The Space builds the Dockerfile automatically. Once it's live, your API
     is reachable at `https://<you>-<space-name>.hf.space`.
  4. Free-tier Spaces sleep after a period of inactivity and take a bit to
     wake back up on the next request (same tradeoff as free Render/Railway).

### Option B — frontend on Cloudflare Pages, backend elsewhere
1. **Backend:** deploy `app.py` using Option A (Hugging Face Spaces, or
   Render/Railway/Fly.io if you don't mind the card requirement). Note the
   URL it gives you, e.g. `https://you-space-name.hf.space`.
2. **Frontend:** create a Cloudflare Pages project from this GitHub repo.
   Since `index.html` now lives at the repo root, you can leave Pages'
   "build output directory" as `/` (the default) — no build command needed,
   it's static files. (Pages will also upload the `.py` files,
   `requirements.txt`, and `Dockerfile` as inert static assets since they're
   in the repo; they aren't executed and aren't secret, just unused —
   harmless either way.)
3. Set `API_URL` at the top of `script.js` to your backend's full URL from
   step 1, then redeploy the Pages site. CORS is already enabled in `app.py`
   via `flask-cors`, so the cross-origin calls from your `.pages.dev` domain
   to your backend's domain work out of the box.

## Notes
- Draw digits reasonably large and spaced apart so they don't touch —
  touching digits get segmented as one blob.
- Retraining with more epochs or data augmentation in `train_model.py` will
  improve accuracy further.
