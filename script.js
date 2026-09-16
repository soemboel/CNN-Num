// Everything here runs in the browser — no backend server involved.
// The trained CNN (model.onnx, ~9KB) is loaded once and run locally via
// onnxruntime-web (WebAssembly), so this page works as pure static hosting.

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const MODEL_URL = "model.onnx";

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultNumber = document.getElementById("resultNumber");
const statusEl = document.getElementById("status");
const chartRow = document.getElementById("chartRow");
const chartPlaceholder = document.getElementById("chartPlaceholder");
const digitCardTemplate = document.getElementById("digitCardTemplate");

let drawing = false;
let lastX = 0;
let lastY = 0;
let lastPrediction = null; // kept so charts can be redrawn if the theme flips

// ---- load the model once, up front ----
predictBtn.disabled = true;
statusEl.textContent = "Loading model...";
const sessionPromise = ort.InferenceSession.create(MODEL_URL, {
  executionProviders: ["wasm"],
}).then((session) => {
  predictBtn.disabled = false;
  statusEl.textContent = "";
  return session;
}).catch((err) => {
  statusEl.textContent = "Could not load the model. Try reloading the page.";
  console.error(err);
  throw err;
});

// ---- drawing ----

function resetCanvas() {
  ctx.fillStyle = "#14110c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#fff8f0";
}
resetCanvas();

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = evt.touches ? evt.touches[0] : evt;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

function startDraw(evt) {
  evt.preventDefault();
  drawing = true;
  const { x, y } = getPos(evt);
  lastX = x;
  lastY = y;
}

function draw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x;
  lastY = y;
}

function stopDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDraw);

canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", stopDraw);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  resultNumber.textContent = "--";
  statusEl.textContent = "";
  chartRow.hidden = true;
  chartRow.innerHTML = "";
  chartPlaceholder.hidden = false;
  lastPrediction = null;
});

// ---- digit segmentation (connected-component labeling, replaces cv2.findContours) ----

function thresholdToBinary(imageData, width, height) {
  const { data } = imageData;
  const binary = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    binary[i] = data[i * 4] > 100 ? 1 : 0; // red channel; strokes are near-white, bg near-black
  }
  return binary;
}

function findComponents(binary, width, height) {
  const visited = new Uint8Array(width * height);
  const stackX = new Int32Array(width * height);
  const stackY = new Int32Array(width * height);
  const boxes = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!binary[idx] || visited[idx]) continue;

      let sp = 0;
      stackX[sp] = x;
      stackY[sp] = y;
      sp++;
      visited[idx] = 1;

      let minX = x, maxX = x, minY = y, maxY = y;

      while (sp > 0) {
        sp--;
        const cx = stackX[sp];
        const cy = stackY[sp];
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx, ny = cy + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const nidx = ny * width + nx;
            if (binary[nidx] && !visited[nidx]) {
              visited[nidx] = 1;
              stackX[sp] = nx;
              stackY[sp] = ny;
              sp++;
            }
          }
        }
      }

      const w = maxX - minX + 1;
      const h = maxY - minY + 1;
      if (w * h < 30) continue; // skip tiny noise specks
      boxes.push({ x: minX, y: minY, w, h });
    }
  }

  boxes.sort((a, b) => a.x - b.x); // left to right
  return boxes;
}

// pad each digit to a square (with a border, like real MNIST images) and
// downscale to 28x28 - mirrors what the old server-side OpenCV code did
function extractDigitTensor(binary, width, height, box) {
  const { x, y, w, h } = box;
  const side = Math.max(w, h);
  const pad = Math.floor(side / 4) + 2;
  const squareSize = side + 2 * pad;
  const xOff = pad + Math.floor((side - w) / 2);
  const yOff = pad + Math.floor((side - h) / 2);

  const squareData = new Uint8ClampedArray(squareSize * squareSize * 4);
  for (let i = 3; i < squareData.length; i += 4) squareData[i] = 255; // opaque

  for (let ry = 0; ry < h; ry++) {
    for (let rx = 0; rx < w; rx++) {
      const val = binary[(y + ry) * width + (x + rx)] ? 255 : 0;
      const dstIdx = ((yOff + ry) * squareSize + (xOff + rx)) * 4;
      squareData[dstIdx] = val;
      squareData[dstIdx + 1] = val;
      squareData[dstIdx + 2] = val;
    }
  }

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = squareSize;
  srcCanvas.height = squareSize;
  srcCanvas.getContext("2d").putImageData(new ImageData(squareData, squareSize, squareSize), 0, 0);

  const dstCanvas = document.createElement("canvas");
  dstCanvas.width = 28;
  dstCanvas.height = 28;
  const dctx = dstCanvas.getContext("2d");
  dctx.imageSmoothingEnabled = true;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(srcCanvas, 0, 0, squareSize, squareSize, 0, 0, 28, 28);

  const pixels = dctx.getImageData(0, 0, 28, 28).data;
  const tensor = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    tensor[i] = pixels[i * 4] / 255;
  }
  return tensor;
}

function softmax(row) {
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

predictBtn.addEventListener("click", async () => {
  statusEl.textContent = "Thinking...";
  resultNumber.textContent = "--";

  try {
    const session = await sessionPromise;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const binary = thresholdToBinary(imageData, canvas.width, canvas.height);
    const boxes = findComponents(binary, canvas.width, canvas.height);

    if (boxes.length === 0) {
      statusEl.textContent = "No digits detected. Try drawing bigger.";
      return;
    }

    const tensors = boxes.map((box) => extractDigitTensor(binary, canvas.width, canvas.height, box));
    const batch = new Float32Array(tensors.length * 28 * 28);
    tensors.forEach((t, i) => batch.set(t, i * 28 * 28));

    const inputTensor = new ort.Tensor("float32", batch, [tensors.length, 1, 28, 28]);
    const results = await session.run({ input: inputTensor });
    const logits = results.logits.data; // Float32Array, shape [N, 10]

    const digits = [];
    const confidences = [];
    const probabilities = [];
    for (let i = 0; i < tensors.length; i++) {
      const row = Array.from(logits.slice(i * 10, i * 10 + 10));
      const probs = softmax(row);
      let best = 0;
      for (let d = 1; d < 10; d++) if (probs[d] > probs[best]) best = d;
      digits.push(best);
      confidences.push(probs[best]);
      probabilities.push(probs);
    }

    const number = digits.join("");
    resultNumber.textContent = number;
    const avgConfidence = (confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100;
    statusEl.textContent = `Confidence: ${avgConfidence.toFixed(1)}%`;

    const data = { digits, confidences, probabilities, number };
    lastPrediction = data;
    renderBreakdown(data);
  } catch (err) {
    statusEl.textContent = "Something went wrong running the model.";
    console.error(err);
  }
});

// ---- prediction breakdown: one bar chart (digit 0-9 probabilities) per detected digit ----

function readChartColors() {
  const styles = getComputedStyle(document.documentElement);
  return {
    surface: styles.getPropertyValue("--chart-surface").trim(),
    bar: styles.getPropertyValue("--chart-bar").trim(),
    barMuted: styles.getPropertyValue("--chart-bar-muted").trim(),
    grid: styles.getPropertyValue("--chart-grid").trim(),
    textPrimary: styles.getPropertyValue("--text-primary").trim(),
    textSecondary: styles.getPropertyValue("--text-secondary").trim(),
  };
}

function drawProbabilityChart(canvasEl, probabilities, predictedDigit) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvasEl.width;   // logical size set in HTML (280x150)
  const cssHeight = canvasEl.height;
  canvasEl.width = cssWidth * dpr;
  canvasEl.height = cssHeight * dpr;
  const g = canvasEl.getContext("2d");
  g.scale(dpr, dpr);

  const colors = readChartColors();

  const padding = { top: 22, right: 8, bottom: 20, left: 8 };
  const plotW = cssWidth - padding.left - padding.right;
  const plotH = cssHeight - padding.top - padding.bottom;
  const barCount = probabilities.length; // 10
  const gap = 5;
  const barWidth = (plotW - gap * (barCount - 1)) / barCount;
  const baselineY = padding.top + plotH;

  g.clearRect(0, 0, cssWidth, cssHeight);

  // recessive baseline only - no full grid, keeps the chart quiet
  g.strokeStyle = colors.grid;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(padding.left, baselineY + 0.5);
  g.lineTo(padding.left + plotW, baselineY + 0.5);
  g.stroke();

  probabilities.forEach((p, digit) => {
    const barHeight = Math.max(2, p * plotH);
    const x = padding.left + digit * (barWidth + gap);
    const y = baselineY - barHeight;
    const isPredicted = digit === predictedDigit;
    const radius = Math.min(4, barWidth / 2);

    g.fillStyle = isPredicted ? colors.bar : colors.barMuted;
    roundRectTop(g, x, y, barWidth, barHeight, radius);
    g.fill();

    // x-axis digit label
    g.fillStyle = isPredicted ? colors.textPrimary : colors.textSecondary;
    g.font = isPredicted ? "600 11px Fredoka, sans-serif" : "11px Quicksand, sans-serif";
    g.textAlign = "center";
    g.fillText(String(digit), x + barWidth / 2, baselineY + 14);

    // direct label on the winning bar so identity isn't color-only
    if (isPredicted) {
      g.fillStyle = colors.textPrimary;
      g.font = "600 11px Fredoka, sans-serif";
      g.fillText(`${Math.round(p * 100)}%`, x + barWidth / 2, Math.max(11, y - 6));
    }
  });
}

function roundRectTop(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x, y + h);
  g.lineTo(x, y + r);
  g.arcTo(x, y, x + r, y, r);
  g.lineTo(x + w - r, y);
  g.arcTo(x + w, y, x + w, y + r, r);
  g.lineTo(x + w, y + h);
  g.closePath();
}

function renderBreakdown(data) {
  chartRow.innerHTML = "";

  data.digits.forEach((digit, i) => {
    const node = digitCardTemplate.content.cloneNode(true);
    node.querySelector(".digit-value").textContent = digit;
    node.querySelector(".digit-confidence").textContent =
      `${Math.round(data.confidences[i] * 100)}% sure`;

    const chartCanvas = node.querySelector(".digit-chart");
    chartRow.appendChild(node);
    drawProbabilityChart(chartCanvas, data.probabilities[i], digit);
  });

  chartPlaceholder.hidden = true;
  chartRow.hidden = false;
}

// redraw charts if the user's OS theme flips between light/dark mid-session
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (lastPrediction) renderBreakdown(lastPrediction);
});
