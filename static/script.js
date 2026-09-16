// If you host the frontend separately from the Flask API, change this
// to the full URL of your API, e.g. "https://your-api.onrender.com"
const API_URL = "http://localhost:5000";

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

predictBtn.addEventListener("click", async () => {
  statusEl.textContent = "Thinking...";
  resultNumber.textContent = "--";

  const dataURL = canvas.toDataURL("image/png");

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
    });

    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = data.error || "Something went wrong.";
      return;
    }

    if (!data.number) {
      statusEl.textContent = data.message || "No digits detected. Try drawing bigger.";
      return;
    }

    resultNumber.textContent = data.number;
    const avgConfidence =
      (data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length) * 100;
    statusEl.textContent = `Confidence: ${avgConfidence.toFixed(1)}%`;

    lastPrediction = data;
    renderBreakdown(data);
  } catch (err) {
    statusEl.textContent = "Could not reach the API. Is app.py running?";
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
