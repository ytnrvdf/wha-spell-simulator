// Browser port of the Python glyph GUI: draw one symbol, see the 56x56 render
// the net receives and how the learned siamese recognizer ranks it, side by
// side with the built-in template matcher. A developer tool for evaluating the
// siamese engine and its prototypes against the existing recognizer.

import { CONFIG } from "../src/config.js";
import { loadDictionary } from "../src/dictionary/dictionaryLoader.js";
import { DrawingCapture } from "../src/input/drawingCapture.js";
import { createStrokeStore } from "../src/input/strokeStore.js";
import {
  allPoints,
  angleDegFromCenter,
  angularDifference,
  boundsForStrokes,
  centerOfBounds,
  clamp,
  directedStrokeAngle,
  dominantAxisOrientationDeg,
  endpointClosedness,
  strokeLength
} from "../src/utils/geometry.js";
import { cleanStrokes } from "../src/parser/strokeCleaner.js";
import { recognizeCandidates as recognizeWithTemplates } from "../src/parser/symbolRecognizer.js";
import { recognizeCandidatesSiamese, warmupSiamese } from "../src/parser/siameseRecognizer.js";
import { rasterizeStrokes } from "../src/parser/glyphRasterizer.js";
import { drawStrokes } from "../src/renderer/glyphOverlayRenderer.js";
import { drawPaper } from "../src/renderer/paperRenderer.js";

const elements = {
  canvas: document.querySelector("#drawCanvas"),
  render: document.querySelector("#renderCanvas"),
  undoButton: document.querySelector("#undoButton"),
  clearButton: document.querySelector("#clearButton"),
  siameseStatus: document.querySelector("#siameseStatus"),
  siameseMatches: document.querySelector("#siameseMatches"),
  templateStatus: document.querySelector("#templateStatus"),
  templateMatches: document.querySelector("#templateMatches")
};

const ctx = elements.canvas.getContext("2d");
const renderCtx = elements.render.getContext("2d");
const store = createStrokeStore();
let capture = null;
let dictionary = null;
let analyzeToken = 0;

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function classifyRadialFacing(directedAngle, radialAngle) {
  const candidates = {
    outward: angularDifference(directedAngle, radialAngle),
    inward: angularDifference(directedAngle, radialAngle + 180),
    counterclockwise: angularDifference(directedAngle, radialAngle + 90),
    clockwise: angularDifference(directedAngle, radialAngle - 90)
  };
  let bestKey = "unclear";
  let bestValue = 48;
  for (const [key, value] of Object.entries(candidates)) {
    if (value < bestValue) {
      bestValue = value;
      bestKey = key;
    }
  }
  return bestKey;
}

// Mirrors sigilSignDetectorLab.buildStandaloneCandidate: turn the drawn strokes
// into a single candidate the recognizers can consume, with a synthetic ring.
function buildStandaloneCandidate(strokes) {
  if (!strokes.length || !allPoints(strokes).length) {
    return null;
  }
  const points = allPoints(strokes);
  const bounds = boundsForStrokes(strokes);
  const center = centerOfBounds(bounds);
  const canvasCenter = { x: elements.canvas.width / 2, y: elements.canvas.height / 2 };
  const ringRadius = Math.min(elements.canvas.width, elements.canvas.height) * 0.42;
  const length = strokes.reduce((sum, stroke) => sum + strokeLength(stroke), 0);
  const size = Math.max(bounds.width, bounds.height);
  const directedOrientationDeg = directedStrokeAngle(strokes);
  const angleDeg = angleDegFromCenter(center, canvasCenter);
  const compactPerimeter = Math.max(1, (bounds.width + bounds.height) * 2);
  const overdrawAmount = clamp(length / compactPerimeter - 0.72, 0, 1);

  return {
    candidateId: "lab-candidate",
    strokeIds: strokes.map((stroke) => stroke.id),
    rawStrokeCount: strokes.length,
    cleanedStrokeCount: strokes.length,
    bounds,
    center,
    radiusNorm: 0.5,
    angleDeg,
    layer: "any",
    nearBoundary: false,
    sizeNorm: size / Math.max(1, ringRadius * 2),
    lengthNorm: length / Math.max(1, Math.PI * 2 * ringRadius),
    orientationDeg: dominantAxisOrientationDeg(points),
    directedOrientationDeg,
    radialFacing: classifyRadialFacing(directedOrientationDeg, angleDeg),
    closedness: endpointClosedness(strokes, Math.max(1, size)),
    overdrawAmount,
    neatness: clamp(0.92 - overdrawAmount * 0.28 - Math.max(0, strokes.length - 4) * 0.035),
    strokes
  };
}

function drawNetInput(candidate) {
  const image = renderCtx.createImageData(56, 56);
  if (candidate) {
    const points = candidate.strokes.map((stroke) => stroke.points ?? stroke);
    const raster = rasterizeStrokes(points, { size: 56 });
    for (let index = 0; index < raster.length; index += 1) {
      const value = raster[index];
      image.data[index * 4] = value;
      image.data[index * 4 + 1] = value;
      image.data[index * 4 + 2] = value;
      image.data[index * 4 + 3] = 255;
    }
  } else {
    for (let index = 0; index < 56 * 56; index += 1) {
      image.data[index * 4 + 3] = 255;
    }
  }
  renderCtx.putImageData(image, 0, 0);
}

function renderTemplateMatches(recognition) {
  if (!recognition) {
    elements.templateStatus.textContent = "—";
    elements.templateMatches.innerHTML = "";
    return;
  }
  elements.templateStatus.innerHTML = recognition.recognized
    ? `<b>${recognition.displayName ?? recognition.id}</b> · ${percent(recognition.confidence)} · ${recognition.recognitionStatus}`
    : `no confident match · ${recognition.recognitionStatus}`;
  const top = recognition.diagnostics?.topMatches ?? [];
  elements.templateMatches.innerHTML = top
    .map((match, index) => matchRow(match.id, match.kind, match.confidence, index === 0))
    .join("");
}

function renderSiameseMatches(recognition) {
  if (!recognition) {
    elements.siameseStatus.textContent = "draw a symbol";
    elements.siameseMatches.innerHTML = "";
    return;
  }
  elements.siameseStatus.innerHTML = recognition.recognized
    ? `<b>${recognition.displayName ?? recognition.id}</b> · ${percent(recognition.confidence)} · ${recognition.recognitionStatus}`
    : `no confident match · ${recognition.recognitionStatus}`;
  const top = recognition.diagnostics?.topMatches ?? [];
  elements.siameseMatches.innerHTML = top
    .map((match, index) =>
      matchRow(match.id, match.kind, match.confidence, index === 0, match.cosine, match.rotationDeg)
    )
    .join("");
}

function matchRow(id, kind, confidence, best, cosine, rotationDeg) {
  const extra =
    cosine === undefined
      ? ""
      : ` · cos ${cosine.toFixed(3)} · ${Math.round(rotationDeg ?? 0)}°`;
  return `
    <div class="match">
      <div class="match-head">
        <span class="${best ? "best" : ""}">${id} <span class="muted">(${kind})</span></span>
        <span>${percent(confidence)}${extra}</span>
      </div>
      <div class="bar"><span style="width: ${Math.round((confidence ?? 0) * 100)}%"></span></div>
    </div>
  `;
}

// runSiamese is false during live drawing (onPreview): the siamese pass is 12
// ONNX inferences on a single wasm thread and would jank the UI on every
// pointer move. We refresh the cheap render + template matcher continuously and
// only run the learned recognizer when a stroke is committed.
async function analyze(runSiamese = true) {
  if (!dictionary) {
    return;
  }
  const token = ++analyzeToken;
  const current = capture?.getCurrentStroke();
  const rawStrokes = current ? [...store.getStrokes(), current] : store.getStrokes();
  const candidate = buildStandaloneCandidate(cleanStrokes(rawStrokes, CONFIG));
  drawNetInput(candidate);
  elements.undoButton.disabled = store.count() === 0;

  if (!candidate) {
    renderSiameseMatches(null);
    renderTemplateMatches(null);
    return;
  }

  // Template engine is synchronous; run it immediately on every update.
  renderTemplateMatches(recognizeWithTemplates([candidate], dictionary, CONFIG)[0] ?? null);

  if (!runSiamese) {
    elements.siameseStatus.textContent = "release to recognize…";
    return;
  }

  // Siamese engine is async (ONNX inference); guard against stale completions.
  elements.siameseStatus.textContent = "recognizing…";
  try {
    const [recognition] = await recognizeCandidatesSiamese([candidate], dictionary, CONFIG);
    if (token === analyzeToken) {
      renderSiameseMatches(recognition ?? null);
    }
  } catch (error) {
    if (token === analyzeToken) {
      elements.siameseStatus.textContent = `siamese error: ${error.message}`;
    }
  }
}

function render() {
  drawPaper(ctx, elements.canvas.width, elements.canvas.height);
  drawStrokes(ctx, store.getStrokes(), capture?.getCurrentStroke(), CONFIG);
  requestAnimationFrame(render);
}

async function init() {
  elements.undoButton.addEventListener("click", () => {
    store.undo();
    analyze();
  });
  elements.clearButton.addEventListener("click", () => {
    store.clear();
    analyze();
  });

  capture = new DrawingCapture(elements.canvas, store, CONFIG, {
    onPreview: () => analyze(false),
    onCommit: () => analyze(true)
  });

  try {
    dictionary = await loadDictionary();
    capture.enable();
    elements.siameseStatus.textContent = "loading model…";
    warmupSiamese(CONFIG).then((ok) => {
      if (store.count() === 0) {
        elements.siameseStatus.textContent = ok ? "draw a symbol" : "model failed to load";
      }
    });
    await analyze(false);
    requestAnimationFrame(render);
  } catch (error) {
    console.error(error);
    elements.siameseStatus.textContent = `init failed: ${error.message}`;
  }
}

init();
