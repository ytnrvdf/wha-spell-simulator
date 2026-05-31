import { CONFIG } from "./config.js";
import { loadDictionary } from "./dictionary/dictionaryLoader.js";
import { DrawingCapture } from "./input/drawingCapture.js";
import { createStrokeStore } from "./input/strokeStore.js";
import { classifyDrawing } from "./parser/drawingClassifier.js";
import { warmupRecognizer } from "./parser/recognizerEngine.js";
import { compileSpell } from "./compiler/spellBuilder.js";
import { CanvasRenderer } from "./renderer/canvasRenderer.js";
import { setupCanvasSizing as setupResponsiveCanvasSizing } from "./ui/canvasSizing.js";
import { updateDiagnostics, updateDiagnosticsMode } from "./ui/diagnosticsView.js";
import { getElements } from "./ui/elements.js";
import { renderDictionaryReference } from "./ui/dictionaryReferenceView.js";
import { updateStatus, updateSummary } from "./ui/spellSummaryView.js";
import { setupTabs } from "./ui/tabs.js";

// Allow overriding the recognizer engine from the URL (?engine=siamese) so both
// engines can be compared on the same build without editing config + rebuilding.
// The committed default in config.js stays "template".
const engineOverride = new URLSearchParams(window.location.search).get("engine");
const APP_CONFIG = engineOverride
  ? { ...CONFIG, recognition: { ...CONFIG.recognition, engine: engineOverride } }
  : CONFIG;

const elements = getElements();
const store = createStrokeStore();
let dictionary = null;
let renderer = null;
let capture = null;
let pipeline = null;
let spellIR = null;
let previousRing = null;
let resizeObserver = null;

function setupCanvasSizing() {
  resizeObserver = setupResponsiveCanvasSizing({
    elements,
    store,
    onCanvasResized: () => {
      previousRing = null;
      recompute();
    }
  });
}

let recomputeToken = 0;

async function recompute() {
  if (!dictionary) {
    return;
  }

  // Recognition can be async (the siamese engine awaits ONNX inference). Guard
  // against out-of-order completions so a slow run can't overwrite a newer one.
  const token = ++recomputeToken;
  const nextPipeline = await classifyDrawing({
    strokes: store.getStrokes(),
    previousRing,
    dictionary,
    config: APP_CONFIG
  });
  if (token !== recomputeToken) {
    return;
  }

  pipeline = nextPipeline;
  previousRing = pipeline.ring;
  spellIR = compileSpell({ glyphAST: pipeline.glyphAST, dictionary, config: APP_CONFIG });
  updateSummary({ elements, store, capture, pipeline, spellIR });
  updateDiagnostics({ elements, store, pipeline, spellIR });
}

function animationFrame(timestamp) {
  renderer.renderGlyph({
    strokes: store.getStrokes(),
    currentStroke: capture.getCurrentStroke(),
    pipeline,
    showGuides: elements.guidesToggle.checked,
    showDebug: elements.diagnosticsToggle.checked
  });

  if (spellIR.active) {
    renderer.renderActivatedGlyph({
      activatedAt: spellIR.activatedAt,
      duration: spellIR.duration,
      strokes: store.getStrokes(),
      pipeline,
      timestamp
    });
  }
  
  renderer.renderEffect({
    spellIR,
    ring: pipeline?.ring,
    timestamp,
    showGuides: elements.guidesToggle.checked
  });
  requestAnimationFrame(animationFrame);
}

function setupControls() {
  elements.undoButton.addEventListener("click", () => {
    store.undo();
    previousRing = null;
    recompute();
  });

  elements.clearButton.addEventListener("click", () => {
    store.clear();
    previousRing = null;
    recompute();
  });

  elements.guidesToggle.addEventListener("change", () => {
    updateSummary({ elements, store, capture, pipeline, spellIR });
    updateDiagnostics({ elements, store, pipeline, spellIR });
  });

  elements.diagnosticsToggle.addEventListener("change", () => {
    updateDiagnosticsMode(elements);
    updateDiagnostics({ elements, store, pipeline, spellIR });
  });

  updateDiagnosticsMode(elements);
}

async function init() {
  setupTabs(elements);
  setupControls();
  setupCanvasSizing();
  renderer = new CanvasRenderer({
    glyphCanvas: elements.glyphCanvas,
    effectCanvas: elements.effectCanvas,
    config: CONFIG
  });
  capture = new DrawingCapture(elements.glyphCanvas, store, CONFIG, {
    onPreview: () => {},
    onCommit: recompute
  });

  try {
    dictionary = await loadDictionary();
    renderDictionaryReference(elements, dictionary);
    capture.enable();
    warmupRecognizer(APP_CONFIG); // preload the siamese model if that engine is selected
    await recompute();
    requestAnimationFrame(animationFrame);
  } catch (error) {
    console.error(error);
    updateStatus(elements, "Dictionary load failed", "invalid");
  }
}

init();
