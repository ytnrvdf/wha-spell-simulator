import { CONFIG } from "./config.js";
import { loadDictionary } from "./dictionary/dictionaryLoader.js";
import { DrawingCapture } from "./input/drawingCapture.js";
import { createStrokeStore } from "./input/strokeStore.js";
import { createPlacementStore } from "./input/placementStore.js";
import { buildShapeLibrary, defaultTransformForShape } from "./input/shapeLibrary.js";
import { bakePlacementToStrokes, placementHandles } from "./input/shapeBaker.js";
import { PlacementController } from "./input/placementController.js";
import { classifyDrawing } from "./parser/drawingClassifier.js";
import { compileSpell } from "./compiler/spellBuilder.js";
import { CanvasRenderer } from "./renderer/canvasRenderer.js";
import { setupCanvasSizing as setupResponsiveCanvasSizing } from "./ui/canvasSizing.js";
import { updateDiagnostics, updateDiagnosticsMode } from "./ui/diagnosticsView.js";
import { getElements } from "./ui/elements.js";
import { renderDictionaryReference } from "./ui/dictionaryReferenceView.js";
import {
  clearArmedHighlight,
  renderShapePalette,
  syncShapeInspectorValues,
  updateShapeInspector
} from "./ui/shapePaletteView.js";
import { updateStatus, updateSummary } from "./ui/spellSummaryView.js";
import { setupTabs } from "./ui/tabs.js";

const elements = getElements();
const store = createStrokeStore();
const placements = createPlacementStore();
const actionHistory = [];
let dictionary = null;
let shapeLibrary = null;
let renderer = null;
let capture = null;
let controller = null;
let pipeline = null;
let spellIR = null;
let previousRing = null;
let resizeObserver = null;
let currentStrokes = [];
let armedShape = null;
let selectedPlacementId = null;
let arrangeMode = false;

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

function mergedStrokes() {
  const stamped = placements.getPlacements().flatMap(bakePlacementToStrokes);
  return [...store.getStrokes(), ...stamped];
}

function refreshViews() {
  updateSummary({ elements, store, capture, pipeline, spellIR, arrangeMode, placementCount: placements.count() });
  updateDiagnostics({ elements, strokes: currentStrokes, pipeline, spellIR });
}

function recompute() {
  if (!dictionary) {
    return;
  }

  currentStrokes = mergedStrokes();
  pipeline = classifyDrawing({
    strokes: currentStrokes,
    previousRing,
    dictionary,
    config: CONFIG
  });
  previousRing = pipeline.ring;
  spellIR = compileSpell({ glyphAST: pipeline.glyphAST, dictionary, config: CONFIG });
  refreshViews();
}

function selectedPlacement() {
  return selectedPlacementId ? placements.get(selectedPlacementId) : null;
}

function refreshInspector() {
  updateShapeInspector(elements, selectedPlacement(), {
    onChange: (patch) => {
      if (selectedPlacementId) {
        placements.update(selectedPlacementId, patch);
        recompute();
      }
    },
    onCommit: () => {
      if (selectedPlacementId) {
        commitPlacement(selectedPlacementId);
        recompute();
      }
    },
    onRemove: () => {
      if (selectedPlacementId) {
        deletePlacement(selectedPlacementId);
      }
    }
  });
}

// Bake a placement into the stroke store as permanent ink, then drop the editable
// placement so it behaves exactly like hand-drawn strokes.
function commitPlacement(id) {
  const placement = placements.get(id);
  if (!placement) {
    return;
  }
  const baked = bakePlacementToStrokes(placement);
  baked.forEach((stroke) => store.addStroke(stroke.points));
  placements.remove(id);

  const index = actionHistory.findIndex((entry) => entry.kind === "placement" && entry.id === id);
  if (index >= 0) {
    actionHistory.splice(index, 1);
  }
  actionHistory.push({ kind: "strokeGroup", count: baked.length });

  if (selectedPlacementId === id) {
    setSelected(null);
  }
}

function commitAllPlacements() {
  placements
    .getPlacements()
    .map((placement) => placement.id)
    .forEach((id) => commitPlacement(id));
}

function setSelected(id) {
  selectedPlacementId = id;
  refreshInspector();
}

function placeArmedShape(point) {
  if (!armedShape) {
    return null;
  }
  const placement = placements.add({
    kind: armedShape.kind,
    sourceId: armedShape.sourceId,
    baseStrokes: armedShape.baseStrokes,
    transform: defaultTransformForShape(armedShape, point, elements.glyphCanvas)
  });
  actionHistory.push({ kind: "placement", id: placement.id });
  armedShape = null;
  clearArmedHighlight(elements);
  setSelected(placement.id);
  recompute();
  return placement.id;
}

function deletePlacement(id) {
  placements.remove(id);
  const index = actionHistory.findIndex((entry) => entry.kind === "placement" && entry.id === id);
  if (index >= 0) {
    actionHistory.splice(index, 1);
  }
  if (selectedPlacementId === id) {
    setSelected(null);
  }
  recompute();
}

function setArrangeMode(on) {
  arrangeMode = on;
  elements.arrangeToggle.checked = on;
  controller?.setActive(on);
  elements.glyphCanvas.style.cursor = on ? "default" : "crosshair";
  if (!on) {
    armedShape = null;
    clearArmedHighlight(elements);
    commitAllPlacements();
    setSelected(null);
  }
  recompute();
}

function armShape(item) {
  armedShape = item;
  if (!arrangeMode) {
    setArrangeMode(true);
  }
}

function animationFrame(timestamp) {
  const placement = arrangeMode ? selectedPlacement() : null;
  renderer.renderGlyph({
    strokes: currentStrokes,
    currentStroke: capture.getCurrentStroke(),
    pipeline,
    showGuides: elements.guidesToggle.checked,
    showDebug: elements.diagnosticsToggle.checked,
    selection: placement ? placementHandles(placement) : null
  });

  if (spellIR.active) {
    renderer.renderActivatedGlyph({
      activatedAt: spellIR.activatedAt,
      duration: spellIR.duration,
      strokes: currentStrokes,
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
    const last = actionHistory.pop();
    if (last?.kind === "placement") {
      placements.remove(last.id);
      if (selectedPlacementId === last.id) {
        setSelected(null);
      }
    } else if (last?.kind === "strokeGroup") {
      for (let index = 0; index < last.count; index += 1) {
        store.undo();
      }
    } else {
      store.undo();
    }
    previousRing = null;
    recompute();
  });

  elements.clearButton.addEventListener("click", () => {
    store.clear();
    placements.clear();
    actionHistory.length = 0;
    armedShape = null;
    setSelected(null);
    clearArmedHighlight(elements);
    previousRing = null;
    recompute();
  });

  elements.guidesToggle.addEventListener("change", refreshViews);

  elements.diagnosticsToggle.addEventListener("change", () => {
    updateDiagnosticsMode(elements);
    refreshViews();
  });

  elements.arrangeToggle.addEventListener("change", () => {
    setArrangeMode(elements.arrangeToggle.checked);
  });

  window.addEventListener("keydown", (event) => {
    if (!arrangeMode || !selectedPlacementId) {
      return;
    }
    const target = event.target;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deletePlacement(selectedPlacementId);
    }
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
    onCommit: () => {
      actionHistory.push({ kind: "stroke" });
      recompute();
    }
  });
  controller = new PlacementController(elements.glyphCanvas, placements, {
    getSelectedId: () => selectedPlacementId,
    setSelectedId: setSelected,
    getArmedShape: () => armedShape,
    placeShape: placeArmedShape,
    onChange: () => {
      recompute();
      syncShapeInspectorValues(elements, selectedPlacement());
    }
  });
  controller.enable();
  controller.setActive(false);
  elements.glyphCanvas.style.cursor = "crosshair";

  try {
    dictionary = await loadDictionary();
    shapeLibrary = buildShapeLibrary(dictionary);
    renderDictionaryReference(elements, dictionary);
    renderShapePalette(elements, shapeLibrary, armShape);
    refreshInspector();
    capture.enable();
    recompute();
    requestAnimationFrame(animationFrame);
  } catch (error) {
    console.error(error);
    updateStatus(elements, "Dictionary load failed", "invalid");
  }
}

init();
