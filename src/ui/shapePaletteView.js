import { renderStrokePreview } from "./dictionaryReferenceView.js";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

function renderShapeCard(item) {
  const elementBadge = item.element ? `<span>${escapeHtml(item.element)}</span>` : "";
  return `
    <button type="button" class="shape-card" data-shape-id="${escapeHtml(item.id)}">
      ${renderStrokePreview(item.baseStrokes)}
      <span class="shape-card-label">
        <strong>${escapeHtml(item.label)}</strong>
        ${elementBadge}
      </span>
    </button>
  `;
}

function renderShapeGroup(title, items) {
  if (!items.length) {
    return "";
  }
  return `
    <div class="shape-group">
      <h3 class="shape-group-title">${escapeHtml(title)}</h3>
      <div class="shape-card-grid">${items.map(renderShapeCard).join("")}</div>
    </div>
  `;
}

export function renderShapePalette(elements, library, onArm) {
  elements.shapePaletteCards.innerHTML = [
    renderShapeGroup("Ring", [library.ring]),
    renderShapeGroup("Sigils", library.sigils),
    renderShapeGroup("Signs", library.signs)
  ].join("");

  elements.shapePaletteCards.querySelectorAll(".shape-card").forEach((button) => {
    button.addEventListener("click", () => {
      const item = library.items.find((entry) => entry.id === button.dataset.shapeId);
      if (item) {
        elements.shapePaletteCards.querySelectorAll(".shape-card").forEach((card) => card.classList.remove("armed"));
        button.classList.add("armed");
        onArm(item);
      }
    });
  });
}

export function clearArmedHighlight(elements) {
  elements.shapePaletteCards.querySelectorAll(".shape-card.armed").forEach((card) => card.classList.remove("armed"));
}

export function updateShapeInspector(elements, placement, handlers) {
  const container = elements.shapeInspector;
  if (!placement) {
    container.innerHTML = `<p class="panel-description">Pick a shape above, then click the canvas to place it. Select a placed shape to move, scale, elongate, or rotate it.</p>`;
    return;
  }

  const { transform } = placement;
  container.innerHTML = `
    <div class="shape-inspector-card">
      <div class="reference-card-header">
        <strong>${escapeHtml(placement.sourceId)}</strong>
        <span>${escapeHtml(placement.kind)}</span>
      </div>
      <label class="shape-field">Width
        <input type="range" min="24" max="1200" data-field="scaleX" value="${Math.round(transform.scaleX)}">
      </label>
      <label class="shape-field">Height
        <input type="range" min="24" max="1200" data-field="scaleY" value="${Math.round(transform.scaleY)}">
      </label>
      <label class="shape-field">Rotation
        <input type="range" min="-180" max="180" data-field="rotationDeg" value="${Math.round(transform.rotationDeg)}">
      </label>
      <button type="button" class="shape-commit" data-action="commit">Place shape (lock)</button>
      <button type="button" class="shape-remove" data-action="remove">Remove shape</button>
      <p class="panel-description">Locking bakes the shape into the drawing as ink. It can no longer be moved, but you can draw over it.</p>
    </div>
  `;

  container.querySelectorAll("input[data-field]").forEach((input) => {
    input.addEventListener("input", () => {
      handlers.onChange({ [input.dataset.field]: Number(input.value) });
    });
  });
  container.querySelector("[data-action='commit']")?.addEventListener("click", () => handlers.onCommit());
  container.querySelector("[data-action='remove']")?.addEventListener("click", () => handlers.onRemove());
}

// Refresh slider positions during a canvas drag without rebuilding the inspector.
export function syncShapeInspectorValues(elements, placement) {
  if (!placement) {
    return;
  }
  const inputs = elements.shapeInspector.querySelectorAll("input[data-field]");
  inputs.forEach((input) => {
    const value = placement.transform[input.dataset.field];
    if (typeof value === "number") {
      input.value = String(Math.round(value));
    }
  });
}
