function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

function getStrokeTemplate(entry) {
  return entry.strokeTemplate ?? null;
}

function entryRecognitionLabel(entry) {
  return getStrokeTemplate(entry)?.strokes?.length ? "stroke reference" : "not configured";
}

export function renderStrokePreview(strokes) {
  if (!strokes?.length) {
    return "";
  }

  const polylines = strokes
    .map((stroke) => {
      const points = stroke
        .map((point) => {
          const x = Number(point.x);
          const y = Number(point.y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            return null;
          }
          const previewX = 8 + x * 84;
          const previewY = 8 + y * 84;
          return `${Math.round(previewX * 10) / 10},${Math.round(previewY * 10) / 10}`;
        })
        .filter(Boolean)
        .join(" ");

      return points ? `<polyline points="${points}"></polyline>` : "";
    })
    .join("");

  return `
    <div class="reference-preview" aria-hidden="true">
      <svg viewBox="0 0 100 100" role="img" focusable="false">
        ${polylines}
      </svg>
    </div>
  `;
}

function renderStrokeTemplatePreview(entry) {
  return renderStrokePreview(getStrokeTemplate(entry)?.strokes);
}

function renderReferenceCard(entry, kind) {
  const layerText = entry.allowedLayers?.join(", ") || "any";
  const elementText = kind === "sigil" && entry.element ? entry.element : "";
  const elementBadge = elementText ? `<span>${escapeHtml(elementText)}</span>` : "";
  const hasStrokeReference = Boolean(getStrokeTemplate(entry)?.strokes?.length);
  const sourceDetails =
    kind === "sign" && entry.sourceNotes
      ? `
        <details class="reference-source">
          <summary>Source notes</summary>
          <p>${escapeHtml(entry.sourceNotes)}</p>
        </details>
      `
      : "";
  return `
    <article class="reference-card ${hasStrokeReference ? "has-template" : ""}">
      ${renderStrokeTemplatePreview(entry)}
      <div>
        <div class="reference-card-header">
          <strong>${escapeHtml(entry.displayName ?? entry.id)}</strong>
          ${elementBadge}
        </div>
        <dl>
          <div><dt>Layers</dt><dd>${escapeHtml(layerText)}</dd></div>
          <div><dt>Recognition</dt><dd>${escapeHtml(entryRecognitionLabel(entry))}</dd></div>
        </dl>
        ${sourceDetails}
      </div>
    </article>
  `;
}

function renderSampleSpellCard(sample) {
  const manifestations = sample.manifestations?.length ? sample.manifestations.join(", ") : "none";
  const hasStrokeReference = Boolean(sample.strokes?.length);
  return `
    <article class="reference-card ${hasStrokeReference ? "has-template" : ""}">
      ${renderStrokePreview(sample.strokes)}
      <div>
        <div class="reference-card-header">
          <strong>${escapeHtml(sample.displayName ?? sample.id)}</strong>
          ${sample.element ? `<span>${escapeHtml(sample.element)}</span>` : ""}
        </div>
        <p class="reference-card-description">${escapeHtml(sample.description)}</p>
        <dl>
          <div><dt>Element</dt><dd>${escapeHtml(sample.element ?? "none")}</dd></div>
          <div><dt>Manifestations</dt><dd>${escapeHtml(manifestations)}</dd></div>
        </dl>
      </div>
    </article>
  `;
}

export function renderDictionaryReference(elements, dictionary) {
  if (!dictionary) {
    return;
  }

  elements.sampleSpellReferenceCards.innerHTML = (dictionary.sampleSpells ?? [])
    .map((sample) => renderSampleSpellCard(sample))
    .join("");
  elements.sigilReferenceCards.innerHTML = (dictionary.sigils ?? [])
    .map((entry) => renderReferenceCard(entry, "sigil"))
    .join("");
  elements.signReferenceCards.innerHTML = (dictionary.signs ?? [])
    .map((entry) => renderReferenceCard(entry, "sign"))
    .join("");
}
