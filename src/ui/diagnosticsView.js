import { buildDiagnosticState } from "../debug/diagnosticState.js";
import { writeJson } from "../debug/debugOverlay.js";

export function updateDiagnostics({ elements, strokes, pipeline, spellIR }) {
  const diagnosticState = buildDiagnosticState({
    rawStrokes: strokes,
    pipeline,
    spellIR
  });

  writeJson(elements.astPanel, diagnosticState.glyphAST);
  writeJson(elements.irPanel, diagnosticState.spellIR);
  writeJson(elements.parserPanel, {
    rawStrokes: diagnosticState.rawStrokes,
    ring: diagnosticState.ring,
    classifications: diagnosticState.classifications,
    candidates: diagnosticState.candidates,
    recognitions: diagnosticState.recognitions
  });
}

export async function copyDiagnosticPanel(panelId, button) {
  const panel = document.getElementById(panelId);
  const text = panel?.dataset.rawJson ?? panel?.textContent ?? "";
  if (!text.trim()) {
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = "Copy";
    }, 900);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(panel);
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
}

export function updateDiagnosticsMode(elements) {
  document.body.classList.toggle("diagnostics-visible", elements.diagnosticsToggle.checked);
}
