// Chooses the symbol recognizer at runtime. The default "template" engine is
// the app's original hand-tuned stroke matcher (synchronous); the "siamese"
// engine is the learned ONNX recognizer (asynchronous). Both honor the exact
// same recognizeCandidates(candidates, dictionary, config) contract, so the
// rest of the parser is agnostic to which one runs.
//
// The dispatcher is async and always returns a promise; the template path just
// resolves synchronously, so switching engines never changes call sites.

import { recognizeCandidates as recognizeWithTemplates } from "./symbolRecognizer.js";

export function recognizerEngineName(config) {
  return config.recognition?.engine ?? "template";
}

export async function recognizeCandidates(candidates, dictionary, config) {
  if (recognizerEngineName(config) === "siamese") {
    // Lazy import so the onnxruntime-web bundle is only fetched when the
    // siamese engine is actually selected.
    const { recognizeCandidatesSiamese } = await import("./siameseRecognizer.js");
    return recognizeCandidatesSiamese(candidates, dictionary, config);
  }
  return recognizeWithTemplates(candidates, dictionary, config);
}

// Optional: warm the selected engine ahead of first use (no-op for template).
export async function warmupRecognizer(config) {
  if (recognizerEngineName(config) === "siamese") {
    const { warmupSiamese } = await import("./siameseRecognizer.js");
    return warmupSiamese(config);
  }
  return true;
}
