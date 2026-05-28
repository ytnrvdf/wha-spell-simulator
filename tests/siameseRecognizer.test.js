import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  recognizeCandidatesSiamese,
  __setSiameseRuntime,
  __resetSiameseRuntime
} from "../src/parser/siameseRecognizer.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..");

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf-8"));
}

function loadDictionary() {
  return {
    sigils: readJson(join(repo, "src", "dictionary", "sigils.json")),
    signs: readJson(join(repo, "src", "dictionary", "signs.json"))
  };
}

// Build a candidate straight from a dictionary entry's own stroke template:
// the "perfect drawing" of that glyph. It should recognize as itself.
function candidateFromEntry(entry, kind) {
  return {
    candidateId: `c-${entry.id}`,
    strokeIds: [entry.id],
    rawStrokeCount: entry.strokeTemplate.strokes.length,
    layer: "any",
    nearBoundary: false,
    radiusNorm: 0.3,
    angleDeg: 0,
    sizeNorm: 0.2,
    lengthNorm: 0.2,
    orientationDeg: 0,
    directedOrientationDeg: 0,
    radialFacing: "unclear",
    overdrawAmount: 0,
    neatness: 0.9,
    strokes: entry.strokeTemplate.strokes.map((stroke) => ({
      points: stroke.map((point) => ({ x: point.x, y: point.y }))
    }))
  };
}

async function buildRuntime() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = join(repo, "node_modules", "onnxruntime-web", "dist", "/");
  const assets = join(repo, "src", "parser", "siamese-assets");
  const modelBytes = readFileSync(join(assets, "model.onnx"));
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const prototypeData = readJson(join(assets, "prototypes.json"));
  return {
    session,
    prototypes: prototypeData.glyphs.map((glyph) => ({
      id: glyph.id,
      kind: glyph.kind,
      embedding: Float32Array.from(glyph.embedding)
    })),
    imageSize: prototypeData.imageSize ?? 56,
    inputName: session.inputNames[0],
    outputName: session.outputNames[0],
    settings: {
      minConfidence: 0.48,
      rotationsDeg: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      temperature: 0.08
    }
  };
}

test("siamese recognizer identifies every dictionary glyph from its own template", async (t) => {
  let runtime;
  try {
    runtime = await buildRuntime();
  } catch (error) {
    t.skip(`onnxruntime-web / artifacts unavailable in Node: ${error.message}`);
    return;
  }
  __setSiameseRuntime(runtime);
  t.after(__resetSiameseRuntime);

  const dictionary = loadDictionary();
  const entries = [
    ...dictionary.sigils.map((entry) => ({ entry, kind: "sigil" })),
    ...dictionary.signs.map((entry) => ({ entry, kind: "sign" }))
  ];
  const candidates = entries.map(({ entry, kind }) => candidateFromEntry(entry, kind));

  const recognitions = await recognizeCandidatesSiamese(candidates, dictionary, {
    recognition: {}
  });

  for (let index = 0; index < entries.length; index += 1) {
    const expected = entries[index].entry.id;
    const recognition = recognitions[index];
    const top = recognition.diagnostics.topMatches[0];
    // Nearest prototype must be the glyph itself.
    assert.equal(top.id, expected, `nearest prototype for ${expected} was ${top.id}`);
    // And a clean rendering should clear the acceptance bar.
    assert.equal(recognition.recognized, true, `${expected} not recognized (status ${recognition.recognitionStatus})`);
    assert.equal(recognition.id, expected);
  }
});
