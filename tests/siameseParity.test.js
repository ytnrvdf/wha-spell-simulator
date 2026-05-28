import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { rasterizeStrokes } from "../src/parser/glyphRasterizer.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = join(here, "fixtures", "siameseParity.json");

let fixture = null;
try {
  fixture = JSON.parse(readFileSync(fixturePath, "utf-8"));
} catch {
  fixture = null;
}

function toPointArrays(strokes) {
  return strokes.map((stroke) => stroke.map(([x, y]) => ({ x, y })));
}

// HARD GATE: the JS rasterizer must reproduce the Python training rasterizer
// pixel-for-pixel, or prototypes (built in Python) and browser candidates live
// in different input distributions (train/serve skew).
test("glyphRasterizer matches the Python dict_raster pixel-for-pixel", () => {
  assert.ok(
    fixture,
    `Missing ${fixturePath}. Generate it with siamese/make_parity_fixture.py.`
  );

  for (const caseData of fixture.rasterCases) {
    const raster = rasterizeStrokes(toPointArrays(caseData.strokes), {
      size: caseData.size,
      rotationDeg: caseData.rotationDeg
    });
    assert.equal(
      raster.length,
      caseData.raster.length,
      `${caseData.name}@${caseData.rotationDeg}: length mismatch`
    );
    let firstDiff = -1;
    for (let index = 0; index < raster.length; index += 1) {
      if (raster[index] !== caseData.raster[index]) {
        firstDiff = index;
        break;
      }
    }
    assert.equal(
      firstDiff,
      -1,
      `${caseData.name}@${caseData.rotationDeg}: pixel differs at index ${firstDiff} ` +
        `(js=${raster[firstDiff]} py=${caseData.raster[firstDiff]})`
    );
  }
});

// BEST-EFFORT: end-to-end embedding parity through onnxruntime-web. Skipped if
// the wasm runtime cannot initialize under plain Node (export_onnx.py already
// guards torch<->onnx parity, so this is a secondary cross-runtime check).
test("onnxruntime-web embedding matches the Python embedding", async (t) => {
  if (!fixture?.embeddingCase) {
    t.skip("no embedding case in fixture");
    return;
  }

  let ort;
  let session;
  try {
    ort = await import("onnxruntime-web");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = join(here, "..", "node_modules", "onnxruntime-web", "dist", "/");
    const modelBytes = readFileSync(join(here, "..", "src", "parser", "siamese-assets", "model.onnx"));
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  } catch (error) {
    t.skip(`onnxruntime-web unavailable in Node: ${error.message}`);
    return;
  }

  const { embeddingCase, size } = fixture;
  const raster = rasterizeStrokes(toPointArrays(embeddingCase.strokes), {
    size,
    rotationDeg: embeddingCase.rotationDeg
  });
  const input = Float32Array.from(raster, (value) => value / 255);
  const tensor = new ort.Tensor("float32", input, [1, 1, size, size]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  const embedding = output[session.outputNames[0]].data;

  const expected = embeddingCase.embedding;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let index = 0; index < expected.length; index += 1) {
    dot += embedding[index] * expected[index];
    na += embedding[index] * embedding[index];
    nb += expected[index] * expected[index];
  }
  const cosine = dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
  assert.ok(cosine > 0.999, `embedding cosine too low: ${cosine}`);
});
