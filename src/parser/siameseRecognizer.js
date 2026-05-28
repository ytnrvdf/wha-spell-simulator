// Drop-in alternative to symbolRecognizer.recognizeCandidates that classifies
// each grouped candidate with a learned siamese embedding instead of the
// hand-tuned stroke-template ink overlap. It deliberately consumes the SAME
// candidate objects produced by strokeGrouper (so the app's stroke grouping /
// separated-symbol handling is reused untouched) and returns the SAME output
// shape downstream code expects.
//
// Pipeline per candidate: rasterize its strokes (glyphRasterizer, a port of the
// Python training rasterizer) at several rotations -> embed each via the ONNX
// model -> cosine-match against per-glyph prototype embeddings -> softmax with
// temperature -> map to recognized / ambiguous / unknown.

import * as ort from "onnxruntime-web";
import { clamp } from "../utils/geometry.js";
import { rasterizeStrokes, rasterToModelInput } from "./glyphRasterizer.js";

const RECOGNITION_AMBIGUITY_GAP = 0.065;
const MESSY_NEATNESS = 0.74;
const MESSY_OVERDRAW = 0.24;

let runtimePromise = null;

function siameseConfig(config) {
  const recognition = config.recognition ?? {};
  const siamese = recognition.siamese ?? {};
  return {
    minConfidence: recognition.minConfidence ?? 0.48,
    // When unset, the model + prototypes are loaded from the bundled assets
    // (dynamic imports below) so they resolve correctly from any page and under
    // the gh-pages subpath. Override with an explicit URL only for custom hosting.
    modelUrl: siamese.modelUrl ?? null,
    prototypesUrl: siamese.prototypesUrl ?? null,
    wasmPaths:
      siamese.wasmPaths ??
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/",
    rotationsDeg: siamese.rotationsDeg ?? [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    temperature: siamese.temperature ?? 0.08
  };
}

async function loadPrototypeData(settings) {
  if (settings.prototypesUrl) {
    const response = await fetch(settings.prototypesUrl);
    if (!response.ok) {
      throw new Error(`Failed to load prototypes: ${response.status}`);
    }
    return response.json();
  }
  // Bundled JSON (Vite inlines it; resolved relative to this module).
  return (await import("./siamese-assets/prototypes.json")).default;
}

async function resolveModelUrl(settings) {
  if (settings.modelUrl) {
    return settings.modelUrl;
  }
  // Vite emits a hashed, correctly-based asset URL (works from any page/subpath).
  return (await import("./siamese-assets/model.onnx?url")).default;
}

// Loads the ONNX session + prototypes once and caches them. The asset imports
// are dynamic so this module stays loadable under plain Node (tests inject a
// runtime via __setSiameseRuntime and never reach here).
function getRuntime(config) {
  if (runtimePromise) {
    return runtimePromise;
  }
  const settings = siameseConfig(config);
  runtimePromise = (async () => {
    ort.env.wasm.wasmPaths = settings.wasmPaths;
    // Single-threaded wasm: the multithreaded build needs SharedArrayBuffer,
    // which requires cross-origin isolation (COOP/COEP headers) that GitHub
    // Pages (and vite preview) do not send. Without this the worker never
    // initializes and inference hangs forever. The model is tiny, so one
    // thread is plenty.
    ort.env.wasm.numThreads = 1;
    const [modelUrl, prototypeData] = await Promise.all([
      resolveModelUrl(settings),
      loadPrototypeData(settings)
    ]);
    const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
    const prototypes = prototypeData.glyphs.map((glyph) => ({
      id: glyph.id,
      kind: glyph.kind,
      embedding: Float32Array.from(glyph.embedding)
    }));
    return {
      session,
      prototypes,
      imageSize: prototypeData.imageSize ?? 56,
      inputName: session.inputNames[0],
      outputName: session.outputNames[0],
      settings
    };
  })().catch((error) => {
    runtimePromise = null; // allow a later retry after a transient failure
    throw error;
  });
  return runtimePromise;
}

function l2normalize(vector, offset, length) {
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    const value = vector[offset + index];
    sum += value * value;
  }
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    out[index] = vector[offset + index] / norm;
  }
  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    sum += a[index] * b[index];
  }
  return sum;
}

function dictionaryIndex(dictionary) {
  const index = new Map();
  for (const entry of dictionary.sigils ?? []) {
    index.set(`sigil:${entry.id}`, entry);
  }
  for (const entry of dictionary.signs ?? []) {
    index.set(`sign:${entry.id}`, entry);
  }
  return index;
}

function strokePointArrays(candidate) {
  return candidate.strokes
    .map((stroke) => stroke.points ?? stroke)
    .filter((points) => points && points.length);
}

// Gentle layer prior: a candidate sitting in a layer the glyph never occupies
// (and not near a boundary) is down-weighted, mirroring the template engine's
// allowedLayerScore without re-implementing its full heuristics.
function layerFactor(entry, candidate) {
  const allowed = entry?.allowedLayers;
  if (!allowed?.length || candidate.layer === "any") {
    return 1;
  }
  if (allowed.includes(candidate.layer)) {
    return 1;
  }
  return candidate.nearBoundary ? 0.85 : 0.55;
}

function buildEmbeddingBatch(candidates, rotations, imageSize) {
  const stride = imageSize * imageSize;
  const perCandidate = rotations.length;
  const data = new Float32Array(candidates.length * perCandidate * stride);
  let cursor = 0;
  for (const candidate of candidates) {
    const points = strokePointArrays(candidate);
    for (const rotationDeg of rotations) {
      const raster = rasterizeStrokes(points, { size: imageSize, rotationDeg });
      data.set(rasterToModelInput(raster), cursor);
      cursor += stride;
    }
  }
  return { data, perCandidate, stride };
}

function publicCandidate(candidate) {
  return {
    candidateId: candidate.candidateId,
    strokeIds: candidate.strokeIds,
    rawStrokeCount: candidate.rawStrokeCount,
    layer: candidate.layer,
    nearBoundary: candidate.nearBoundary,
    radiusNorm: candidate.radiusNorm,
    angleDeg: candidate.angleDeg,
    sizeNorm: candidate.sizeNorm,
    lengthNorm: candidate.lengthNorm,
    orientationDeg: candidate.orientationDeg,
    directedOrientationDeg: candidate.directedOrientationDeg,
    radialFacing: candidate.radialFacing,
    overdrawAmount: candidate.overdrawAmount,
    neatness: candidate.neatness
  };
}

function scoreCandidate(candidate, embeddings, runtime, index, dictByKey) {
  const { prototypes, settings } = runtime;
  const { perCandidate, dim } = embeddings;
  // Best cosine over all tested rotations, per prototype.
  const best = prototypes.map(() => ({ cosine: -Infinity, rotationDeg: 0 }));
  for (let slot = 0; slot < perCandidate; slot += 1) {
    const offset = (index * perCandidate + slot) * dim;
    const embedding = l2normalize(embeddings.data, offset, dim);
    for (let p = 0; p < prototypes.length; p += 1) {
      const cosine = dot(embedding, prototypes[p].embedding);
      if (cosine > best[p].cosine) {
        best[p] = { cosine, rotationDeg: settings.rotationsDeg[slot] };
      }
    }
  }

  // Softmax over per-glyph best cosines -> confidence, with a layer prior.
  const scaled = prototypes.map((proto, p) => {
    const entry = dictByKey.get(`${proto.kind}:${proto.id}`);
    const factor = layerFactor(entry, candidate);
    return { proto, entry, cosine: best[p].cosine, rotationDeg: best[p].rotationDeg, factor };
  });
  const maxCosine = Math.max(...scaled.map((item) => item.cosine));
  let partition = 0;
  for (const item of scaled) {
    item.weight = Math.exp((item.cosine - maxCosine) / settings.temperature) * item.factor;
    partition += item.weight;
  }
  for (const item of scaled) {
    item.confidence = clamp(item.weight / (partition || 1));
  }
  scaled.sort((a, b) => b.confidence - a.confidence);
  return scaled;
}

function recognitionStatus(candidate, best, second, accepted) {
  if (!best) {
    return "unknown";
  }
  if (best.confidence - (second?.confidence ?? 0) < RECOGNITION_AMBIGUITY_GAP) {
    return "ambiguous";
  }
  if (!accepted) {
    return "unknown";
  }
  if (candidate.neatness <= MESSY_NEATNESS || candidate.overdrawAmount >= MESSY_OVERDRAW) {
    return "valid_messy";
  }
  return "valid";
}

function toRecognition(candidate, scored, settings) {
  const best = scored[0];
  const second = scored[1] ?? { confidence: 0 };
  const acceptedByConfidence = Boolean(best && best.confidence >= settings.minConfidence);
  const status = recognitionStatus(candidate, best, second, acceptedByConfidence);
  const accepted = acceptedByConfidence && (status === "valid" || status === "valid_messy");
  const entry = best?.entry ?? null;
  const topMatches = scored.slice(0, 3).map((item) => ({
    kind: item.proto.kind,
    id: item.proto.id,
    confidence: item.confidence,
    cosine: item.cosine,
    rotationDeg: item.rotationDeg
  }));

  return {
    ...publicCandidate(candidate),
    recognized: accepted,
    recognitionStatus: status,
    kind: accepted ? best.proto.kind : "unknown",
    id: accepted ? best.proto.id : null,
    displayName: accepted ? entry?.displayName ?? best.proto.id : null,
    element: accepted ? entry?.element ?? null : null,
    semantic: accepted ? entry?.semantic ?? null : null,
    confidence: accepted ? best.confidence : 0,
    shape: { strokeCount: candidate.strokes.length },
    diagnostics: {
      engine: "siamese",
      bestGuess: accepted
        ? null
        : best
          ? { kind: best.proto.kind, id: best.proto.id, confidence: best.confidence }
          : null,
      recognitionRotationDeg: best?.rotationDeg ?? 0,
      topMatches
    }
  };
}

// Kick off model + prototype loading ahead of the first recognition (e.g. while
// the user is still drawing), so the first commit isn't stalled by the download.
export function warmupSiamese(config) {
  return getRuntime(config).then(
    () => true,
    () => false
  );
}

export async function recognizeCandidatesSiamese(candidates, dictionary, config) {
  if (!candidates.length) {
    return [];
  }
  const runtime = await getRuntime(config);
  const { session, settings, imageSize, inputName, outputName } = runtime;
  const dictByKey = dictionaryIndex(dictionary);

  const { data, perCandidate } = buildEmbeddingBatch(candidates, settings.rotationsDeg, imageSize);
  const batch = candidates.length * perCandidate;
  const tensor = new ort.Tensor("float32", data, [batch, 1, imageSize, imageSize]);
  const output = await session.run({ [inputName]: tensor });
  const flat = output[outputName].data;
  const dim = flat.length / batch;
  const embeddings = { data: flat, perCandidate, dim };

  return candidates.map((candidate, index) => {
    const scored = scoreCandidate(candidate, embeddings, runtime, index, dictByKey);
    return toRecognition(candidate, scored, settings);
  });
}

// Test seams: reset the cached runtime, or inject a prebuilt one so the engine
// can be exercised under Node (where /-relative fetch and BASE_URL are absent).
export function __resetSiameseRuntime() {
  runtimePromise = null;
}

export function __setSiameseRuntime(runtime) {
  runtimePromise = Promise.resolve(runtime);
}
