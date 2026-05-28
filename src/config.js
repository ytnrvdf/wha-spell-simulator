export const CONFIG = {
  // String version tag shown in diagnostics.
  appVersion: "0.1.0-poc",
  input: {
    // Canvas pixels; lower keeps more pointer samples, higher smooths noisy input.
    minPointDistance: 1.4,

    // Canvas pixels; strokes shorter than this are ignored.
    minStrokeLength: 7,

    // Integer passes, usually 0..3; set to 0 to use raw points.
    // Higher softens hand jitter but can distort sharp symbols.
    smoothingPasses: 1
  },
  ring: {
    // Canvas pixels; smallest ring radius accepted as spell paper boundary.
    minRadius: 70
  },
  // spell ring layer (three layers); sigils are usually in the center layer
  layers: {
    // 0..1 normalized radius; center symbol area.
    centerMax: 0.32,

    // 0..1 normalized radius; middle symbol area.
    middleMax: 0.66,

    // 0..1 normalized radius; outer sign area.
    outerMax: 0.94,

    // 0..1 normalized radius; strokes beyond this are outside the spell boundary.
    boundaryMax: 1.06,

    // 0..1 normalized radius; marks symbols close to layer edges as ambiguous.
    boundaryTolerance: 0.055
  },
  recognition: {
    // Symbol recognizer engine: "template" (built-in hand-tuned stroke matcher,
    // synchronous) or "siamese" (learned ONNX embedding, asynchronous). The
    // siamese engine reuses the exact same stroke grouping; it only replaces
    // the per-candidate classification step. See siamese/README.md.
    engine: "template",

    // 0..1 final recognizer score floor (shared by both engines).
    minConfidence: 0.48,

    // Settings for the "siamese" engine (ignored by the template engine).
    siamese: {
      // The model + prototypes ship as bundled assets in
      // src/parser/siamese-assets/ and load automatically. Set these only to
      // override with custom hosting (absolute URL).
      modelUrl: null,
      prototypesUrl: null,
      // onnxruntime-web WASM binaries; CDN by default to keep the bundle light.
      wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/",
      // Rotations (degrees) tested per candidate; best cosine is kept. Covers
      // ring-relative sign orientation without the template engine's rotation
      // machinery.
      rotationsDeg: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      // Softmax temperature over per-glyph cosine similarities.
      temperature: 0.08
    }
  },
  compiler: {
    // 0..1 confidence; minimum primary sigil confidence before a spell is valid.
    minimumPrimarySigilConfidence: 0.62,

    // Count; unknown symbols above this increase instability.
    maxUnknownsBeforeInstability: 4
  },
  renderer: {
    // CSS color; drawn ink color.
    inkColor: "#241b16",

    // CSS color; guide line color.
    guideColor: "rgba(92, 74, 54, 0.28)",

    // Count; default effect particle budget before element-specific scaling.
    particleBaseCount: 130,

    // Count; hard cap for effect particles.
    particleCap: 360,

    effectSize: {
      // 0..1+ multiplier; baseline effect size even for compact sigils.
      baseScale: 1.28,

      // Multiplier added from primary sigil sizeNorm; larger sigils produce larger effects.
      sigilSizeInfluence: 2.1,

      // 0..1+ multiplier; lower visual effect scale clamp.
      minScale: 1,

      // 0..1+ multiplier; upper visual effect scale clamp.
      maxScale: 2.35
    }
  }
};
