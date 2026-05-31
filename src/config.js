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
    smoothingPasses: 1,

    // Canvas pixels; radius around eraser path to remove strokes.
    eraserThreshold: 22
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
    // 0..1 final recognizer score floor.
    minConfidence: 0.48
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
