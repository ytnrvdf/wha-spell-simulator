import {
  drawCandidateDebug,
  drawRingDebug,
  drawStrokeIdDebug,
  drawStrokes,
  drawActiveGlyphGlow,
  drawRejectedRingGlow
} from "./glyphOverlayRenderer.js";
import { drawGuides, drawPaper } from "./paperRenderer.js";
import { SpellEffectRenderer } from "./spellEffectRenderer.js";

const REJECTED_RING_GLOW_MS = 820;

const GOOD_MAGIC_QUALITY = 0.6;
const GOOD_MAGIC_STABILITY = 0.5;

function magicGrade(quality, stability) {
  return (quality ?? 0) >= GOOD_MAGIC_QUALITY && (stability ?? 0) >= GOOD_MAGIC_STABILITY ? "good" : "bad";
}

function getActivatedStrokeIds(pipeline) {
  if (!pipeline) {
    return new Set();
  }

  const { ring, primarySigil, signs } = pipeline.glyphAST;
  return new Set([
    ...(ring?.strokeIds ?? []),
    ...(primarySigil?.strokeIds ?? []),
    ...((signs ?? []).flatMap((sign) => sign.strokeIds ?? []))
  ]);
}

export class CanvasRenderer {
  constructor({ glyphCanvas, effectCanvas, config }) {
    this.glyphCanvas = glyphCanvas;
    this.glyphCtx = glyphCanvas.getContext("2d");
    this.effectRenderer = new SpellEffectRenderer(effectCanvas, config);
    this.config = config;
  }

  renderGlyph({ strokes, currentStroke, pipeline, showGuides, showDebug }) {
    const width = this.glyphCanvas.width;
    const height = this.glyphCanvas.height;
    drawPaper(this.glyphCtx, width, height);

    if (showGuides) {
      drawGuides(this.glyphCtx, pipeline?.ring, width, height, this.config);
    }

    drawStrokes(this.glyphCtx, strokes, currentStroke, this.config);

    if (showGuides && pipeline?.ring?.found) {
      drawRingDebug(this.glyphCtx, pipeline.ring);
    }

    if (showDebug) {
      drawCandidateDebug(this.glyphCtx, pipeline?.candidates, pipeline?.recognitions);
      drawStrokeIdDebug(this.glyphCtx, strokes);
    }
  }

  renderActivatedGlyph({ activatedAt, duration, quality, stability, strokes, pipeline, timestamp }) {
    const activatedStrokeIds = getActivatedStrokeIds(pipeline);
    const glowDuration = Math.max(250, duration * 1000);
    drawActiveGlyphGlow(
      this.glyphCtx,
      activatedAt,
      activatedStrokeIds,
      strokes,
      glowDuration,
      magicGrade(quality, stability),
      timestamp
    );
  }

  renderRejectedRing({ rejectedAt, strokes, pipeline, timestamp }) {
    const ringStrokeIds = new Set(pipeline?.ring?.strokeIds ?? []);
    drawRejectedRingGlow(this.glyphCtx, rejectedAt, ringStrokeIds, strokes, REJECTED_RING_GLOW_MS, timestamp);
  }

  renderEffect({ spellIR, ring, timestamp, showGuides }) {
    this.effectRenderer.render(spellIR, ring, timestamp, { showGuides });
  }
}
