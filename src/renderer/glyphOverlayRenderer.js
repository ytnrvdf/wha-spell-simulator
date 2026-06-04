const GLOW_LAYERS = [
  {
    shadowColor: "rgb(110, 185, 212)",
    shadowBlur: ({ pulse, flicker, glowAlpha }) => (24 + pulse * 18 + flicker * 10) * glowAlpha,
    strokeStyle: ({ pulse, glowAlpha }) => `rgba(120, 220, 255, ${(0.18 + pulse * 0.12) * glowAlpha})`,
    lineWidth: ({ pulse, glowAlpha }) => 4 + (8 + pulse * 2) * glowAlpha
  },
  {
    shadowColor: "rgb(117, 150, 161)",
    shadowBlur: ({ pulse, glowAlpha }) => (10 + pulse * 6) * glowAlpha,
    strokeStyle: ({ pulse, glowAlpha }) => `rgba(187, 225, 237, ${(0.88 + pulse * 0.12) * glowAlpha})`,
    lineWidth: ({ pulse, glowAlpha }) => 1.8 + (2 + pulse * 0.6) * glowAlpha
  }
];

const REJECTED_GLOW_LAYERS = [
  {
    shadowColor: "rgba(214, 84, 56, 0.95)",
    shadowBlur: (glowAlpha) => 28 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(202, 70, 46, ${(0.55 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 7 + 9 * glowAlpha
  },
  {
    shadowColor: "rgba(255, 255, 255, 0.9)",
    shadowBlur: (glowAlpha) => 13 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(255, 255, 255, ${(0.72 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 2.2 + 3 * glowAlpha
  }
];

const GOOD_GLOW_LAYERS = [
  {
    shadowColor: "rgba(70, 210, 110, 0.95)",
    shadowBlur: (glowAlpha) => 30 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(54, 196, 96, ${(0.55 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 7 + 9 * glowAlpha
  },
  {
    shadowColor: "rgba(206, 255, 214, 0.9)",
    shadowBlur: (glowAlpha) => 13 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(226, 255, 230, ${(0.72 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 2.2 + 3 * glowAlpha
  }
];

const BAD_GLOW_LAYERS = [
  {
    shadowColor: "rgba(255, 255, 255, 0.95)",
    shadowBlur: (glowAlpha) => 30 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(248, 248, 252, ${(0.6 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 8 + 10 * glowAlpha
  },
  {
    shadowColor: "rgba(0, 0, 0, 0.92)",
    shadowBlur: (glowAlpha) => 14 * glowAlpha,
    strokeStyle: (glowAlpha, pulse) => `rgba(8, 8, 10, ${(0.72 + pulse * 0.2) * glowAlpha})`,
    lineWidth: (glowAlpha) => 2.6 + 3 * glowAlpha
  }
];

const ACTIVE_GLOW_PALETTES = {
  good: GOOD_GLOW_LAYERS,
  bad: BAD_GLOW_LAYERS
};

function hasStrokePoints(stroke) {
  return Boolean(stroke?.points?.length);
}

function traceStrokePath(ctx, stroke) {
  const firstPoint = stroke.points[0];
  ctx.beginPath();
  ctx.moveTo(firstPoint.x, firstPoint.y);
  for (let index = 1; index < stroke.points.length; index += 1) {
    const point = stroke.points[index];
    ctx.lineTo(point.x, point.y);
  }
}

function drawSingleStroke(ctx, stroke, options = {}) {
  if (!hasStrokePoints(stroke)) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = options.color ?? "#241b16";
  ctx.lineWidth = options.lineWidth ?? 4.2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = options.alpha ?? 1;
  traceStrokePath(ctx, stroke);
  ctx.stroke();
  ctx.restore();
}

function strokeLabelAnchor(stroke) {
  if (!hasStrokePoints(stroke)) {
    return null;
  }
  if (stroke.points.length === 1) {
    return stroke.points[0];
  }

  let totalLength = 0;
  for (let index = 1; index < stroke.points.length; index += 1) {
    const previous = stroke.points[index - 1];
    const current = stroke.points[index];
    totalLength += Math.hypot(current.x - previous.x, current.y - previous.y);
  }

  const targetLength = totalLength / 2;
  let walkedLength = 0;
  for (let index = 1; index < stroke.points.length; index += 1) {
    const previous = stroke.points[index - 1];
    const current = stroke.points[index];
    const segmentLength = Math.hypot(current.x - previous.x, current.y - previous.y);
    if (walkedLength + segmentLength >= targetLength) {
      const local = segmentLength <= 0 ? 0 : (targetLength - walkedLength) / segmentLength;
      return {
        x: previous.x + (current.x - previous.x) * local,
        y: previous.y + (current.y - previous.y) * local
      };
    }
    walkedLength += segmentLength;
  }

  return stroke.points[stroke.points.length - 1];
}

function clampLabelPosition(ctx, x, y, width, height) {
  return {
    x: Math.max(4, Math.min(ctx.canvas.width - width - 4, x)),
    y: Math.max(height + 4, Math.min(ctx.canvas.height - 4, y))
  };
}

function drawGlowingStrokeLayer(ctx, stroke, glow, layer) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowBlur = layer.shadowBlur(glow);
  ctx.shadowColor = layer.shadowColor;
  ctx.strokeStyle = layer.strokeStyle(glow);
  ctx.lineWidth = layer.lineWidth(glow);
  traceStrokePath(ctx, stroke);
  ctx.stroke();
  ctx.restore();
}

function drawSingleGlowingStroke(ctx, stroke, timestamp, glowAlpha = 1) {
  if (!hasStrokePoints(stroke)) {
    return;
  }

  const glow = {
    pulse: 0.5 + Math.sin(timestamp * 0.004) * 0.5,
    flicker: Math.random() * 0.08,
    glowAlpha
  };

  for (const layer of GLOW_LAYERS) {
    drawGlowingStrokeLayer(ctx, stroke, glow, layer);
  }
}

export function drawStrokes(ctx, strokes, currentStroke, config) {
  for (const stroke of strokes) {
    drawSingleStroke(ctx, stroke, {
      color: config.renderer.inkColor,
      lineWidth: 4.4,
      alpha: 0.94
    });
  }

  if (currentStroke) {
    drawSingleStroke(ctx, currentStroke, {
      color: config.renderer.inkColor,
      lineWidth: 4.4,
      alpha: 0.72
    });
  }
}

function activeGlowStrokes(activatedStrokeIds, strokes) {
  const glowingStrokes = [];

  for (const stroke of strokes) {
    if (activatedStrokeIds.has(stroke.id)) {
      glowingStrokes.push(stroke);
    }
  }

  return glowingStrokes;
}

function glowAlphaAt(timestamp, activatedAt, duration) {
  const elapsed = timestamp - activatedAt;
  const t = Math.min(1, elapsed / duration);
  return Math.pow(1 - t, 2);
}

export function drawGlowingStrokes(
  ctx,
  activatedAt,
  activatedStrokeIds,
  strokes,
  duration,
  timestamp = performance.now()
) {
  if (!activatedStrokeIds?.size || !activatedAt) {
    return;
  }

  const glowAlpha = glowAlphaAt(timestamp, activatedAt, duration);
  if (glowAlpha <= 0) {
    return;
  }

  for (const stroke of activeGlowStrokes(activatedStrokeIds, strokes)) {
    drawSingleGlowingStroke(ctx, stroke, timestamp, glowAlpha);
  }
}

function rejectedPulseEnvelope(t) {
  if (t <= 0 || t >= 1) {
    return 0;
  }
  const rise = Math.min(1, t / 0.18);
  const fall = Math.pow(1 - Math.max(0, (t - 0.18) / 0.82), 1.7);
  return rise * fall;
}

// Source-over (not additive) so dark and saturated hues stay visible on the light paper.
function drawPaletteGlowLayer(ctx, stroke, glowAlpha, pulse, layer) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowBlur = layer.shadowBlur(glowAlpha);
  ctx.shadowColor = layer.shadowColor;
  ctx.strokeStyle = layer.strokeStyle(glowAlpha, pulse);
  ctx.lineWidth = layer.lineWidth(glowAlpha);
  traceStrokePath(ctx, stroke);
  ctx.stroke();
  ctx.restore();
}

function drawPaletteGlowStroke(ctx, stroke, glowAlpha, pulse, layers) {
  if (!hasStrokePoints(stroke)) {
    return;
  }
  for (const layer of layers) {
    drawPaletteGlowLayer(ctx, stroke, glowAlpha, pulse, layer);
  }
}

export function drawActiveGlyphGlow(
  ctx,
  activatedAt,
  activatedStrokeIds,
  strokes,
  duration,
  grade,
  timestamp = performance.now()
) {
  if (!activatedStrokeIds?.size || !activatedAt) {
    return;
  }

  const glowAlpha = glowAlphaAt(timestamp, activatedAt, duration);
  if (glowAlpha <= 0) {
    return;
  }

  const layers = ACTIVE_GLOW_PALETTES[grade] ?? GOOD_GLOW_LAYERS;
  const pulse = 0.5 + Math.sin(timestamp * 0.006) * 0.5;
  for (const stroke of activeGlowStrokes(activatedStrokeIds, strokes)) {
    drawPaletteGlowStroke(ctx, stroke, glowAlpha, pulse, layers);
  }
}

export function drawRejectedRingGlow(
  ctx,
  rejectedAt,
  ringStrokeIds,
  strokes,
  duration,
  timestamp = performance.now()
) {
  if (!ringStrokeIds?.size || !rejectedAt) {
    return;
  }

  const glowAlpha = rejectedPulseEnvelope((timestamp - rejectedAt) / duration);
  if (glowAlpha <= 0) {
    return;
  }

  const pulse = 0.5 + Math.sin(timestamp * 0.012) * 0.5;
  for (const stroke of activeGlowStrokes(ringStrokeIds, strokes)) {
    drawPaletteGlowStroke(ctx, stroke, glowAlpha, pulse, REJECTED_GLOW_LAYERS);
  }
}

export function drawRingDebug(ctx, ring) {
  if (!ring?.found) {
    return;
  }

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = ring.complete ? "rgba(184, 69, 49, 0.72)" : "rgba(31, 111, 115, 0.72)";
  ctx.setLineDash(ring.complete ? [] : [10, 10]);
  ctx.beginPath();
  ctx.arc(ring.center.x, ring.center.y, ring.radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(36, 27, 22, 0.62)";
  ctx.beginPath();
  ctx.arc(ring.center.x, ring.center.y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export function drawStrokeIdDebug(ctx, strokes) {
  ctx.save();
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;

  for (const stroke of strokes ?? []) {
    const anchor = strokeLabelAnchor(stroke);
    if (!anchor || !stroke.id) {
      continue;
    }

    const label = stroke.id;
    const paddingX = 5;
    const paddingY = 3;
    const textMetrics = ctx.measureText(label);
    const boxWidth = Math.ceil(textMetrics.width + paddingX * 2);
    const boxHeight = 18;
    const position = clampLabelPosition(ctx, anchor.x + 7, anchor.y - 9, boxWidth, boxHeight);

    ctx.fillStyle = "rgba(255, 251, 233, 0.88)";
    ctx.strokeStyle = "rgba(36, 27, 22, 0.34)";
    ctx.beginPath();
    ctx.roundRect(position.x, position.y - boxHeight / 2, boxWidth, boxHeight, 5);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "rgba(36, 27, 22, 0.86)";
    ctx.fillText(label, position.x + paddingX, position.y + paddingY - 2);
  }

  ctx.restore();
}

export function drawCandidateDebug(ctx, candidates, recognitions) {
  const byCandidate = new Map((recognitions ?? []).map((recognition) => [recognition.candidateId, recognition]));

  ctx.save();
  ctx.lineWidth = 1.5;
  for (const candidate of candidates ?? []) {
    const recognition = byCandidate.get(candidate.candidateId);
    const accepted = recognition?.recognized;
    ctx.strokeStyle = accepted ? "rgba(31, 111, 115, 0.82)" : "rgba(184, 69, 49, 0.74)";
    ctx.fillStyle = accepted ? "rgba(31, 111, 115, 0.92)" : "rgba(184, 69, 49, 0.92)";
    ctx.strokeRect(candidate.bounds.minX, candidate.bounds.minY, candidate.bounds.width, candidate.bounds.height);
    const label = accepted
      ? `${recognition.id} ${Math.round(recognition.confidence * 100)}`
      : `${candidate.candidateId}`;
    ctx.fillText(label, candidate.bounds.minX, Math.max(12, candidate.bounds.minY - 5));
  }
  ctx.restore();
}
