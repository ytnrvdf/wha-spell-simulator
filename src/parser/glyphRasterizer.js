// Rasterizes normalized strokes into a grayscale glyph image for the siamese
// recognizer. This is a bit-for-bit port of siamese/dict_raster.py: the same
// unit-box fit, supersampled disk stamping, and box downsample. The Python
// prototypes and the browser candidates MUST go through identical pixels, so
// any change here has to be mirrored in dict_raster.py (the parity test guards
// this). Pure integer / deterministic float math, no canvas, no library AA.

export const PADDING_RATIO = 0.28;
export const STROKE_WIDTH_RATIO = 0.045;
export const SUPERSAMPLE = 4;

// floor(v + 0.5): matches Python dict_raster._round_half_up exactly.
function roundHalfUp(value) {
  return Math.floor(value + 0.5);
}

function fitUnitBox(strokes) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const stroke of strokes) {
    for (const point of stroke) {
      if (point.x < minX) minX = point.x;
      if (point.x > maxX) maxX = point.x;
      if (point.y < minY) minY = point.y;
      if (point.y > maxY) maxY = point.y;
    }
  }
  const width = Math.max(1e-6, maxX - minX);
  const height = Math.max(1e-6, maxY - minY);
  const scale = 1.0 / Math.max(width, height);
  const offsetX = (1.0 - width * scale) / 2.0;
  const offsetY = (1.0 - height * scale) / 2.0;
  return strokes.map((stroke) =>
    stroke.map((point) => ({
      x: (point.x - minX) * scale + offsetX,
      y: (point.y - minY) * scale + offsetY
    }))
  );
}

function rotateStrokes(strokes, degrees) {
  if (!degrees) {
    return strokes;
  }
  const radians = (degrees * Math.PI) / 180.0;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return strokes.map((stroke) =>
    stroke.map((point) => {
      const x = point.x - 0.5;
      const y = point.y - 0.5;
      return { x: x * cos - y * sin + 0.5, y: x * sin + y * cos + 0.5 };
    })
  );
}

function stampDisk(pixels, cx, cy, radius, size) {
  const centerX = roundHalfUp(cx);
  const centerY = roundHalfUp(cy);
  const radiusSq = radius * radius;
  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    const py = centerY + offsetY;
    if (py < 0 || py >= size) {
      continue;
    }
    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      if (offsetX * offsetX + offsetY * offsetY > radiusSq) {
        continue;
      }
      const px = centerX + offsetX;
      if (px >= 0 && px < size) {
        pixels[py * size + px] = 255;
      }
    }
  }
}

function boxDownsample(hi, hiSize, size, factor) {
  const out = new Uint8Array(size * size);
  const area = factor * factor;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let sum = 0;
      for (let dy = 0; dy < factor; dy += 1) {
        const row = (y * factor + dy) * hiSize + x * factor;
        for (let dx = 0; dx < factor; dx += 1) {
          sum += hi[row + dx];
        }
      }
      out[y * size + x] = Math.floor(sum / area);
    }
  }
  return out;
}

// strokes: array of point arrays ([{x, y}, ...]) in any coordinate space (they
// are fit to the unit box internally). Returns a Uint8Array[size*size], white
// ink (255) on black (0), row-major.
export function rasterizeStrokes(
  strokes,
  {
    size = 56,
    rotationDeg = 0,
    strokeWidthRatio = STROKE_WIDTH_RATIO,
    paddingRatio = PADDING_RATIO,
    supersample = SUPERSAMPLE
  } = {}
) {
  let fitted = fitUnitBox(strokes);
  if (rotationDeg) {
    fitted = fitUnitBox(rotateStrokes(fitted, rotationDeg));
  }

  const hiSize = size * supersample;
  const hi = new Uint8Array(hiSize * hiSize);
  const inner = hiSize * (1.0 - 2.0 * paddingRatio);
  const origin = hiSize * paddingRatio;
  const radius = Math.max(1, roundHalfUp((hiSize * strokeWidthRatio) / 2.0));

  for (const stroke of fitted) {
    const points = stroke.map((point) => ({
      x: origin + point.x * inner,
      y: origin + point.y * inner
    }));
    if (points.length === 1) {
      stampDisk(hi, points[0].x, points[0].y, radius, hiSize);
      continue;
    }
    for (let index = 1; index < points.length; index += 1) {
      const a = points[index - 1];
      const b = points[index];
      const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y)));
      for (let step = 0; step <= steps; step += 1) {
        const t = step / steps;
        stampDisk(hi, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, radius, hiSize);
      }
    }
  }

  return boxDownsample(hi, hiSize, size, supersample);
}

// Convert a raster (Uint8Array) into the model input tensor data: Float32 in
// [0, 1], shape [1, 1, size, size] flattened row-major.
export function rasterToModelInput(raster) {
  const input = new Float32Array(raster.length);
  for (let index = 0; index < raster.length; index += 1) {
    input[index] = raster[index] / 255.0;
  }
  return input;
}
