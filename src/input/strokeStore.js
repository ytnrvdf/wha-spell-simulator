export function createStrokeStore() {
  let strokes = [];
  let nextId = 1;

  return {
    addStroke(points) {
      const now = performance.now();
      const stroke = {
        id: `s${nextId++}`,
        points: points.map((point) => ({ ...point })),
        startedAt: points[0]?.t ?? now,
        endedAt: points[points.length - 1]?.t ?? now
      };
      strokes = [...strokes, stroke];
      return stroke;
    },

    undo() {
      const removed = strokes[strokes.length - 1] ?? null;
      strokes = strokes.slice(0, -1);
      return removed;
    },

    clear() {
      strokes = [];
      nextId = 1;
    },

    removeStrokesNearPath(eraserPath, threshold = 22) {
      if (!eraserPath?.length) {
        return [];
      }
      const thresholdSq = threshold * threshold;
      const removed = [];
      strokes = strokes.filter((stroke) => {
        for (const ep of eraserPath) {
          for (const sp of stroke.points) {
            const dx = ep.x - sp.x;
            const dy = ep.y - sp.y;
            if (dx * dx + dy * dy <= thresholdSq) {
              removed.push(stroke);
              return false;
            }
          }
        }
        return true;
      });
      return removed;
    },

    scale(scaleX, scaleY) {
      strokes = strokes.map((stroke) => ({
        ...stroke,
        points: stroke.points.map((point) => ({
          ...point,
          x: point.x * scaleX,
          y: point.y * scaleY
        }))
      }));
    },

    getStrokes() {
      return strokes.map((stroke) => ({
        ...stroke,
        points: stroke.points.map((point) => ({ ...point }))
      }));
    },

    count() {
      return strokes.length;
    }
  };
}
