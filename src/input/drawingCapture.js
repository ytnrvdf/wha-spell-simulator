import { canvasPointFromEvent, shouldKeepPoint } from "./pointerNormalizer.js";
import { pathLength } from "../utils/geometry.js";

export class DrawingCapture {
  constructor(canvas, strokeStore, config, callbacks = {}) {
    this.canvas = canvas;
    this.strokeStore = strokeStore;
    this.config = config;
    this.callbacks = callbacks;
    this.currentPoints = [];
    this.pointerId = null;
    this.enabled = false;
    this.locked = false;
    this.tool = "pen";

    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
  }

  setLocked(locked) {
    this.locked = locked;
    if (locked) {
      this.clearPreview();
    }
  }

  enable() {
    if (this.enabled) {
      return;
    }
    this.canvas.addEventListener("pointerdown", this.handlePointerDown);
    this.canvas.addEventListener("pointermove", this.handlePointerMove);
    this.canvas.addEventListener("pointerup", this.handlePointerUp);
    this.canvas.addEventListener("pointercancel", this.handlePointerUp);
    this.enabled = true;
  }

  disable() {
    this.canvas.removeEventListener("pointerdown", this.handlePointerDown);
    this.canvas.removeEventListener("pointermove", this.handlePointerMove);
    this.canvas.removeEventListener("pointerup", this.handlePointerUp);
    this.canvas.removeEventListener("pointercancel", this.handlePointerUp);
    this.enabled = false;
  }

  getCurrentStroke() {
    if (this.currentPoints.length === 0) {
      return null;
    }
    return {
      id: "preview",
      points: this.currentPoints.map((point) => ({ ...point }))
    };
  }

  clearPreview() {
    this.currentPoints = [];
    this.pointerId = null;
  }

  handlePointerDown(event) {
    if (this.locked) {
      event.preventDefault();
      return;
    }
    if (event.button !== undefined && event.button !== 0) {
      return;
    }
    event.preventDefault();
    this.pointerId = event.pointerId;
    this.canvas.setPointerCapture?.(event.pointerId);
    this.currentPoints = [canvasPointFromEvent(event, this.canvas)];
    this.callbacks.onPreview?.(this.getCurrentStroke());
  }

  handlePointerMove(event) {
    if (this.locked) {
      event.preventDefault();
      return;
    }
    if (this.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const point = canvasPointFromEvent(event, this.canvas);
    if (shouldKeepPoint(this.currentPoints, point, this.config.input.minPointDistance)) {
      this.currentPoints.push(point);
      this.callbacks.onPreview?.(this.getCurrentStroke());
    }
  }

  handlePointerUp(event) {
    if (this.locked) {
      event.preventDefault();
      this.clearPreview();
      return;
    }
    if (this.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    this.canvas.releasePointerCapture?.(event.pointerId);

    const points = this.currentPoints;
    this.clearPreview();

    if (this.tool === "eraser") {
      if (points.length >= 1) {
        this.strokeStore.removeStrokesNearPath(points, this.config.input.eraserThreshold);
        this.callbacks.onCommit?.();
      }
      this.callbacks.onPreview?.(null);
      return;
    }

    if (points.length >= 2 && pathLength(points) >= this.config.input.minStrokeLength) {
      this.strokeStore.addStroke(points);
      this.callbacks.onCommit?.();
      return;
    }

    this.callbacks.onPreview?.(null);
  }
}
