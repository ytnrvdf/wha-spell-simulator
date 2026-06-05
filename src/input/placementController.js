import { canvasPointFromEvent } from "./pointerNormalizer.js";
import { distance } from "../utils/geometry.js";
import {
  clampShapeSize,
  hitTestHandles,
  hitTestPlacement,
  rotationDegToPoint,
  toLocalPoint
} from "./shapeBaker.js";

// Drives shape placement and transform handles on the glyph canvas while arrange mode
// is active. Freehand drawing capture is locked during this time, so the two pointer
// handlers never fight over the same gesture.
export class PlacementController {
  constructor(canvas, store, api = {}) {
    this.canvas = canvas;
    this.store = store;
    this.api = api;
    this.active = false;
    this.dragOp = null;

    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
  }

  enable() {
    this.canvas.addEventListener("pointerdown", this.handlePointerDown);
    this.canvas.addEventListener("pointermove", this.handlePointerMove);
    this.canvas.addEventListener("pointerup", this.handlePointerUp);
    this.canvas.addEventListener("pointercancel", this.handlePointerUp);
  }

  setActive(active) {
    this.active = active;
    if (!active) {
      this.dragOp = null;
    }
  }

  handlePointerDown(event) {
    if (!this.active || (event.button !== undefined && event.button !== 0)) {
      return;
    }
    event.preventDefault();
    const point = canvasPointFromEvent(event, this.canvas);

    const selected = this.api.getSelectedId?.() ? this.store.get(this.api.getSelectedId()) : null;
    if (selected) {
      const handle = hitTestHandles(selected, point);
      if (handle) {
        this.beginHandleDrag(event, selected, handle);
        return;
      }
    }

    const hit = [...this.store.getPlacements()].reverse().find((placement) => hitTestPlacement(placement, point));
    if (hit) {
      this.api.setSelectedId?.(hit.id);
      this.beginDrag(event, { type: "move", id: hit.id, last: point });
      return;
    }

    if (this.api.getArmedShape?.()) {
      const placed = this.api.placeShape?.(point);
      if (placed) {
        this.beginDrag(event, { type: "move", id: placed, last: point });
      }
      return;
    }

    this.api.setSelectedId?.(null);
  }

  beginHandleDrag(event, placement, handle) {
    const point = canvasPointFromEvent(event, this.canvas);
    if (handle.type === "scale") {
      const center = { x: placement.transform.cx, y: placement.transform.cy };
      this.beginDrag(event, {
        type: "scale",
        id: placement.id,
        startDistance: Math.max(1, distance(center, point)),
        startScaleX: placement.transform.scaleX,
        startScaleY: placement.transform.scaleY
      });
    } else {
      this.beginDrag(event, { type: handle.type, id: placement.id });
    }
  }

  beginDrag(event, dragOp) {
    this.dragOp = dragOp;
    this.canvas.setPointerCapture?.(event.pointerId);
    this.pointerId = event.pointerId;
  }

  handlePointerMove(event) {
    if (!this.active || !this.dragOp || this.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const point = canvasPointFromEvent(event, this.canvas);
    const placement = this.store.get(this.dragOp.id);
    if (!placement) {
      return;
    }

    const transform = placement.transform;
    if (this.dragOp.type === "move") {
      this.store.update(this.dragOp.id, {
        cx: transform.cx + (point.x - this.dragOp.last.x),
        cy: transform.cy + (point.y - this.dragOp.last.y)
      });
      this.dragOp.last = point;
    } else if (this.dragOp.type === "scale") {
      const center = { x: transform.cx, y: transform.cy };
      const factor = distance(center, point) / this.dragOp.startDistance;
      this.store.update(this.dragOp.id, {
        scaleX: clampShapeSize(this.dragOp.startScaleX * factor),
        scaleY: clampShapeSize(this.dragOp.startScaleY * factor)
      });
    } else if (this.dragOp.type === "elongate-x") {
      this.store.update(this.dragOp.id, { scaleX: clampShapeSize(Math.abs(toLocalPoint(transform, point).x) * 2) });
    } else if (this.dragOp.type === "elongate-y") {
      this.store.update(this.dragOp.id, { scaleY: clampShapeSize(Math.abs(toLocalPoint(transform, point).y) * 2) });
    } else if (this.dragOp.type === "rotate") {
      this.store.update(this.dragOp.id, { rotationDeg: rotationDegToPoint(transform, point) });
    }

    this.api.onChange?.();
  }

  handlePointerUp(event) {
    if (!this.dragOp || this.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    this.canvas.releasePointerCapture?.(event.pointerId);
    this.dragOp = null;
    this.pointerId = null;
    this.api.onChange?.();
  }
}
