import assert from "node:assert/strict";
import test from "node:test";

import { CONFIG } from "../src/config.js";
import { classifyDrawing } from "../src/parser/drawingClassifier.js";
import { detectRing } from "../src/parser/ringDetector.js";
import { degreesToRadians } from "../src/utils/geometry.js";

function arcStroke(id, centerX, centerY, radius, startDeg, endDeg, steps) {
  const points = [];

  for (let index = 0; index <= steps; index += 1) {
    const deg = startDeg + (endDeg - startDeg) * (index / steps);
    const radians = degreesToRadians(deg);
    points.push({
      x: centerX + Math.cos(radians) * radius,
      y: centerY + Math.sin(radians) * radius
    });
  }

  return { id, points };
}

function openRingStroke(id = "s1") {
  return arcStroke(id, 400, 300, 180, 25, 335, 160);
}

function closingStroke(id = "s2") {
  return arcStroke(id, 400, 300, 180, 335, 385, 32);
}

function outsideStroke(id = "s2") {
  return {
    id,
    points: [
      { x: 900, y: 130 },
      { x: 940, y: 150 },
      { x: 930, y: 190 }
    ]
  };
}

function openRingAt(id, centerX, centerY, radius = 120) {
  return arcStroke(id, centerX, centerY, radius, 25, 335, 128);
}

function closingStrokeAt(id, centerX, centerY, radius = 120) {
  return arcStroke(id, centerX, centerY, radius, 335, 385, 28);
}

test("detects a sealed ring without outside strokes", () => {
  const openRing = openRingStroke();
  const closing = closingStroke();
  const prepared = detectRing([openRing], null, CONFIG);
  const sealed = detectRing([openRing, closing], prepared, CONFIG);

  assert.equal(prepared.found, true);
  assert.equal(prepared.complete, false);
  assert.equal(sealed.complete, true);
  assert.equal(sealed.activationEvent, true);
  assert.deepEqual(sealed.strokeIds, ["s1", "s2"]);
  assert.equal(Object.hasOwn(sealed, "topology"), false);
  assert.equal(Object.hasOwn(sealed, "coverageBinCount"), false);
});

test("keeps a short closing stroke in sealed ring ids", () => {
  const openRing = arcStroke("s1", 400, 300, 180, 5, 355, 180);
  const closing = arcStroke("s2", 400, 300, 180, 355, 365, 8);
  const prepared = detectRing([openRing], null, CONFIG);
  const sealed = detectRing([openRing, closing], prepared, CONFIG);

  assert.equal(prepared.found, true);
  assert.equal(prepared.complete, false);
  assert.equal(sealed.complete, true);
  assert.equal(sealed.activationEvent, true);
  assert.deepEqual(sealed.strokeIds, ["s1", "s2"]);
});

test("reports multiple open rings as unsupported", () => {
  const firstRing = openRingAt("s1", 260, 300);
  const secondRing = openRingAt("s2", 620, 300);
  const detected = detectRing([firstRing, secondRing], null, CONFIG);

  assert.equal(detected.found, true);
  assert.equal(detected.complete, false);
  assert.equal(detected.activationEvent, false);
  assert.equal(detected.unsupportedMultipleRings.length, 1);
});

test("does not activate when closing one of multiple rings", () => {
  const firstRing = openRingAt("s1", 260, 300);
  const secondRing = openRingAt("s2", 620, 300);
  const firstClosingStroke = closingStrokeAt("s3", 260, 300);
  const prepared = detectRing([firstRing, secondRing], null, CONFIG);
  const sealed = detectRing([firstRing, secondRing, firstClosingStroke], prepared, CONFIG);

  assert.equal(sealed.found, true);
  assert.equal(sealed.activationEvent, false);
  assert.equal(sealed.unsupportedMultipleRings.length, 1);
});

test("ignores outside strokes when sealing a prepared ring", () => {
  const openRing = openRingStroke("s1");
  const outside = outsideStroke("s2");
  const closing = closingStroke("s3");
  const prepared = detectRing([openRing], null, CONFIG);
  const preparedWithOutsideMark = detectRing([openRing, outside], prepared, CONFIG);
  const sealed = detectRing([openRing, outside, closing], preparedWithOutsideMark, CONFIG);

  assert.equal(preparedWithOutsideMark.found, true);
  assert.equal(preparedWithOutsideMark.complete, false);
  assert.equal(sealed.complete, true);
  assert.equal(sealed.activationEvent, true);
  assert.deepEqual(sealed.strokeIds, ["s1", "s3"]);
});

test("ignores outside strokes when closed ring is evaluated without prior state", () => {
  const openRing = openRingStroke("s1");
  const outside = outsideStroke("s2");
  const closing = closingStroke("s3");
  const sealed = detectRing([openRing, outside, closing], null, CONFIG);

  assert.equal(sealed.complete, true);
  assert.equal(sealed.activationEvent, false);
  assert.deepEqual(sealed.strokeIds, ["s1", "s3"]);
});

test("classifies symbols inside a detected ring without crashing", async () => {
  const openRing = openRingStroke("s1");
  const closing = closingStroke("s2");
  const columnStem = {
    id: "s3",
    points: [
      { x: 400, y: 285 },
      { x: 400, y: 365 }
    ]
  };
  const columnBase = {
    id: "s4",
    points: [
      { x: 360, y: 365 },
      { x: 440, y: 365 }
    ]
  };
  const dictionary = {
    sigils: [],
    signs: [
      {
        id: "column",
        displayName: "Column",
        allowedLayers: ["center", "middle", "outer"],
        semantic: {
          manifestation: "column",
          directionMode: "inward"
        },
        strokeTemplate: {
          sourceAspectRatio: 1,
          strokes: [
            [
              { x: 0.5, y: 0.12 },
              { x: 0.5, y: 0.8 }
            ],
            [
              { x: 0.18, y: 0.8 },
              { x: 0.82, y: 0.8 }
            ]
          ]
        }
      }
    ]
  };

  const result = await classifyDrawing({
    strokes: [openRing, closing, columnStem, columnBase],
    previousRing: null,
    dictionary,
    config: CONFIG
  });

  assert.equal(result.ring.complete, true);
  assert.ok(result.candidates.length >= 1);
  assert.equal(result.recognitions[0].id, "column");
  assert.ok(result.recognitions[0].diagnostics);
  assert.equal(result.glyphAST.signs[0].id, "column");
  assert.equal(Object.hasOwn(result.glyphAST.signs[0], "diagnostics"), false);
});
