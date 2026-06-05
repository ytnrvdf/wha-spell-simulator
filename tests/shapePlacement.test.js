import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CONFIG } from "../src/config.js";
import { buildShapeLibrary, defaultTransformForShape } from "../src/input/shapeLibrary.js";
import { bakePlacementToStrokes } from "../src/input/shapeBaker.js";
import { classifyDrawing } from "../src/parser/drawingClassifier.js";
import { compileSpell } from "../src/compiler/spellBuilder.js";

function loadJson(relativePath) {
  return JSON.parse(readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), "utf8"));
}

const dictionary = {
  sigils: loadJson("../src/dictionary/sigils.json"),
  signs: loadJson("../src/dictionary/signs.json"),
  sampleSpells: []
};

const library = buildShapeLibrary(dictionary);
const canvas = { width: 1200, height: 800 };
const center = { x: 600, y: 400 };

function place(id, item, point, overrides = {}) {
  return {
    id,
    kind: item.kind,
    sourceId: item.sourceId,
    baseStrokes: item.baseStrokes,
    transform: { ...defaultTransformForShape(item, point, canvas), ...overrides }
  };
}

function arcStroke(centerPoint, radius, startDeg, endDeg, steps = 16) {
  const points = [];
  for (let index = 0; index <= steps; index += 1) {
    const angle = ((startDeg + ((endDeg - startDeg) * index) / steps) * Math.PI) / 180;
    points.push({ x: centerPoint.x + Math.cos(angle) * radius, y: centerPoint.y + Math.sin(angle) * radius });
  }
  return { id: "close", points, startedAt: 0, endedAt: 0 };
}

function compile(strokes) {
  const pipeline = classifyDrawing({ strokes, previousRing: null, dictionary, config: CONFIG });
  return { pipeline, spellIR: compileSpell({ glyphAST: pipeline.glyphAST, dictionary, config: CONFIG }) };
}

function compileFromPlacements(placements, extraStrokes = []) {
  return compile([...placements.flatMap(bakePlacementToStrokes), ...extraStrokes]);
}

test("a baked ring is detected as a prepared, not yet sealed, boundary", () => {
  const { pipeline } = compileFromPlacements([place("p1", library.ring, center)]);
  assert.equal(pipeline.ring.found, true);
  assert.equal(pipeline.ring.complete, false);
});

test("a baked ring with a baked fire sigil compiles to a prepared fire spell", () => {
  const fire = library.sigils.find((item) => item.sourceId === "fire");
  assert.ok(fire, "fire sigil exists in the library");

  const { spellIR } = compileFromPlacements([
    place("p1", library.ring, center),
    place("p2", fire, center)
  ]);

  assert.equal(spellIR.valid, true);
  assert.equal(spellIR.prepared, true);
  assert.equal(spellIR.active, false);
  assert.equal(spellIR.element, "fire");
});

test("closing the ring gap by hand activates the fire spell", () => {
  const fire = library.sigils.find((item) => item.sourceId === "fire");
  const ringRadius = (800 * 0.62) / 2;
  const closingStroke = arcStroke(center, ringRadius, -30, 30);

  const { spellIR } = compileFromPlacements(
    [place("p1", library.ring, center), place("p2", fire, center)],
    [closingStroke]
  );

  assert.equal(spellIR.active, true);
  assert.equal(spellIR.element, "fire");
});
