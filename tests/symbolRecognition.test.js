import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { CONFIG } from "../src/config.js";
import {
  boundsForStrokes,
  centerOfBounds,
  degreesToRadians,
  endpointClosedness,
  pathLength
} from "../src/utils/geometry.js";
import { recognizeCandidates } from "../src/parser/symbolRecognizer.js";

const lineTemplate = {
  sourceAspectRatio: 0.05,
  strokes: [
    [
      { x: 0.5, y: 0 },
      { x: 0.5, y: 1 }
    ]
  ]
};

const dictionary = {
  sigils: [
    {
      id: "line-sign",
      displayName: "Line Sign",
      allowedLayers: ["middle"],
      recognitionRotationInvariant: true,
      strokeTemplate: lineTemplate
    }
  ],
  signs: []
};

const realDictionary = {
  sigils: JSON.parse(readFileSync(new URL("../src/dictionary/sigils.json", import.meta.url), "utf8")),
  signs: JSON.parse(readFileSync(new URL("../src/dictionary/signs.json", import.meta.url), "utf8"))
};

function stroke(id, points) {
  return { id, points };
}

function candidate(strokes) {
  const bounds = boundsForStrokes(strokes);
  const center = centerOfBounds(bounds);
  const length = strokes.reduce((sum, item) => sum + pathLength(item.points), 0);
  const size = Math.max(bounds.width, bounds.height, 1);
  const compactPerimeter = Math.max(1, (bounds.width + bounds.height) * 2);
  const overdrawAmount = Math.max(0, Math.min(1, length / compactPerimeter - 0.72));

  return {
    candidateId: "c1",
    strokeIds: strokes.map((item) => item.id),
    rawStrokeCount: strokes.length,
    cleanedStrokeCount: strokes.length,
    bounds,
    center,
    radiusNorm: 0.5,
    angleDeg: 0,
    layer: "middle",
    nearBoundary: false,
    sizeNorm: size / 300,
    lengthNorm: length / 900,
    orientationDeg: 90,
    directedOrientationDeg: 90,
    radialFacing: "unclear",
    closedness: endpointClosedness(strokes, size),
    overdrawAmount,
    neatness: Math.max(0, Math.min(1, 0.92 - overdrawAmount * 0.28 - Math.max(0, strokes.length - 4) * 0.035)),
    strokes
  };
}

function candidateFromTemplate(entry, duplicateOffset, duplicateStrokeIndex = null) {
  const strokes = [];
  let nextId = 1;

  for (let index = 0; index < entry.strokeTemplate.strokes.length; index += 1) {
    const templateStroke = entry.strokeTemplate.strokes[index];
    const points = templateStroke.map((point) => ({
      x: 100 + point.x * 220,
      y: 100 + point.y * 220
    }));
    strokes.push(stroke(`s${nextId++}`, points));
    if (duplicateStrokeIndex !== null && duplicateStrokeIndex !== index) {
      continue;
    }
    strokes.push(
      stroke(
        `s${nextId++}`,
        points.map((point) => ({
          x: point.x + duplicateOffset,
          y: point.y + duplicateOffset
        }))
      )
    );
  }

  const result = candidate(strokes);
  return {
    ...result,
    layer: "center",
    radiusNorm: 0.35,
    overdrawAmount: 0.5,
    neatness: 0.65
  };
}

function cleanCandidateFromTemplate(entry, options = {}) {
  const scale = options.scale ?? 180;
  const scaleX = options.scaleX ?? scale;
  const scaleY = options.scaleY ?? scale;
  const origin = options.origin ?? 120;
  const rotationDeg = options.rotationDeg ?? 0;
  const rotate = rotationTransform(rotationDeg);
  const center = {
    x: origin + scaleX * 0.5,
    y: origin + scaleY * 0.5
  };
  const strokes = entry.strokeTemplate.strokes.map((templateStroke, index) =>
    stroke(
      `s${index + 1}`,
      templateStroke.map((point) =>
        rotatePoint(
          {
            x: origin + point.x * scaleX,
            y: origin + point.y * scaleY
          },
          center,
          rotate
        )
      )
    )
  );

  const result = candidate(strokes);
  return {
    ...result,
    layer: options.layer ?? "any",
    radiusNorm: options.radiusNorm ?? 0.5,
    angleDeg: options.angleDeg ?? 270,
    overdrawAmount: 0,
    neatness: 0.92
  };
}

function rotationTransform(degrees) {
  if (!degrees) {
    return null;
  }
  const radians = degreesToRadians(degrees);
  return {
    cos: Math.cos(radians),
    sin: Math.sin(radians)
  };
}

function rotatePoint(point, center, transform) {
  if (!transform) {
    return point;
  }

  const x = point.x - center.x;
  const y = point.y - center.y;
  return {
    x: center.x + x * transform.cos - y * transform.sin,
    y: center.y + x * transform.sin + y * transform.cos
  };
}

test("recognizes overdraw as valid messy instead of rejecting the symbol", () => {
  const messyLine = candidate([
    stroke("s1", [
      { x: 99, y: 0 },
      { x: 99, y: 100 }
    ]),
    stroke("s2", [
      { x: 101, y: 0 },
      { x: 101, y: 100 }
    ]),
    stroke("s3", [
      { x: 100, y: 0 },
      { x: 100, y: 100 }
    ])
  ]);

  const [recognition] = recognizeCandidates([messyLine], dictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(recognition.id, "line-sign");
  assert.equal(recognition.recognitionStatus, "valid_messy");
  assert.equal(recognition.diagnostics.template.unexplainedInkRatio, 0);
});

test("keeps fire classified when one ray is bolded", () => {
  const fire = realDictionary.sigils.find((entry) => entry.id === "fire");
  const fireWithBoldedRay = candidateFromTemplate(fire, 12, 1);

  const [recognition] = recognizeCandidates([fireWithBoldedRay], realDictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(recognition.id, "fire");
  assert.equal(recognition.recognitionStatus, "valid_messy");
  assert.equal(recognition.diagnostics.topMatches[0].id, "fire");
  assert.equal(recognition.diagnostics.topMatches.length, 3);
  assert.equal(Object.hasOwn(recognition, "scoreboard"), false);
});

test("keeps column classified as column instead of a larger sigil", () => {
  const column = realDictionary.signs.find((entry) => entry.id === "column");
  assert.ok(column);

  const drawnColumn = cleanCandidateFromTemplate(column, { layer: "any" });

  const [recognition] = recognizeCandidates([drawnColumn], realDictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(recognition.kind, "sign");
  assert.equal(recognition.id, "column");
  assert.equal(recognition.diagnostics.topMatches[0].id, "column");
});

test("recognizes signs in the ring-relative orientation for their position", () => {
  const column = realDictionary.signs.find((entry) => entry.id === "column");
  assert.ok(column);

  const bottomColumn = cleanCandidateFromTemplate(column, { layer: "outer", angleDeg: 270 });
  const topColumn = cleanCandidateFromTemplate(column, {
    layer: "outer",
    angleDeg: 90,
    rotationDeg: 180
  });
  const rightColumn = cleanCandidateFromTemplate(column, {
    layer: "outer",
    angleDeg: 0,
    rotationDeg: 270
  });
  const upperRightColumn = cleanCandidateFromTemplate(column, {
    layer: "outer",
    angleDeg: 45,
    rotationDeg: 225
  });
  const leftColumn = cleanCandidateFromTemplate(column, {
    layer: "outer",
    angleDeg: 180,
    rotationDeg: 90
  });

  const recognitions = recognizeCandidates(
    [bottomColumn, topColumn, rightColumn, upperRightColumn, leftColumn],
    realDictionary,
    CONFIG
  );

  assert.deepEqual(
    recognitions.map((recognition) => recognition.id),
    ["column", "column", "column", "column", "column"]
  );
  assert.deepEqual(
    recognitions.map((recognition) => recognition.diagnostics.recognitionRotationDeg),
    [0, 180, 90, 135, 270]
  );
});

test("recognizes convergence when the triangle point faces the spell center", () => {
  const convergence = realDictionary.signs.find((entry) => entry.id === "convergence");
  assert.ok(convergence);

  const bottomInwardConvergence = cleanCandidateFromTemplate(convergence, {
    layer: "outer",
    angleDeg: 270,
    rotationDeg: 180
  });
  const bottomOutwardConvergence = cleanCandidateFromTemplate(convergence, {
    layer: "outer",
    angleDeg: 270
  });

  const [inwardRecognition, outwardRecognition] = recognizeCandidates(
    [bottomInwardConvergence, bottomOutwardConvergence],
    realDictionary,
    CONFIG
  );

  assert.equal(inwardRecognition.recognized, true);
  assert.equal(inwardRecognition.kind, "sign");
  assert.equal(inwardRecognition.id, "convergence");
  assert.notEqual(outwardRecognition.id, "convergence");
});

test("rejects a sign drawn in the wrong orientation for its ring position", () => {
  const column = realDictionary.signs.find((entry) => entry.id === "column");
  assert.ok(column);

  const topButUnrotatedColumn = cleanCandidateFromTemplate(column, {
    layer: "outer",
    angleDeg: 90
  });

  const [recognition] = recognizeCandidates([topButUnrotatedColumn], realDictionary, CONFIG);

  assert.notEqual(recognition.id, "column");
});

test("does not recognize a lone line as the column sign", () => {
  const lineOnly = candidate([
    stroke("s1", [
      { x: 100, y: 100 },
      { x: 100, y: 260 }
    ])
  ]);
  const outerLine = {
    ...lineOnly,
    layer: "outer",
    radiusNorm: 0.72
  };

  const [recognition] = recognizeCandidates([outerLine], realDictionary, CONFIG);

  assert.equal(recognition.recognized, false);
  assert.notEqual(recognition.id, "column");
  assert.ok(
    recognition.diagnostics.topMatches.every((match) => match.confidence < CONFIG.recognition.minConfidence)
  );
  assert.equal(recognition.diagnostics.bestGuess.confidence < CONFIG.recognition.minConfidence, true);
});

test("keeps larger and naturally stretched column signs as column", () => {
  const column = realDictionary.signs.find((entry) => entry.id === "column");
  assert.ok(column);

  const largerColumn = cleanCandidateFromTemplate(column, { layer: "outer", scale: 280 });
  const stretchedColumn = cleanCandidateFromTemplate(column, { layer: "outer", scaleX: 180, scaleY: 320 });

  const [largerRecognition, stretchedRecognition] = recognizeCandidates(
    [largerColumn, stretchedColumn],
    realDictionary,
    CONFIG
  );

  assert.equal(largerRecognition.recognized, true);
  assert.equal(largerRecognition.id, "column");
  assert.ok(largerRecognition.sizeNorm > cleanCandidateFromTemplate(column, { layer: "outer", scale: 180 }).sizeNorm);
  assert.equal(stretchedRecognition.recognized, true);
  assert.equal(stretchedRecognition.id, "column");
  assert.ok(stretchedRecognition.shape.elongation > largerRecognition.shape.elongation);
});

test("does not mistake a sideways-stretched column for levitation", () => {
  const column = realDictionary.signs.find((entry) => entry.id === "column");
  assert.ok(column);

  const sidewaysColumn = cleanCandidateFromTemplate(column, { layer: "outer", scaleX: 320, scaleY: 180 });
  const [recognition] = recognizeCandidates([sidewaysColumn], realDictionary, CONFIG);

  assert.notEqual(recognition.id, "levitation");
  assert.notEqual(recognition.diagnostics.topMatches[0].id, "levitation");
});

test("marks unrelated extra ink as contaminated", () => {
  const contaminatedLine = candidate([
    stroke("s1", [
      { x: 99, y: 0 },
      { x: 99, y: 100 }
    ]),
    stroke("s2", [
      { x: 101, y: 0 },
      { x: 101, y: 100 }
    ]),
    stroke("s3", [
      { x: 100, y: 0 },
      { x: 100, y: 100 }
    ]),
    stroke("s4", [
      { x: 140, y: 0 },
      { x: 140, y: 100 }
    ])
  ]);

  const [recognition] = recognizeCandidates([contaminatedLine], dictionary, CONFIG);

  assert.equal(recognition.recognized, false);
  assert.equal(recognition.recognitionStatus, "contaminated");
  assert.equal(recognition.diagnostics.bestGuess.id, "line-sign");
  assert.ok(recognition.diagnostics.template.unexplainedInkRatio > 0.6);
});

test("keeps matcher internals under diagnostics", () => {
  const cleanLine = candidate([
    stroke("s1", [
      { x: 100, y: 0 },
      { x: 100, y: 100 }
    ])
  ]);

  const [recognition] = recognizeCandidates([cleanLine], dictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(Object.hasOwn(recognition, "scoreboard"), false);
  assert.equal(Object.hasOwn(recognition, "recognitionConfidence"), false);
  assert.equal(Object.hasOwn(recognition, "inkScore"), false);
  assert.equal(Object.hasOwn(recognition, "suspectedId"), false);
  assert.ok(recognition.diagnostics.template.inkScore > 0);
  assert.ok(recognition.diagnostics.topMatches.length > 0);
});

test("keeps drawn orientation while using rotation-invariant recognition", () => {
  const horizontalLine = candidate([
    stroke("s1", [
      { x: 0, y: 100 },
      { x: 100, y: 100 }
    ])
  ]);

  const [recognition] = recognizeCandidates([horizontalLine], dictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(recognition.id, "line-sign");
  assert.notEqual(recognition.diagnostics.recognitionRotationDeg, 0);
  assert.equal(recognition.orientationDeg, 90);
});

test("fixed-orientation templates do not rotate during recognition", () => {
  const fixedDictionary = {
    sigils: [
      {
        id: "fixed-line",
        displayName: "Fixed Line",
        allowedLayers: ["middle"],
        recognitionRotationInvariant: false,
        strokeTemplate: lineTemplate
      }
    ],
    signs: []
  };
  const horizontalLine = candidate([
    stroke("s1", [
      { x: 0, y: 100 },
      { x: 100, y: 100 }
    ])
  ]);

  const [recognition] = recognizeCandidates([horizontalLine], fixedDictionary, CONFIG);

  assert.equal(recognition.diagnostics.topMatches[0].id, "fixed-line");
  assert.equal(recognition.diagnostics.topMatches[0].recognitionRotationDeg, 0);
});

test("keeps a bolded fire sigil classified as fire", () => {
  const fire = realDictionary.sigils.find((entry) => entry.id === "fire");
  const boldedFire = candidateFromTemplate(fire, 12);

  const [recognition] = recognizeCandidates([boldedFire], realDictionary, CONFIG);

  assert.equal(recognition.recognized, true);
  assert.equal(recognition.id, "fire");
  assert.equal(recognition.recognitionStatus, "valid_messy");
  assert.equal(recognition.diagnostics.topMatches[0].id, "fire");
});
