# Dictionary Authoring

- [Introduction](#introduction)
- [Entry Types](#entry-types)
  - [Sigils](#sigils)
  - [Signs](#signs)
  - [Sample Spells](#sample-spells)
- [Common Properties](#common-properties)
- [`strokeTemplate`](#stroketemplate)
- [Sigil-Only Properties](#sigil-only-properties)
- [Sign-Only Properties](#sign-only-properties)
- [Creating A Template With The Reference Maker](#creating-a-template-with-the-reference-maker)
- [Viewing A Template As An Image](#viewing-a-template-as-an-image)
- [Adding A New Sigil](#adding-a-new-sigil)
- [Adding A New Sign](#adding-a-new-sign)
- [Adding A Sample Spell](#adding-a-sample-spell)
- [Testing Notes](#testing-notes)

## Introduction

This document explains how to add or edit sigils, signs, and sample spells in:

- `src/dictionary/sigils.json`
- `src/dictionary/signs.json`
- `src/dictionary/sample-spells.json`

The sigil and sign dictionaries define both recognition data and spell meaning. Recognition mostly comes from `strokeTemplate`; semantics tell the compiler what the recognized symbol should do. Sample spells are drawing references shown in the Dictionary panel, not compiler grammar.

## Entry Types

### Sigils

Sigils define the spell's primary element or elemental variant. The current compiler selects one primary sigil and emits it as `SpellIR.element`.

The current compiler supports one recognized sigil per spell. If the parser recognizes extra sigils, the spell compiles as unsupported until multi-element behavior is designed.

Current sigil ids:

- `fire`
- `water`
- `wind-directs-air`
- `earth`
- `light`

### Signs

Signs modify how the primary element manifests. They affect `manifestations`, direction, force, focus, spread, range, and lifetime bias.

Current sign ids:

- `column`
- `levitation`
- `convergence`

### Sample Spells

Sample spells are complete seal layouts stored in `src/dictionary/sample-spells.json` and shown in the Dictionary panel. They are reference-only examples for us to copy by eye. They do not load strokes into the canvas, feed recognition, or affect compiler output.

## Common Properties

### `id`

Stable machine-readable identifier. Use lowercase kebab-case.

Example:

```json
"id": "wind-directs-air"
```

Do not rename an id casually; parser output, tests, and known spell fixtures may refer to it.

### `displayName`

Human-readable label shown in the Dictionary panel.

Example:

```json
"displayName": "Wind (Directs Air)"
```

### `allowedLayers`

Where the symbol is expected inside the ring.

Current values:

- `center`
- `middle`
- `outer`

Typical use:

- Sigils: `["center", "middle", "outer"]`
- Signs: `["middle", "outer"]`
- Directional signs often prefer `["outer"]`

Layer matching is part of recognition confidence. A good shape in the wrong layer can still score lower.

### `sourceNotes`

Longer explanatory notes. Currently used mainly by signs and shown under the Signs tab expander.

Example:

```json
"sourceNotes": "Column causes the magic of its glyph to manifest in a column or beam above the glyph."
```

### `recognitionRotationInvariant`

Controls sigil recognition only.

If `true`, the matcher can rotate the drawn sigil candidate to match the template. Current sigils should usually use `false` because players draw them upright on the paper and sigil orientation has no meaning.

Default in code is effectively `true` for template matching.

Use `false` only if a symbol's identity should be recognized only at a fixed orientation.

Signs do not use this field. Sign templates are authored in a canonical bottom-of-ring pose, at `270` degrees, and the parser rotates sign matching from that pose to the candidate's ring position.

## `strokeTemplate`

`strokeTemplate` is the normalized reference drawing used for template matching.

Shape:

```json
"strokeTemplate": {
  "sourceAspectRatio": 1,
  "strokes": [
    [
      { "x": 0.1, "y": 0.2 },
      { "x": 0.2, "y": 0.3 }
    ]
  ]
}
```

Properties:

- `sourceAspectRatio`: width divided by height of the source reference bounds.
- `strokes`: list of strokes.
- Each stroke is a list of normalized points.
- Each point uses `x` and `y` from `0` to `1`.

Important conventions:

- Preserve stroke order when possible.
- Draw cleanly, but do not over-optimize. The user drawing will be imperfect.
- The tool for stroke template maker exports only the `strokeTemplate` object. Paste that object into the dictionary entry.

## Sigil-Only Properties

### `element`

Primary element emitted into `SpellIR.element` by the current single-element compiler.

Current values:

- `fire`
- `water`
- `wind`
- `earth`
- `light`

This field is required for sigils. Missing or unsupported elements compile to invalid `SpellIR` instead of falling back to another effect.

Multiple sigils may share one element when they represent variants.

Even though sigils can be recognized in the center, middle, or outer layer, the current compiler still accepts only one primary sigil in a spell.

### `semantic`

Sigil semantic data modifies base spell parameters.

Shape:

```json
"semantic": {
  "force": 0.12,
  "focus": 0.04,
  "spread": 0.02,
  "range": 0.08,
  "lifetimeBias": 0.08
}
```

Numeric semantic fields are deltas applied by the compiler:

- `force`: power/intensity
- `focus`: concentration/narrowness
- `spread`: width/dispersal
- `range`: reach
- `lifetimeBias`: bias applied to compiled spell lifetime

`lifetimeBias` is not direct seconds. The compiler combines it with glyph quality and neatness, then emits `SpellIR.duration` as the active spell lifetime in seconds.

Suggested range for deltas:

- `-0.20` to `0.20` for normal modifiers
- Up to about `0.35` for a strong defining behavior

Keep values conservative. Signs also modify these parameters, and everything is clamped later.

## Sign-Only Properties

### `semantic.manifestation`

The sign's main behavior. The compiler aggregates signs into `SpellIR.manifestations`, so different sign types can coexist in one spell.

Current values:

- `column`
- `levitation`
- `convergence`

### `semantic.directionMode`

How the sign contributes to spell direction.

Current values:

- `position`: direction comes from the sign's position around the ring.
- `orientation`: direction comes from the drawn stroke orientation.
- `inward`: direction points from the sign toward the ring center.

Use `inward` when a sign placed on one side should push the effect toward the opposite side of the ring. For example, a `column` sign on the left side sends the effect rightward.

### Numeric Sign Semantics

Signs use the same numeric delta fields as sigils:

- `force`
- `focus`
- `spread`
- `range`
- `lifetimeBias`

These are multiplied by sign influence. Sign influence currently depends on recognition confidence, neatness, size, stroke length, layer, and distance from center. Elongated signs also get a small built-in boost to direction weight, force, and focus, so a longer column stem naturally reads as a stronger directional push (as in the anime). Uniformly larger or naturally elongated signs should keep their identity, while sideways distortion or missing simple sign strokes can lower structural confidence.

### Sign Template Orientation

Author sign `strokeTemplate` drawings as if the sign is placed at the bottom of the ring, at `270` degrees. When a player draws the same sign at another ring position, the recognizer rotates the candidate back into this canonical pose before comparing ink.

For example, a sign at the top of the ring, `90` degrees, is compared with a 180 degree rotation back to the bottom reference. A sign at the right side of the ring, `0` degrees, is compared with a 270 degree rotation back to the bottom reference.

If an existing template is authored in the opposite pose, set `recognitionRotationOffsetDeg` on that sign instead of changing the spell semantics. For example, `convergence` uses `180` so the triangle point must face the spell center even though its stored template ink is inverted.

## Creating A Template With The Reference Maker

The tool is currently at:

```txt
/tools/strokeTemplateMaker.html
```

Workflow:

1. Start the local static server.
2. Open the reference maker.
3. Draw one clean sigil or sign reference on the paper.
4. Click `Export`.
5. Copy the exported JSON object.
6. Paste it into the matching dictionary entry as `strokeTemplate`.

## Viewing A Template As An Image

Use the reverse viewer when you want to inspect a `strokeTemplate` from JSON:

```txt
/tools/strokeTemplateViewer.html
```

The viewer accepts either:

- a raw `strokeTemplate` object
- a full dictionary entry containing `strokeTemplate`

Workflow:

1. Paste the JSON into the Template JSON box.
2. Click `Render`.
3. Inspect the reconstructed strokes on the paper preview.
4. Check the metrics panel for aspect ratio, stroke count, and point count.

This is useful after editing dictionary JSON by hand because it shows whether the stored normalized strokes still look like the intended sigil or sign.

The export should look like:

```json
{
  "sourceAspectRatio": 1.12,
  "strokes": [
    [
      { "x": 0, "y": 0.5 },
      { "x": 1, "y": 0.5 }
    ]
  ]
}
```

Paste it like:

```json
{
  "id": "example-sign",
  "displayName": "Example Sign",
  "strokeTemplate": {
    "sourceAspectRatio": 1.12,
    "strokes": []
  }
}
```

## Adding A New Sigil

Use this checklist:

1. Add a new entry to `src/dictionary/sigils.json`.
2. Choose a stable `id`.
3. Set `displayName`.
4. Set `element`.
5. Set `allowedLayers`.
6. Add `semantic`.
7. Create and paste `strokeTemplate`.
8. Set `recognitionRotationInvariant`.
9. Test in the main app with Diagnostics enabled.

Minimal shape:

```json
{
  "id": "example-sigil",
  "displayName": "Example Sigil",
  "element": "light",
  "allowedLayers": ["center", "middle", "outer"],
  "strokeTemplate": {
    "sourceAspectRatio": 1,
    "strokes": []
  },
  "recognitionRotationInvariant": false,
  "semantic": {
    "force": 0,
    "focus": 0,
    "spread": 0,
    "range": 0,
    "lifetimeBias": 0
  }
}
```

## Adding A New Sign

Use this checklist:

1. Add a new entry to `src/dictionary/signs.json`.
2. Choose a stable `id`.
3. Set `displayName`.
4. Set `allowedLayers`.
5. Set `sourceNotes` when there is useful player-facing context.
6. Add `semantic.manifestation`.
7. Add `semantic.directionMode`.
8. Add numeric semantic deltas.
9. Create and paste `strokeTemplate`.
10. Test in the main app with Diagnostics enabled.

Minimal shape:

```json
{
  "id": "example-sign",
  "displayName": "Example Sign",
  "allowedLayers": ["middle", "outer"],
  "sourceNotes": "Longer explanation shown in the Signs tab.",
  "semantic": {
    "manifestation": "column",
    "directionMode": "orientation",
    "force": 0,
    "focus": 0,
    "spread": 0,
    "range": 0,
    "lifetimeBias": 0
  },
  "strokeTemplate": {
    "sourceAspectRatio": 1,
    "strokes": []
  }
}
```

## Adding A Sample Spell

Use this checklist:

1. Add a new entry to `src/dictionary/sample-spells.json`.
2. Choose a stable `id`.
3. Set `displayName` and a short `description`.
4. Set `element` to the primary element shown by the sample.
5. Set `manifestations` to the sign behaviors shown by the sample.
6. Add normalized `strokes` for the complete reference drawing.
7. Check the Dictionary panel preview.

Minimal shape:

```json
{
  "id": "example-sample",
  "displayName": "Example Sample",
  "description": "Short description shown on the sample card.",
  "element": "water",
  "manifestations": ["convergence"],
  "strokes": []
}
```

## Testing Notes

After adding or changing an entry:

1. Open the main app.
2. Enable Diagnostics.
3. Draw a ring, the sigil, and optionally signs.
4. Check Parser, AST, and IR output.
5. Confirm the expected id appears in recognitions and `SpellIR.manifestations`.

For sample spells, check the Dictionary panel preview instead. Sample spell entries are visual references and should not change Parser, AST, or IR output.

If recognition fails, check:

- Is the symbol grouped as one candidate?
- Is it inside an allowed layer?
- Is the drawing big enough to be legible without becoming a merged or distorted candidate?
- Is the final recognition score below `CONFIG.recognition.minConfidence`?
- Is another same-kind entry scoring too close?
