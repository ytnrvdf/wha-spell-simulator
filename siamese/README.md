# Siamese glyph recognizer

A learned, optional alternative to the app's built-in stroke-template matcher.
A small siamese CNN (metric-trained on Omniglot, then lightly adapted to this
project's dictionary) maps a rendered glyph to an embedding; recognition is a
cosine match against one prototype embedding per dictionary entry.

It **reuses the app's stroke grouping untouched** — it only replaces the
per-candidate classification step (`recognizeCandidates`). The model runs in the
browser via `onnxruntime-web`, so the static GitHub Pages deploy is preserved.

## Why this might help

The app README notes the template matcher "works best with clean, deliberate
drawings." A learned embedding is more tolerant of messy strokes, and adding a
new glyph is just "provide a template/example" — no hand-tuned thresholds.

## How the pieces fit

```
              trainer.py            finetune.py           export_onnx.py        build_prototypes.py
Omniglot ───► base .pt  ──────────► finetuned .pt ──────► model.onnx ─────────► prototypes.json
                                    (adapt to dict)        (+ .meta.json)        (one embedding/glyph)
                                                                │                      │
                                                                ▼                      ▼
                                                        src/parser/siamese-assets/  ◄──── browser: glyphRasterizer
                                                                                + siameseRecognizer.js
```

`dict_raster.py` (Python) and `src/parser/glyphRasterizer.js` (JS) are
**bit-for-bit identical** rasterizers — the supersampled disk-stamp renderer is
pure integer math on purpose. `tests/siameseParity.test.js` enforces this; if
they ever diverge, prototypes (built in Python) and browser candidates would
live in different input distributions (train/serve skew).

## Setup

```bash
cd siamese
uv sync
```

## Workflow

```bash
# 1. (optional) Train the base encoder on Omniglot — slow; or use the release asset.
uv run trainer.py --dataset omniglot --omniglot-split background --steps 10000

# 2. Adapt to this project's dictionary (fast). Works for ANY dictionary size.
uv run finetune.py --dict-dir ../src/dictionary

# 3. Export the fine-tuned checkpoint to ONNX (verifies torch<->onnx parity).
uv run export_onnx.py --checkpoint checkpoints/finetuned_glyph_net.pt \
    --out ../src/parser/siamese-assets/model.onnx

# 4. Build the prototype embeddings (runs through the exported model.onnx).
uv run build_prototypes.py --model ../src/parser/siamese-assets/model.onnx \
    --dict-dir ../src/dictionary --out ../src/parser/siamese-assets/prototypes.json

# 5. Regenerate the JS parity fixture (run after any rasterizer/model change).
uv run make_parity_fixture.py --model ../src/parser/siamese-assets/model.onnx \
    --dict-dir ../src/dictionary --out ../tests/fixtures/siameseParity.json
```

Then flip `recognition.engine` to `"siamese"` in `src/config.js` and run the
app, or open `tools/siameseRecognizerLab.html` to compare engines side by side.

## Adding or changing glyphs

Because recognition is few-shot (cosine to prototypes), a new glyph usually only
needs steps 2–5 re-run after editing the dictionary — no architecture change.
For best accuracy on a substantially larger or restyled dictionary, re-run
`finetune.py` (use `--classes-per-batch` to bound the batch for large
dictionaries).

## Evaluation

`finetune.py` reports top-1 on two fixed synthetic distributions — an
"operating" one (jitter matched to training) and a wider "stress" one — using a
held-out RNG seed disjoint from training. Latest run (5 sigils + 3 signs):

| distribution | base (Omniglot only) | fine-tuned |
| --- | --- | --- |
| operating | 62.7% | **99.8%** |
| stress | 33.8% | **84.2%** |

`poc_eval.py` prints the prototype-vs-prototype cosine matrix (class
separation) and a montage of the rendered glyphs.

### Honest caveats

These numbers are **synthetic**: training and evaluation both render the same
dictionary stroke templates (no independent hand-drawn dataset exists yet). The
disjoint eval seed and wider stress augmentation reduce — but do not eliminate —
that circularity. Treat them as an upper bound and validate on real drawings
(the lab tool, or the side-by-side compare) before trusting the engine in
production. The default engine remains `"template"`.

## Files

- `model/`, `data/`, `trainer.py`, `visualize_latent.py`, `glyph_gui.py` — the
  training stack (siamese encoder + samplers + augmentation).
- `dict_raster.py` — the shared rasterizer (mirror of the JS one).
- `finetune.py`, `export_onnx.py`, `build_prototypes.py`, `poc_eval.py`,
  `make_parity_fixture.py` — the adapt → export → ship pipeline.
