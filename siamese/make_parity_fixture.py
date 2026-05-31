"""Generate the cross-language parity fixture consumed by the JS test suite.

Writes tests/fixtures/siameseParity.json with:
  - rasterCases: synthetic + real stroke sets rendered by dict_raster, so the
    JS glyphRasterizer can be asserted to reproduce the exact same pixels
    (this is the train/serve-skew guard).
  - embeddingCase: one raster fed through the exported ONNX model, so the JS
    onnxruntime-web inference can be checked against the Python embedding.

Run AFTER export_onnx.py:
    .venv/bin/python siamese/make_parity_fixture.py \
        --model ../wha-spell-simulator/src/parser/siamese-assets/model.onnx \
        --dict-dir ../wha-spell-simulator/src/dictionary \
        --out ../wha-spell-simulator/tests/fixtures/siameseParity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dict_raster import image_to_array, load_dictionary, rasterize_strokes

# A few deterministic synthetic stroke sets (lists of [x, y] polylines).
SYNTHETIC = {
    "diagonal": [[[0.1, 0.1], [0.9, 0.9]]],
    "vee": [[[0.2, 0.2], [0.5, 0.85], [0.8, 0.2]]],
    "box": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]],
    "two_strokes": [[[0.2, 0.5], [0.8, 0.5]], [[0.5, 0.2], [0.5, 0.8]]],
}


def strokes_to_tuples(strokes):
    return [[(float(x), float(y)) for x, y in stroke] for stroke in strokes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dict-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta_path = args.model.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    size = int(meta.get("imageSize", 56))

    raster_cases = []

    cases = {f"synthetic:{name}": strokes for name, strokes in SYNTHETIC.items()}
    # Include the first two real dictionary glyphs for realistic coverage.
    for entry in load_dictionary(args.dict_dir)[:2]:
        cases[f"dict:{entry.id}"] = entry.strokes

    for name, strokes in cases.items():
        tuples = strokes_to_tuples(strokes)
        for rotation in (0.0, 37.0):
            raster = np.asarray(rasterize_strokes(tuples, size=size, rotation_deg=rotation))
            raster_cases.append({
                "name": name,
                "rotationDeg": rotation,
                "size": size,
                "strokes": [[[float(x), float(y)] for x, y in stroke] for stroke in tuples],
                "raster": raster.flatten().astype(int).tolist(),
            })

    # Embedding case: render one glyph, run ONNX, store input + embedding.
    embed_strokes = strokes_to_tuples(SYNTHETIC["vee"])
    embed_raster = rasterize_strokes(embed_strokes, size=size, rotation_deg=0.0)
    model_input = image_to_array(embed_raster, size=size)[None].astype(np.float32)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    embedding = session.run(
        [meta.get("outputName", "embedding")],
        {meta.get("inputName", "glyph"): model_input},
    )[0][0]

    payload = {
        "size": size,
        "rasterCases": raster_cases,
        "embeddingCase": {
            "strokes": [[[float(x), float(y)] for x, y in stroke] for stroke in embed_strokes],
            "rotationDeg": 0.0,
            "embedding": [round(float(v), 6) for v in embedding.tolist()],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote parity fixture ({len(raster_cases)} raster cases) -> {args.out}")


if __name__ == "__main__":
    main()
