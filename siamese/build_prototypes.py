"""Build per-glyph prototype embeddings for the in-browser recognizer.

For every dictionary entry, rasterize its stroke template (with a few small
rotations as light test-time augmentation), embed each render through the
EXPORTED ONNX model (not PyTorch), and average + L2-normalize into one
prototype vector. The browser recognizer compares a candidate's embedding
against these by cosine similarity.

Embedding via onnxruntime here (rather than torch) guarantees the prototypes
live in the exact numeric space the browser will use.

Run:
    .venv/bin/python siamese/build_prototypes.py \
        --model ../wha-spell-simulator/src/parser/siamese-assets/model.onnx \
        --dict-dir ../wha-spell-simulator/src/dictionary \
        --out ../wha-spell-simulator/src/parser/siamese-assets/prototypes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dict_raster import image_to_array, load_dictionary, rasterize_strokes

TTA_ROTATIONS = (-8.0, 0.0, 8.0)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dict-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta_path = args.model.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    size = int(meta.get("imageSize", 56))
    input_name = meta.get("inputName", "glyph")
    output_name = meta.get("outputName", "embedding")

    entries = load_dictionary(args.dict_dir)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    glyphs = []
    for entry in entries:
        batch = np.stack([
            image_to_array(rasterize_strokes(entry.strokes, size=size, rotation_deg=rot), size=size)
            for rot in TTA_ROTATIONS
        ]).astype(np.float32)
        embeddings = session.run([output_name], {input_name: batch})[0]
        prototype = l2_normalize(embeddings.mean(axis=0))
        glyphs.append({
            "id": entry.id,
            "kind": entry.kind,
            "displayName": entry.display_name,
            "element": entry.element,
            "embedding": [round(float(v), 6) for v in prototype.tolist()],
        })

    payload = {
        "imageSize": size,
        "embeddingDim": len(glyphs[0]["embedding"]) if glyphs else 0,
        "ttaRotations": list(TTA_ROTATIONS),
        "glyphs": glyphs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {len(glyphs)} prototypes -> {args.out}  (dim={payload['embeddingDim']})")


if __name__ == "__main__":
    main()
