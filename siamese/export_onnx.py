"""Export a trained siamese checkpoint to ONNX for in-browser inference.

The exported graph takes a float32 tensor [batch, 1, size, size] with pixels in
[0, 1] (white ink on black, same as the trainer) and returns L2-normalized
embeddings [batch, embedding_dim]. After export it verifies the ONNX Runtime
output matches PyTorch within tolerance, so a silent op-translation bug can't
slip through to the browser.

Run:
    .venv/bin/python siamese/export_onnx.py \
        --checkpoint siamese/checkpoints/finetuned_glyph_net.pt \
        --out ../wha-spell-simulator/src/parser/siamese-assets/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model.siam_net import build_siamese_model, model_config_from_checkpoint


def _model_config_with_image_size(config: dict) -> dict:
    config = dict(config)
    if "image_size" not in config:
        arch = config.get("architecture", "conv_attention")
        config["image_size"] = 56 if arch == "conv_attention" else 28
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetuned_glyph_net.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cpu")  # export on CPU for portable graph
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = _model_config_with_image_size(model_config_from_checkpoint(checkpoint))
    size = int(config["image_size"])
    model = build_siamese_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.rand(2, int(config.get("in_channels", 1)), size, size)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["glyph"],
        output_names=["embedding"],
        dynamic_axes={"glyph": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        # Legacy TorchScript exporter: the dynamo path lacks an ONNX mapping for
        # adaptive_max_pool2d, while TorchScript emits GlobalMaxPool cleanly.
        dynamo=False,
    )
    print(f"exported -> {args.out}  (image_size={size}, opset={args.opset})")

    # Numerical parity check: PyTorch vs ONNX Runtime.
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["embedding"], {"glyph": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"max |torch - onnx| = {max_diff:.2e}")
    if max_diff > args.tolerance:
        raise SystemExit(f"ONNX output diverges from PyTorch (> {args.tolerance}); aborting.")
    print("ONNX parity OK")

    # Emit a tiny sidecar so the JS side knows the expected input geometry.
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        '{\n'
        f'  "imageSize": {size},\n'
        f'  "inChannels": {int(config.get("in_channels", 1))},\n'
        f'  "embeddingDim": {int(config.get("embedding_dim", 64))},\n'
        '  "inputName": "glyph",\n'
        '  "outputName": "embedding",\n'
        '  "normalized": true\n'
        '}\n',
        encoding="utf-8",
    )
    print(f"wrote sidecar -> {meta_path}")


if __name__ == "__main__":
    main()
