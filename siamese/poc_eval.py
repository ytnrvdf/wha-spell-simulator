"""PoC: does the Omniglot-trained siamese net transfer to the WHA dictionary?

Builds a prototype embedding per dictionary glyph from its rasterized stroke
template, then evaluates recognition by re-rendering each glyph under random
augmentation (rotation / scale jitter / stroke-width jitter) and checking
whether the nearest prototype is the correct class. Prints top-1 accuracy, a
per-class breakdown, and the prototype-vs-prototype cosine matrix (how
separable the classes are in embedding space). Saves a montage of the rendered
glyphs for visual sanity checking.

Run:
    .venv/bin/python siamese/poc_eval.py \
        --checkpoint siamese/checkpoints/best_siamese_glyph_net.pt \
        --dict-dir ../wha-spell-simulator/src/dictionary
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dict_raster import (
    DictEntry,
    image_to_array,
    load_dictionary,
    rasterize_strokes,
)
from model.siam_net import build_siamese_model, model_config_from_checkpoint


def _model_config_with_image_size(config: dict) -> dict:
    config = dict(config)
    if "image_size" not in config:
        arch = config.get("architecture", "conv_attention")
        config["image_size"] = 56 if arch == "conv_attention" else 28
    return config


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = _model_config_with_image_size(model_config_from_checkpoint(checkpoint))
    model = build_siamese_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(config["image_size"])


@torch.no_grad()
def embed_images(model, images: list[Image.Image], size: int, device: torch.device) -> torch.Tensor:
    batch = torch.from_numpy(
        np.stack([image_to_array(img, size=size) for img in images])
    ).to(device)
    return model(batch).cpu()


def build_prototypes(model, entries, size, device, *, jitter_rotations=(0.0,)):
    images, owners = [], []
    for index, entry in enumerate(entries):
        for rot in jitter_rotations:
            images.append(rasterize_strokes(entry.strokes, size=size, rotation_deg=rot))
            owners.append(index)
    embeddings = embed_images(model, images, size, device)
    protos = []
    for index in range(len(entries)):
        mask = [owner == index for owner in owners]
        rows = embeddings[torch.tensor(mask)]
        protos.append(F.normalize(rows.mean(dim=0), p=2, dim=0))
    return torch.stack(protos)


def augmented_renders(entry: DictEntry, size: int, count: int, rng: random.Random):
    images = []
    for _ in range(count):
        rot = rng.uniform(-18.0, 18.0)
        width = rng.uniform(0.04, 0.075)
        pad = rng.uniform(0.22, 0.34)
        images.append(
            rasterize_strokes(
                entry.strokes,
                size=size,
                rotation_deg=rot,
                stroke_width_ratio=width,
                padding_ratio=pad,
            )
        )
    return images


def save_montage(entries, size, path: Path):
    cols = len(entries)
    montage = Image.new("L", (cols * size, size), 0)
    for index, entry in enumerate(entries):
        montage.paste(rasterize_strokes(entry.strokes, size=size), (index * size, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("siamese/checkpoints/best_siamese_glyph_net.pt"))
    parser.add_argument("--dict-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--montage", type=Path, default=Path("siamese/outputs/dict_montage.png"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    rng = random.Random(args.seed)

    entries = load_dictionary(args.dict_dir)
    model, size = load_model(args.checkpoint, device)
    print(f"Loaded {len(entries)} dictionary glyphs; image size {size}; device {device}")
    save_montage(entries, size, args.montage)
    print(f"Saved render montage -> {args.montage}")

    # Prototypes from a few rotated renders each (light test-time augmentation).
    protos = build_prototypes(
        model, entries, size, device, jitter_rotations=(-8.0, 0.0, 8.0)
    )

    # Prototype separation matrix.
    sim = protos @ protos.T
    names = [e.id for e in entries]
    print("\n=== prototype cosine similarity (off-diagonal = confusability) ===")
    print("            " + " ".join(f"{n[:8]:>8}" for n in names))
    for i, name in enumerate(names):
        row = " ".join(f"{sim[i, j].item():8.3f}" for j in range(len(names)))
        print(f"{name[:11]:>11} {row}")
    off = sim - torch.eye(len(entries))
    print(f"\nmax off-diagonal similarity: {off.max().item():.3f} "
          f"(lower is better; >0.9 means two glyphs look identical to the net)")

    # Recognition accuracy on augmented renders.
    print(f"\n=== nearest-prototype recognition on {args.samples} augmented renders/glyph ===")
    total_correct = total = 0
    for index, entry in enumerate(entries):
        images = augmented_renders(entry, size, args.samples, rng)
        emb = F.normalize(embed_images(model, images, size, device), p=2, dim=1)
        sims = emb @ protos.T
        probs = torch.softmax(sims / args.temperature, dim=1)
        conf, pred = probs.max(dim=1)
        correct = int((pred == index).sum())
        total_correct += correct
        total += len(images)
        mean_conf = conf[pred == index].mean().item() if correct else 0.0
        print(f"  {entry.id:>16} ({entry.kind:>5}): "
              f"{correct:>3}/{len(images)} top-1  mean_conf={mean_conf:5.2f}")
    print(f"\nOVERALL top-1 accuracy: {total_correct}/{total} = "
          f"{100.0 * total_correct / max(1, total):.1f}%")


if __name__ == "__main__":
    main()
