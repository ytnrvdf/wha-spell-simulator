"""Light contrastive fine-tune of the siamese net on the WHA dictionary.

This adapts the Omniglot-trained embedding to the *style* of the dictionary
glyphs using a batch-hard triplet objective over heavily augmented renders of
each stroke template. It deliberately stays metric-learning (no classifier
head) and uses a small learning rate for few steps, so the few-shot
"drop in a new glyph" property and Omniglot generalization are mostly kept.

Honesty note: training and evaluation both draw from the same 8 stroke
templates (no independent hand-drawn data exists yet), so the reported top-1
measures *robustness to augmentation*, not true generalization to unseen
drawings. Evaluation uses a disjoint RNG seed and wider jitter than training to
reduce — not eliminate — that circularity. Treat the number as an upper bound;
validate on real drawings before trusting it.

Run:
    .venv/bin/python siamese/finetune.py --dict-dir /path/to/src/dictionary
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dict_raster import DictEntry, image_to_array, load_dictionary, rasterize_strokes
from model.siam_net import build_siamese_model, model_config_from_checkpoint


def _model_config_with_image_size(config: dict) -> dict:
    config = dict(config)
    if "image_size" not in config:
        arch = config.get("architecture", "conv_attention")
        config["image_size"] = 56 if arch == "conv_attention" else 28
    return config


def _smooth_displacements(n: int, sigma: float, rng: random.Random, *, passes: int = 4) -> list[float]:
    """Low-frequency 1-D displacement field: white noise smoothed along the
    point sequence so neighboring points move together (correlated hand wobble,
    not high-frequency zigzag). Amplitude renormalized back to ~sigma."""
    if n == 0:
        return []
    values = [rng.gauss(0.0, sigma) for _ in range(n)]
    for _ in range(passes):
        smoothed = []
        for i in range(n):
            lo, hi = max(0, i - 1), min(n - 1, i + 1)
            smoothed.append(sum(values[lo:hi + 1]) / (hi - lo + 1))
        values = smoothed
    std = (sum(v * v for v in values) / n) ** 0.5
    if std > 1e-9:
        gain = sigma / std
        values = [v * gain for v in values]
    return values


def jitter_strokes(
    strokes: list[list[tuple[float, float]]],
    rng: random.Random,
    *,
    scale_x: float,
    scale_y: float,
    jitter_sigma: float,
) -> list[list[tuple[float, float]]]:
    """Anisotropic scale about center + smooth correlated per-stroke wobble.

    Translation and uniform scale are intentionally NOT applied: the renderer
    re-fits strokes to the unit box, mirroring how candidates are normalized at
    recognition time, so those would be washed out anyway. The wobble is a
    low-frequency displacement field (see _smooth_displacements) so it looks
    like real shaky drawing rather than per-point salt noise.
    """
    out = []
    for stroke in strokes:
        n = len(stroke)
        dx = _smooth_displacements(n, jitter_sigma, rng)
        dy = _smooth_displacements(n, jitter_sigma, rng)
        out.append(
            [
                (
                    (x - 0.5) * scale_x + 0.5 + dx[i],
                    (y - 0.5) * scale_y + 0.5 + dy[i],
                )
                for i, (x, y) in enumerate(stroke)
            ]
        )
    return out


def render_augmented(
    entry: DictEntry,
    size: int,
    rng: random.Random,
    *,
    rot_range: float,
    jitter_sigma: float,
    aspect_range: float,
    width_range: tuple[float, float],
    pad_range: tuple[float, float],
) -> Image.Image:
    aspect = math.exp(rng.uniform(-aspect_range, aspect_range))
    strokes = jitter_strokes(
        entry.strokes,
        rng,
        scale_x=aspect,
        scale_y=1.0 / aspect,
        jitter_sigma=jitter_sigma,
    )
    return rasterize_strokes(
        strokes,
        size=size,
        rotation_deg=rng.uniform(-rot_range, rot_range),
        stroke_width_ratio=rng.uniform(*width_range),
        padding_ratio=rng.uniform(*pad_range),
    )


def renders_to_batch(images: list[Image.Image], size: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([image_to_array(img, size=size) for img in images])).to(device)


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    # Cosine distance = 1 - cos_sim on L2-normalized embeddings.
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    positive = same & ~eye
    negative = ~same
    # Hardest positive: max distance to a same-class sample.
    hardest_pos = (dist.masked_fill(~positive, -1.0)).max(dim=1).values
    # Hardest negative: min distance to a different-class sample.
    hardest_neg = (dist.masked_fill(~negative, float("inf"))).min(dim=1).values
    valid = hardest_pos >= 0
    losses = F.relu(hardest_pos - hardest_neg + margin)[valid]
    return losses.mean() if losses.numel() else embeddings.new_zeros(())


# Two fixed eval distributions. "operating" matches the training augmentation
# width (the expected real-use jitter); "stress" is deliberately harsher to
# probe robustness. Both are reported so the gain is not cherry-picked.
EVAL_PRESETS = {
    "operating": dict(rot_range=18.0, jitter_sigma=0.03, aspect_range=0.2,
                      width_range=(0.04, 0.075), pad_range=(0.22, 0.34)),
    "stress": dict(rot_range=24.0, jitter_sigma=0.05, aspect_range=0.30,
                   width_range=(0.035, 0.08), pad_range=(0.2, 0.36)),
}


@torch.no_grad()
def evaluate(model, entries, size, device, *, samples, seed, preset="operating"):
    rng = random.Random(seed)  # fixed -> deterministic held-out set
    # Prototypes from light TTA (clean + small rotations).
    proto_imgs, proto_owner = [], []
    for index, entry in enumerate(entries):
        for rot in (-8.0, 0.0, 8.0):
            proto_imgs.append(rasterize_strokes(entry.strokes, size=size, rotation_deg=rot))
            proto_owner.append(index)
    emb = model(renders_to_batch(proto_imgs, size, device)).cpu()
    protos = torch.stack([
        F.normalize(emb[torch.tensor([o == i for o in proto_owner])].mean(0), p=2, dim=0)
        for i in range(len(entries))
    ])

    params = EVAL_PRESETS[preset]
    correct = total = 0
    for index, entry in enumerate(entries):
        imgs = [render_augmented(entry, size, rng, **params) for _ in range(samples)]
        e = F.normalize(model(renders_to_batch(imgs, size, device)).cpu(), p=2, dim=1)
        pred = (e @ protos.T).argmax(dim=1)
        correct += int((pred == index).sum())
        total += len(imgs)
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_siamese_glyph_net.pt"))
    parser.add_argument("--dict-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/finetuned_glyph_net.pt"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--renders-per-class", type=int, default=6)
    parser.add_argument("--classes-per-batch", type=int, default=0,
                        help="Classes sampled per step for large dictionaries; 0 = all classes.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    train_rng = random.Random(args.seed)
    val_seed = args.seed + 10_000    # early-stopping set
    test_seed = args.seed + 20_000   # unbiased headline set
    stress_seed = args.seed + 30_000  # robustness set

    entries = load_dictionary(args.dict_dir)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = _model_config_with_image_size(model_config_from_checkpoint(checkpoint))
    size = int(config["image_size"])
    model = build_siamese_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    def report(tag):
        results = {}
        for preset, seed in (("operating", test_seed), ("stress", stress_seed)):
            c, t = evaluate(model, entries, size, device, samples=args.eval_samples,
                            seed=seed, preset=preset)
            results[preset] = 100.0 * c / t
        print(f"[{tag}]  operating top-1: {results['operating']:.1f}%   "
              f"stress top-1: {results['stress']:.1f}%")
        return results

    model.eval()
    base_results = report("base")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # For large dictionaries, sample a subset of classes per step so the batch
    # stays bounded; 0 (default) uses every class each step (fine for small dicts).
    classes_per_batch = args.classes_per_batch or len(entries)
    c0v, t0v = evaluate(model, entries, size, device, samples=args.eval_samples,
                        seed=val_seed, preset="operating")
    best_acc = c0v / t0v
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for step in range(1, args.steps + 1):
        model.train()
        class_ids = (
            train_rng.sample(range(len(entries)), classes_per_batch)
            if classes_per_batch < len(entries)
            else list(range(len(entries)))
        )
        images, label_list = [], []
        for class_id in class_ids:
            for _ in range(args.renders_per_class):
                images.append(render_augmented(
                    entries[class_id], size, train_rng, rot_range=18.0, jitter_sigma=0.03,
                    aspect_range=0.2, width_range=(0.04, 0.075), pad_range=(0.22, 0.34),
                ))
                label_list.append(class_id)
        labels = torch.tensor(label_list, device=device)
        embeddings = model(renders_to_batch(images, size, device))
        loss = batch_hard_triplet_loss(embeddings, labels, args.margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            cv, tv = evaluate(model, entries, size, device, samples=args.eval_samples,
                              seed=val_seed, preset="operating")
            acc = cv / tv
            flag = ""
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                flag = " *"
            print(f"step {step:>4}  loss={loss.item():.4f}  val top-1: "
                  f"{cv}/{tv} = {100.0 * acc:.1f}%{flag}")

    # Restore best-on-val weights, then report unbiased test + stress numbers.
    model.load_state_dict(best_state)
    model.eval()
    print()
    tuned_results = report("fine-tuned (best-on-val)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": best_state, "model_config": config,
         "finetuned_from": str(args.checkpoint),
         "eval_operating_top1": tuned_results["operating"],
         "eval_stress_top1": tuned_results["stress"]},
        args.out,
    )
    print(f"\noperating top-1: base {base_results['operating']:.1f}%  ->  "
          f"tuned {tuned_results['operating']:.1f}%")
    print(f"stress    top-1: base {base_results['stress']:.1f}%  ->  "
          f"tuned {tuned_results['stress']:.1f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
