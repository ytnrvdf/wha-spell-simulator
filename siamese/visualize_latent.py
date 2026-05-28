from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from data.mnist import MnistGlyphDataset
from model.siam_net import build_siamese_model, model_config_from_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="checkpoints/best_siamese_glyph_net.pt")
    parser.add_argument("--data-root", default="datasets/mnist")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--method", choices=("umap", "tsne"), default="umap")
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/latent_space.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    dataset = MnistGlyphDataset(
        args.data_root,
        train=not args.test,
        download=args.download,
        seed=args.seed,
    )

    if args.random_weights:
        model = build_siamese_model().to(device)
    else:
        model = load_checkpoint(Path(args.checkpoint_path), device)
    model.eval()

    embeddings, glyph_ids = collect_embeddings(
        model,
        dataset,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        device=device,
    )
    points = project_embeddings(
        embeddings,
        method=args.method,
        seed=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
    )
    save_plot(
        points,
        glyph_ids,
        dataset.glyph_ids,
        output=Path(args.output),
        rng=rng,
        title=f"{args.method.upper()} latent space",
    )
    print(f"saved latent plot: {args.output}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_siamese_model(model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    dataset: MnistGlyphDataset,
    *,
    samples_per_class: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    images = []
    glyph_ids = []
    for glyph_id in dataset.glyph_ids:
        for _ in range(samples_per_class):
            images.append(dataset.random_image(glyph_id).unsqueeze(0))
            glyph_ids.append(glyph_id)

    embeddings = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size]).to(device)
        embeddings.append(model(batch).cpu())

    return torch.cat(embeddings).numpy(), np.asarray(glyph_ids)


def project_embeddings(
    embeddings: np.ndarray,
    *,
    method: str,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
    tsne_perplexity: float,
) -> np.ndarray:
    if method == "umap":
        from umap import UMAP

        return UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(embeddings)
    if method == "tsne":
        return TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(embeddings)
    raise ValueError(f"unknown projection method: {method}")


def save_plot(
    points: np.ndarray,
    glyph_ids: np.ndarray,
    classes: tuple[int, ...],
    *,
    output: Path,
    rng: np.random.Generator,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {glyph_id: rng.random(3) for glyph_id in classes}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    for glyph_id in classes:
        mask = glyph_ids == glyph_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=10,
            color=colors[glyph_id],
            alpha=0.78,
            label=str(glyph_id),
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(title="glyph_id", markerscale=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
