from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

import torch

from data import GlyphDataset


@dataclass(frozen=True)
class TripletGlyphBatch:
    main: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    main_ids: torch.Tensor
    negative_ids: torch.Tensor


class TripletBatchSampler(Protocol):
    def sample(self) -> TripletGlyphBatch:
        """Return one aligned triplet batch."""


class UniformTripletGlyphSampler:
    """Samples three aligned batches for Siamese/triplet training."""

    def __init__(
        self,
        dataset: GlyphDataset,
        *,
        classes_per_batch: int,
        seed: int | None = None,
    ) -> None:
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be >= 1")
        if len(dataset.glyph_ids) < 2:
            raise ValueError("dataset must contain at least two glyph ids")
        if classes_per_batch > len(dataset.glyph_ids):
            raise ValueError("classes_per_batch exceeds available glyph ids")

        self.dataset = dataset
        self.classes_per_batch = classes_per_batch
        self.rng = Random(seed)

    def sample(self) -> TripletGlyphBatch:
        glyph_ids = self.rng.sample(list(self.dataset.glyph_ids), self.classes_per_batch)
        negative_pool = glyph_ids if len(glyph_ids) > 1 else list(self.dataset.glyph_ids)
        negative_ids = [
            self.rng.choice([candidate for candidate in negative_pool if candidate != glyph_id])
            for glyph_id in glyph_ids
        ]
        return _build_triplet_batch(self.dataset, glyph_ids, negative_ids)


class ClassFilterSampler:
    """Samples negatives from a mutable per-class allow list."""

    def __init__(
        self,
        dataset: GlyphDataset,
        *,
        classes_per_batch: int,
        allowed_classes: Sequence[Sequence[int]],
        seed: int | None = None,
    ) -> None:
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be >= 1")
        if len(dataset.glyph_ids) < 2:
            raise ValueError("dataset must contain at least two glyph ids")
        if classes_per_batch > len(dataset.glyph_ids):
            raise ValueError("classes_per_batch exceeds available glyph ids")

        max_glyph_id = max(dataset.glyph_ids)
        if len(allowed_classes) <= max_glyph_id:
            raise ValueError("allowed_classes must be indexable by every glyph id")

        self.dataset = dataset
        self.dataset_glyph_ids = set(dataset.glyph_ids)
        self.classes_per_batch = classes_per_batch
        self.allowed_classes = allowed_classes
        self.rng = Random(seed)

    def sample(self) -> TripletGlyphBatch:
        glyph_ids = self.rng.sample(list(self.dataset.glyph_ids), self.classes_per_batch)
        negative_ids = [self._sample_negative_id(glyph_id) for glyph_id in glyph_ids]
        return _build_triplet_batch(self.dataset, glyph_ids, negative_ids)

    def _sample_negative_id(self, glyph_id: int) -> int:
        candidates = [
            int(candidate)
            for candidate in self.allowed_classes[int(glyph_id)]
            if int(candidate) in self.dataset_glyph_ids and int(candidate) != int(glyph_id)
        ]
        if not candidates:
            candidates = [
                candidate
                for candidate in self.dataset.glyph_ids
                if int(candidate) != int(glyph_id)
            ]
        return self.rng.choice(candidates)


def _build_triplet_batch(
    dataset: GlyphDataset,
    glyph_ids: Sequence[int],
    negative_ids: Sequence[int],
) -> TripletGlyphBatch:
    main_images = []
    positive_images = []
    negative_images = []

    for glyph_id, negative_id in zip(glyph_ids, negative_ids, strict=True):
        main_images.append(_as_image_batch_item(dataset.random_image(glyph_id)))
        positive_images.append(_as_image_batch_item(dataset.random_image(glyph_id)))
        negative_images.append(_as_image_batch_item(dataset.random_image(negative_id)))

    return TripletGlyphBatch(
        main=torch.stack(main_images),
        positive=torch.stack(positive_images),
        negative=torch.stack(negative_images),
        main_ids=torch.tensor(glyph_ids, dtype=torch.long),
        negative_ids=torch.tensor(negative_ids, dtype=torch.long),
    )


def _as_image_batch_item(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().clone().float()
    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3:
        return image
    raise ValueError(f"expected image tensor with 2 or 3 dims, got {image.shape}")


TripletGlyphSampler = UniformTripletGlyphSampler


__all__ = [
    "ClassFilterSampler",
    "TripletBatchSampler",
    "TripletGlyphBatch",
    "TripletGlyphSampler",
    "UniformTripletGlyphSampler",
]
