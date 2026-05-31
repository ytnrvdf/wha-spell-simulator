from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Protocol

import numpy as np
import torch

from data import GlyphDataset
from data.sampler import (
    ClassFilterSampler,
    TripletBatchSampler,
    TripletGlyphBatch,
    UniformTripletGlyphSampler,
)


TrainContext = dict[str, Any]

CLASS_ALLOWED_KEY = "class_sampling_candidates"
CLASS_BUFFERS_KEY = "class_embedding_buffers"
CLASS_MEANS_KEY = "class_distribution_means"
CLASS_STDS_KEY = "class_distribution_stds"
BBOX_RADIUS_MULTIPLIER = 1.0


@dataclass(frozen=True)
class TripletGlyphEmbeddings:
    main: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor


@dataclass
class CandidateRebuildStats:
    fallback_classes: int = 0
    candidate_topup_classes: int = 0
    candidate_topup_added: int = 0
    min_class_candidates: int = 0
    pair_tests: int = 0
    bbox_intersections: int = 0
    workers: int = 1


class TrainingPlugin(Protocol):
    def build_sampler(
        self,
        dataset: GlyphDataset,
        config: Any,
        train_ctx: TrainContext,
    ) -> TripletBatchSampler:
        """Build the sampler used by the trainer."""

    def pre_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> None:
        """Run before an epoch starts."""

    def post_train_step(
        self,
        train_ctx: TrainContext,
        epoch: int,
        step: int,
        batch: TripletGlyphBatch,
        embeddings: TripletGlyphEmbeddings,
        metrics: dict[str, float],
    ) -> None:
        """Run after one optimizer step."""

    def post_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> dict[str, float]:
        """Run after an epoch ends and return metrics to log."""


class UniformSamplerPlugin:
    def build_sampler(
        self,
        dataset: GlyphDataset,
        config: Any,
        train_ctx: TrainContext,
    ) -> TripletBatchSampler:
        return UniformTripletGlyphSampler(
            dataset,
            classes_per_batch=config.classes_per_batch,
            seed=config.seed,
        )

    def pre_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> None:
        return None

    def post_train_step(
        self,
        train_ctx: TrainContext,
        epoch: int,
        step: int,
        batch: TripletGlyphBatch,
        embeddings: TripletGlyphEmbeddings,
        metrics: dict[str, float],
    ) -> None:
        return None

    def post_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> dict[str, float]:
        return {}


class ClassStdSamplerPlugin:
    """Updates class-filtered sampling candidates from embedding distributions."""

    def __init__(
        self,
        *,
        distribution_ema_decay: float = 0.9,
        nearest_fallback_count: int = 10,
        min_class_candidates: int = 0,
        std_floor: float = 1e-6,
        intersection_workers: int = -1,
    ) -> None:
        if not 0.0 <= distribution_ema_decay < 1.0:
            raise ValueError("distribution_ema_decay must be in [0, 1)")
        if nearest_fallback_count < 1:
            raise ValueError("nearest_fallback_count must be >= 1")
        if min_class_candidates < 0:
            raise ValueError("min_class_candidates must be >= 0")
        if std_floor < 0:
            raise ValueError("std_floor must be >= 0")
        if intersection_workers < -1 or intersection_workers == 0:
            raise ValueError("intersection_workers must be -1 or >= 1")

        self.distribution_ema_decay = distribution_ema_decay
        self.nearest_fallback_count = nearest_fallback_count
        self.min_class_candidates = min_class_candidates
        self.std_floor = std_floor
        self.intersection_workers = _resolve_worker_count(intersection_workers)
        self.glyph_ids: list[int] = []
        self.allowed_classes: list[list[int]] = []

    def build_sampler(
        self,
        dataset: GlyphDataset,
        config: Any,
        train_ctx: TrainContext,
    ) -> TripletBatchSampler:
        self.glyph_ids = list(dataset.glyph_ids)
        self.allowed_classes = _all_other_classes(self.glyph_ids)
        train_ctx[CLASS_ALLOWED_KEY] = self.allowed_classes
        _ensure_class_lists(train_ctx, self.glyph_ids)
        return ClassFilterSampler(
            dataset,
            classes_per_batch=config.classes_per_batch,
            allowed_classes=self.allowed_classes,
            seed=config.seed,
        )

    def pre_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> None:
        _ensure_class_lists(train_ctx, self.glyph_ids)

    def post_train_step(
        self,
        train_ctx: TrainContext,
        epoch: int,
        step: int,
        batch: TripletGlyphBatch,
        embeddings: TripletGlyphEmbeddings,
        metrics: dict[str, float],
    ) -> None:
        buffers = _ensure_class_lists(train_ctx, self.glyph_ids)[CLASS_BUFFERS_KEY]
        _append_embeddings(buffers, batch.main_ids, embeddings.main)
        _append_embeddings(buffers, batch.main_ids, embeddings.positive)
        _append_embeddings(buffers, batch.negative_ids, embeddings.negative)

    def post_epoch(self, train_ctx: TrainContext, epoch: int, step: int) -> dict[str, float]:
        class_lists = _ensure_class_lists(train_ctx, self.glyph_ids)
        buffers = class_lists[CLASS_BUFFERS_KEY]
        means = class_lists[CLASS_MEANS_KEY]
        stds = class_lists[CLASS_STDS_KEY]

        collected_embeddings = sum(len(buffers[glyph_id]) for glyph_id in self.glyph_ids)
        classes_with_epoch_samples = 0

        for glyph_id in self.glyph_ids:
            class_embeddings = buffers[glyph_id]
            if not class_embeddings:
                continue

            values = np.stack(class_embeddings).astype(np.float32, copy=False)
            epoch_mean = values.mean(axis=0)
            epoch_std = np.maximum(values.std(axis=0), self.std_floor)
            if means[glyph_id] is None:
                means[glyph_id] = epoch_mean
                stds[glyph_id] = epoch_std
            else:
                previous_mean = means[glyph_id]
                previous_std = stds[glyph_id]
                means[glyph_id] = (
                    self.distribution_ema_decay * previous_mean
                    + (1.0 - self.distribution_ema_decay) * epoch_mean
                )
                stds[glyph_id] = np.maximum(
                    self.distribution_ema_decay * previous_std
                    + (1.0 - self.distribution_ema_decay) * epoch_std,
                    self.std_floor,
                )
            classes_with_epoch_samples += 1

        rebuild_stats = _rebuild_allowed_classes(
            self.glyph_ids,
            means,
            stds,
            self.allowed_classes,
            nearest_fallback_count=self.nearest_fallback_count,
            min_class_candidates=self.min_class_candidates,
            intersection_workers=self.intersection_workers,
        )

        for glyph_id in self.glyph_ids:
            buffers[glyph_id].clear()

        candidate_counts = [len(self.allowed_classes[glyph_id]) for glyph_id in self.glyph_ids]
        distribution_metrics = _distribution_metrics(
            self.glyph_ids,
            means,
            stds,
            self.allowed_classes,
            nearest_fallback_count=self.nearest_fallback_count,
        )
        return {
            "epoch_collected_embeddings": float(collected_embeddings),
            "epoch_classes_with_samples": float(classes_with_epoch_samples),
            "epoch_classes_with_stats": float(
                sum(means[glyph_id] is not None for glyph_id in self.glyph_ids)
            ),
            "epoch_candidate_min": float(min(candidate_counts) if candidate_counts else 0),
            "epoch_candidate_mean": float(np.mean(candidate_counts) if candidate_counts else 0.0),
            "epoch_candidate_max": float(max(candidate_counts) if candidate_counts else 0),
            "epoch_fallback_classes": float(rebuild_stats.fallback_classes),
            "epoch_min_class_candidates": float(rebuild_stats.min_class_candidates),
            "epoch_candidate_topup_classes": float(rebuild_stats.candidate_topup_classes),
            "epoch_candidate_topup_added": float(rebuild_stats.candidate_topup_added),
            "epoch_bbox_pair_tests": float(rebuild_stats.pair_tests),
            "epoch_bbox_intersections": float(rebuild_stats.bbox_intersections),
            "epoch_intersection_workers": float(rebuild_stats.workers),
            **distribution_metrics,
        }


def _ensure_class_lists(train_ctx: TrainContext, glyph_ids: list[int]) -> TrainContext:
    max_glyph_id = max(glyph_ids)
    class_count = max_glyph_id + 1
    if CLASS_BUFFERS_KEY not in train_ctx:
        train_ctx[CLASS_BUFFERS_KEY] = [[] for _ in range(class_count)]
    elif len(train_ctx[CLASS_BUFFERS_KEY]) < class_count:
        train_ctx[CLASS_BUFFERS_KEY].extend(
            [] for _ in range(class_count - len(train_ctx[CLASS_BUFFERS_KEY]))
        )
    if CLASS_MEANS_KEY not in train_ctx:
        train_ctx[CLASS_MEANS_KEY] = [None for _ in range(class_count)]
    elif len(train_ctx[CLASS_MEANS_KEY]) < class_count:
        train_ctx[CLASS_MEANS_KEY].extend(
            None for _ in range(class_count - len(train_ctx[CLASS_MEANS_KEY]))
        )
    if CLASS_STDS_KEY not in train_ctx:
        train_ctx[CLASS_STDS_KEY] = [None for _ in range(class_count)]
    elif len(train_ctx[CLASS_STDS_KEY]) < class_count:
        train_ctx[CLASS_STDS_KEY].extend(
            None for _ in range(class_count - len(train_ctx[CLASS_STDS_KEY]))
        )
    return train_ctx


def _append_embeddings(
    buffers: list[list[np.ndarray]],
    class_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> None:
    class_ids_list = class_ids.detach().cpu().tolist()
    vectors = embeddings.detach().float().cpu().numpy()
    for class_id, vector in zip(class_ids_list, vectors, strict=True):
        buffers[int(class_id)].append(vector.copy())


def _all_other_classes(glyph_ids: list[int]) -> list[list[int]]:
    max_glyph_id = max(glyph_ids)
    allowed_classes = [[] for _ in range(max_glyph_id + 1)]
    for glyph_id in glyph_ids:
        allowed_classes[glyph_id] = [candidate for candidate in glyph_ids if candidate != glyph_id]
    return allowed_classes


def _rebuild_allowed_classes(
    glyph_ids: list[int],
    means: list[np.ndarray | None],
    stds: list[np.ndarray | None],
    allowed_classes: list[list[int]],
    *,
    nearest_fallback_count: int,
    min_class_candidates: int,
    intersection_workers: int = 1,
) -> CandidateRebuildStats:
    stats = CandidateRebuildStats()
    stats.workers = intersection_workers
    stats.min_class_candidates = min_class_candidates
    all_other = _all_other_classes(glyph_ids)
    intersections_by_id = {glyph_id: [] for glyph_id in glyph_ids}
    glyph_ids_with_stats = [
        glyph_id
        for glyph_id in glyph_ids
        if means[glyph_id] is not None and stds[glyph_id] is not None
    ]

    pair_chunks = list(_bbox_pair_chunks(glyph_ids_with_stats))
    if intersection_workers == 1 or len(pair_chunks) <= 1:
        chunk_results = [
            _check_bbox_pair_chunk(chunk, means, stds)
            for chunk in pair_chunks
        ]
    else:
        worker_stats = _build_worker_distribution_stats(glyph_ids_with_stats, means, stds)
        with ProcessPoolExecutor(max_workers=intersection_workers) as executor:
            chunk_results = list(
                executor.map(
                    _check_bbox_pair_chunk_from_stats,
                    [(chunk, worker_stats) for chunk in pair_chunks],
                )
            )

    for chunk_stats, intersections in chunk_results:
        _merge_rebuild_stats(stats, chunk_stats)
        for glyph_id, other_id in intersections:
            intersections_by_id[glyph_id].append(other_id)
            intersections_by_id[other_id].append(glyph_id)

    for glyph_id in glyph_ids:
        mean = means[glyph_id]
        std = stds[glyph_id]
        if mean is None or std is None:
            allowed_classes[glyph_id] = all_other[glyph_id]
            stats.fallback_classes += 1
            continue

        intersections = list(intersections_by_id[glyph_id])
        if intersections:
            candidates, added_count = _top_up_with_nearest_classes(
                glyph_id,
                glyph_ids,
                means,
                intersections,
                min_class_candidates,
            )
            allowed_classes[glyph_id] = candidates
            if added_count > 0:
                stats.candidate_topup_classes += 1
                stats.candidate_topup_added += added_count
            continue

        fallback_count = max(nearest_fallback_count, min_class_candidates)
        nearest = _nearest_classes(glyph_id, glyph_ids, means, fallback_count)
        allowed_classes[glyph_id] = nearest if nearest else all_other[glyph_id]
        stats.fallback_classes += 1
        if min_class_candidates > 0 and nearest:
            stats.candidate_topup_classes += 1
            stats.candidate_topup_added += len(nearest)

    return stats


def _resolve_worker_count(requested_workers: int) -> int:
    if requested_workers == -1:
        return max(1, min(os.cpu_count() or 1, 8))
    return requested_workers


def _bbox_pair_chunks(
    glyph_ids: list[int],
    *,
    chunk_size: int = 4096,
) -> list[list[tuple[int, int]]]:
    chunks: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for index, glyph_id in enumerate(glyph_ids):
        for other_id in glyph_ids[index + 1 :]:
            current.append((glyph_id, other_id))
            if len(current) >= chunk_size:
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)
    return chunks


def _check_bbox_pair_chunk(
    pairs: list[tuple[int, int]],
    means: list[np.ndarray | None],
    stds: list[np.ndarray | None],
) -> tuple[CandidateRebuildStats, list[tuple[int, int]]]:
    stats = CandidateRebuildStats()
    intersections = []
    for glyph_id, other_id in pairs:
        mean = means[glyph_id]
        std = stds[glyph_id]
        other_mean = means[other_id]
        other_std = stds[other_id]
        if mean is None or std is None or other_mean is None or other_std is None:
            continue
        stats.pair_tests += 1
        intersects = _axis_aligned_bboxes_intersect(mean, std, other_mean, other_std)
        if not intersects:
            continue
        intersections.append((glyph_id, other_id))
        stats.bbox_intersections += 1
    return stats, intersections


def _build_worker_distribution_stats(
    glyph_ids: list[int],
    means: list[np.ndarray | None],
    stds: list[np.ndarray | None],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        glyph_id: (means[glyph_id], stds[glyph_id])
        for glyph_id in glyph_ids
        if means[glyph_id] is not None and stds[glyph_id] is not None
    }


def _check_bbox_pair_chunk_from_stats(
    args: tuple[
        list[tuple[int, int]],
        dict[int, tuple[np.ndarray, np.ndarray]],
    ],
) -> tuple[CandidateRebuildStats, list[tuple[int, int]]]:
    pairs, distribution_stats = args
    stats = CandidateRebuildStats()
    intersections = []
    for glyph_id, other_id in pairs:
        mean, std = distribution_stats[glyph_id]
        other_mean, other_std = distribution_stats[other_id]
        stats.pair_tests += 1
        intersects = _axis_aligned_bboxes_intersect(mean, std, other_mean, other_std)
        if not intersects:
            continue
        intersections.append((glyph_id, other_id))
        stats.bbox_intersections += 1
    return stats, intersections


def _merge_rebuild_stats(target: CandidateRebuildStats, source: CandidateRebuildStats) -> None:
    target.pair_tests += source.pair_tests
    target.bbox_intersections += source.bbox_intersections


def _axis_aligned_bboxes_intersect(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    mean_b: np.ndarray,
    std_b: np.ndarray,
) -> bool:
    radius_a = np.maximum(BBOX_RADIUS_MULTIPLIER * std_a, np.finfo(np.float64).eps).astype(
        np.float64,
        copy=False,
    )
    radius_b = np.maximum(BBOX_RADIUS_MULTIPLIER * std_b, np.finfo(np.float64).eps).astype(
        np.float64,
        copy=False,
    )
    center_a = mean_a.astype(np.float64, copy=False)
    center_b = mean_b.astype(np.float64, copy=False)
    return bool(np.all(np.abs(center_a - center_b) <= radius_a + radius_b))


def _distribution_metrics(
    glyph_ids: list[int],
    means: list[np.ndarray | None],
    stds: list[np.ndarray | None],
    allowed_classes: list[list[int]],
    *,
    nearest_fallback_count: int,
) -> dict[str, float]:
    center_norms = []
    std_norms = []
    candidate_distances = []
    nearest_distances = []
    for glyph_id in glyph_ids:
        mean = means[glyph_id]
        std = stds[glyph_id]
        if mean is None or std is None:
            continue
        center_norms.append(float(np.linalg.norm(mean)))
        std_norms.append(float(np.linalg.norm(std)))

        for other_id in allowed_classes[glyph_id]:
            other_mean = means[other_id]
            if other_mean is not None:
                candidate_distances.append(float(np.linalg.norm(mean - other_mean)))

        distances = [
            float(np.linalg.norm(mean - means[other_id]))
            for other_id in glyph_ids
            if other_id != glyph_id and means[other_id] is not None
        ]
        distances.sort()
        nearest_distances.extend(distances[:nearest_fallback_count])

    return {
        "epoch_center_norm_mean": float(np.mean(center_norms) if center_norms else 0.0),
        "epoch_std_norm_mean": float(np.mean(std_norms) if std_norms else 0.0),
        "epoch_candidate_center_distance_mean": float(
            np.mean(candidate_distances) if candidate_distances else 0.0
        ),
        "epoch_nearest_center_distance_mean": float(
            np.mean(nearest_distances) if nearest_distances else 0.0
        ),
    }


def _nearest_classes(
    glyph_id: int,
    glyph_ids: list[int],
    means: list[np.ndarray | None],
    nearest_fallback_count: int,
) -> list[int]:
    mean = means[glyph_id]
    if mean is None:
        return []

    distances = []
    for other_id in glyph_ids:
        other_mean = means[other_id]
        if other_id == glyph_id or other_mean is None:
            continue
        distances.append((float(np.linalg.norm(mean - other_mean)), other_id))
    distances.sort(key=lambda item: item[0])
    return [other_id for _, other_id in distances[:nearest_fallback_count]]


def _top_up_with_nearest_classes(
    glyph_id: int,
    glyph_ids: list[int],
    means: list[np.ndarray | None],
    candidates: list[int],
    min_class_candidates: int,
) -> tuple[list[int], int]:
    if min_class_candidates <= len(candidates):
        return candidates, 0

    result = list(candidates)
    seen = set(result)
    nearest = _nearest_classes(glyph_id, glyph_ids, means, len(glyph_ids) - 1)
    for other_id in nearest:
        if other_id in seen:
            continue
        result.append(other_id)
        seen.add(other_id)
        if len(result) >= min_class_candidates:
            break
    return result, len(result) - len(candidates)


__all__ = [
    "CLASS_ALLOWED_KEY",
    "CLASS_BUFFERS_KEY",
    "CLASS_MEANS_KEY",
    "CLASS_STDS_KEY",
    "ClassStdSamplerPlugin",
    "TrainContext",
    "TrainingPlugin",
    "TripletGlyphEmbeddings",
    "UniformSamplerPlugin",
]
