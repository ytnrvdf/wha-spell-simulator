from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from aim import Run
from PIL import Image
from torch import nn
from torch.nn import functional as F

from data.augment import AugmentationConfig, RandomGlyphAugment
from data import GlyphDataset
from data.mnist import MnistGlyphDataset
from data.omniglot import OmniglotGlyphDataset
from data.sampler import TripletBatchSampler
from data.sampler_plugin import (
    ClassStdSamplerPlugin,
    TrainContext,
    TrainingPlugin,
    TripletGlyphEmbeddings,
    UniformSamplerPlugin,
)
from model.siam_net import SiameseGlyphNet, model_config_from_trainer_config


@dataclass(frozen=True)
class TrainerConfig:
    steps: int = 10_000
    steps_per_epoch: int = 1_000
    classes_per_batch: int = 10
    learning_rate: float = 1e-4
    margin: float = 1.0
    checkpoint_path: str = "checkpoints/best_siamese_glyph_net.pt"
    aim_repo: str = ".aim"
    aim_experiment: str = "siamese-glyphs"
    device: str = "auto"
    log_every: int = 10
    validation_every: int = 1_000
    validation_support_per_class: int = 5
    validation_batch_size: int = 512
    class_center_weight: float = 0.02
    seed: int = 1
    augment: bool = True
    rotation_degrees: float = 20.0
    translate: float = 0.18
    scale_min: float = 0.85
    scale_max: float = 1.15
    flip_probability: float = 0.0
    shear_degrees: float = 10.0
    sampler_plugin: str = "class-std"
    distribution_ema_decay: float = 0.9
    nearest_fallback_count: int = 10
    min_class_candidates: int = 0
    class_intersection_workers: int = -1
    model_architecture: str = "conv_attention"
    embedding_dim: int = 64
    base_channels: int = 32
    attention_heads: int = 4
    attention_layers: int = 2
    projection_hidden_dim: int = 512
    model_dropout: float = 0.1


@dataclass(frozen=True)
class ValidationEpisode:
    support_images: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    glyph_ids: tuple[int, ...]


class SiameseTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        sampler: TripletBatchSampler,
        plugin: TrainingPlugin,
        config: TrainerConfig,
        model_config: dict[str, object] | None = None,
        train_ctx: TrainContext | None = None,
        validation_dataset: GlyphDataset | None = None,
        train_glyph_ids: tuple[int, ...] | None = None,
        class_center_validation: bool = False,
    ) -> None:
        self.config = config
        if config.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        if config.class_center_weight < 0.0:
            raise ValueError("class_center_weight must be >= 0")
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.model_config = model_config or {}
        self.sampler = sampler
        self.plugin = plugin
        self.train_ctx = train_ctx if train_ctx is not None else {}
        self.class_glyph_ids = tuple(int(glyph_id) for glyph_id in (train_glyph_ids or ()))
        self.glyph_id_to_class_index = {
            glyph_id: index for index, glyph_id in enumerate(self.class_glyph_ids)
        }
        self.class_center_validation = class_center_validation
        self.validation_episode = (
            build_validation_episode(
                validation_dataset,
                support_per_class=config.validation_support_per_class,
            )
            if validation_dataset is not None and config.validation_every >= 1
            else None
        )
        self.class_centers = self._build_class_centers()
        optimizer_params: list[dict[str, object]] = [{"params": self.model.parameters()}]
        if self.class_centers is not None:
            optimizer_params.append(
                {
                    "params": self.class_centers.parameters(),
                    "weight_decay": 0.0,
                }
            )
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=config.learning_rate)
        self.augment = RandomGlyphAugment(
            AugmentationConfig(
                enabled=config.augment,
                rotation_degrees=config.rotation_degrees,
                translate=config.translate,
                scale_min=config.scale_min,
                scale_max=config.scale_max,
                flip_probability=config.flip_probability,
                shear_degrees=config.shear_degrees,
            )
        )
        self.best_validation_accuracy = float("-inf")

    def _build_class_centers(self) -> nn.Embedding | None:
        if self.config.class_center_weight <= 0.0:
            return None
        if not self.class_glyph_ids:
            raise ValueError("train_glyph_ids must be provided when class centers are enabled")

        embedding_dim = int(self.model_config.get("embedding_dim", self.config.embedding_dim))
        centers = nn.Embedding(len(self.class_glyph_ids), embedding_dim).to(self.device)
        with torch.no_grad():
            nn.init.normal_(centers.weight)
            centers.weight.copy_(F.normalize(centers.weight, dim=1))
        return centers

    def train(self) -> Path:
        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        run = Run(repo=self.config.aim_repo, experiment=self.config.aim_experiment)
        run["hparams"] = asdict(self.config)
        run["model"] = self.model.__class__.__name__

        try:
            epoch = 0
            for step in range(1, self.config.steps + 1):
                if (step - 1) % self.config.steps_per_epoch == 0:
                    epoch += 1
                    self.plugin.pre_epoch(self.train_ctx, epoch, step)

                metrics = self.train_step(epoch=epoch, step=step)
                for name, value in metrics.items():
                    run.track(value, name=name, step=step)

                validation_metrics = self.maybe_validate(step)
                if validation_metrics:
                    for name, value in validation_metrics.items():
                        run.track(value, name=name, step=step)
                    metrics.update(validation_metrics)
                    class_center_accuracy = validation_metrics.get(
                        "validation_class_center_accuracy"
                    )
                    class_center_text = (
                        f" validation_class_center_accuracy={class_center_accuracy:.4f}"
                        if class_center_accuracy is not None
                        else ""
                    )
                    print(
                        f"step={step} "
                        f"validation_accuracy="
                        f"{validation_metrics['validation_accuracy']:.4f}"
                        f"{class_center_text}"
                    )
                    if (
                        validation_metrics["validation_accuracy"]
                        > self.best_validation_accuracy
                    ):
                        self.best_validation_accuracy = validation_metrics[
                            "validation_accuracy"
                        ]
                        self.save_checkpoint(checkpoint_path, step, metrics)
                        print(
                            f"saved best checkpoint: path={checkpoint_path} "
                            f"step={step} "
                            f"validation_accuracy="
                            f"{validation_metrics['validation_accuracy']:.4f}"
                        )

                if self.config.log_every > 0 and step % self.config.log_every == 0:
                    center_distance = metrics.get("class_center_distance")
                    center_text = (
                        f" center={center_distance:.4f}"
                        if center_distance is not None
                        else ""
                    )
                    print(
                        f"step={step} loss={metrics['loss']:.4f} "
                        f"pos={metrics['positive_distance']:.4f} "
                        f"neg={metrics['main_negative_distance']:.4f}"
                        f"{center_text}"
                    )

                if step % self.config.steps_per_epoch == 0 or step == self.config.steps:
                    epoch_metrics = self.plugin.post_epoch(self.train_ctx, epoch, step)
                    for name, value in epoch_metrics.items():
                        run.track(value, name=name, step=step)
                    if epoch_metrics:
                        print(_format_epoch_metrics(epoch, step, epoch_metrics))
        finally:
            run.close()

        return checkpoint_path

    def train_step(self, *, epoch: int, step: int) -> dict[str, float]:
        self.model.train()
        batch = self.sampler.sample()
        main = batch.main.to(self.device)
        positive = batch.positive.to(self.device)
        negative = batch.negative.to(self.device)
        main = self.augment(main)
        positive = self.augment(positive)
        negative = self.augment(negative)

        main_embedding = self.model(main)
        positive_embedding = self.model(positive)
        negative_embedding = self.model(negative)

        loss, distances = three_point_loss(
            main_embedding,
            positive_embedding,
            negative_embedding,
            margin=self.config.margin,
        )
        center_loss, center_metrics = self.class_center_regularization(
            main_embedding=main_embedding,
            positive_embedding=positive_embedding,
            negative_embedding=negative_embedding,
            main_ids=batch.main_ids,
            negative_ids=batch.negative_ids,
        )
        loss = loss + center_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        metrics = {name: value.detach().item() for name, value in distances.items()}
        metrics.update({name: value.detach().item() for name, value in center_metrics.items()})
        metrics["loss"] = loss.detach().item()
        self.plugin.post_train_step(
            self.train_ctx,
            epoch,
            step,
            batch,
            TripletGlyphEmbeddings(
                main=main_embedding,
                positive=positive_embedding,
                negative=negative_embedding,
            ),
            metrics,
        )
        return metrics

    def class_center_regularization(
        self,
        *,
        main_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: torch.Tensor,
        main_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.class_centers is None:
            zero = main_embedding.new_zeros(())
            return zero, {}

        main_indices = self._class_indices(main_ids)
        negative_indices = self._class_indices(negative_ids)
        main_centers = self._normalized_class_centers(main_indices)
        negative_centers = self._normalized_class_centers(negative_indices)
        main_center_distance = F.pairwise_distance(main_embedding, main_centers).mean()
        positive_center_distance = F.pairwise_distance(positive_embedding, main_centers).mean()
        negative_center_distance = F.pairwise_distance(
            negative_embedding,
            negative_centers,
        ).mean()

        raw_loss = (
            F.mse_loss(main_embedding, main_centers)
            + F.mse_loss(positive_embedding, main_centers)
            + F.mse_loss(negative_embedding, negative_centers)
        ) / 3.0
        weighted_loss = self.config.class_center_weight * raw_loss

        batch_embeddings = torch.cat(
            [main_embedding, positive_embedding, negative_embedding],
            dim=0,
        )
        batch_indices = torch.cat([main_indices, main_indices, negative_indices], dim=0)
        all_centers = self._normalized_all_class_centers()
        center_predictions = torch.cdist(
            F.normalize(batch_embeddings, dim=1),
            all_centers,
        ).argmin(dim=1)
        center_accuracy = (center_predictions == batch_indices).float().mean()

        return weighted_loss, {
            "class_center_loss": raw_loss,
            "weighted_class_center_loss": weighted_loss,
            "class_center_distance": (
                main_center_distance
                + positive_center_distance
                + negative_center_distance
            ) / 3.0,
            "main_class_center_distance": main_center_distance,
            "positive_class_center_distance": positive_center_distance,
            "negative_class_center_distance": negative_center_distance,
            "batch_class_center_accuracy": center_accuracy,
        }

    def _class_indices(self, glyph_ids: torch.Tensor) -> torch.Tensor:
        indices = [
            self.glyph_id_to_class_index[int(glyph_id)]
            for glyph_id in glyph_ids.detach().cpu().tolist()
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _normalized_class_centers(self, indices: torch.Tensor) -> torch.Tensor:
        if self.class_centers is None:
            raise RuntimeError("class centers are disabled")
        return F.normalize(self.class_centers(indices), dim=1)

    def _normalized_all_class_centers(self) -> torch.Tensor:
        if self.class_centers is None:
            raise RuntimeError("class centers are disabled")
        return F.normalize(self.class_centers.weight, dim=1)

    def maybe_validate(self, step: int) -> dict[str, float]:
        if self.validation_episode is None:
            return {}
        if self.config.validation_every < 1:
            return {}
        if step % self.config.validation_every != 0 and step != self.config.steps:
            return {}
        metrics = evaluate_classification_accuracy(
            self.model,
            self.validation_episode,
            device=self.device,
            batch_size=self.config.validation_batch_size,
        )
        if self.class_centers is not None and self.class_center_validation:
            metrics.update(
                evaluate_class_center_accuracy(
                    self.model,
                    self.validation_episode,
                    class_centers=self.class_centers,
                    glyph_id_to_class_index=self.glyph_id_to_class_index,
                    device=self.device,
                    batch_size=self.config.validation_batch_size,
                )
            )
        return metrics

    def save_checkpoint(
        self,
        path: Path,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        torch.save(
            {
                "step": step,
                "metrics": metrics,
                "config": asdict(self.config),
                "model_config": self.model_config,
                "model_state_dict": self.model.state_dict(),
                "class_center_state_dict": (
                    self.class_centers.state_dict()
                    if self.class_centers is not None
                    else None
                ),
                "class_center_glyph_ids": self.class_glyph_ids,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )


def three_point_loss(
    main_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive_distance = F.pairwise_distance(main_embedding, positive_embedding)
    main_negative_distance = F.pairwise_distance(main_embedding, negative_embedding)
    positive_negative_distance = F.pairwise_distance(positive_embedding, negative_embedding)

    pull_positive = positive_distance.square().mean()
    push_main_negative = F.relu(margin - main_negative_distance).square().mean()
    push_positive_negative = F.relu(margin - positive_negative_distance).square().mean()
    loss = pull_positive + push_main_negative + push_positive_negative

    return loss, {
        "positive_distance": positive_distance.mean(),
        "main_negative_distance": main_negative_distance.mean(),
        "positive_negative_distance": positive_negative_distance.mean(),
        "pull_positive": pull_positive,
        "push_main_negative": push_main_negative,
        "push_positive_negative": push_positive_negative,
    }


def train_dataset(
    config: TrainerConfig,
    *,
    dataset_name: str,
    data_root: str,
    download: bool,
    omniglot_split: str,
    image_size: int,
) -> Path:
    torch.manual_seed(config.seed)
    if dataset_name == "mnist":
        dataset = MnistGlyphDataset(data_root, train=True, download=download, seed=config.seed)
        validation_dataset = MnistGlyphDataset(
            data_root,
            train=False,
            download=download,
            seed=config.seed + 1,
        )
    elif dataset_name == "omniglot":
        dataset = OmniglotGlyphDataset(
            data_root,
            split=omniglot_split,
            download=download,
            image_size=image_size,
            seed=config.seed,
        )
        validation_split = _omniglot_validation_split(omniglot_split)
        validation_dataset = OmniglotGlyphDataset(
            data_root,
            split=validation_split,
            download=download,
            image_size=image_size,
            seed=config.seed + 1,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    train_ctx: TrainContext = {}
    plugin = build_training_plugin(config)
    sampler = plugin.build_sampler(dataset, config, train_ctx)
    model_config = model_config_from_trainer_config(config, image_size=image_size)
    model = SiameseGlyphNet(**model_config)
    trainer = SiameseTrainer(
        model=model,
        sampler=sampler,
        plugin=plugin,
        config=config,
        model_config=model_config,
        train_ctx=train_ctx,
        validation_dataset=validation_dataset,
        train_glyph_ids=dataset.glyph_ids,
        class_center_validation=(
            dataset_name == "mnist"
            or (dataset_name == "omniglot" and omniglot_split == "all")
        ),
    )
    return trainer.train()


def build_validation_episode(
    dataset: GlyphDataset,
    *,
    support_per_class: int,
) -> ValidationEpisode:
    if support_per_class < 1:
        raise ValueError("support_per_class must be >= 1")

    support_images = []
    query_images = []
    query_labels = []
    glyph_ids = tuple(dataset.glyph_ids)

    for class_index, glyph_id in enumerate(glyph_ids):
        images = _as_image_batch(dataset.images_for_glyph(glyph_id))
        if len(images) <= support_per_class:
            raise ValueError(
                f"validation glyph {glyph_id} has {len(images)} images, "
                f"but support_per_class={support_per_class} requires at least "
                f"{support_per_class + 1}"
            )
        support_images.append(images[:support_per_class])
        queries = images[support_per_class:]
        query_images.append(queries)
        query_labels.append(
            torch.full((len(queries),), class_index, dtype=torch.long)
        )

    return ValidationEpisode(
        support_images=torch.stack(support_images),
        query_images=torch.cat(query_images),
        query_labels=torch.cat(query_labels),
        glyph_ids=glyph_ids,
    )


def evaluate_classification_accuracy(
    model: nn.Module,
    episode: ValidationEpisode,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        class_count, support_per_class = episode.support_images.shape[:2]
        support_embeddings = _batched_embeddings(
            model,
            episode.support_images.flatten(0, 1),
            device=device,
            batch_size=batch_size,
        )
        prototype_tensor = support_embeddings.reshape(
            class_count,
            support_per_class,
            -1,
        ).mean(dim=1)
        prototype_tensor = F.normalize(prototype_tensor, dim=1).to(device)

        correct = 0
        total = 0
        for start in range(0, len(episode.query_images), batch_size):
            end = start + batch_size
            query_images = episode.query_images[start:end].to(device)
            query_labels = episode.query_labels[start:end].to(device)
            query_embeddings = F.normalize(model(query_images), dim=1)
            distances = torch.cdist(query_embeddings, prototype_tensor)
            predictions = distances.argmin(dim=1)
            correct += (predictions == query_labels).sum().item()
            total += int(predictions.numel())

    if was_training:
        model.train()

    return {
        "validation_accuracy": correct / total if total else 0.0,
        "validation_correct": float(correct),
        "validation_total": float(total),
    }


def evaluate_class_center_accuracy(
    model: nn.Module,
    episode: ValidationEpisode,
    *,
    class_centers: nn.Embedding,
    glyph_id_to_class_index: dict[int, int],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not all(glyph_id in glyph_id_to_class_index for glyph_id in episode.glyph_ids):
        return {}

    was_training = model.training
    model.eval()

    with torch.no_grad():
        episode_center_indices = torch.tensor(
            [glyph_id_to_class_index[glyph_id] for glyph_id in episode.glyph_ids],
            dtype=torch.long,
            device=device,
        )
        center_tensor = F.normalize(class_centers(episode_center_indices), dim=1)
        correct = 0
        total = 0
        for start in range(0, len(episode.query_images), batch_size):
            end = start + batch_size
            query_images = episode.query_images[start:end].to(device)
            query_labels = episode.query_labels[start:end].to(device)
            query_embeddings = F.normalize(model(query_images), dim=1)
            distances = torch.cdist(query_embeddings, center_tensor)
            predictions = distances.argmin(dim=1)
            correct += (predictions == query_labels).sum().item()
            total += int(predictions.numel())

    if was_training:
        model.train()

    return {
        "validation_class_center_accuracy": correct / total if total else 0.0,
        "validation_class_center_correct": float(correct),
        "validation_class_center_total": float(total),
    }


def _batched_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(images), batch_size):
        embeddings.append(model(images[start : start + batch_size].to(device)).cpu())
    return torch.cat(embeddings)


def _as_image_batch(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().clone().float()
    if images.ndim == 3:
        return images.unsqueeze(1)
    if images.ndim == 4:
        return images
    raise ValueError(f"expected image tensor with 3 or 4 dims, got {images.shape}")


def _omniglot_validation_split(train_split: str) -> str:
    if train_split == "background":
        return "evaluation"
    if train_split == "evaluation":
        return "background"
    return "all"


def build_dataset(
    *,
    dataset_name: str,
    data_root: str,
    download: bool,
    omniglot_split: str,
    image_size: int,
    seed: int,
) -> MnistGlyphDataset | OmniglotGlyphDataset:
    if dataset_name == "mnist":
        return MnistGlyphDataset(data_root, train=True, download=download, seed=seed)
    if dataset_name == "omniglot":
        return OmniglotGlyphDataset(
            data_root,
            split=omniglot_split,
            download=download,
            image_size=image_size,
            seed=seed,
        )
    raise ValueError(f"unknown dataset: {dataset_name}")


def save_augmented_samples(
    config: TrainerConfig,
    *,
    dataset_name: str,
    data_root: str,
    download: bool,
    omniglot_split: str,
    image_size: int,
    sample_count: int,
    output_dir: str,
) -> Path:
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    torch.manual_seed(config.seed)
    dataset = build_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        download=download,
        omniglot_split=omniglot_split,
        image_size=image_size,
        seed=config.seed,
    )
    augment = RandomGlyphAugment(
        AugmentationConfig(
            enabled=config.augment,
            rotation_degrees=config.rotation_degrees,
            translate=config.translate,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            flip_probability=config.flip_probability,
            shear_degrees=config.shear_degrees,
        )
    )

    generator = torch.Generator().manual_seed(config.seed)
    glyph_ids = torch.randint(
        low=0,
        high=len(dataset.glyph_ids),
        size=(sample_count,),
        generator=generator,
    )
    selected_glyph_ids = [dataset.glyph_ids[index] for index in glyph_ids.tolist()]
    originals = torch.stack(
        [
            _as_image_batch_item(dataset.random_image(glyph_id))
            for glyph_id in selected_glyph_ids
        ]
    )
    augmented = augment(originals)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for index, glyph_id in enumerate(selected_glyph_ids):
        stem = f"{index:04d}_class_{glyph_id}"
        _save_tensor_image(originals[index], output_path / f"{stem}_original.png")
        _save_tensor_image(augmented[index], output_path / f"{stem}_augmented.png")
        _save_tensor_image(
            torch.cat([originals[index], augmented[index]], dim=2),
            output_path / f"{stem}_compare.png",
        )
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "omniglot"), default="mnist")
    parser.add_argument("--data-root")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-augmented-samples", type=int)
    parser.add_argument("--augment-output-dir", default="outputs/augmented_samples")
    parser.add_argument(
        "--omniglot-split",
        choices=("background", "evaluation", "all"),
        default="background",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Input image size. Defaults to 56 for the fixed conv_attention model.",
    )
    parser.add_argument("--steps", type=int, default=TrainerConfig.steps)
    parser.add_argument("--steps-per-epoch", type=int, default=TrainerConfig.steps_per_epoch)
    parser.add_argument("--classes-per-batch", type=int, default=TrainerConfig.classes_per_batch)
    parser.add_argument("--learning-rate", type=float, default=TrainerConfig.learning_rate)
    parser.add_argument("--margin", type=float, default=TrainerConfig.margin)
    parser.add_argument("--checkpoint-path", default=TrainerConfig.checkpoint_path)
    parser.add_argument("--aim-repo", default=TrainerConfig.aim_repo)
    parser.add_argument("--aim-experiment", default=TrainerConfig.aim_experiment)
    parser.add_argument("--device", default=TrainerConfig.device)
    parser.add_argument("--log-every", type=int, default=TrainerConfig.log_every)
    parser.add_argument(
        "--validation-every",
        type=int,
        default=TrainerConfig.validation_every,
        help="Evaluate validation classification accuracy every N steps. 0 disables.",
    )
    parser.add_argument(
        "--validation-support-per-class",
        type=int,
        default=TrainerConfig.validation_support_per_class,
        help="Fixed validation prototype images per class. All remaining validation images are evaluated.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=TrainerConfig.validation_batch_size,
        help="Batch size for full validation evaluation.",
    )
    parser.add_argument(
        "--class-center-weight",
        type=float,
        default=TrainerConfig.class_center_weight,
        help="Weight for trainable per-class center regularization. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=TrainerConfig.seed)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=TrainerConfig.rotation_degrees)
    parser.add_argument("--translate", type=float, default=TrainerConfig.translate)
    parser.add_argument("--scale-min", type=float, default=TrainerConfig.scale_min)
    parser.add_argument("--scale-max", type=float, default=TrainerConfig.scale_max)
    parser.add_argument("--flip-probability", type=float, default=TrainerConfig.flip_probability)
    parser.add_argument("--shear-degrees", type=float, default=TrainerConfig.shear_degrees)
    parser.add_argument(
        "--plugin",
        choices=("class-std", "uniform"),
        default=TrainerConfig.sampler_plugin,
    )
    parser.add_argument(
        "--distribution-ema-decay",
        type=float,
        default=TrainerConfig.distribution_ema_decay,
    )
    parser.add_argument(
        "--nearest-fallback-count",
        type=int,
        default=TrainerConfig.nearest_fallback_count,
    )
    parser.add_argument(
        "--min-class-candidates",
        type=int,
        default=TrainerConfig.min_class_candidates,
        help="Top up each class candidate list with nearest classes until this count. 0 disables.",
    )
    parser.add_argument(
        "--j",
        dest="class_intersection_workers",
        type=int,
        default=TrainerConfig.class_intersection_workers,
        help="Class intersection worker processes. -1 picks a capped auto value, 1 is sequential.",
    )
    parser.add_argument(
        "--model-architecture",
        choices=("conv_attention", "residual", "legacy"),
        default=TrainerConfig.model_architecture,
    )
    parser.add_argument("--embedding-dim", type=int, default=TrainerConfig.embedding_dim)
    parser.add_argument("--base-channels", type=int, default=TrainerConfig.base_channels)
    parser.add_argument("--attention-heads", type=int, default=TrainerConfig.attention_heads)
    parser.add_argument("--attention-layers", type=int, default=TrainerConfig.attention_layers)
    parser.add_argument(
        "--projection-hidden-dim",
        type=int,
        default=TrainerConfig.projection_hidden_dim,
    )
    parser.add_argument("--model-dropout", type=float, default=TrainerConfig.model_dropout)
    args = parser.parse_args(argv)

    config = TrainerConfig(
        steps=args.steps,
        steps_per_epoch=args.steps_per_epoch,
        classes_per_batch=args.classes_per_batch,
        learning_rate=args.learning_rate,
        margin=args.margin,
        checkpoint_path=args.checkpoint_path,
        aim_repo=args.aim_repo,
        aim_experiment=args.aim_experiment,
        device=args.device,
        log_every=args.log_every,
        validation_every=args.validation_every,
        validation_support_per_class=args.validation_support_per_class,
        validation_batch_size=args.validation_batch_size,
        class_center_weight=args.class_center_weight,
        seed=args.seed,
        augment=not args.no_augment,
        rotation_degrees=args.rotation_degrees,
        translate=args.translate,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        flip_probability=args.flip_probability,
        shear_degrees=args.shear_degrees,
        sampler_plugin=args.plugin,
        distribution_ema_decay=args.distribution_ema_decay,
        nearest_fallback_count=args.nearest_fallback_count,
        min_class_candidates=args.min_class_candidates,
        class_intersection_workers=args.class_intersection_workers,
        model_architecture=args.model_architecture,
        embedding_dim=args.embedding_dim,
        base_channels=args.base_channels,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        projection_hidden_dim=args.projection_hidden_dim,
        model_dropout=args.model_dropout,
    )
    data_root = args.data_root
    if data_root is None:
        data_root = "datasets/omniglot" if args.dataset == "omniglot" else "datasets/mnist"
    image_size = args.image_size
    if image_size is None:
        image_size = 56
    if args.save_augmented_samples is not None:
        output_path = save_augmented_samples(
            config,
            dataset_name=args.dataset,
            data_root=data_root,
            download=args.download,
            omniglot_split=args.omniglot_split,
            image_size=image_size,
            sample_count=args.save_augmented_samples,
            output_dir=args.augment_output_dir,
        )
        print(f"saved augmented samples: {output_path}")
        return
    checkpoint_path = train_dataset(
        config,
        dataset_name=args.dataset,
        data_root=data_root,
        download=args.download,
        omniglot_split=args.omniglot_split,
        image_size=image_size,
    )
    print(f"best checkpoint: {checkpoint_path}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_image_batch_item(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().clone().float()
    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3:
        return image
    raise ValueError(f"expected image tensor with 2 or 3 dims, got {image.shape}")


def _save_tensor_image(image: torch.Tensor, path: Path) -> None:
    image = image.detach().cpu().float()
    if image.ndim == 3:
        image = image.squeeze(0)
    if image.ndim != 2:
        raise ValueError(f"expected image tensor with 2 or 3 dims, got {image.shape}")
    array = (image.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def build_training_plugin(config: TrainerConfig) -> TrainingPlugin:
    if config.sampler_plugin == "class-std":
        return ClassStdSamplerPlugin(
            distribution_ema_decay=config.distribution_ema_decay,
            nearest_fallback_count=config.nearest_fallback_count,
            min_class_candidates=config.min_class_candidates,
            intersection_workers=config.class_intersection_workers,
        )
    if config.sampler_plugin == "uniform":
        return UniformSamplerPlugin()
    raise ValueError(f"unknown sampler plugin: {config.sampler_plugin}")


def _format_epoch_metrics(epoch: int, step: int, metrics: dict[str, float]) -> str:
    return (
        f"epoch={epoch} step={step} "
        f"embeddings={metrics.get('epoch_collected_embeddings', 0):.0f} "
        f"classes={metrics.get('epoch_classes_with_stats', 0):.0f} "
        f"candidates=min/mean/max "
        f"{metrics.get('epoch_candidate_min', 0):.0f}/"
        f"{metrics.get('epoch_candidate_mean', 0):.2f}/"
        f"{metrics.get('epoch_candidate_max', 0):.0f} "
        f"fallback={metrics.get('epoch_fallback_classes', 0):.0f} "
        f"min_cand={metrics.get('epoch_min_class_candidates', 0):.0f} "
        f"topup={metrics.get('epoch_candidate_topup_classes', 0):.0f}/"
        f"{metrics.get('epoch_candidate_topup_added', 0):.0f} "
        f"center_norm={metrics.get('epoch_center_norm_mean', 0):.3f} "
        f"std_norm={metrics.get('epoch_std_norm_mean', 0):.3f} "
        f"cand_dist={metrics.get('epoch_candidate_center_distance_mean', 0):.3f} "
        f"near_dist={metrics.get('epoch_nearest_center_distance_mean', 0):.3f} "
        f"bbox={metrics.get('epoch_bbox_intersections', 0):.0f}/"
        f"{metrics.get('epoch_bbox_pair_tests', 0):.0f} "
        f"workers={metrics.get('epoch_intersection_workers', 1):.0f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SiameseTrainer",
    "TrainerConfig",
    "build_dataset",
    "build_training_plugin",
    "save_augmented_samples",
    "three_point_loss",
    "train_dataset",
]
