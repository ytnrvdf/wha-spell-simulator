from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_CONFIG_KEYS = {
    "architecture",
    "in_channels",
    "image_size",
    "embedding_dim",
    "base_channels",
    "attention_heads",
    "attention_layers",
    "projection_hidden_dim",
    "dropout",
    "normalize_embeddings",
}


class SiameseGlyphNet(nn.Module):
    """CNN encoder that maps a glyph image to a latent embedding."""

    def __init__(
        self,
        *,
        architecture: str = "conv_attention",
        in_channels: int = 1,
        image_size: int = 28,
        embedding_dim: int = 64,
        base_channels: int = 32,
        attention_heads: int = 4,
        attention_layers: int = 2,
        projection_hidden_dim: int = 512,
        dropout: float = 0.1,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        if architecture not in {"conv_attention", "legacy", "residual"}:
            raise ValueError("architecture must be one of: conv_attention, legacy, residual")
        if architecture == "conv_attention" and image_size != 56:
            raise ValueError("conv_attention architecture requires image_size=56")
        if projection_hidden_dim < 1:
            raise ValueError("projection_hidden_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.architecture = architecture
        self.normalize_embeddings = normalize_embeddings
        feature_channels = base_channels * 4

        if architecture == "legacy":
            feature_size = image_size // 4
            self.backbone = nn.Sequential(
                _conv_block(in_channels, base_channels),
                nn.MaxPool2d(2),
                _conv_block(base_channels, base_channels * 2),
                nn.MaxPool2d(2),
                _conv_block(base_channels * 2, feature_channels),
            )
            self.projection = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(feature_channels * feature_size * feature_size, embedding_dim),
            )
        elif architecture == "conv_attention":
            feature_size = image_size // 8
            self.backbone = nn.Sequential(
                _conv_block(in_channels, base_channels),
                _conv_block(base_channels, base_channels),
                nn.MaxPool2d(2),
                _conv_block(base_channels, base_channels * 2),
                _conv_block(base_channels * 2, base_channels * 2),
                nn.MaxPool2d(2),
                _conv_block(base_channels * 2, feature_channels),
                _conv_block(feature_channels, feature_channels),
                nn.MaxPool2d(2),
            )
            self.projection = SpatialProjectionHead(
                channels=feature_channels,
                feature_size=feature_size,
                embedding_dim=embedding_dim,
                hidden_dim=projection_hidden_dim,
                dropout=dropout,
            )
        else:
            feature_size = image_size // 4
            self.backbone = nn.Sequential(
                _conv_block(in_channels, base_channels),
                ResidualStage(base_channels, base_channels, blocks=2, stride=1),
                ResidualStage(base_channels, base_channels * 2, blocks=2, stride=2),
                ResidualStage(base_channels * 2, feature_channels, blocks=2, stride=2),
                ResidualStage(feature_channels, feature_channels, blocks=2, stride=1),
            )
            self.projection = SpatialProjectionHead(
                channels=feature_channels,
                feature_size=feature_size,
                embedding_dim=embedding_dim,
                hidden_dim=projection_hidden_dim,
                dropout=dropout,
            )

        self.self_attention = SpatialSelfAttention(
            feature_channels,
            feature_size=feature_size,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=dropout,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        features = self.self_attention(features)
        embedding = self.projection(features)
        if self.normalize_embeddings:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class ResidualStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, blocks: int, stride: int) -> None:
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        layers: list[nn.Module] = [
            ResidualConvBlock(in_channels, out_channels, stride=stride),
        ]
        for _ in range(blocks - 1):
            layers.append(ResidualConvBlock(out_channels, out_channels, stride=1))
        super().__init__(*layers)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(features) + self.shortcut(features))


class SpatialSelfAttention(nn.Module):
    """All-to-all self-attention over spatial feature tokens."""

    def __init__(
        self,
        channels: int,
        *,
        feature_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if feature_size < 1:
            raise ValueError("feature_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_size = feature_size
        self.register_buffer("position", build_2d_sincos_position(feature_size, channels))
        self.norm = nn.LayerNorm(channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        if height != self.feature_size or width != self.feature_size:
            raise ValueError(
                f"expected feature map {self.feature_size}x{self.feature_size}, "
                f"got {height}x{width}"
            )

        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.position
        tokens = self.norm(tokens)
        tokens = self.encoder(tokens)
        attended = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.output_norm(attended)


class SpatialProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        feature_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pooled_channels = channels * 2
        flattened_channels = channels * feature_size * feature_size
        self.flatten = nn.Flatten(start_dim=1)
        self.projection = nn.Sequential(
            nn.Linear(flattened_channels + pooled_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(features)
        avg = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        max_values = F.adaptive_max_pool2d(features, output_size=1).flatten(1)
        return self.projection(torch.cat([flat, avg, max_values], dim=1))


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def build_2d_sincos_position(feature_size: int, channels: int) -> torch.Tensor:
    if feature_size < 1:
        raise ValueError("feature_size must be >= 1")
    if channels < 4:
        raise ValueError("channels must be >= 4")

    y, x = torch.meshgrid(
        torch.arange(feature_size, dtype=torch.float32),
        torch.arange(feature_size, dtype=torch.float32),
        indexing="ij",
    )
    y = y.flatten()
    x = x.flatten()

    y_channels = channels // 2
    x_channels = channels - y_channels
    position = torch.cat(
        [
            _build_1d_sincos_position(y, y_channels),
            _build_1d_sincos_position(x, x_channels),
        ],
        dim=1,
    )
    return position.unsqueeze(0)


def _build_1d_sincos_position(values: torch.Tensor, channels: int) -> torch.Tensor:
    frequencies = torch.arange((channels + 1) // 2, dtype=torch.float32)
    frequencies = 1.0 / (10_000 ** (frequencies / max(1, frequencies.numel())))
    angles = values[:, None] * frequencies[None, :]
    position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
    return position[:, :channels]


def model_config_from_trainer_config(config: Any, *, image_size: int) -> dict[str, Any]:
    return {
        "architecture": getattr(config, "model_architecture", "conv_attention"),
        "image_size": image_size,
        "embedding_dim": getattr(config, "embedding_dim", 64),
        "base_channels": getattr(config, "base_channels", 32),
        "attention_heads": getattr(config, "attention_heads", 4),
        "attention_layers": getattr(config, "attention_layers", 2),
        "projection_hidden_dim": getattr(config, "projection_hidden_dim", 512),
        "dropout": getattr(config, "model_dropout", 0.1),
        "normalize_embeddings": getattr(config, "normalize_embeddings", True),
    }


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = dict(checkpoint.get("model_config") or {})
    if config:
        return {key: value for key, value in config.items() if key in MODEL_CONFIG_KEYS}

    state_dict = checkpoint.get("model_state_dict", {})
    architecture = "legacy" if "projection.1.weight" in state_dict else "residual"
    return {"architecture": architecture}


def build_siamese_model(config: dict[str, Any] | None = None) -> SiameseGlyphNet:
    kwargs = {
        key: value
        for key, value in (config or {}).items()
        if key in MODEL_CONFIG_KEYS
    }
    return SiameseGlyphNet(**kwargs)


__all__ = [
    "ResidualConvBlock",
    "ResidualStage",
    "SiameseGlyphNet",
    "SpatialProjectionHead",
    "SpatialSelfAttention",
    "build_2d_sincos_position",
    "build_siamese_model",
    "model_config_from_checkpoint",
    "model_config_from_trainer_config",
]
