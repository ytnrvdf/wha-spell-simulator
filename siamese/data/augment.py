from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    rotation_degrees: float = 20.0
    translate: float = 0.18
    scale_min: float = 0.85
    scale_max: float = 1.15
    flip_probability: float = 0.0
    shear_degrees: float = 10.0


class RandomGlyphAugment:
    def __init__(self, config: AugmentationConfig) -> None:
        if config.translate < 0:
            raise ValueError("translate must be >= 0")
        if config.scale_min <= 0 or config.scale_max <= 0:
            raise ValueError("scale range must be positive")
        if config.scale_min > config.scale_max:
            raise ValueError("scale_min must be <= scale_max")
        if not 0.0 <= config.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        self.config = config

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return images
        if images.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got {images.shape}")

        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        rotation = _uniform(
            batch_size,
            -self.config.rotation_degrees,
            self.config.rotation_degrees,
            device=device,
            dtype=dtype,
        ) * (pi / 180.0)
        shear_x = _uniform(
            batch_size,
            -self.config.shear_degrees,
            self.config.shear_degrees,
            device=device,
            dtype=dtype,
        ) * (pi / 180.0)
        shear_y = _uniform(
            batch_size,
            -self.config.shear_degrees,
            self.config.shear_degrees,
            device=device,
            dtype=dtype,
        ) * (pi / 180.0)
        scale = _uniform(
            batch_size,
            self.config.scale_min,
            self.config.scale_max,
            device=device,
            dtype=dtype,
        )
        flip = torch.where(
            torch.rand(batch_size, device=device) < self.config.flip_probability,
            -torch.ones(batch_size, device=device, dtype=dtype),
            torch.ones(batch_size, device=device, dtype=dtype),
        )
        translate_x = _uniform(
            batch_size,
            -self.config.translate,
            self.config.translate,
            device=device,
            dtype=dtype,
        )
        translate_y = _uniform(
            batch_size,
            -self.config.translate,
            self.config.translate,
            device=device,
            dtype=dtype,
        )

        theta = _affine_matrices(
            rotation=rotation,
            scale=scale,
            flip=flip,
            shear_x=shear_x,
            shear_y=shear_y,
            translate_x=translate_x,
            translate_y=translate_y,
        )
        grid = F.affine_grid(theta, size=images.shape, align_corners=False)
        return F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).clamp(0.0, 1.0)


def _affine_matrices(
    *,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    flip: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    translate_x: torch.Tensor,
    translate_y: torch.Tensor,
) -> torch.Tensor:
    cos = rotation.cos()
    sin = rotation.sin()
    tan_x = shear_x.tan()
    tan_y = shear_y.tan()

    rotate = torch.stack(
        [
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin, cos], dim=1),
        ],
        dim=1,
    )
    shear = torch.stack(
        [
            torch.stack([torch.ones_like(tan_x), tan_x], dim=1),
            torch.stack([tan_y, torch.ones_like(tan_y)], dim=1),
        ],
        dim=1,
    )
    scale_flip = torch.stack(
        [
            torch.stack([scale * flip, torch.zeros_like(scale)], dim=1),
            torch.stack([torch.zeros_like(scale), scale], dim=1),
        ],
        dim=1,
    )
    linear = rotate @ shear @ scale_flip
    theta = torch.zeros(rotation.shape[0], 2, 3, device=rotation.device, dtype=rotation.dtype)
    theta[:, :, :2] = linear
    theta[:, 0, 2] = translate_x
    theta[:, 1, 2] = translate_y
    return theta


def _uniform(
    size: int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(size, device=device, dtype=dtype).uniform_(low, high)


__all__ = ["AugmentationConfig", "RandomGlyphAugment"]
