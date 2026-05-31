from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class GlyphDataset(ABC):
    """Base source of glyph images for one-shot learning experiments."""

    def __init__(self, glyph_ids: Sequence[int]) -> None:
        if not glyph_ids:
            raise ValueError("glyph_ids must not be empty")
        self.glyph_ids = tuple(int(glyph_id) for glyph_id in glyph_ids)

    @abstractmethod
    def random_image(self, glyph_id: int) -> torch.Tensor:
        """Return a random glyph image tensor with shape (height, width)."""

    def images_for_glyph(self, glyph_id: int) -> torch.Tensor:
        """Return all images for one glyph with shape (count, height, width)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose full glyph images")


__all__ = ["GlyphDataset"]
