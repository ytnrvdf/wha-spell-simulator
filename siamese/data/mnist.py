from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

from data import GlyphDataset


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class MnistGlyphDataset(GlyphDataset):
    """MNIST digits exposed through the GlyphDataset abstraction."""

    def __init__(
        self,
        root: str | Path = "datasets/mnist",
        *,
        train: bool = True,
        download: bool = False,
        normalize: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(range(10))
        self.root = Path(root)
        self.train = train
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        if download:
            self.download()

        split = "train" if train else "test"
        images_path = self.root / MNIST_FILES[f"{split}_images"]
        labels_path = self.root / MNIST_FILES[f"{split}_labels"]

        images = _read_idx_images(images_path)
        labels = _read_idx_labels(labels_path)
        if len(images) != len(labels):
            raise ValueError("MNIST images and labels have different lengths")

        self._images_by_glyph = {
            glyph_id: images[labels == glyph_id] for glyph_id in self.glyph_ids
        }
        missing = [
            glyph_id
            for glyph_id, glyph_images in self._images_by_glyph.items()
            if len(glyph_images) == 0
        ]
        if missing:
            raise ValueError(f"MNIST split is missing glyph ids: {missing}")

    def download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for filename in MNIST_FILES.values():
            target = self.root / filename
            if target.exists():
                continue
            url = f"{MNIST_BASE_URL}/{filename}"
            urllib.request.urlretrieve(url, target)

    def random_image(self, glyph_id: int) -> torch.Tensor:
        glyph_id = int(glyph_id)
        if glyph_id not in self._images_by_glyph:
            raise KeyError(f"unknown glyph_id: {glyph_id}")

        glyph_images = self._images_by_glyph[glyph_id]
        image = glyph_images[self.rng.integers(0, len(glyph_images))]
        return self._image_to_tensor(image)

    def images_for_glyph(self, glyph_id: int) -> torch.Tensor:
        glyph_id = int(glyph_id)
        if glyph_id not in self._images_by_glyph:
            raise KeyError(f"unknown glyph_id: {glyph_id}")

        glyph_images = self._images_by_glyph[glyph_id]
        tensor = torch.from_numpy(glyph_images.astype(np.float32, copy=False))
        if self.normalize:
            tensor = tensor / 255.0
        return tensor

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.astype(np.float32, copy=False))
        if self.normalize:
            tensor = tensor / 255.0
        return tensor


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count)


__all__ = ["MnistGlyphDataset"]
