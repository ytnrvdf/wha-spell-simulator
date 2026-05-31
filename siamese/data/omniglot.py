from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import GlyphDataset


OMNIGLOT_BASE_URL = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
OMNIGLOT_SPLITS = {
    "background": "images_background",
    "evaluation": "images_evaluation",
}


class OmniglotGlyphDataset(GlyphDataset):
    """Omniglot characters exposed through the GlyphDataset abstraction."""

    def __init__(
        self,
        root: str | Path = "datasets/omniglot",
        *,
        split: str = "background",
        download: bool = False,
        image_size: int = 56,
        normalize: bool = True,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        if split not in {"background", "evaluation", "all"}:
            raise ValueError("split must be one of: background, evaluation, all")
        if image_size < 1:
            raise ValueError("image_size must be >= 1")

        split_dirs = self._prepare_split_dirs(download=download)
        images_by_glyph, glyph_names = self._index_images(split_dirs)
        if not images_by_glyph:
            raise ValueError(f"no Omniglot glyph images found in {self.root}")

        self._images_by_glyph = images_by_glyph
        self.glyph_names = glyph_names
        super().__init__(sorted(images_by_glyph))

    def random_image(self, glyph_id: int) -> torch.Tensor:
        glyph_id = int(glyph_id)
        if glyph_id not in self._images_by_glyph:
            raise KeyError(f"unknown glyph_id: {glyph_id}")

        paths = self._images_by_glyph[glyph_id]
        path = paths[self.rng.integers(0, len(paths))]
        return self._load_image(path)

    def images_for_glyph(self, glyph_id: int) -> torch.Tensor:
        glyph_id = int(glyph_id)
        if glyph_id not in self._images_by_glyph:
            raise KeyError(f"unknown glyph_id: {glyph_id}")

        return torch.stack(
            [self._load_image(path) for path in self._images_by_glyph[glyph_id]]
        )

    def _prepare_split_dirs(self, *, download: bool) -> list[Path]:
        split_names = (
            tuple(OMNIGLOT_SPLITS.values())
            if self.split == "all"
            else (OMNIGLOT_SPLITS[self.split],)
        )
        split_dirs = []
        for split_name in split_names:
            split_dir = self.root / "python" / split_name
            if not split_dir.exists():
                zip_path = self.root / "python" / f"{split_name}.zip"
                if download and not zip_path.exists():
                    download_omniglot_zip(zip_path, split_name)
                if zip_path.exists():
                    extract_omniglot_zip(zip_path, zip_path.parent)
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"missing {split_dir}; clone Omniglot or pass download=True"
                )
            split_dirs.append(split_dir)
        return split_dirs

    def _index_images(self, split_dirs: list[Path]) -> tuple[dict[int, list[Path]], dict[int, str]]:
        images_by_glyph: dict[int, list[Path]] = {}
        glyph_names: dict[int, str] = {}
        next_id = 0

        for split_dir in split_dirs:
            for character_dir in sorted(split_dir.glob("*/*")):
                if not character_dir.is_dir():
                    continue
                paths = sorted(character_dir.glob("*.png"))
                if not paths:
                    continue
                glyph_id = next_id
                next_id += 1
                alphabet = character_dir.parent.name
                character = character_dir.name
                glyph_names[glyph_id] = f"{alphabet}/{character}"
                images_by_glyph[glyph_id] = paths

        return images_by_glyph, glyph_names

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        array = np.asarray(image, dtype=np.float32)
        array = 255.0 - array
        if self.normalize:
            array = array / 255.0
        return torch.from_numpy(array)


def download_omniglot_zip(zip_path: Path, split_name: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{OMNIGLOT_BASE_URL}/{split_name}.zip"
    urllib.request.urlretrieve(url, zip_path)


def extract_omniglot_zip(zip_path: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or zip_path.parent
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


__all__ = ["OmniglotGlyphDataset", "download_omniglot_zip", "extract_omniglot_zip"]
