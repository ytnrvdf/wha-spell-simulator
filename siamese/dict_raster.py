"""Rasterize dictionary stroke templates into glyph images for the siamese net.

The browser app stores each sigil/sign as a `strokeTemplate.strokes` array of
normalized polylines in the unit box ([0, 1] x [0, 1], already fit to bounds).
This module renders those polylines the same way the siamese training pipeline
expects its inputs: white ink on a black background, the glyph centered in a
square frame with a fixed padding margin (mirrors
`render_glyph_for_recognition` / `pad_to_square` in the reference parser).

Keeping this rasterization in one place is important: the JS recognizer must
reproduce it pixel-for-pixel, and the parity test compares the two.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# Fraction of the frame left as empty margin around the glyph on each side.
# Mirrors pad_to_square(padding_ratio=0.28): a glyph of side S is drawn into a
# canvas of side S * (1 + 2 * 0.28), so the glyph fills ~64% of the frame.
PADDING_RATIO = 0.28

# Stroke width as a fraction of the FINAL frame size. 0.045 -> ~2.5px pen at 56.
STROKE_WIDTH_RATIO = 0.045

# Supersampling factor: strokes are stamped on a (size * SUPERSAMPLE) canvas and
# box-averaged down to `size`. This anti-aliases thin pen lines so fine glyph
# detail survives the 56px bottleneck, while staying pure integer math (no
# library AA) so the JS recognizer reproduces it bit-for-bit.
SUPERSAMPLE = 4


@dataclass(frozen=True)
class DictEntry:
    id: str
    kind: str  # "sigil" | "sign"
    display_name: str
    element: str | None
    strokes: list[list[tuple[float, float]]]


def load_dictionary(dict_dir: Path) -> list[DictEntry]:
    entries: list[DictEntry] = []
    for kind, filename in (("sigil", "sigils.json"), ("sign", "signs.json")):
        raw = json.loads((dict_dir / filename).read_text(encoding="utf-8"))
        for item in raw:
            template = item.get("strokeTemplate") or {}
            strokes = [
                [(float(point["x"]), float(point["y"])) for point in stroke]
                for stroke in template.get("strokes", [])
                if stroke
            ]
            if not strokes:
                continue
            entries.append(
                DictEntry(
                    id=item["id"],
                    kind=kind,
                    display_name=item.get("displayName", item["id"]),
                    element=item.get("element"),
                    strokes=strokes,
                )
            )
    return entries


def _fit_unit_box(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    points = [point for stroke in strokes for point in stroke]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    scale = 1.0 / max(width, height)
    # Center the (possibly non-square) glyph inside the unit box.
    offset_x = (1.0 - width * scale) / 2.0
    offset_y = (1.0 - height * scale) / 2.0
    return [
        [((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y) for x, y in stroke]
        for stroke in strokes
    ]


def _rotate(
    strokes: list[list[tuple[float, float]]], degrees: float
) -> list[list[tuple[float, float]]]:
    if not degrees:
        return strokes
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    out = []
    for stroke in strokes:
        out.append(
            [
                (
                    (x - 0.5) * cos - (y - 0.5) * sin + 0.5,
                    (x - 0.5) * sin + (y - 0.5) * cos + 0.5,
                )
                for x, y in stroke
            ]
        )
    return out


def _round_half_up(value: float) -> int:
    # Match JS Math.floor(v + 0.5) exactly (Python's round() is banker's).
    return int(math.floor(value + 0.5))


def _disk_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = range(-radius, radius + 1)
    oy, ox = np.meshgrid(np.array(offsets), np.array(offsets), indexing="ij")
    inside = (ox * ox + oy * oy) <= radius * radius
    return ox[inside].ravel(), oy[inside].ravel()


def _stamp_centers(
    pixels: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    radius: int,
    size: int,
) -> None:
    """Stamp filled disks at integer pixel centers. Vectorized but produces the
    exact same pixel set as a per-pixel `ox^2+oy^2<=r^2` test (setting 255 is
    idempotent and order-independent), so JS parity is preserved."""
    if centers_x.size == 0:
        return
    off_x, off_y = _disk_offsets(radius)
    px = centers_x[:, None] + off_x[None, :]
    py = centers_y[:, None] + off_y[None, :]
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    pixels[py[valid], px[valid]] = 255


def _stamp_disk(pixels: np.ndarray, cx: float, cy: float, radius: int, size: int) -> None:
    _stamp_centers(
        pixels,
        np.array([_round_half_up(cx)]),
        np.array([_round_half_up(cy)]),
        radius,
        size,
    )


def _box_downsample(hi: np.ndarray, size: int, factor: int) -> np.ndarray:
    """Average each factor x factor block down to one pixel (floor division).

    Integer accumulate-then-floor so JS reproduces it exactly.
    """
    blocks = hi.reshape(size, factor, size, factor)
    summed = blocks.sum(axis=(1, 3), dtype=np.int32)
    return (summed // (factor * factor)).astype(np.uint8)


def rasterize_strokes(
    strokes: list[list[tuple[float, float]]],
    *,
    size: int = 56,
    rotation_deg: float = 0.0,
    stroke_width_ratio: float = STROKE_WIDTH_RATIO,
    padding_ratio: float = PADDING_RATIO,
    supersample: int = SUPERSAMPLE,
) -> Image.Image:
    """Render normalized strokes to an anti-aliased white-on-black glyph image.

    Strokes are disk-stamped on a `size * supersample` canvas, then box-averaged
    down to `size`. All math is integer / deterministic float so the JS
    recognizer reproduces it bit-for-bit (see the parity test).
    """
    fitted = _fit_unit_box(strokes)
    if rotation_deg:
        fitted = _rotate(fitted, rotation_deg)
        fitted = _fit_unit_box(fitted)

    hi_size = size * supersample
    pixels = np.zeros((hi_size, hi_size), dtype=np.uint8)
    inner = hi_size * (1.0 - 2.0 * padding_ratio)
    origin = hi_size * padding_ratio
    radius = max(1, _round_half_up(hi_size * stroke_width_ratio / 2.0))

    # Collect every disk-center (rounded to integer pixels) first, then stamp in
    # one vectorized pass. Duplicate centers are harmless (idempotent), so this
    # is identical to stamping each sample individually but far faster.
    centers_x: list[int] = []
    centers_y: list[int] = []
    for stroke in fitted:
        points = [(origin + x * inner, origin + y * inner) for x, y in stroke]
        if len(points) == 1:
            centers_x.append(_round_half_up(points[0][0]))
            centers_y.append(_round_half_up(points[0][1]))
            continue
        for index in range(1, len(points)):
            ax, ay = points[index - 1]
            bx, by = points[index]
            steps = max(1, math.ceil(math.hypot(bx - ax, by - ay)))
            for step in range(steps + 1):
                t = step / steps
                centers_x.append(_round_half_up(ax + (bx - ax) * t))
                centers_y.append(_round_half_up(ay + (by - ay) * t))

    _stamp_centers(pixels, np.array(centers_x), np.array(centers_y), radius, hi_size)
    return Image.fromarray(_box_downsample(pixels, size, supersample), mode="L")


def image_to_array(image: Image.Image, *, size: int = 56) -> np.ndarray:
    """Match the trainer's image_to_tensor: [1, size, size] float32 in [0, 1]."""
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, :, :]
