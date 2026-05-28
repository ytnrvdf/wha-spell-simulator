from __future__ import annotations

import argparse
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from torch.nn import functional as F

from model.siam_net import build_siamese_model, model_config_from_checkpoint


CANVAS_SIZE = 360


@dataclass(frozen=True)
class GlyphPrototype:
    name: str
    embedding: torch.Tensor
    count: int


class GlyphRecognizer:
    def __init__(
        self,
        *,
        glyph_dir: Path,
        checkpoint_path: Path | None,
        device: torch.device,
        temperature: float,
    ) -> None:
        self.device = device
        self.temperature = temperature
        self.lock = threading.Lock()
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_config = _model_config_with_image_size(model_config_from_checkpoint(checkpoint))
            self.image_size = int(model_config["image_size"])
            self.model = build_siamese_model(model_config).to(device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_config = {"architecture": "conv_attention", "image_size": 56}
            self.image_size = int(model_config["image_size"])
            self.model = build_siamese_model(model_config).to(device)
        self.model.eval()
        self.prototypes = self._load_prototypes(glyph_dir)

    @torch.no_grad()
    def recognize(self, image: Image.Image, *, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.prototypes:
            return []

        tensor = image_to_tensor(image, image_size=self.image_size).to(self.device)
        with self.lock:
            embedding = self.model(tensor.unsqueeze(0)).squeeze(0).cpu()
        prototype_embeddings = torch.stack([item.embedding for item in self.prototypes])
        similarities = F.cosine_similarity(embedding.unsqueeze(0), prototype_embeddings, dim=1)
        probabilities = torch.softmax(similarities / self.temperature, dim=0)
        top_count = min(top_k, len(self.prototypes))
        values, indices = torch.topk(probabilities, k=top_count)
        return [
            (self.prototypes[index].name, float(value.item() * 100.0))
            for value, index in zip(values, indices, strict=True)
        ]

    @torch.no_grad()
    def _load_prototypes(self, glyph_dir: Path) -> list[GlyphPrototype]:
        grouped_paths: dict[str, list[Path]] = {}
        for path in sorted(glyph_dir.glob("*.png")):
            name = parse_glyph_name(path)
            if name is None:
                continue
            grouped_paths.setdefault(name, []).append(path)

        prototypes = []
        for name, paths in grouped_paths.items():
            tensors = [
                image_to_tensor(load_glyph_image(path), image_size=self.image_size).to(self.device)
                for path in paths
            ]
            batch = torch.stack(tensors)
            with self.lock:
                embeddings = self.model(batch).cpu()
            mean_embedding = F.normalize(embeddings.mean(dim=0), p=2, dim=0)
            prototypes.append(GlyphPrototype(name=name, embedding=mean_embedding, count=len(paths)))

        return sorted(prototypes, key=lambda item: item.name)


class GlyphGui:
    def __init__(self, root: tk.Tk, recognizer: GlyphRecognizer) -> None:
        self.root = root
        self.recognizer = recognizer
        self.tool = tk.StringVar(value="brush")
        self.brush_size = tk.IntVar(value=18)
        self.status = tk.StringVar(
            value=(
                f"loaded {len(recognizer.prototypes)} glyphs, "
                f"input {recognizer.image_size}x{recognizer.image_size}"
            )
        )
        self.last_point: tuple[int, int] | None = None
        self.recognition_job: str | None = None
        self.recognition_version = 0

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self._build()
        self.update_preview()
        self.schedule_recognition()

    def _build(self) -> None:
        self.root.title("Glyph Recognizer")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            main,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#444",
        )
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        panel = ttk.Frame(main, padding=(12, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="new")

        ttk.Label(panel, text="Tool").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(panel, text="Brush", value="brush", variable=self.tool).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Radiobutton(panel, text="Eraser", value="eraser", variable=self.tool).grid(
            row=2, column=0, sticky="w"
        )

        ttk.Label(panel, text="Size").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Scale(
            panel,
            from_=4,
            to=48,
            variable=self.brush_size,
            orient="horizontal",
            length=180,
        ).grid(row=4, column=0, sticky="ew")

        ttk.Button(panel, text="Clear", command=self.clear).grid(row=5, column=0, sticky="ew", pady=12)

        ttk.Label(panel, text="Nearest glyphs").grid(row=6, column=0, sticky="w")
        self.results = tk.Listbox(panel, width=32, height=8)
        self.results.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        ttk.Label(panel, textvariable=self.status).grid(row=8, column=0, sticky="w", pady=(12, 0))

    def on_press(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        self.paint(event.x, event.y, event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        if self.last_point is None:
            self.last_point = (event.x, event.y)
        x0, y0 = self.last_point
        self.paint(x0, y0, event.x, event.y)
        self.last_point = (event.x, event.y)

    def on_release(self, _event: tk.Event) -> None:
        self.last_point = None

    def paint(self, x0: int, y0: int, x1: int, y1: int) -> None:
        radius = int(self.brush_size.get())
        color = 255 if self.tool.get() == "brush" else 0
        self.draw.line((x0, y0, x1, y1), fill=color, width=radius, joint="curve")
        half = radius // 2
        self.draw.ellipse((x1 - half, y1 - half, x1 + half, y1 + half), fill=color)
        self.update_preview()
        self.schedule_recognition()

    def clear(self) -> None:
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.update_preview()
        self.schedule_recognition()

    def update_preview(self) -> None:
        self.preview = ImageTk.PhotoImage(self.image.convert("RGB"))
        self.canvas.create_image(0, 0, image=self.preview, anchor="nw")

    def schedule_recognition(self) -> None:
        self.recognition_version += 1
        if self.recognition_job is not None:
            self.root.after_cancel(self.recognition_job)
        self.recognition_job = self.root.after(120, self.recognize_async, self.recognition_version)

    def recognize_async(self, version: int) -> None:
        image = self.image.copy()
        threading.Thread(
            target=self._recognize_worker,
            args=(image, version),
            daemon=True,
        ).start()

    def _recognize_worker(self, image: Image.Image, version: int) -> None:
        try:
            results = self.recognizer.recognize(image)
        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: messagebox.showerror("Recognition error", message))
            return
        self.root.after(0, self.show_results, results, version)

    def show_results(self, results: list[tuple[str, float]], version: int) -> None:
        if version != self.recognition_version:
            return
        self.results.delete(0, tk.END)
        if not results:
            self.results.insert(tk.END, "No glyph references found")
            return
        for name, percent in results:
            self.results.insert(tk.END, f"{name}: {percent:5.1f}%")


def parse_glyph_name(path: Path) -> str | None:
    stem = path.stem
    if "_" not in stem:
        return None
    name, index = stem.rsplit("_", 1)
    if not name or not index:
        return None
    return name


def load_glyph_image(path: Path) -> Image.Image:
    return Image.open(path)


def image_to_tensor(image: Image.Image, *, image_size: int) -> torch.Tensor:
    glyph = normalize_glyph_image(image)
    glyph = glyph.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(glyph, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _model_config_with_image_size(model_config: dict[str, object]) -> dict[str, object]:
    config = dict(model_config)
    if "image_size" not in config:
        architecture = config.get("architecture", "conv_attention")
        config["image_size"] = 56 if architecture == "conv_attention" else 28
    return config


def normalize_glyph_image(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    gray = np.asarray(rgba.convert("L"), dtype=np.float32) / 255.0

    if alpha.max() > 0 and alpha.mean() < 0.98:
        glyph = alpha
    else:
        glyph = gray
        if glyph.mean() > 0.5:
            glyph = 1.0 - glyph

    glyph = np.clip(glyph * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(glyph, mode="L")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyph-dir", default="glyphs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--random-weights", action="store_true")
    args = parser.parse_args(argv)

    glyph_dir = Path(args.glyph_dir)
    glyph_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint_path = None
    if not args.random_weights:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else find_latest_checkpoint(Path(args.checkpoint_dir))
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Train first or pass --random-weights for UI testing."
            )

    recognizer = GlyphRecognizer(
        glyph_dir=glyph_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        temperature=args.temperature,
    )

    root = tk.Tk()
    GlyphGui(root, recognizer)
    root.mainloop()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
