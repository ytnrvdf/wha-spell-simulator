# Checkpoints

Trained `.pt` weights are not committed (they are large; the app ships
`src/parser/siamese-assets/model.onnx` for inference). This folder is where the training
and export scripts read/write them.

- **`best_siamese_glyph_net.pt`** — the base encoder, metric-trained on
  Omniglot. Produce it with `trainer.py` (see `../README.md`), or grab it from
  the PR's release assets.
- **`finetuned_glyph_net.pt`** — the base encoder lightly adapted to this
  project's dictionary by `finetune.py`. Regenerate any time:

  ```bash
  uv run finetune.py --dict-dir ../../src/dictionary
  ```

`export_onnx.py` then turns a checkpoint into `src/parser/siamese-assets/model.onnx`.
