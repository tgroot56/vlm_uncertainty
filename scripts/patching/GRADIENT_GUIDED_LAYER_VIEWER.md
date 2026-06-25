# Gradient-Guided Layer Viewer

This viewer browses existing `obj_yes` layer figures across the supported LLaVA
and Qwen gradient-guided token patching outputs.

From the repository root, run:

```bash
python scripts/patching/serve_gradient_guided_layer_viewer.py --port 8000
```

Then open:

```text
http://localhost:8000/
```

Do not use `python -m http.server` for this viewer. The dedicated server is
needed to discover and serve figures from multiple model and sample directories.

Controls:

- Select a model, then a sample ID.
- `Space` or Right Arrow: next available layer.
- Left Arrow: previous available layer.

After selecting a sample, the embedded PDF fills the browser window. The viewer
prefers `layer<number>.pdf`, falls back to PNG only when a PDF is unavailable,
and preloads layer figures for faster keyboard navigation. Missing samples and
figures are reported in the interface.
