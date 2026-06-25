#!/usr/bin/env python3
"""Create one layer-by-layer animation for each matching sample directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LAYER_RE = re.compile(r"layer(\d+)\.png$")
VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{sample_name} layer viewer</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #111827;
      color: #f9fafb;
      font: 16px/1.4 system-ui, sans-serif;
      overflow: hidden;
    }}
    header {{
      height: 112px;
      padding: 12px 20px;
      background: #1f2937;
      border-bottom: 1px solid #4b5563;
    }}
    .top-row, .controls {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .top-row {{ justify-content: space-between; margin-bottom: 10px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    #counter {{ font-size: 22px; font-weight: 700; }}
    button {{
      padding: 7px 14px;
      border: 1px solid #9ca3af;
      border-radius: 6px;
      background: #374151;
      color: #fff;
      cursor: pointer;
    }}
    button:hover {{ background: #4b5563; }}
    input[type="range"] {{ flex: 1; min-width: 160px; }}
    .hint {{ color: #d1d5db; white-space: nowrap; }}
    main {{
      height: calc(100vh - 112px);
      overflow: auto;
      padding: 12px;
      text-align: center;
    }}
    img {{
      display: block;
      margin: 0 auto;
      max-width: 100%;
      max-height: calc(100vh - 136px);
      width: auto;
      height: auto;
      object-fit: contain;
      background: white;
    }}
  </style>
</head>
<body>
  <header>
    <div class="top-row">
      <h1>{sample_name} interactive layer viewer</h1>
      <div id="counter"></div>
    </div>
    <div class="controls">
      <button id="previous" type="button">Previous</button>
      <input id="slider" type="range" min="0" max="{max_index}" value="0">
      <button id="next" type="button">Next</button>
      <span class="hint">Space / Right: next &nbsp; Backspace / Left: previous</span>
    </div>
  </header>
  <main><img id="layer-image" alt=""></main>
  <script>
    const layers = {layers_json};
    let index = 0;
    const image = document.getElementById("layer-image");
    const counter = document.getElementById("counter");
    const slider = document.getElementById("slider");

    function showLayer(nextIndex) {{
      index = (nextIndex + layers.length) % layers.length;
      const layer = layers[index];
      image.src = layer.file;
      image.alt = `Layer ${{layer.number}}`;
      counter.textContent = `Layer ${{layer.number}} (${{index + 1}} / ${{layers.length}})`;
      slider.value = index;
      history.replaceState(null, "", `#layer-${{layer.number}}`);
    }}

    document.getElementById("previous").addEventListener("click", () => showLayer(index - 1));
    document.getElementById("next").addEventListener("click", () => showLayer(index + 1));
    slider.addEventListener("input", () => showLayer(Number(slider.value)));
    document.addEventListener("keydown", (event) => {{
      if (event.code === "Space" || event.key === "ArrowRight") {{
        event.preventDefault();
        showLayer(index + 1);
      }} else if (event.key === "Backspace" || event.key === "ArrowLeft") {{
        event.preventDefault();
        showLayer(index - 1);
      }} else if (event.key === "Home") {{
        event.preventDefault();
        showLayer(0);
      }} else if (event.key === "End") {{
        event.preventDefault();
        showLayer(layers.length - 1);
      }}
    }});

    const requested = Number(location.hash.replace("#layer-", ""));
    const requestedIndex = layers.findIndex((layer) => layer.number === requested);
    showLayer(requestedIndex >= 0 ? requestedIndex : 0);
  </script>
</body>
</html>
"""
VIEWER_README = """Interactive layer viewer
========================

Open layer_viewer.html directly in a web browser.

Alternatively, run this command from any directory:

    python -m http.server 8000 --directory "{objective_dir}"

Then open http://localhost:8000/layer_viewer.html in a browser. If running on a
remote machine, forward port 8000 first.

Controls:
- Space or Right Arrow: next layer
- Backspace or Left Arrow: previous layer
- Home / End: first / last layer
- Previous / Next buttons and the slider also select layers
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch numerically ordered layer PNGs into animated WebP files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing sample_<id>/<objective>/layer<number>.png.",
    )
    parser.add_argument(
        "--objective",
        default="obj_yes",
        help="Objective subdirectory to animate (default: obj_yes).",
    )
    parser.add_argument(
        "--output-name",
        default="layers.webp",
        help="Output filename created inside each objective directory.",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=650,
        help="Milliseconds to show each layer (default: 650).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Downscale frames wider than this value (default: 1600).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="Animated WebP quality from 0 to 100 (default: 85).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace animations that already exist.",
    )
    parser.add_argument(
        "--viewer-name",
        default="layer_viewer.html",
        help="Interactive viewer filename created inside each objective directory.",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Create interactive viewers without creating animations.",
    )
    return parser.parse_args()


def layer_number(path: Path) -> int:
    match = LAYER_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a layer PNG: {path}")
    return int(match.group(1))


def load_frame(path: Path, max_width: int) -> Image.Image:
    with Image.open(path) as source:
        frame = source.convert("RGB")

    if frame.width > max_width:
        height = round(frame.height * max_width / frame.width)
        frame = frame.resize((max_width, height), Image.Resampling.LANCZOS)
    return frame


def find_layers(objective_dir: Path) -> list[Path]:
    return sorted(
        (path for path in objective_dir.glob("layer*.png") if LAYER_RE.fullmatch(path.name)),
        key=layer_number,
    )


def write_viewer(objective_dir: Path, viewer_name: str) -> bool:
    layers = find_layers(objective_dir)
    if not layers:
        print(f"skip: no layer PNGs in {objective_dir}")
        return False

    output_path = objective_dir / viewer_name
    layer_data = [
        {"number": layer_number(path), "file": path.name}
        for path in layers
    ]
    output_path.write_text(
        VIEWER_TEMPLATE.format(
            sample_name=objective_dir.parent.name,
            max_index=len(layers) - 1,
            layers_json=json.dumps(layer_data),
        ),
        encoding="utf-8",
    )
    (objective_dir / "LAYER_VIEWER_README.txt").write_text(
        VIEWER_README.format(objective_dir=objective_dir.resolve()),
        encoding="utf-8",
    )
    print(f"wrote: {output_path} ({len(layers)} layers)")
    return True


def animate_directory(
    objective_dir: Path,
    output_name: str,
    frame_ms: int,
    max_width: int,
    quality: int,
    overwrite: bool,
) -> bool:
    output_path = objective_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"skip: {output_path}")
        return False

    layers = find_layers(objective_dir)
    if not layers:
        print(f"skip: no layer PNGs in {objective_dir}")
        return False

    loaded_frames = [load_frame(path, max_width) for path in layers]
    canvas_width = max(frame.width for frame in loaded_frames)
    frame_height = max(frame.height for frame in loaded_frames)
    header_height = max(36, canvas_width // 32)
    canvas_height = frame_height + header_height
    font = ImageFont.load_default(size=max(18, canvas_width // 70))
    frames = []
    for frame, path in zip(loaded_frames, layers):
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(frame, ((canvas_width - frame.width) // 2, header_height))
        label = f"{objective_dir.parent.name} | layer {layer_number(path)}"
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((canvas_width - (bbox[2] - bbox[0])) // 2, (header_height - (bbox[3] - bbox[1])) // 2),
            label,
            fill=(0, 0, 0),
            font=font,
        )
        frames.append(canvas)

    frames[0].save(
        output_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        quality=quality,
        method=4,
    )
    print(f"wrote: {output_path} ({len(frames)} frames)")
    return True


def main() -> None:
    args = parse_args()
    objective_dirs = sorted(
        args.root.glob(f"sample_*/{args.objective}"),
        key=lambda path: int(path.parent.name.removeprefix("sample_")),
    )
    if not objective_dirs:
        raise SystemExit(f"No sample_*/{args.objective} directories found under {args.root}")

    viewers_written = sum(
        write_viewer(objective_dir, args.viewer_name)
        for objective_dir in objective_dirs
    )
    animations_written = 0
    if not args.viewer_only:
        animations_written = sum(
            animate_directory(
                objective_dir,
                args.output_name,
                args.frame_ms,
                args.max_width,
                args.quality,
                args.overwrite,
            )
            for objective_dir in objective_dirs
        )
    print(
        f"done: wrote {viewers_written} viewer(s) and {animations_written} animation(s); "
        f"found {len(objective_dirs)} sample(s)"
    )


if __name__ == "__main__":
    main()
