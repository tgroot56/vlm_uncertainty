#!/usr/bin/env python3
"""Serve a browser for gradient-guided token patching layer figures."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse


DEFAULT_ROOT = Path("/projects/prjs2014/patching/gradient_guided_token_patching")
MODELS = ("llava-hf_llava-1-5-7b-hf", "Qwen_Qwen3-5-9B")
OBJECTIVE = "obj_yes"
LAYER_RE = re.compile(r"layer(\d+)\.(png|pdf)$")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gradient-guided layer viewer</title>
  <style>
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { background: #111827; color: #f9fafb; font: 16px/1.4 system-ui, sans-serif; }
    #chooser {
      display: grid;
      place-items: center;
      width: 100%;
      height: 100%;
      padding: 24px;
    }
    #chooser-panel {
      width: min(620px, 100%);
      padding: 28px;
      border: 1px solid #4b5563;
      border-radius: 12px;
      background: #1f2937;
    }
    h1 { margin: 0 0 22px; font-size: 23px; }
    label { display: block; margin: 15px 0 6px; font-weight: 700; }
    select {
      width: 100%;
      padding: 10px;
      border: 1px solid #9ca3af;
      border-radius: 6px;
      background: #374151;
      color: #fff;
      font: inherit;
    }
    select:disabled { opacity: 0.5; }
    #status { min-height: 24px; margin-top: 18px; color: #fbbf24; font-weight: 600; }
    #viewer { display: none; width: 100%; height: 100%; background: white; }
    body.viewing #chooser { display: none; }
    body.viewing #viewer { display: block; }
    #pdf {
      display: block;
      width: 100%;
      height: 100%;
      border: 0;
      background: white;
      pointer-events: none;
    }
    #layer-badge, #error {
      position: fixed;
      z-index: 3;
      padding: 7px 11px;
      border-radius: 7px;
      background: rgba(17, 24, 39, 0.86);
      color: white;
      font-weight: 700;
    }
    #layer-badge { top: 10px; right: 10px; }
    #error {
      display: none;
      top: 50%;
      left: 50%;
      max-width: min(700px, 90%);
      transform: translate(-50%, -50%);
      color: #fca5a5;
      text-align: center;
    }
  </style>
</head>
<body>
  <section id="chooser">
    <div id="chooser-panel">
      <h1>Gradient-guided token patching figures</h1>
      <label for="model">Model</label>
      <select id="model">
        <option value="">Select a model</option>
        <option value="llava-hf_llava-1-5-7b-hf">llava-hf_llava-1-5-7b-hf</option>
        <option value="Qwen_Qwen3-5-9B">Qwen_Qwen3-5-9B</option>
      </select>
      <label for="sample">Sample ID</label>
      <select id="sample" disabled><option value="">Select a model first</option></select>
      <div id="status">Select a model, then a sample ID.</div>
    </div>
  </section>
  <main id="viewer">
    <embed id="pdf" type="application/pdf">
    <div id="layer-badge"></div>
    <div id="error"></div>
  </main>
  <script>
    const model = document.getElementById("model");
    const sample = document.getElementById("sample");
    const status = document.getElementById("status");
    const pdf = document.getElementById("pdf");
    const badge = document.getElementById("layer-badge");
    const errorBox = document.getElementById("error");
    let layers = [];
    let index = 0;

    async function fetchJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
      return data;
    }

    function figureUrl(layer) {
      const query = new URLSearchParams({model: model.value, sample: sample.value, layer});
      return `/figure?${query}#toolbar=0&navpanes=0&scrollbar=0&view=Fit`;
    }

    function preloadAdjacent() {
      for (const candidate of [layers[index - 1], layers[index + 1]]) {
        if (candidate) fetch(figureUrl(candidate.number).split("#")[0]).catch(() => {});
      }
    }

    function preloadRemaining() {
      for (const layer of layers) {
        if (layer !== layers[index]) fetch(figureUrl(layer.number).split("#")[0]).catch(() => {});
      }
    }

    function showLayer(nextIndex) {
      if (!layers.length) return;
      index = Math.max(0, Math.min(nextIndex, layers.length - 1));
      const layer = layers[index];
      errorBox.style.display = "none";
      pdf.src = figureUrl(layer.number);
      badge.textContent = `${model.value} | sample ${sample.value} | obj_yes | layer ${layer.number}`;
      preloadAdjacent();
    }

    async function loadSamples() {
      sample.disabled = true;
      sample.innerHTML = '<option value="">Loading samples...</option>';
      if (!model.value) {
        sample.innerHTML = '<option value="">Select a model first</option>';
        status.textContent = "Select a model, then a sample ID.";
        return;
      }
      try {
        const data = await fetchJson(`/api/samples?model=${encodeURIComponent(model.value)}`);
        sample.innerHTML = '<option value="">Select a sample ID</option>';
        for (const id of data.samples) sample.add(new Option(id, id));
        sample.disabled = !data.samples.length;
        status.textContent = data.samples.length ? "Select a sample ID." : "No samples found.";
      } catch (error) {
        status.textContent = error.message;
      }
    }

    async function loadLayers() {
      if (!sample.value) return;
      status.textContent = "Loading layers...";
      try {
        const query = new URLSearchParams({model: model.value, sample: sample.value});
        const data = await fetchJson(`/api/layers?${query}`);
        layers = data.layers;
        if (!layers.length) throw new Error("No PDF layer figures found for this sample.");
        document.body.classList.add("viewing");
        showLayer(0);
        setTimeout(preloadRemaining, 500);
      } catch (error) {
        status.textContent = error.message;
      }
    }

    pdf.addEventListener("error", () => {
      errorBox.textContent = "Could not load this PDF layer figure.";
      errorBox.style.display = "block";
    });
    model.addEventListener("change", loadSamples);
    sample.addEventListener("change", loadLayers);
    document.addEventListener("keydown", (event) => {
      if (!document.body.classList.contains("viewing")) return;
      if (event.code === "Space" || event.key === "ArrowRight") {
        event.preventDefault();
        showLayer(index + 1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        showLayer(index - 1);
      }
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve an interactive gradient-guided token patching layer viewer."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Gradient-guided output root (default: {DEFAULT_ROOT}).",
    )
    return parser.parse_args()


def sample_directory(root: Path, model: str, sample: str) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not sample.isdigit():
        raise ValueError("Sample ID must contain digits only.")
    return root / model / "pope" / "random" / f"sample_{sample}" / OBJECTIVE


def discover_samples(root: Path, model: str) -> list[int]:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    random_dir = root / model / "pope" / "random"
    samples = []
    for objective_dir in random_dir.glob(f"sample_*/{OBJECTIVE}"):
        sample = objective_dir.parent.name.removeprefix("sample_")
        if sample.isdigit():
            samples.append(int(sample))
    return sorted(samples)


def discover_layers(objective_dir: Path) -> list[dict[str, object]]:
    by_number: dict[int, dict[str, object]] = {}
    for path in objective_dir.glob("layer*.*"):
        match = LAYER_RE.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group(1))
        extension = match.group(2)
        candidate = {"number": number, "extension": extension, "file": path.name}
        if number not in by_number or extension == "pdf":
            by_number[number] = candidate
    return [by_number[number] for number in sorted(by_number)]


def figure_path(root: Path, model: str, sample: str, layer: str) -> Path:
    if not layer.isdigit() or int(layer) < 1:
        raise ValueError("Layer must be a positive integer.")
    objective_dir = sample_directory(root, model, sample)
    pdf = objective_dir / f"layer{int(layer)}.pdf"
    png = objective_dir / f"layer{int(layer)}.png"
    if pdf.is_file():
        return pdf
    if png.is_file():
        return png
    raise FileNotFoundError(f"Layer {int(layer)} not found for model {model}, sample {sample}.")


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif parsed.path == "/api/samples":
                    model = self.query_value(query, "model")
                    self.send_json({"model": model, "samples": discover_samples(root, model)})
                elif parsed.path == "/api/layers":
                    model = self.query_value(query, "model")
                    sample = self.query_value(query, "sample")
                    objective_dir = sample_directory(root, model, sample)
                    if not objective_dir.is_dir():
                        raise FileNotFoundError(
                            f"No {OBJECTIVE} figures found for model {model}, sample {sample}."
                        )
                    self.send_json(
                        {
                            "model": model,
                            "sample": sample,
                            "objective": OBJECTIVE,
                            "layers": discover_layers(objective_dir),
                        }
                    )
                elif parsed.path == "/figure":
                    model = self.query_value(query, "model")
                    sample = self.query_value(query, "sample")
                    layer = self.query_value(query, "layer")
                    self.send_file(figure_path(root, model, sample, layer))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (ValueError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
            except Exception as error:
                self.send_json(
                    {"error": f"Unexpected server error: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        @staticmethod
        def query_value(query: dict[str, list[str]], name: str) -> str:
            values = query.get(name)
            if not values or not values[0]:
                raise ValueError(f"Missing query parameter: {name}")
            return values[0]

        def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def send_file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_bytes(path.read_bytes(), content_type, cache_control="public, max-age=86400")

        def send_bytes(
            self,
            content: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache_control: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return ViewerHandler


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Gradient-guided output root does not exist: {root}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    query = urlencode({"model": MODELS[0]})
    print(f"Serving figures from: {root}")
    print(f"Open: http://localhost:{args.port}/")
    print(f"Example samples API: http://localhost:{args.port}/api/samples?{query}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
