from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generalization.evaluate_msts import (
    DEFAULT_MSTS_RUN_DIR,
    build_annotated_msts_dataset,
    _apply_checkpoint_normalization,
    _load_model_for_checkpoint,
    _metrics,
    _predict_all,
    _select_features,
)


DEFAULT_ANNOTATION_CSV = Path(
    "/projects/prjs2014/msts/llava-hf/llava-1.5-7b-hf/run_14423ff29d/annotation_template.csv"
)
DEFAULT_RESULTS_DIR = Path(
    "/projects/prjs2014/tgroot/probe_results/msts_eval/llava/all_datasets_lm_fullspan_lasttok_layer_16"
)
DEFAULT_FEATURE_NAME = "lm_fullspan_lasttok_layer_16"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/projects/prjs2014/probe_results/generalization/all_datasets/llava/_shared_train/"
    "lm_fullspan_lasttok_layer_16/seed_runs"
)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    checkpoint_path: Path


def default_probe_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(
            name=f"all_datasets_seed_{seed}",
            checkpoint_path=(
                DEFAULT_CHECKPOINT_ROOT
                / f"seed_{seed}"
                / "train_all_datasets"
                / "shared_train"
                / "mlp"
                / DEFAULT_FEATURE_NAME
                / "best_model.pt"
            ),
        )
        for seed in (42, 123, 456)
    ]


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MSTS Across-Dataset Probe Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #fff;
      --ink: #1f2933;
      --muted: #637083;
      --line: #d7dde6;
      --accent: #155e75;
      --accent-soft: #dff3f8;
      --danger: #b91c1c;
      --ok: #166534;
      --shadow: 0 1px 4px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    h2 { margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 410px) minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      position: sticky;
      top: 58px;
      align-self: start;
      max-height: calc(100vh - 58px);
      display: flex;
      flex-direction: column;
    }
    .filters {
      padding: 12px;
      display: grid;
      gap: 8px;
      border-bottom: 1px solid var(--line);
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    .sample-list { overflow: auto; padding: 8px; display: grid; gap: 6px; }
    .sample-button {
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      border-radius: 7px;
      padding: 9px 10px;
      cursor: pointer;
      color: var(--ink);
    }
    .sample-button:hover { background: #f1f4f7; }
    .sample-button.active { border-color: var(--accent); background: var(--accent-soft); }
    .sample-top { display: flex; justify-content: space-between; gap: 10px; font-weight: 650; }
    .sample-prompt {
      margin-top: 4px;
      color: var(--muted);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .workspace { padding: 18px; display: grid; gap: 14px; align-content: start; }
    .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
    }
    .sample-detail {
      display: grid;
      grid-template-columns: minmax(260px, 38%) minmax(0, 1fr);
      gap: 16px;
    }
    .image-wrap {
      background: #eef1f5;
      border: 1px solid var(--line);
      border-radius: 7px;
      display: grid;
      place-items: center;
      min-height: 260px;
      overflow: hidden;
    }
    .image-wrap img {
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      display: block;
    }
    .prompt { font-size: 18px; font-weight: 650; margin: 0 0 10px; }
    .answer { white-space: pre-wrap; margin: 0; color: #293441; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }
    .meta span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f7;
      color: var(--ink);
      font-weight: 650;
      white-space: nowrap;
    }
    .pill.ok { color: var(--ok); background: #dcfce7; }
    .pill.bad { color: var(--danger); background: #fee2e2; }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
    th { color: var(--muted); font-weight: 650; background: var(--panel); }
    .prob-cell { display: grid; grid-template-columns: 54px 1fr; align-items: center; gap: 8px; min-width: 160px; }
    .bar { height: 8px; border-radius: 999px; background: #e4e9f0; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: var(--accent); }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; max-height: 420px; border-right: 0; border-bottom: 1px solid var(--line); }
      .sample-detail { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MSTS Across-Dataset Final Prompt Token Probe</h1>
    <div class="status" id="status">Loading...</div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <input id="search" type="search" placeholder="Search prompts, answers, hazards..." />
        <select id="labelFilter">
          <option value="all">All labels</option>
          <option value="1">Safe label 1</option>
          <option value="0">Unsafe label 0</option>
        </select>
      </div>
      <div class="sample-list" id="sampleList"></div>
    </aside>
    <section class="workspace">
      <div class="section sample-detail">
        <div class="image-wrap"><img id="sampleImage" alt="MSTS sample image" /></div>
        <div>
          <p class="prompt" id="promptText"></p>
          <div class="meta-grid" id="metaGrid"></div>
          <h2>LLaVA Answer</h2>
          <p class="answer" id="responseText"></p>
        </div>
      </div>
      <div class="section">
        <h2>Mean Predictions By Label</h2>
        <div class="toolbar">
          <select id="probeFilter"><option value="all">All probes</option></select>
          <select id="sortMode">
            <option value="probe">Sort by probe</option>
            <option value="gap_desc">Largest unsafe-safe gap</option>
            <option value="safe_desc">Highest safe mean</option>
            <option value="unsafe_desc">Highest unsafe mean</option>
          </select>
          <select id="rowFilter">
            <option value="all">Show all rows</option>
            <option value="ensemble">Ensemble only</option>
            <option value="seeds">Seed probes only</option>
          </select>
        </div>
        <table>
          <thead>
            <tr>
              <th>Probe</th>
              <th>Feature</th>
              <th>Safe Mean</th>
              <th>Unsafe Mean</th>
              <th>Unsafe - Safe</th>
              <th>Counts</th>
              <th>Brier</th>
              <th>AUROC</th>
              <th>ECE</th>
            </tr>
          </thead>
          <tbody id="aggregateBody"></tbody>
        </table>
        <h2 style="margin-top:14px;">Selected Sample Predictions</h2>
        <table>
          <thead>
            <tr>
              <th>Probe</th>
              <th>Feature</th>
              <th>Prediction</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody id="predictionBody"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const state = { samples: [], predictions: [], metrics: [], selectedIdx: 0 };
    const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "";
    const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const labelPill = (label) => {
      const cls = String(label) === "1" ? "ok" : "bad";
      return `<span class="pill ${cls}">${String(label) === "1" ? "safe / label 1" : "unsafe / label 0"}</span>`;
    };

    function fillSelect(id, values, label) {
      const el = document.getElementById(id);
      el.innerHTML = `<option value="all">${label}</option>` + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
    }

    function populateFilters() {
      fillSelect("probeFilter", [...new Set(state.predictions.map(p => p.probe_name))].sort(), "All probes");
    }

    function filteredSamples() {
      const q = document.getElementById("search").value.trim().toLowerCase();
      const lf = document.getElementById("labelFilter").value;
      return state.samples.filter(sample => {
        if (lf !== "all" && String(sample.manual_label) !== lf) return false;
        if (!q) return true;
        const haystack = [
          sample.prompt_id, sample.case_id, sample.prompt_text, sample.response,
          sample.hazard_category, sample.hazard_subcategory, sample.unsafe_image_description
        ].join(" ").toLowerCase();
        return haystack.includes(q);
      });
    }

    function visiblePredictions() {
      const probe = document.getElementById("probeFilter").value;
      const rowFilter = document.getElementById("rowFilter").value;
      let rows = state.predictions;
      if (probe !== "all") rows = rows.filter(row => row.probe_name === probe);
      if (rowFilter === "ensemble") rows = rows.filter(row => row.probe_name === "ensemble_mean");
      if (rowFilter === "seeds") rows = rows.filter(row => row.probe_name !== "ensemble_mean");
      return rows;
    }

    function renderList() {
      const list = document.getElementById("sampleList");
      const samples = filteredSamples();
      document.getElementById("status").textContent = `${samples.length} / ${state.samples.length} samples`;
      list.innerHTML = samples.map(sample => `
        <button class="sample-button ${Number(sample.idx) === state.selectedIdx ? "active" : ""}" data-idx="${sample.idx}">
          <div class="sample-top">
            <span>#${sample.idx} ${escapeHtml(sample.prompt_id)}</span>
            ${labelPill(sample.manual_label)}
          </div>
          <div class="sample-prompt">${escapeHtml(sample.prompt_text)}</div>
        </button>
      `).join("");
      list.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          state.selectedIdx = Number(btn.dataset.idx);
          render();
        });
      });
    }

    function renderDetail() {
      const sample = state.samples.find(s => Number(s.idx) === state.selectedIdx) || state.samples[0];
      if (!sample) return;
      document.getElementById("sampleImage").src = `/image?idx=${encodeURIComponent(sample.idx)}`;
      document.getElementById("promptText").textContent = sample.prompt_text;
      document.getElementById("responseText").textContent = sample.response;
      document.getElementById("metaGrid").innerHTML = [
        ["Label", labelPill(sample.manual_label)],
        ["Hazard", escapeHtml(sample.hazard_category)],
        ["Subcategory", escapeHtml(sample.hazard_subcategory || "")],
        ["Image", escapeHtml(sample.unsafe_image_description || sample.unsafe_image_id)],
        ["Content warning", escapeHtml(sample.unsafe_image_cw || "")],
        ["License", escapeHtml(sample.unsafe_image_license || "")]
      ].map(([k, v]) => `<div class="meta"><span>${k}</span>${v}</div>`).join("");
    }

    function aggregateRows() {
      const sampleByIdx = new Map(state.samples.map(s => [Number(s.idx), s]));
      const grouped = new Map();
      for (const row of visiblePredictions()) {
        const key = row.probe_name;
        const sample = sampleByIdx.get(Number(row.idx));
        if (!sample) continue;
        if (!grouped.has(key)) {
          const metric = state.metrics.find(m => m.probe_name === key) || {};
          grouped.set(key, {
            probe_name: key,
            feature_name: row.feature_name,
            safe_sum: 0,
            safe_n: 0,
            unsafe_sum: 0,
            unsafe_n: 0,
            brier: metric.brier,
            auroc: metric.auroc,
            ece: metric.ece,
          });
        }
        const group = grouped.get(key);
        if (String(sample.manual_label) === "1") {
          group.safe_sum += Number(row.prediction);
          group.safe_n += 1;
        } else if (String(sample.manual_label) === "0") {
          group.unsafe_sum += Number(row.prediction);
          group.unsafe_n += 1;
        }
      }
      const sortMode = document.getElementById("sortMode").value;
      const rows = [...grouped.values()].map(group => {
        const safeMean = group.safe_n ? group.safe_sum / group.safe_n : NaN;
        const unsafeMean = group.unsafe_n ? group.unsafe_sum / group.unsafe_n : NaN;
        return { ...group, safeMean, unsafeMean, gap: unsafeMean - safeMean };
      });
      if (sortMode === "gap_desc") rows.sort((a, b) => b.gap - a.gap);
      if (sortMode === "safe_desc") rows.sort((a, b) => b.safeMean - a.safeMean);
      if (sortMode === "unsafe_desc") rows.sort((a, b) => b.unsafeMean - a.unsafeMean);
      if (sortMode === "probe") rows.sort((a, b) => a.probe_name.localeCompare(b.probe_name));
      return rows;
    }

    function renderAggregates() {
      document.getElementById("aggregateBody").innerHTML = aggregateRows().map(row => `
        <tr>
          <td>${escapeHtml(row.probe_name)}</td>
          <td>${escapeHtml(row.feature_name)}</td>
          <td>${fmt(row.safeMean)}</td>
          <td>${fmt(row.unsafeMean)}</td>
          <td>${fmt(row.gap)}</td>
          <td><span class="muted">safe ${row.safe_n}, unsafe ${row.unsafe_n}</span></td>
          <td>${fmt(row.brier)}</td>
          <td>${fmt(row.auroc)}</td>
          <td>${fmt(row.ece)}</td>
        </tr>
      `).join("");
    }

    function renderPredictions() {
      let rows = visiblePredictions().filter(row => Number(row.idx) === state.selectedIdx);
      rows.sort((a, b) => a.probe_name.localeCompare(b.probe_name));
      document.getElementById("predictionBody").innerHTML = rows.map(row => {
        const width = Math.max(0, Math.min(100, Number(row.prediction) * 100));
        return `<tr>
          <td>${escapeHtml(row.probe_name)}</td>
          <td>${escapeHtml(row.feature_name)}</td>
          <td><div class="prob-cell"><span>${fmt(row.prediction)}</span><div class="bar"><span style="width:${width}%"></span></div></div></td>
          <td>${labelPill(row.label)}</td>
        </tr>`;
      }).join("");
    }

    function render() {
      renderList();
      renderDetail();
      renderAggregates();
      renderPredictions();
    }

    async function boot() {
      const res = await fetch("/api/data");
      const data = await res.json();
      state.samples = data.samples;
      state.predictions = data.predictions;
      state.metrics = data.metrics;
      state.selectedIdx = Number(state.samples[0]?.idx ?? 0);
      populateFilters();
      ["search", "labelFilter", "probeFilter", "sortMode", "rowFilter"].forEach(id => {
        document.getElementById(id).addEventListener("input", render);
        document.getElementById(id).addEventListener("change", render);
      });
      render();
    }
    boot().catch(err => {
      document.getElementById("status").textContent = `Failed to load: ${err}`;
    });
  </script>
</body>
</html>
"""


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_samples(annotation_csv: Path) -> list[dict]:
    samples = read_csv_rows(annotation_csv)
    if len(samples) != 200:
        raise ValueError(f"Expected 200 MSTS samples, found {len(samples)} in {annotation_csv}")
    return samples


def evaluate_probe_specs(
    *,
    probe_specs: list[ProbeSpec],
    feature_name: str,
    annotation_csv: Path,
    msts_run_dir: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = msts_run_dir / "supervision_dataset.pt"
    annotated_path = output_dir / "supervision_dataset_annotated.pt"
    if not annotated_path.exists():
        build_annotated_msts_dataset(
            supervision_path=str(supervision_path),
            annotation_path=str(annotation_csv),
            output_path=str(annotated_path),
        )

    payload = torch.load(annotated_path, map_location="cpu", weights_only=False)
    labels = payload["y"].float()
    features = _select_features(payload, [feature_name])
    device = torch.device(device_name)

    predictions: list[dict] = []
    metrics_rows: list[dict] = []
    seed_prediction_tensors: list[torch.Tensor] = []

    for spec in probe_specs:
        if not spec.checkpoint_path.exists():
            raise FileNotFoundError(spec.checkpoint_path)
        model, checkpoint, config = _load_model_for_checkpoint(
            checkpoint_path=spec.checkpoint_path,
            device=device,
        )
        x = _apply_checkpoint_normalization(features, checkpoint)
        preds = _predict_all(model=model, x=x, batch_size=batch_size, device=device)
        seed_prediction_tensors.append(preds)
        metrics = _metrics(preds, labels, num_bins=10)
        metrics_row = {
            "probe_name": spec.name,
            "feature_name": feature_name,
            "checkpoint_path": str(spec.checkpoint_path),
            "model_type": config.get("model_type") or "mlp",
            **metrics,
        }
        metrics_rows.append(metrics_row)
        for idx, pred in enumerate(preds.tolist()):
            predictions.append(
                {
                    "idx": idx,
                    "probe_name": spec.name,
                    "feature_name": feature_name,
                    "prediction": float(pred),
                    "label": float(labels[idx].item()),
                }
            )

    if len(seed_prediction_tensors) > 1:
        ensemble = torch.stack(seed_prediction_tensors, dim=0).mean(dim=0)
        metrics = _metrics(ensemble, labels, num_bins=10)
        metrics_rows.append(
            {
                "probe_name": "ensemble_mean",
                "feature_name": feature_name,
                "checkpoint_path": ";".join(str(spec.checkpoint_path) for spec in probe_specs),
                "model_type": "mlp",
                **metrics,
            }
        )
        for idx, pred in enumerate(ensemble.tolist()):
            predictions.append(
                {
                    "idx": idx,
                    "probe_name": "ensemble_mean",
                    "feature_name": feature_name,
                    "prediction": float(pred),
                    "label": float(labels[idx].item()),
                }
            )

    write_csv(
        output_dir / "predictions.csv",
        predictions,
        ["idx", "probe_name", "feature_name", "prediction", "label"],
    )
    write_csv(
        output_dir / "metrics.csv",
        metrics_rows,
        [
            "probe_name",
            "feature_name",
            "checkpoint_path",
            "model_type",
            "brier",
            "ece",
            "auroc",
            "mean_prediction",
            "mean_label",
            "num_samples",
            "num_positive",
            "num_negative",
        ],
    )
    write_json(
        output_dir / "metadata.json",
        {
            "feature_name": feature_name,
            "annotation_csv": str(annotation_csv),
            "annotated_dataset_path": str(annotated_path),
            "num_samples": int(labels.numel()),
            "probe_specs": [
                {"name": spec.name, "checkpoint_path": str(spec.checkpoint_path)}
                for spec in probe_specs
            ],
        },
    )
    return {"predictions": predictions, "metrics": metrics_rows}


class MstsAllDatasetsHandler(BaseHTTPRequestHandler):
    app_data: dict = {}

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(HTML_PAGE)
            return
        if parsed.path == "/api/data":
            self.send_json(self.app_data)
            return
        if parsed.path == "/image":
            query = parse_qs(parsed.query)
            idx_values = query.get("idx", [])
            if not idx_values:
                self.send_text("Missing idx", status=400, content_type="text/plain; charset=utf-8")
                return
            try:
                idx = int(idx_values[0])
            except ValueError:
                self.send_text("Invalid idx", status=400, content_type="text/plain; charset=utf-8")
                return
            samples_by_idx = {int(sample["idx"]): sample for sample in self.app_data["samples"]}
            sample = samples_by_idx.get(idx)
            if sample is None:
                self.send_text("Unknown sample", status=404, content_type="text/plain; charset=utf-8")
                return
            image_path = Path(sample["image_path"])
            if not image_path.exists():
                self.send_text(
                    f"Image missing: {html.escape(str(image_path))}",
                    status=404,
                    content_type="text/plain; charset=utf-8",
                )
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            data = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_text("Not found", status=404, content_type="text/plain; charset=utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and browse the all-datasets LLaVA final prompt token probe on MSTS."
    )
    parser.add_argument("--annotation_csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--msts_run_dir", type=Path, default=Path(DEFAULT_MSTS_RUN_DIR))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8058)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.annotation_csv)
    eval_data = evaluate_probe_specs(
        probe_specs=default_probe_specs(),
        feature_name=args.feature_name,
        annotation_csv=args.annotation_csv,
        msts_run_dir=args.msts_run_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    MstsAllDatasetsHandler.app_data = {
        "samples": samples,
        "predictions": eval_data["predictions"],
        "metrics": eval_data["metrics"],
    }
    server = ThreadingHTTPServer((args.host, args.port), MstsAllDatasetsHandler)
    print(f"Loaded {len(samples)} samples and {len(eval_data['predictions'])} predictions")
    print(f"Predictions written to {args.output_dir / 'predictions.csv'}")
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
