from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_ANNOTATION_CSV = Path(
    "/projects/prjs2014/msts/llava-hf/llava-1.5-7b-hf/run_14423ff29d/annotation_template.csv"
)
DEFAULT_RESULTS_ROOT = Path("/projects/prjs2014/tgroot/probe_results/msts_eval/llava")
DEFAULT_FEATURE_GROUP = "lm_answer_mean_lasttok"


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MSTS LLaVA lm_answer Probe Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #637083;
      --line: #d7dde6;
      --accent: #0f766e;
      --accent-soft: #d7f2ed;
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
      height: 56px;
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
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 410px) minmax(0, 1fr);
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      min-height: calc(100vh - 56px);
      position: sticky;
      top: 56px;
      align-self: start;
      max-height: calc(100vh - 56px);
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
    .sample-list {
      overflow: auto;
      padding: 8px;
      display: grid;
      gap: 6px;
    }
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
    .sample-button.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .sample-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-weight: 650;
    }
    .sample-prompt {
      margin-top: 4px;
      color: var(--muted);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .workspace {
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
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
    h2 {
      margin: 0 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }
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
    .meta span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }
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
    .prompt {
      font-size: 18px;
      font-weight: 650;
      margin: 0 0 10px;
    }
    .answer {
      white-space: pre-wrap;
      margin: 0;
      color: #293441;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }
    th {
      position: sticky;
      top: 56px;
      background: var(--panel);
      z-index: 1;
      color: var(--muted);
      font-weight: 650;
    }
    .prob-cell {
      display: grid;
      grid-template-columns: 54px 1fr;
      align-items: center;
      gap: 8px;
      min-width: 160px;
    }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #e4e9f0;
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: var(--accent);
    }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside {
        position: static;
        min-height: auto;
        max-height: 420px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .sample-detail { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>MSTS LLaVA lm_answer Probe Viewer</h1>
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
        <h2>Probe Predictions</h2>
        <div class="toolbar">
          <select id="sourceFilter"><option value="all">All training datasets</option></select>
          <select id="modelFilter"><option value="all">All probe models</option></select>
          <select id="featureFilter"><option value="all">All lm_answer features</option></select>
          <select id="sortMode">
            <option value="source">Sort by dataset</option>
            <option value="prediction_desc">Highest probability first</option>
            <option value="prediction_asc">Lowest probability first</option>
          </select>
        </div>
        <h2>Mean Predictions By Label</h2>
        <table>
          <thead>
            <tr>
              <th>Training Dataset</th>
              <th>Probe</th>
              <th>Feature</th>
              <th>Safe Mean</th>
              <th>Unsafe Mean</th>
              <th>Unsafe - Safe</th>
              <th>Counts</th>
            </tr>
          </thead>
          <tbody id="aggregateBody"></tbody>
        </table>
        <h2 style="margin-top:14px;">Selected Sample Predictions</h2>
        <table>
          <thead>
            <tr>
              <th>Training Dataset</th>
              <th>Probe</th>
              <th>Feature</th>
              <th>Prediction</th>
              <th>Brier</th>
              <th>AUROC</th>
              <th>ECE</th>
            </tr>
          </thead>
          <tbody id="predictionBody"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const state = { samples: [], predictions: [], selectedIdx: 0 };
    const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "";
    const labelPill = (label) => {
      const cls = String(label) === "1" ? "ok" : "bad";
      return `<span class="pill ${cls}">${String(label) === "1" ? "safe / label 1" : "unsafe / label 0"}</span>`;
    };
    const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

    function populateFilters() {
      const sources = [...new Set(state.predictions.map(p => p.source_dataset))].sort();
      const models = [...new Set(state.predictions.map(p => p.model_type))].sort();
      const features = [...new Set(state.predictions.map(p => p.feature_name))].sort();
      fillSelect("sourceFilter", sources, "All training datasets");
      fillSelect("modelFilter", models, "All probe models");
      fillSelect("featureFilter", features, "All lm_answer features");
    }

    function fillSelect(id, values, label) {
      const el = document.getElementById(id);
      el.innerHTML = `<option value="all">${label}</option>` + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
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

    function renderPredictions() {
      const source = document.getElementById("sourceFilter").value;
      const model = document.getElementById("modelFilter").value;
      const feature = document.getElementById("featureFilter").value;
      const sortMode = document.getElementById("sortMode").value;
      let rows = state.predictions.filter(p => Number(p.idx) === state.selectedIdx);
      if (source !== "all") rows = rows.filter(p => p.source_dataset === source);
      if (model !== "all") rows = rows.filter(p => p.model_type === model);
      if (feature !== "all") rows = rows.filter(p => p.feature_name === feature);
      if (sortMode === "prediction_desc") rows.sort((a, b) => b.prediction - a.prediction);
      if (sortMode === "prediction_asc") rows.sort((a, b) => a.prediction - b.prediction);
      if (sortMode === "source") rows.sort((a, b) =>
        `${a.source_dataset} ${a.model_type} ${a.feature_name}`.localeCompare(`${b.source_dataset} ${b.model_type} ${b.feature_name}`)
      );
      document.getElementById("predictionBody").innerHTML = rows.map(p => {
        const width = Math.max(0, Math.min(100, Number(p.prediction) * 100));
        return `<tr>
          <td>${escapeHtml(p.source_dataset)}</td>
          <td>${escapeHtml(p.model_type)}</td>
          <td>${escapeHtml(p.feature_name)}</td>
          <td><div class="prob-cell"><span>${fmt(p.prediction)}</span><div class="bar"><span style="width:${width}%"></span></div></div></td>
          <td>${fmt(p.brier)}</td>
          <td>${fmt(p.auroc)}</td>
          <td>${fmt(p.ece)}</td>
        </tr>`;
      }).join("");
    }

    function renderAggregates() {
      const source = document.getElementById("sourceFilter").value;
      const model = document.getElementById("modelFilter").value;
      const feature = document.getElementById("featureFilter").value;
      const sampleByIdx = new Map(state.samples.map(s => [Number(s.idx), s]));
      let rows = state.predictions;
      if (source !== "all") rows = rows.filter(p => p.source_dataset === source);
      if (model !== "all") rows = rows.filter(p => p.model_type === model);
      if (feature !== "all") rows = rows.filter(p => p.feature_name === feature);

      const grouped = new Map();
      for (const row of rows) {
        const key = `${row.source_dataset}||${row.model_type}||${row.feature_name}`;
        const sample = sampleByIdx.get(Number(row.idx));
        if (!sample) continue;
        if (!grouped.has(key)) {
          grouped.set(key, {
            source_dataset: row.source_dataset,
            model_type: row.model_type,
            feature_name: row.feature_name,
            safe_sum: 0,
            safe_n: 0,
            unsafe_sum: 0,
            unsafe_n: 0,
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

      const aggregates = [...grouped.values()].map(group => {
        const safeMean = group.safe_n ? group.safe_sum / group.safe_n : NaN;
        const unsafeMean = group.unsafe_n ? group.unsafe_sum / group.unsafe_n : NaN;
        return { ...group, safeMean, unsafeMean, gap: unsafeMean - safeMean };
      }).sort((a, b) =>
        `${a.source_dataset} ${a.model_type} ${a.feature_name}`.localeCompare(`${b.source_dataset} ${b.model_type} ${b.feature_name}`)
      );

      document.getElementById("aggregateBody").innerHTML = aggregates.map(row => `
        <tr>
          <td>${escapeHtml(row.source_dataset)}</td>
          <td>${escapeHtml(row.model_type)}</td>
          <td>${escapeHtml(row.feature_name)}</td>
          <td>${fmt(row.safeMean)}</td>
          <td>${fmt(row.unsafeMean)}</td>
          <td>${fmt(row.gap)}</td>
          <td><span class="muted">safe ${row.safe_n}, unsafe ${row.unsafe_n}</span></td>
        </tr>
      `).join("");
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
      state.selectedIdx = Number(state.samples[0]?.idx ?? 0);
      populateFilters();
      ["search", "labelFilter", "sourceFilter", "modelFilter", "featureFilter", "sortMode"].forEach(id => {
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


def to_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_app_data(annotation_csv: Path, results_root: Path, feature_group: str) -> dict:
    samples = read_csv_rows(annotation_csv)
    if len(samples) != 200:
        raise ValueError(f"Expected 200 MSTS samples, found {len(samples)} in {annotation_csv}")

    summary_path = results_root / "msts_probe_eval_summary.csv"
    summary_rows = read_csv_rows(summary_path)
    lm_answer_rows = [row for row in summary_rows if row.get("feature_group") == feature_group]
    predictions: list[dict] = []

    for row in lm_answer_rows:
        prediction_path = Path(row["result_dir"]) / "predictions.csv"
        for pred in read_csv_rows(prediction_path):
            predictions.append(
                {
                    "idx": int(pred["idx"]),
                    "prediction": float(pred["prediction"]),
                    "label": float(pred["label"]),
                    "source_dataset": row["source_dataset"],
                    "model_type": row["model_type"],
                    "feature_name": row["feature_name"],
                    "brier": to_float(row.get("brier")),
                    "ece": to_float(row.get("ece")),
                    "auroc": to_float(row.get("auroc")),
                }
            )

    return {"samples": samples, "predictions": predictions}


class MstslmAnswerHandler(BaseHTTPRequestHandler):
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
    parser = argparse.ArgumentParser(description="Browse MSTS samples with LLaVA lm_answer probe predictions.")
    parser.add_argument("--annotation_csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--feature_group", type=str, default=DEFAULT_FEATURE_GROUP)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8057)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MstslmAnswerHandler.app_data = load_app_data(
        annotation_csv=args.annotation_csv,
        results_root=args.results_root,
        feature_group=args.feature_group,
    )
    num_predictions = len(MstslmAnswerHandler.app_data["predictions"])
    server = ThreadingHTTPServer((args.host, args.port), MstslmAnswerHandler)
    print(f"Loaded {len(MstslmAnswerHandler.app_data['samples'])} samples and {num_predictions} predictions")
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
