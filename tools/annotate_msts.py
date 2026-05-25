"""Local web app for safe/unsafe annotation of VLM responses.

Usage:
    python tools/annotate_msts.py [--csv PATH] [--host 127.0.0.1] [--port 8765]

Keys: 0 = unsafe, 1 = safe, u = undo, ArrowLeft/ArrowRight = navigate without labeling.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SERVER_SIGNATURE = "MSTS-ANNOTATE-SERVER/1"

DEFAULT_CSV = "/projects/prjs2014/msts/llava-hf/llava-1.5-7b-hf/run_14423ff29d/annotation_template.csv"
LABEL_COL = "manual_label"
IMAGE_COL = "image_path"
PROMPT_COL = "prompt_text"
RESPONSE_COL = "response"


class CSVStore:
    """In-memory CSV with atomic on-disk writes after every label."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        for col in (LABEL_COL, IMAGE_COL, PROMPT_COL, RESPONSE_COL):
            if col not in self.fieldnames:
                raise SystemExit(f"CSV missing required column: {col!r}")

    def first_unlabeled(self) -> int:
        for i, r in enumerate(self.rows):
            if not (r.get(LABEL_COL) or "").strip():
                return i
        return len(self.rows) - 1  # all labeled — show last

    def label_counts(self) -> dict:
        labeled = sum(1 for r in self.rows if (r.get(LABEL_COL) or "").strip())
        return {"labeled": labeled, "total": len(self.rows)}

    def set_label(self, idx: int, value: str | None) -> None:
        with self._lock:
            if not (0 <= idx < len(self.rows)):
                raise IndexError(idx)
            self.rows[idx][LABEL_COL] = "" if value is None else str(value)
            self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.path)

    def row_payload(self, idx: int) -> dict:
        idx = max(0, min(idx, len(self.rows) - 1))
        r = self.rows[idx]
        return {
            "idx": idx,
            "total": len(self.rows),
            "labeled": self.label_counts()["labeled"],
            "prompt": r.get(PROMPT_COL, ""),
            "response": r.get(RESPONSE_COL, ""),
            "label": (r.get(LABEL_COL) or "").strip(),
            "hazard_category": r.get("hazard_category", ""),
            "hazard_subcategory": r.get("hazard_subcategory", ""),
            "image_path": r.get(IMAGE_COL, ""),
        }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MSTS Annotation</title>
<style>
  :root { --bg:#111; --fg:#eee; --muted:#888; --safe:#2ecc71; --unsafe:#e74c3c; --card:#1c1c1c; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--fg); }
  header { padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
           border-bottom: 1px solid #333; }
  header h1 { font-size: 16px; margin: 0; font-weight: 500; }
  header .progress { color: var(--muted); font-variant-numeric: tabular-nums; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px;
         max-width: 1600px; margin: 0 auto; }
  .img-wrap { background: var(--card); border-radius: 8px; padding: 12px; display: flex;
              align-items: center; justify-content: center; min-height: 300px; }
  .img-wrap img { max-width: 100%; max-height: 75vh; border-radius: 4px; }
  .text-wrap { background: var(--card); border-radius: 8px; padding: 16px;
               display: flex; flex-direction: column; gap: 14px; }
  .field-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
                 color: var(--muted); margin-bottom: 4px; }
  .field-value { font-size: 15px; line-height: 1.5; white-space: pre-wrap; }
  .prompt { font-weight: 600; }
  .meta { color: var(--muted); font-size: 12px; }
  .controls { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
  button { padding: 10px 16px; border: none; border-radius: 6px; cursor: pointer;
           font-size: 14px; font-weight: 600; color: white; }
  .btn-unsafe { background: var(--unsafe); }
  .btn-safe { background: var(--safe); }
  .btn-undo, .btn-nav { background: #333; color: var(--fg); font-weight: 500; }
  .current-label { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }
  .current-label.safe { background: var(--safe); color: white; }
  .current-label.unsafe { background: var(--unsafe); color: white; }
  .current-label.empty { background: #333; color: var(--muted); }
  kbd { background: #222; border: 1px solid #444; border-radius: 3px; padding: 1px 6px;
        font-size: 11px; font-family: monospace; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <h1>MSTS safety annotation</h1>
  <div class="progress" id="progress">—</div>
</header>
<main>
  <div class="img-wrap"><img id="img" alt="image"></div>
  <div class="text-wrap">
    <div class="meta" id="meta"></div>
    <div>
      <div class="field-label">Prompt</div>
      <div class="field-value prompt" id="prompt"></div>
    </div>
    <div>
      <div class="field-label">Model response</div>
      <div class="field-value" id="response"></div>
    </div>
    <div>
      <div class="field-label">Current label</div>
      <span class="current-label empty" id="curlabel">unlabeled</span>
    </div>
    <div class="controls">
      <button class="btn-unsafe" onclick="label(0)">Unsafe (0)</button>
      <button class="btn-safe" onclick="label(1)">Safe (1)</button>
      <button class="btn-undo" onclick="undo()">Undo (u)</button>
      <button class="btn-nav" onclick="nav(-1)">← Prev</button>
      <button class="btn-nav" onclick="nav(1)">Next →</button>
    </div>
    <div class="hint">
      <kbd>0</kbd> unsafe &nbsp; <kbd>1</kbd> safe &nbsp; <kbd>u</kbd> undo &nbsp;
      <kbd>←</kbd>/<kbd>→</kbd> navigate without labeling
    </div>
  </div>
</main>
<script>
let curIdx = 0;
let lastLabeled = null;  // {idx, prev} for undo

async function fetchRow(idx) {
  const url = idx === null ? "/row?next_unlabeled=1" : `/row?idx=${idx}`;
  const r = await fetch(url);
  return r.json();
}
function render(data) {
  curIdx = data.idx;
  document.getElementById("progress").textContent =
    `row ${data.idx + 1} / ${data.total}  ·  labeled ${data.labeled} / ${data.total}`;
  document.getElementById("prompt").textContent = data.prompt;
  document.getElementById("response").textContent = data.response;
  document.getElementById("meta").textContent =
    `idx ${data.idx} · ${data.hazard_category}${data.hazard_subcategory ? " / " + data.hazard_subcategory : ""}`;
  const lbl = document.getElementById("curlabel");
  if (data.label === "1") { lbl.textContent = "safe (1)"; lbl.className = "current-label safe"; }
  else if (data.label === "0") { lbl.textContent = "unsafe (0)"; lbl.className = "current-label unsafe"; }
  else { lbl.textContent = "unlabeled"; lbl.className = "current-label empty"; }
  document.getElementById("img").src = `/image?idx=${data.idx}&t=${Date.now()}`;
}
async function label(value) {
  const prev = (document.getElementById("curlabel").textContent.includes("safe (1)") ? "1"
              : document.getElementById("curlabel").textContent.includes("unsafe (0)") ? "0" : "");
  const r = await fetch("/label", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({idx: curIdx, label: String(value)}),
  });
  const data = await r.json();
  lastLabeled = {idx: curIdx, prev: prev};
  // advance to next unlabeled
  const next = await fetchRow(null);
  // if everything labeled, /row?next_unlabeled=1 returns the last row; that's fine
  render(next);
}
async function undo() {
  if (!lastLabeled) {
    // fallback: just clear current label
    await fetch("/label", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({idx: curIdx, label: null}),
    });
    render(await fetchRow(curIdx));
    return;
  }
  await fetch("/label", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({idx: lastLabeled.idx, label: lastLabeled.prev || null}),
  });
  const target = lastLabeled.idx;
  lastLabeled = null;
  render(await fetchRow(target));
}
async function nav(delta) {
  const target = Math.max(0, curIdx + delta);
  render(await fetchRow(target));
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "0") { e.preventDefault(); label(0); }
  else if (e.key === "1") { e.preventDefault(); label(1); }
  else if (e.key === "u" || e.key === "U") { e.preventDefault(); undo(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); nav(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); nav(1); }
});
(async () => { render(await fetchRow(null)); })();
</script>
</body>
</html>
"""


def make_handler(store: CSVStore, allowed_images: set[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            pass

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, ctype: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if u.path == "/health":
                self._send_json({"signature": SERVER_SIGNATURE, "csv": str(store.path)})
                return
            if u.path == "/row":
                if q.get("next_unlabeled", ["0"])[0] == "1":
                    idx = store.first_unlabeled()
                else:
                    try:
                        idx = int(q.get("idx", ["0"])[0])
                    except ValueError:
                        idx = 0
                self._send_json(store.row_payload(idx))
                return
            if u.path == "/image":
                try:
                    idx = int(q.get("idx", ["0"])[0])
                except ValueError:
                    self.send_error(400); return
                path = store.rows[idx].get(IMAGE_COL, "")
                if path not in allowed_images or not os.path.isfile(path):
                    self.send_error(404); return
                ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                with open(path, "rb") as f:
                    data = f.read()
                self._send_bytes(data, ctype)
                return
            self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/label":
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            idx = int(payload.get("idx", 0))
            label = payload.get("label")
            if label not in (None, "0", "1", 0, 1):
                self._send_json({"error": "label must be 0, 1, or null"}, status=400); return
            store.set_label(idx, None if label is None else str(label))
            self._send_json({"ok": True, **store.row_payload(idx)})

    return Handler


def _probe_existing(host: str, port: int, expected_csv: Path) -> tuple[bool, str]:
    """Check whether something on (host, port) is already our server.
    Returns (is_ours, info_string)."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.5) as r:
            data = json.loads(r.read())
        if data.get("signature") == SERVER_SIGNATURE:
            same = Path(data.get("csv", "")).resolve() == expected_csv.resolve()
            return True, data.get("csv", "?") + ("" if same else "  (different CSV!)")
    except Exception:
        pass
    return False, ""


def _try_bind(host: str, port: int) -> ThreadingHTTPServer | None:
    """Attempt to bind a ThreadingHTTPServer to (host, port). None if port is taken."""
    try:
        return ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)
    except OSError as e:
        if e.errno in (socket.EADDRINUSE if hasattr(socket, "EADDRINUSE") else 98, 98, 48):
            return None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8765, type=int)
    ap.add_argument("--port-search-range", default=20, type=int,
                    help="If --port is busy, try up to this many subsequent ports.")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    store = CSVStore(args.csv)
    allowed = {r.get(IMAGE_COL, "") for r in store.rows if r.get(IMAGE_COL)}

    # If our requested port already speaks our protocol, just reuse it.
    is_ours, info = _probe_existing(args.host, args.port, args.csv)
    if is_ours:
        counts = store.label_counts()
        print(f"An annotation server is already running on {args.host}:{args.port}")
        print(f"  CSV: {info}")
        print(f"  progress: {counts['labeled']} / {counts['total']} labeled")
        print(f"\nJust open:  http://{args.host}:{args.port}/")
        print(f"(remote? tunnel with:  ssh -L {args.port}:127.0.0.1:{args.port} <this-host>)")
        return

    # Find a free port starting at --port, walking forward.
    chosen_port = None
    chosen_server = None
    for offset in range(args.port_search_range + 1):
        candidate = args.port + offset
        probe = _try_bind(args.host, candidate)
        if probe is not None:
            probe.server_close()
            chosen_port = candidate
            break
        # taken — see if it's another annotator we should point the user at
        is_ours, info = _probe_existing(args.host, candidate, args.csv)
        if is_ours:
            counts = store.label_counts()
            print(f"An annotation server is already running on {args.host}:{candidate}")
            print(f"  CSV: {info}")
            print(f"  progress: {counts['labeled']} / {counts['total']} labeled")
            print(f"\nJust open:  http://{args.host}:{candidate}/")
            print(f"(remote? tunnel with:  ssh -L {candidate}:127.0.0.1:{candidate} <this-host>)")
            return
    if chosen_port is None:
        raise SystemExit(
            f"Could not bind any port in range {args.port}..{args.port + args.port_search_range}"
        )

    if chosen_port != args.port:
        print(f"Port {args.port} was busy — using {chosen_port} instead.")

    counts = store.label_counts()
    print(f"Loaded {args.csv}")
    print(f"  rows: {counts['total']}, already labeled: {counts['labeled']}")
    print(f"  resuming at row {store.first_unlabeled()}")
    print(f"\nServing on http://{args.host}:{chosen_port}/  (Ctrl-C to stop)")
    if args.host == "127.0.0.1":
        print(f"  Remote? tunnel with:")
        print(f"  ssh -L {chosen_port}:127.0.0.1:{chosen_port} <this-host>\n")

    httpd = ThreadingHTTPServer((args.host, chosen_port), make_handler(store, allowed))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
