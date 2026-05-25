from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote, urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
BY_UID = {str(s["uid"]): s for s in DATA["samples"]}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith('/image/'):
            uid = unquote(parsed.path[len('/image/'):])
            kind = parse_qs(parsed.query).get('kind', ['clean'])[0]
            sample = BY_UID.get(uid)
            if not sample:
                self.send_error(404, 'unknown sample')
                return
            key = 'corrupted_image_path' if kind == 'corrupted' else 'clean_image_path'
            path = Path(sample.get(key, ''))
            if not path.exists() or not path.is_file():
                self.send_error(404, 'image not found')
                return
            ctype = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.end_headers()
            with path.open('rb') as f:
                self.wfile.write(f.read())
            return
        return super().do_GET()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8766'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f"Serving {ROOT} at http://127.0.0.1:{port}")
    server.serve_forever()
