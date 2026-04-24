"""Minimal web UI for search + ingestion without extra framework dependencies."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from src.search_ingestion import build_request, execute_search_ingestion_sync


def _render_form(result_json: str = "", error: str = "") -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Search And Ingest</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background: #f7f7f5; color: #1a1a1a; }}
    form {{ display: grid; gap: 0.9rem; max-width: 900px; }}
    label {{ display: grid; gap: 0.35rem; font-weight: 600; }}
    input, textarea {{ padding: 0.7rem; border: 1px solid #c8c8c8; border-radius: 8px; font: inherit; }}
    button {{ width: 220px; padding: 0.8rem 1rem; border: 0; border-radius: 999px; background: #14532d; color: white; font: inherit; }}
    pre {{ background: #101418; color: #d5f5e3; padding: 1rem; border-radius: 10px; overflow: auto; }}
    .error {{ color: #9f1239; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Search And Ingest</h1>
  <form method="post">
    <label>Keywords (comma separated)<input name="keywords" required /></label>
    <label>Document Types (comma separated)<input name="document_types" placeholder="pdf,image,office,text" /></label>
    <label>Extensions (comma separated)<input name="extensions" placeholder=".pdf,.docx,.png" /></label>
    <label>Input Location<input name="input_location" placeholder="data/raw or urls.txt" /></label>
    <label>URL Subdomain Pattern<input name="url_subdomain_pattern" placeholder="docs.example.com or *.example.com" /></label>
    <label>Output Location<input name="output_location" placeholder="./output" /></label>
    <label>Parse Method<input name="parse_method" value="auto" /></label>
    <label>Workers<input name="workers" value="1" /></label>
    <label><input type="checkbox" name="recursive" checked /> Recursive</label>
    <label><input type="checkbox" name="dry_run" /> Dry Run</label>
    <button type="submit">Run Search + Ingest</button>
  </form>
  <p class="error">{html.escape(error)}</p>
  <pre>{html.escape(result_json)}</pre>
</body>
</html>"""


class SearchIngestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send_html(_render_form())

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body)

        try:
            request = build_request(
                keywords=form.get("keywords", [""])[0],
                document_types=form.get("document_types", [""])[0],
                extensions=form.get("extensions", [""])[0],
                input_location=form.get("input_location", [""])[0] or None,
                url_subdomain_pattern=form.get("url_subdomain_pattern", [""])[0] or None,
                output_location=form.get("output_location", [""])[0] or None,
                parse_method=form.get("parse_method", ["auto"])[0],
                workers=int(form.get("workers", ["1"])[0]),
                recursive="recursive" in form,
                dry_run="dry_run" in form,
            )
            result = execute_search_ingestion_sync(request)
            payload = json.dumps(result.to_dict(), indent=2)
            self._send_html(_render_form(result_json=payload))
        except Exception as exc:
            self._send_html(_render_form(error=str(exc)))

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve_web_ui(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), SearchIngestHandler)
    print(f"Search UI listening on http://{host}:{port}")
    server.serve_forever()
