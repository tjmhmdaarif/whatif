"""Run a stand-in Ollama server for manual testing.

Usage:
    python scripts/fake_ollama.py [port]

Prints the host URL on startup. Point whatif.toml's ollama.host at it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 32


def fake_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send(
                {
                    "models": [
                        {"name": "llama3.1:latest"},
                        {"name": "nomic-embed-text:latest"},
                    ]
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8")) if raw else {}

        if self.path == "/api/embed":
            value = payload.get("input", "")
            texts = value if isinstance(value, list) else [value]
            self._send({"embeddings": [fake_vector(str(t)) for t in texts]})
        elif self.path == "/api/embeddings":
            self._send({"embedding": fake_vector(str(payload.get("prompt", "")))})
        elif self.path == "/api/generate":
            self._send({"response": "A generated answer. [ESTIMATE]"})
        else:
            self._send({"error": "not found"}, status=404)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11434
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Fake Ollama listening on http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
