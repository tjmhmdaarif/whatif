"""Run a stand-in Ollama server for manual testing.

Usage:
    python scripts/fake_ollama.py [port]

Prints the host URL on startup. Point whatif.toml's ollama.host at it.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 32

# The names this stand-in reports itself under. They are deliberately not the
# names of the real models, so that a report produced against this server always
# says which embedder produced it.
STANDIN_EMBED_MODEL = "standin-hash-embed"
STANDIN_LLM_MODEL = "standin-canned-llm"

# Common English words carry no signal about what a decision was about. A real
# embedding model does not confuse "the rest of the team would move to the new
# tool" with "the extra tuition for in-person classes would be worth it", so the
# stand-in drops stopwords too. Otherwise every pair of English sentences looks
# 80% similar and the evaluation number means nothing.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing have has had having
    i me my mine we our ours you your yours he him his she her hers it its
    they them their theirs who whom which what when where why how
    in on at to for of with by from as into about over under again further
    not no nor so too very can will just should now
    would could shall may might must
    one two three four five
    s t don
    """.split()
)


def content_words(text: str) -> list[str]:
    """Return the meaningful words of a text, lowercased."""
    words = []
    for raw in text.lower().split():
        word = raw.strip(".,!?;:'\"()[]")
        if not word or word in STOPWORDS:
            continue
        words.append(word)
    return words


def fake_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding built from the content words."""
    words = content_words(text)
    raw = [0.0] * DIM
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        raw[digest[0] % DIM] += digest[1] / 255.0
    if not any(raw):
        raw[0] = 1.0
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def generated_brief(prompt: str) -> str:
    """Build a canned brief that cites a note id the prompt actually allowed.

    A real model would read the prompt; this one just picks the first id out of
    the "NOTE IDS YOU MAY CITE" section, which is enough to exercise the whole
    pipeline including tag validation.
    """
    allowed: list[str] = []
    marker = "=== NOTE IDS YOU MAY CITE ==="
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
        for line in tail.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                break
            allowed = [part.strip() for part in line.split(",") if part.strip()]
            break

    cited = allowed[0] if allowed else "2026-01-15-course-choice"
    relevance = (
        f"{cited}: A past decision where the cheaper option lost for reasons that "
        f"were not about price"
        if allowed
        else ""
    )

    return f"""<decision_summary>
The user is choosing between two options that differ mainly on price and speed [ESTIMATE]
The cheaper option scored better on price but the faster one carried the weighted total in the note [EVIDENCE: {cited}]
</decision_summary>
<similar_notes>
{relevance}
</similar_notes>
<counterargument>
A similar decision did not pay off the last time it was made [EVIDENCE: {cited}]
</counterargument>
<open_questions>
What is the real budget once the second option is included? [ESTIMATE]
How much does the speed matter after the deadline? [ESTIMATE]
</open_questions>"""


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
            # The stand-in reports itself under its own name rather than claiming
            # to be nomic-embed-text. Any evaluation run that uses it will name it
            # in the report, so a reader can never mistake these numbers for
            # numbers produced by a real embedding model.
            self._send(
                {
                    "models": [
                        {"name": f"{STANDIN_EMBED_MODEL}:latest"},
                        {"name": f"{STANDIN_LLM_MODEL}:latest"},
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
            self._send({"response": generated_brief(str(payload.get("prompt", "")))})
        else:
            self._send({"error": "not found"}, status=404)


class DualStackServer(HTTPServer):
    """HTTPServer that accepts both IPv6 and IPv4 loopback connections.

    On Windows, ``localhost`` resolves to ``::1`` before ``127.0.0.1``. A plain
    IPv4 server makes every request pay a connect timeout before falling back to
    IPv4, which is slow enough to look like the tool is hanging.
    """

    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11434
    try:
        server = DualStackServer(("::1", port), Handler)
    except OSError:
        server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Fake Ollama listening on http://localhost:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
