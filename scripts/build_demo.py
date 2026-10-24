"""Build a static, self-contained demo of the WhatIf web interface.

    python scripts/build_demo.py [--out promo/demo]

The landing page links to a working prototype, but the real interface is a Python
server that needs Ollama. This script runs both for real -- the stand-in Ollama
from ``scripts/fake_ollama.py`` plus the actual WhatIf code, against a temporary
copy of the sample vault -- asks the code for every response the interface can
produce, and writes the answers into the output folder.

The result needs no server of any kind. ``data.js`` holds the captured payloads,
``shim.js`` answers ``fetch()`` from that file, and ``index.html`` is the real
interface with its asset paths made relative. Nothing under ``whatif/`` is
changed, so the demo cannot drift from the interface it copies.

The stand-in names itself honestly in every payload it touches, so a reader can
never mistake these numbers for numbers produced by a real embedding model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whatif.brief import BriefGenerator  # noqa: E402
from whatif.config import Config, load_config  # noqa: E402
from whatif.eval import format_report, run_eval  # noqa: E402
from whatif.webapp import WebApp  # noqa: E402

logger = logging.getLogger("build_demo")

WEB_ROOT = ROOT / "whatif" / "web"
SAMPLE_VAULT = ROOT / "vault"
DEMO_CONFIG = ROOT / "whatif.demo.toml"
FAKE_OLLAMA = ROOT / "scripts" / "fake_ollama.py"
DEFAULT_OUT = ROOT / "promo" / "demo"

DEMO_SCRIPTS = "\n".join(
    [
        '<script src="./data.js"></script>',
        '<script src="./shim.js"></script>',
        '<script src="./app.js"></script>',
        '<script src="./demo.js"></script>',
    ]
)

DEMO_STYLE = """<style>
  .demo-bar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 10px 18px;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--text-2);
    background: #0a100e;
    border-bottom: 1px solid var(--line);
  }
  .demo-bar code {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--accent);
  }
  .demo-bar a {
    color: var(--accent);
    text-decoration: none;
    white-space: nowrap;
  }
  .demo-bar a:hover { text-decoration: underline; }
  @media (max-width: 820px) {
    .demo-bar { flex-direction: column; align-items: flex-start; gap: 6px; }
  }
</style>"""

DEMO_BAR = """<div class="demo-bar">
  <span><strong>Interactive demo</strong> &mdash; a static snapshot of <code>whatif serve</code> running the 10-note sample vault. No server and no model is running here.</span>
  <a href="../">&larr; Back to the landing page</a>
</div>"""

SHIM_JS = '''/* Answer every API call from the captured payloads in data.js.

   The interface only ever talks to the network through fetch(), so overriding
   that one function turns the real app.js into a static page. Anything not in
   data.js is reported as a 404 rather than sent to a server that does not exist.
*/
(function () {
  var DATA = window.__WHATIF_DEMO__ || {};
  var JSON_HEADERS = { "Content-Type": "application/json; charset=utf-8" };

  function reply(status, payload) {
    return Promise.resolve(
      new Response(JSON.stringify(payload), { status: status, headers: JSON_HEADERS })
    );
  }

  window.fetch = function (input, init) {
    var url =
      typeof input === "string" ? input : (input && input.url) || "";
    var method = (
      (init && init.method) ||
      (input && input.method) ||
      "GET"
    ).toUpperCase();
    var key = method + " " + url.split("?")[0];

    if (Object.prototype.hasOwnProperty.call(DATA, key)) {
      return reply(200, DATA[key]);
    }
    if (method !== "GET") {
      var readKey = "GET " + url.split("?")[0];
      if (Object.prototype.hasOwnProperty.call(DATA, readKey)) {
        return reply(200, DATA[readKey]);
      }
    }
    return reply(404, { error: "Not part of this static demo: " + key, status: 404 });
  };
})();
'''

DEMO_JS = '''/* Replace the header's "Open localhost" button with a link back to the landing page.

   The real interface opens http://127.0.0.1:8765, which only exists if the
   visitor is running the tool themselves. Registered after app.js so it runs
   after wire() has attached its handler; swapping the node discards that handler.
*/
document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("btnServe");
  if (!button) return;
  var link = document.createElement("a");
  link.id = "btnServe";
  link.className = "btn btn-ghost";
  link.href = "../";
  link.textContent = "Landing page";
  button.replaceWith(link);
});
'''


# ---------------------------------------------------------------------------
# Stand-in server and scratch vault
# ---------------------------------------------------------------------------


def load_stand_in() -> ModuleType:
    """Import scripts/fake_ollama.py by path so the demo uses the same stand-in."""
    spec = importlib.util.spec_from_file_location("whatif_fake_ollama", FAKE_OLLAMA)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import the stand-in from {FAKE_OLLAMA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_stand_in(module: ModuleType) -> tuple[HTTPServer, int]:
    """Start the stand-in on an ephemeral IPv4 loopback port.

    An explicit 127.0.0.1 avoids the Windows habit of resolving ``localhost`` to
    ``::1`` first, which would make every request pay a connect timeout.
    """
    server = HTTPServer(("127.0.0.1", 0), module.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def prepare_vault(scratch: Path) -> Path:
    """Copy the sample decisions into a scratch vault.

    Building against a copy keeps the demo's index, briefs and canvases out of
    the working vault, and guarantees a clean embedding dimension no matter what
    the working index was built with.
    """
    vault = scratch / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SAMPLE_VAULT / "Decisions", vault / "Decisions", dirs_exist_ok=True)
    return vault


# ---------------------------------------------------------------------------
# Payload capture
# ---------------------------------------------------------------------------


def brief_payload(brief: Any) -> dict[str, Any]:
    """Flatten a generated brief into the shape the interface renders.

    Mirrors the POST handler in whatif/webapp.py. The rendered Markdown is left
    out: the interface never reads it, and it is the largest field.
    """
    return {
        "path": f"vault/Briefs/{brief.decision.id}.md",
        "sections": brief.sections,
        "relevance": brief.relevance,
        "similar": [note.to_dict() for note in brief.similar],
        "failed": [item.to_dict() for item in brief.failed_assumptions],
        "scoring": brief.scoring.to_dict() if brief.scoring else {},
        "leading": brief.leading,
        "validation": brief.validation.as_dict() if brief.validation else {},
        "retries": brief.retries,
    }


def build_payloads(
    config: Config, app: WebApp, declared_host: str
) -> dict[str, Any]:
    """Return every response the interface fetches, keyed by "METHOD /path"."""
    payloads: dict[str, Any] = {}

    health = app.health_payload()
    health["vault"] = "vault"
    health["host"] = declared_host
    payloads["GET /api/health"] = health

    decisions, errors = app.decisions()
    payloads["GET /api/notes"] = {
        "notes": [app.note_summary(d) for d in decisions],
        "errors": errors,
    }

    for decision in decisions:
        payloads[f"GET /api/scoring/{decision.id}"] = app.scoring_payload(decision)
        payloads[f"GET /api/retrieval/{decision.id}"] = app.retrieval_payload(decision)
        payloads[f"GET /api/canvas/{decision.id}"] = app.canvas_payload(decision)

    with app.store() as store:
        client = app.ollama_client(config.models.llm_model)
        retriever = app.retriever(store)
        generator = BriefGenerator(
            config, client, retriever, note_bodies=store.get_note_bodies()
        )
        for decision in decisions:
            full = brief_payload(generator.generate(decision, k=5))
            # A generated brief answers both the read and the write. The read
            # carries exists:true so the brief panel renders it in full.
            payloads[f"POST /api/brief/{decision.id}"] = full
            payloads[f"GET /api/brief/{decision.id}"] = {"exists": True, **full}

        client_embed = app.ollama_client(config.models.embed_model)
        report = run_eval(config, store, embedder=client_embed.embed)

    eval_path = "vault/.whatif/eval-results.json"
    payloads["GET /api/eval"] = {
        "exists": True,
        "path": eval_path,
        "results": report.as_dict(),
    }
    payloads["POST /api/eval"] = {
        "path": eval_path,
        "text": format_report(report),
        "results": report.as_dict(),
    }
    payloads["POST /api/index"] = index_stats_payload(config)

    return payloads


def index_stats_payload(config: Config) -> dict[str, Any]:
    """Report an index run over the already-indexed scratch vault."""
    from whatif.store import index_decisions

    return index_decisions(config).as_dict()


def scrub(value: Any, vault: Path) -> Any:
    """Replace the scratch vault's absolute path with a stable label.

    Every path the interface shows comes from the config, so a scratch vault
    would otherwise leak a local directory (and a username) into a public page.
    """
    if isinstance(value, dict):
        return {key: scrub(item, vault) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, vault) for item in value]
    if isinstance(value, str):
        for form in {str(vault), vault.as_posix()}:
            if form:
                value = value.replace(form, "vault")
        return value
    return value


# ---------------------------------------------------------------------------
# Static site
# ---------------------------------------------------------------------------


def build_index_html() -> str:
    """Turn the real interface's index.html into the demo's index.html."""
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    replacements = [
        (
            '<link rel="stylesheet" href="/static/styles.css">',
            '<link rel="stylesheet" href="./styles.css">',
        ),
        ("<script src=\"/static/app.js\"></script>", DEMO_SCRIPTS),
        ("<title>WhatIf</title>", "<title>WhatIf &mdash; interactive demo</title>"),
        ("<body>", "<body>\n" + DEMO_STYLE + "\n" + DEMO_BAR),
        (
            ">Open</button>",
            ">Landing page</button>",
        ),
    ]
    for old, new in replacements:
        if old not in source:
            raise SystemExit(f"web/index.html no longer contains: {old!r}")
        source = source.replace(old, new, 1)
    return source


def write_site(out: Path, payloads: dict[str, Any]) -> None:
    """Write the demo folder."""
    out.mkdir(parents=True, exist_ok=True)

    data = "window.__WHATIF_DEMO__ = " + json.dumps(
        payloads, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"
    (out / "data.js").write_text(data, encoding="utf-8")
    (out / "shim.js").write_text(SHIM_JS, encoding="utf-8")
    (out / "demo.js").write_text(DEMO_JS, encoding="utf-8")
    (out / "index.html").write_text(build_index_html(), encoding="utf-8")

    for name in ("styles.css", "app.js"):
        shutil.copyfile(WEB_ROOT / name, out / name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Folder to write (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    stand_in = load_stand_in()
    server, port = start_stand_in(stand_in)

    scratch = Path(tempfile.mkdtemp(prefix="whatif-demo-", dir=str(ROOT.parent)))
    try:
        vault = prepare_vault(scratch)
        config = load_config(DEMO_CONFIG, vault_path=vault)
        declared_host = config.ollama.host
        config.ollama.host = f"http://127.0.0.1:{port}"

        from whatif.store import index_decisions

        stats = index_decisions(config)
        logger.warning("scratch index: %s", stats.summary())

        app = WebApp(config)
        payloads = scrub(build_payloads(config, app, declared_host), vault)
        write_site(args.out, payloads)

        size = sum(f.stat().st_size for f in args.out.iterdir())
        logger.warning(
            "wrote %s (%d routes, %.0f KB)",
            args.out,
            len(payloads),
            size / 1024,
        )
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(scratch, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
