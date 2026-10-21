"""Start the whole demo in one command.

    python scripts/demo.py [--port 8765]

Starts the stand-in Ollama server (unless a real one is already answering) and
the WhatIf web interface, then waits until you press Ctrl+C.

If you have real Ollama running on port 11434, point whatif.toml at it and use
`whatif serve` on its own instead; this script will detect the running server
and leave it alone.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEMO_CONFIG = "whatif.demo.toml"
OLLAMA_PORT = 11434


def port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Return True if something is listening on the port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ollama_answers(port: int) -> bool:
    """Return True if the port speaks the Ollama API."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/api/tags", timeout=1.5
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_for(url: str, timeout: float = 15.0) -> bool:
    """Poll a URL until it answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the WhatIf demo locally.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the web interface")
    parser.add_argument(
        "--config",
        default=DEMO_CONFIG,
        help=f"Config file to use (default: {DEMO_CONFIG})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    standin: subprocess.Popen[bytes] | None = None

    if ollama_answers(OLLAMA_PORT):
        print(f"Using the Ollama server already running on port {OLLAMA_PORT}.")
    else:
        print(f"Starting the stand-in Ollama server on port {OLLAMA_PORT}...")
        standin = subprocess.Popen(
            [sys.executable, str(root / "scripts" / "fake_ollama.py"), str(OLLAMA_PORT)],
            cwd=str(root),
        )
        if not wait_for(f"http://localhost:{OLLAMA_PORT}/api/tags"):
            print("The stand-in server did not come up.", file=sys.stderr)
            if standin:
                standin.terminate()
            return 1

    # Make sure the vault is indexed before anyone opens the page.
    print("Indexing the vault...")
    index = subprocess.run(
        [sys.executable, "-m", "whatif.cli", "index", "--config", str(config_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    for line in index.stdout.splitlines():
        if "Indexed" in line:
            print(f"  {line.strip()}")
    if index.returncode != 0:
        print(index.stderr.strip()[:400], file=sys.stderr)

    url = f"http://127.0.0.1:{args.port}"
    print()
    print(f"  WhatIf is running at  {url}")
    print("  Press Ctrl+C to stop.")
    print()

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "whatif.cli",
            "serve",
            "--config",
            str(config_path),
            "--port",
            str(args.port),
        ],
        cwd=str(root),
    )

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
        if standin:
            standin.terminate()
            try:
                standin.wait(timeout=5)
            except subprocess.TimeoutExpired:
                standin.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
