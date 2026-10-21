"""Extract frames from the promotional video into still images.

The video is a WebM recorded by Playwright. There is no ffmpeg on this machine,
so the frames are pulled out with a browser: load the video into a <video>
element, seek, draw to a canvas, and POST each frame back to a tiny local
receiver that writes it to disk.

Usage:
    python scripts/extract_frames.py [video] [outdir] [t1,t2,...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class FrameReceiver(SimpleHTTPRequestHandler):
    """Writes POSTed frame bodies to disk."""

    outdir: Path = Path(".")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        query = parse_qs(urlparse(self.path).query)
        stamp = (query.get("t") or ["0"])[0]
        name = f"frame-{int(float(stamp)):03d}.jpg"
        (self.outdir / name).write_bytes(body)
        self.send_response(200)
        self._cors()
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


class VideoHost(SimpleHTTPRequestHandler):
    """Serves the video, plus a blank page on the same origin.

    The page and the video must share an origin, otherwise drawing the video into
    a canvas taints it and toBlob refuses to produce an image.
    """

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path in ("/", "/blank.html"):
            body = b"<!doctype html><title>frames</title><body></body>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "promo" / "whatif-promo.webm"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "promo" / "frames"
    times = (
        [float(t) for t in sys.argv[3].split(",")]
        if len(sys.argv) > 3
        else [4, 16, 24, 36, 46, 62, 74, 86, 96, 101]
    )

    if not video.exists():
        print(f"No video at {video}")
        return 1

    outdir.mkdir(parents=True, exist_ok=True)

    # Receiver for the frames.
    receiver = ThreadingHTTPServer(("127.0.0.1", 0), FrameReceiver)
    receiver.RequestHandlerClass.outdir = outdir
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    receiver_url = f"http://127.0.0.1:{receiver.server_port}"

    # Static server for the video, on the same origin as the blank page.
    static = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(VideoHost, directory=str(video.parent))
    )
    threading.Thread(target=static.serve_forever, daemon=True).start()
    static_url = f"http://127.0.0.1:{static.server_port}"

    script = f"""
    async (page) => {{
      const times = {json.dumps(times)};
      await page.goto('{static_url}/blank.html');
      const result = await page.evaluate(async ({{ times, videoUrl, receiver }}) => {{
        const v = document.createElement('video');
        v.src = videoUrl;
        v.muted = true;
        v.playsInline = true;
        const ready = await Promise.race([
          new Promise((r) => v.addEventListener('loadeddata', () => r(true), {{ once: true }})),
          new Promise((r) => setTimeout(() => r(false), 20000)),
        ]);
        if (!ready) return 'video failed to load';

        const c = document.createElement('canvas');
        c.width = 1280; c.height = 720;
        const ctx = c.getContext('2d');
        const pending = times.slice().sort((a, b) => a - b);
        const sent = [];

        await v.play();

        // The WebM has no seek index, so the only way to reach a frame is to play
        // through and grab it as it goes.
        const grab = async (t) => {{
          ctx.drawImage(v, 0, 0, 1280, 720);
          const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.75));
          if (blob) {{
            await fetch(receiver + '/frame?t=' + t, {{ method: 'POST', body: blob }});
            sent.push(t);
          }}
        }};

        if (v.requestVideoFrameCallback) {{
          await new Promise((resolve) => {{
            const onFrame = async (now, meta) => {{
              const t = meta.mediaTime;
              while (pending.length && t >= pending[0] - 0.05) {{
                await grab(pending.shift());
              }}
              if (!pending.length) {{ resolve(); return; }}
              v.requestVideoFrameCallback(onFrame);
            }};
            v.requestVideoFrameCallback(onFrame);
          }});
        }} else {{
          while (pending.length) {{
            const t = pending[0];
            if (v.currentTime >= t - 0.05) {{
              await grab(pending.shift());
            }} else {{
              await new Promise((r) => setTimeout(r, 20));
            }}
          }}
        }}
        v.pause();
        return 'sent ' + sent.length;
      }}, {{ times: times, videoUrl: '{static_url}/{video.name}', receiver: '{receiver_url}' }});
      return result;
    }}
    """

    script_path = outdir / "_extract.js"
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        ["playwright-cli", "run-code", "--filename", str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        # On Windows the CLI is a .cmd shim, which needs the shell to resolve.
        shell=True,
    )
    script_path.unlink(missing_ok=True)

    frames = sorted(outdir.glob("frame-*.jpg"))
    for path in frames:
        print(f"  {path.name}  {path.stat().st_size // 1024} KB")
    print(f"Wrote {len(frames)} frames to {outdir}")

    if not frames:
        print(result.stdout[-600:])
        print(result.stderr[-600:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
