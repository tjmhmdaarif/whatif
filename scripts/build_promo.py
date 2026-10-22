"""Build the standalone promotional page.

The page in ``promo/src.html`` is a template. This script inlines every asset it
references, so the result is a single file that works from ``file://`` and can be
dropped on any static host without a build step on the reader's side.

    python scripts/build_promo.py

Writes ``promo/whatif-promo.html``.
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path


def compress_image(path: Path, max_width: int = 1500, quality: int = 72) -> str:
    """Downscale an image and return it as a base64 data URI.

    The full PNG is 727 KB. As a JPEG at a sensible width it is around 90 KB,
    which keeps the standalone page small enough to serve comfortably.
    """
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build(root: Path) -> Path:
    """Inline the assets and write the standalone page."""
    source = root / "promo" / "src.html"
    target = root / "promo" / "whatif-promo.html"
    shot = root / "docs" / "screenshots" / "retrieval.png"

    html = source.read_text(encoding="utf-8")

    if "{{RETRIEVAL_IMAGE}}" not in html:
        print("The template has no {{RETRIEVAL_IMAGE}} placeholder.", file=sys.stderr)
        return target

    print("Compressing the product screenshot...")
    data_uri = compress_image(shot)
    size_kb = len(data_uri) // 1024
    print(f"  {shot.name} -> data URI, {size_kb} KB")

    html = html.replace("{{RETRIEVAL_IMAGE}}", data_uri)

    # Nothing may reference an external file any more.
    leftovers = re.findall(r'(?:src|href)="(?!data:|#|https://127\.0\.0\.1)([^"]+)"', html)
    leftovers = [
        item
        for item in leftovers
        if not item.startswith("http://") and not item.startswith("https://")
    ]
    if leftovers:
        print(f"WARNING: still referencing external files: {leftovers}", file=sys.stderr)

    target.write_text(html, encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size // 1024} KB)")
    return target


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    build(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
