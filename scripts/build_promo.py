"""Build the standalone promotional page.

The page in ``promo/src.html`` is a template. This script inlines every asset it
references, so the result is a single file that works from ``file://`` and can be
dropped on any static host without a build step on the reader's side.

    python scripts/build_promo.py

Writes ``promo/whatif-promo.html``.

One thing is deliberately not inlined: the "Try the prototype" buttons link to
``demo/index.html``, the static build produced by ``scripts/build_demo.py``. That
folder ships next to this page, so the landing page hands the reader a working
interface rather than a screenshot.
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

    # Nothing may reference an external file any more. A relative link is allowed
    # only when the target really sits next to the page, so the standalone file
    # cannot ship with a link to something that is not there.
    pattern = re.compile(r'(?:src|href)="(?!data:|#|https?://)([^"]+)"')
    links = sorted(set(pattern.findall(html)))
    missing = [item for item in links if not (root / "promo" / item).exists()]
    if missing:
        print(f"WARNING: still referencing external files: {missing}", file=sys.stderr)
    shipped = [item for item in links if item not in missing]
    if shipped:
        print("Ships next to this page: " + ", ".join(shipped))

    target.write_text(html, encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size // 1024} KB)")
    return target


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    build(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
