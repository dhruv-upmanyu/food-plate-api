"""Check a running meal-photo API end to end. Standard library only, so it runs from any machine with Python 3.8+.

    python api_smoke_test.py                                    # http://127.0.0.1:8000, photos in samples/
    python api_smoke_test.py --url https://food.example.com --key <API key> photo1.jpg photo2.jpg

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

FIELDS = ["kcal", "protein_g", "carb_g", "fat_g", "fibre_g"]


def call(url: str, key: str | None, body: bytes | None = None, content_type: str | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    if key:
        req.add_header("X-API-Key", key)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def multipart(name: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n").encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="*", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")   # not "localhost": Windows tries IPv6 first (+2 s)
    ap.add_argument("--key", help="API key, if the server sets PLATE_API_KEY")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    photos = a.photos or sorted(p for p in (Path(__file__).parent / "samples").glob("*") if p.suffix.lower() in
                                {".jpg", ".jpeg", ".png", ".webp", ".heic"})
    failures = []

    def check(ok: bool, what: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            failures.append(what)

    status, h = call(f"{url}/health", None)
    check(status == 200 and h.get("status") == "ok", f"GET /health -> {status} {h}")
    status, dishes = call(f"{url}/v1/dishes", a.key)
    check(status == 200 and isinstance(dishes, list) and len(dishes) > 100,
          f"GET /v1/dishes -> {status}, {len(dishes) if isinstance(dishes, list) else dishes} dishes")
    body, ctype = multipart("not_a_photo.jpg", b"this is not an image")
    status, err = call(f"{url}/v1/analyze", a.key, body, ctype)
    check(status == 400, f"POST /v1/analyze with a non-image -> {status} (expected 400)")

    for p in photos:
        body, ctype = multipart(p.name, p.read_bytes())
        t = time.time()
        status, meal = call(f"{url}/v1/analyze", a.key, body, ctype)
        wall = time.time() - t
        if status != 200:
            check(False, f"POST /v1/analyze {p.name} -> {status} {meal}")
            continue
        print(f"\n{p.name}: {len(meal['items'])} dishes, server {meal['processing_ms']} ms, round trip {wall:.1f} s")
        print(f"  {'dish':28} {'count':>5}  {'grams':>5} {'kcal':>6} {'prot':>5} {'carb':>5} {'fat':>5} {'fibre':>5}  conf")
        for it in meal["items"]:
            print(f"  {it['name'][:28]:28} {it['count']:>5g}  {it['grams']:>5g} {it['kcal']:>6.0f} {it['protein_g']:>5.1f} "
                  f"{it['carb_g']:>5.1f} {it['fat_g']:>5.1f} {it['fibre_g']:>5.1f}  {it['confidence']:.2f}"
                  + ("  (unsure)" if it["unsure"] else ""))
        t_ = meal["totals"]
        print(f"  {'MEAL TOTAL':28} {'':>5}  {t_['grams']:>5g} {t_['kcal']:>6.0f} {t_['protein_g']:>5.1f} "
              f"{t_['carb_g']:>5.1f} {t_['fat_g']:>5.1f} {t_['fibre_g']:>5.1f}")
        check(all(abs(t_[f] - sum(i[f] for i in meal["items"])) <= 0.5 for f in FIELDS), "totals = sum of items")
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
