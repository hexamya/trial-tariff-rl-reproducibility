"""Download the LCL smart-meter release (all households, half-hourly) into data/raw/lcl/meters/.

Source: London Datastore, "SmartMeter Energy Consumption Data in London Households"
(UK Power Networks, Low Carbon London). The documented link
``Power-Networks-LCL-June2015(withAcornGps).csv_Pieces.zip`` now redirects to
``Partitioned LCL Data.zip`` (795,722,689 bytes, last modified 2025-10-21).

The server gives each connection only ~100 kB/s but honours range requests, so the
file is fetched in 5 MiB parts over parallel connections. The parts match the S3
multipart upload behind the server's ETag (``<md5 of part md5s>-<n parts>``), which is
checked before the parts are joined. An interrupted run resumes from the parts on disk.

Requests go through the ``curl`` executable (the server's CDN answers Python's urllib
with 403). curl uses HTTPS_PROXY from the environment when set.

    python scripts/fetch_lcl_meters.py [--workers 16]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

from ttrl.paths import RAW

URL = ("https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/"
       "04feba67-f1a3-4563-98d0-f3071e3d56d1/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces.zip")
OUT_DIR = RAW / "lcl" / "meters"
ZIP_PATH = OUT_DIR / "partitioned_lcl_data.zip"
PARTS_DIR = OUT_DIR / ".parts"
PART = 5 * 1024 * 1024  # S3 multipart part size behind the ETag
EXPECTED_SIZE = 795_722_689
EXPECTED_ETAG = "f3e020799f8feae94a2d9036dfae3331-152"


def _get_headers(url: str, extra: tuple[str, ...] = ()) -> tuple[int, dict[str, str]]:
    """Status and headers (lower-case keys) of a GET whose body is discarded.

    A proxy's 'Connection established' block comes first in curl's output; the last
    block is the server's response.
    """
    out = subprocess.run(["curl", "-sS", "--max-time", "120", "-D", "-", "-o", os.devnull, *extra, url],
                         capture_output=True, text=True, check=True).stdout
    block = [b for b in out.replace("\r", "").split("\n\n") if b.strip()][-1].split("\n")
    headers = dict(line.split(":", 1) for line in block[1:] if ":" in line)
    return int(block[0].split()[1]), {k.strip().lower(): v.strip() for k, v in headers.items()}


def probe(url: str) -> tuple[str, int, str]:
    """Resolve the redirect and return (final URL, total bytes, ETag)."""
    status, h = _get_headers(url)
    if status in (301, 302, 303, 307, 308):
        url = urljoin(url, h["location"])
    status, h = _get_headers(url, ("-r", "0-0"))
    if status != 206:
        raise OSError(f"range request answered with HTTP {status}")
    return url, int(h["content-range"].rsplit("/", 1)[1]), h["etag"].strip('"')


def n_parts(total: int) -> int:
    return -(-total // PART)


def fetch_part(url: str, i: int, total: int, retries: int = 8) -> int:
    """Download byte range ``i`` (5 MiB) into PARTS_DIR, skipping it if already complete."""
    a = i * PART
    b = min(total, a + PART) - 1
    path = PARTS_DIR / f"part_{i:04d}.bin"
    if path.exists() and path.stat().st_size == b - a + 1:
        return i
    tmp = path.with_suffix(".tmp")
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sS", "-f", "--max-time", "600", "-r", f"{a}-{b}", "-o", str(tmp),
                            "-w", "%{http_code}", url], capture_output=True, text=True)
        size = tmp.stat().st_size if tmp.exists() else 0
        if r.returncode == 0 and r.stdout == "206" and size == b - a + 1:
            tmp.replace(path)
            return i
        err = f"curl exit {r.returncode}, HTTP {r.stdout}, {size} of {b - a + 1} bytes {r.stderr.strip()}"
        time.sleep(5 * (attempt + 1))
    raise OSError(f"part {i}: {err}")


def multipart_etag(paths) -> str:
    digests = b"".join(hashlib.md5(p.read_bytes()).digest() for p in paths)
    return f"{hashlib.md5(digests).hexdigest()}-{len(paths)}"


def file_etag(path) -> str:
    """S3 multipart ETag of a local file, computed with the same 5 MiB parts."""
    h = []
    with open(path, "rb") as f:
        while chunk := f.read(PART):
            h.append(hashlib.md5(chunk).digest())
    return f"{hashlib.md5(b''.join(h)).hexdigest()}-{len(h)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == EXPECTED_SIZE and file_etag(ZIP_PATH) == EXPECTED_ETAG:
        print(f"already present: {ZIP_PATH}")
        return 0

    url, total, etag = probe(URL)
    if (total, etag) != (EXPECTED_SIZE, EXPECTED_ETAG):
        print(f"server file changed: {total:,} bytes, ETag {etag} (expected {EXPECTED_SIZE:,}, "
              f"{EXPECTED_ETAG}); check the release before using it", file=sys.stderr)
        return 1
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    n = n_parts(total)
    print(f"{url}\n{total:,} bytes in {n} parts, {args.workers} connections", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_part, url, i, total) for i in range(n)]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 8 == 0 or done == n:
                mb = sum(p.stat().st_size for p in PARTS_DIR.glob("part_*.bin")) / 2**20
                print(f"{done}/{n} parts, {mb:,.0f} MB, {time.time() - t0:,.0f} s", flush=True)

    parts = [PARTS_DIR / f"part_{i:04d}.bin" for i in range(n)]
    got = multipart_etag(parts)
    if got != etag:
        print(f"ETag mismatch: {got} != {etag}", file=sys.stderr)
        return 1
    tmp = ZIP_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    tmp.replace(ZIP_PATH)
    shutil.rmtree(PARTS_DIR)
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = z.namelist()
    print(f"saved {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes, ETag ok, {len(members)} members)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
