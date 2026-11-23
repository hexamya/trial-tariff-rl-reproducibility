"""Download the LCL 2013 price/temperature/aggregate workbook into data/raw/lcl/.

Source: GitHub mirror groupoasys/Data_Low_Carbon_London (Data_features.xlsx).
Uses HTTPS_PROXY from the environment when set. Skips the download if the
file is already present with the expected checksum.

    python scripts/fetch_lcl_features.py
"""
import hashlib
import sys
import urllib.request

from ttrl.data.lcl_features import XLSX_PATH

URL = ("https://raw.githubusercontent.com/groupoasys/"
       "Data_Low_Carbon_London/master/Data_features.xlsx")
SHA256_PREFIX = "f49b6821195ba9ad"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if XLSX_PATH.exists() and sha256(XLSX_PATH).startswith(SHA256_PREFIX):
        print(f"already present: {XLSX_PATH}")
        return 0
    XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as resp:
        XLSX_PATH.write_bytes(resp.read())
    digest = sha256(XLSX_PATH)
    if not digest.startswith(SHA256_PREFIX):
        print(f"checksum mismatch: {digest[:16]} != {SHA256_PREFIX}", file=sys.stderr)
        return 1
    print(f"saved {XLSX_PATH} ({XLSX_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
