#!/usr/bin/env python3
"""Download the TDTD source report PDF (30 MB, not included in the repo) and check its SHA-256.

The extracted county table (tdtd_county_visitor_spending.csv) is included, so code/run_all.py does not need the PDF.
The PDF is only needed to audit the extraction (see extraction_script.py and SOURCE_NOTES.md).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FNAME = "2025 Economic Impact of Travel on Tennessee (1).pdf"
URL = ("https://industry.tnvacation.com/sites/default/files/2026-09/"
       "2025%20Economic%20Impact%20of%20Travel%20on%20Tennessee%20%281%29.pdf")
EXPECTED = "66ba797692e239868e752314a1d18b86aa1b532ab2c470f9bc49631862f4f058"


def main():
    out = HERE / FNAME
    if not out.exists():
        print(f"Downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
            f.write(r.read())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"sha256: {digest}")
    if digest != EXPECTED:
        print("MISMATCH against MANIFEST.sha256 — the file on the server has changed; "
              "treat the bundled CSV as the frozen extraction of record.", file=sys.stderr)
        sys.exit(1)
    print("Verified against MANIFEST.sha256.")


if __name__ == "__main__":
    main()
