#!/usr/bin/env python3
"""Parse the county-level table of the TDTD 2025 Economic Impact of Travel on Tennessee report.

Input: directory of per-page text files p<N>.txt (N = PDF page), made with `pdftotext -layout`; pass it as the first argument.
Extracts the 2024 and 2025 values for each of the 95 counties and checks component sums, statewide totals and the 2025 rankings table.
Output: county_blocks.json in the current directory.

Usage: python extraction_script.py <page_text_dir>
"""
import re, glob, sys
from collections import OrderedDict

EIDIR = sys.argv[1] if len(sys.argv) > 1 else "eipages"

COUNTIES = """ANDERSON BEDFORD BENTON BLEDSOE BLOUNT BRADLEY CAMPBELL CANNON CARROLL CARTER
CHEATHAM CHESTER CLAIBORNE CLAY COCKE COFFEE CROCKETT CUMBERLAND DAVIDSON DECATUR
DEKALB DICKSON DYER FAYETTE FENTRESS FRANKLIN GIBSON GILES GRAINGER GREENE
GRUNDY HAMBLEN HAMILTON HANCOCK HARDEMAN HARDIN HAWKINS HAYWOOD HENDERSON HENRY
HICKMAN HOUSTON HUMPHREYS JACKSON JEFFERSON JOHNSON KNOX LAKE LAUDERDALE LAWRENCE
LEWIS LINCOLN LOUDON MCMINN MCNAIRY MACON MADISON MARION MARSHALL MAURY
MEIGS MONROE MONTGOMERY MOORE MORGAN OBION OVERTON PERRY PICKETT POLK
PUTNAM RHEA ROANE ROBERTSON RUTHERFORD SCOTT SEQUATCHIE SEVIER SHELBY SMITH
STEWART SULLIVAN SUMNER TIPTON TROUSDALE UNICOI UNION VAN BUREN WARREN WASHINGTON
WAYNE WEAKLEY WHITE WILLIAMSON WILSON""".split()
COUNTY_SET = set()
i = 0
raw = ("ANDERSON|BEDFORD|BENTON|BLEDSOE|BLOUNT|BRADLEY|CAMPBELL|CANNON|CARROLL|CARTER|"
       "CHEATHAM|CHESTER|CLAIBORNE|CLAY|COCKE|COFFEE|CROCKETT|CUMBERLAND|DAVIDSON|DECATUR|"
       "DEKALB|DICKSON|DYER|FAYETTE|FENTRESS|FRANKLIN|GIBSON|GILES|GRAINGER|GREENE|"
       "GRUNDY|HAMBLEN|HAMILTON|HANCOCK|HARDEMAN|HARDIN|HAWKINS|HAYWOOD|HENDERSON|HENRY|"
       "HICKMAN|HOUSTON|HUMPHREYS|JACKSON|JEFFERSON|JOHNSON|KNOX|LAKE|LAUDERDALE|LAWRENCE|"
       "LEWIS|LINCOLN|LOUDON|MCMINN|MCNAIRY|MACON|MADISON|MARION|MARSHALL|MAURY|"
       "MEIGS|MONROE|MONTGOMERY|MOORE|MORGAN|OBION|OVERTON|PERRY|PICKETT|POLK|"
       "PUTNAM|RHEA|ROANE|ROBERTSON|RUTHERFORD|SCOTT|SEQUATCHIE|SEVIER|SHELBY|SMITH|"
       "STEWART|SULLIVAN|SUMNER|TIPTON|TROUSDALE|UNICOI|UNION|VAN BUREN|WARREN|WASHINGTON|"
       "WAYNE|WEAKLEY|WHITE|WILLIAMSON|WILSON").split("|")
COUNTY_SET = set(raw)
assert len(COUNTY_SET) == 95, len(COUNTY_SET)

LABELS = ["SPENDING", "LODGING", "FOOD & BEVERAGES", "RETAIL", "RECREATION",
          "TRANSPORTATION", "LABOR INCOME", "EMPLOYMENT", "STATE TAXES",
          "LOCAL TAXES", "TAX SAVINGS PER HOUSEHOLD"]
MONEY = re.compile(r"\$([\d,]+)")
NUM = re.compile(r"(?<![\d$.,])([\d,]{2,})(?![\d%])")

def money_pair(line):
    vals = MONEY.findall(line)
    return [int(v.replace(",", "")) for v in vals[:2]] if len(vals) >= 2 else None

blocks = OrderedDict()   # county -> {label: (v2024, v2025)}
page_of = {}             # county -> (pdf_page, report_page)
cur = None
for path in sorted(glob.glob(f"{EIDIR}/p*.txt"), key=lambda p: int(re.search(r"p(\d+)\.txt", p).group(1))):
    pdfpage = int(re.search(r"p(\d+)\.txt", path).group(1))
    if pdfpage < 16:      # alphabetical county table starts at PDF page 16 (report p.14)
        continue
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    # report page = PDF page - 2
    report_page = pdfpage - 2
    for ln in lines:
        s = ln.strip()
        if s in COUNTY_SET:
            cur = s
            if cur not in blocks:
                blocks[cur] = {}
                page_of[cur] = (pdfpage, report_page)
            continue
        if cur is None:
            continue
        for lab in LABELS:
            if s.startswith(lab):
                rest = s[len(lab):]
                if lab == "EMPLOYMENT":
                    vals = [int(t.replace(",", "")) for t in rest.split()
                            if re.fullmatch(r"[\d,]+", t)]
                    if len(vals) >= 2:
                        blocks[cur][lab] = (vals[0], vals[1])
                else:
                    pair = money_pair(rest)
                    if pair:
                        blocks[cur][lab] = (pair[0], pair[1])
                break

print(f"counties parsed: {len(blocks)}")
missing = COUNTY_SET - set(blocks)
print("missing counties:", missing or "none")
incomplete = {c: [l for l in LABELS if l not in b] for c, b in blocks.items()}
incomplete = {c: m for c, m in incomplete.items() if m}
print("incomplete blocks:", incomplete or "none")

# component sums against the total, both years
bad = []
for c, b in blocks.items():
    for yi, yr in ((0, 2024), (1, 2025)):
        comp = sum(b[k][yi] for k in ["LODGING", "FOOD & BEVERAGES", "RETAIL", "RECREATION", "TRANSPORTATION"])
        if abs(comp - b["SPENDING"][yi]) > 5:
            bad.append((c, yr, comp, b["SPENDING"][yi]))
print("component mismatches (> $5):", bad or "none")

# county sums against printed statewide totals
for yi, yr, printed in ((0, 2024, 31662137672), (1, 2025, 32527758351)):
    tot = sum(b["SPENDING"][yi] for b in blocks.values())
    print(f"{yr}: sum of 95 counties = {tot:,} | printed statewide = {printed:,} | diff = {tot-printed:+,}")

# check 2025 spending against the rankings table (PDF pages 14-15)
rank = {}
for p in (14, 15):
    for ln in open(f"{EIDIR}/p{p}.txt", encoding="utf-8", errors="replace"):
        m = re.finditer(r"\b(\d{1,2})\s+([A-Z][A-Z ]+?)\s+\$([\d,]+)\s+(-?\d+)\b", ln)
        for g in m:
            name = g.group(2).strip()
            if name in COUNTY_SET:
                rank[name] = int(g.group(3).replace(",", ""))
print(f"rankings parsed: {len(rank)}")
mism = [(c, rank.get(c), blocks[c]["SPENDING"][1]) for c in blocks if rank.get(c) != blocks[c]["SPENDING"][1]]
print("rankings vs alphabetical mismatches:", mism or "none")

# save parsed blocks as JSON
import json
out = {c: {k: list(v) for k, v in b.items()} for c, b in blocks.items()}
json.dump({"blocks": out, "page_of": {c: list(v) for c, v in page_of.items()}}, open("county_blocks.json", "w"), indent=1)
print("page range of blocks:", min(page_of.values()), "->", max(page_of.values()))
