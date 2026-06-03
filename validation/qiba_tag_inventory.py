"""QIBA-CT-Liver-Phantom — DICOM tag inventory & factorial reconstruction.

Walks a directory of QIBA DICOM files (TCIA download), builds a one-row-per-series
inventory of the acquisition parameters, and locates where iterative-reconstruction
strength (GE ASIR / Siemens SAFIRE-IR) is actually recorded.

Why this matters: the GammaMetric "metadata is insufficient for comparability"
thesis hinges on whether recon strength is recoverable from standard DICOM tags.
QIBA varies ASIR systematically on a fixed object, so it is the clean test case:
  - If ASIR% only appears in SeriesDescription / private tags (not a standard,
    queryable field), then "the tag can't tell these scans apart" is a demonstrated
    fact, not an argument — and the pixel fingerprint is the only recovery path.

Outputs (into --out, default ./qiba_inventory_out):
  series_inventory.csv      one row per series, all parsed params
  factorial_summary.txt     unique values per axis + per-vendor cross-tabs
  asir_findings.txt         every element (incl. private) mentioning ASIR/iterative
  sample_tags_<MFR>.txt      full element dump of one series per manufacturer

Usage:
  python validation/qiba_tag_inventory.py --root "C:/path/to/QIBA download"
  (defaults to scanning the current directory if --root is omitted)
"""
from __future__ import annotations

import os
import re
import csv
import argparse
from collections import defaultdict, Counter

import pydicom
from pydicom.misc import is_dicom

# --------------------------------------------------------------------------- #
# Standard tags we want, by keyword. Missing tags are tolerated.
# --------------------------------------------------------------------------- #
STD_KEYWORDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "XRayTubeCurrent",       # mA            (0018,1151)
    "Exposure",              # mAs           (0018,1152)
    "ExposureTime",          # ms            (0018,1150)
    "SpiralPitchFactor",     # pitch         (0018,9311) — often absent on classic CT
    "SliceThickness",        # mm            (0018,0050)
    "SpacingBetweenSlices",  # mm            (0018,0088)
    "ConvolutionKernel",     #               (0018,1210)
    "ReconstructionDiameter",# mm            (0018,1100)
    "FilterType",            #               (0018,1160)
]

# Regexes over SeriesDescription (e.g. "690 mA 5 MM 70%ASIR", "Recon 7: AXIAL SOFT TISSUE")
RE_MA    = re.compile(r"(\d+(?:\.\d+)?)\s*ma\b", re.IGNORECASE)
RE_MM    = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
RE_ASIR  = re.compile(r"(\d+)\s*%?\s*asir", re.IGNORECASE)
RE_RECON = re.compile(r"recon\s*(\d+)", re.IGNORECASE)

# Strings that flag an iterative-recon-related element anywhere in the dataset
IR_HINTS = ("asir", "asir-v", "asirv", "iterative", "safire", "admire",
            "veo", "idose", "aidr", "imr", "clear", "recon algorithm")


def find_dicom_files(root: str):
    """Yield paths of DICOM files under root (TCIA files often have no extension)."""
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                if is_dicom(p):
                    yield p
            except (OSError, PermissionError):
                continue


def parse_description(desc: str) -> dict:
    desc = desc or ""
    ma = RE_MA.search(desc)
    mm = RE_MM.search(desc)
    asir = RE_ASIR.search(desc)
    recon = RE_RECON.search(desc)
    return {
        "desc_mA": ma.group(1) if ma else "",
        "desc_slice_mm": mm.group(1) if mm else "",
        "desc_asir_pct": asir.group(1) if asir else "",
        "desc_recon_n": recon.group(1) if recon else "",
    }


def scan_for_ir(ds) -> list:
    """Return [(tag, name, VR, value)] for any element whose name or value mentions
    an iterative-recon keyword. Walks sequences recursively."""
    hits = []

    def walk(dataset):
        for elem in dataset:
            try:
                name = (elem.keyword or elem.name or "").lower()
                val = ""
                if elem.VR != "SQ":
                    val = str(elem.value)
                blob = f"{name} {val}".lower()
                if any(h in blob for h in IR_HINTS):
                    hits.append((str(elem.tag), elem.keyword or elem.name,
                                 elem.VR, (val[:120] if val else "")))
            except Exception:
                pass
            if elem.VR == "SQ":
                for item in (elem.value or []):
                    walk(item)

    walk(ds)
    return hits


def get(ds, keyword, default=""):
    v = ds.get(keyword, default)
    return "" if v is None else v


def main():
    ap = argparse.ArgumentParser(description="QIBA DICOM tag inventory")
    ap.add_argument("--root", default=".", help="Root folder of the QIBA download")
    ap.add_argument("--out", default="qiba_inventory_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- pass 1: group files by series, pick a representative, count instances ---
    print(f"Scanning {os.path.abspath(args.root)} ...")
    repr_path = {}              # series_uid -> representative file path
    inst_count = Counter()      # series_uid -> n instances
    n_seen = 0
    for p in find_dicom_files(args.root):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
            uid = str(ds.get("SeriesInstanceUID", ""))
        except Exception:
            continue
        if not uid:
            continue
        inst_count[uid] += 1
        if uid not in repr_path:
            repr_path[uid] = p
        n_seen += 1
        if n_seen % 2000 == 0:
            print(f"  ...{n_seen} files, {len(repr_path)} series so far")

    print(f"Found {len(repr_path)} series across {n_seen} DICOM files.")
    if not repr_path:
        print("No DICOM series found. Check --root.")
        return

    # ---- pass 2: read representatives, extract tags --------------------------- #
    rows = []
    ir_findings = []            # (series_desc, mfr, [hits])
    sample_dumped = set()       # manufacturers already dumped in full

    for uid, p in sorted(repr_path.items()):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception as e:
            print(f"  ! could not read {p}: {e}")
            continue

        row = {k: get(ds, k) for k in STD_KEYWORDS}
        # normalize ConvolutionKernel (can be a multi-value)
        ck = row["ConvolutionKernel"]
        if isinstance(ck, (list, pydicom.multival.MultiValue)):
            row["ConvolutionKernel"] = "\\".join(str(x) for x in ck)
        row["n_instances"] = inst_count[uid]
        row.update(parse_description(str(row["SeriesDescription"])))
        rows.append(row)

        # IR scan
        hits = scan_for_ir(ds)
        if hits:
            ir_findings.append((str(row["SeriesDescription"]), str(row["Manufacturer"]), hits))

        # full dump, one per manufacturer
        mfr = (str(row["Manufacturer"]) or "UNKNOWN").split()[0].upper()
        if mfr not in sample_dumped:
            dump_path = os.path.join(args.out, f"sample_tags_{mfr}.txt")
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(f"Representative series for {row['Manufacturer']}\n")
                f.write(f"SeriesDescription: {row['SeriesDescription']}\n")
                f.write(f"File: {p}\n")
                f.write("=" * 70 + "\n")
                f.write(str(ds))
            sample_dumped.add(mfr)
            print(f"  wrote full tag dump for {mfr} -> {dump_path}")

    # ---- write inventory CSV ------------------------------------------------- #
    cols = STD_KEYWORDS + ["n_instances", "desc_mA", "desc_slice_mm",
                           "desc_asir_pct", "desc_recon_n"]
    inv_path = os.path.join(args.out, "series_inventory.csv")
    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} series -> {inv_path}")

    # ---- ASIR / IR findings -------------------------------------------------- #
    asir_path = os.path.join(args.out, "asir_findings.txt")
    with open(asir_path, "w", encoding="utf-8") as f:
        if not ir_findings:
            f.write("No elements mentioning ASIR/iterative-recon keywords were found "
                    "in any series' tags.\n\nIMPLICATION: recon strength is NOT in a "
                    "readable tag — only (if anywhere) in SeriesDescription. This is "
                    "the strongest form of the metadata-insufficiency claim.\n")
        else:
            f.write("Elements mentioning iterative-recon keywords (tag : name : VR : value):\n\n")
            for desc, mfr, hits in ir_findings:
                f.write(f"[{mfr}] {desc}\n")
                for tag, name, vr, val in hits:
                    f.write(f"    {tag}  {name}  ({vr}) = {val}\n")
                f.write("\n")
    print(f"Wrote ASIR/IR findings -> {asir_path}")

    # ---- factorial summary --------------------------------------------------- #
    sum_path = os.path.join(args.out, "factorial_summary.txt")
    by_mfr = defaultdict(list)
    for r in rows:
        by_mfr[str(r["Manufacturer"]) or "UNKNOWN"].append(r)

    def uniq(rs, key):
        return sorted({str(r[key]) for r in rs if str(r[key]) != ""})

    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(f"QIBA series inventory — {len(rows)} series\n")
        f.write("=" * 70 + "\n\n")
        for mfr, rs in by_mfr.items():
            f.write(f"### {mfr}  ({len(rs)} series)\n")
            f.write(f"  Models:           {uniq(rs, 'ManufacturerModelName')}\n")
            f.write(f"  KVP:              {uniq(rs, 'KVP')}\n")
            f.write(f"  Tube current mA:  {uniq(rs, 'XRayTubeCurrent')}\n")
            f.write(f"  Exposure mAs:     {uniq(rs, 'Exposure')}\n")
            f.write(f"  Pitch:            {uniq(rs, 'SpiralPitchFactor')}\n")
            f.write(f"  SliceThickness:   {uniq(rs, 'SliceThickness')}\n")
            f.write(f"  ConvolutionKernel:{uniq(rs, 'ConvolutionKernel')}\n")
            f.write(f"  FilterType:       {uniq(rs, 'FilterType')}\n")
            f.write(f"  desc mA:          {uniq(rs, 'desc_mA')}\n")
            f.write(f"  desc slice mm:    {uniq(rs, 'desc_slice_mm')}\n")
            f.write(f"  desc ASIR %:      {uniq(rs, 'desc_asir_pct')}\n")
            # kernel x asir cross-tab (the key one)
            ct = Counter((str(r["ConvolutionKernel"]), str(r["desc_asir_pct"])) for r in rs)
            f.write("  kernel x ASIR%(from desc) counts:\n")
            for (k, a), c in sorted(ct.items()):
                f.write(f"      kernel={k!r:18} asir={a!r:5} -> {c} series\n")
            f.write("\n")
    print(f"Wrote factorial summary -> {sum_path}")

    print("\nDone. Read factorial_summary.txt first, then asir_findings.txt "
          "to see whether ASIR is tag-hidden.")


if __name__ == "__main__":
    main()
