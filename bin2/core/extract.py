"""
core.extract - command-line .nxs extraction/inspection tool

Usage:
  python -m core.extract /path/to/file.nxs
  # Optional custom output dir:
  python -m core.extract /path/to/file.nxs --outdir /path/to/extract

Consumes any canonical OperaXN file (schema v1/v2/v3) via core.nxs_reader and
writes:
  * xrd 1D            -> scan_NNNNNN/xrd_oned.csv (with errors when stored)
  * xrd 2D (embedded) -> scan_NNNNNN/xrd_twod.npy
  * neutron banks     -> scan_NNNNNN/bank_N_{tof,d}.csv
  * operando echem    -> operando_echem.csv
  * standard echem    -> standard_echem_*.csv
  * summary.json      -> global metadata + per-scan summary
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from .model import ExperimentModel, ScanData
from .nxs_reader import load

MAX_SCANS_DEFAULT = 5


def ensure_dir(p: str) -> None:
    """Create the directory (and parents) if it does not exist."""
    os.makedirs(p, exist_ok=True)


def write_xy_csv(path: str, x, y, e=None, x_name="x", y_name="y") -> None:
    """Write x/y (and optional error) columns as a CSV."""
    cols = {x_name: x, y_name: y}
    if e is not None:
        cols["error"] = e
    pd.DataFrame(cols).to_csv(path, index=False)


def extract_scan(scan: ScanData, outdir_scan: str) -> Dict[str, Any]:
    """Write one scan's 1D/2D/bank data files and return its summary dict."""
    s: Dict[str, Any] = {
        "scan_num": scan.scan_num,
        "timestamp": scan.timestamp,
        "voltage (V)": scan.echem,
        "current (mA)": scan.current,
        "voltage_timestamp": scan.echem_timestamp,
        "exposure_time": scan.exposure_time,
    }
    if scan.monitor:
        s["monitor"] = scan.monitor
    if scan.neutron_start:
        s["neutron_start"] = scan.neutron_start
        s["neutron_end"] = scan.neutron_end

    if scan.oned is not None:
        write_xy_csv(os.path.join(outdir_scan, "xrd_oned.csv"),
                     scan.oned["x"], scan.oned["y"], scan.oned.get("e"),
                     "two_theta", "intensity")
        s["xrd_oned_points"] = int(len(scan.oned["x"]))
        s["xrd_oned_has_errors"] = "e" in scan.oned
        s["xrd_oned_source"] = scan.oned.get("source")

    if scan.twod is not None:
        np.save(os.path.join(outdir_scan, "xrd_twod.npy"), scan.twod)
        s["xrd_twod_shape"] = list(scan.twod.shape)
        s["xrd_twod_stored"] = "embedded"
    elif scan.twod_source:
        s["xrd_twod_stored"] = "external"
        s["xrd_twod_source"] = scan.twod_source

    if scan.neutron:
        banks = []
        for bank_num, bank in sorted(scan.neutron.items()):
            info: Dict[str, Any] = {"bank": bank_num}
            for key, (x_name, y_name) in {
                "tof": ("tof", "intensity"),
                "d": ("d_spacing", "intensity"),
            }.items():
                if key in bank:
                    write_xy_csv(
                        os.path.join(outdir_scan, f"bank_{bank_num}_{key}.csv"),
                        bank[key]["x"], bank[key]["y"], bank[key].get("e"),
                        x_name, y_name)
                    info[f"{key}_points"] = int(len(bank[key]["x"]))
                    info[f"{key}_has_errors"] = "e" in bank[key]
                    info[f"{key}_source"] = bank[key].get("source")
            banks.append(info)
        s["neutron_banks"] = banks

    return s


def write_echem_csv(outdir: str, name: str, df: pd.DataFrame) -> int:
    """Write an echem DataFrame as CSV; returns the number of rows."""
    out = pd.DataFrame({"timestamp": df["timestamp"].astype(str),
                        "voltage_V": df["echem_data"]})
    if "current" in df.columns:
        out["current_mA"] = df["current"]
    out.to_csv(os.path.join(outdir, name), index=False)
    return len(out)


def print_summary(model: ExperimentModel, summary: Dict[str, Any]) -> None:
    """Print the experiment metadata and a one-line digest per scan."""
    gm = model.global_metadata or {}
    print(f"     data_source: {model.data_source}")
    for key in ("title", "start_time", "end_time", "experiment_identifier",
                "correlation_method", "generator", "generator_version"):
        if key in gm:
            print(f"     {key}: {gm[key]}")
    for group in ("instrument", "sample", "user"):
        if isinstance(gm.get(group), dict):
            name = gm[group].get("name")
            if name:
                print(f"     {group}: {name}")

    if summary.get("operando_echem_points"):
        print(f"     operando electrochemistry: {summary['operando_echem_points']} points")
    if summary.get("standard_echem_files"):
        print(f"     standard electrochemistry: {summary['standard_echem_files']} files")

    for s in summary["scans"]:
        bits = [f"scan_{s['scan_num']:06d}"]
        if "xrd_oned_points" in s:
            err = "+err" if s.get("xrd_oned_has_errors") else ""
            bits.append(f"1D={s['xrd_oned_points']}pts{err}")
        if s.get("xrd_twod_stored") == "embedded":
            bits.append(f"2D={s.get('xrd_twod_shape')}")
        elif s.get("xrd_twod_stored") == "external":
            bits.append("2D=external")
        if s.get("voltage (V)") is not None:
            bits.append(f"V={s['voltage (V)']}")
        if s.get("neutron_banks"):
            bits.append(f"neutron={len(s['neutron_banks'])}banks")
        print("     - " + " | ".join(bits))


def main() -> None:
    """CLI entry point: extract a canonical .nxs into CSVs plus summary.json."""
    ap = argparse.ArgumentParser(
        description="Extract scans and echem from a canonical OperaXN .nxs file.")
    ap.add_argument("nxs_file", help="Path to a .nxs file")
    ap.add_argument("--outdir", help="Output directory (default: <file_stem>_extract)")
    ap.add_argument("--max-scans", type=int, default=MAX_SCANS_DEFAULT,
                    help=f"Number of scans to extract (default {MAX_SCANS_DEFAULT}; 0 = all)")
    args = ap.parse_args()

    nxs_path = os.path.abspath(args.nxs_file)
    if not os.path.isfile(nxs_path):
        raise SystemExit(f"File not found: {nxs_path}")

    outdir = args.outdir or (os.path.splitext(os.path.basename(nxs_path))[0] + "_extract")
    outdir = os.path.abspath(outdir)
    ensure_dir(outdir)

    model = load(nxs_path)

    summary: Dict[str, Any] = {
        "file": os.path.basename(nxs_path),
        "data_source": model.data_source,
        "total_scans": len(model.scans),
        "global_metadata": model.global_metadata,
        "scans": [],
    }

    if model.echem_df is not None and not model.echem_df.empty:
        summary["operando_echem_points"] = write_echem_csv(
            outdir, "operando_echem.csv", model.echem_df)
    else:
        summary["operando_echem_points"] = 0

    summary["standard_echem_files"] = len(model.standard_echem)
    for item in model.standard_echem:
        write_echem_csv(outdir, f"standard_echem_{item['name']}.csv", item["data"])

    scans = model.scans if args.max_scans == 0 else model.scans[:args.max_scans]
    summary["scans_extracted"] = len(scans)

    for scan in scans:
        scan_out = os.path.join(outdir, f"scan_{scan.scan_num:06d}")
        ensure_dir(scan_out)
        summary["scans"].append(extract_scan(scan, scan_out))

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"[OK] {os.path.basename(nxs_path)} -> {outdir}")
    print(f"     scans extracted: {summary['scans_extracted']} / {summary['total_scans']}")
    print_summary(model, summary)


if __name__ == "__main__":
    main()
