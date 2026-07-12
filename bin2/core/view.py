"""
core.view - print the raw stored structure of a canonical .nxs file

Shows the file exactly as stored: every group with its NX_class, every
dataset with shape/dtype and a value preview, all attributes, and soft
links displayed as links (not followed). Companion to core.extract, which
interprets the data; this shows the architecture itself.

Usage:
  python -m core.view /path/to/file.nxs
  python -m core.view /path/to/file.nxs --scans 0     # all scans
  python -m core.view /path/to/file.nxs --scans 5     # first 5 (default 2)
"""

import argparse
import os
import re

import h5py
import numpy as np

_SCAN_RE = re.compile(r"scan_\d+$")
INDENT = "    "


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="ignore")
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


def _fmt_attrs(obj) -> str:
    parts = []
    for k, v in obj.attrs.items():
        v = _decode(v)
        if isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"@{k}={v}")
    return ("  " + ", ".join(parts)) if parts else ""


def _fmt_value(ds: h5py.Dataset) -> str:
    """Preview of a dataset's stored value."""
    try:
        if ds.shape == ():
            return f"= {_decode(ds[()])!r}"
        if ds.ndim == 1 and ds.size <= 4:
            return "= [" + ", ".join(str(_decode(x)) for x in ds[()]) + "]"
        if ds.ndim == 1:
            first, last = _decode(ds[0]), _decode(ds[-1])
            if isinstance(first, float):
                first, last = f"{first:g}", f"{last:g}"
            return f"= [{first} ... {last}]"
        return ""
    except Exception:
        return ""


def _print_dataset(name: str, ds: h5py.Dataset, depth: int) -> None:
    shape = "scalar" if ds.shape == () else "x".join(map(str, ds.shape))
    print(f"{INDENT * depth}{name}  [{shape} {ds.dtype}] "
          f"{_fmt_value(ds)}{_fmt_attrs(ds)}")


def _print_group(group: h5py.Group, depth: int, max_scans: int) -> None:
    scan_names = sorted(k for k in group.keys() if _SCAN_RE.fullmatch(k))
    shown_scans = 0

    for key in group.keys():
        link = group.get(key, getlink=True)
        if isinstance(link, h5py.SoftLink):
            print(f"{INDENT * depth}{key}/  --> soft link to {link.path}")
            continue
        if isinstance(link, h5py.ExternalLink):
            print(f"{INDENT * depth}{key}/  --> external link to "
                  f"{link.filename}:{link.path}")
            continue

        obj = group[key]
        if isinstance(obj, h5py.Group):
            if key in scan_names:
                shown_scans += 1
                if max_scans and shown_scans > max_scans:
                    continue
            print(f"{INDENT * depth}{key}/{_fmt_attrs(obj)}")
            _print_group(obj, depth + 1, max_scans)
        else:
            _print_dataset(key, obj, depth)

    if max_scans and len(scan_names) > max_scans:
        print(f"{INDENT * depth}... (+{len(scan_names) - max_scans} more scans, "
              f"use --scans 0 to show all)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print the raw stored structure of a canonical .nxs file.")
    ap.add_argument("nxs_file", help="Path to a .nxs file")
    ap.add_argument("--scans", type=int, default=2,
                    help="Number of scan subentries to print (default 2; 0 = all)")
    args = ap.parse_args()

    path = os.path.abspath(args.nxs_file)
    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        size_mb = os.path.getsize(path) / 1e6
        print(f"{os.path.basename(path)}  ({size_mb:.1f} MB)")
        print(f"/{_fmt_attrs(f)}")
        _print_group(f, 1, args.scans)


if __name__ == "__main__":
    main()
