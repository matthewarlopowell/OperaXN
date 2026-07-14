"""
NeXus/HDF5 file reader for the core pipeline.

Loads a canonical .nxs into an ExperimentModel. Dispatches on layout:
- schema 3.0: single /entry NXentry with scan_NNNNNN NXsubentries
  (environment / monitor / data / bank_N groups; NXmonopd/NXtofnpd-based)
- schema 2.0 and v1 nexusgen files: root-level scan_NNNN NXentry groups
  (metadata / xrd_data / neutron_data groups)

Within each layout the reader dispatches on which groups are present, so
mixed or partial files render whatever they contain.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd

from .model import ExperimentModel, ScanData

logger = logging.getLogger(__name__)

_SCAN_GROUP_RE = re.compile(r"scan_(\d+)$")


def is_canonical_nxs(path: str) -> bool:
    """True if the file looks like one of our generated files, as opposed to a
    raw beamline .nxs."""
    try:
        with h5py.File(path, 'r') as f:
            # v3 layout: /entry with our attrs or scan_ subgroups
            entry = f.get('entry')
            if entry is not None and ('generator' in entry.attrs or any(
                    _SCAN_GROUP_RE.fullmatch(k) for k in entry.keys())):
                return True
            # v1/v2 layout: root-level scan groups or generator attribute
            if any(_SCAN_GROUP_RE.fullmatch(k) for k in f.keys()):
                return True
            gm = f.get('global_metadata')
            return gm is not None and 'generator' in gm.attrs
    except Exception:
        return False


def load(path: str) -> ExperimentModel:
    """Load a canonical .nxs file into an ExperimentModel."""
    model = ExperimentModel(source_path=os.path.abspath(path))

    with h5py.File(path, 'r') as f:
        entry = f.get('entry')
        is_v3 = (entry is not None and isinstance(entry, h5py.Group)
                 and ('generator' in entry.attrs
                      or any(_SCAN_GROUP_RE.fullmatch(k) for k in entry.keys())))

        if is_v3:
            _load_v3(entry, model)
        else:
            _load_legacy(f, model)

    logger.info(f"Loaded {len(model.scans)} scans from {os.path.basename(path)}")
    return model


def _scan_groups(parent: h5py.Group):
    """Yield (name, group) for scan_N children in numeric order."""
    names = sorted(
        (k for k in parent.keys() if _SCAN_GROUP_RE.fullmatch(k)),
        key=lambda n: int(_SCAN_GROUP_RE.fullmatch(n).group(1))
    )
    for name in names:
        yield name, parent[name]


# ============================================================================
# Schema 3.0
# ============================================================================

def _load_v3(entry: h5py.Group, model: ExperimentModel) -> None:
    """Populate the model from a /entry NXentry (schema 3.0 layout)."""
    _read_global_metadata_v3(entry, model)
    _read_operando_echem(entry, model)
    _read_standard_echem(entry, model)

    for name, grp in _scan_groups(entry):
        model.scans.append(_read_scan_v3(grp, name))


def _read_global_metadata_v3(entry: h5py.Group, model: ExperimentModel) -> None:
    """Entry attrs, top-level fields, and instrument/sample/user groups."""
    out: Dict[str, Any] = {}

    for attr_name, attr_val in entry.attrs.items():
        out[attr_name] = _decode(attr_val)

    for field in ('title', 'start_time', 'end_time', 'experiment_identifier'):
        val = _dataset_scalar(entry, field)
        if val is not None:
            out[field] = _decode(val)

    for group_name in ('instrument', 'sample', 'user'):
        grp = entry.get(group_name)
        if isinstance(grp, h5py.Group):
            out[group_name] = _flatten_group(grp)

    model.global_metadata = out
    model.data_source = out.get('data_source')


def _flatten_group(grp: h5py.Group, max_depth: int = 3) -> Dict[str, Any]:
    """Recursively flatten a metadata group into nested dicts of decoded values."""
    out: Dict[str, Any] = {}
    for attr_name, attr_val in grp.attrs.items():
        out[attr_name] = _decode(attr_val)
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            if item.size <= 100:
                out[key] = _decode(item[()])
        elif isinstance(item, h5py.Group) and max_depth > 0:
            out[key] = _flatten_group(item, max_depth - 1)
    return out


def _read_scan_v3(sub: h5py.Group, name: str) -> ScanData:
    """One scan_N NXsubentry -> ScanData (environment, monitor, data, banks)."""
    scan_num = int(_SCAN_GROUP_RE.fullmatch(name).group(1))
    scan = ScanData(scan_num=int(sub.attrs.get('scan_number', scan_num)))

    start = _decode(_dataset_scalar(sub, 'start_time'))
    end = _decode(_dataset_scalar(sub, 'end_time'))

    # Environment: correlated electrochemistry + timestamps
    env = sub.get('environment')
    if isinstance(env, h5py.Group):
        scan.timestamp = _decode(_dataset_scalar(env, 'scan_timestamp')) or start
        scan.midpoint_timestamp = _decode(
            _dataset_scalar(env, 'midpoint_adjusted_timestamp'))
        scan.echem_timestamp = _decode(_dataset_scalar(env, 'voltage_timestamp'))
        scan.echem = _float_or_none(_dataset_scalar(env, 'voltage (V)'))
        scan.current = _float_or_none(_dataset_scalar(env, 'current (mA)'))
        scan.exposure_time = _float_or_none(_dataset_scalar(env, 'exposure_time'))
    else:
        scan.timestamp = start

    # Neutron acquisition window (present when end_time was written)
    if end:
        scan.neutron_start = start
        scan.neutron_end = end

    # Monitor record
    mon = sub.get('monitor')
    if isinstance(mon, h5py.Group):
        monitor: Dict[str, Any] = {}
        for key in mon.keys():
            val = _dataset_scalar(mon, key)
            if val is not None:
                monitor[key] = _decode(val)
        if monitor:
            scan.monitor = monitor

    # XRD data group
    data_grp = sub.get('data')
    if isinstance(data_grp, h5py.Group):
        _read_xrd_v3(data_grp, scan)

    # Neutron banks
    banks: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in sub.keys():
        if key.startswith('bank_') and isinstance(sub[key], h5py.Group):
            bank = _read_bank_v3(sub[key])
            if bank:
                banks[key[len('bank_'):]] = bank
    if banks:
        scan.neutron = banks

    return scan


def _read_xrd_v3(grp: h5py.Group, scan: ScanData) -> None:
    """1D pattern (with errors) plus the 2D image or its source reference."""
    x = _dataset_1d(grp, 'polar_angle')
    y = _dataset_1d(grp, 'data')
    if x is not None and y is not None and len(x) == len(y):
        oned = {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "source": _decode(grp.attrs.get('oned_source_file')),
        }
        e = _dataset_1d(grp, 'errors')
        if e is not None and len(e) == len(x):
            oned["e"] = np.asarray(e, dtype=float)
        scan.oned = oned

    scan.twod_source = _decode(grp.attrs.get('twod_source_path')) or \
        _decode(grp.attrs.get('twod_source'))
    embedded = grp.attrs.get('twod_embedded')
    scan.twod_embedded = bool(embedded) if embedded is not None else False

    if scan.twod_embedded and 'twod_image' in grp:
        scan.twod = np.asarray(grp['twod_image'][()], dtype=float)


def _read_bank_v3(b: h5py.Group) -> Dict[str, Dict[str, Any]]:
    """TOF and d-spacing traces of one detector bank as {'tof'|'d': arrays}."""
    bank: Dict[str, Dict[str, Any]] = {}

    for key, (x_name, y_name, e_name) in {
        'tof': ('time_of_flight', 'data', 'errors'),
        'd': ('d_spacing', 'd_data', 'd_errors'),
    }.items():
        x = _dataset_1d(b, x_name)
        y = _dataset_1d(b, y_name)
        if x is not None and y is not None and len(x) == len(y):
            entry = {"x": np.asarray(x, dtype=float),
                     "y": np.asarray(y, dtype=float),
                     "source": _decode(b.attrs.get(f'{key}_source_file'))}
            e = _dataset_1d(b, e_name)
            if e is not None and len(e) == len(x):
                entry["e"] = np.asarray(e, dtype=float)
            bank[key] = entry

    return bank


# ============================================================================
# Legacy layouts (schema 2.0 and v1 nexusgen files)
# ============================================================================

def _load_legacy(f: h5py.File, model: ExperimentModel) -> None:
    """Populate the model from root-level scan groups (schema 2.0 / v1)."""
    _read_global_metadata_legacy(f, model)
    _read_operando_echem(f, model)
    _read_standard_echem(f, model)

    for name, grp in _scan_groups(f):
        model.scans.append(_read_scan_legacy(grp, name))


def _read_global_metadata_legacy(f: h5py.File, model: ExperimentModel) -> None:
    """Contents of the root global_metadata group, when present."""
    if 'global_metadata' not in f:
        return

    gm = f['global_metadata']
    out: Dict[str, Any] = {}

    for attr_name, attr_val in gm.attrs.items():
        out[attr_name] = _decode(attr_val)

    for key in gm.keys():
        item = gm[key]
        if isinstance(item, h5py.Group):
            out[key] = _flatten_group(item)
        elif isinstance(item, h5py.Dataset):
            out[key] = _decode(item[()])

    model.global_metadata = out
    model.data_source = out.get('data_source')


def _read_scan_legacy(scan_grp: h5py.Group, name: str) -> ScanData:
    """One legacy scan group -> ScanData (metadata/xrd_data/neutron_data)."""
    scan_num = int(_SCAN_GROUP_RE.fullmatch(name).group(1))
    scan = ScanData(scan_num=int(scan_grp.attrs.get('scan_number', scan_num)))

    if 'metadata' in scan_grp:
        md = scan_grp['metadata']
        scan.timestamp = _decode(_dataset_scalar(md, 'scan_timestamp'))
        scan.midpoint_timestamp = _decode(_dataset_scalar(md, 'midpoint_adjusted_timestamp'))
        scan.echem_timestamp = _decode(_dataset_scalar(md, 'voltage_timestamp'))
        scan.echem = _float_or_none(_dataset_scalar(md, 'voltage (V)'))
        scan.current = _float_or_none(_dataset_scalar(md, 'current (mA)'))
        scan.exposure_time = _float_or_none(_dataset_scalar(md, 'exposure_time'))

    if 'xrd_data' in scan_grp:
        xrd = scan_grp['xrd_data']

        th = _dataset_1d(xrd, 'oned_2theta')
        inten = _dataset_1d(xrd, 'oned_intensity')
        if th is not None and inten is not None and len(th) == len(inten):
            scan.oned = {
                "x": np.asarray(th, dtype=float),
                "y": np.asarray(inten, dtype=float),
                "source": _decode(xrd.attrs.get('oned_source_file')),
            }

        scan.twod_source = _decode(xrd.attrs.get('twod_source_path')) or \
            _decode(xrd.attrs.get('twod_source'))
        embedded = xrd.attrs.get('twod_embedded')
        scan.twod_embedded = bool(embedded) if embedded is not None else False

        if scan.twod_embedded and 'twod_image' in xrd:
            scan.twod = np.asarray(xrd['twod_image'][()], dtype=float)

    if 'neutron_data' in scan_grp:
        ng = scan_grp['neutron_data']
        scan.neutron_start = _decode(ng.attrs.get('start_time'))
        scan.neutron_end = _decode(ng.attrs.get('end_time'))

        banks: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for bname in ng.keys():
            if not bname.startswith('bank_'):
                continue
            b = ng[bname]
            bank: Dict[str, Dict[str, Any]] = {}

            tof = _dataset_1d(b, 'tof')
            tofi = _dataset_1d(b, 'tof_intensity')
            if tof is not None and tofi is not None and len(tof) == len(tofi):
                bank['tof'] = {"x": np.asarray(tof, dtype=float),
                               "y": np.asarray(tofi, dtype=float),
                               "source": _decode(b.attrs.get('tof_source_file'))}

            d = _dataset_1d(b, 'd_spacing')
            di = _dataset_1d(b, 'd_intensity')
            if d is not None and di is not None and len(d) == len(di):
                bank['d'] = {"x": np.asarray(d, dtype=float),
                             "y": np.asarray(di, dtype=float),
                             "source": _decode(b.attrs.get('d_source_file'))}

            if bank:
                banks[bname[len('bank_'):]] = bank

        if banks:
            scan.neutron = banks

    return scan


# ============================================================================
# Shared section readers (same group names in all schema versions)
# ============================================================================

def _read_operando_echem(parent: h5py.Group, model: ExperimentModel) -> None:
    """Cycling protocol -> model.echem_df (empty frame when absent/invalid)."""
    if 'operando_electrochemistry' not in parent:
        model.echem_df = pd.DataFrame(columns=["timestamp", "echem_data", "current"])
        return

    e = parent['operando_electrochemistry']
    ts = _dataset_1d(e, 'timestamps')
    v = _dataset_1d(e, 'voltage (V)')
    i = _dataset_1d(e, 'current (mA)')

    if ts is None or v is None or len(ts) != len(v):
        model.echem_df = pd.DataFrame(columns=["timestamp", "echem_data", "current"])
        return

    timestamps = pd.to_datetime([_decode(t) for t in ts])
    df = pd.DataFrame({"timestamp": timestamps, "echem_data": v.astype(float)})
    if i is not None and len(i) == len(ts):
        df["current"] = i.astype(float)
    else:
        df["current"] = np.nan

    model.echem_df = df


def _read_standard_echem(parent: h5py.Group, model: ExperimentModel) -> None:
    """Each stored file_N dataset -> model.standard_echem entry."""
    if 'standard_electrochemistry' not in parent:
        return

    se = parent['standard_electrochemistry']
    for name in sorted(se.keys()):
        if not name.startswith('file_'):
            continue
        grp = se[name]
        ts = _dataset_1d(grp, 'timestamps')
        v = _dataset_1d(grp, 'voltage (V)')
        i = _dataset_1d(grp, 'current (mA)')

        if ts is None or v is None or len(ts) != len(v):
            continue

        timestamps = pd.to_datetime([_decode(t) for t in ts])
        df = pd.DataFrame({"timestamp": timestamps, "echem_data": v.astype(float)})
        if i is not None and len(i) == len(ts):
            df["current"] = i.astype(float)

        model.standard_echem.append({
            "name": name,
            "source_file": _decode(grp.attrs.get('source_file')),
            "data": df,
        })


# ============================================================================
# HDF5 helpers
# ============================================================================

def _dataset_scalar(grp: h5py.Group, name: str) -> Optional[Any]:
    """Raw value of a scalar dataset, or None when absent/unreadable."""
    try:
        if name in grp:
            return grp[name][()]
    except Exception:
        pass
    return None


def _dataset_1d(grp: h5py.Group, name: str) -> Optional[np.ndarray]:
    """Dataset flattened to 1D, or None when absent/unreadable."""
    try:
        if name in grp:
            return np.array(grp[name][()]).reshape(-1)
    except Exception:
        pass
    return None


def _float_or_none(value: Any) -> Optional[float]:
    """Value as float, or None when missing or non-numeric."""
    if value is None:
        return None
    try:
        return float(np.array(value).squeeze())
    except (ValueError, TypeError):
        return None


def _decode(value: Any) -> Optional[Any]:
    """Decode HDF5 bytes/numpy scalars to native Python values."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    if isinstance(value, np.bytes_):
        return bytes(value).decode('utf-8', errors='ignore')
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode(value.item())
        return [_decode(x) for x in value.tolist()]
    return value
