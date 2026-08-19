"""
NeXus/HDF5 file reader for the core pipeline.

Loads a canonical .nxs into an ExperimentModel. Dispatches on layout:
- schema 4.0/3.0: single /entry with scan_NNNNNN NXsubentries; one loader
  reads both, trying v4 names first (voltage, time/@start, bank_N +
  bank_N_d, image_source) with v3 fallbacks ("voltage (V)", string
  timestamps, combined bank groups, twod_* attributes).
- schema 2.0 / legacy v1 files: root-level scan_NNNN groups
  (metadata / xrd_data / neutron_data).

Partial files render whatever they contain. Timestamps handed to the GUI
are normalised to 'YYYY-MM-DD HH:MM:SS' whatever the file stores.
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
_BANK_GROUP_RE = re.compile(r"bank_(\d+)(_d)?$")
_RELATIVE_TS_RE = re.compile(r"\d{1,3}:\d{2}:\d{2}$")


def _display_ts(value: Any) -> Any:
    """Normalise a stored timestamp to the GUI's 'YYYY-MM-DD HH:MM:SS' form.
    v4 stores ISO 8601; relative HH:MM:SS strings pass through unchanged."""
    if not value or not isinstance(value, str):
        return value
    if _RELATIVE_TS_RE.fullmatch(value):
        return value
    try:
        return pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return value


def _entry_is_canonical(entry) -> bool:
    """True when /entry carries our provenance or scan_ subgroups."""
    return (entry is not None and isinstance(entry, h5py.Group)
            and ('generator' in entry.attrs
                 or 'program_name' in entry
                 or 'definition' in entry
                 or any(_SCAN_GROUP_RE.fullmatch(k) for k in entry.keys())))


def is_canonical_nxs(path: str) -> bool:
    """True if the file looks like one of our generated files, as opposed to a
    raw beamline .nxs."""
    try:
        with h5py.File(path, 'r') as f:
            # v4/v3 layout: /entry with our provenance or scan_ subgroups
            if _entry_is_canonical(f.get('entry')):
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
        is_v3 = _entry_is_canonical(entry)

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
# Schema 4.0 / 3.0 (one loader, v4 names first with v3 fallbacks)
# ============================================================================

def _load_v3(entry: h5py.Group, model: ExperimentModel) -> None:
    """Populate the model from a /entry NXentry (schema 4.0/3.0 layouts,
    v4 names first with v3 fallbacks)."""
    _read_global_metadata_v3(entry, model)
    _read_operando_echem(entry, model)
    _read_standard_echem(entry, model)

    for name, grp in _scan_groups(entry):
        model.scans.append(_read_scan_v3(grp, name))


def _read_global_metadata_v3(entry: h5py.Group, model: ExperimentModel) -> None:
    """Entry fields/attrs + instrument/sample/user groups -> flat metadata.
    v4 provenance (program_name, process) lands on the same keys the v3
    entry attrs used, so GUI consumers see one shape for both schemas."""
    out: Dict[str, Any] = {}

    for attr_name, attr_val in entry.attrs.items():
        out[attr_name] = _decode(attr_val)

    for field in ('title', 'start_time', 'end_time', 'experiment_identifier',
                  'definition', 'pre_sample_flightpath'):
        val = _dataset_scalar(entry, field)
        if val is not None:
            out[field] = _decode(val)
    for field in ('start_time', 'end_time'):
        if field in out:
            out[field] = _display_ts(out[field])

    # v4 provenance: program_name/@version + process fields
    program = entry.get('program_name')
    if isinstance(program, h5py.Dataset):
        out['generator'] = _decode(program[()])
        version = program.attrs.get('version')
        if version is not None:
            out['generator_version'] = _decode(version)

    process = entry.get('process')
    if isinstance(process, h5py.Group):
        for field in ('data_source', 'correlation_method', 'total_scans',
                      'twod_included', 'twod_max_display_size',
                      'echem_time_tolerance', 'date'):
            val = _dataset_scalar(process, field)
            if val is not None:
                out[field] = _decode(val)

    for group_name in ('instrument', 'sample', 'user', 'cycling_protocol'):
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

    start = _display_ts(_decode(_dataset_scalar(sub, 'start_time')))
    end = _display_ts(_decode(_dataset_scalar(sub, 'end_time')))

    # Environment: correlated electrochemistry + timestamps
    env = sub.get('environment')
    if isinstance(env, h5py.Group):
        scan.timestamp = _display_ts(
            _decode(_dataset_scalar(env, 'scan_timestamp'))) or start
        scan.midpoint_timestamp = _display_ts(_decode(
            _dataset_scalar(env, 'midpoint_adjusted_timestamp')))
        scan.echem_timestamp = _display_ts(
            _decode(_dataset_scalar(env, 'voltage_timestamp')))
        scan.echem = _float_or_none(
            _first_scalar(env, 'voltage', 'voltage (V)'))
        scan.current = _float_or_none(
            _first_scalar(env, 'current', 'current (mA)'))
        scan.exposure_time = _float_or_none(_dataset_scalar(env, 'exposure_time'))
        scan.voltage_min = _float_or_none(_dataset_scalar(env, 'voltage_min'))
        scan.voltage_max = _float_or_none(_dataset_scalar(env, 'voltage_max'))
        scan.current_min = _float_or_none(_dataset_scalar(env, 'current_min'))
        scan.current_max = _float_or_none(_dataset_scalar(env, 'current_max'))
        scan.echem_index_start = _int_or_none(
            _first_scalar(env, 'echem_index_first', 'echem_index_start'))
        scan.echem_index_end = _int_or_none(
            _first_scalar(env, 'echem_index_last', 'echem_index_end'))
        scan.capacity = _float_or_none(_dataset_scalar(env, 'capacity'))
    else:
        scan.timestamp = start

    # Neutron acquisition window (both bounds written only for neutron scans;
    # v4 XRD scans carry end_time = start + exposure, which is not a window).
    # Gated on the NXtofnpd definition (bank groups as fallback) so an XRD
    # scan whose data failed to read is not misread as a neutron window.
    definition = _decode(_dataset_scalar(sub, 'definition'))
    is_neutron = (definition == 'NXtofnpd' if definition else
                  any(_BANK_GROUP_RE.fullmatch(k) for k in sub.keys()))
    if end and is_neutron and not (sub.get('data') or sub.get('image_source')):
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

    # v4 2D image reference/embed (subentry level; overrides v3 data attrs)
    note = sub.get('image_source')
    if isinstance(note, h5py.Group):
        scan.twod_source = _decode(_dataset_scalar(note, 'file_name'))
        embedded = _dataset_scalar(note, 'embedded')
        scan.twod_embedded = bool(embedded) if embedded is not None else False
        image = sub.get('image_data')
        if (scan.twod_embedded and isinstance(image, h5py.Group)
                and 'data' in image):
            scan.twod = np.asarray(image['data'][()], dtype=float)

    # Neutron banks: v4 bank_N (TOF) + bank_N_d (d-spacing) pairs, with the
    # v3 combined-group fields as fallback inside bank_N
    banks: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in sub.keys():
        m = _BANK_GROUP_RE.fullmatch(key)
        if not m or not isinstance(sub[key], h5py.Group):
            continue
        b = sub[key]
        bank_num, is_d = m.group(1), bool(m.group(2))
        if is_d:
            trace = _read_bank_trace(b, 'd_spacing', 'data', 'errors',
                                     ('source_file',))
            if trace:
                banks.setdefault(bank_num, {})['d'] = trace
        else:
            trace = _read_bank_trace(b, 'time_of_flight', 'data', 'errors',
                                     ('source_file', 'tof_source_file'))
            if trace:
                banks.setdefault(bank_num, {})['tof'] = trace
            # v3 combined group: d family lives alongside TOF
            trace = _read_bank_trace(b, 'd_spacing', 'd_data', 'd_errors',
                                     ('d_source_file',))
            if trace:
                banks.setdefault(bank_num, {})['d'] = trace
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


def _read_bank_trace(b: h5py.Group, x_name: str, y_name: str, e_name: str,
                     source_attrs: tuple) -> Optional[Dict[str, Any]]:
    """One x/y(/e) trace from a bank NXdata group, or None when absent."""
    x = _dataset_1d(b, x_name)
    y = _dataset_1d(b, y_name)
    if x is None or y is None or len(x) != len(y):
        return None

    source = None
    for attr in source_attrs:
        source = _decode(b.attrs.get(attr))
        if source is not None:
            break

    trace = {"x": np.asarray(x, dtype=float),
             "y": np.asarray(y, dtype=float),
             "source": source}
    e = _dataset_1d(b, e_name)
    if e is not None and len(e) == len(x):
        trace["e"] = np.asarray(e, dtype=float)
    return trace


# ============================================================================
# Legacy layouts (schema 2.0 and v1 files)
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

def _read_echem_series(grp: h5py.Group) -> Optional[pd.DataFrame]:
    """One echem NXdata group -> timestamp/echem_data/current DataFrame.
    Reads the v4 time/@start convention first, then the v3 string arrays."""
    # v4: time/@start + voltage/current
    v = _dataset_1d(grp, 'voltage')
    time_item = grp.get('time')
    if v is not None and isinstance(time_item, h5py.Dataset):
        start = _decode(time_item.attrs.get('start'))
        seconds = np.asarray(time_item[()], dtype=float).reshape(-1)
        if start is not None and len(seconds) == len(v):
            start_ts = pd.to_datetime(start)
            if start_ts.tzinfo is not None:
                # in-memory model uses naive local wall time throughout
                start_ts = start_ts.tz_localize(None)
            # float-seconds storage can be ~1 ns off; source data (and the
            # v3 string format) never exceeds microsecond resolution
            timestamps = (start_ts + pd.to_timedelta(seconds, unit='s')).round('us')
            df = pd.DataFrame({"timestamp": timestamps,
                               "echem_data": np.asarray(v, dtype=float)})
            i = _dataset_1d(grp, 'current')
            df["current"] = (np.asarray(i, dtype=float)
                             if i is not None and len(i) == len(v) else np.nan)
            return df

    # v3: string timestamps + unit-suffixed names
    ts = _dataset_1d(grp, 'timestamps')
    v = _dataset_1d(grp, 'voltage (V)')
    if ts is None or v is None or len(ts) != len(v):
        return None
    i = _dataset_1d(grp, 'current (mA)')
    timestamps = pd.to_datetime([_decode(t) for t in ts])
    df = pd.DataFrame({"timestamp": timestamps, "echem_data": v.astype(float)})
    df["current"] = (i.astype(float)
                     if i is not None and len(i) == len(ts) else np.nan)
    return df


def _read_operando_echem(parent: h5py.Group, model: ExperimentModel) -> None:
    """Cycling protocol -> model.echem_df (empty frame when absent/invalid)."""
    df = None
    if 'operando_electrochemistry' in parent:
        df = _read_echem_series(parent['operando_electrochemistry'])
    if df is None:
        df = pd.DataFrame(columns=["timestamp", "echem_data", "current"])
    model.echem_df = df


def _read_standard_echem(parent: h5py.Group, model: ExperimentModel) -> None:
    """Each stored file_N dataset -> model.standard_echem entry."""
    if 'standard_electrochemistry' not in parent:
        return

    se = parent['standard_electrochemistry']
    for name in sorted(se.keys()):
        if not name.startswith('file_'):
            continue
        df = _read_echem_series(se[name])
        if df is None:
            continue
        # Preserve the historical shape: standard-echem frames only carry a
        # current column when the file actually had current data
        if df["current"].isna().all():
            df = df.drop(columns=["current"])

        model.standard_echem.append({
            "name": name,
            "source_file": _decode(se[name].attrs.get('source_file')),
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


def _first_scalar(grp: h5py.Group, *names: str) -> Optional[Any]:
    """First present scalar dataset among names (v4 name, then v3 fallback)."""
    for name in names:
        val = _dataset_scalar(grp, name)
        if val is not None:
            return val
    return None


def _int_or_none(value: Any) -> Optional[int]:
    """Value as int, or None when missing or non-numeric."""
    if value is None:
        return None
    try:
        return int(np.array(value).squeeze())
    except (ValueError, TypeError):
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
    """Native Python value for HDF5 bytes/numpy scalars (arrays -> lists)."""
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
