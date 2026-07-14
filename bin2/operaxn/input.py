"""
Input adapter: bridges the OperaXN GUI to the core NXS-canonical pipeline.

Every upload goes through the canonical .nxs file:
- raw data  -> core.generate() into a cache .nxs -> core.load()
- .nxs file -> core.load() directly

This module exposes the same API the GUI has always consumed
(process_paths, make_*_arrays, get_correlated_data), but the data now comes
from the loaded ExperimentModel instead of re-reading raw files per scan.
Relative time is applied here as a display-only view transform; the .nxs
always stores absolute timestamps.
"""

import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

from core.config import SYNCHROTRON_MAX_DISPLAY_SIZE
from core.classify import (  # noqa: F401  (GUI scan labels)
    NeutronFileGrouper,
    SynchrotronFileGrouper,
)
from core.correlate import EchemParser, format_relative_time, parse_relative_time
from core.generate import (
    cache_dir, convert_tabular_to_txt, default_cache_path, generate,
)
from core.model import (  # noqa: F401  (re-exported for package API)
    DataSourceType,
    DataType,
    FileType,
    ScanData,
    TimeMethod,
)
from core.nxs_reader import is_canonical_nxs, load
from core.readers import (  # noqa: F401  (clear_global_cache used by main.py)
    DataReaderFactory,
    HDFReader,
    clear_global_cache,
)

from .config import OPERAXNTheme, WINDOW_SIZES

logger = logging.getLogger(__name__)

# Context of the most recent load, so the GUI can export the canonical file
# and the unsaved cache file can be cleaned up.
_session: Dict[str, Any] = {
    "nxs_path": None,
    "input_paths": None,  # None = opened a .nxs directly (user's own file)
    "exported": False,
    "standard_echem": [],
}


def cleanup_session_cache() -> None:
    """Delete the session's auto-generated cache .nxs if it was never exported.

    Never touches user files: only paths inside the cache directory that came
    from a raw-data load are eligible. Called on app exit and whenever a new
    load replaces the session.
    """
    path = _session.get("nxs_path")
    if not path or _session.get("exported"):
        return
    if _session.get("input_paths") is None:
        return  # direct .nxs upload: the file is the user's own

    try:
        in_cache = os.path.dirname(os.path.abspath(path)) == os.path.abspath(cache_dir())
        if in_cache and os.path.isfile(path):
            os.remove(path)
            logger.debug(f"Removed unsaved cache file: {path}")
    except OSError as e:
        logger.debug(f"Could not remove cache file {path}: {e}")


# ============================================================================
# Model -> GUI scan dict conversion
# ============================================================================

def _scan_to_dict(sd: ScanData) -> Dict[str, Any]:
    """Convert a loaded ScanData into the dict shape the GUI consumes.

    `oned`/`twod`/`neutron_files` hold source *names* for display; the arrays
    themselves live on the attached ScanData under the private `_data` key.
    """
    neutron_files = None
    if sd.neutron:
        neutron_files = {}
        for meas, bank in sd.neutron.items():
            entry = {}
            for key in ("tof", "d"):
                if key in bank:
                    entry[key] = bank[key].get("source") or f"bank_{meas}_{key}.dat"
            neutron_files[meas] = entry

    twod_available = sd.twod is not None or bool(
        sd.twod_source and os.path.isfile(str(sd.twod_source)))

    timestamp_for_correlation = None
    for candidate in (sd.midpoint_timestamp, sd.timestamp):
        if candidate:
            try:
                timestamp_for_correlation = pd.to_datetime(candidate)
                break
            except (ValueError, TypeError):
                pass

    return {
        "scan_num": sd.scan_num,
        "oned": sd.oned.get("source") if sd.oned else None,
        "twod": sd.twod_source if twod_available else None,
        "echem": sd.echem,
        "current": sd.current,
        "echem_timestamp": sd.echem_timestamp,
        "neutron_meta": None,
        "neutron_files": neutron_files,
        "timestamp": sd.timestamp,
        "original_timestamp": sd.timestamp,
        "exposure_time": sd.exposure_time,
        "oned_exposure": None,
        "twod_exposure": None,
        "neutron_start": sd.neutron_start,
        "neutron_end": sd.neutron_end,
        "timestamp_for_correlation": timestamp_for_correlation,
        "_data": sd,
    }


def _apply_relative_view(scan_dicts: List[Dict[str, Any]],
                         echem_df: Optional[pd.DataFrame],
                         data_source: DataSourceType) -> None:
    """Display-only transform: rewrite timestamps as HH:MM:SS offsets.

    Scan t=0 is the earliest correlation midpoint (XRD scans only for
    XRD sources, all scans for neutron); echem t=0 is its earliest point.
    Mirrors the old ScanProcessor._apply_relative_time behaviour.
    """
    if data_source == DataSourceType.NEUTRON:
        candidates = scan_dicts
    else:
        candidates = [s for s in scan_dicts if s.get("oned") or s.get("twod")]

    refs = [s["timestamp_for_correlation"] for s in candidates
            if s.get("timestamp_for_correlation") is not None]
    reference_time = min(refs) if refs else None

    if reference_time is not None:
        for scan in scan_dicts:
            base_time = scan.get("timestamp_for_correlation")
            if base_time is None and scan.get("timestamp"):
                try:
                    base_time = pd.to_datetime(scan["timestamp"])
                except (ValueError, TypeError):
                    base_time = None

            if base_time is not None:
                scan["original_timestamp"] = scan["timestamp"]
                relative_seconds = (base_time - reference_time).total_seconds()
                scan["timestamp"] = format_relative_time(relative_seconds)

    if echem_df is not None and not echem_df.empty:
        timestamps = pd.to_datetime(echem_df["timestamp"])
        echem_reference = timestamps.min()
        relative_seconds = (timestamps - echem_reference).dt.total_seconds()

        echem_df["original_timestamp"] = echem_df["timestamp"].copy()
        echem_df["timestamp"] = [format_relative_time(s) for s in relative_seconds]


# ============================================================================
# Main API: process_paths
# ============================================================================

def process_paths(selected_paths: List[str],
                  progress_callback: Optional[Callable] = None,
                  time_method: Optional[TimeMethod] = None,
                  data_source: DataSourceType = DataSourceType.INHOUSE,
                  standard_echem_files: Optional[List[str]] = None,
                  include_2d_images: bool = False,
                  twod_max_display_size: int = 0,
                  title: Optional[str] = None,
                  sample_name: Optional[str] = None
                  ) -> tuple:
    """Load an experiment: generate the canonical .nxs (or take one directly),
    load it, and return (scan_dicts, echem_df, time_method_str).

    `standard_echem_files`, `include_2d_images`, `twod_max_display_size`,
    `title`, and `sample_name` are generation options: they determine what is
    stored in the .nxs and are ignored when a .nxs is opened directly."""
    paths = [str(p) for p in selected_paths]
    empty = ([], pd.DataFrame(columns=["timestamp", "echem_data", "current"]),
             TimeMethod.ABSOLUTE.value)

    if not paths:
        return empty

    if time_method is None:
        time_method = TimeSortingDialog.ask_method()
    if time_method is None:  # dialog dismissed
        time_method = TimeMethod.ABSOLUTE

    # A new load replaces the session: drop the previous unsaved cache file
    cleanup_session_cache()

    # Direct canonical .nxs upload skips generation
    direct_nxs = (len(paths) == 1 and paths[0].lower().endswith(".nxs")
                  and is_canonical_nxs(paths[0]))

    if direct_nxs:
        nxs_path = paths[0]
        if progress_callback:
            progress_callback("Opening NeXus file...")
    else:
        nxs_path = default_cache_path(paths)
        success, messages = generate(
            paths, nxs_path,
            data_source=data_source,
            time_method=time_method,
            standard_echem_files=standard_echem_files,
            include_2d_images=include_2d_images,
            max_display_size=twod_max_display_size,
            title=title,
            sample_name=sample_name,
            progress_callback=progress_callback,
        )
        if not success:
            logger.error(f"NeXus generation failed: {messages}")
            return empty

    _session["nxs_path"] = os.path.abspath(nxs_path)
    _session["input_paths"] = None if direct_nxs else paths
    _session["exported"] = False

    model = load(nxs_path)
    _session["global_metadata"] = dict(model.global_metadata or {})
    _session["standard_echem"] = [dict(item) for item in model.standard_echem]

    if model.data_source and model.data_source != data_source.value:
        logger.warning(
            f"Loaded file was generated as '{model.data_source}' but "
            f"'{data_source.value}' was selected; displaying file contents as-is")

    scan_dicts = [_scan_to_dict(sd) for sd in model.scans]
    echem_df = model.echem_df if model.echem_df is not None else pd.DataFrame(
        columns=["timestamp", "echem_data", "current"])

    if time_method == TimeMethod.RELATIVE:
        _apply_relative_view(scan_dicts, echem_df, data_source)

    if progress_callback:
        progress_callback(f"Loaded {len(scan_dicts)} scans")

    return scan_dicts, echem_df, time_method.value


# ============================================================================
# NeXus export
# ============================================================================

def get_loaded_nxs_path() -> Optional[str]:
    """Path of the canonical .nxs backing the currently loaded experiment."""
    return _session["nxs_path"]


def get_experiment_metadata() -> Dict[str, Any]:
    """Experiment-level metadata of the loaded .nxs (for exports/display)."""
    return dict(_session.get("global_metadata") or {})


def get_standard_echem() -> List[Dict[str, Any]]:
    """Standard (non-operando) echem datasets stored in the loaded .nxs.

    Each entry has 'name', 'source_file', and 'data' (a DataFrame with
    timestamp/echem_data and, when present in the file, current columns)."""
    return list(_session.get("standard_echem") or [])


def _write_standard_echem_group(nxs_path: str,
                                datasets: List[Dict[str, Any]]) -> None:
    """Replace the standard-echem collection in a canonical NeXus file."""
    with h5py.File(nxs_path, "a") as nxs:
        parent = nxs["entry"] if "entry" in nxs else nxs
        if "standard_electrochemistry" in parent:
            del parent["standard_electrochemistry"]
        container = parent.create_group("standard_electrochemistry")
        container.attrs["NX_class"] = "NXcollection"
        container.attrs["num_files"] = len(datasets)
        string_type = h5py.string_dtype(encoding="utf-8")

        for index, item in enumerate(datasets, start=1):
            data = item["data"]
            group = container.create_group(f"file_{index:03d}")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["source_file"] = str(item.get("source_file") or "echem")
            timestamps = data["timestamp"].astype(str).to_numpy(dtype=object)
            group.create_dataset("timestamps", data=timestamps, dtype=string_type)
            group.create_dataset(
                "voltage (V)", data=data["echem_data"].to_numpy(dtype=float))
            if "current" in data.columns:
                current = data["current"].to_numpy(dtype=float)
                if not np.all(np.isnan(current)):
                    group.create_dataset("current (mA)", data=current)


def add_standard_echem_files(paths: List[str]) -> List[Dict[str, Any]]:
    """Append echem files to the loaded experiment without replacing its data.

    The backing cache is updated for subsequent NeXus export. A directly
    opened user NeXus file is copied into the cache before it is modified.
    Returns the successfully parsed additions.
    """
    if not _session.get("nxs_path"):
        raise RuntimeError("Load experiment data before adding electrochemistry")

    parser = EchemParser()
    additions = []
    for path in paths:
        source_path = str(path)
        parse_path = convert_tabular_to_txt(source_path)
        try:
            data = parser.parse(parse_path)
        finally:
            if parse_path != source_path:
                try:
                    os.remove(parse_path)
                except OSError:
                    pass
        if data is None or data.empty:
            logger.warning("No electrochemistry data found in %s", source_path)
            continue
        additions.append({
            "source_file": os.path.basename(source_path),
            "data": data,
        })

    if not additions:
        return []

    combined = list(_session.get("standard_echem") or [])
    positions = {
        str(item.get("source_file", "")).casefold(): index
        for index, item in enumerate(combined)
    }
    for addition in additions:
        key = addition["source_file"].casefold()
        if key in positions:
            combined[positions[key]] = addition
        else:
            positions[key] = len(combined)
            combined.append(addition)

    nxs_path = os.path.abspath(_session["nxs_path"])
    if _session.get("input_paths") is None:
        stem = os.path.splitext(os.path.basename(nxs_path))[0]
        working_path = os.path.join(cache_dir(), f"{stem}_working.nxs")
        shutil.copy2(nxs_path, working_path)
        nxs_path = working_path
        _session["nxs_path"] = nxs_path
        _session["input_paths"] = [os.path.abspath(paths[0])]

    _write_standard_echem_group(nxs_path, combined)
    _session["standard_echem"] = combined
    _session["exported"] = False
    return additions


def export_nxs(destination: str,
               progress_callback: Optional[Callable] = None) -> tuple:
    """Save the loaded experiment's canonical .nxs to `destination`.

    A plain copy: the file already contains everything chosen at generation
    time (standard echem, 2D images or references). Returns (success, messages)."""
    source = _session["nxs_path"]
    if not source or not os.path.isfile(source):
        return False, ["No experiment loaded"]

    if progress_callback:
        progress_callback("Exporting NeXus file...")
    if os.path.abspath(source) != os.path.abspath(destination):
        shutil.copyfile(source, destination)
    _session["exported"] = True
    return True, ["NeXus file exported"]


# ============================================================================
# Plot array builders (data comes from the loaded model, not raw files)
# ============================================================================

def _scan_data(scan: Dict[str, Any]) -> Optional[ScanData]:
    """The ScanData carried on a GUI scan dict (None for synthetic entries)."""
    return scan.get("_data")


def _load_twod_image(sd: ScanData, state: Any = None) -> Optional[np.ndarray]:
    """Return the 2D image: embedded array, or lazy-load the referenced source
    file (2D is stored by reference unless embedding was requested)."""
    if sd.twod is not None:
        return sd.twod

    path = sd.twod_source
    if not path or not os.path.isfile(str(path)):
        return None

    try:
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".hdf":
            max_size = SYNCHROTRON_MAX_DISPLAY_SIZE
            if state is not None and hasattr(state, "synchrotron_max_size"):
                max_size = state.synchrotron_max_size
            reader = HDFReader(max_display_size=max_size)
            return reader.read(str(path))
        return DataReaderFactory.read_file(str(path))
    except Exception as e:
        logger.error(f"Error loading 2D source {path}: {e}")
        return None


def make_oned_arrays(scans: List[Dict[str, Any]], state: Any = None
                     ) -> Dict[int, Dict[str, Any]]:
    """Build {scan_num: {x, y, timestamp, echem, current}} from 1D scans."""
    plot_data = {}

    for scan in scans:
        sd = _scan_data(scan)
        if sd is None or sd.oned is None:
            continue

        plot_data[scan["scan_num"]] = {
            "x": sd.oned["x"],
            "y": sd.oned["y"],
            "timestamp": scan["timestamp"],
            "echem": scan.get("echem"),
            "current": scan.get("current"),
        }

    return plot_data


def make_twod_arrays(scans: List[Dict[str, Any]], state: Any = None
                     ) -> Dict[int, Dict[str, Any]]:
    """Build {scan_num: {image, timestamp, echem, current}} from 2D scans."""
    plot_data = {}

    for scan in scans:
        sd = _scan_data(scan)
        if sd is None or not scan.get("twod"):
            continue

        image = _load_twod_image(sd, state)

        if image is not None:
            plot_data[scan["scan_num"]] = {
                "image": image,
                "timestamp": scan["timestamp"],
                "echem": scan.get("echem"),
                "current": scan.get("current"),
            }
        else:
            plot_data[scan["scan_num"]] = {"error": True}

    return plot_data


def make_neutron_arrays(scans: List[Dict[str, Any]], state: Any = None
                        ) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Build {scan_num: {measurement: {type: {x, y}}}} from neutron scans."""
    plot_data = {}

    for scan in scans:
        sd = _scan_data(scan)
        if sd is None or not sd.neutron:
            continue

        scan_data = {}
        for measurement_num, bank in sd.neutron.items():
            measurement_data = {}
            for key in ("tof", "d"):
                if key in bank:
                    measurement_data[key] = {"x": bank[key]["x"], "y": bank[key]["y"]}
            if measurement_data:
                scan_data[measurement_num] = measurement_data

        if scan_data:
            plot_data[scan["scan_num"]] = scan_data

    return plot_data


def make_echem_arrays(echem_df: Optional[pd.DataFrame],
                      time_method: str = "absolute"
                      ) -> Dict[str, Union[np.ndarray, List]]:
    """Convert echem DataFrame to {x, y, current, timestamps} arrays."""
    if echem_df is None or echem_df.empty:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "current": np.array([]),
            "timestamps": []
        }

    if time_method == TimeMethod.RELATIVE.value:
        time_seconds = np.array([
            parse_relative_time(ts) or 0 for ts in echem_df["timestamp"]
        ])
    else:
        try:
            timestamps = pd.to_datetime(echem_df["timestamp"])
            time_seconds = (timestamps - timestamps.min()).dt.total_seconds().values
        except Exception as e:
            logger.error(f"Error processing timestamps: {e}")
            time_seconds = np.arange(len(echem_df))

    echem_values = echem_df["echem_data"].values
    current_values = (echem_df["current"].values if "current" in echem_df.columns
                      else np.full(len(echem_df), np.nan))

    return {
        "x": time_seconds,
        "y": echem_values,
        "current": current_values,
        "timestamps": echem_df["timestamp"].tolist()
    }


def get_correlated_data(scans: List[Dict[str, Any]],
                        echem_df: Optional[pd.DataFrame],
                        scan_num: int,
                        state: Any = None) -> Optional[Dict[str, Any]]:
    """Collect all data (1D, 2D, neutron, echem values) for a single scan."""
    scan = next((s for s in scans if s["scan_num"] == scan_num), None)
    if not scan:
        return None

    sd = _scan_data(scan)
    if sd is None:
        return None

    result = {
        "scan_num": scan_num,
        "timestamp": scan["timestamp"],
        "echem_value": scan.get("echem"),
        "current_value": scan.get("current")
    }

    result["oned"] = ({"x": sd.oned["x"], "y": sd.oned["y"]}
                      if sd.oned is not None else None)

    result["twod"] = _load_twod_image(sd, state) if scan.get("twod") else None

    if sd.neutron:
        neutron_data = {}
        for measurement_num, bank in sd.neutron.items():
            measurement_data = {}
            for key in ("tof", "d"):
                if key in bank:
                    measurement_data[key] = {"x": bank[key]["x"], "y": bank[key]["y"]}
            if measurement_data:
                neutron_data[measurement_num] = measurement_data
        if neutron_data:
            result["neutron"] = neutron_data

    return result


# ============================================================================
# Time method dialog
# ============================================================================

class TimeSortingDialog:
    """Tkinter dialog for choosing absolute vs. relative time display."""

    @staticmethod
    def ask_method() -> Optional[TimeMethod]:
        """Show modal dialog and return chosen TimeMethod (None if dismissed)."""
        import tkinter as tk

        dialog = tk.Toplevel()
        dialog.title("Time Sorting Method")
        dialog.geometry(WINDOW_SIZES['time'])
        dialog.transient()

        dialog.configure(bg=OPERAXNTheme.COLORS['bg_primary'])

        result = {"method": TimeMethod.ABSOLUTE}

        dialog.protocol("WM_DELETE_WINDOW",
                        lambda: [result.update({"method": None}), dialog.destroy()])

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        tk.Label(
            dialog,
            text="How should timestamps be handled?",
            font=OPERAXNTheme.FONTS['heading'],
            bg=OPERAXNTheme.COLORS['bg_primary'],
            fg=OPERAXNTheme.COLORS['text_primary']
        ).pack(pady=20)

        var = tk.StringVar(value=TimeMethod.ABSOLUTE.value)

        options = [
            (TimeMethod.ABSOLUTE.value, "Absolute time (use actual timestamps)"),
            (TimeMethod.RELATIVE.value, "Relative time (XRD and echem each start at 00:00:00)")
        ]

        for value, text in options:
            tk.Radiobutton(
                dialog,
                text=text,
                variable=var,
                value=value,
                font=OPERAXNTheme.FONTS['body'],
                bg=OPERAXNTheme.COLORS['bg_primary'],
                fg=OPERAXNTheme.COLORS['text_primary'],
                activebackground=OPERAXNTheme.COLORS['bg_primary'],
                activeforeground=OPERAXNTheme.COLORS['accent_primary'],
                selectcolor=OPERAXNTheme.COLORS['bg_tertiary']
            ).pack(anchor="w", padx=30, pady=5)

        def confirm():
            result["method"] = TimeMethod(var.get())
            dialog.destroy()

        button_frame = tk.Frame(dialog, bg=OPERAXNTheme.COLORS['bg_primary'])
        button_frame.pack(pady=10)

        confirm_btn = tk.Button(
            button_frame,
            text="OK",
            command=confirm,
            width=12,
            bg=OPERAXNTheme.COLORS['accent_primary'],
            fg=OPERAXNTheme.COLORS['bg_primary'],
            font=OPERAXNTheme.FONTS['button'],
            relief=tk.FLAT,
            cursor='hand2'
        )
        confirm_btn.pack()

        confirm_btn.bind("<Enter>", lambda e: confirm_btn.config(bg=OPERAXNTheme.COLORS['accent_hover']))
        confirm_btn.bind("<Leave>", lambda e: confirm_btn.config(bg=OPERAXNTheme.COLORS['accent_primary']))

        dialog.wait_window()
        return result["method"]
