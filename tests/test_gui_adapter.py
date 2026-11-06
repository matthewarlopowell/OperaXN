"""operaxn.input adapter and main window: the GUI contract over the core pipeline, exports and lifecycle."""
import argparse
import logging
import os
import shutil
import warnings
import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.gui
pytest.importorskip("tkinter")

import h5py

import core
import core.readers as readers
import operaxn
from core import nxs_writer
from operaxn import input as gui_input
from operaxn.dialog import GIFSettings, UploadOptions, UploadOptionsDialog
from operaxn.gui import OPERAXN, PerformanceMonitor
from operaxn.main import main as cli_main
from operaxn.input import (
    DataSourceType,
    TimeMethod,
    add_standard_echem_files,
    cleanup_session_cache,
    experiment_metadata_rows,
    export_nxs,
    get_correlated_data,
    get_experiment_metadata,
    get_experiment_stem,
    get_loaded_data_source,
    get_loaded_nxs_path,
    get_standard_echem,
    make_echem_arrays,
    make_neutron_arrays,
    make_oned_arrays,
    make_twod_arrays,
    process_paths,
    reset_session,
)
from operaxn.output import get_scan_time_positions

import _builders
from _helpers import needs_fabio


def _reset_session():
    """Restore operaxn.input._session to its pristine module state."""
    gui_input._session.update({"nxs_path": None, "input_paths": None,
                               "exported": False, "standard_echem": [],
                               "stem": None, "data_source": None, "images_nxs": None})
    gui_input._session.pop("global_metadata", None)


@pytest.fixture(autouse=True)
def _own_session():
    """Give each test a fresh module-global session and drop its cache .nxs after."""
    _reset_session()
    yield
    if gui_input._session.get("input_paths") is not None:
        gui_input._session["exported"] = False
    cleanup_session_cache()
    _reset_session()


def _discard(path):
    """Remove a cache file the test left behind in the shared cache dir."""
    if path and os.path.isfile(path):
        os.remove(path)


def _write_edf(path):
    """Write an 8x8 EDF image whose Date pairs it with in-house scan 1."""
    import fabio
    img = fabio.edfimage.EdfImage(
        data=np.arange(64, dtype=np.float32).reshape(8, 8) + 1.0,
        header={"Date": _builders.SCAN_TIMES[0], "ExposureTime": "120.0",
                "WaveLength": "1.54"})
    img.write(path)


@pytest.fixture
def inhouse_edf_dir(inhouse_dir):
    """Return the in-house dataset plus an EDF 2D image paired with scan 1 by Date."""
    _write_edf(os.path.join(inhouse_dir, "image_001.edf"))
    return inhouse_dir


def _std_echem_file(path, **overrides):
    """Write a standard (non-operando) echem file distinct from the operando series."""
    options = dict(n_rows=10, v0=4.0, dv=0.01, i0=0.5, di=0.0,
                   date="06/02/2024", hour=9)
    options.update(overrides)
    _builders.write_echem_txt(path, **options)


def test_package_exports_core_enums():
    """Full package import succeeds and config re-exports the core enum."""
    assert operaxn.config.DataSourceType is core.DataSourceType


def test_raw_upload_absolute(inhouse_dir):
    """Raw in-house upload yields correlated scan dicts with midpoint matching."""
    progress = []
    scans, _, method = process_paths(
        [inhouse_dir], progress.append,
        time_method=TimeMethod.ABSOLUTE, data_source=DataSourceType.INHOUSE)
    assert method == "absolute", method
    assert len(scans) == 3, f"got {len(scans)}"
    assert len(progress) > 0
    s1 = scans[0]
    assert s1["echem"] == 3.76, f"got {s1['echem']}"
    assert s1["echem_timestamp"] == "2024-02-05 10:06:00", \
        f"got {s1['echem_timestamp']}"
    assert s1["oned"] == "scan_001.dat", f"got {s1['oned']}"
    assert s1["timestamp_for_correlation"] == pd.Timestamp("2024-02-05 10:06:00"), \
        f"got {s1['timestamp_for_correlation']}"


@needs_fabio
def test_raw_upload_pairs_edf_and_lazy_loads(inhouse_edf_dir):
    """Scan 1 gains a 2D reference and make_twod_arrays lazy-loads the EDF."""
    scans, _, _ = process_paths(
        [inhouse_edf_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    assert bool(scans[0]["twod"]), f"got {scans[0]['twod']}"
    twod_arrays = make_twod_arrays(scans)
    assert 1 in twod_arrays and "image" in twod_arrays[1], \
        str(twod_arrays.get(1, {}).keys())
    assert twod_arrays[1]["image"].shape == (8, 8)


def test_make_oned_arrays(inhouse_dir):
    """1D plot arrays carry exact data and the correlated voltage per scan."""
    scans, _, _ = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    oned_arrays = make_oned_arrays(scans)
    assert set(oned_arrays.keys()) == {1, 2, 3}
    assert np.allclose(oned_arrays[1]["x"], _builders.ONED_X)
    assert np.allclose(oned_arrays[1]["y"], _builders.inhouse_scan_y(1))
    assert oned_arrays[1]["echem"] == 3.76


def test_make_echem_arrays_absolute(inhouse_dir):
    """Echem arrays cover all points with x as seconds from the first point."""
    _, echem_df, method = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    echem_arrays = make_echem_arrays(echem_df, method)
    assert len(echem_arrays["x"]) == 40
    assert echem_arrays["x"][0] == 0.0


def test_get_correlated_data(inhouse_dir):
    """get_correlated_data bundles the 1D arrays and voltage for one scan."""
    scans, _, _ = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    corr = get_correlated_data(scans, 1)
    assert corr["oned"] is not None
    assert np.allclose(corr["oned"]["x"], _builders.ONED_X)
    assert corr["echem_value"] == 3.76


def test_direct_nxs_upload_matches_raw(inhouse_dir, tmp_path):
    """Opening the generated .nxs directly reproduces the raw-upload results."""
    scans, echem_df, _ = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    oned_arrays = make_oned_arrays(scans)
    nxs_copy = str(tmp_path / "shared_experiment.nxs")
    shutil.copy(core.default_cache_path([inhouse_dir]), nxs_copy)

    scans2, echem_df2, _ = process_paths(
        [nxs_copy], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    assert len(scans2) == 3, f"got {len(scans2)}"
    assert [s["echem"] for s in scans2] == [s["echem"] for s in scans]
    assert np.allclose(make_oned_arrays(scans2)[1]["y"], oned_arrays[1]["y"])
    assert np.allclose(echem_df2["echem_data"].values,
                       echem_df["echem_data"].values)


def test_relative_view_transform(inhouse_dir):
    """Relative view rewrites timestamps as offsets and re-zeroes matching."""
    scans_r, echem_r, method_r = process_paths(
        [inhouse_dir], time_method=TimeMethod.RELATIVE,
        data_source=DataSourceType.INHOUSE)
    assert method_r == "relative", method_r
    assert scans_r[0]["timestamp"] == "00:00:00", f"got {scans_r[0]['timestamp']}"
    assert scans_r[1]["timestamp"] == "00:05:00", f"got {scans_r[1]['timestamp']}"
    assert scans_r[0]["original_timestamp"] == "2024-02-05 10:05:00", \
        f"got {scans_r[0]['original_timestamp']}"
    assert echem_r["timestamp"].iloc[0] == "00:00:00", \
        f"got {echem_r['timestamp'].iloc[0]}"
    assert scans_r[0]["echem"] == 3.70, f"got {scans_r[0]['echem']}"

    rel_echem_arrays = make_echem_arrays(echem_r, method_r)
    assert rel_echem_arrays["x"][5] == 300.0, f"got {rel_echem_arrays['x'][5]}"


@needs_fabio
def test_relative_view_reference_survives_unreadable_data(tmp_path):
    """t=0 is the earliest midpoint of all scans, as at write time, even when that scan's data is gone."""
    src = tmp_path / "src"; src.mkdir()
    _builders.write_edf_image(str(src / "image_000.edf"), "2024-02-05T10:00:00")
    _builders.write_xrd_dat(str(src / "scan_002.dat"), "2024-02-05T10:10:00")
    _builders.write_xrd_dat(str(src / "scan_003.dat"), "2024-02-05T10:15:00")
    _builders.write_echem_txt(str(src / "echem.txt"))
    out = str(tmp_path / "rel.nxs")
    assert core.generate([str(src)], out, DataSourceType.INHOUSE, time_method=TimeMethod.RELATIVE)[0]
    (src / "image_000.edf").unlink()
    scans, echem_df, method = process_paths([out], time_method=TimeMethod.RELATIVE,
                                            data_source=DataSourceType.INHOUSE)
    reset_session()
    assert [s["timestamp"] for s in scans] == ["00:00:00", "00:10:00", "00:15:00"]
    assert get_scan_time_positions(scans, echem_df, method) == [0, 600, 900]


@needs_fabio
def test_relative_view_of_edf_only_file_with_no_reachable_image(tmp_path):
    """An EDF-only file whose images are all gone still gets relative labels instead of raising."""
    src = tmp_path / "src"; src.mkdir()
    for i, ts in enumerate(["2024-02-05T10:05:00", "2024-02-05T10:10:00"], start=1):
        _builders.write_edf_image(str(src / f"image_{i:03d}.edf"), ts)
    _builders.write_echem_txt(str(src / "echem.txt"))
    out = str(tmp_path / "edf.nxs")
    assert core.generate([str(src)], out, DataSourceType.INHOUSE, time_method=TimeMethod.RELATIVE)[0]
    for f in src.glob("*.edf"):
        f.unlink()
    scans, echem_df, method = process_paths([out], time_method=TimeMethod.RELATIVE,
                                            data_source=DataSourceType.INHOUSE)
    reset_session()
    assert get_scan_time_positions(scans, echem_df, method) == [0, 300]


def test_neutron_adapter(neutron_dir):
    """Neutron upload yields one correlated scan with per-bank display names."""
    scans_n, _, _ = process_paths(
        [neutron_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.NEUTRON)
    assert len(scans_n) == 1, f"got {len(scans_n)}"
    sn = scans_n[0]
    assert sn["echem"] == 3.85, f"got {sn['echem']}"
    assert sn["neutron_files"] is not None, str(sn["neutron_files"])
    tof_name, d_name = _builders.wish_bank_names("123456", 1)
    assert sn["neutron_files"]["1"]["tof"] == tof_name, str(sn["neutron_files"])
    assert sn["neutron_files"]["1"]["d"] == d_name, str(sn["neutron_files"])

    n_arrays = make_neutron_arrays(scans_n)
    assert set(n_arrays[1].keys()) == {"1", "2"}
    assert np.allclose(n_arrays[1]["1"]["tof"]["x"], _builders.TOF_X)


def test_one_scan_load_progress_reads_singular(neutron_dir):
    """A one-run load's last progress line is 'Loaded 1 scan'."""
    lines = []
    process_paths([neutron_dir], lines.append, time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    assert lines[-1] == "Loaded 1 scan", lines


@needs_fabio
def test_embedded_twod_survives_export_and_source_deletion(inhouse_edf_dir, tmp_path):
    """Images embedded at upload survive the copy export and load once the source EDF is gone."""
    process_paths([inhouse_edf_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE, include_2d_images=True)
    exported = str(tmp_path / "exported_embedded.nxs")
    ok, msgs = export_nxs(exported)
    assert ok, str(msgs)
    os.unlink(os.path.join(inhouse_edf_dir, "image_001.edf"))

    model = core.load(exported)
    assert model.scans[0].twod_embedded and model.scans[0].twod.shape == (8, 8)
    scans, _, _ = process_paths([exported], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    twod = make_twod_arrays(scans)
    assert 1 in twod and "image" in twod[1], str(twod.get(1, {}).keys())
    assert twod[1]["image"].shape == (8, 8)
    _discard(core.default_cache_path([inhouse_edf_dir]))


def test_export_copies_canonical_file(inhouse_dir, tmp_path):
    """Export copies the generated file with scans and correlated voltages."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    export_dest = str(tmp_path / "exported.nxs")
    ok, msgs = export_nxs(export_dest)
    assert ok, str(msgs)
    exported_model = core.load(export_dest)
    assert len(exported_model.scans) == 3
    assert exported_model.scans[0].echem == 3.76


@needs_fabio
def test_export_default_keeps_twod_by_reference(inhouse_edf_dir, tmp_path):
    """Default generation stores 2D by reference, and export preserves that."""
    process_paths([inhouse_edf_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    export_dest = str(tmp_path / "exported.nxs")
    ok, msgs = export_nxs(export_dest)
    assert ok, str(msgs)
    exported_model = core.load(export_dest)
    assert not exported_model.scans[0].twod_embedded
    assert exported_model.scans[0].twod is None


def test_export_failure_keeps_previous_destination(inhouse_dir, tmp_path, monkeypatch):
    """A copy that fails part way leaves the earlier export whole and no .part beside it."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    dest = str(tmp_path / "exported.nxs")
    assert export_nxs(dest)[0]

    def half_copy(src, dst):
        with open(dst, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", half_copy)
    with pytest.raises(OSError, match="disk full"):
        export_nxs(dest)
    assert len(core.load(dest).scans) == 3
    assert not os.path.exists(dest + ".part")
    _discard(core.default_cache_path([inhouse_dir]))


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive file names")
def test_export_case_only_rename_is_a_save(inhouse_dir, tmp_path):
    """Exporting an opened Run3.nxs as run3.nxs is the same file: a save, not SameFileError."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    user_file = str(tmp_path / "Run3.nxs")
    assert export_nxs(user_file)[0]
    _discard(core.default_cache_path([inhouse_dir]))
    process_paths([user_file], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    ok, msgs = export_nxs(str(tmp_path / "run3.nxs"))
    assert ok, msgs
    assert len(core.load(user_file).scans) == 3


def test_direct_nxs_load_survives_cleanup_and_exports(inhouse_dir, tmp_path):
    """A directly opened user .nxs is never deleted by session cleanup and still exports."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    kept_export = str(tmp_path / "kept.nxs")
    export_nxs(kept_export)

    process_paths([kept_export], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cleanup_session_cache()
    assert os.path.isfile(kept_export)
    copy2 = str(tmp_path / "copy_of_direct.nxs")
    ok, _ = export_nxs(copy2)
    assert ok and os.path.isfile(copy2)
    _discard(core.default_cache_path([inhouse_dir]))


def test_cache_unsaved_removed_on_exit(inhouse_dir):
    """The unsaved session cache .nxs is deleted by cleanup on app exit."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    assert cache_file and os.path.isfile(cache_file)
    cleanup_session_cache()
    assert not os.path.exists(cache_file)


@pytest.mark.display
def test_closing_main_window_cleans_session_cache(tk_root, inhouse_dir):
    """The main window's registered close handler deletes the unsaved session cache."""
    from tkinter import Toplevel
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    assert cache_file and os.path.isfile(cache_file)
    win = Toplevel(tk_root)  # the handler destroys its master; keep the shared root
    OPERAXN(master=win)
    win.tk.eval(win.protocol("WM_DELETE_WINDOW"))
    assert not os.path.exists(cache_file), "exit cleanup must run on window close"
    assert not win.winfo_exists()


@pytest.mark.display
def test_declining_to_clear_cancels_the_upload(tk_root, monkeypatch):
    """Answering No to "Clear all data?" keeps the loaded scans and starts no load."""
    from tkinter import Toplevel, messagebox
    import operaxn.gui as gui_module
    win = Toplevel(tk_root)
    try:
        app = OPERAXN(master=win)
        loaded = [{"scan_num": 1}]
        app.state.scans = loaded
        options = UploadOptions(paths=["/new/data"], data_source=DataSourceType.INHOUSE,
                                time_method=TimeMethod.ABSOLUTE, display_size=0,
                                include_2d=False)
        monkeypatch.setattr(gui_module, "UploadOptionsDialog",
                            lambda *_a, **_k: SimpleNamespace(get_result=lambda: options))
        started = []
        monkeypatch.setattr(app, "_process_files_async", started.append)
        monkeypatch.setattr(messagebox, "askyesno", lambda *_a, **_k: False)
        app._upload_files()
        assert started == [] and app.state.scans is loaded, "No must cancel the load"
        monkeypatch.setattr(messagebox, "askyesno", lambda *_a, **_k: True)
        app._upload_files()
        assert started == [options] and not app.state.scans
    finally:
        if win.winfo_exists():
            win.destroy()


def test_cache_exported_file_kept_on_exit(inhouse_dir, tmp_path):
    """An exported session cache .nxs survives cleanup on app exit."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    export_nxs(str(tmp_path / "kept.nxs"))
    cleanup_session_cache()
    assert os.path.isfile(cache_file)


# ============================================================================
# ZIP 2D sidecar lifecycle (persist -> resolve -> export -> cleanup)
# ============================================================================

def _write_twod_zip(tmp_path):
    """Write a ZIP of one XRD .dat and one EDF image at one timestamp and return its path."""
    return _builders.write_twod_zip(str(tmp_path / "cell.zip"), tmp_path)


@needs_fabio
def test_cache_zip_images_dir_cleaned(tmp_path):
    """The images dir persisted for a ZIP input is removed with its cache file."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    images_dir = os.path.splitext(cache_file)[0] + "_images"
    assert os.path.isdir(images_dir), "images dir not persisted for ZIP input"
    cleanup_session_cache()
    assert not os.path.exists(cache_file)
    assert not os.path.isdir(images_dir)


@needs_fabio
def test_exported_zip_twod_resolves_after_cache_cleanup(tmp_path):
    """Export copies the ZIP 2D sidecar, and 2D resolves from it once the cache copy is gone."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_images = os.path.splitext(get_loaded_nxs_path())[0] + "_images"
    dest = str(tmp_path / "kept.nxs")
    ok, _ = export_nxs(dest)
    assert ok
    edfs = [f for _, _, fs in os.walk(str(tmp_path / "kept_images"))
            for f in fs if f.endswith(".edf")]
    assert edfs == ["image_001.edf"], f"sidecar not copied on export: {edfs}"
    shutil.rmtree(cache_images)  # simulate the cache being cleaned later
    scans, _, _ = process_paths([dest], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    arrays = make_twod_arrays(scans)
    assert 1 in arrays and arrays[1]["image"].shape == (8, 8), \
        "2D not resolved from the exported sibling images dir"


@needs_fabio
def test_exported_session_cleanup_drops_cache_sidecar_keeps_nxs(tmp_path):
    """Exported-session cleanup keeps the cache .nxs but drops the sidecar the export has copied."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    cache_images = os.path.splitext(cache_file)[0] + "_images"
    dest = str(tmp_path / "kept.nxs")
    ok, _ = export_nxs(dest)
    assert ok and os.path.isdir(cache_images)
    cleanup_session_cache()  # exported flag still True: app-exit behaviour
    assert os.path.isfile(cache_file), "exported cache .nxs must be kept"
    assert not os.path.isdir(cache_images), "cache sidecar must not leak"
    assert os.path.isdir(str(tmp_path / "kept_images")), "export copy intact"
    _discard(cache_file)


@needs_fabio
def test_foreign_images_dir_preserved_and_never_served(tmp_path):
    """Export spares a user folder named like the sidecar, warns and never serves it as 2D."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_images = os.path.splitext(get_loaded_nxs_path())[0] + "_images"
    dest = str(tmp_path / "kept.nxs")
    foreign = tmp_path / "kept_images"
    os.makedirs(foreign)
    (foreign / "precious.txt").write_text("user data")
    (foreign / "image_001.edf").write_bytes(b"not an image")
    ok, msgs = export_nxs(dest)
    assert ok, "export itself must still succeed"
    assert any(m.startswith("Warning") for m in msgs), \
        "sidecar copy failure must be reported to the caller"
    assert (foreign / "precious.txt").read_text() == "user data", \
        "user's folder was clobbered by the sidecar copy"
    shutil.rmtree(cache_images)
    scans, _, _ = process_paths([dest], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    assert make_twod_arrays(scans) == {}, \
        "foreign unmarked folder must not be served as 2D data"


@needs_fabio
def test_working_copy_carries_images_sidecar(tmp_path):
    """Add-echem to a reopened export keeps 2D viewable in the next export."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    v1 = str(tmp_path / "master_v1.nxs")
    ok, _ = export_nxs(v1)
    assert ok
    process_paths([v1], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    extra = str(tmp_path / "extra_echem.txt")
    _builders.write_echem_txt(extra, n_rows=5)
    added = add_standard_echem_files([extra])
    assert added, "standard echem must append"
    v2 = str(tmp_path / "master_v2.nxs")
    ok, _ = export_nxs(v2)
    assert ok
    assert os.path.isdir(str(tmp_path / "master_v2_images")), \
        "sidecar lost through the working-copy flow"
    scans, _, _ = process_paths([v2], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    arrays = make_twod_arrays(scans)
    assert 1 in arrays and arrays[1]["image"].shape == (8, 8), \
        "2D not viewable in the second-generation export"


def _add_echem_as_the_sidecar_copy_fails(tmp_path, monkeypatch, partial):
    """Reopen an exported 3-image ZIP load and add echem as its sidecar copy fails; return both.
    partial: the copy stops before scan_003.edf (a member path too long); else it copies nothing."""
    src = tmp_path / "src"
    _builders.write_dated_pairs(str(src), _builders.SCAN_TIMES, _builders.SCAN_TIMES)
    process_paths([_builders.zip_tree(str(src), str(tmp_path / "failcopy.zip"))],
                  time_method=TimeMethod.ABSOLUTE, data_source=DataSourceType.INHOUSE)
    zip_cache = get_loaded_nxs_path()
    opened = str(tmp_path / "failcopy.nxs")
    ok, _ = export_nxs(opened)
    assert ok
    scans, _, _ = process_paths([opened], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    _discard(zip_cache)

    def failing_copytree(source, destination, **kwargs):
        if not partial:
            raise OSError("no space left on device")
        for root, _, names in os.walk(source):
            target = os.path.join(destination, os.path.relpath(root, source))
            os.makedirs(target, exist_ok=True)
            for name in names:
                if name != "scan_003.edf":
                    shutil.copy2(os.path.join(root, name), target)
        raise shutil.Error([(source, destination, "path too long")])

    extra = str(tmp_path / "extra_echem.txt")
    _builders.write_echem_txt(extra, n_rows=5)
    with monkeypatch.context() as patch:
        patch.setattr(gui_input.shutil, "copytree", failing_copytree)
        assert add_standard_echem_files([extra]), "standard echem must append"
    return opened, scans


@needs_fabio
def test_failed_sidecar_copy_keeps_reading_the_opened_files_images(tmp_path, monkeypatch):
    """Add-echem whose sidecar copy fails keeps showing 2D from the opened file's sidecar."""
    _, scans = _add_echem_as_the_sidecar_copy_fails(tmp_path, monkeypatch, partial=False)
    core.clear_global_cache()
    shapes = {n: getattr(a.get("image"), "shape", a) for n, a in make_twod_arrays(scans).items()}
    assert shapes == {1: (8, 8), 2: (8, 8), 3: (8, 8)}, shapes


@needs_fabio
def test_export_after_a_partial_sidecar_copy_keeps_every_image(tmp_path, monkeypatch):
    """After a partial sidecar copy, saving over the opened file or elsewhere keeps every image."""
    opened, _ = _add_echem_as_the_sidecar_copy_fails(tmp_path, monkeypatch, partial=True)

    def images(nxs):
        return sorted(f for _, _, fs in os.walk(os.path.splitext(nxs)[0] + "_images")
                      for f in fs if f.endswith(".edf"))

    every = ["scan_001.edf", "scan_002.edf", "scan_003.edf"]
    assert export_nxs(opened)[0] and images(opened) == every, \
        "saving over the opened file lost images"
    elsewhere = str(tmp_path / "copy" / "failcopy_echem.nxs")
    os.makedirs(os.path.dirname(elsewhere))
    assert export_nxs(elsewhere)[0] and images(elsewhere) == every, \
        "the export elsewhere lost images"


@needs_fabio
def test_resolution_handles_foreign_separator_paths(tmp_path):
    """A recorded path with the other OS's separators still resolves."""
    zpath = _write_twod_zip(tmp_path)
    process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    dest = str(tmp_path / "kept.nxs")
    ok, _ = export_nxs(dest)
    assert ok
    process_paths([dest], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    recorded = "/tmp/operaxn_cache/cell_images/zip1/image_001.edf"  # POSIX form, one level per archive
    resolved = gui_input._resolve_twod_source(recorded)
    assert resolved and resolved.endswith("image_001.edf") and \
        os.path.isfile(resolved), f"got {resolved}"


@needs_fabio
def test_failed_zip_generate_leaves_no_cache_sidecar(tmp_path, monkeypatch):
    """A ZIP load whose write fails leaves neither a partial .nxs nor its _images sidecar."""
    zpath = _builders.write_twod_zip(tmp_path / "cellz.zip", tmp_path)

    def fail(self, entry, scan):
        raise OSError("simulated disk full")

    monkeypatch.setattr(nxs_writer.NXSWriter, "_write_scan", fail)
    with pytest.raises(RuntimeError, match="simulated disk full"):
        process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                      data_source=DataSourceType.INHOUSE)
    cache = core.default_cache_path([zpath])
    assert not os.path.exists(cache) and not os.path.exists(cache + ".part")
    assert not os.path.isdir(os.path.splitext(cache)[0] + "_images")


@needs_fabio
@pytest.mark.parametrize("stage", ["extract", "warm_up", "classify", "sidecar", "write"])
def test_cancelled_zip_load_raises_and_leaves_no_files(tmp_path, monkeypatch, stage):
    """A cancel mid ZIP load raises and leaves no extraction folder, .part, .nxs or sidecar."""
    import importlib
    import tempfile
    import threading
    import zipfile
    generate_module = importlib.import_module("core.generate")
    src = tmp_path / "src"
    _builders.write_dated_pairs(str(src), _builders.SCAN_TIMES, _builders.SCAN_TIMES)
    zpath = _builders.zip_tree(str(src), str(tmp_path / "cellc.zip"))
    monkeypatch.setattr(generate_module, "PARALLEL_PROCESSING_THRESHOLD", 0)
    cancel = threading.Event()
    owner, name = {"extract": (zipfile.ZipFile, "extract"),
                   "warm_up": (generate_module.FileProcessor, "_warm_up"),
                   "classify": (core.DATClassifier, "classify"),
                   "sidecar": (generate_module.shutil, "move"),
                   "write": (nxs_writer.NXSWriter, "_write_scan")}[stage]
    original = getattr(owner, name)

    def cancel_then_run(*args, **kwargs):
        cancel.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name,
                        staticmethod(cancel_then_run) if stage == "warm_up" else cancel_then_run)
    tmp, cache = tempfile.gettempdir(), core.cache_dir()
    before = set(os.listdir(tmp))
    with pytest.raises(RuntimeError, match="Generation cancelled"):
        process_paths([zpath], time_method=TimeMethod.ABSOLUTE,
                      data_source=DataSourceType.INHOUSE, cancel=cancel)
    assert set(os.listdir(tmp)) == before, "extraction folder left"
    assert [n for n in os.listdir(cache) if n.startswith("cellc")] == []


# ============================================================================
# Session cache lifecycle
# ============================================================================

def test_cache_replaced_by_new_load(inhouse_dir, neutron_dir):
    """Loading a new dataset removes the previous unsaved cache file."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    prev_cache = get_loaded_nxs_path()
    process_paths([neutron_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    assert not os.path.exists(prev_cache)
    assert os.path.isfile(get_loaded_nxs_path())


def test_standard_echem_stored_and_survives_export(inhouse_dir, tmp_path):
    """Standard echem is stored in the .nxs, kept out of operando data and carried by the export."""
    std_echem_path = str(tmp_path / "standard_cycling.txt")
    _std_echem_file(std_echem_path)

    scans_se, echem_se, _ = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE,
        standard_echem_files=[std_echem_path])
    model_s = core.load(get_loaded_nxs_path())
    assert len(model_s.standard_echem) == 1, \
        str([(e["source_file"], len(e["data"])) for e in model_s.standard_echem])
    assert len(model_s.standard_echem[0]["data"]) == 10
    assert model_s.standard_echem[0]["source_file"] == "standard_cycling.txt"
    assert abs(model_s.standard_echem[0]["data"]["echem_data"].iloc[0] - 4.0) < 1e-9

    assert len(echem_se) == 40, f"got {len(echem_se)}"
    assert scans_se[0]["echem"] == 3.76

    std_export = str(tmp_path / "with_std.nxs")
    export_nxs(std_export)
    assert len(core.load(std_export).standard_echem) == 1


def test_standard_echem_append_after_user_nxs(inhouse_dir, tmp_path):
    """Appending echem to a loaded user .nxs edits a cache copy, leaving the original untouched."""
    std_echem_path = str(tmp_path / "standard_cycling.txt")
    _std_echem_file(std_echem_path)
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE,
                  standard_echem_files=[std_echem_path])
    std_export = str(tmp_path / "with_std.nxs")
    export_nxs(std_export)

    std_echem_later = str(tmp_path / "standard_added_later.txt")
    _std_echem_file(std_echem_later, n_rows=8, v0=3.5, i0=-0.5, date="06/03/2024")
    process_paths([std_export], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    added_later = add_standard_echem_files([std_echem_later])
    assert len(added_later) == 1
    assert os.path.abspath(get_loaded_nxs_path()) != os.path.abspath(std_export)
    assert len(core.load(std_export).standard_echem) == 1
    assert [item["source_file"]
            for item in core.load(get_loaded_nxs_path()).standard_echem] \
        == ["standard_cycling.txt", "standard_added_later.txt"]
    _discard(core.default_cache_path([inhouse_dir]))


def test_added_csv_echem_is_named_in_its_unit_line(inhouse_dir, tmp_path, caplog):
    """Add echem names a .csv file in its current-unit line, not the converted temp copy."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    csv = tmp_path / "added7.csv"
    csv.write_text("Time,Ewe/V,I/A\n" + "".join(
        f"06/02/2024 09:{m:02d}:00,{4.0 + m * 0.01:.4f},0.0005\n" for m in range(10)))
    with caplog.at_level(logging.INFO, logger="core"):
        assert len(add_standard_echem_files([str(csv)])) == 1
    lines = [r.getMessage() for r in caplog.records if "current column" in r.getMessage()]
    assert lines == ["Echem current column 'I/A' in added7.csv converted to mA (x1000)"], lines
    _discard(core.default_cache_path([inhouse_dir]))


def test_unconverted_added_table_is_named_once(inhouse_dir, tmp_path, caplog):
    """Add echem names a missing or unreadable .csv/.xlsx once, by its conversion error."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    (tmp_path / "broken.xlsx").write_bytes(b"not a workbook")
    for name in ("gone.csv", "broken.xlsx"):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert add_standard_echem_files([str(tmp_path / name)]) == []
        named = [r.getMessage() for r in caplog.records if name in r.getMessage()]
        assert len(named) == 1 and named[0].startswith("Error converting tabular file "), named
    _discard(core.default_cache_path([inhouse_dir]))


@needs_fabio
def test_standard_echem_combines_with_embedded_twod(inhouse_edf_dir, tmp_path):
    """Standard echem and 2D inclusion can be combined at generation."""
    std_echem_path = str(tmp_path / "standard_cycling.txt")
    _std_echem_file(std_echem_path)
    process_paths([inhouse_edf_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE,
                  standard_echem_files=[std_echem_path],
                  include_2d_images=True, twod_max_display_size=0)
    m2 = core.load(get_loaded_nxs_path())
    assert len(m2.standard_echem) == 1
    assert m2.scans[0].twod_embedded and m2.scans[0].twod is not None


def test_upload_dialog_api():
    """The combined upload dialog exposes the expected headless API."""
    assert callable(UploadOptionsDialog)
    assert UploadOptionsDialog.DISPLAY_OPTIONS == \
        ["No downsampling", "4096", "2048", "1024", "512"]
    assert {"paths", "data_source", "time_method", "display_size", "include_2d",
            "title", "sample_name", "sample_description",
            "sample_preparation_date", "cycling_protocol",
            "standard_echem_files", "echem_time_tolerance"} \
        <= set(UploadOptions.__dataclass_fields__.keys())


def test_process_paths_threads_protocol_kwargs(inhouse_dir):
    """process_paths forwards preparation date and cycling protocol into the .nxs."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE,
                  sample_preparation_date="2024-01-15",
                  cycling_protocol={"technique": "GCPL", "c_rate": "C/10"})
    meta = core.load(get_loaded_nxs_path()).global_metadata
    assert meta["cycling_protocol"]["technique"] == "GCPL", str(meta)
    assert meta["cycling_protocol"]["C_rate"] == "C/10", str(meta)
    assert str(meta["sample"]["preparation_date"]).startswith("2024-01-15"), \
        str(meta.get("sample"))


def test_process_paths_threads_tolerance(inhouse_dir):
    """The echem time tolerance reaches metadata and Excel rows; None gives the core default."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE, echem_time_tolerance=120)
    meta = get_experiment_metadata()
    assert meta.get("echem_time_tolerance") == 120.0, str(meta)
    rows = experiment_metadata_rows(meta)
    assert {"field": "echem time tolerance", "value": "120 s"} in rows, rows
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    assert get_experiment_metadata().get("echem_time_tolerance") == \
        core.ECHEM_TIME_TOLERANCE


def test_process_paths_raises_the_generate_error(tmp_path):
    """A failed generate reaches the caller as an error naming the reason."""
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "notes.txt").write_text("nothing to see\n")
    with pytest.raises(RuntimeError, match="No scan data available"):
        process_paths([str(junk)], time_method=TimeMethod.ABSOLUTE,
                      data_source=DataSourceType.INHOUSE)


def test_process_paths_names_a_damaged_nxs(inhouse_dir, tmp_path):
    """A truncated .nxs is reported by name as damaged, not as 'No scan data available'."""
    nxs = str(tmp_path / "cut.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE)
    assert ok, msgs
    with open(nxs, "r+b") as f:
        f.truncate(os.path.getsize(nxs) // 2)
    with pytest.raises(RuntimeError, match=r"^cut\.nxs could not be opened \(damaged, incomplete "
                                           r"or in use by another program: .+\)$"):
        process_paths([nxs], time_method=TimeMethod.ABSOLUTE, data_source=DataSourceType.INHOUSE)


@pytest.mark.display
def test_upload_dialog_tolerance_field(tk_root, monkeypatch, inhouse_dir, tmp_path):
    """The tolerance entry starts at the core default and accepts only finite positive numbers,
    as a float capped silently at core's limit; a directly opened .nxs greys it out and drops it."""
    errors = []
    monkeypatch.setattr("operaxn.dialog.messagebox.showerror",
                        lambda *args, **kwargs: errors.append(args))
    nxs = str(tmp_path / "direct.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)

    dlg = UploadOptionsDialog(tk_root)
    try:
        assert dlg.tolerance_var.get() == f"{core.ECHEM_TIME_TOLERANCE:g}"
        dlg._paths = [inhouse_dir]
        dlg._update_option_states()
        assert str(dlg.tolerance_entry.cget("state")) == "normal"
        for bad in ("0", "-5", "abc", "nan", "inf"):
            dlg.tolerance_var.set(bad)
            dlg._confirm()
            assert dlg.result is None, bad
        assert len(errors) == 5 and all(a[0] == "Echem time tolerance" for a in errors)

        dlg._paths = [nxs]
        dlg._update_option_states()
        assert str(dlg.tolerance_entry.cget("state")) == "disabled"
        dlg.tolerance_var.set("abc")
        dlg._confirm()
        assert dlg.result is not None and dlg.result.echem_time_tolerance is None
        assert len(errors) == 5, "direct .nxs must not validate the greyed field"
    finally:
        if dlg.winfo_exists():
            dlg.destroy()

    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg._paths = [inhouse_dir]
        dlg.tolerance_var.set(" 120.5 ")
        dlg._confirm()
        assert dlg.result is not None and dlg.result.echem_time_tolerance == 120.5
    finally:
        if dlg.winfo_exists():
            dlg.destroy()

    # Above core's limit the value quietly becomes the limit, with no error box
    from core.config import MAX_ECHEM_TIME_TOLERANCE as limit
    for text in ("1e12", f"{limit:g}"):
        dlg = UploadOptionsDialog(tk_root)
        try:
            dlg._paths = [inhouse_dir]
            dlg.tolerance_var.set(text)
            dlg._confirm()
            result = dlg.result
            assert result is not None and result.echem_time_tolerance == limit, text
        finally:
            if dlg.winfo_exists():
                dlg.destroy()
    assert len(errors) == 5, "the fallback is silent"

    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg._paths = [inhouse_dir]
        dlg.tolerance_var.set("  ")
        dlg._confirm()
        assert dlg.result is not None and dlg.result.echem_time_tolerance is None, \
            "a blank field means the core default"
    finally:
        if dlg.winfo_exists():
            dlg.destroy()

    # Appending standard echem alone ignores the generation metadata, so it is not checked
    std = tmp_path / "std_echem.txt"
    _builders.write_echem_txt(str(std))
    dlg = UploadOptionsDialog(tk_root, has_loaded_data=True)
    try:
        dlg._std_echem = [str(std)]
        dlg.tolerance_var.set("abc")
        dlg.prep_date_var.set("notadate")
        dlg._confirm()
        assert dlg.result is not None and dlg.result.standard_echem_files == [str(std)]
        assert dlg.result.echem_time_tolerance is None and len(errors) == 5
    finally:
        if dlg.winfo_exists():
            dlg.destroy()


def test_display_size_caps_stored_synchrotron_images(tmp_path):
    """The single display-size setting also caps stored synchrotron images."""
    syn_dir = tmp_path / "synchrotron"
    syn_dir.mkdir()
    with h5py.File(str(syn_dir / "sample_123456.nxs"), "w") as f:
        f["/entry1/start_time"] = b"2024-02-05T10:05:00"
        f["/entry1/end_time"] = b"2024-02-05T10:07:00"
    with h5py.File(str(syn_dir / "sample_123456.hdf"), "w") as f:
        f["/entry/data"] = np.arange(256, dtype=np.float64).reshape(16, 16) + 1.0
    _builders.write_echem_txt(str(syn_dir / "echem.txt"))

    process_paths([str(syn_dir)], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.SYNCHROTRON,
                  include_2d_images=True, twod_max_display_size=8)
    m_syn = core.load(get_loaded_nxs_path())
    assert len(m_syn.scans) == 1, f"got {len(m_syn.scans)}"
    assert m_syn.scans[0].twod is not None, "no stored image"
    assert m_syn.scans[0].twod.shape == (8, 8), str(m_syn.scans[0].twod.shape)
    assert m_syn.scans[0].echem == 3.76, f"got {m_syn.scans[0].echem}"


def test_experiment_metadata_rows_bank_geometry(polaris_src):
    """Experiment rows list the flight path, each detector bank and each monitor element.
    A scalar detector field (lab EDF distance) or ragged monitor table yields no element rows."""
    assert experiment_metadata_rows({}) == []
    lab = {"instrument": {"name": "lab", "detector": {"distance": 0.1}}}
    assert [r["field"] for r in experiment_metadata_rows(lab)] == ["instrument"]
    synthetic = {"pre_sample_flightpath": 14.0,
                 "instrument": {"detector": {"detector_number": np.array([1, 2]),
                                             "distance": np.array([2.248, 1.783]),
                                             "polar_angle": np.array([10.4, 25.99])}},
                 "monitor": {"mode": "timer", "detector_number": np.array([671]),
                             "distance": np.array([3.27]),
                             "polar_angle": np.array([0.0])}}
    assert experiment_metadata_rows(synthetic) == [
        {"field": "pre_sample_flightpath", "value": "14.0"},
        {"field": "detector bank 1", "value": "L2 2.248 m, 2theta 10.40 deg"},
        {"field": "detector bank 2", "value": "L2 1.783 m, 2theta 25.99 deg"},
        {"field": "monitor mode", "value": "timer"},
        {"field": "monitor 671", "value": "3.270 m, 2theta 0.00 deg"}]
    # An in-house monitor (mode and preset, no element arrays) and a monitor
    # table whose arrays disagree in length yield no element rows
    inhouse = {"monitor": {"mode": "timer", "preset": 120.0}}
    assert experiment_metadata_rows(inhouse) == [
        {"field": "monitor mode", "value": "timer"},
        {"field": "monitor preset", "value": "120 s"}]
    ragged = {"monitor": {"mode": "timer", "detector_number": np.array([611, 612]),
                          "distance": np.array([7.641]),
                          "polar_angle": np.array([180.0, 180.0])}}
    assert experiment_metadata_rows(ragged) == [{"field": "monitor mode", "value": "timer"}]

    process_paths([polaris_src], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    fields = {r["field"]: r["value"]
              for r in experiment_metadata_rows(get_experiment_metadata())}
    assert fields["instrument"] == "POLARIS", str(fields)
    assert fields["pre_sample_flightpath"] == "14.0", str(fields)
    assert fields["detector bank 5"] == "L2 1.251 m, 2theta 146.72 deg", str(fields)
    assert fields["monitor mode"] == "timer", str(fields)
    assert fields["monitor 611"] == "7.641 m, 2theta 180.00 deg", str(fields)


def test_cli_profile_option(tmp_path, capsys):
    """operaxn --profile prints a Mantid file's INSTRUMENT_PROFILES block, exits 1 if unreadable."""
    path = str(tmp_path / "TESTINST1-2.nxs")
    _builders.write_mantid_processed_file(path)
    assert cli_main(["--profile", path]) == 0
    out = capsys.readouterr().out
    assert '"testinst": {' in out and '"pre_sample_flightpath_m": 14.0' in out
    assert '"monitor_mode": None' in out and '"monitors": {' in out
    assert cli_main(["--profile", str(tmp_path / "missing.nxs")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_prints_the_version_and_has_no_dependency_check(monkeypatch, capsys):
    """The CLI prints its version, names itself operaxn however launched and has no --check-deps."""
    from operaxn.config import APP_VERSION
    with pytest.raises(SystemExit) as info:
        cli_main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"OperaXN {APP_VERSION}"
    monkeypatch.setattr("sys.argv", ["main.py"])
    with pytest.raises(SystemExit) as info:
        cli_main(["--help"])
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith("usage: operaxn [-h]"), "argv[0] must not name it"
    with pytest.raises(SystemExit) as info:
        cli_main(["--check-deps"])
    assert info.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


def test_excel_sheet_widths_survive_nan_and_empty_frames():
    """Column fitting handles NaN (pandas 3 keeps it through astype(str)) and a header-only sheet."""
    from openpyxl import Workbook
    frames = (pd.DataFrame({"Voltage (V)": pd.Series([3.71, np.nan], dtype=object),
                            "note": ["a", None]}),
              pd.DataFrame(columns=["scan number", "Has Echem"]))
    for df in frames:
        worksheet = Workbook().active
        worksheet.append(list(df.columns))
        OPERAXN._format_sheet(worksheet, df)
        widths = [worksheet.column_dimensions[c].width for c in ("A", "B")]
        assert widths == [len(col) + 3 for col in df.columns], widths
        assert worksheet["A1"].font.bold and worksheet.freeze_panes == "A2"


def test_neutron_statistics_sheet_reads_echem_columns(tmp_path):
    """Neutron Statistics rows carry the correlated voltage and current (mA); uncorrelated say No."""
    def scan(num, volts, amps):
        return {"scan_num": num, "timestamp": "2026-06-01 10:00:00", "echem": volts,
                "current": amps, "echem_timestamp": "", "exposure_time": 120.0,
                "neutron_start": "2026-06-01 10:00:00", "neutron_end": "2026-06-01 10:02:00",
                "neutron_files": {"1": {"tof": f"/data/run{num}_b1_tof.dat"}}, "neutron_meta": None}
    fake = SimpleNamespace(state=SimpleNamespace(
        scans=[scan(1, 3.71, 0.25), scan(2, None, None)],
        data_source=core.DataSourceType.NEUTRON, time_method="absolute"))
    summary = pd.DataFrame([OPERAXN._create_scan_summary(fake, s) for s in fake.state.scans])
    rows = [OPERAXN._calculate_neutron_statistics(fake, r) for _, r in summary.iterrows()]
    assert rows[0]["Voltage (V)"] == 3.71 and rows[0]["Current (mA)"] == 0.25
    assert rows[0]["Has Echem"] == "Yes" and rows[0]["Scan Duration"] == "120.0 s"
    assert rows[1]["Voltage (V)"] == "" and rows[1]["Current (mA)"] == ""
    assert rows[1]["Has Echem"] == "No"
    assert "Current (A)" not in rows[0], "values are mA"

    # The sheet gets the shared formatting: bold frozen header, capped widths
    from openpyxl import load_workbook
    fake._calculate_neutron_statistics = (
        lambda row: OPERAXN._calculate_neutron_statistics(fake, row))
    fake._format_sheet = OPERAXN._format_sheet
    book = tmp_path / "stats.xlsx"
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        OPERAXN._add_neutron_statistics_sheet(fake, writer, summary)
    sheet = load_workbook(book)["Neutron Statistics"]
    assert sheet["A1"].font.bold and sheet.freeze_panes == "A2"
    assert all(dim.width <= 50 for dim in sheet.column_dimensions.values())


def test_excel_export_asks_for_the_file_before_its_thread(monkeypatch):
    """The save dialog runs on the calling (Tk) thread; cancelling it starts no export thread."""
    import threading
    from tkinter import filedialog
    import operaxn.gui as gui_module
    asked, started = [], []
    monkeypatch.setattr(filedialog, "asksaveasfilename",
                        lambda **_k: asked.append(threading.current_thread()) or "")
    monkeypatch.setattr(gui_module.threading, "Thread",
                        lambda *a, **k: started.append(k) or SimpleNamespace(start=lambda: None))
    fake = SimpleNamespace(master=None, state=SimpleNamespace(
        scans=[{"scan_num": 1}], data_source=core.DataSourceType.INHOUSE),
        _warn_if_busy=lambda action: False)
    OPERAXN._export_to_excel(fake)
    assert asked == [threading.current_thread()] and started == []


def _excel_fake(monkeypatch, path, messages):
    """Return an OPERAXN stand-in whose export thread runs inline."""
    from tkinter import filedialog
    import operaxn.gui as gui_module
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: path)

    class InlineThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(gui_module.threading, "Thread", InlineThread)
    fake = SimpleNamespace(
        master=None, performance=PerformanceMonitor(),
        state=SimpleNamespace(scans=[{"scan_num": 1}], echem_df=None,
                              data_source=core.DataSourceType.INHOUSE),
        _warn_if_busy=lambda action: False, _set_busy=lambda busy: None,
        after=lambda ms, fn: fn(), _finish_busy=lambda outcome: outcome and outcome(),
        _show_message=lambda *a: messages.append(a),
        _create_summary_dataframe=lambda: pd.DataFrame({"scan number": [1],
                                                        "voltage (V)": [3.7]}))
    fake._save_to_csv = lambda df, name: OPERAXN._save_to_csv(fake, df, name)
    fake._save_to_excel = lambda df, name: OPERAXN._save_to_excel(fake, df, name)
    fake._format_sheet = OPERAXN._format_sheet
    fake._add_operando_echem_sheet = lambda writer: None
    fake._add_experiment_sheet = lambda writer: None
    return fake


def test_excel_export_writes_a_workbook(tmp_path, monkeypatch):
    """An Excel export writes a workbook with its Scan Summary sheet and leaves no .part."""
    from openpyxl import load_workbook
    messages = []
    path = str(tmp_path / "summary.xlsx")
    OPERAXN._export_to_excel(_excel_fake(monkeypatch, path, messages))
    assert messages[-1][0] == "Export complete", messages
    assert load_workbook(path).sheetnames == ["Scan Summary"]
    assert not os.path.exists(path + ".part")


@pytest.mark.parametrize("filename, kind", [("summary.xlsx", "Excel"), ("summary.csv", "CSV")])
def test_failed_excel_export_keeps_previous_file(tmp_path, monkeypatch, filename, kind):
    """A save that fails part way names the format written and leaves the earlier file untouched."""
    messages = []
    path = tmp_path / filename
    path.write_bytes(b"previous export")
    fake = _excel_fake(monkeypatch, str(path), messages)

    def half_save(df, name):
        with open(name, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    fake._save_to_excel = fake._save_to_csv = half_save
    OPERAXN._export_to_excel(fake)
    assert messages[-1] == ("Export failed", f"{kind} file could not be exported: disk full",
                            "error"), messages
    assert path.read_bytes() == b"previous export"
    assert not os.path.exists(str(path) + ".part")


@pytest.mark.parametrize("export_type, what", [("current", "Image"), ("multiple", "Images")])
def test_failed_image_export_says_what_could_not_be_exported(export_type, what):
    """An image or range export that raises closes its page and names what was not exported."""
    from contextlib import nullcontext
    from operaxn.dialog import ExportOptions
    shown, closed = [], []

    def boom(*_a):
        raise OSError("disk full")

    fake = SimpleNamespace(_exporting_on_tk_thread=nullcontext, _export_current_scan=boom,
                           _export_multiple_scans=boom, _close_dialog=closed.append,
                           _show_message=lambda *a: shown.append(a))
    OPERAXN._perform_export(fake, ExportOptions(export_type, 100, {"oned": True}, [1]), "page")
    assert closed == ["page"]
    assert shown == [("Export failed", f"{what} could not be exported: disk full", "error")]


@pytest.mark.parametrize("result, text", [
    pytest.param((False, ["No experiment loaded"]),
                 "NeXus file could not be exported:\nNo experiment loaded", id="refused"),
    pytest.param(OSError("disk full"), "NeXus file could not be exported: disk full", id="raises")])
def test_failed_nexus_export_says_what_could_not_be_exported(tmp_path, monkeypatch, result, text):
    """A NeXus export that is refused or raises says the NeXus file could not be exported."""
    messages = []

    def export(_destination):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("operaxn.gui.export_nxs", export)
    OPERAXN._export_nxs(_excel_fake(monkeypatch, str(tmp_path / "cell.nxs"), messages))
    assert messages == [("Export failed", text, "error")], messages


def test_upper_case_csv_name_gets_csv(tmp_path, monkeypatch):
    """summary.CSV is written as CSV text, not a workbook under a .CSV name."""
    messages = []
    path = str(tmp_path / "summary.CSV")
    OPERAXN._export_to_excel(_excel_fake(monkeypatch, path, messages))
    assert messages[-1][0] == "Export complete" and "CSV file" in messages[-1][1]
    with open(path, "rb") as f:
        assert f.read(11) == b"scan number"


def test_config_summary_logs_after_setup(caplog):
    """The debug configuration summary logs its lines when called after logging setup."""
    from operaxn.config import log_config_summary
    with caplog.at_level(logging.DEBUG, logger="operaxn.config"):
        log_config_summary()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "OperaXN Configuration" in text and "Max Workers:" in text


def test_splash_fill_stops_at_track_end():
    """The splash progress fill is drawn no further than 100 % of its track."""
    from operaxn.main import EnhancedSplashScreen

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    drawn = []

    class Canvas:
        def coords(self, item, *xy):
            if xy:
                drawn.append(xy)
            return [50, 10, 60, 20]

        def itemconfig(self, *_args, **_kwargs):
            pass

    fake = SimpleNamespace(
        progress_var=Var(99), canvas=Canvas(), _schedule_animation=lambda *_args: None,
        _animate_progress=None,  # rescheduling target; the stub never calls it
        progress_target=100,
        ui_elements={"bar_width": 300, "bar_center_x": 200,
                     "progress_rect": 1, "progress_glow": 2})
    EnhancedSplashScreen._animate_progress(fake)
    assert fake.progress_var.get() == 100
    assert drawn[0][2] == 350, "the fill must end at the track's right edge (50 + 300)"


@pytest.mark.display
def test_startup_failure_closes_splash_and_exits(monkeypatch):
    """A main-window build error is shown once, the splash goes and run() returns 1."""
    import operaxn.main as main_module

    def broken(*_a, **_k):
        raise RuntimeError("broken build")

    monkeypatch.setattr(main_module, "OPERAXN", broken)
    monkeypatch.setattr(main_module, "SPLASH_DURATION_MS", 400)
    monkeypatch.setattr(main_module, "setup_logging",
                        lambda debug=False: logging.getLogger("operaxn.main"))
    errors = []
    monkeypatch.setattr(main_module.ApplicationManager, "_show_error",
                        lambda self, message: errors.append(message))
    original_ui = main_module.ApplicationManager._initialise_ui

    def ui_with_watchdog(self):
        original_ui(self)
        self.root.after(8000, self.root.destroy)  # ends a main loop the failure left running

    monkeypatch.setattr(main_module.ApplicationManager, "_initialise_ui", ui_with_watchdog)
    manager = main_module.ApplicationManager(argparse.Namespace(debug=False))
    assert manager.run() == 1
    assert len(errors) == 1 and "broken build" in errors[0]


@pytest.mark.display
def test_plot_settings_apply_redraws_once(tk_root):
    """Applying plot settings redraws through the update callback exactly once."""
    from operaxn.dialog import PlotSettingsDialog
    from operaxn.gui import VisualiserConfig
    calls = []
    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(), lambda: calls.append(1))
    try:
        dlg._apply_settings()
    finally:
        dlg.destroy()
    assert calls == [1]


@pytest.mark.display
def test_plot_settings_axis_choices_stay_plain_text(tk_root):
    """Plot Settings keeps plain Tk text for TOF, d-spacing and 2θ (figure labels use mathtext)."""
    import tkinter as tk
    from operaxn.dialog import PlotSettingsDialog
    from operaxn.gui import VisualiserConfig

    def texts(widget, kinds=(tk.Label, tk.Checkbutton, tk.Radiobutton)):
        own = [w.cget("text") for w in widget.winfo_children() if isinstance(w, kinds)]
        return own + [t for w in widget.winfo_children() for t in texts(w, kinds)]

    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(data_source=DataSourceType.NEUTRON), None)
    try:
        assert texts(dlg, tk.Radiobutton) == ["Time of flight (μs)", "d-spacing (Å)"]
        assert not [t for t in texts(dlg) if "$" in t]
    finally:
        dlg.destroy()
    dlg = PlotSettingsDialog(tk_root, VisualiserConfig(), None)
    try:
        shown = texts(dlg)
        assert {"X-axis range (2θ [°]):", "Show d-spacing instead of 2θ"} <= set(shown)
        assert not [t for t in shown if "$" in t]
    finally:
        dlg.destroy()


def test_direct_nxs_opens_as_its_recorded_source(neutron_dir, tmp_path):
    """A directly opened .nxs is read and displayed as the source it records,
    whatever was selected; an unreadable path gives None."""
    process_paths([neutron_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    neutron_nxs = str(tmp_path / "neutron_run.nxs")
    shutil.copy(get_loaded_nxs_path(), neutron_nxs)
    assert core.read_data_source(neutron_nxs) == "neutron"
    assert core.read_data_source(str(tmp_path / "missing.nxs")) is None

    scans, _, _ = process_paths([neutron_nxs], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    assert get_loaded_data_source() == DataSourceType.NEUTRON
    assert make_neutron_arrays(scans), "neutron banks must load from a directly opened file"


def test_direct_nxs_uses_its_recorded_time_method(inhouse_dir, tmp_path):
    """A relative-recorded .nxs opened with Absolute selected shows relative time, so every
    scan marker sits on the voltage stored for that scan."""
    nxs = str(tmp_path / "rel.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE,
                             time_method=core.TimeMethod.RELATIVE)
    assert ok, msgs
    assert core.read_correlation_method(nxs) == "relative"
    scans, echem_df, method = process_paths(
        [nxs], time_method=TimeMethod.ABSOLUTE, data_source=DataSourceType.INHOUSE)
    assert method == "relative"
    arrays = make_echem_arrays(echem_df, method)
    for scan, t in zip(scans, get_scan_time_positions(scans, echem_df, method)):
        assert abs(np.interp(t, arrays["x"], arrays["y"]) - scan["echem"]) < 0.011


def test_working_copy_keeps_the_export_stem(inhouse_dir, tmp_path):
    """Adding echem to an opened .nxs edits <stem>_working.nxs, but exports keep <stem>."""
    from operaxn.gui import _experiment_stem
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    user_nxs = str(tmp_path / "MyCell_run3.nxs")
    export_nxs(user_nxs)
    process_paths([user_nxs], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    assert get_experiment_stem() == "MyCell_run3"

    std_echem_path = str(tmp_path / "standard_cycling.txt")
    _std_echem_file(std_echem_path)
    assert add_standard_echem_files([std_echem_path])
    assert os.path.basename(get_loaded_nxs_path()) == "MyCell_run3_working.nxs"
    assert get_experiment_stem() == "MyCell_run3" and _experiment_stem() == "MyCell_run3"
    _discard(core.default_cache_path([inhouse_dir]))


def test_reset_session_forgets_the_experiment(inhouse_dir, tmp_path):
    """reset_session drops the unsaved cache file and every session field, so an echem
    append afterwards has no experiment to target."""
    std_echem_path = str(tmp_path / "standard_cycling.txt")
    _std_echem_file(std_echem_path)
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE,
                  standard_echem_files=[std_echem_path])
    cache_file = get_loaded_nxs_path()
    assert get_standard_echem()
    reset_session()
    assert not os.path.exists(cache_file)
    assert get_loaded_nxs_path() is None and get_experiment_stem() is None
    assert get_standard_echem() == [] and get_experiment_metadata() == {}
    assert get_loaded_data_source() is None
    with pytest.raises(RuntimeError):
        add_standard_echem_files([std_echem_path])


@needs_fabio
def test_reset_session_empties_the_reader_cache(inhouse_dir):
    """Clear All (reset_session) drops the forgotten experiment's images from the cache."""
    _builders.write_edf_image(os.path.join(inhouse_dir, "image_001.edf"),
                              _builders.SCAN_TIMES[0])
    scans, _, _ = process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    assert make_twod_arrays(scans)[1]["image"] is not None
    assert readers._file_cache.cache, "the shown image goes through the cache"
    reset_session()
    assert not readers._file_cache.cache and readers._file_cache.current_size == 0


@needs_fabio
def test_twod_arrays_read_each_image_only_when_shown(inhouse_dir, monkeypatch):
    """Building, sizing and membership read nothing; indexing reads that one scan."""
    _builders.write_edf_image(os.path.join(inhouse_dir, "image_001.edf"),
                              _builders.SCAN_TIMES[0])
    scans, _, _ = process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    assert scans[0]["twod"]
    reads = []
    real = gui_input._load_twod_image
    monkeypatch.setattr(gui_input, "_load_twod_image",
                        lambda sd, state=None: reads.append(sd.scan_num) or real(sd, state))
    arrays = make_twod_arrays(scans)
    assert bool(arrays) and len(arrays) == 1 and 1 in arrays and 2 not in arrays
    assert list(arrays) == [1] and arrays.get(2) is None
    assert reads == [], f"images read before any scan was shown: {reads}"
    assert arrays[1]["image"].shape == (8, 8)
    assert reads == [1]


@needs_fabio
def test_unreadable_twod_image_maps_to_error_once(inhouse_dir, monkeypatch):
    """A 2D source that fails to read shows the error entry and is not re-read on every visit."""
    edf = os.path.join(inhouse_dir, "image_001.edf")
    _builders.write_edf_image(edf, _builders.SCAN_TIMES[0])
    scans, _, _ = process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    with open(edf, "wb") as f:
        f.write(b"not an image")
    reads = []
    real = gui_input._load_twod_image
    monkeypatch.setattr(gui_input, "_load_twod_image",
                        lambda sd, state=None: reads.append(sd.scan_num) or real(sd, state))
    arrays = make_twod_arrays(scans)
    assert arrays[1] == {"error": True}
    assert arrays.get(1) == {"error": True} and reads == [1]


@needs_fabio
def test_twod_scan_shown_while_its_source_was_away_shows_the_image_once_it_is_back(tmp_path):
    """Moving the image folder away turns the panel red; moving it back heals viewed scans."""
    root = tmp_path / "set"
    src, images = str(root / "xrd"), str(root / "images")
    _builders.write_lab_dir(src)
    os.makedirs(images)
    for i, ts in enumerate(_builders.SCAN_TIMES, start=1):
        _builders.write_edf_image(os.path.join(images, f"image_{i:03d}.edf"), ts)
    scans, _, _ = process_paths([str(root)], time_method=TimeMethod.ABSOLUTE,
                                data_source=DataSourceType.INHOUSE)
    arrays = make_twod_arrays(scans)
    assert "image" in arrays[1]
    os.rename(images, images + "_away")
    assert arrays[1] == {"error": True} and arrays[2] == {"error": True}
    os.rename(images + "_away", images)
    assert "image" in arrays[1] and "image" in arrays[2] and "image" in arrays[3]


def _gui_app(tk_root, monkeypatch):
    """Return (window, app, messages), message boxes recorded. Workers run on the Tk thread:
    after() from another thread needs a running mainloop, and tests pump with update()."""
    from tkinter import Toplevel
    import operaxn.gui as gui_module
    win = Toplevel(tk_root)

    class DeferredThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            win.after(0, self._target)

    monkeypatch.setattr(gui_module, "threading", SimpleNamespace(Thread=DeferredThread))
    app = OPERAXN(master=win)
    messages = []
    monkeypatch.setattr(app, "_show_message", lambda *args: messages.append(args))
    return win, app, messages


def _wait_for_worker(win, app, timeout=120.0):
    """Pump Tk events until the app's worker task (load, append or export) has finished."""
    import time
    deadline = time.monotonic() + timeout
    while app._is_busy():
        assert time.monotonic() < deadline, "worker task did not finish"
        win.update()
        time.sleep(0.01)
    win.update()


@pytest.mark.display
def test_upload_and_clear_locked_while_loading(tk_root, monkeypatch, neutron_dir, tmp_path):
    """Upload and Clear are disabled and refused while a load runs, and come back after;
    a neutron .nxs opened under the default Laboratory source still loads as neutron."""
    process_paths([neutron_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    neutron_nxs = str(tmp_path / "neutron_run.nxs")
    shutil.copy(get_loaded_nxs_path(), neutron_nxs)

    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        opened = []
        import operaxn.gui as gui_module
        monkeypatch.setattr(gui_module, "UploadOptionsDialog",
                            lambda *_a, **_k: opened.append(1))
        options = UploadOptions(paths=[neutron_nxs], data_source=DataSourceType.INHOUSE,
                                time_method=TimeMethod.ABSOLUTE, display_size=0,
                                include_2d=False)
        app._process_files_async(options)
        buttons = app.button_panel.buttons
        assert app._is_busy()
        assert str(buttons["upload_files"].cget("state")) == "disabled"
        assert str(buttons["clear_all"].cget("state")) == "disabled"
        app._upload_files()
        assert opened == [], "Upload must not open a dialog mid-load"
        assert app._clear_all() is False, "Clear must be refused mid-load"

        _wait_for_worker(win, app)
        assert str(buttons["upload_files"].cget("state")) == "normal"
        assert str(buttons["clear_all"].cget("state")) == "normal"
        assert app.state.data_source == DataSourceType.NEUTRON
        assert app.state.neutron_arrays, messages
        assert messages and messages[-1][0] == "Success", messages
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_one_run_neutron_load_message_reads_singular(tk_root, monkeypatch, neutron_dir):
    """A one-run neutron load says 'Successfully loaded 1 neutron scan'."""
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        app._process_files_async(UploadOptions(paths=[neutron_dir],
                                               data_source=DataSourceType.NEUTRON,
                                               time_method=TimeMethod.ABSOLUTE, display_size=0,
                                               include_2d=False))
        _wait_for_worker(win, app)
        assert messages[-1] == ("Success", "Successfully loaded 1 neutron scan using absolute "
                                "timestamps", "info"), messages
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_replot_closes_old_figure_and_rebinds_plot_settings(tk_root, monkeypatch, inhouse_dir):
    """Each Plot frees the figure it replaces, an open Plot Settings dialog reopens on the
    new config, and Clear frees the figure and forgets the session."""
    import matplotlib.pyplot as plt
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        options = UploadOptions(paths=[inhouse_dir], data_source=DataSourceType.INHOUSE,
                                time_method=TimeMethod.ABSOLUTE, display_size=0,
                                include_2d=False)
        app._process_files_async(options)
        _wait_for_worker(win, app)
        assert app.state.scans, messages

        def open_figures():
            # Compare objects: pyplot hands a closed figure's number to the next one
            return [manager.canvas.figure
                    for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]

        before = len(open_figures())
        app._plot_data()
        first = app.fig
        app._show_plot_settings()
        old_dialog = app._plot_settings_dialog
        app._plot_data()
        assert app.fig is not first, messages
        assert all(fig is not first for fig in open_figures()), "replaced figure not closed"
        assert len(open_figures()) == before + 1
        new_dialog = app._plot_settings_dialog
        assert new_dialog is not None and new_dialog is not old_dialog
        assert not old_dialog.winfo_exists()
        assert new_dialog.config is app.config, "Plot Settings must edit the live config"

        monkeypatch.setattr("operaxn.gui.messagebox.askyesno", lambda *_a, **_k: True)
        last = app.fig
        assert app._clear_all()
        assert all(fig is not last for fig in open_figures()), "Clear must close the figure"
        assert get_loaded_nxs_path() is None and get_standard_echem() == []
    finally:
        if win.winfo_exists():
            win.destroy()


def _load_inhouse(win, app, inhouse_dir):
    """Run a raw in-house load through the app's worker and wait for it."""
    app._process_files_async(UploadOptions(paths=[inhouse_dir],
                                           data_source=DataSourceType.INHOUSE,
                                           time_method=TimeMethod.ABSOLUTE,
                                           display_size=0, include_2d=False))
    _wait_for_worker(win, app)
    assert app.state.scans


@pytest.mark.display
def test_failed_load_unlocks_upload_and_clear(tk_root, monkeypatch, inhouse_dir):
    """A load whose processing raises shows the error and leaves Upload and Clear usable."""
    import operaxn.gui as gui_module
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        def boom(*_a, **_k):
            raise RuntimeError("unreadable input")

        monkeypatch.setattr(gui_module, "process_paths", boom)
        app._process_files_async(UploadOptions(paths=[inhouse_dir],
                                               data_source=DataSourceType.INHOUSE,
                                               time_method=TimeMethod.ABSOLUTE,
                                               display_size=0, include_2d=False))
        _wait_for_worker(win, app)
        assert messages and messages[-1][0] == "Error" and "unreadable input" in messages[-1][1]
        buttons = app.button_panel.buttons
        assert str(buttons["upload_files"].cget("state")) == "normal"
        assert str(buttons["clear_all"].cget("state")) == "normal"
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
@pytest.mark.parametrize("failure", ["generate", "arrays", "rows shown"])
def test_failed_load_leaves_the_window_as_clear_all_does(tk_root, monkeypatch, tmp_path,
                                                         neutron_dir, failure):
    """A failed neutron load leaves no scans, rows or view settings behind: the window and its
    settings are as Clear All leaves them."""
    import operaxn.gui as gui_module
    from operaxn.gui import VisualiserConfig
    path = neutron_dir
    if failure == "generate":
        path = tmp_path / "junk"
        path.mkdir()
        (path / "notes.txt").write_text("nothing to see\n")
    win, app, messages = _gui_app(tk_root, monkeypatch)
    if failure == "arrays":
        def boom(*_a, **_k):
            raise MemoryError("simulated neutron build failure")
        monkeypatch.setattr(gui_module, "make_neutron_arrays", boom)
    elif failure == "rows shown":
        def boom(*_a, **_k):
            raise RuntimeError("simulated failure after the rows are shown")
        monkeypatch.setattr(app.scan_selector, "set_range", boom)  # first step after the rows
    try:
        app._apply_data_source(DataSourceType.NEUTRON, 0)  # as _upload_files does first
        app._process_files_async(UploadOptions(
            paths=[str(path)], data_source=DataSourceType.NEUTRON,
            time_method=TimeMethod.ABSOLUTE, display_size=0, include_2d=False))
        _wait_for_worker(win, app)
        assert messages[-1][0] == "Error", messages
        assert app.state.scans == []
        assert app.config == VisualiserConfig()
        assert app.file_list.listbox.size() == 0, app.file_list.listbox.get(0, "end")
    finally:
        if win.winfo_exists():
            win.destroy()


def _inhouse_options(path):
    """Return the options of an absolute in-house load of path without 2D images."""
    return UploadOptions(paths=[path], data_source=DataSourceType.INHOUSE,
                         time_method=TimeMethod.ABSOLUTE, display_size=0,
                         include_2d=False)


@pytest.mark.display
def test_failed_generate_shows_the_reason_and_clears_progress(tk_root, monkeypatch, tmp_path):
    """The box names the generate error and the progress row does not stay behind."""
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "notes.txt").write_text("nothing to see\n")
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        app.file_list.show_progress("Processing laboratory data...")
        app._process_files_async(_inhouse_options(str(junk)))
        _wait_for_worker(win, app)
        assert messages[-1][0] == "Error" and "No scan data available" in messages[-1][1]
        assert app.file_list.listbox.size() == 0
        assert app.state.scans == []
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_array_failure_drops_the_half_load(tk_root, monkeypatch, inhouse_dir):
    """An error while building arrays is shown, and the scans, session and progress row go."""
    import operaxn.gui as gui_module

    def boom(*_a, **_k):
        raise MemoryError("simulated 2D build failure")

    monkeypatch.setattr(gui_module, "make_twod_arrays", boom)
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        app._process_files_async(_inhouse_options(inhouse_dir))
        _wait_for_worker(win, app)
        assert messages and messages[-1][0] == "Error", messages
        assert "simulated 2D build failure" in messages[-1][1]
        assert app.state.scans == [] and app.file_list.listbox.size() == 0
        assert gui_input.get_loaded_nxs_path() is None
        assert not os.path.exists(core.default_cache_path([inhouse_dir]))
        buttons = app.button_panel.buttons
        assert str(buttons["upload_files"].cget("state")) == "normal"
        assert str(buttons["plot_data"].cget("state")) == "disabled"
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_load_without_plottable_data_is_dropped(tk_root, monkeypatch, inhouse_dir):
    """Scans with no readable pattern give the warning and no orphan scans."""
    import operaxn.gui as gui_module
    monkeypatch.setattr(gui_module, "make_oned_arrays", lambda *_a, **_k: {})
    monkeypatch.setattr(gui_module, "make_twod_arrays", lambda *_a, **_k: {})
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        app._process_files_async(_inhouse_options(inhouse_dir))
        _wait_for_worker(win, app)
        assert messages[-1][0] == "Warning" and "No valid data" in messages[-1][1]
        assert app.state.scans == [] and app.file_list.listbox.size() == 0
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_plot_keeps_or_ends_a_worker_busy_state_correctly(tk_root, monkeypatch, inhouse_dir):
    """Plot during an echem append keeps the append's lock; if the append finishes inside
    Plot's error box, the lock ends rather than coming back when the box closes."""
    from operaxn.gui import UIState
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        _load_inhouse(win, app, inhouse_dir)

        app._set_busy(UIState.LOADING)  # an append is running
        app._plot_data()
        assert app._is_busy(), "a successful Plot must not end the append's lock"

        # The append's completion runs while Plot's error box pumps events
        def failing_figure(*_a, **_k):
            raise RuntimeError("draw failed")

        def message_box(*args):
            messages.append(args)
            app._finish_busy(None)

        monkeypatch.setattr(app, "_create_figure", failing_figure)
        monkeypatch.setattr(app, "_show_message", message_box)
        app._plot_data()
        assert messages[-1][0] == "Error"
        assert not app._is_busy(), "the finished append must stay finished"
        assert str(app.button_panel.buttons["clear_all"].cget("state")) == "normal"
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_echem_append_while_busy_is_refused_with_a_message(tk_root, monkeypatch, tmp_path):
    """Echem picked while an export started (e.g. during ICI's file picker) is not dropped
    silently: the user is told to add it again."""
    from operaxn.gui import UIState
    import operaxn.gui as gui_module
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        started = []
        monkeypatch.setattr(gui_module, "add_standard_echem_files", started.append)
        app._set_busy(UIState.EXPORTING)
        app._append_standard_echem_async([str(tmp_path / "std.txt")])
        win.update()
        assert started == [] and messages[-1][0] == "Busy", messages
        assert "add electrochemistry" in messages[-1][1]
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
@pytest.mark.parametrize("count, title, body", [
    (0, "No electrochemistry data",
     "No valid electrochemistry data was found in the selected files."),
    (1, "Electrochemistry added without current",
     "The selected file was loaded, but it has no usable current column and therefore "
     "cannot be used for ICI or Capacity analysis."),
    (2, "Electrochemistry added without current",
     "The selected files were loaded, but they have no usable current column and therefore "
     "cannot be used for ICI or Capacity analysis."),
], ids=["none", "one", "two"])
def test_added_echem_outcomes_say_electrochemistry(tk_root, monkeypatch, count, title, body):
    """Add Electrochemistry outcomes: one title word, the no-current text by the file count."""
    import operaxn.gui as gui_module
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        voltage_only = pd.DataFrame({
            "timestamp": pd.date_range("2024-02-05 10:00", periods=3, freq="min"),
            "echem_data": [3.7, 3.8, 3.9]})
        monkeypatch.setattr(gui_module, "add_standard_echem_files",
                            lambda paths: [{"source_file": p, "data": voltage_only}
                                           for p in paths[:count]])
        app._append_standard_echem_async(["v1.txt", "v2.txt"])
        _wait_for_worker(win, app)
        assert messages[-1][:2] == (title, body), messages
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_echem_only_upload_needs_a_loaded_experiment(tk_root, monkeypatch, tmp_path):
    """Standard echem alone (dialog opened with data that a Clear then removed) is refused."""
    import operaxn.gui as gui_module
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        options = UploadOptions(paths=[], data_source=DataSourceType.INHOUSE,
                                time_method=TimeMethod.ABSOLUTE, display_size=0,
                                include_2d=False,
                                standard_echem_files=[str(tmp_path / "std.txt")])
        monkeypatch.setattr(gui_module, "UploadOptionsDialog",
                            lambda *_a, **_k: SimpleNamespace(get_result=lambda: options))
        appended = []
        monkeypatch.setattr(app, "_append_standard_echem_async", appended.append)
        app._upload_files()
        assert appended == [] and messages[-1][0] == "No experiment loaded", messages
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_exports_lock_upload_and_clear(tk_root, monkeypatch, inhouse_dir, tmp_path):
    """NeXus and Excel exports hold the busy lock, so Clear cannot delete the file being
    copied or the scans being summarised."""
    from operaxn.gui import UIState
    win, app, messages = _gui_app(tk_root, monkeypatch)
    try:
        _load_inhouse(win, app, inhouse_dir)
        cache_file = get_loaded_nxs_path()
        target = {"path": str(tmp_path / "exported.nxs")}
        monkeypatch.setattr("operaxn.gui.filedialog.asksaveasfilename",
                            lambda *_a, **_k: target["path"])
        monkeypatch.setattr("operaxn.gui.messagebox.askyesno", lambda *_a, **_k: True)

        app._export_nxs()
        assert app.state.ui_state == UIState.EXPORTING
        assert app._clear_all() is False and os.path.isfile(cache_file)
        app._export_nxs()
        assert messages[-1][0] == "Busy", "a second export must wait"
        _wait_for_worker(win, app)
        assert messages[-1][0] == "Export complete", messages
        assert os.path.isfile(target["path"])

        target["path"] = str(tmp_path / "summary.csv")
        app._export_to_excel()
        assert app.state.ui_state == UIState.EXPORTING
        assert app._clear_all() is False and app.state.scans
        _wait_for_worker(win, app)
        assert messages[-1][0] == "Export complete", messages
        assert os.path.isfile(target["path"])

        assert str(app.button_panel.buttons["clear_all"].cget("state")) == "normal"
        assert app._clear_all() and os.path.isfile(str(tmp_path / "exported.nxs"))
        _discard(cache_file)
    finally:
        if win.winfo_exists():
            win.destroy()


def _pump_until_closed(win, timeout=10.0):
    """Pump Tk events until win is destroyed or timeout seconds pass."""
    import time
    deadline = time.monotonic() + timeout
    while win.winfo_exists() and time.monotonic() < deadline:
        win.update()
        time.sleep(0.01)


@pytest.mark.display
def test_quit_while_busy_asks_first(tk_root, monkeypatch):
    """Closing during an export asks; No keeps the window, Yes hides it until the export ends."""
    from operaxn.gui import UIState
    win, app, _ = _gui_app(tk_root, monkeypatch)
    answers, asked, hidden = [False, True], [], []
    monkeypatch.setattr("operaxn.gui.messagebox.askyesno",
                        lambda *a, **k: asked.append(a) or answers.pop(0))
    monkeypatch.setattr(win, "withdraw", lambda: hidden.append(True))
    try:
        app._set_busy(UIState.EXPORTING)
        app._on_closing()
        assert asked and win.winfo_exists() and not hidden, "No must keep the window and its export"
        app._on_closing()
        assert hidden and win.winfo_exists(), "Yes hides the window while the export runs"
        app._finish_busy(None)  # the export's worker ends
        _pump_until_closed(win)
        assert not win.winfo_exists()
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_quit_closes_at_the_deadline_when_the_work_never_ends(tk_root, monkeypatch):
    """Work that never reaches a checkpoint keeps the app open for QUIT_WAIT_S at most."""
    from operaxn.gui import UIState
    monkeypatch.setattr("operaxn.gui.QUIT_WAIT_S", 0.2)
    win, app, _ = _gui_app(tk_root, monkeypatch)
    monkeypatch.setattr("operaxn.gui.messagebox.askyesno", lambda *a, **k: True)
    try:
        app._set_busy(UIState.LOADING)
        app._on_closing()
        _pump_until_closed(win)
        assert not win.winfo_exists() and app._is_busy()
    finally:
        if win.winfo_exists():
            win.destroy()


@needs_fabio
@pytest.mark.display
@pytest.mark.parametrize("stage", ["warm_up", "write"])
def test_quit_mid_load_closes_once_the_load_has_cleaned_up(tk_root, monkeypatch, tmp_path, stage):
    """Quit anyway mid ZIP load closes the window only after the stopped load has cleaned up."""
    import gc
    import importlib
    import tempfile
    import threading
    import time
    from tkinter import Toplevel
    generate_module = importlib.import_module("core.generate")
    src = tmp_path / "src"
    _builders.write_dated_pairs(str(src), _builders.SCAN_TIMES, _builders.SCAN_TIMES)
    zpath = _builders.zip_tree(str(src), str(tmp_path / "cellq.zip"))
    monkeypatch.setattr(generate_module, "PARALLEL_PROCESSING_THRESHOLD", 0)
    monkeypatch.setattr("operaxn.gui.messagebox.askyesno", lambda *a, **k: True)
    reached, pressed = threading.Event(), threading.Event()
    owner, name = {"warm_up": (generate_module.FileProcessor, "_warm_up"),
                   "write": (nxs_writer.NXSWriter, "_write_scan")}[stage]
    original = getattr(owner, name)

    def held(*args, **kwargs):  # the stage's first call waits for the close button
        if not reached.is_set():
            reached.set()
            pressed.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, staticmethod(held) if stage == "warm_up" else held)
    tmp, cache = tempfile.gettempdir(), core.cache_dir()
    before = set(os.listdir(tmp))
    win = Toplevel(tk_root)
    app = OPERAXN(master=win)
    shown, left = [], []
    monkeypatch.setattr(app, "_show_message", lambda *a: shown.append(a))
    # Earlier tests' Tk objects are freed here: freed on the worker, their Tk call can block it
    gc.collect()
    others = set(threading.enumerate())
    app._process_files_async(UploadOptions(paths=[zpath], data_source=DataSourceType.INHOUSE,
                                           time_method=TimeMethod.ABSOLUTE, display_size=0,
                                           include_2d=False))
    worker = (set(threading.enumerate()) - others).pop()
    deadline = time.monotonic() + 30

    def poll():
        if reached.is_set() and not pressed.is_set():
            app._on_closing()
            pressed.set()
        if win.winfo_exists() and time.monotonic() < deadline:
            tk_root.after(10, poll)
            return
        left.extend(set(os.listdir(tmp)) - before)  # as the window closes
        left.extend(n for n in os.listdir(cache) if n.startswith("cellq"))
        tk_root.quit()

    try:
        tk_root.after(10, poll)
        tk_root.mainloop()
        worker.join(2)
        assert not win.winfo_exists() and not worker.is_alive(), "the load must stop first"
        assert left == [] and shown == [], (left, shown)
    finally:
        pressed.set()
        worker.join(30)
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_quit_during_an_image_or_gif_export_asks_and_stops_it(tk_root, monkeypatch, inhouse_dir,
                                                              tmp_path):
    """Closing mid range export or GIF asks; No lets it finish, Yes stops it as Stop does."""
    import glob
    from operaxn.dialog import ExportOptions
    win, app, _ = _gui_app(tk_root, monkeypatch)
    monkeypatch.delattr(app, "_show_message")  # the real one: no box once the quit is confirmed
    shown, answers = [], [False, False, True]
    for kind in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(f"operaxn.gui.messagebox.{kind}", lambda *a, **k: shown.append(a))
    monkeypatch.setattr("operaxn.gui.messagebox.askyesno", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr("operaxn.gui.filedialog.askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr("operaxn.gui.filedialog.asksaveasfilename",
                        lambda **k: str(tmp_path / "movie.gif"))

    def run(open_dialog, attr, start, settings):
        open_dialog()
        dialog = getattr(app, attr)
        original = dialog.update_progress

        def update_progress(value, text):  # the close button, clicked as the export pumps events
            if value == 1:
                app._on_closing()
            return original(value, text)

        dialog.update_progress = update_progress
        start(settings, dialog)

    scans = ExportOptions("multiple", 40, {"oned": True}, [1, 2, 3])
    pngs = lambda: glob.glob(str(tmp_path / "*.png"))  # noqa: E731
    try:
        _load_inhouse(win, app, inhouse_dir)
        app._plot_data()
        run(app._export_data, "_export_options_dialog", app._perform_export, scans)
        assert win.winfo_exists() and len(answers) == 2, "a quit during a range export must ask"
        run(app._create_gif, "_gif_dialog", app._run_gif,
            GIFSettings(fps=2, dpi=30, loop=0, scan_list=[1, 2, 3]))
        assert win.winfo_exists() and len(answers) == 1, "a quit during a GIF must ask"
        assert len(pngs()) == 3 and os.path.isfile(tmp_path / "movie.gif"), "No lets both finish"
        for png in pngs():
            os.remove(png)
        boxes = len(shown)
        run(app._export_data, "_export_options_dialog", app._perform_export, scans)
        _pump_until_closed(win)
        assert not win.winfo_exists() and not answers
        assert len(pngs()) == 1 and not glob.glob(str(tmp_path / "*.part")), "stopped after scan 1"
        assert len(shown) == boxes, "no box after a confirmed quit"
    finally:
        if win.winfo_exists():
            win.destroy()


@pytest.mark.display
def test_upload_dialog_takes_source_from_nxs(tk_root, neutron_dir, inhouse_dir, tmp_path):
    """Selecting a directly opened .nxs shows and locks its recorded source, with a grey note;
    choosing raw data again restores the user's own choice."""
    from operaxn.config import OPERAXNTheme
    process_paths([neutron_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.NEUTRON)
    neutron_nxs = str(tmp_path / "neutron_run.nxs")
    shutil.copy(get_loaded_nxs_path(), neutron_nxs)

    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg.source_var.set("synchrotron")
        dlg._paths = [neutron_nxs]
        dlg._update_option_states()
        assert dlg.source_var.get() == "neutron"
        assert all(str(r.cget("state")) == "disabled" for r in dlg._source_radios)
        assert "source and generation options" in dlg.note_var.get()
        assert dlg._note_label.cget("fg") == OPERAXNTheme.COLORS['text_secondary']

        dlg._paths = [inhouse_dir]
        dlg._update_option_states()
        assert dlg.source_var.get() == "synchrotron"
        assert all(str(r.cget("state")) == "normal" for r in dlg._source_radios)
        assert dlg.note_var.get() == ""
    finally:
        if dlg.winfo_exists():
            dlg.destroy()


@pytest.mark.display
def test_upload_dialog_takes_time_method_from_nxs(tk_root, inhouse_dir, tmp_path):
    """Picking a relative-recorded .nxs shows and locks Relative; raw data restores the choice."""
    nxs = str(tmp_path / "rel.nxs")
    ok, msgs = core.generate([inhouse_dir], nxs, core.DataSourceType.INHOUSE,
                             time_method=core.TimeMethod.RELATIVE)
    assert ok, msgs
    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg._paths = [nxs]
        dlg._update_option_states()
        assert dlg.time_var.get() == "relative"
        assert all(str(r.cget("state")) == "disabled" for r in dlg._time_radios)
        dlg._paths = [inhouse_dir]
        dlg._update_option_states()
        assert dlg.time_var.get() == "absolute"
        assert all(str(r.cget("state")) == "normal" for r in dlg._time_radios)
    finally:
        if dlg.winfo_exists():
            dlg.destroy()


@pytest.mark.display
def test_upload_dialog_wheel_scrolls_on_every_platform_and_unbinds_on_close(tk_root):
    """The wheel scrolls on every platform, a zero delta does not, and closing inside unbinds it."""
    sequences = ("<MouseWheel>", "<Button-4>", "<Button-5>")
    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg._canvas.event_generate("<Enter>")
        assert all(tk_root.bind_all(s) for s in sequences)
        steps = []
        dlg._scrollbar.winfo_ismapped = lambda: True
        dlg._canvas.yview_scroll = lambda n, _what: steps.append(n)
        for delta, num in ((120, "??"), (-240, "??"), (1, "??"), (-1, "??"), (0, "??"), (0, 4), (0, 5)):
            dlg._on_wheel(SimpleNamespace(delta=delta, num=num))
        assert steps == [-1, 2, -1, 1, -1, 1]
        dlg.cancel()  # what <Escape> runs
    finally:
        if dlg.winfo_exists():
            dlg.destroy()
    assert not any(tk_root.bind_all(s) for s in sequences)


@pytest.mark.display
def test_upload_file_picker_separates_patterns_by_spaces(tk_root, monkeypatch):
    """The 'All Supported' filter lists its patterns with spaces, as Tk's Linux file dialog needs."""
    asked = []
    monkeypatch.setattr("operaxn.dialog.filedialog.askopenfilenames",
                        lambda **kwargs: asked.append(kwargs["filetypes"]) or ())
    dlg = UploadOptionsDialog(tk_root)
    try:
        dlg._select_files()
    finally:
        dlg.destroy()
    patterns = dict(asked[0])["All Supported"]
    assert patterns == "*.zip *.dat *.edf *.txt *.xlsx *.csv *.hdf *.nxs *.xy", patterns


@pytest.mark.display
def test_upload_standard_echem_count_reads_singular(tk_root, monkeypatch):
    """The standard echem field reads '1 file' for one file and '2 files' for two."""
    picks = [("a.txt",), ("a.txt", "b.txt")]
    monkeypatch.setattr("operaxn.dialog.filedialog.askopenfilenames",
                        lambda **kwargs: picks.pop(0))
    dlg = UploadOptionsDialog(tk_root)
    try:
        shown = []
        for _ in range(2):
            dlg._select_std_echem()
            shown.append(dlg.std_echem_var.get())
    finally:
        dlg.destroy()
    assert shown == ["1 file", "2 files"], shown


# ============================================================================
# GIF export
# ============================================================================

def test_gif_export_needs_no_imageio_and_keeps_its_bytes(tmp_path, monkeypatch):
    """With imageio unavailable a GIF still exports, byte for byte the GIF Pillow writes from the
    frames' RGB pixels at 1000 / fps ms per frame and the chosen loop count."""
    import sys
    from PIL import Image
    rng = np.random.default_rng(0)
    files = []
    for i in range(4):
        rgba = rng.integers(0, 256, (30, 40, 4), dtype=np.uint8)
        rgba[..., 3] = 255 if i % 2 else rng.integers(0, 256, (30, 40), dtype=np.uint8)
        files.append(str(tmp_path / f"f{i}.png"))
        Image.fromarray(rgba).save(files[-1])
    files.append(files[3])  # the writer merges a repeated frame
    expected = str(tmp_path / "expected.gif")
    pixels = [Image.fromarray(np.asarray(Image.open(f))[..., :3]) for f in files]
    pixels[0].save(expected, format="GIF", save_all=True, append_images=pixels[1:],
                   duration=1000 / 3, loop=2)

    monkeypatch.setitem(sys.modules, "imageio", None)  # any import of imageio now fails
    assert not hasattr(operaxn.gui, "imageio"), "the GIF writer must not hold imageio"
    messages = []

    def frame(temp_dir, index, scan, dpi, layout=None):
        path = os.path.join(temp_dir, f"frame_{index:04d}.png")
        shutil.copyfile(files[index], path)
        return path
    fake = SimpleNamespace(state=SimpleNamespace(scans=[{"scan_num": i + 1} for i in range(5)]),
                           _gif_layout=lambda nums: (True, False, False, False),
                           _create_gif_frame=frame, _show_message=lambda *a: messages.append(a),
                           _close_dialog=lambda dialog: None)
    dialog = SimpleNamespace(begin_progress=lambda *a: None, update_progress=lambda *a: True)
    out = str(tmp_path / "anim.gif")
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=3, dpi=100, loop=2, scan_list=[1, 2, 3, 4, 5]),
                          dialog)
    assert messages and messages[-1][0] == "Export complete", messages
    with open(out, "rb") as written, open(expected, "rb") as reference:
        assert written.read() == reference.read()


def _gif_app(monkeypatch, n):
    """Return a stand-in app with n scans of distinct 1D data and a message log."""
    import operaxn.gui as gui_module
    x = np.linspace(10, 80, 50)

    def correlated(scans, scan_num, state=None):
        return {"scan_num": scan_num, "oned": {"x": x, "y": np.arange(50.0) * scan_num}}

    monkeypatch.setattr(gui_module, "get_correlated_data", correlated)
    source = gui_module.DataSourceType.INHOUSE
    fake = SimpleNamespace(
        state=SimpleNamespace(data_source=source, scans=[{"scan_num": i} for i in range(1, n + 1)],
                              echem_df=pd.DataFrame(), echem_arrays={"x": np.array([])},
                              intensity_limits={}, time_method="absolute",
                              oned_arrays={i: {} for i in range(1, n + 1)}, twod_arrays={},
                              neutron_arrays={}),
        config=operaxn.VisualiserConfig(data_source=source),
        _gif_placeholder=OPERAXN._gif_placeholder,
    )
    fake._gif_layout = lambda nums: OPERAXN._gif_layout(fake, nums)
    fake._create_gif_frame = lambda *a: OPERAXN._create_gif_frame(fake, *a)
    fake.messages = []
    fake._show_message = lambda *a: fake.messages.append(a)
    fake._close_dialog = lambda dialog: None
    return fake


DIALOG = SimpleNamespace(begin_progress=lambda *a: None, update_progress=lambda *a: True)


def test_generate_gif_keeps_frame_timing_without_fps_keyword(monkeypatch, tmp_path):
    """Each frame lasts 1000 / fps ms and the loop count is kept, with no fps deprecation."""
    from PIL import Image
    fake = _gif_app(monkeypatch, 3)
    out = str(tmp_path / "timed.gif")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OPERAXN._generate_gif(fake, out, GIFSettings(fps=4, dpi=40, loop=2, scan_list=[1, 2, 3]),
                              DIALOG)
    assert fake.messages and fake.messages[-1][0] == "Export complete", fake.messages
    assert not [w for w in caught if "fps" in str(w.message)], [str(w.message) for w in caught]
    with Image.open(out) as gif:
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
        assert gif.info["loop"] == 2
    assert durations == [250, 250, 250], durations


def test_generate_gif_streams_frames_to_the_writer(monkeypatch, tmp_path):
    """The writer holds a fixed few full-colour frames (first, previous, next), not one per scan."""
    from PIL import Image
    refs = []
    peak = [0]
    original = Image.Image.save

    def counting(self, fp, format=None, **params):
        # Count the frames handed to the GIF writer that are still alive as each new one
        # arrives; PIL images are unhashable, so by weak reference
        if not params.get("save_all"):  # the PNG frames matplotlib writes
            return original(self, fp, format, **params)
        refs.append(weakref.ref(self))

        def frames(source):
            for image in source:
                refs.append(weakref.ref(image))
                peak[0] = max(peak[0], sum(ref() is not None for ref in refs))
                yield image
        params["append_images"] = frames(params.get("append_images", []))
        return original(self, fp, format, **params)

    monkeypatch.setattr(Image.Image, "save", counting)
    fake = _gif_app(monkeypatch, 8)
    out = str(tmp_path / "long.gif")
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=5, dpi=40, loop=0,
                                                 scan_list=list(range(1, 9))), DIALOG)
    assert fake.messages and fake.messages[-1][0] == "Export complete", fake.messages
    with Image.open(out) as gif:
        assert gif.n_frames == 8
    assert peak[0] <= 3, peak[0]


def _frames(tmp_path, n=3):
    """Write n distinct opaque PNG frames and return their paths."""
    from PIL import Image
    paths = []
    for i in range(n):
        p = str(tmp_path / f"f{i}.png")
        Image.fromarray(np.full((40, 60, 4), (40 * i, 80, 120, 255), np.uint8)).save(p)
        paths.append(p)
    return paths


def _gif(out, files, monkeypatch=None, fail_on=None):
    """Run the real _generate_gif on prepared frames; return the last message."""
    from PIL import Image
    if fail_on is not None:
        real, calls = Image.open, [0]

        def flaky(path, *a, **k):
            calls[0] += 1
            if calls[0] == fail_on:
                raise OSError("disk full")
            return real(path, *a, **k)
        monkeypatch.setattr(Image, "open", flaky)
    msgs = []

    def frame(temp_dir, idx, scan, dpi, layout=None):
        dst = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        shutil.copyfile(files[idx], dst)
        return dst
    fake = SimpleNamespace(state=SimpleNamespace(scans=[{"scan_num": i + 1} for i in range(len(files))]),
                           _gif_layout=lambda nums: (True, False, False, False),
                           _create_gif_frame=frame, _show_message=lambda *a: msgs.append(a),
                           _close_dialog=lambda d: None)
    dialog = SimpleNamespace(begin_progress=lambda *a: None, update_progress=lambda *a: True)
    OPERAXN._generate_gif(fake, out, GIFSettings(fps=2, dpi=100, loop=0,
                                                 scan_list=list(range(1, len(files) + 1))), dialog)
    return msgs[-1]


def test_failed_gif_write_keeps_the_previous_gif(tmp_path, monkeypatch):
    """A write that fails part-way shows the error, leaves the old GIF byte for byte and no .part."""
    files = _frames(tmp_path)
    out = str(tmp_path / "anim.gif")
    assert _gif(out, files)[0] == "Export complete"
    before = open(out, "rb").read()
    title, text, _ = _gif(out, files[::-1], monkeypatch, fail_on=2)
    assert (title, text) == ("Export failed", "GIF could not be exported: disk full")
    assert open(out, "rb").read() == before
    assert not os.path.exists(out + ".part")
