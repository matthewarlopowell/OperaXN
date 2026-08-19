"""Checks for the operaxn.input adapter (GUI contract over the core pipeline)."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.gui
pytest.importorskip("tkinter")

import h5py

import core
import operaxn
from operaxn import input as gui_input
from operaxn.dialog import UploadOptions, UploadOptionsDialog
from operaxn.gui import OPERAXN, ButtonPanel
from operaxn.input import (
    DataSourceType,
    TimeMethod,
    add_standard_echem_files,
    cleanup_session_cache,
    export_nxs,
    get_correlated_data,
    get_loaded_nxs_path,
    make_echem_arrays,
    make_neutron_arrays,
    make_oned_arrays,
    make_twod_arrays,
    process_paths,
)

import _builders

needs_fabio = pytest.mark.skipif(not core.FABIO_AVAILABLE,
                                 reason="fabio not installed")


def _reset_session():
    """Restore operaxn.input._session to its pristine module state."""
    gui_input._session.update({"nxs_path": None, "input_paths": None,
                               "exported": False, "standard_echem": []})
    gui_input._session.pop("global_metadata", None)


@pytest.fixture(autouse=True)
def _own_session():
    """Each test owns the module-global session; its cache .nxs is dropped after."""
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
    """8x8 EDF image whose Date pairs it with in-house scan 1."""
    import fabio
    img = fabio.edfimage.EdfImage(
        data=np.arange(64, dtype=np.float32).reshape(8, 8) + 1.0,
        header={"Date": _builders.SCAN_TIMES[0], "ExposureTime": "120.0",
                "WaveLength": "1.54"})
    img.write(path)


@pytest.fixture
def inhouse_edf_dir(inhouse_dir):
    """In-house dataset plus an EDF 2D image paired with scan 1 by Date."""
    _write_edf(os.path.join(inhouse_dir, "image_001.edf"))
    return inhouse_dir


def _std_echem_file(path, **overrides):
    """Standard (non-operando) echem file distinct from the operando series."""
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
    scans, echem_df, method = process_paths(
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
    scans, echem_df, _ = process_paths(
        [inhouse_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    corr = get_correlated_data(scans, echem_df, 1)
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


def test_neutron_adapter(neutron_dir):
    """Neutron upload yields one correlated scan with per-bank display names."""
    scans_n, _, _ = process_paths(
        [neutron_dir], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.NEUTRON)
    assert len(scans_n) == 1, f"got {len(scans_n)}"
    sn = scans_n[0]
    assert sn["echem"] == 3.85, f"got {sn['echem']}"
    assert sn["neutron_files"] is not None, str(sn["neutron_files"])
    assert sn["neutron_files"]["1"]["tof"] == "123456-1-0.dat", \
        str(sn["neutron_files"])
    assert sn["neutron_files"]["1"]["d"] == "123456-1-d-0.dat", \
        str(sn["neutron_files"])

    n_arrays = make_neutron_arrays(scans_n)
    assert set(n_arrays[1].keys()) == {"1", "2"}
    assert np.allclose(n_arrays[1]["1"]["tof"]["x"], _builders.TOF_X)


@needs_fabio
def test_embedded_twod_survives_source_deletion(inhouse_edf_dir, tmp_path):
    """An embedded 2D image loads from the .nxs after the source EDF is gone."""
    embedded_nxs = str(tmp_path / "embedded.nxs")
    ok, _ = core.generate([inhouse_edf_dir], embedded_nxs,
                          data_source=core.DataSourceType.INHOUSE,
                          include_2d_images=True)
    assert ok

    os.unlink(os.path.join(inhouse_edf_dir, "image_001.edf"))

    scans_e, _, _ = process_paths(
        [embedded_nxs], time_method=TimeMethod.ABSOLUTE,
        data_source=DataSourceType.INHOUSE)
    twod_e = make_twod_arrays(scans_e)
    assert 1 in twod_e and "image" in twod_e[1], str(twod_e.get(1, {}).keys())
    assert twod_e[1]["image"].shape == (8, 8)


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


@needs_fabio
def test_export_carries_generation_embedded_twod(inhouse_edf_dir, tmp_path):
    """Images embedded at upload land in the .nxs and survive the copy export."""
    process_paths([inhouse_edf_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE, include_2d_images=True)
    export_embedded = str(tmp_path / "exported_embedded.nxs")
    ok, msgs = export_nxs(export_embedded)
    assert ok, str(msgs)
    embedded_model = core.load(export_embedded)
    assert embedded_model.scans[0].twod_embedded
    assert embedded_model.scans[0].twod is not None
    assert embedded_model.scans[0].twod.shape == (8, 8)


def test_export_after_direct_nxs_load_copies_user_file(inhouse_dir, tmp_path):
    """Export after a direct .nxs load still copies the user's file."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    export_dest = str(tmp_path / "exported.nxs")
    export_nxs(export_dest)

    process_paths([export_dest], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    copy2 = str(tmp_path / "copy_of_direct.nxs")
    ok, _ = export_nxs(copy2)
    assert ok and os.path.isfile(copy2)
    _discard(core.default_cache_path([inhouse_dir]))


def test_gui_export_button_wiring():
    """The GUI declares the Export NeXus button and its handler (headless)."""
    assert any(k == "export_nxs" for k, *_ in ButtonPanel.BUTTON_CONFIGS)
    assert callable(getattr(OPERAXN, "_export_nxs", None))


def test_cache_unsaved_removed_on_exit(inhouse_dir):
    """The unsaved session cache .nxs is deleted by cleanup on app exit."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    assert cache_file and os.path.isfile(cache_file)
    cleanup_session_cache()
    assert not os.path.exists(cache_file)


def test_cache_exported_file_kept_on_exit(inhouse_dir, tmp_path):
    """An exported session cache .nxs survives cleanup on app exit."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cache_file = get_loaded_nxs_path()
    export_nxs(str(tmp_path / "kept.nxs"))
    cleanup_session_cache()
    assert os.path.isfile(cache_file)


def test_cache_user_nxs_never_deleted(inhouse_dir, tmp_path):
    """Cleanup never deletes a directly opened user .nxs file."""
    process_paths([inhouse_dir], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    kept_export = str(tmp_path / "kept.nxs")
    export_nxs(kept_export)
    process_paths([kept_export], time_method=TimeMethod.ABSOLUTE,
                  data_source=DataSourceType.INHOUSE)
    cleanup_session_cache()
    assert os.path.isfile(kept_export)
    _discard(core.default_cache_path([inhouse_dir]))


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
    """Standard echem is stored in the .nxs, kept out of the GUI's operando data,
    and carried along by the copy export."""
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
    """Appending echem to a loaded user .nxs works on a cache copy, leaving the
    original file untouched."""
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
            "standard_echem_files"} \
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
