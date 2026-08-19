"""Schema v4 on-disk structure: monopd and tofnpd group layout, units, and attrs."""
import os

import h5py
import numpy as np
import pytest

import core

import _builders
from _helpers import s, wall, tz_aware, assert_pynxtools_valid


def test_monopd_root_and_entry(inhouse_nxs):
    """@default chains to the operando echem plot, provenance lives on
    definition/program_name, and start_time is tz-aware."""
    with h5py.File(inhouse_nxs) as f:
        assert f.attrs.get("default") == "entry"
        e = f["entry"]
        assert e.attrs.get("default") == "operando_electrochemistry"
        assert "generator" not in e.attrs and "data_source" not in e.attrs, \
            "v3 provenance attrs left on entry"
        assert s(e["definition"]) == "NXoperando_monopd"
        assert e["definition"].attrs.get("version") == core.config.DEFINITION_VERSION
        assert e["definition"].attrs.get("URL", "").endswith("NXoperando_monopd.nxdl.xml")
        assert s(e["program_name"]) == "operaxn-core"
        assert e["program_name"].attrs.get("version") == core.config.GENERATOR_VERSION
        assert wall(e["start_time"]) == "2024-02-05T10:05:00"
        assert tz_aware(e["start_time"]), "entry start_time not tz-aware"


def test_monopd_process(inhouse_nxs):
    """Process group carries provenance fields and the correlation tolerance."""
    with h5py.File(inhouse_nxs) as f:
        p = f["entry/process"]
        assert p.attrs.get("NX_class") == "NXprocess"
        assert s(p["data_source"]) == "inhouse"
        assert s(p["correlation_method"]) == "absolute"
        assert p["total_scans"][()] == 2
        assert p["echem_time_tolerance"][()] == 300.0
        assert p["echem_time_tolerance"].attrs.get("units") == "s"


def test_monopd_scan_group(inhouse_nxs):
    """Each scan is an NXmonopd subentry whose end_time is start +
    exposure and whose instrument is a soft link."""
    with h5py.File(inhouse_nxs) as f:
        s1 = f["entry/scan_000001"]
        assert s1.attrs.get("NX_class") == "NXsubentry"
        assert s(s1["definition"]) == "NXmonopd"
        assert wall(s1["start_time"]) == "2024-02-05T10:05:00"
        assert wall(s1["end_time"]) == "2024-02-05T10:07:00", \
            "end_time should be start + exposure"
        assert isinstance(s1.get("instrument", getlink=True), h5py.SoftLink)


def test_monopd_environment(inhouse_nxs):
    """Per-scan NXenvironment holds correlated values with units, the
    window min/max, and echem index pointers; no voltage_log for XRD."""
    with h5py.File(inhouse_nxs) as f:
        env = f["entry/scan_000001/environment"]
        assert env.attrs.get("NX_class") == "NXenvironment"
        assert "voltage (V)" not in env and "current (mA)" not in env, \
            "unit-suffixed names present"
        assert abs(env["voltage"][()] - 3.76) < 1e-9
        assert env["voltage"].attrs.get("units") == "V"
        assert abs(env["current"][()] - 0.106) < 1e-9
        assert env["current"].attrs.get("units") == "mA"
        assert abs(env["voltage_min"][()] - 3.75) < 1e-9
        assert abs(env["voltage_max"][()] - 3.77) < 1e-9
        assert abs(env["current_min"][()] - 0.105) < 1e-9
        assert abs(env["current_max"][()] - 0.107) < 1e-9
        assert env["echem_index_first"][()] == 5
        assert env["echem_index_last"][()] == 7
        assert "echem_index_end" not in env, "reserved suffix name present"
        assert wall(env["scan_timestamp"]) == "2024-02-05T10:05:00"
        assert wall(env["midpoint_adjusted_timestamp"]) == "2024-02-05T10:06:00"
        assert env["exposure_time"][()] == 120.0
        assert env["exposure_time"].attrs.get("units") == "s"
        assert "voltage_log" not in env, "XRD scans should not carry a voltage_log"


def test_monopd_monitor_and_data(inhouse_nxs):
    """The monitor preset carries seconds units and the 1D NXdata keeps
    signal/axes attrs with the Sigma column stored as errors."""
    with h5py.File(inhouse_nxs) as f:
        mon = f["entry/scan_000001/monitor"]
        assert mon["preset"][()] == 120.0
        assert mon["preset"].attrs.get("units") == "s"
        d = f["entry/scan_000001/data"]
        assert d["polar_angle"].attrs.get("units") == "degrees"
        assert d.attrs.get("polar_angle_indices") == 0
        assert d.attrs.get("signal") == "data"
        assert "errors" in d, "1D errors dropped"
        assert np.allclose(d["errors"][()], np.sqrt(100.0 + np.arange(30)))


def test_monopd_operando_electrochemistry(inhouse_nxs):
    """Operando echem group follows the NXlog time convention with NXdata attrs."""
    with h5py.File(inhouse_nxs) as f:
        oe = f["entry/operando_electrochemistry"]
        assert "timestamps" not in oe and "time" in oe
        assert oe["time"].attrs.get("units") == "s"
        assert wall(oe["time"].attrs.get("start")) == "2024-02-05T10:00:00"
        assert tz_aware(oe["time"].attrs.get("start"))
        assert np.allclose(oe["time"][:3], [0.0, 60.0, 120.0])
        assert oe["voltage"].attrs.get("units") == "V"
        assert oe["current"].attrs.get("units") == "mA"
        assert oe.attrs.get("signal") == "voltage"
        assert oe.attrs.get("axes") == "time"
        assert list(oe.attrs.get("auxiliary_signals")) == ["current"]


def test_monopd_standard_electrochemistry(inhouse_nxs):
    """Standard echem NXenvironment holds per-file NXdata groups with time/@start."""
    with h5py.File(inhouse_nxs) as f:
        se = f["entry/standard_electrochemistry"]
        assert se.attrs.get("NX_class") == "NXenvironment"
        assert se.attrs.get("num_files") == 1
        f1 = se["file_001"]
        assert f1.attrs.get("NX_class") == "NXdata"
        assert "time" in f1 and "voltage" in f1
        assert wall(f1["time"].attrs.get("start")) == "2024-02-05T10:00:00"


@pytest.mark.skipif(not core.FABIO_AVAILABLE, reason="fabio not installed")
def test_monopd_edf_provenance(inhouse_nxs):
    """EDF headers feed the instrument (wavelength in angstrom), monitor
    integral, and image_source NXnote; data carries no twod_* attrs."""
    with h5py.File(inhouse_nxs) as f:
        e = f["entry"]
        assert abs(e["instrument/crystal/wavelength"][()] - 1.541891) < 1e-6, \
            "wavelength not converted m->angstrom"
        assert e["instrument/detector/distance"].attrs.get("units") == "m"
        mon = e["scan_000001/monitor"]
        assert mon["integral"][()] == 155717.0
        assert mon["integral"].attrs.get("units") == "counts"
        i1 = e["scan_000001"]
        assert i1["image_source"].attrs.get("NX_class") == "NXnote"
        assert s(i1["image_source/file_name"]).endswith("image_001.edf")
        assert s(i1["image_source/type"]) == "application/x-esrf-edf"
        assert bool(i1["image_source/embedded"][()]) is False
        assert not any(k.startswith("twod_") for k in i1["data"].attrs), \
            "v3 twod_* attrs left on data"


@pytest.mark.skipif(not core.FABIO_AVAILABLE, reason="fabio not installed")
def test_monopd_image_only_scans(tmp_path):
    """A dataset of only EDF images yields image_source scans with no data group."""
    src = str(tmp_path / "images_only")
    os.makedirs(src)
    for i, ts in enumerate(_builders.DETAILED_SCAN_TIMES, start=1):
        _builders.write_edf_image(os.path.join(src, f"image_{i:03d}.edf"), ts,
                                  offset=float(i))
    out = str(tmp_path / "images_only.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        scan_names = sorted(k for k in f["entry"] if k.startswith("scan_"))
        assert len(scan_names) == 2, str(scan_names)
        for name in scan_names:
            grp = f["entry"][name]
            assert "data" not in grp, f"{name} must not carry a data group"
            assert grp["image_source"].attrs.get("NX_class") == "NXnote"
    model = core.load(out)
    assert all(x.oned is None for x in model.scans)
    assert all(x.twod_source for x in model.scans)
    with_monitor = [x for x in model.scans if x.monitor and "integral" in x.monitor]
    assert len(with_monitor) == 2, f"{len(with_monitor)}/2 scans carry monitor integrals"
    assert_pynxtools_valid(out, "image-only")


def test_tofnpd_entry_and_scan(neutron_nxs):
    """Entry/subentry declare NXoperando_tofnpd/NXtofnpd; scan
    start/end times come from the logbook."""
    with h5py.File(neutron_nxs) as f:
        e = f["entry"]
        assert s(e["definition"]) == "NXoperando_tofnpd"
        s1 = e["scan_000001"]
        assert s(s1["definition"]) == "NXtofnpd"
        assert wall(s1["start_time"]) == "2024-02-05T10:00:00"
        assert wall(s1["end_time"]) == "2024-02-05T10:30:00"


def test_tofnpd_environment(neutron_nxs):
    """Neutron scans carry correlated voltage, a steady-state window,
    voltage/current NXlogs, and the raw logbook entry."""
    with h5py.File(neutron_nxs) as f:
        env = f["entry/scan_000001/environment"]
        assert abs(env["voltage"][()] - 3.85) < 1e-9
        assert abs(env["voltage_min"][()] - 3.70) < 1e-9, "steady-state window min"
        assert abs(env["voltage_max"][()] - 4.00) < 1e-9, "steady-state window max"
        assert env["echem_index_first"][()] == 0
        assert env["echem_index_last"][()] == 30
        vlog = env.get("voltage_log")
        assert vlog is not None, "voltage_log NXlog not written"
        assert vlog.attrs.get("NX_class") == "NXlog"
        assert len(vlog["time"]) == 31
        assert wall(vlog["time"].attrs.get("start")) == "2024-02-05T10:00:00"
        assert np.allclose(vlog["value"][()], 3.7 + 0.01 * np.arange(31))
        clog = env.get("current_log")
        assert clog is not None, "current_log NXlog not written"
        assert np.allclose(clog["value"][()], 0.1 + 0.001 * np.arange(31))
        assert b"Op-Run1" in env["logbook_entry"][()]


def test_tofnpd_banks(neutron_nxs):
    """Banks split into TOF + d NXdata groups with Mantid-mapped units."""
    with h5py.File(neutron_nxs) as f:
        s1 = f["entry/scan_000001"]
        assert "bank_1" in s1 and "bank_1_d" in s1
        assert "bank_2" in s1 and "bank_2_d" in s1
        b1, b1d = s1["bank_1"], s1["bank_1_d"]
        assert {"time_of_flight", "data", "errors"} <= set(b1.keys())
        assert "d_spacing" not in b1
        assert b1["time_of_flight"].attrs.get("units") == "microsecond"
        assert b1.attrs.get("x_label") == "Time-of-flight"
        assert {"d_spacing", "data", "errors"} <= set(b1d.keys())
        assert b1d["d_spacing"].attrs.get("units") == "angstrom"
        assert b1.attrs.get("signal") == "data"
        assert b1.attrs.get("axes") == "time_of_flight"
        assert b1d.attrs.get("axes") == "d_spacing"
        assert b1.attrs.get("spectrum") == 1
        assert np.allclose(b1["errors"][()], 0.3 + 0.001 * np.arange(30))


def test_tofnpd_banks_without_mantid_header(tmp_path):
    """Bank axes carry units even when the source .dat has no Mantid header.

    The shared builders (and real reduced data) often omit the
    "X-axis unit is:" line; NXoperando_tofnpd types these axes, so a missing
    units attribute makes the whole file invalid."""
    src = str(tmp_path / "plain_neutron")
    _builders.write_neutron_dir(src)
    with open(os.path.join(src, "123456-1-0.dat")) as fh:
        assert "X-axis unit is" not in fh.read(), "fixture must stay header-less"
    out = str(tmp_path / "plain_neutron.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        s1 = f["entry/scan_000001"]
        assert s1["bank_1/time_of_flight"].attrs.get("units") == "microsecond"
        assert s1["bank_1_d/d_spacing"].attrs.get("units") == "angstrom"
    assert_pynxtools_valid(out, "neutron without Mantid header")


@pytest.mark.parametrize("fixture", ["inhouse_nxs", "neutron_nxs"])
def test_entry_window_spans_all_scans(request, fixture):
    """The entry time window contains every acquisition it holds."""
    with h5py.File(request.getfixturevalue(fixture)) as f:
        entry = f["entry"]
        e_start, e_end = wall(entry["start_time"]), wall(entry["end_time"])
        scans = [k for k in entry if k.startswith("scan_")]
        assert scans, "no scan subentries"
        for name in scans:
            sub = entry[name]
            assert wall(sub["start_time"]) >= e_start,                 f"{name} starts before the entry window"
            if "end_time" in sub:
                assert wall(sub["end_time"]) <= e_end,                     f"{name} ends after the entry window"


@pytest.mark.parametrize("nxs_fixture", ["inhouse_nxs", "neutron_nxs"])
def test_scan_timestamp_always_written(request, nxs_fixture):
    """Every scan subentry carries environment/scan_timestamp."""
    with h5py.File(request.getfixturevalue(nxs_fixture)) as f:
        names = [k for k in f["entry"] if k.startswith("scan_")]
        assert names
        for name in names:
            assert "scan_timestamp" in f["entry"][name]["environment"], name
