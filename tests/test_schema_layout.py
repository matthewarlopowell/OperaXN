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
    definition and process, and start_time is tz-aware."""
    with h5py.File(inhouse_nxs) as f:
        assert f.attrs.get("default") == "entry"
        e = f["entry"]
        assert e.attrs.get("default") == "operando_electrochemistry"
        assert "generator" not in e.attrs and "data_source" not in e.attrs, \
            "v3 provenance attrs left on entry"
        assert s(e["definition"]) == "NXoperando_monopd"
        assert e["definition"].attrs.get("version") == core.config.DEFINITION_VERSION
        assert e["definition"].attrs.get("URL", "").endswith("NXoperando_monopd.nxdl.xml")
        assert "program_name" not in e, "duplicate of process/program"
        assert wall(e["start_time"]) == "2024-02-05T10:05:00"
        assert tz_aware(e["start_time"]), "entry start_time not tz-aware"


def test_monopd_process(inhouse_nxs):
    """Process group is the single provenance record: program, version,
    data source, correlation method and scan count; no tolerance field."""
    with h5py.File(inhouse_nxs) as f:
        p = f["entry/process"]
        assert p.attrs.get("NX_class") == "NXprocess"
        assert s(p["program"]) == "operaxn"
        assert s(p["version"]) == core.config.GENERATOR_VERSION
        assert s(p["data_source"]) == "inhouse"
        assert s(p["correlation_method"]) == "absolute"
        assert p["total_scans"][()] == 2
        assert "echem_time_tolerance" not in p
        assert "twod_included" not in p and "twod_max_display_size" not in p


def test_monopd_scan_group(inhouse_nxs):
    """Each scan is a bare NXsubentry: a soft link to the shared instrument,
    environment (whose end_time is start + exposure), the EDF beam-monitor
    record when the header carries one, and data; nothing that repeats
    entry-level facts."""
    with h5py.File(inhouse_nxs) as f:
        s1 = f["entry/scan_000001"]
        assert s1.attrs.get("NX_class") == "NXsubentry"
        assert s1.attrs.get("scan_number") == 1
        env = s1["environment"]
        assert wall(env["start_time"]) == "2024-02-05T10:05:00"
        assert wall(env["end_time"]) == "2024-02-05T10:07:00", \
            "end_time should be start + exposure"
        assert isinstance(s1.get("instrument", getlink=True), h5py.SoftLink)
        for name in ("definition", "title", "start_time", "end_time"):
            assert name not in s1, f"{name} repeated in the subentry"


def test_monopd_environment(inhouse_nxs):
    """Per-scan NXenvironment holds the window, the correlated values with
    units, the window extrema and the timestamps; no index pointers or logs."""
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
        for name in ("echem_index_first", "echem_index_last", "capacity",
                     "voltage_log", "current_log"):
            assert name not in env, f"{name} is derivable and must not be stored"
        assert wall(env["scan_timestamp"]) == "2024-02-05T10:05:00"
        assert wall(env["midpoint_adjusted_timestamp"]) == "2024-02-05T10:06:00"
        assert env["exposure_time"][()] == 120.0
        assert env["exposure_time"].attrs.get("units") == "s"


def test_monopd_data(inhouse_nxs):
    """The 1D NXdata keeps signal/axes attrs with the Sigma column stored
    as data_errors."""
    with h5py.File(inhouse_nxs) as f:
        d = f["entry/scan_000001/data"]
        assert d["polar_angle"].attrs.get("units") == "degrees"
        assert "polar_angle_indices" not in d.attrs, "redundant for a 1-D axis"
        assert d.attrs.get("signal") == "data"
        assert "data_errors" in d, "1D uncertainties dropped"
        assert "errors" not in d, "deprecated NXdata errors name written"
        assert np.allclose(d["data_errors"][()], np.sqrt(100.0 + np.arange(30)))


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
        assert "time_indices" not in oe.attrs, "redundant NXdata attribute written"
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
    """EDF headers feed the instrument (wavelength in angstrom), the
    entry-level monitor (timer mode, the shared 120 s preset, no integral:
    the header's Monitor counter is 0) and the image_source NXnote; data
    carries no twod_* attrs."""
    with h5py.File(inhouse_nxs) as f:
        e = f["entry"]
        assert abs(e["instrument/crystal/wavelength"][()] - 1.541891) < 1e-6, \
            "wavelength not converted m->angstrom"
        assert e["instrument/detector/distance"].attrs.get("units") == "m"
        mon = e["monitor"]
        assert mon.attrs.get("NX_class") == "NXmonitor"
        assert s(mon["mode"]) == "timer"
        assert mon["preset"][()] == 120.0 and mon["preset"].attrs.get("units") == "s"
        assert "integral" not in mon and "pilct1" not in mon, list(mon)
        assert "monitor" not in e["scan_000001"], "Monitor counter 0: no integral"
        assert "monitor" not in e["scan_000002"], "no EDF, so no monitor"
        i1 = e["scan_000001"]
        assert i1["image_source"].attrs.get("NX_class") == "NXnote"
        assert s(i1["image_source/file_name"]).endswith("image_001.edf")
        assert s(i1["image_source/type"]) == "application/x-esrf-edf"
        assert bool(i1["image_source/embedded"][()]) is False
        assert "max_display_size" not in i1["image_source"]
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
    assert all(not x.monitor for x in model.scans), "no beam monitor, no integrals"
    with h5py.File(out) as f:
        assert s(f["entry/monitor/mode"]) == "timer"
        assert f["entry/monitor/preset"][()] == 120.0
    assert_pynxtools_valid(out, "image-only")


def test_tofnpd_entry_and_scan(neutron_nxs):
    """The entry declares NXoperando_tofnpd; each scan's acquisition window
    in environment comes from the logbook, and nothing else is repeated."""
    with h5py.File(neutron_nxs) as f:
        e = f["entry"]
        assert s(e["definition"]) == "NXoperando_tofnpd"
        s1 = e["scan_000001"]
        env = s1["environment"]
        assert wall(env["start_time"]) == "2024-02-05T10:00:00"
        assert wall(env["end_time"]) == "2024-02-05T10:30:00"
        assert isinstance(s1.get("instrument", getlink=True), h5py.SoftLink)
        for name in ("definition", "title", "start_time", "end_time", "monitor"):
            assert name not in s1, f"{name} repeated in the subentry"


def test_tofnpd_environment(neutron_nxs):
    """Neutron scans carry the correlated voltage, the logbook window and
    the raw logbook entry; no derived window summary or log slices."""
    with h5py.File(neutron_nxs) as f:
        env = f["entry/scan_000001/environment"]
        assert abs(env["voltage"][()] - 3.85) < 1e-9
        assert abs(env["voltage_min"][()] - 3.70) < 1e-9, "window min"
        assert abs(env["voltage_max"][()] - 4.00) < 1e-9, "window max"
        assert wall(env["start_time"]) == "2024-02-05T10:00:00"
        assert wall(env["end_time"]) == "2024-02-05T10:30:00"
        assert "voltage_log" not in env and "current_log" not in env
        assert b"Op-Run1" in env["logbook_entry"][()]


def test_tofnpd_banks(neutron_nxs):
    """Banks split into TOF + d NXdata groups with Mantid-mapped units."""
    with h5py.File(neutron_nxs) as f:
        s1 = f["entry/scan_000001"]
        assert "bank_1" in s1 and "bank_1_d" in s1
        assert "bank_2" in s1 and "bank_2_d" in s1
        b1, b1d = s1["bank_1"], s1["bank_1_d"]
        assert {"time_of_flight", "data", "data_errors"} <= set(b1.keys())
        assert "d_spacing" not in b1
        assert b1["time_of_flight"].attrs.get("units") == "microsecond"
        assert b1.attrs.get("x_label") == "Time-of-flight"
        assert {"d_spacing", "data", "data_errors"} <= set(b1d.keys())
        assert b1d["d_spacing"].attrs.get("units") == "angstrom"
        assert b1.attrs.get("signal") == "data"
        assert b1.attrs.get("axes") == "time_of_flight"
        assert b1d.attrs.get("axes") == "d_spacing"
        assert b1.attrs.get("spectrum") == 1
        assert np.allclose(b1["data_errors"][()], 0.3 + 0.001 * np.arange(30))


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
        assert "monitor" not in f["entry"], "unprofiled instrument must write no monitor"
        assert "detector" not in f["entry/instrument"], "empty detector group written"
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
            assert wall(sub["environment/start_time"]) >= e_start,                 f"{name} starts before the entry window"
            if "end_time" in sub["environment"]:
                assert wall(sub["environment/end_time"]) <= e_end,                     f"{name} ends after the entry window"


@pytest.mark.parametrize("nxs_fixture", ["inhouse_nxs", "neutron_nxs"])
def test_scan_timestamp_always_written(request, nxs_fixture):
    """Every scan subentry carries environment/scan_timestamp."""
    with h5py.File(request.getfixturevalue(nxs_fixture)) as f:
        names = [k for k in f["entry"] if k.startswith("scan_")]
        assert names
        for name in names:
            assert "scan_timestamp" in f["entry"][name]["environment"], name


# --- member order: creation-order tracking, parent definition sequence ---

# Expected on-disk order per group: the NXmonopd / NXtofnpd sequence with
# OperaXN's additions directly after the item they extend. Names absent from
# a given file (fabio-dependent harvests, optional fields) are skipped, so
# each assertion is on the relative order of what is present.
ENTRY_ORDER = ["title", "start_time", "end_time", "definition",
               "experiment_identifier", "process",
               "pre_sample_flightpath", "user", "instrument", "sample",
               "cycling_protocol", "monitor", "operando_electrochemistry",
               "standard_electrochemistry"]
INSTRUMENT_ORDER = ["name", "source", "crystal", "detector",
                    "edf_metadata", "synchrotron_metadata"]
SOURCE_ORDER = ["type", "name", "probe"]
SCAN_ORDER = ["instrument", "environment", "monitor", "data", "image_source",
              "image_data"]
ENVIRONMENT_ORDER = ["start_time", "end_time", "voltage", "current",
                     "voltage_min", "voltage_max", "current_min", "current_max",
                     "scan_timestamp", "midpoint_adjusted_timestamp",
                     "voltage_timestamp", "exposure_time", "logbook_entry"]
ENTRY_MONITOR_ORDER = ["mode", "preset"]
MONITOR_ORDER = ["integral"]
PROTOCOL_ORDER = ["technique", "voltage_window_lower", "voltage_window_upper",
                  "C_rate", "instrument", "software", "raw_data_file"]
IMAGE_SOURCE_ORDER = ["file_name", "type", "description", "embedded",
                      "original_shape"]


def _members(group):
    """Member names as h5py lists them: creation order for tracked groups."""
    return list(group.keys())


def _tracks_order(group):
    """True when the group records link creation order."""
    flags = group.id.get_create_plist().get_link_creation_order()
    return bool(flags & h5py.h5p.CRT_ORDER_TRACKED)


def _assert_order(group, expected, tag):
    """The members of group named in expected appear in that order."""
    present = [n for n in expected if n in group]
    listed = [n for n in _members(group) if n in expected]
    assert listed == present, f"{tag}: on disk {listed}, expected {present}"


@pytest.mark.parametrize("fixture", ["inhouse_nxs", "neutron_nxs", "protocol_nxs"])
def test_creation_order_tracked_everywhere(request, fixture):
    """The root and every group record creation order, so readers see the
    written sequence instead of HDF5's alphabetical index."""
    with h5py.File(request.getfixturevalue(fixture)) as f:
        assert _tracks_order(f["/"]), "root group"
        untracked = []

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and not _tracks_order(obj):
                untracked.append(name)
        f.visititems(visit)
        assert not untracked, untracked


def test_monopd_member_order(inhouse_nxs):
    """NXmonopd sequence (title, start_time, definition, instrument, sample)
    with additions after the item they extend, source fields type/name/probe,
    and the scans last in order."""
    with h5py.File(inhouse_nxs) as f:
        e = f["entry"]
        _assert_order(e, ENTRY_ORDER, "entry")
        names = _members(e)
        assert names[-2:] == ["scan_000001", "scan_000002"], names
        inst = e["instrument"]
        _assert_order(inst, INSTRUMENT_ORDER, "instrument")
        assert _members(inst["source"]) == SOURCE_ORDER
        s1 = e["scan_000001"]
        assert _members(s1)[:2] == ["instrument", "environment"]
        _assert_order(s1, SCAN_ORDER, "scan")
        assert _members(s1["environment"])[:2] == ["start_time", "end_time"]
        _assert_order(s1["environment"], ENVIRONMENT_ORDER, "environment")
        assert _members(e["monitor"]) == ENTRY_MONITOR_ORDER
        if "monitor" in s1:
            assert _members(s1["monitor"]) == MONITOR_ORDER
        assert _members(s1["data"]) == ["polar_angle", "data", "data_errors"]


def test_monopd_protocol_member_order(protocol_nxs):
    """cycling_protocol follows sample and lists its fields in definition
    order; sample keeps name before preparation_date."""
    with h5py.File(protocol_nxs) as f:
        e = f["entry"]
        _assert_order(e, ENTRY_ORDER, "entry")
        names = _members(e)
        assert names.index("cycling_protocol") == names.index("sample") + 1
        assert _members(e["cycling_protocol"]) == PROTOCOL_ORDER
        assert _members(e["sample"]) == ["name", "preparation_date"]


def test_tofnpd_member_order(neutron_nxs):
    """NXtofnpd sequence: pre_sample_flightpath and user before instrument,
    monitor after the sample block, detector_number leading the bank
    geometry, data before the bank axis, banks in numeric order with TOF
    before d-spacing."""
    with h5py.File(neutron_nxs) as f:
        e = f["entry"]
        assert _members(e) == [
            "title", "start_time", "end_time", "definition",
            "experiment_identifier", "process",
            "pre_sample_flightpath", "user", "instrument", "sample",
            "monitor", "operando_electrochemistry", "scan_000001"]
        assert _members(e["monitor"]) == [
            "mode", "detector_number", "distance", "polar_angle", "azimuthal_angle"]
        inst = e["instrument"]
        assert _members(inst) == ["name", "source", "detector"]
        assert _members(inst["source"]) == SOURCE_ORDER
        assert _members(inst["detector"]) == [
            "description", "detector_number", "distance", "polar_angle"]
        s1 = e["scan_000001"]
        assert _members(s1) == [
            "instrument", "environment", "bank_1", "bank_1_d", "bank_2", "bank_2_d"]
        assert _members(s1["environment"])[:2] == ["start_time", "end_time"]
        _assert_order(s1["environment"], ENVIRONMENT_ORDER, "environment")
        assert _members(s1["bank_1"]) == ["data", "data_errors", "time_of_flight"]
        assert _members(s1["bank_1_d"]) == ["data", "data_errors", "d_spacing"]


@pytest.mark.skipif(not core.FABIO_AVAILABLE, reason="fabio not installed")
def test_monopd_image_member_order(tmp_path):
    """image_source precedes image_data and lists its fields in definition
    order, with original_shape present only once the image was embedded."""
    src = str(tmp_path / "twod")
    _builders.write_twod_dir(src)
    for embed in (False, True):
        out = str(tmp_path / f"twod_{int(embed)}.nxs")
        ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE,
                                 include_2d_images=embed)
        assert ok, str(msgs)
        with h5py.File(out) as f:
            entry = f["entry"]
            subs = [entry[n] for n in entry
                    if n.startswith("scan_") and "image_source" in entry[n]]
            assert subs, "no subentry carries image_source"
            sub = subs[0]
            _assert_order(sub, SCAN_ORDER, f"scan (embed={embed})")
            assert ("image_data" in sub) is embed
            expected = [n for n in IMAGE_SOURCE_ORDER
                        if embed or n != "original_shape"]
            assert _members(sub["image_source"]) == expected


def test_tofnpd_unreadable_bank_leaves_no_shell(tmp_path):
    """A bank file that fails to read yields no NXdata group for it (the
    definition requires data in every bank group); the other banks are intact."""
    src = str(tmp_path / "neutron_bad_bank")
    _builders.write_neutron_dir(src)
    with open(os.path.join(src, "123456-2-0.dat"), "w") as f:
        f.write("# Time-of-flight Y E\nnot numbers at all\n")
    nxs = str(tmp_path / "bad_bank.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.NEUTRON)
    assert ok, msgs
    with h5py.File(nxs) as f:
        s1 = f["entry/scan_000001"]
        assert "bank_2" not in s1, "empty NXdata shell left for an unreadable bank"
        assert "bank_2_d" in s1 and "bank_1" in s1 and "bank_1_d" in s1
        assert s1["bank_2_d"].attrs.get("signal") == "data"
    assert_pynxtools_valid(nxs, "unreadable bank")


@pytest.mark.skipif(not core.FABIO_AVAILABLE, reason="fabio not installed")
def test_image_source_path_normalised(tmp_path):
    """Collected paths are normalised, so a source folder given with forward
    slashes (the file dialog's form on Windows) records one separator style."""
    src = str(tmp_path / "twod_slash")
    _builders.write_twod_dir(src)
    nxs = str(tmp_path / "twod_slash.nxs")
    ok, msgs = core.generate([src.replace(os.sep, "/")], nxs,
                             core.DataSourceType.INHOUSE)
    assert ok, msgs
    with h5py.File(nxs) as f:
        name = s(f["entry/scan_000001/image_source/file_name"])
        assert name == os.path.normpath(name), name


@pytest.mark.skipif(not core.FABIO_AVAILABLE, reason="fabio not installed")
def test_monopd_monitor_integral_from_positive_counter(tmp_path):
    """A positive Monitor counter in an EDF header yields that scan's
    monitor/integral; a zero counter yields no per-scan group; the counting
    mode and shared preset sit once at entry level."""
    src = str(tmp_path / "monitored")
    os.makedirs(src)
    _builders.write_edf_image(os.path.join(src, "image_001.edf"),
                              _builders.DETAILED_SCAN_TIMES[0], monitor="5000")
    _builders.write_edf_image(os.path.join(src, "image_002.edf"),
                              _builders.DETAILED_SCAN_TIMES[1], offset=2.0)
    out = str(tmp_path / "monitored.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        e = f["entry"]
        assert _members(e["monitor"]) == ENTRY_MONITOR_ORDER
        assert s(e["monitor/mode"]) == "timer" and e["monitor/preset"][()] == 120.0
        mon = e["scan_000001/monitor"]
        assert _members(mon) == MONITOR_ORDER
        assert mon["integral"][()] == 5000.0
        assert mon["integral"].attrs.get("units") == "counts"
        assert "monitor" not in e["scan_000002"], "zero counter: no integral"
    assert_pynxtools_valid(out, "monitor integral")
