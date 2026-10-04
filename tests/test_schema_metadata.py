"""Schema v4 provenance: reader round-trips, overrides, v3 back-compat, validation."""
import ast
import os

import h5py
import numpy as np
import pytest

import core

import _builders
from _helpers import s, wall, tz_aware, assert_pynxtools_valid


def test_monopd_reader_roundtrip(inhouse_nxs):
    """A generated file loads back with the same shapes and the v4
    fields (window, extrema, echem_df, metadata) populated."""
    m = core.load(inhouse_nxs)
    m1 = m.scans[0]
    assert len(m.scans) == 2
    assert m1.echem == 3.76
    assert m1.timestamp == "2024-02-05 10:05:00"
    assert m1.midpoint_timestamp == "2024-02-05 10:06:00"
    assert m1.voltage_min == 3.75 and m1.voltage_max == 3.77
    assert len(m.echem_df) == 40, "echem_df not reconstructed from time/@start"
    assert str(m.echem_df["timestamp"].iloc[0]) == "2024-02-05 10:00:00"
    assert abs(m.echem_df["echem_data"].iloc[5] - 3.75) < 1e-9
    assert len(m.standard_echem) == 1
    assert len(m.standard_echem[0]["data"]) == 10
    assert "current" in m.standard_echem[0]["data"].columns
    assert m.global_metadata.get("generator") == "operaxn"
    assert m.global_metadata.get("generator_version") == core.config.GENERATOR_VERSION
    assert m.global_metadata.get("total_scans") == 2
    assert m.global_metadata.get("correlation_method") == "absolute"
    assert m.data_source == "inhouse"


def test_tofnpd_reader_roundtrip(neutron_nxs):
    """Split banks remap to the tof+d model with timestamps and window extrema."""
    nm = core.load(neutron_nxs)
    n1 = nm.scans[0]
    assert n1.neutron is not None
    assert set(n1.neutron["1"].keys()) == {"tof", "d"}
    assert "e" in n1.neutron["1"]["tof"]
    assert np.allclose(n1.neutron["1"]["d"]["y"], 71.0 + np.arange(30))
    assert n1.timestamp == "2024-02-05 10:15:00"
    assert n1.neutron_start == "2024-02-05 10:00:00"
    assert n1.neutron_end == "2024-02-05 10:30:00"
    assert n1.echem == 3.85
    assert n1.voltage_min == 3.70 and n1.voltage_max == 4.00


def test_v3_backcompat(legacy_v3_nxs):
    """Schema-v3 files still load: fallback names, combined-bank split, entry attrs."""
    assert core.is_canonical_nxs(legacy_v3_nxs)
    lm = core.load(legacy_v3_nxs)
    l1 = lm.scans[0]
    assert l1.echem == 3.76 and l1.current == 0.106 and l1.exposure_time == 120.0, \
        "env values via fallback names"
    assert l1.twod_source == r"C:\raw\img.edf"
    assert l1.twod_embedded is False
    assert l1.neutron is not None
    assert set(l1.neutron["1"].keys()) == {"tof", "d"}, "combined bank not split on read"
    assert l1.neutron["1"]["tof"]["source"] == "t.dat"
    assert l1.neutron["1"]["d"]["source"] == "d.dat"
    assert np.allclose(l1.neutron["1"]["d"]["y"], np.arange(30.0) + 1)
    assert lm.echem_df is not None and len(lm.echem_df) == 10
    assert str(lm.echem_df["timestamp"].iloc[1]) == "2024-02-05 10:01:00"
    assert lm.global_metadata.get("generator_version") == "2.0.0"
    assert lm.data_source == "inhouse"


def test_overrides(override_nxs):
    """Title, sample name, and sample description overrides land in the file."""
    with h5py.File(override_nxs) as f:
        assert s(f["entry/title"]) == "My Operando Study"
        assert s(f["entry/sample/name"]) == "NMC811 pouch"
        assert s(f["entry/sample/description"]) == "NMC811/graphite single-layer pouch"


def test_cycling_protocol_written(protocol_nxs):
    """The cycling protocol lands as an NXnote with all seven datasets and
    voltage units."""
    with h5py.File(protocol_nxs) as f:
        cp = f["entry/cycling_protocol"]
        assert cp.attrs.get("NX_class") == "NXnote"
        assert set(cp.keys()) == {"technique", "voltage_window_lower",
                                  "voltage_window_upper", "C_rate", "instrument",
                                  "software", "raw_data_file"}
        assert s(cp["technique"]) == "GCPL C/10 with 1 h rest"
        assert abs(cp["voltage_window_lower"][()] - 2.8) < 1e-9
        assert cp["voltage_window_lower"].attrs.get("units") == "V"
        assert cp["voltage_window_upper"].attrs.get("units") == "V"
        assert s(cp["C_rate"]) == "C/10"
        assert s(cp["raw_data_file"]) == "doi:10.5281/zenodo.0000000"


def test_raw_data_file_defaults_to_echem_sources(tmp_path, inhouse_src):
    """An unsupplied raw_data_file records the echem files actually ingested."""
    out = str(tmp_path / "auto_raw.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE,
                             cycling_protocol={"technique": "GCPL"})
    assert ok, str(msgs)
    with h5py.File(out) as f:
        assert s(f["entry/cycling_protocol/raw_data_file"]) == "echem.txt"
    assert_pynxtools_valid(out, "auto raw_data_file")


def test_raw_data_file_uses_original_xlsx_name(tmp_path):
    """xlsx echem provenance records the user's filename, not the temp copy."""
    src = tmp_path / "xlsx_src"
    src.mkdir()
    for i, ts in enumerate(_builders.SCAN_TIMES, start=1):
        _builders.write_xrd_dat(str(src / f"scan_{i:03d}.dat"), ts,
                                y=_builders.inhouse_scan_y(i))
    _builders.write_arbin_xlsx(str(src / "cellA.xlsx"))
    out = str(tmp_path / "xlsx_raw.nxs")
    ok, msgs = core.generate([str(src)], out, core.DataSourceType.INHOUSE,
                             cycling_protocol={"technique": "GCPL"})
    assert ok, str(msgs)
    with h5py.File(out) as f:
        assert s(f["entry/cycling_protocol/raw_data_file"]) == "cellA.xlsx"
    assert_pynxtools_valid(out, "xlsx raw_data_file")


def test_standard_echem_xlsx_provenance(tmp_path, inhouse_src):
    """A standard-echem .xlsx keeps its own filename as source_file."""
    xlsx = tmp_path / "reference.xlsx"
    _builders.write_arbin_xlsx(str(xlsx), n_rows=10)
    out = str(tmp_path / "std_xlsx.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE,
                             standard_echem_files=[str(xlsx)])
    assert ok, str(msgs)
    with h5py.File(out) as f:
        group = f["entry/standard_electrochemistry/file_001"]
        assert group.attrs["source_file"] == "reference.xlsx"
        assert group["voltage"].shape == (10,)
    assert_pynxtools_valid(out, "standard-echem xlsx")


def test_sample_preparation_date(protocol_nxs):
    """sample/preparation_date is written as a tz-aware ISO timestamp."""
    with h5py.File(protocol_nxs) as f:
        pd_ds = f["entry/sample/preparation_date"]
        assert wall(pd_ds) == "2024-01-15T00:00:00"
        assert tz_aware(pd_ds), "preparation_date not tz-aware"


def test_protocol_reader_roundtrip(protocol_nxs):
    """The reader exposes the cycling protocol and preparation date in global_metadata."""
    m = core.load(protocol_nxs)
    cp = m.global_metadata.get("cycling_protocol")
    assert cp is not None, "cycling_protocol missing from global_metadata"
    assert cp.get("technique") == "GCPL C/10 with 1 h rest"
    assert cp.get("C_rate") == "C/10"
    assert abs(cp.get("voltage_window_lower") - 2.8) < 1e-9
    assert wall(m.global_metadata["sample"]["preparation_date"]) == "2024-01-15T00:00:00"


def test_cycling_protocol_absent_when_not_given(inhouse_nxs):
    """No cycling_protocol group or preparation_date appears without user input."""
    with h5py.File(inhouse_nxs) as f:
        assert "cycling_protocol" not in f["entry"]
        assert "preparation_date" not in f["entry/sample"]


def test_cycling_protocol_dropped_without_technique(tmp_path, inhouse_src):
    """A protocol without technique writes no group at all and the file stays valid."""
    scans, echem_df = core.process_raw([inhouse_src], core.DataSourceType.INHOUSE)
    out = str(tmp_path / "no_technique.nxs")
    writer = core.NXSWriter(core.DataSourceType.INHOUSE,
                            cycling_protocol={"c_rate": "C/10",
                                              "voltage_window_lower": 2.8})
    writer.write(out, scans, echem_df)
    with h5py.File(out) as f:
        assert "cycling_protocol" not in f["entry"]
    assert_pynxtools_valid(out, "no-technique")


def test_tofnpd_geometry_placeholders_skipped(tmp_path, polaris_src, monkeypatch):
    """None geometry placeholders in a profile write nothing, not even an
    empty detector or monitor group (the shipped POLARIS values are patched
    back to placeholders for the check)."""
    profile = core.config.INSTRUMENT_PROFILES["polaris"]
    monkeypatch.setitem(profile, "pre_sample_flightpath_m", None)
    monkeypatch.setitem(profile, "detector_banks", {
        bank: {"distance_m": None, "polar_angle_deg": None,
               "azimuthal_angle_deg": None}
        for bank in (1, 2, 3, 4, 5)})
    monkeypatch.setitem(profile, "monitor_mode", None)
    monkeypatch.setitem(profile, "detector_description", None)
    out = str(tmp_path / "polaris.nxs")
    ok, msgs = core.generate([polaris_src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        e = f["entry"]
        assert s(e["instrument/name"]) == "POLARIS", "polaris profile not selected"
        assert "pre_sample_flightpath" not in e
        assert "detector" not in e["instrument"], \
            "nothing to write, so no empty detector group"
        assert "monitor" not in e, "None monitor_mode must write no monitor group"


def test_polaris_profile_geometry_written(tmp_path, polaris_src):
    """The shipped POLARIS profile writes the flight path and the per-bank
    distance/2theta arrays; the ring banks carry no azimuth."""
    out = str(tmp_path / "polaris_geometry.nxs")
    ok, msgs = core.generate([polaris_src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        e = f["entry"]
        assert s(e["instrument/name"]) == "POLARIS", "polaris profile not selected"
        assert e["pre_sample_flightpath"][()] == 14.0
        assert e["pre_sample_flightpath"].attrs.get("units") == "m"
        det = e["instrument/detector"]
        assert list(det["detector_number"][()]) == [1, 2, 3, 4, 5]
        assert np.allclose(det["distance"][()], [2.248, 1.783, 1.206, 0.898, 1.251])
        assert det["distance"].attrs.get("units") == "m"
        assert np.allclose(det["polar_angle"][()],
                           [10.40, 25.99, 52.21, 92.59, 146.72])
        assert det["polar_angle"].attrs.get("units") == "degrees"
        assert "azimuthal_angle" not in det, "ring banks have no azimuth"
        assert "ZnS" in s(det["description"]), "profile detector description"
        mon = e["monitor"]
        assert mon.attrs.get("NX_class") == "NXmonitor"
        assert s(mon["mode"]) == "timer"
        assert list(mon["detector_number"][()]) == [
            611, 612, 621, 622, 631, 632, 641, 642, 651, 652, 661, 662, 671, 672]
        assert np.allclose(mon["distance"][()][:2], [7.641, 7.641])
        assert np.allclose(mon["distance"][()][-2:], [3.270, 3.270])
        assert mon["distance"].attrs.get("units") == "m"
        assert np.allclose(mon["polar_angle"][()], [180.0] * 12 + [0.0, 0.0])
        assert np.allclose(mon["azimuthal_angle"][()], 0.0)
        assert "preset" not in mon
    m = core.load(out)
    assert m.global_metadata.get("pre_sample_flightpath") == 14.0
    det_meta = m.global_metadata["instrument"]["detector"]
    assert list(det_meta["detector_number"]) == [1, 2, 3, 4, 5]
    assert m.global_metadata["monitor"]["mode"] == "timer"
    # The profile is selected case-insensitively by the harvested name
    profile = core.get_profile(core.DataSourceType.NEUTRON, "POLARIS")
    assert profile["pre_sample_flightpath_m"] == 14.0
    assert all(b["azimuthal_angle_deg"] is None
               for b in profile["detector_banks"].values())
    assert profile["monitor_mode"] == "timer" and len(profile["monitors"]) == 14
    assert_pynxtools_valid(out, "polaris geometry")


def test_tofnpd_geometry_from_profile(tmp_path, polaris_src, monkeypatch):
    """Profile geometry values become entry/pre_sample_flightpath and per-bank
    detector arrays; a bank missing one key drops only that array."""
    profile = core.config.INSTRUMENT_PROFILES["polaris"]
    monkeypatch.setitem(profile, "pre_sample_flightpath_m", 14.0)
    monkeypatch.setitem(profile, "monitors", {
        671: {"distance_m": 3.27, "polar_angle_deg": 0.0, "azimuthal_angle_deg": 0.0},
        611: {"distance_m": 7.641, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0}})
    monkeypatch.setitem(profile, "detector_banks", {
        1: {"distance_m": 1.5, "polar_angle_deg": 52.0, "azimuthal_angle_deg": 0.0},
        2: {"distance_m": 2.0, "polar_angle_deg": 92.0, "azimuthal_angle_deg": 0.0},
    })
    out = str(tmp_path / "geometry.nxs")
    ok, msgs = core.generate([polaris_src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        e = f["entry"]
        assert s(e["instrument/name"]) == "POLARIS"
        assert e["pre_sample_flightpath"][()] == 14.0
        assert e["pre_sample_flightpath"].attrs.get("units") == "m"
        det = e["instrument/detector"]
        assert list(det["detector_number"][()]) == [1, 2]
        assert np.allclose(det["distance"][()], [1.5, 2.0])
        assert det["distance"].attrs.get("units") == "m"
        assert np.allclose(det["polar_angle"][()], [52.0, 92.0])
        assert det["polar_angle"].attrs.get("units") == "degrees"
        assert np.allclose(det["azimuthal_angle"][()], [0.0, 0.0])
        assert det["azimuthal_angle"].attrs.get("units") == "degrees"
        assert list(e["monitor/detector_number"][()]) == [611, 671], "sorted by number"
        assert np.allclose(e["monitor/distance"][()], [7.641, 3.27])
        assert e["monitor/distance"].attrs.get("units") == "m"
        assert np.allclose(e["monitor/polar_angle"][()], [180.0, 0.0])
        assert list(e["monitor"].keys()) == [
            "mode", "detector_number", "distance", "polar_angle", "azimuthal_angle"]
    m = core.load(out)
    assert m.global_metadata.get("pre_sample_flightpath") == 14.0
    assert list(m.global_metadata["monitor"]["detector_number"]) == [611, 671]
    assert_pynxtools_valid(out, "geometry")

    # Partial: bank 2 lacks a distance -> distance array skipped, others kept
    monkeypatch.setitem(profile, "detector_banks", {
        1: {"distance_m": 1.5, "polar_angle_deg": 52.0, "azimuthal_angle_deg": 0.0},
        2: {"distance_m": None, "polar_angle_deg": 92.0, "azimuthal_angle_deg": 0.0},
    })
    core.clear_global_cache()
    out2 = str(tmp_path / "geometry_partial.nxs")
    ok, msgs = core.generate([polaris_src], out2, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out2) as f:
        det = f["entry/instrument/detector"]
        assert "distance" not in det
        assert np.allclose(det["polar_angle"][()], [52.0, 92.0])
        assert list(det["detector_number"][()]) == [1, 2]


def test_xrd_data_group_omitted_on_read_failure(tmp_path, inhouse_src, monkeypatch):
    """A failed 1D read leaves no empty NXdata shell behind."""
    def _boom(cls, *args, **kwargs):
        raise IOError("simulated read failure")
    monkeypatch.setattr(core.nxs_writer.DataReaderFactory, "read_file",
                        classmethod(_boom))
    out = str(tmp_path / "read_failure.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        names = [k for k in f["entry"] if k.startswith("scan_")]
        assert names
        for name in names:
            assert "data" not in f["entry"][name], f"{name} carries an empty data group"
        empty_nxdata = []
        def _visit(path, obj):
            if isinstance(obj, h5py.Group) and obj.attrs.get("NX_class") == "NXdata" \
                    and len(obj.keys()) == 0:
                empty_nxdata.append(path)
        f.visititems(_visit)
        assert not empty_nxdata, empty_nxdata


def test_definition_version():
    """The definition version advertised in files is 1.0.0."""
    assert core.config.DEFINITION_VERSION == "1.0.0"


@pytest.mark.parametrize("nxs_fixture", ["inhouse_nxs", "neutron_nxs", "protocol_nxs"])
def test_pynxtools_validation(request, nxs_fixture):
    """pynxtools validate_nexus declares the generated files valid."""
    assert_pynxtools_valid(request.getfixturevalue(nxs_fixture), nxs_fixture)


def test_profile_from_mantid_nexus(tmp_path):
    """L1 from the IDF, per-bank means and ranges from the focussed
    positions; a ring gets no azimuth, a localised panel does."""
    path = str(tmp_path / "TESTINST1-2.nxs")
    _builders.write_mantid_processed_file(path)
    p = core.profile_from_mantid_nexus(path)
    assert p["instrument_name"] == "TESTINST"
    assert p["pre_sample_flightpath_m"] == 14.0
    assert sorted(p["detector_banks"]) == [1, 2]
    assert p["monitor_mode"] is None
    assert p["monitors"] == {
        11: {"distance_m": 7.6, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
        12: {"distance_m": 7.6, "polar_angle_deg": 180.0, "azimuthal_angle_deg": 0.0},
        21: {"distance_m": 3.3, "polar_angle_deg": 0.0, "azimuthal_angle_deg": 0.0}}

    ring = p["detector_banks"][1]
    assert ring["distance_m"] == 2.25 and ring["polar_angle_deg"] == 10.0
    assert ring["azimuthal_angle_deg"] is None, "a ring has no azimuth"
    panel = p["detector_banks"][2]
    assert panel["distance_m"] == 1.0 and panel["polar_angle_deg"] == 90.0
    assert panel["azimuthal_angle_deg"] == 90.0

    info = p["info"]
    assert info["banks"][1]["elements"] == 8
    assert info["banks"][1]["polar_angle_range_deg"] == (10.0, 10.0)
    assert info["banks"][2]["distance_range_m"] == (1.0, 1.0)
    assert info["banks"][2]["polar_angle_range_deg"] == (88.0, 92.0)
    assert info["banks"][1]["azimuthal_coherence"] < 0.5 \
        <= info["banks"][2]["azimuthal_coherence"]
    assert "monitors" not in info, "the monitor table is a written key, not info"
    assert info["idf_name"] == "Test_upgrade"
    assert info["idf_valid_from"] == "2020-01-01 00:00:00"
    assert info["source_type_name"] == "H2O_moderator"
    assert info["mantid_version"] == "6.16.1"
    assert p["provenance"] == (
        "Mantid IDF Test_upgrade (valid-from 2020-01-01 00:00:00), "
        "read from TESTINST1-2.nxs (Mantid 6.16.1)")


def test_format_profile_block_is_paste_ready(tmp_path):
    """The printed block is a valid INSTRUMENT_PROFILES entry with the
    informational values as comments."""
    path = str(tmp_path / "TESTINST1-2.nxs")
    _builders.write_mantid_processed_file(path)
    block = core.format_profile_block(core.profile_from_mantid_nexus(path))
    entry = ast.literal_eval("{" + block + "}")
    assert entry["testinst"]["instrument_name"] == "TESTINST"
    assert entry["testinst"]["pre_sample_flightpath_m"] == 14.0
    assert entry["testinst"]["detector_banks"][1]["azimuthal_angle_deg"] is None
    assert entry["testinst"]["detector_banks"][2]["azimuthal_angle_deg"] == 90.0
    assert "# bank 1: 8 elements, L2 2.25-2.25 m, 2theta 10.0-10.0 deg" in block
    assert entry["testinst"]["monitor_mode"] is None
    assert sorted(entry["testinst"]["monitors"]) == [11, 12, 21]
    assert entry["testinst"]["monitors"][21] == {
        "distance_m": 3.3, "polar_angle_deg": 0.0, "azimuthal_angle_deg": 0.0}
    assert "# 2theta 180 is upstream of the sample, 0 downstream" in block


def test_profile_from_non_mantid_file(tmp_path):
    """A file without mantid_workspace_N entries is rejected, not guessed."""
    path = str(tmp_path / "plain.nxs")
    with h5py.File(path, "w") as f:
        f.create_group("entry")
    with pytest.raises(ValueError):
        core.profile_from_mantid_nexus(path)


def test_wish_profile_minimal(tmp_path):
    """WISH ships source and name only: the file names ISIS as the source and
    carries no detector, flight path or monitor group."""
    profile = core.get_profile(core.DataSourceType.NEUTRON, "WISH")
    assert profile["source_name"] == "ISIS" and profile["instrument_name"] == "WISH"
    assert not any(k in profile for k in ("pre_sample_flightpath_m", "detector_banks",
                                          "monitors", "monitor_mode")), str(profile)
    src = tmp_path / "wish"
    _builders.write_detailed_neutron_dir(str(src))
    for dat in src.glob("*.dat"):
        dat.write_text(dat.read_text().replace("Instrument POLARIS", "Instrument WISH"))
    nxs = str(tmp_path / "wish.nxs")
    ok, msgs = core.generate([str(src)], nxs, core.DataSourceType.NEUTRON)
    assert ok, msgs
    with h5py.File(nxs) as f:
        e = f["entry"]
        assert s(e["instrument/name"]) == "WISH"
        assert s(e["instrument/source/name"]) == "ISIS"
        assert "detector" not in e["instrument"] and "monitor" not in e
        assert "pre_sample_flightpath" not in e
    assert_pynxtools_valid(nxs, "WISH profile")


def test_synchrotron_collection_keeps_count_time(tmp_path):
    """The synchrotron_metadata collection keeps the detector count_time (a
    setting) while the per-scan start/end times and scan identifier stay out."""
    src = str(tmp_path / "synchrotron")
    os.makedirs(src)
    _builders.write_synchrotron_scan(src, "000001", "2024-02-05T10:05:00",
                                     "2024-02-05T10:05:39", count_time=30.0)
    _builders.write_echem_txt(os.path.join(src, "echem.txt"))
    nxs = str(tmp_path / "synchrotron.nxs")
    ok, msgs = core.generate([src], nxs, core.DataSourceType.SYNCHROTRON)
    assert ok, msgs
    with h5py.File(nxs) as f:
        coll = f["entry/instrument/synchrotron_metadata"]
        assert coll["entry1_instrument_pixium_hdf_count_time"][()] == 30.0, list(coll)
        assert "entry1_start_time" not in coll and "entry1_end_time" not in coll, list(coll)
        assert f["entry/scan_000001/environment/exposure_time"][()] == 39.0


def test_reader_accepts_pre_rename_errors_name(tmp_path, inhouse_src):
    """Files written before the data_errors name carry NXdata's deprecated
    `errors`; the reader still returns the uncertainties as the trace's e."""
    nxs = str(tmp_path / "old_name.nxs")
    ok, msgs = core.generate([inhouse_src], nxs, core.DataSourceType.INHOUSE)
    assert ok, msgs
    with h5py.File(nxs, "r+") as f:
        f["entry/scan_000001/data"].move("data_errors", "errors")
    m = core.load(nxs)
    assert "e" in m.scans[0].oned and len(m.scans[0].oned["e"]) == 30

    src = str(tmp_path / "neutron_old")
    _builders.write_neutron_dir(src)
    nxs2 = str(tmp_path / "old_name_neutron.nxs")
    ok, msgs = core.generate([src], nxs2, core.DataSourceType.NEUTRON)
    assert ok, msgs
    with h5py.File(nxs2, "r+") as f:
        f["entry/scan_000001/bank_1"].move("data_errors", "errors")
    m2 = core.load(nxs2)
    traces = [t for bk in m2.scans[0].neutron.values() for t in bk.values()]
    assert traces and all("e" in t for t in traces), [sorted(t) for t in traces]


def test_is_canonical_requires_operaxn_provenance(tmp_path, inhouse_nxs):
    """Generic NeXus members do not make a file ours: a raw facility file
    with /entry/definition and /entry/process is left to generation, while
    OperaXN provenance in any of its forms (an NXoperando definition,
    process/program operaxn, the pre-17-Sep program_name, or scan_
    subentries) is recognised."""
    def nxs_with(name, build):
        path = str(tmp_path / f"{name}.nxs")
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            build(entry)
        return path

    def foreign(entry):
        entry["definition"] = "NXmonopd"
        process = entry.create_group("process")
        process.attrs["NX_class"] = "NXprocess"
        process["program"] = "Mantid"
        process["version"] = "6.16.1"

    def our_definition(entry):
        entry["definition"] = "NXoperando_tofnpd"

    def our_process(entry):
        entry.create_group("process")["program"] = "operaxn"

    def pre_rename(entry):
        entry["program_name"] = "operaxn-core"

    def scans_only(entry):
        entry.create_group("scan_000001").attrs["NX_class"] = "NXsubentry"

    assert not core.is_canonical_nxs(nxs_with("foreign", foreign))
    assert not core.is_canonical_nxs(nxs_with("empty", lambda entry: None))
    for name, build in (("definition", our_definition), ("process", our_process),
                        ("program_name", pre_rename), ("scans", scans_only)):
        assert core.is_canonical_nxs(nxs_with(name, build)), name
    assert core.is_canonical_nxs(inhouse_nxs)
