"""Schema v4 provenance: reader round-trips, overrides, v3 back-compat, validation."""
import h5py
import numpy as np
import pytest

import core

from _helpers import s, wall, tz_aware, assert_pynxtools_valid


def test_monopd_reader_roundtrip(inhouse_nxs):
    """A generated file loads back with the same shapes and the v4
    fields (window, indices, echem_df, metadata) populated."""
    m = core.load(inhouse_nxs)
    m1 = m.scans[0]
    assert len(m.scans) == 2
    assert m1.echem == 3.76
    assert m1.timestamp == "2024-02-05 10:05:00"
    assert m1.midpoint_timestamp == "2024-02-05 10:06:00"
    assert m1.voltage_min == 3.75 and m1.voltage_max == 3.77
    assert m1.echem_index_start == 5 and m1.echem_index_end == 7
    assert len(m.echem_df) == 40, "echem_df not reconstructed from time/@start"
    assert str(m.echem_df["timestamp"].iloc[0]) == "2024-02-05 10:00:00"
    assert abs(m.echem_df["echem_data"].iloc[5] - 3.75) < 1e-9
    assert len(m.standard_echem) == 1
    assert len(m.standard_echem[0]["data"]) == 10
    assert "current" in m.standard_echem[0]["data"].columns
    assert m.global_metadata.get("generator") == "operaxn-core"
    assert m.global_metadata.get("generator_version") == core.config.GENERATOR_VERSION
    assert m.global_metadata.get("total_scans") == 2
    assert m.global_metadata.get("correlation_method") == "absolute"
    assert m.data_source == "inhouse"


def test_tofnpd_reader_roundtrip(neutron_nxs):
    """Split banks remap to the tof+d model with timestamps and window summary."""
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


def test_capacity_roundtrip(tmp_path, inhouse_src):
    """A set Scan.capacity is written with mAh units, read back, and validates."""
    scans, echem_df = core.process_raw([inhouse_src], core.DataSourceType.INHOUSE)
    scans[0].capacity = 1.25
    out = str(tmp_path / "capacity.nxs")
    core.NXSWriter(core.DataSourceType.INHOUSE).write(out, scans, echem_df)
    with h5py.File(out) as f:
        cap = f["entry/scan_000001/environment/capacity"]
        assert abs(cap[()] - 1.25) < 1e-9
        assert cap.attrs.get("units") == "mAh"
        assert "capacity" not in f["entry/scan_000002/environment"]
    m = core.load(out)
    assert m.scans[0].capacity == 1.25
    assert m.scans[1].capacity is None
    assert_pynxtools_valid(out, "capacity")


def test_tofnpd_geometry_placeholders_skipped(tmp_path, polaris_src):
    """The POLARIS profile's None geometry placeholders write nothing."""
    out = str(tmp_path / "polaris.nxs")
    ok, msgs = core.generate([polaris_src], out, core.DataSourceType.NEUTRON)
    assert ok, str(msgs)
    with h5py.File(out) as f:
        e = f["entry"]
        assert s(e["instrument/name"]) == "POLARIS", "polaris profile not selected"
        assert "pre_sample_flightpath" not in e
        det = e["instrument/detector"]
        for name in ("detector_number", "distance", "polar_angle", "azimuthal_angle"):
            assert name not in det, f"placeholder geometry {name} was written"


def test_tofnpd_geometry_from_profile(tmp_path, polaris_src, monkeypatch):
    """Profile geometry values become entry/pre_sample_flightpath and per-bank
    detector arrays; a bank missing one key drops only that array."""
    profile = core.profiles.INSTRUMENT_PROFILES["polaris"]
    monkeypatch.setitem(profile, "pre_sample_flightpath_m", 14.0)
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
    m = core.load(out)
    assert m.global_metadata.get("pre_sample_flightpath") == 14.0
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
