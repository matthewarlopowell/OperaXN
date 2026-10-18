"""Shared fixtures over synthetic tmp_path datasets.

Imports prefer the installed package and fall back to bin/ for bare checkouts.
"""
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import core
except ImportError:
    # Bare dev checkout with no installed package: import from bin/. When the
    # package is installed (CI runs `pip install .[test]`) the installed
    # packages are used, so the wheel's bin->core/operaxn mapping stays tested.
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    import core

import h5py
import numpy as np
import pytest

import _builders


@pytest.fixture(scope="session", autouse=True)
def _private_tempdir(tmp_path_factory):
    """Point tempfile, and so the session cache, at a pytest folder, never the user's real cache."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("systemp"))
    yield
    tempfile.tempdir = previous


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Clear the path/size/mtime-keyed reader cache around each test, as tmp_paths get reused."""
    core.clear_global_cache()
    yield
    core.clear_global_cache()


@pytest.fixture
def tk_root():
    """Yield a withdrawn Tk root, retrying Tk() startup (flaky on busy displays) before skipping."""
    tk = pytest.importorskip("tkinter")
    root = None
    for _ in range(3):
        try:
            root = tk.Tk()
            root.withdraw()
            break
        except Exception:
            time.sleep(0.5)
    if root is None:
        pytest.skip("no display available")
    yield root
    root.destroy()


@pytest.fixture
def inhouse_dir(tmp_path):
    """Return a dir of 3 XRD scans at 10:05/10:10/10:15 + echem (60 s cadence from 10:00,
    V = 3.7 + 0.01/step). Known-good: scan 1 midpoint 10:06 -> V = 3.76."""
    d = tmp_path / "inhouse"
    _builders.write_inhouse_dir(str(d))
    return str(d)


@pytest.fixture
def neutron_dir(tmp_path):
    """Return a dir of a 10:00-10:30 logbook + 2 banks x (TOF, d) + echem.
    Known-good: midpoint 10:15 -> V = 3.85."""
    d = tmp_path / "neutron"
    _builders.write_neutron_dir(str(d))
    return str(d)


# Schema fixtures over the detailed builders. The module-scoped ones are built
# once per consuming test module, so the two schema modules each get their own.


@pytest.fixture(scope="module")
def inhouse_src(tmp_path_factory):
    """Return the detailed in-house source directory, shared across the module."""
    d = str(tmp_path_factory.mktemp("schema_src") / "inhouse")
    _builders.write_detailed_inhouse_dir(d)
    return d


@pytest.fixture(scope="module")
def inhouse_nxs(tmp_path_factory, inhouse_src):
    """Return a generated NXoperando_monopd file with a standard-echem sidecar."""
    base = tmp_path_factory.mktemp("schema_inhouse")
    std = str(base / "standard.txt")
    _builders.write_echem_txt(std, n_rows=10)
    out = str(base / "inhouse.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE,
                             standard_echem_files=[std])
    assert ok, f"in-house generation failed: {msgs}"
    return out


@pytest.fixture(scope="module")
def neutron_nxs(tmp_path_factory):
    """Return a generated NXoperando_tofnpd file from the Mantid-style fixture."""
    base = tmp_path_factory.mktemp("schema_neutron")
    src = str(base / "neutron")
    _builders.write_detailed_neutron_dir(src)
    out = str(base / "neutron.nxs")
    ok, msgs = core.generate([src], out, core.DataSourceType.NEUTRON)
    assert ok, f"neutron generation failed: {msgs}"
    return out


@pytest.fixture(scope="module")
def override_nxs(tmp_path_factory, inhouse_src):
    """Return a generated file with title/sample_name/sample_description overrides."""
    out = str(tmp_path_factory.mktemp("schema_override") / "override.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE,
                             title="My Operando Study", sample_name="NMC811 pouch",
                             sample_description="NMC811/graphite single-layer pouch")
    assert ok, f"override generation failed: {msgs}"
    return out


@pytest.fixture(scope="module")
def protocol_nxs(tmp_path_factory, inhouse_src):
    """Return a generated file with a sample preparation date and full cycling protocol."""
    out = str(tmp_path_factory.mktemp("schema_protocol") / "protocol.nxs")
    ok, msgs = core.generate([inhouse_src], out, core.DataSourceType.INHOUSE,
                             sample_preparation_date="2024-01-15",
                             cycling_protocol=dict(_builders.CYCLING_PROTOCOL))
    assert ok, f"protocol generation failed: {msgs}"
    return out


@pytest.fixture
def polaris_src(tmp_path):
    """Return a neutron source dir whose Mantid headers name POLARIS (selects the profile)."""
    d = str(tmp_path / "polaris")
    _builders.write_detailed_neutron_dir(d)
    return d


@pytest.fixture(scope="module")
def legacy_v3_nxs(tmp_path_factory):
    """Return a hand-built schema-v3 file: legacy attrs, unit-suffixed names, combined banks."""
    path = str(tmp_path_factory.mktemp("schema_v3") / "legacy_v3.nxs")
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.attrs["generator"] = "operaxn-core"
        e.attrs["generator_version"] = "2.0.0"
        e.attrs["data_source"] = "inhouse"
        e.attrs["correlation_method"] = "absolute"
        e.attrs["total_scans"] = 1
        sub = e.create_group("scan_000001")
        sub.attrs["NX_class"] = "NXsubentry"
        sub.attrs["scan_number"] = 1
        sub.create_dataset("definition", data="NXmonopd")
        sub.create_dataset("start_time", data="2024-02-05 10:05:00")
        env = sub.create_group("environment")
        env.attrs["NX_class"] = "NXcollection"
        env.create_dataset("scan_timestamp", data="2024-02-05 10:05:00")
        env.create_dataset("voltage (V)", data=3.76)
        env.create_dataset("current (mA)", data=0.106)
        env.create_dataset("exposure_time", data=120.0)
        d = sub.create_group("data")
        d.attrs["NX_class"] = "NXdata"
        d.attrs["signal"] = "data"
        d.attrs["axes"] = "polar_angle"
        d.attrs["twod_source"] = "img.edf"
        d.attrs["twod_source_path"] = r"C:\raw\img.edf"
        d.attrs["twod_embedded"] = False
        d.create_dataset("polar_angle", data=_builders.DETAILED_ONED_X)
        d.create_dataset("data", data=np.arange(30.0))
        b = sub.create_group("bank_1")
        b.attrs["NX_class"] = "NXdata"
        b.attrs["tof_source_file"] = "t.dat"
        b.attrs["d_source_file"] = "d.dat"
        b.create_dataset("time_of_flight", data=_builders.DETAILED_TOF_X)
        b.create_dataset("data", data=np.arange(30.0))
        b.create_dataset("d_spacing", data=_builders.DETAILED_D_X)
        b.create_dataset("d_data", data=np.arange(30.0) + 1)
        oe = e.create_group("operando_electrochemistry")
        oe.attrs["NX_class"] = "NXdata"
        ts = np.array([f"2024-02-05 10:{mm:02d}:00" for mm in range(10)], dtype=object)
        oe.create_dataset("timestamps", data=ts, dtype=h5py.string_dtype())
        oe.create_dataset("voltage (V)", data=3.7 + 0.01 * np.arange(10))
        oe.create_dataset("current (mA)", data=0.1 + 0.001 * np.arange(10))
    return path
