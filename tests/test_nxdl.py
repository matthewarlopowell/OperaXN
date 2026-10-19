"""NXDL definitions: validity against the XSD pinned in schema/README.md, plus Figure-4 elements."""
from pathlib import Path

import pytest

pytest.importorskip("lxml")
from lxml import etree  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schema" / "nxdl.xsd"
NXDL_FILES = sorted((REPO / "definitions").glob("*.nxdl.xml"))
NS = {"nx": "http://definition.nexusformat.org/nxdl/3.1"}


@pytest.fixture(scope="module")
def xsd():
    """Return the compiled nxdl.xsd schema."""
    return etree.XMLSchema(etree.parse(str(SCHEMA)))


def test_definitions_found():
    """At least the two NXoperando application definitions exist."""
    assert len(NXDL_FILES) >= 2, f"found {len(NXDL_FILES)} in {REPO / 'definitions'}"


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_valid_against_schema(path, xsd):
    """Each definition obeys the NXDL meta-schema; failures report the full XSD error log."""
    doc = etree.parse(str(path))
    ok = xsd.validate(doc)
    assert ok, "; ".join(str(e) for e in xsd.error_log)


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_filename_matches_definition_name(path):
    """The file is named <definition name>.nxdl.xml."""
    def_name = etree.parse(str(path)).getroot().get("name", "")
    assert path.name == f"{def_name}.nxdl.xml", f"definition name is {def_name!r}"


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_category_is_application(path):
    """Each definition declares category='application': a strict contract, not a base class."""
    assert etree.parse(str(path)).getroot().get("category") == "application"


def _one(root, xpath):
    """Return the one element under root matching xpath (namespace-aware); assert it is unique."""
    hits = root.xpath(xpath, namespaces=NS)
    assert len(hits) == 1, f"{xpath}: {len(hits)} matches"
    return hits[0]


def _required(el):
    """True when el carries neither optional nor recommended attributes."""
    return el.get("optional") is None and el.get("recommended") is None


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_figure4_elements_present(path):
    """Each definition declares the published Figure-4 elements and optionality."""
    root = etree.parse(str(path)).getroot()
    entry = _one(root, "nx:group[@type='NXentry']")

    proto = _one(entry, "nx:group[@name='cycling_protocol'][@type='NXnote']")
    assert proto.get("recommended") == "true"
    assert _required(_one(proto, "nx:field[@name='technique']"))
    for name in ("voltage_window_lower", "voltage_window_upper", "C_rate",
                 "instrument", "software", "raw_data_file"):
        assert _one(proto, f"nx:field[@name='{name}']").get("optional") == "true"
    for name in ("voltage_window_lower", "voltage_window_upper"):
        f = _one(proto, f"nx:field[@name='{name}']")
        assert (f.get("type"), f.get("units")) == ("NX_FLOAT", "NX_VOLTAGE")

    prep = _one(entry, "nx:group[@name='sample']/nx:field[@name='preparation_date']")
    assert (prep.get("type"), prep.get("optional")) == ("NX_DATE_TIME", "true")

    scan = _one(entry, "nx:group[@name='SCAN'][@type='NXsubentry']")
    env = _one(scan, "nx:group[@name='environment']")
    # Window extrema are declared optional (Figure 4 requires them); the
    # index pointers, log slices and capacity are not declared at all.
    for name, units in (("voltage_min", "NX_VOLTAGE"), ("voltage_max", "NX_VOLTAGE"),
                        ("current_min", "NX_CURRENT"), ("current_max", "NX_CURRENT")):
        f = _one(env, f"nx:field[@name='{name}']")
        assert (f.get("type"), f.get("units"), f.get("optional")) == (
            "NX_FLOAT", units, "true"), name
    for name in ("capacity", "echem_index_first", "echem_index_last"):
        assert not env.xpath(f"nx:field[@name='{name}']", namespaces=NS), name
    assert not env.xpath("nx:group[@type='NXlog']", namespaces=NS)
    assert _required(_one(env, "nx:field[@name='scan_timestamp']"))

    assert not entry.xpath("nx:field[@name='program_name']", namespaces=NS), \
        "program_name duplicates process/program"
    process = _one(entry, "nx:group[@name='process']")
    assert _required(process)
    assert _required(_one(process, "nx:field[@name='correlation_method']"))
    # The correlation tolerance is a recommended scalar of the echem record
    # it bounds (an OperaXN addition, so a Figure 4-only file still
    # validates), not a process field.
    echem = _one(entry, "nx:group[@name='operando_electrochemistry'][@type='NXdata']")
    tolerance = _one(echem, "nx:field[@name='echem_time_tolerance']")
    assert (tolerance.get("type"), tolerance.get("units"),
            tolerance.get("recommended")) == ("NX_FLOAT", "NX_TIME", "true")
    assert not tolerance.xpath("nx:dimensions", namespaces=NS)
    assert not process.xpath("nx:field[@name='echem_time_tolerance']", namespaces=NS)
    # A subentry holds its environment (with the acquisition window) and its
    # data: no definition/title/time fields, no declared link. Only monopd
    # declares a per-scan monitor (the integral of an acquisition with a
    # beam-monitor reading), optional; mode and preset sit at entry level.
    for name in ("definition", "title", "start_time", "end_time"):
        assert not scan.xpath(f"nx:field[@name='{name}']", namespaces=NS), name
    scan_monitors = scan.xpath("nx:group[@type='NXmonitor']", namespaces=NS)
    if root.get("name") == "NXoperando_monopd":
        assert len(scan_monitors) == 1 and scan_monitors[0].get("optional") == "true"
        assert _required(_one(scan_monitors[0], "nx:field[@name='integral']"))
        assert not scan_monitors[0].xpath("nx:field[@name='mode']", namespaces=NS)
    else:
        assert not scan_monitors
    assert _required(_one(env, "nx:field[@name='start_time']"))
    assert _one(env, "nx:field[@name='end_time']").get("recommended") == "true"
    assert _one(entry, "nx:field[@name='end_time']").get("optional") == "true"
    assert not root.xpath("//nx:link", namespaces=NS)

    mon = _one(entry, "nx:group[@name='monitor'][@type='NXmonitor']")
    assert mon.get("optional") == "true"
    mode = _one(mon, "nx:field[@name='mode']")
    assert _required(mode)
    if root.get("name") == "NXoperando_tofnpd":
        fp = _one(entry, "nx:field[@name='pre_sample_flightpath']")
        assert (fp.get("type"), fp.get("units"), fp.get("optional")) == (
            "NX_FLOAT", "NX_LENGTH", "true")
        det = _one(entry, "nx:group[@name='instrument']/nx:group[@name='detector']")
        for group, symbol in ((det, "nBank"), (mon, "nMon")):
            for name, typ, units in (("detector_number", "NX_INT", None),
                                     ("distance", "NX_FLOAT", "NX_LENGTH"),
                                     ("polar_angle", "NX_FLOAT", "NX_ANGLE"),
                                     ("azimuthal_angle", "NX_FLOAT", "NX_ANGLE")):
                f = _one(group, f"nx:field[@name='{name}']")
                assert (f.get("type"), f.get("units"), f.get("optional")) == (
                    typ, units, "true"), name
                assert _one(f, "nx:dimensions[@rank='1']/nx:dim").get("value") == symbol
            _one(root, f"nx:symbols/nx:symbol[@name='{symbol}']")
        assert [i.get("value") for i in
                mode.xpath("nx:enumeration/nx:item", namespaces=NS)] == ["monitor", "timer"]
        for name in ("preset", "integral", "data", "time_of_flight"):
            assert not mon.xpath(f"nx:field[@name='{name}']", namespaces=NS), name
    else:
        preset = _one(mon, "nx:field[@name='preset']")
        assert (preset.get("type"), preset.get("optional")) == ("NX_FLOAT", "true")
        assert not mon.xpath("nx:field[@name='integral']", namespaces=NS)
