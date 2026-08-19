"""NXDL definitions validated against the XSD pinned in schema/README.md."""
from pathlib import Path

import pytest

lxml = pytest.importorskip("lxml")
from lxml import etree  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schema" / "nxdl.xsd"
NXDL_FILES = sorted((REPO / "definitions").glob("*.nxdl.xml"))
PLUGIN_BUNDLE = (REPO / "pynxtools-operaxn" / "src" / "pynxtools_operaxn"
                 / "nexus_definitions" / "applications")
NS = {"nx": "http://definition.nexusformat.org/nxdl/3.1"}


@pytest.fixture(scope="module")
def xsd():
    """Compiled nxdl.xsd schema."""
    return etree.XMLSchema(etree.parse(str(SCHEMA)))


@pytest.mark.parametrize("filename", ["nxdl.xsd", "nxdlTypes.xsd"])
def test_schema_file_present(filename):
    """The vendored NXDL XSDs are present under schema/."""
    assert (REPO / "schema" / filename).is_file(), f"schema/{filename} missing"


def test_definitions_found():
    """At least the two NXoperando application definitions exist."""
    assert len(NXDL_FILES) >= 2, f"found {len(NXDL_FILES)} in {REPO / 'definitions'}"


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_well_formed_xml(path):
    """Each definition parses as XML; a failure here is a syntax error."""
    etree.parse(str(path))


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_valid_against_schema(path, xsd):
    """Each definition obeys the NXDL meta-schema; failures report the
    full XSD error log."""
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
    """Each definition declares category='application' so validators
    treat it as a strict contract, not a base class."""
    assert etree.parse(str(path)).getroot().get("category") == "application"


@pytest.mark.parametrize("path", NXDL_FILES, ids=lambda p: p.name)
def test_plugin_bundle_in_sync(path):
    """The local pynxtools-operaxn bundle carries a byte-identical copy of
    each repo definition."""
    if not PLUGIN_BUNDLE.is_dir():
        pytest.skip("pynxtools-operaxn plugin checkout not present")
    bundled = PLUGIN_BUNDLE / path.name
    assert bundled.is_file(), f"{bundled} missing"
    assert bundled.read_bytes() == path.read_bytes(), (
        f"{path.name} differs between definitions/ and the plugin bundle")


def _one(root, xpath):
    """Exactly one element matching xpath (namespace-aware) under root."""
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
    cap = _one(env, "nx:field[@name='capacity']")
    assert (cap.get("type"), cap.get("units"), cap.get("optional")) == (
        "NX_FLOAT", "NX_CHARGE", "true")
    assert _required(_one(env, "nx:field[@name='scan_timestamp']"))

    assert _required(_one(entry, "nx:field[@name='program_name']"))
    process = _one(entry, "nx:group[@name='process']")
    assert _required(process)
    assert _required(_one(process, "nx:field[@name='correlation_method']"))
    assert _required(_one(scan, "nx:group[@name='monitor'][@type='NXmonitor']"))
    assert _one(entry, "nx:field[@name='end_time']").get("optional") == "true"
    assert not root.xpath("//nx:link", namespaces=NS)

    if root.get("name") == "NXoperando_tofnpd":
        fp = _one(entry, "nx:field[@name='pre_sample_flightpath']")
        assert (fp.get("type"), fp.get("units"), fp.get("optional")) == (
            "NX_FLOAT", "NX_LENGTH", "true")
        det = _one(entry, "nx:group[@name='instrument']/nx:group[@name='detector']")
        for name, typ, units in (("detector_number", "NX_INT", None),
                                 ("distance", "NX_FLOAT", "NX_LENGTH"),
                                 ("polar_angle", "NX_FLOAT", "NX_ANGLE"),
                                 ("azimuthal_angle", "NX_FLOAT", "NX_ANGLE")):
            f = _one(det, f"nx:field[@name='{name}']")
            assert (f.get("type"), f.get("units"), f.get("optional")) == (
                typ, units, "true"), name
            dim = _one(f, "nx:dimensions[@rank='1']/nx:dim")
            assert dim.get("value") == "nBank"
        _one(root, "nx:symbols/nx:symbol[@name='nBank']")
