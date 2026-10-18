"""Assertion and skip helpers shared by the test suite.

The writer stamps timestamps with the generating machine's local UTC
offset; the reader returns naive local wall time. Assertions therefore
compare wall-clock strings only, never raw ISO strings.
"""
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

import core


def need_dataset(relpath):
    """Return a realdata path under OPERAXN_TEST_DATA, skipping when absent."""
    root = os.environ.get("OPERAXN_TEST_DATA")
    if not root:
        pytest.skip("OPERAXN_TEST_DATA not set")
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        pytest.skip(f"{relpath} not present under {root}")
    return path


needs_fabio = pytest.mark.skipif(not core.FABIO_AVAILABLE,
                                 reason="fabio not installed")


def s(dataset):
    """Return a scalar string dataset's value, decoded from bytes."""
    v = dataset[()]
    return v.decode() if isinstance(v, bytes) else v


def _iso(value):
    """Return a stored ISO dataset or attribute value, decoded from bytes."""
    if hasattr(value, "attrs"):
        value = value[()]
    return value.decode() if isinstance(value, bytes) else value


def wall(value):
    """Return a stored ISO value's naive wall-clock 'YYYY-MM-DDTHH:MM:SS'."""
    t = pd.to_datetime(_iso(value))
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def tz_aware(value):
    """True when a stored ISO value carries a UTC offset."""
    return pd.to_datetime(_iso(value)).tzinfo is not None


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _same_bytes(a, b):
    """True when files a and b both exist with identical contents."""
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def _register_definitions():
    """Copy definitions/*.nxdl.xml into pynxtools; return False on failure.
    Repo copies are authoritative and overwrite site-packages when bytes differ;
    validate_nexus crashes on an unresolvable definition, so callers skip on False."""
    try:
        import pynxtools
        dest = os.path.join(os.path.dirname(pynxtools.__file__),
                            "definitions", "contributed_definitions")
        if not os.path.isdir(dest):
            return False
        src_dir = os.path.join(_REPO, "definitions")
        for name in os.listdir(src_dir):
            if name.endswith(".nxdl.xml"):
                src = os.path.join(src_dir, name)
                target = os.path.join(dest, name)
                if not _same_bytes(src, target):
                    shutil.copy2(src, target)
        return True
    except Exception:
        return False


def pynxtools_verdict(path):
    """Return validate_nexus stdout+stderr for path, or None when unavailable.
    None means no CLI, unregistrable definitions or a 300 s timeout. The CLI is
    sought beside sys.executable first (unactivated venvs), then on PATH."""
    exe = os.path.join(os.path.dirname(sys.executable),
                       "validate_nexus.exe" if os.name == "nt" else "validate_nexus")
    if not os.path.isfile(exe):
        exe = shutil.which("validate_nexus")
    if not exe or not _register_definitions():
        return None
    try:
        result = subprocess.run([exe, path], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    return result.stdout + result.stderr


def assert_pynxtools_valid(path, tag=""):
    """Assert validate_nexus declares path valid; skip when validation is unavailable.
    OPERAXN_REQUIRE_VALIDATION (set on every CI leg) turns that skip into a failure."""
    out = pynxtools_verdict(path)
    if out is None:
        if os.environ.get("OPERAXN_REQUIRE_VALIDATION"):
            pytest.fail("pynxtools validation unavailable but "
                        "OPERAXN_REQUIRE_VALIDATION is set")
        pytest.skip("pynxtools validation unavailable")
    tail = out.strip().splitlines()[-1] if out.strip() else "no output"
    assert "is valid" in out, (f"{tag}: {tail}" if tag else tail)
