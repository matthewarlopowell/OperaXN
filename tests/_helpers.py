"""Assertion helpers for NeXus file checks.

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


def s(dataset):
    """Scalar string dataset value."""
    v = dataset[()]
    return v.decode() if isinstance(v, bytes) else v


def wall(value):
    """Wall-clock 'YYYY-MM-DDTHH:MM:SS' of a stored ISO value (offset dropped)
    so assertions pass regardless of the generating machine's timezone."""
    if hasattr(value, "attrs"):
        value = value[()]
    if isinstance(value, bytes):
        value = value.decode()
    t = pd.to_datetime(value)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def tz_aware(value):
    """True when a stored ISO string carries a UTC offset."""
    if hasattr(value, "attrs"):
        value = value[()]
    if isinstance(value, bytes):
        value = value.decode()
    return pd.to_datetime(value).tzinfo is not None


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _same_bytes(a, b):
    """True when files a and b both exist with identical contents."""
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def _register_definitions():
    """Make the NXoperando_* definitions resolvable by pynxtools; False if not.

    The repo's definitions/*.nxdl.xml are AUTHORITATIVE for the test suite:
    after optionally running the pynxtools-operaxn plugin's installer (which
    prefers its own bundled copies), the repo copies are ALWAYS written over
    the installed pynxtools contributed_definitions copies whenever the bytes
    differ -- a write into site-packages, the same step definitions/README.md
    documents for manual validation. Without this an edited NXDL would be
    silently ignored by validation. validate_nexus crashes (rather than
    reporting invalid) on a file whose entry/definition it cannot resolve, so
    validation must be skipped unless registration succeeds."""
    try:
        from pynxtools_operaxn.reader import ensure_definitions_installed
        ensure_definitions_installed()
    except ImportError:
        pass
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
    """Combined stdout+stderr of validate_nexus on path, or None when
    validation is unavailable (CLI absent, definitions unregistrable, or the
    validator exceeded a 300 s timeout -- callers skip in all three cases).

    The CLI is looked up next to sys.executable first so unactivated venvs
    work, then on PATH."""
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
    """Assert validate_nexus declares path valid.

    Skips when validation is unavailable so the suite still runs in a
    stripped environment -- unless OPERAXN_REQUIRE_VALIDATION is set, which
    CI does on every leg so a lost validator fails loudly instead of
    quietly reducing the run."""
    out = pynxtools_verdict(path)
    if out is None:
        if os.environ.get("OPERAXN_REQUIRE_VALIDATION"):
            pytest.fail("pynxtools validation unavailable but "
                        "OPERAXN_REQUIRE_VALIDATION is set")
        pytest.skip("pynxtools validation unavailable")
    tail = out.strip().splitlines()[-1] if out.strip() else "no output"
    assert "is valid" in out, (f"{tag}: {tail}" if tag else tail)
