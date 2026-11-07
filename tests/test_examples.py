"""The example generator: its walkthrough files and its ICI analysis log."""
import filecmp
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import _builders
import core

pytestmark = pytest.mark.gui

pytest.importorskip("tkinter")
from operaxn.capacity import assign_cycles, compute_capacity  # noqa: E402
from operaxn.ici import _compute_all_pulses, _detect_pulses  # noqa: E402

GENERATOR = Path(__file__).resolve().parents[1] / "examples" / "make_example_data.py"


def _generator():
    """Import examples/make_example_data.py as a module."""
    spec = importlib.util.spec_from_file_location("make_example_data", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_matches_builders(tmp_path):
    """The walkthrough's laboratory and neutron files are byte-identical to tests/_builders.py."""
    gen = _generator()
    gen.make_laboratory(str(tmp_path / "ex" / "laboratory"))
    gen.make_neutron(str(tmp_path / "ex" / "neutron"))
    _builders.write_inhouse_dir(str(tmp_path / "b" / "laboratory"))
    _builders.write_neutron_dir(str(tmp_path / "b" / "neutron"))
    for sub in ("laboratory", "neutron"):
        cmp = filecmp.dircmp(tmp_path / "ex" / sub, tmp_path / "b" / sub)
        assert not (cmp.left_only or cmp.right_only), sub
        _, differ, errors = filecmp.cmpfiles(tmp_path / "ex" / sub, tmp_path / "b" / sub,
                                             cmp.common_files, shallow=False)
        assert not differ and not errors, (sub, differ, errors)


def test_example_ici_log_drives_ici_and_capacity(tmp_path):
    """The ICI log: 2 cycles of 0.5 mAh legs, each of 10 pulses that fit the R and k it encodes."""
    gen = _generator()
    gen.make_analysis(str(tmp_path))
    df = core.EchemParser().parse(str(tmp_path / "ici_echem.txt"))
    cycles = assign_cycles(df)
    assert sorted(c for c in cycles["cycle"].unique() if c > 0) == [1, 2]
    for cycle in (1, 2):
        for phase in ("charge", "discharge"):
            seg = cycles[(cycles["cycle"] == cycle) & (cycles["phase"] == phase)]
            assert np.isclose(compute_capacity(seg)[-1], 0.5, rtol=1e-3), (cycle, phase)
            fits = _compute_all_pulses(seg, _detect_pulses(seg, max_rest=300.0),
                                       lambda idx: (0.5, 1.0))
            assert len(fits) == gen.ICI_PULSES, (cycle, phase)
            for fit in fits:
                done = fit["pulse_idx"] / gen.ICI_PULSES
                r_true, k_true = gen.ici_parameters(done if phase == "charge" else 1.0 - done,
                                                    cycle - 1)
                assert np.isclose(fit["R"], r_true, rtol=1e-3), (cycle, phase, fit["pulse_idx"])
                assert np.isclose(fit["k"], k_true, rtol=1e-2), (cycle, phase, fit["pulse_idx"])
