# Contributing to OperaXN

Thank you for your interest in OperaXN. Bug reports, feature requests and
pull requests are all welcome.

## Reporting bugs

Open an issue on the
[issue tracker](https://github.com/matthewarlopowell/OperaXN/issues) and
include:

- your operating system and Python version
- the OperaXN version (`pip show operaxn`)
- the data source (laboratory XRD, synchrotron XRD or neutron) and file
  formats involved
- the full traceback, or the log output from running with `operaxn --debug`

If the problem is specific to a data file and you are able to share it (or a
minimal excerpt), that makes reproduction much easier.

## Seeking support

Questions about using OperaXN — loading data, time correlation, the NeXus
output — can also be raised on the issue tracker.

## Contributing changes

1. Fork the repository and create a branch from `main`.
2. Install in development mode:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e '.[test]'
   ```

3. Make your changes, keeping each pull request focused on a single fix or
   feature.
4. Run the test suite (`pytest`) and check the GUI still launches and loads
   data (`operaxn`), then open a pull request describing what changed and
   why.

## Running the tests

Install the test dependencies and run pytest from the repository root:

```bash
pip install -e '.[test]'
pytest
```

Notes:

- Tests run on synthetic data generated at test time; no data files are
  required (the opt-in `realdata` marker below is the one exception).
- Tests marked `gui` import the Tk GUI package: they need `tkinter`
  importable but no display, and skip where it is missing.
- Tests marked `display` open real Tk windows and are skipped automatically
  when no display is available (CI runs them under xvfb).
- Tests marked `realdata` run against local instrument datasets: set
  `OPERAXN_TEST_DATA=<folder>` to enable them (see
  [examples/README.md](examples/README.md) for the expected layout). They
  are skipped otherwise.

The `[test]` extra includes FAIRmat's pynxtools, so the NeXus
schema-validation tests run by default; they skip only if it is missing.
Those tests copy the two NXDL files from `definitions/` into pynxtools'
installed `contributed_definitions` directory — the validator cannot resolve
`NXoperando_monopd` / `NXoperando_tofnpd` otherwise. Run them in a virtual
environment rather than against a system Python install.

Pull requests should keep `pytest` green; CI runs the suite on
Python 3.10–3.13 on Linux (plus Python 3.12 on Windows) on every push to
main and every pull request.

Support for additional instruments, beamlines and file formats is
particularly welcome — an issue describing the format (ideally with an
example file) is a good place to start. Contributed public example
datasets are equally valuable: see the layout in
[examples/README.md](examples/README.md) and extend the real-data
section of `tests/test_pipeline.py` with a test describing what the
dataset should produce.

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
