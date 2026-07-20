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
   pip install -e .
   ```

3. Make your changes, keeping each pull request focused on a single fix or
   feature.
4. Check the GUI still launches and loads data (`operaxn`), then open a pull
   request describing what changed and why.

Support for additional instruments, beamlines and file formats is
particularly welcome — an issue describing the format (ideally with an
example file) is a good place to start.

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
