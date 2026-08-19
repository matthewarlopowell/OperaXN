# NXDL schema (vendored)

`nxdl.xsd` and `nxdlTypes.xsd` are vendored verbatim from the NeXus
International Advisory Committee definitions repository:

- Source: https://github.com/nexusformat/definitions
- Release tag: **v2026.01**
- Upstream license: LGPL-3.0 (the NeXus definitions licence,
  https://www.gnu.org/licenses/lgpl-3.0.txt); these two files are included
  unmodified for schema validation only and remain under that licence,
  separate from the MIT licence covering the rest of this repository.

They are used by the test suite (`tests/test_nxdl.py`) to validate the
application definitions in `definitions/`. When updating, fetch both files
from the same tag and record the new tag here and in `definitions/README.md`.
