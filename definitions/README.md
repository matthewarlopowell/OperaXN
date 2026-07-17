# NeXus application definitions

Custom NeXus application definitions for operando powder diffraction of
electrochemical cells, as described in the OperaXN perspective paper:

- **`NXoperando_monopd.nxdl.xml`** — monochromatic X-ray/neutron diffraction
  (modelled on NXmonopd)
- **`NXoperando_tofnpd.nxdl.xml`** — time-of-flight neutron diffraction
  (modelled on NXtofnpd)

Both hold a single `NXentry` with experiment-level instrument/sample metadata
and the full electrochemical cycling record, plus one `NXsubentry` per
diffraction acquisition (`scan_000001` ...) carrying the electrochemical
state of the cell during that acquisition. Deliberate deviations from the
ratified definitions (processed NX_NUMBER intensities, per-bank native axes)
are documented inside each file. Status: draft, not yet submitted upstream
to nexusformat/definitions or FAIRmat-NFDI.

OperaXN writes files conforming to these definitions; see below for how to
validate a generated file independently.

## Validating generated files

Files can be validated with FAIRmat's
[pynxtools](https://github.com/FAIRmat-NFDI/pynxtools). pynxtools resolves
definition names against its own bundled schema library, so the two NXDL
files here must be copied into it once (and again after every pynxtools
upgrade, which replaces the bundle):

```bash
pip install pynxtools

# 1. locate pynxtools' contributed-definitions folder
python -c "import pynxtools, os; print(os.path.join(os.path.dirname(pynxtools.__file__), 'definitions', 'contributed_definitions'))"

# 2. copy BOTH definitions there
#    Windows:
copy NXoperando_monopd.nxdl.xml <path printed above>
copy NXoperando_tofnpd.nxdl.xml <path printed above>
#    macOS/Linux:
cp NXoperando_*.nxdl.xml <path printed above>

# 3. validate
pynx validate experiment.nxs
```

Success looks like:

```
The entry `entry` in file `experiment.nxs` is valid according to the
`NXoperando_monopd` application definition.
```

(`NXoperando_tofnpd` for neutron files. Older pynxtools versions use
`validate_nexus experiment.nxs` instead of `pynx validate`.)

## Validating the definitions themselves

The NXDL meta-schema is not vendored here. To validate these files, fetch
`nxdl.xsd` and `nxdlTypes.xsd` from https://github.com/nexusformat/definitions
at release tag **v2026.01** (the pinned version these were authored against)
and check with any XSD validator, e.g.:

```python
from lxml import etree
xsd = etree.XMLSchema(etree.parse("nxdl.xsd"))
xsd.assertValid(etree.parse("NXoperando_monopd.nxdl.xml"))
```
