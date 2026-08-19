# NeXus application definitions

Custom NeXus application definitions for operando powder diffraction of
electrochemical cells, as described in the OperaXN perspective paper
(Tan et al., *Standardizing Operando Diffraction Studies for Battery
Systems*, ACS Energy Lett.,
[doi:10.1021/acsenergylett.6c01791](https://doi.org/10.1021/acsenergylett.6c01791)):

- **`NXoperando_monopd.nxdl.xml`** — monochromatic X-ray/neutron diffraction
  (modelled on NXmonopd)
- **`NXoperando_tofnpd.nxdl.xml`** — time-of-flight neutron diffraction
  (modelled on NXtofnpd)

Both hold a single `NXentry` with experiment-level instrument/sample metadata
and the full electrochemical cycling record, plus one `NXsubentry` per
diffraction acquisition (`scan_000001` ...) carrying the electrochemical
state of the cell during that acquisition. Status: draft, not yet submitted
upstream to nexusformat/definitions or FAIRmat-NFDI.

OperaXN writes files conforming to these definitions; see below for how to
validate a generated file independently.

## Version

**1.0.0** — first release, shipped with OperaXN 2.0.0. Follows the published
specification (Figure 4 of the perspective paper). The version is stamped on
`entry/definition@version` in every generated file.

## Deviations from the published specification (Figure 4)

The paper's Figure 4 is the target. Where the shipped definitions differ,
the difference is either *forced* by what the source data can supply (the
writer must never emit a non-conforming file), or a *deferred* tightening of
a human-entered field that will be made required at NIAC submission.

| Item | Figure 4 | Here | Reason | Status |
|---|---|---|---|---|
| `extends` | NXmonopd / NXtofnpd | `NXobject` | The parents require entry-level detector `data`/`polar_angle` that cannot coexist with the per-subentry layout. | forced |
| `sample/description` | required | recommended | Human-entered; cannot be harvested from source data. | tighten at NIAC |
| `cycling_protocol` (NXnote) | required | recommended; `technique` required within the group | Human-entered; the writer omits the whole group when `technique` is empty so no non-conforming file can be produced. | tighten at NIAC |
| `operando_electrochemistry` | required | recommended | Absent from diffraction-only files (no electrochemistry supplied). | forced |
| `environment/voltage`, `environment/current` | required | recommended | Present when the scan was correlated within `process/echem_time_tolerance`; absent for uncorrelated scans and diffraction-only files. | forced |
| `environment/voltage_min`, `voltage_max`, `current_min`, `current_max` | required | optional | Need at least one electrochemistry sample inside the acquisition window; short exposures against a sparse logging cadence have none. | forced |
| `environment/scan_timestamp` | required | required | Always written (presence guarantee equals `start_time`'s). | matches |
| `end_time` (entry) | optional | optional | | matches |
| `program_name`, `process`, `process/correlation_method` | required | required | Always written. | matches |
| `SCAN/monitor` | required | required | Always written with `mode`. | matches |
| `environment/echem_index_first`, `_last` | printed `echem_index_start` / `_end` | `echem_index_first` / `echem_index_last` | `_end` is a reserved NeXus field-name suffix. | naming, forced |
| Electrochemistry array length symbol | `nP` | `nE` | Figure 4 uses `nP` for the electrochemistry series; monopd needs `nP` for the diffraction pattern length, so the electrochemistry arrays use `nE`. | naming |
| `SCAN/data` (monopd) | required | optional | Image-only acquisitions carry `image_source` instead; the writer enforces at least one of `data` / `image_source`. | forced |
| `MONITOR` / `DATA` placement | printed at subentry indent in Fig. 4 (the parents define them at entry level) | one per acquisition subentry | OperaXN stores one monitor and one data group per acquisition subentry. | structural choice |
| `SUBENTRY/instrument` | link | HDF5 soft link with `@target` (documented, not declared) | NXDL links cannot be optional (XSD); pynxtools traverses the soft link and checks `@target` but does not match a `<link>` against a linked group, so declaring one fails every conforming file. Linking avoids N copies of the invariant instrument. | forced |
| `environment/capacity` | optional | optional | Plumbed through model/writer/reader/NXDL but not computed (semantics ambiguous); reserved. | reserved |
| `pre_sample_flightpath` (tofnpd) | required | optional | Not in reduced files; supplied from the per-instrument profile, whose POLARIS values are TODO placeholders — the writer skips `None`. | forced until profile values supplied |

Additions beyond Figure 4 (all optional or recommended, so a file carrying
only the Figure 4 content still validates): `definition@version` and
`definition@URL`; `program_name@version`/`@configuration`; extra `process`
provenance (`program`, `version`, `date`, `data_source`,
`echem_time_tolerance`, `total_scans`, `twod_included`,
`twod_max_display_size`); extra `instrument` fields and the verbatim
`edf_metadata` / `synchrotron_metadata` NXcollection harvests; `user`;
`standard_electrochemistry` (non-correlated electrochemistry files);
`environment/voltage_timestamp` and `logbook_entry`; verbatim EDF
beam-monitor counters under `monitor`; `data/errors`; `image_source` and
`image_data`; per-bank NXdata attributes (`spectrum`, `y_unit`, `x_label`,
`source_file`, `measurement_number`); tofnpd per-bank detector geometry
arrays.

## Deviations from the ratified NXmonopd / NXtofnpd parents

| Item | Parent | Here | Forced/Choice | Reason |
|---|---|---|---|---|
| Intensities (`data`) | NX_INT | NX_NUMBER | forced | Azimuthally averaged / focussed, normalised — processed data, not raw counts. |
| `errors` field | absent | optional NX_NUMBER, `[nP]` in monopd (tofnpd leaves it undimensioned: bank lengths vary) | choice (addition) | Carries the propagated uncertainties (Sigma_I) that reduction software exports. |
| Per-bank axes (tofnpd) | single `time_of_flight` `[nTimeChan]` | one NXdata per bank, `bank_N` (time-of-flight) and optional sibling `bank_N_d` (d-spacing) | choice | Focussed banks have different native axes and lengths. |
| `data` XOR `image_source` (monopd) | `data` required | at least one of `data` / `image_source` (writer-enforced) | forced | Image-only acquisitions have no 1-D pattern yet must be recorded. |
| `instrument` in each subentry | n/a (single entry) | HDF5 soft link + `@target`, not an NXDL `<link>` | forced | See Figure 4 table: `<link>` cannot be optional and pynxtools does not match it against a linked group. |
| `echem_index_first`/`_last` | n/a | `_first`/`_last` naming | forced | `_end` is a reserved suffix; parents have no equivalent. |
| `twod_*` process fields | n/a | absent for neutron | choice | 2-D images do not arise in reduced neutron data. |
| `rotation_angle` (monopd) | required NX_FLOAT | absent | choice | The operando cell is stationary / the angle is not recorded. |
| Detector `polar_angle` / `data` | instrument-level, `[nDet]` under NXdetector | per-scan `SCAN/data` NXdata `[nP]` | structural | Each acquisition is its own subentry with its own pattern. |
| Entry-level NXdata links | `data` NXdata linking to detector fields | direct datasets in each subentry's NXdata | structural | No detector-level arrays to link to; per-subentry storage. |
| `monitor/preset` | required | recommended (monopd) / optional (tofnpd) | forced | Not all sources record it. |
| `monitor/integral` | required | optional | forced | Not all sources record it. |
| `monitor/mode` | enumeration `monitor` / `timer` | free NX_CHAR | forced | Not all sources record the mode; verbatim value kept. |
| `crystal` (monopd) | required NXcrystal with `wavelength [i]` | recommended, scalar `wavelength` | forced/choice | Absent for neutron sources; a monochromatic beam has one wavelength. |
| `source/probe` (monopd) | enumeration incl. electron | narrowed to `x-ray` / `neutron` | choice | Only these probes are in scope. |
| `user` (tofnpd) | required | optional | forced | Harvest-dependent; not present in all reduced files. |
| Detector geometry (tofnpd) | required `[nDet]` `distance`, `polar_angle`, `azimuthal_angle`, `detector_number` | optional per-bank arrays `[nBank]` | forced | Not in reduced files; supplied from the instrument profile, per bank rather than per detector element. `nBank`, not `nDet`. |
| Monitor spectrum (tofnpd) | `distance`, `data [nTimeChan]`, `time_of_flight` | absent | forced | Not present in reduced files. |
| 2-D `[nDet, nTimeChan]` data (tofnpd) | required | dropped | forced | Focussed 1-D pattern per bank. |
| `monitor/integral` (tofnpd) | absent | optional | choice (addition) | Kept when the reduction file carries it. |
| `pre_sample_flightpath` (tofnpd) | required | optional | forced | Until profile values are supplied (see above). |

## Validating generated files

Files can be validated with FAIRmat's
[pynxtools](https://github.com/FAIRmat-NFDI/pynxtools). pynxtools resolves
definition names against its own bundled schema library, so the two NXDL
files here must be copied into it once (and again after every pynxtools
upgrade, which replaces the bundle):

```bash
pip install 'pynxtools>=0.15,<0.16'

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

After editing either NXDL, the copies inside pynxtools' contributed
definitions folder must be refreshed (repeat step 2) or validation silently
runs against the old version. The test suite does this automatically: the
repo `definitions/*.nxdl.xml` are authoritative and are copied over the
installed pynxtools copies whenever the bytes differ.

## Validating the definitions themselves

The NXDL meta-schema (`nxdl.xsd`, `nxdlTypes.xsd`) is vendored in
[../schema/](../schema) at release tag **v2026.01** (the pinned version these
were authored against), and the test suite validates both definitions against
it automatically (`pytest tests/test_nxdl.py`). To check manually with any
XSD validator:

```python
from lxml import etree
xsd = etree.XMLSchema(etree.parse("schema/nxdl.xsd"))
xsd.assertValid(etree.parse("definitions/NXoperando_monopd.nxdl.xml"))
```
