---
title: "OperaXN: A Python tool for correlating, visualising, and standardising operando battery diffraction data"
authors:
  - name: Matthew A. Powell
    orcid: orcid.org/0000-0003-3150-8467
    affiliation: 1, 2
  - name: José J. Arroyo-Gómez
    orcid: 0000-0001-8214-0645
    affiliation: 3, 4
  - name: Tobias A. Bird
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 5
  - name: Gaurav C. Pandey
    orcid: 0000-0001-9823-1672
    affiliation: 1
  - name: Louis F. J. Piper
    orcid: 0000-0002-3421-3210
    affiliation: 1, 2, 7
  - name: Gabriel E. Perez
    orcid: 0000-0003-3150-8467
    affiliation: 7
  - name: Ashok S. Menon
    orcid: 0000-0001-8148-8615
    affiliation: 1, 2, 7
affiliations:
  - name: 'Warwick Manufacturing Group (WMG), University of Warwick, Coventry, UK'
    index: 1
  - name: 'The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK'
    index: 2
  - name: 'Departamento de Almacenamiento de la Energía, Subgerencia Operativa de Energía y Movilidad, Instituto Nacional de Tecnología Industrial (INTI), Buenos Aires, Argentina'
    index: 3
  - name: 'Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET), Buenos Aires, Argentina'
    index: 4
  - name: 'Diamond Light Source Ltd., Harwell Science and Innovation Campus, Didcot UK'
    index: 5
  - name: 'The Hartnoll Centre for Experimental Fuel Technologies, University of Warwick, Coventry, UK'
    index: 6
  - name: 'ISIS Neutron and Muon Source, Harwell Science and Innovation Campus, Didcot OX11 0QX, U.K.'
    index: 7
  
date: DD Month YYYY
bibliography: paper.bib
---

# Summary

Understanding how electrochemical stimuli influence the physicochemical properties of the electroactive materials is key to developing batteries and other electrochemical systems,. As such systems, especially batteries, may be kinetically stabilised, this is best achieved by studying them *in operando*, as they are operating, using complementary probes such as powder diffraction, which disentangles globally averaged electrochemical data using component-specific chemical information.

For battery cells, *operando* X-ray and neutron diffraction experiments involve the collection of powder diffraction patterns while the cell is electrochemically cycled. A typical measurement charges and discharges the cell at a specified rate within a set voltage window on a diffractometer, with diffraction patterns collected continuously throughout. Thus, each experiment produces two streams of data: a sequence of diffraction patterns and a continuous electrochemical record (time, voltage, and current) from the cycling equipment. The flexible nature of the experiment allows for a wide variety of cells to be studied under diverse operating conditions to probe specific research questions. With recent advances in instrumentation and automated analysis, *operando* diffraction experiments are becoming more accessible to the research community. The resulting multi-modal datasets are rich enough to support performance diagnostics studies as well as data-driven prediction and discovery, e.g., from training machine learning models that predict structural evolution to enabling high-throughput screening of electrode materials across operating conditions [@skurtveit2026operando].    

OperaXN is a cross-platform Python desktop application that automates the data processing workflow for *operando* diffraction experiments for electrochemical systems. It ingests raw diffraction and electrochemistry files from laboratory X-ray diffraction (XRD), synchrotron XRD, and neutron diffraction instruments, time-correlates the two data streams, and consolidates the result into a single standardised NeXus file [@konnecke2015nexus] conforming to the custom `NXoperando_monopd` and `NXoperando_tofnpd` application definitions introduced in our recent perspective article [@tan2026standardizing]. The resulting file can be shared with collaborators or deposited in a repository, and be visualised interactively within the application.

# Statement of need

Operando diffraction has grown rapidly as a diagnostic characterisation technique for battery research [@skurtveit2026operando]. A typical experiment generates hundreds of diffraction data files; this can increase by an order of magnitude when experiments span multiple cycles or are performed at synchrotron and neutron facilities where high-frequency data collection is possible. For example, the I11 powder diffraction beamline at Diamond Light Source has a dedicated long-duration experiment facility [@murray2017synchrotron] that can cycle cells for over 500 cycles (which could span >1 year) with periodic diffraction data collection [@xu2021bulk]. Our recent work has further demonstrated that diffraction data can now be collected from single-layer pouch cells at synchrotron and neutron sources in as little as 10 ms [@menon2025spatially] and 3–5 minutes [@menon2026advancing], respectively. Managing these large datasets and reliably correlating the two data streams (diffraction and electrochemistry) sampled at very different frequencies is not straightforward. The task is further complicated by the intermittent nature of diffraction data collection arising from beam interruptions and other instrumental factors. The correlation step, i.e., timestamp alignment followed by nearest-neighbour lookup across two independently clocked instruments, i.e., is currently performed using custom, user-defined scripts that are rarely documented or shared. As highlighted by Drnec and Lyonnard, the field urgently needs reliable, representative, and reproducible characterisation workflows [@drnec2025battery].

Once correlated, there is no established format for storing the combined diffraction-plus-electrochemistry dataset. Groups typically archive the raw files separately, with correlation parameters held in custom, sometimes private, scripts. This makes reproducing figures, sharing data between collaborators, and depositing datasets for publication difficult. The problem is expected to gain significance as AI and machine learning methods and autonomous experimentation workflows become more widely adopted, since these approaches depend directly on well-structured, consistently formatted datasets.

OperaXN addresses both problems. It provides a single, documented pipeline that handles the correlation automatically and stores the result in a standardised, self-describing NeXus file readable by any HDF5 tool without knowledge of OperaXN itself. This enables experimental researchers at synchrotron, neutron, and laboratory X-ray facilities to share their operando datasets seamlessly, and gives computational and modelling researchers direct access to well-structured data without requiring significant pre-processing.

# State of the field

Several tools exist for reducing and analysing diffraction data collected at
large-scale facilities. DAWN [@dawn2012] is the primary data reduction platform
at Diamond Light Source and other European synchrotrons; it handles detector
calibration, azimuthal integration, and peak fitting, but it is not designed to
ingest or correlate electrochemical data streams.
pynxtools [@pynxtools] provides a general framework for converting heterogeneous
instrument outputs into NeXus-compliant HDF5 files and is used by the NOMAD
research data management platform [@nomad]; it does not contain any
electrochemical correlation logic and has no built-in support for the
operando battery experiment workflow.
Mantid [@mantid2014] is the standard reduction environment for neutron
diffraction at ISIS and SNS; it handles raw detector data to integrated
patterns but again has no electrochemical layer.

In practice, the correlation and archiving steps are performed by bespoke
per-group Python or MATLAB scripts. These are rarely published, vary in
quality, and produce proprietary file formats that cannot be read without
access to the original script. OperaXN fills this gap with a tool specifically
designed for the combined diffraction-plus-electrochemistry dataset, built on
top of the outputs of existing reduction tools (integrated patterns in `.xy`,
`.dat`, or `.nxs` format) rather than competing with them.

# Software design

OperaXN is structured as two layers. The `core` package implements the data
pipeline: file collection and format detection, classification of files by
instrument type, time-correlation of the diffraction and electrochemistry
streams, and writing and reading of the canonical NeXus file. The `operaxn`
package implements the Tkinter desktop GUI, which drives the pipeline and
provides interactive visualisation. The GUI operates entirely through the
canonical NeXus file: every load — whether from raw instrument files or an
existing `.nxs` — passes through the same reader, so there is a single
consistent data path for all operations.

The time-correlation algorithm matches each diffraction frame to the nearest
electrochemical datapoint by timestamp, using the midpoint of the acquisition
window (start time plus half the exposure time) rather than the frame start
time. Two modes are provided: absolute time, in which instrument clocks are
assumed to be synchronised, and relative time, in which both streams are
independently zeroed to their first timestamp and correlated by elapsed time.
Per-scan summaries of voltage and current (minimum, maximum, and for long
neutron acquisitions, the full time series) are stored in the NeXus file
alongside the matched scalar values.

The output format is defined by two custom NeXus application definitions,
`NXoperando_monopd` (monochromatic X-ray and neutron) and `NXoperando_tofnpd`
(time-of-flight neutron), shipped in the `definitions/` directory of the
repository. Both extend the ratified `NXmonopd` and `NXtofnpd` base definitions
with an electrochemical layer: a `cycling_protocol` group recording the
experimental protocol, and per-scan `electrochemical_environment` groups
recording the instantaneous cell state. The instrument description is written
once in the top-level `NXentry` and linked symbolically from each scan
subentry, avoiding redundant replication of beamline geometry across hundreds
of frames. Generated files can be validated against these definitions using
pynxtools [@pynxtools] and inspected with any HDF5 browser.

The application supports three instrument families out of the box: laboratory
X-ray diffractometers (`.dat` and `.xy` pattern files), synchrotron XRD
beamlines including Diamond Light Source I11 (`.nxs`, `.hdf`, `.xy`), and
ISIS neutron diffractometers including POLARIS (`.dat`, logbook `.txt`). Input
files may be supplied individually, as a directory, or as a ZIP archive.

Beyond the core correlation and NeXus generation workflow, OperaXN provides
analysis tools directly relevant to battery research: an operando heatmap
(stacked diffraction patterns plotted against scan number or time with the
voltage profile overlaid), a capacity analysis window, and an intermittent
current interruption (ICI) analysis module. Figures can be exported as PNG,
PDF, or SVG, and scan-by-scan data as Excel summaries.

# Research impact

[FILL IN: Describe published or submitted papers that used OperaXN, datasets
deposited with a DOI, or beamlines/facilities that have adopted the tool.
JOSS requires evidence of actual use. For example:
"OperaXN has been used in [Author et al., Year] to study structural evolution
in NMC811 cathodes during cycling at Diamond Light Source beamline I11 and
ISIS POLARIS. The resulting dataset has been deposited at [repository] with
DOI [DOI]."
If no paper is published yet, a submitted preprint qualifies.]

# AI usage

This project was developed with the assistance of Claude (Anthropic). Claude
was used during software development for code review and refactoring assistance.
The text of this paper was written by the authors with Claude being used for proof-reading.

# Acknowledgements

[FILL IN: funding sources, beamtime allocations, facility support]

# References

