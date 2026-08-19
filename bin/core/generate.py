"""
Raw data -> canonical .nxs orchestration for the core pipeline.

`generate()` is the single entry point: collect files (ZIPs extracted,
xlsx/csv converted), classify, correlate, and write the canonical NeXus file.
The GUI then loads that file via nxs_reader — one read path for everything.

Canonical files always store absolute timestamps; relative display is a GUI
view transform, never baked into the file.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import config as _config
from .config import (
    BATCH_SIZE,
    MAX_WORKERS,
    PARALLEL_PROCESSING,
    PARALLEL_PROCESSING_THRESHOLD,
)
from .classify import (
    FileClassificationManager,
    NexusMetadataExtractor,
    SynchrotronFileGrouper,
)
from .correlate import ScanProcessor
from .model import (
    DataSourceType,
    FileRecord,
    FileType,
    Scan,
    SUPPORTED_EXTENSIONS,
    TimeMethod,
)
from .nxs_writer import NXSWriter

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]


# ============================================================================
# Tabular format conversion
# ============================================================================

def convert_tabular_to_txt(path: str) -> str:
    """Convert .xlsx/.csv to tab-delimited .txt; returns original path otherwise
    or on failure."""
    lower = path.lower()
    if lower.endswith('.xlsx'):
        read = pd.read_excel
    elif lower.endswith('.csv'):
        read = pd.read_csv
    else:
        return path

    try:
        df = read(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        fd, txt_path = tempfile.mkstemp(prefix=f"{base_name}_", suffix=".txt")
        os.close(fd)
        df.to_csv(txt_path, sep='\t', index=False)
        logger.info(f"Converted {os.path.basename(path)} to txt format")
        return txt_path
    except Exception as e:
        logger.error(f"Error converting tabular file {path}: {e}")
        return path


# ============================================================================
# File collection
# ============================================================================

class FileProcessor:
    """Collects, extracts, and pre-classifies files from paths, ZIPs, and directories."""

    def __init__(self, data_source: DataSourceType = DataSourceType.INHOUSE):
        self.data_source = data_source
        self.tempdir: Optional[str] = None
        self.processed_files: set = set()
        self.synchrotron_grouper = SynchrotronFileGrouper()
        self.nexus_extractor = NexusMetadataExtractor()

    def __enter__(self) -> 'FileProcessor':
        """Create the ZIP-extraction tempdir for the processing session."""
        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Delete the extraction tempdir (all reads, including the NeXus
        write, must happen inside the context)."""
        if self.tempdir and os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def process_paths(self, selected_paths: List[str]) -> List[FileRecord]:
        """Main entry: collect files, group source-specific ones, process the rest."""
        all_files = self._collect_all_files(selected_paths)

        if not all_files:
            return []

        file_dict = {extracted: original for extracted, original in all_files}

        # Source-specific grouping
        if self.data_source == DataSourceType.NEUTRON:
            records = self._process_neutron_files(all_files)
        elif self.data_source == DataSourceType.SYNCHROTRON:
            synchrotron_groups = self.synchrotron_grouper.group_files(file_dict)
            records = self._process_synchrotron_groups(synchrotron_groups)
        else:
            records = []

        # Process ungrouped files
        remaining_files = [
            (extracted, original) for extracted, original in all_files
            if extracted not in self.processed_files
        ]

        if remaining_files:
            if PARALLEL_PROCESSING and len(remaining_files) > PARALLEL_PROCESSING_THRESHOLD:
                chunk_size = max(BATCH_SIZE, len(remaining_files) // MAX_WORKERS)
                chunks = [remaining_files[i:i + chunk_size]
                          for i in range(0, len(remaining_files), chunk_size)]
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for chunk_records in executor.map(self._process_remaining_chunk, chunks):
                        records.extend(chunk_records)
            else:
                records.extend(self._process_remaining_chunk(remaining_files))

        return records

    def _process_neutron_files(self, all_files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Create records for every collected file (no extension filter)."""
        records = []

        for extracted_path, original_path in all_files:
            self.processed_files.add(extracted_path)
            records.append(FileRecord(path=extracted_path, original_path=original_path))

        logger.info(f"Processed {len(records)} neutron-related files")
        return records

    def _collect_all_files(self, selected_paths: List[str]) -> List[Tuple[str, str]]:
        """Recursively collect files from paths/directories, extract ZIPs, convert xlsx/csv."""
        all_files: List[Tuple[str, str]] = []

        for path in selected_paths:
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext == FileType.ZIP.value:
                    all_files.extend(self._extract_zip_files(path))
                else:
                    all_files.append((path, path))
            elif os.path.isdir(path):
                all_files.extend(self._collect_directory_files(path))

        all_files = self._convert_files_to_txt(all_files)

        logger.info(f"Collected {len(all_files)} total files")
        return all_files

    @staticmethod
    def _convert_files_to_txt(files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Convert .xlsx and .csv to tab-delimited .txt for uniform parsing."""
        return [(convert_tabular_to_txt(extracted), original)
                for extracted, original in files]

    def _extract_zip_files(self, zip_path: str) -> List[Tuple[str, str]]:
        """Extract supported files from ZIP, skipping macOS metadata."""
        extracted: List[Tuple[str, str]] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                supported_exts = SUPPORTED_EXTENSIONS - {FileType.ZIP.value}

                for member in archive.namelist():
                    if (member.endswith("/") or
                            member.startswith("__MACOSX") or
                            member.startswith(".")):
                        continue

                    member_ext = os.path.splitext(member)[1].lower()

                    if member_ext in supported_exts:
                        # Validate path to prevent ZIP Slip (path traversal)
                        target = os.path.realpath(os.path.join(self.tempdir, member))
                        if not target.startswith(os.path.realpath(self.tempdir) + os.sep):
                            logger.warning(f"Skipping ZIP entry with invalid path: {member}")
                            continue
                        extracted_path = archive.extract(member, self.tempdir)
                        original_path = os.path.join(zip_path, member)
                        extracted.append((extracted_path, original_path))

        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")

        return extracted

    def _collect_directory_files(self, directory: str) -> List[Tuple[str, str]]:
        """Walk directory tree, collecting supported files and extracting nested ZIPs."""
        files: List[Tuple[str, str]] = []

        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                file_path = os.path.join(root, filename)
                ext = os.path.splitext(file_path)[1].lower()

                if ext == FileType.ZIP.value:
                    files.extend(self._extract_zip_files(file_path))
                elif ext in SUPPORTED_EXTENSIONS:
                    files.append((file_path, file_path))

        return files

    def _process_synchrotron_groups(
            self, synchrotron_groups: Dict[str, Dict[str, str]]) -> List[FileRecord]:
        """Create records for each synchrotron scan group (requires .nxs anchor)."""
        records: List[FileRecord] = []

        for base_id, file_group in synchrotron_groups.items():
            if 'nxs' in file_group:
                record = self._create_synchrotron_record(file_group)
                if record:
                    records.append(record)
                    for file_path in file_group.values():
                        self.processed_files.add(file_path)

        return records

    def _process_remaining_chunk(self, files: List[Tuple[str, str]]) -> List[FileRecord]:
        """Create simple records for ungrouped .dat, .edf, .txt files."""
        records: List[FileRecord] = []
        for extracted_path, original_path in files:
            ext = os.path.splitext(extracted_path)[1].lower()
            if ext in (FileType.DAT.value, FileType.EDF.value, FileType.TXT.value):
                self.processed_files.add(extracted_path)
                records.append(FileRecord(path=extracted_path, original_path=original_path))
        return records

    def _create_synchrotron_record(self, file_group: Dict[str, str]) -> Optional[FileRecord]:
        """Build a FileRecord for a synchrotron group, enriched with .nxs metadata."""
        nxs_path = file_group.get('nxs')
        if not nxs_path:
            return None

        metadata = self.nexus_extractor.extract(nxs_path) or {}

        return FileRecord(
            path=nxs_path,
            original_path=nxs_path,
            oned=file_group.get('xy'),
            twod=file_group.get('hdf'),
            timestamp=metadata.get('timestamp'),
            exposure_time=metadata.get('exposure_time'),
            source_nxs=nxs_path
        )


# ============================================================================
# Pipeline
# ============================================================================

def _run_pipeline(processor: FileProcessor,
                  input_paths: List[str],
                  data_source: DataSourceType,
                  progress_callback: ProgressCallback,
                  time_method: TimeMethod) -> Tuple[List[Scan], pd.DataFrame]:
    """Collect, classify, and correlate using an already-open FileProcessor.

    The processor's context must stay open while the returned scans' file
    paths are read: see the tempdir constraint on `process_raw`.
    """
    records = processor.process_paths(input_paths)

    if not records:
        return [], pd.DataFrame(columns=["timestamp", "echem_data", "current", "source_file"])

    records_df = pd.DataFrame([r.__dict__ for r in records])

    for col in ["oned", "twod", "echem", "neutron_meta", "timestamp",
                "source_nxs", "path", "original_path"]:
        if col in records_df.columns:
            records_df[col] = records_df[col].astype("object")

    if progress_callback:
        progress_callback("Classifying files...")

    classifier = FileClassificationManager(data_source)
    classified_df = classifier.classify_files(records_df)

    if progress_callback:
        progress_callback("Processing scans and correlating with echem...")

    scan_processor = ScanProcessor(time_method, data_source,
                                   apply_relative_display=False)
    return scan_processor.process_scans(classified_df)


def process_raw(input_paths: List[str],
                data_source: DataSourceType = DataSourceType.INHOUSE,
                progress_callback: ProgressCallback = None,
                time_method: TimeMethod = TimeMethod.ABSOLUTE
                ) -> Tuple[List[Scan], pd.DataFrame]:
    """Collect, classify, and correlate raw data.

    `time_method` selects how scans are *matched* to echem (RELATIVE re-zeroes
    both streams, for unsynchronised clocks). Timestamps in the output are
    always absolute — relative display is a GUI view transform.

    For ZIP inputs the file paths on the returned scans point into a temp
    directory that is removed when this function returns. Use `generate`
    (which keeps the extraction alive until the .nxs is written) when the
    data itself is needed.
    """
    with FileProcessor(data_source) as processor:
        return _run_pipeline(processor, input_paths, data_source,
                             progress_callback, time_method)


def validate(scans: List[Scan], echem_df: pd.DataFrame) -> List[str]:
    """Return validation messages; entries starting with 'Warning' are non-fatal."""
    errors: List[str] = []

    if not scans:
        errors.append("No scan data available")

    if echem_df.empty:
        errors.append("Warning: No echem data available")

    has_xrd = any(scan.oned or scan.twod for scan in scans)
    has_neutron = any(scan.neutron_files for scan in scans)

    if scans and not has_xrd and not has_neutron:
        errors.append("No XRD or neutron data found in scans")

    return errors


def generate(input_paths: List[str],
             output_path: str,
             data_source: DataSourceType = DataSourceType.INHOUSE,
             include_2d_images: bool = False,
             max_display_size: int = 0,
             standard_echem_files: Optional[List[str]] = None,
             progress_callback: ProgressCallback = None,
             time_method: TimeMethod = TimeMethod.ABSOLUTE,
             title: Optional[str] = None,
             sample_name: Optional[str] = None,
             sample_description: Optional[str] = None,
             sample_preparation_date: Optional[str] = None,
             cycling_protocol: Optional[Dict[str, Any]] = None
             ) -> Tuple[bool, List[str]]:
    """Full pipeline: raw paths -> canonical .nxs file at output_path;
    returns (success, messages). `sample_preparation_date` (ISO 8601) and
    `cycling_protocol` (keys per nxs_writer.CYCLING_PROTOCOL_FIELDS) are
    forwarded to the writer."""
    try:
        # The FileProcessor context must span writing too: the writer reads
        # data from the scans' file paths, which for ZIP inputs live in the
        # processor's temp extraction directory.
        with FileProcessor(data_source) as processor:
            scans, echem_df = _run_pipeline(processor, input_paths, data_source,
                                            progress_callback, time_method)

            messages = validate(scans, echem_df)
            fatal = [m for m in messages if not m.startswith("Warning")]
            if fatal:
                return False, messages

            if progress_callback:
                progress_callback("Writing NeXus file...")

            # Convert standard echem xlsx/csv the same way as everything else
            std_files = None
            if standard_echem_files:
                std_files = [convert_tabular_to_txt(p) for p in standard_echem_files]

            writer = NXSWriter(data_source,
                               include_2d_images=include_2d_images,
                               max_display_size=max_display_size,
                               correlation_method=time_method.value,
                               title=title,
                               sample_name=sample_name,
                               sample_description=sample_description,
                               sample_preparation_date=sample_preparation_date,
                               cycling_protocol=cycling_protocol)
            writer.write(output_path, scans, echem_df, std_files)

        if _config.USE_PYNXTOOLS_WRITER:
            _revalidate_with_pynxtools(output_path, data_source)

        return True, messages if messages else ["NeXus file generated successfully"]

    except Exception as e:
        logger.exception("Error generating NeXus file")
        return False, [f"Error: {e}"]


def _revalidate_with_pynxtools(output_path: str,
                               data_source: DataSourceType) -> None:
    """Best-effort rewrite through the pynxtools dataconverter (validates
    against the NXoperando_* definitions). Any failure is logged and the
    plain NXSWriter file is kept — the GUI never breaks on this path."""
    tmp_out = output_path + ".pynx.nxs"
    try:
        from pynxtools_operaxn.reader import ensure_definitions_installed
        from pynxtools.dataconverter.convert import convert

        ensure_definitions_installed()
        nxdl = ('NXoperando_tofnpd' if data_source == DataSourceType.NEUTRON
                else 'NXoperando_monopd')
        convert(input_file=(output_path,), reader='operaxn', nxdl=nxdl,
                output=tmp_out)
        os.replace(tmp_out, output_path)
        logger.info(f"Output validated and rewritten by pynxtools ({nxdl})")
    except Exception as e:
        logger.warning(f"pynxtools validation pass skipped: {e}")
        if os.path.isfile(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass


def cache_dir() -> str:
    """Directory for auto-generated canonical files (created on demand)."""
    d = os.path.join(tempfile.gettempdir(), "operaxn_cache")
    os.makedirs(d, exist_ok=True)
    return d


def default_cache_path(input_paths: List[str]) -> str:
    """Location for the auto-generated canonical file: always a user cache
    directory, never next to the data (beamline mounts are often read-only)."""
    if input_paths:
        stem = os.path.splitext(os.path.basename(os.path.normpath(input_paths[0])))[0]
    else:
        stem = "experiment"
    return os.path.join(cache_dir(), f"{stem}.nxs")
