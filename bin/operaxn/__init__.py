"""
OperaXN: operando diffraction visualisation with NeXus-canonical storage.

The GUI (this package) is a thin layer over the shared `core` pipeline:
every raw-data upload is generated into a canonical .nxs file and the
interface works entirely from that file; opening a .nxs directly skips
generation. See the `core` package for the pipeline itself.
"""

from .config import APP_VERSION, APP_AUTHOR, SUPPORT_EMAIL

__version__ = APP_VERSION
__author__ = APP_AUTHOR
__email__ = SUPPORT_EMAIL
__license__ = "MIT"

from .config import (  # noqa: F401  (package API re-exports)
    DataSourceType,
    OPERAXNTheme
)

from .gui import OPERAXN, UIState, VisualiserConfig

from .capacity import (
    assign_cycles,
    compute_capacity,
    plot_capacity_vs_voltage,
    plot_time_vs_voltage,
    CapacityWindow
)

from .heatmap import HeatmapWindow
from .ici import ICIWindow

from .input import (
    process_paths,
    make_oned_arrays,
    make_twod_arrays,
    make_echem_arrays,
    get_correlated_data,
    add_standard_echem_files,
    TimeMethod
)

from .output import (
    plot_oned_data,
    plot_twod_data,
    plot_echem_data,
    plot_neutron_data,
    create_figure_layout,
    export_single_scan,
    clear_plot_cache,
    clear_plot_axes,
    get_scan_time_positions,
    PlotConfig
)

from .dialog import (
    UploadOptionsDialog,
    PlotSettingsDialog,
    ExportOptionsDialog,
    GIFSettingsDialog
)

__all__ = [
    # Main application
    'OPERAXN',

    # Configuration classes
    'VisualiserConfig',
    'PlotConfig',
    'OPERAXNTheme',

    # Data processing
    'process_paths',
    'make_oned_arrays',
    'make_twod_arrays',
    'make_echem_arrays',
    'get_correlated_data',
    'add_standard_echem_files',
    'get_scan_time_positions',

    # Plotting
    'plot_oned_data',
    'plot_twod_data',
    'plot_echem_data',
    'plot_neutron_data',
    'create_figure_layout',
    'export_single_scan',
    'clear_plot_cache',
    'clear_plot_axes',

    # Analysis
    'assign_cycles',
    'compute_capacity',
    'plot_capacity_vs_voltage',
    'plot_time_vs_voltage',
    'CapacityWindow',
    'ICIWindow',
    'HeatmapWindow',

    # Dialogs
    'UploadOptionsDialog',
    'PlotSettingsDialog',
    'ExportOptionsDialog',
    'GIFSettingsDialog',

    # Enums
    'DataSourceType',
    'UIState',
    'TimeMethod',

    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]
