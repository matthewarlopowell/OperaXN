"""
OperaXN: operando diffraction visualisation with NeXus-canonical storage.

The GUI (this package) is a thin layer over the shared `operaxn.core` pipeline:
every raw-data upload is generated into a canonical .nxs file and the
interface works entirely from that file; opening a .nxs directly skips
generation. See `operaxn.core` for the pipeline itself.

The names below import their module on first use, so `import operaxn.core`
loads no Tk or matplotlib.
"""

import importlib
from typing import Any, List

__license__ = "MIT"

# Public name -> the submodule it comes from
_EXPORTS = {
    # Main application
    'OPERAXN': 'gui',

    # Configuration classes
    'VisualiserConfig': 'gui',
    'PlotConfig': 'output',
    'OPERAXNTheme': 'config',

    # Data processing
    'process_paths': 'input',
    'make_oned_arrays': 'input',
    'make_twod_arrays': 'input',
    'make_echem_arrays': 'input',
    'get_correlated_data': 'input',
    'add_standard_echem_files': 'input',
    'get_scan_time_positions': 'output',

    # Plotting
    'plot_oned_data': 'output',
    'plot_twod_data': 'output',
    'plot_echem_data': 'output',
    'plot_neutron_data': 'output',
    'create_figure_layout': 'output',
    'export_single_scan': 'output',
    'clear_plot_cache': 'output',
    'clear_plot_axes': 'output',

    # Analysis
    'assign_cycles': 'capacity',
    'compute_capacity': 'capacity',
    'plot_capacity_vs_voltage': 'capacity',
    'plot_time_vs_voltage': 'capacity',
    'CapacityWindow': 'capacity',
    'ICIWindow': 'ici',
    'HeatmapWindow': 'heatmap',

    # Dialogs
    'UploadOptionsDialog': 'dialog',
    'PlotSettingsDialog': 'dialog',
    'ExportOptionsDialog': 'dialog',
    'GIFSettingsDialog': 'dialog',

    # Enums
    'DataSourceType': 'core',
    'UIState': 'gui',
    'TimeMethod': 'core',
}

# Version info -> its name in operaxn.config
_VERSION_INFO = {
    '__version__': 'APP_VERSION',
    '__author__': 'APP_AUTHOR',
    '__email__': 'SUPPORT_EMAIL',
}

__all__ = [*_EXPORTS, *_VERSION_INFO, '__license__']


def __getattr__(name: str) -> Any:
    """Import a public name's module on first access and keep the value."""
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    elif name in _VERSION_INFO:
        value = getattr(importlib.import_module(".config", __name__), _VERSION_INFO[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """List the module's own names with the public ones not loaded yet."""
    return sorted(set(globals()) | set(__all__))
