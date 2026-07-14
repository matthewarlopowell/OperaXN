"""
OperaXN application entry point: CLI parsing, logging setup, dependency
checks, splash screen, and window lifecycle management.
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import List, Optional, Tuple, Callable

from .config import (
    DEBUG_MODE,
    APP_NAME,
    APP_VERSION as __version__,
    APP_COPYRIGHT,
    OPERAXNTheme,
    LOG_FORMAT,
    LOG_FILE
)
from .gui import OPERAXN

# ============================================================================
# Constants
# ============================================================================

SPLASH_DURATION_MS = 2500
SPLASH_WIDTH = 500
SPLASH_HEIGHT = 350

REQUIRED_DEPENDENCIES = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'h5py': 'H5py'
}

OPTIONAL_DEPENDENCIES = {
    'fabio': 'Fabio (for EDF file support)',
    'imageio': 'ImageIO (for GIF creation)',
    'psutil': 'PSutil (for performance monitoring)',
    'openpyxl': 'OpenPyXL (for Excel export)'
}


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the application logger."""
    level = logging.DEBUG if (debug or DEBUG_MODE) else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    # Add file handler if configured
    if LOG_FILE and (debug or DEBUG_MODE):
        file_handler = logging.FileHandler(LOG_FILE, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(file_handler)

    # Set modules to debug
    logging.getLogger("operaxn").setLevel(level)
    logging.getLogger("operaxn.input").setLevel(level)
    logging.getLogger("operaxn.output").setLevel(level)
    logging.getLogger("operaxn.gui").setLevel(level)

    # Suppress matplotlib debug
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug("Logging configured - Debug mode ENABLED")
    return logger


def check_dependencies() -> Tuple[List[str], List[str]]:
    """Return (missing_required, missing_optional) dependency names."""
    missing_required = []
    missing_optional = []

    for module, name in REQUIRED_DEPENDENCIES.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(name)

    for module, name in OPTIONAL_DEPENDENCIES.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(name)

    return missing_required, missing_optional


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex colour string to (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# ============================================================================
# Base Window Manager
# ============================================================================

class WindowManager:
    """Shared window positioning and configuration utilities."""

    @staticmethod
    def center_window(window: tk.Misc, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Centre a window on the screen."""
        window.update_idletasks()

        # Get dimensions
        w = width or window.winfo_width()
        h = height or window.winfo_height()

        # Calculate position
        x = (window.winfo_screenwidth() // 2) - (w // 2)
        y = (window.winfo_screenheight() // 2) - (h // 2)

        # Ensure on screen
        x = max(0, min(x, window.winfo_screenwidth() - w))
        y = max(0, min(y, window.winfo_screenheight() - h))

        window.geometry(f"{w}x{h}+{x}+{y}" if width else f"+{x}+{y}")

    @staticmethod
    def configure_window(window: tk.Misc, title: Optional[str] = None,
                         topmost: bool = False, alpha: Optional[float] = None,
                         decorations: bool = True) -> None:
        """Apply title, topmost, alpha, and decoration settings to a window."""
        if title:
            window.title(title)

        if not decorations:
            window.overrideredirect(True)

        if topmost:
            window.attributes('-topmost', True)

        if alpha is not None:
            try:
                window.attributes('-alpha', alpha)
            except (tk.TclError, AttributeError):
                pass  # Alpha not supported on all platforms


# ============================================================================
# Enhanced Splash Screen
# ============================================================================

class EnhancedSplashScreen(WindowManager):
    """Animated splash screen shown during application startup."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialise splash screen."""
        self.root = root
        self.width = SPLASH_WIDTH
        self.height = SPLASH_HEIGHT
        self.splash = None
        self.canvas = None
        self.animation_items = []
        self.animation_running = False
        self.animations = {}  # Store animation callbacks

        # Progress tracking
        self.progress_var = tk.DoubleVar(value=0)
        self.status_text = tk.StringVar(value="Initialising...")

        # UI elements
        self.ui_elements = {}

    def show(self) -> tk.Toplevel:
        """Create, display, and return the splash Toplevel window."""
        self.splash = tk.Toplevel(self.root)

        # Configure window
        self.configure_window(
            self.splash,
            title=APP_NAME,
            decorations=False,
            topmost=True,
            alpha=0.95
        )

        self.splash.geometry(f"{self.width}x{self.height}")
        self.center_window(self.splash, self.width, self.height)

        # Create content
        self._create_content()

        # Start animations
        self._start_animations()

        self.splash.update()
        return self.splash

    def _create_content(self) -> None:
        """Build all splash screen visual elements on the canvas."""
        # Main container
        main_frame = tk.Frame(self.splash, bg=OPERAXNTheme.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True)

        # Canvas for graphics
        self.canvas = tk.Canvas(
            main_frame,
            width=self.width,
            height=self.height,
            bg=OPERAXNTheme.COLORS['bg_primary'],
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)

        # Create all visual elements
        self._draw_gradient_background()
        self._create_logo_section()
        self._create_title_section()
        self._create_progress_section()
        self._create_decorative_corners()
        self._create_diffraction_rings()

    def _draw_gradient_background(self) -> None:
        """Draw vertical gradient from primary to secondary background colour."""
        bg_rgb = hex_to_rgb(OPERAXNTheme.COLORS['bg_primary'])
        bg_light_rgb = hex_to_rgb(OPERAXNTheme.COLORS['bg_secondary'])

        for i in range(self.height):
            ratio = i / self.height
            r = int(bg_rgb[0] + (bg_light_rgb[0] - bg_rgb[0]) * ratio)
            g = int(bg_rgb[1] + (bg_light_rgb[1] - bg_rgb[1]) * ratio)
            b = int(bg_rgb[2] + (bg_light_rgb[2] - bg_rgb[2]) * ratio)

            color = f'#{r:02x}{g:02x}{b:02x}'
            self.canvas.create_line(0, i, self.width, i, fill=color, width=1)

    def _create_logo_section(self) -> None:
        """Draw hexagonal logo with rotating inner element."""
        center_x = self.width // 2
        center_y = 76

        # Create hexagons
        for radius, width, color, tag in [
            (35, 3, OPERAXNTheme.COLORS['accent_primary'], 'outer_hex'),
            (20, 2, OPERAXNTheme.COLORS['accent_secondary'], 'inner_hex')
        ]:
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.extend([x, y])

            hex_id = self.canvas.create_polygon(
                points,
                outline=color,
                fill=OPERAXNTheme.COLORS['bg_secondary'] if tag == 'inner_hex' else '',
                width=width,
                tags=tag
            )

            if tag == 'inner_hex':
                self.animation_items.append(('rotate', hex_id, 0))

        # Center dot
        self.canvas.create_oval(
            center_x - 5, center_y - 5,
            center_x + 5, center_y + 5,
            fill=OPERAXNTheme.COLORS['danger'],
            outline=''
        )

    def _create_title_section(self) -> None:
        """Draw title, subtitle, version badge, and copyright text."""
        center_x = self.width // 2

        # Title font
        title_font = (OPERAXNTheme._SANS, 36, 'bold')
        subtitle_font = (OPERAXNTheme._SANS, 11)
        version_font = (OPERAXNTheme._SANS, 9)

        # Split title for coloring
        opera_text = "OPERA"
        xn_text = "XN"

        # Measure each part so placement is font-independent
        temp = self.canvas.create_text(0, 0, text=opera_text, font=title_font)
        opera_width = (self.canvas.bbox(temp)[2] - self.canvas.bbox(temp)[0]) if self.canvas.bbox(temp) else 120
        self.canvas.delete(temp)

        temp = self.canvas.create_text(0, 0, text=xn_text, font=title_font)
        xn_width = (self.canvas.bbox(temp)[2] - self.canvas.bbox(temp)[0]) if self.canvas.bbox(temp) else 60
        self.canvas.delete(temp)

        # The differently coloured title runs should read as one wordmark.
        gap = 0
        total_width = opera_width + gap + xn_width
        opera_x = center_x - total_width // 2 + opera_width // 2
        xn_x = center_x + total_width // 2 - xn_width // 2

        self.canvas.create_text(
            opera_x, 151,
            text=opera_text,
            font=title_font,
            fill=OPERAXNTheme.COLORS['text_primary'],
            anchor='center'
        )

        self.canvas.create_text(
            xn_x, 151,
            text=xn_text,
            font=title_font,
            fill=OPERAXNTheme.COLORS['danger'],
            anchor='center'
        )

        # Subtitle
        self.canvas.create_text(
            center_x, 204,
            text="OPERAndo X-ray and Neutron data visualisation tool",
            font=subtitle_font,
            fill=OPERAXNTheme.COLORS['text_secondary'],
            anchor='center'
        )

        # Version badge
        self._create_version_badge(center_x, 221, 231, version_font)

        # Copyright
        if APP_COPYRIGHT:
            self.canvas.create_text(
                center_x, 255,
                text=f"© {APP_COPYRIGHT}",
                font=(OPERAXNTheme._SANS, 8),
                fill=OPERAXNTheme.COLORS['text_secondary'],
                anchor='center'
            )

    def _create_version_badge(self, center_x: int, y_top: int, y_text: int, font: Tuple) -> None:
        """Draw bordered version label at the given position."""
        self.canvas.create_rectangle(
            center_x - 40, y_top,
            center_x + 40, y_top + 20,
            fill=OPERAXNTheme.COLORS['bg_secondary'],
            outline=OPERAXNTheme.COLORS['accent_primary'],
            width=1
        )

        self.canvas.create_text(
            center_x, y_text,
            text=f"v{__version__}",
            font=font,
            fill=OPERAXNTheme.COLORS['accent_primary'],
            anchor='center'
        )

    def _create_progress_section(self) -> None:
        """Draw progress bar, status label, and loading dots."""
        center_x = self.width // 2
        bar_width = 300
        bar_height = 5
        bar_y = 274

        # Progress bar background
        self.canvas.create_rectangle(
            center_x - bar_width // 2, bar_y,
            center_x + bar_width // 2, bar_y + bar_height,
            fill=OPERAXNTheme.COLORS['bg_secondary'],
            outline=''
        )

        # Progress bar fill
        self.ui_elements['progress_rect'] = self.canvas.create_rectangle(
            center_x - bar_width // 2, bar_y,
            center_x - bar_width // 2, bar_y + bar_height,
            fill=OPERAXNTheme.COLORS['accent_primary'],
            outline=''
        )

        # Progress glow
        self.ui_elements['progress_glow'] = self.canvas.create_rectangle(
            center_x - bar_width // 2 - 2, bar_y - 2,
            center_x - bar_width // 2 + 2, bar_y + bar_height + 2,
            fill='',
            outline=OPERAXNTheme.COLORS['accent_primary'],
            width=1,
            state='hidden'
        )

        # Status text
        self.ui_elements['status_label'] = self.canvas.create_text(
            center_x, 298,
            text="Initialising application...",
            font=(OPERAXNTheme._SANS, 9),
            fill=OPERAXNTheme.COLORS['text_secondary'],
            anchor='center'
        )

        # Loading dots
        self.ui_elements['loading_dots'] = self.canvas.create_text(
            center_x, 319,
            text="",
            font=(OPERAXNTheme._SANS, 9),
            fill=OPERAXNTheme.COLORS['accent_primary'],
            anchor='center'
        )

        # Store dimensions
        self.ui_elements['bar_width'] = bar_width
        self.ui_elements['bar_center_x'] = center_x

    def _create_decorative_corners(self) -> None:
        """Draw accent-coloured corner brackets."""
        corners = [
            # Top-left
            [(10, 10, 40, 10), (10, 10, 10, 40)],
            # Top-right
            [(self.width - 40, 10, self.width - 10, 10),
             (self.width - 10, 10, self.width - 10, 40)],
            # Bottom-left
            [(10, self.height - 10, 40, self.height - 10),
             (10, self.height - 40, 10, self.height - 10)],
            # Bottom-right
            [(self.width - 40, self.height - 10, self.width - 10, self.height - 10),
             (self.width - 10, self.height - 40, self.width - 10, self.height - 10)]
        ]

        for corner in corners:
            for coords in corner:
                self.canvas.create_line(
                    *coords,
                    fill=OPERAXNTheme.COLORS['accent_primary'],
                    width=2
                )

    def _create_diffraction_rings(self) -> None:
        """Create hidden diffraction ring ovals for later animation."""
        center_x = self.width // 2
        center_y = self.height // 2 - 30

        for i in range(5):
            radius = 40 + i * 30
            ring = self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline=OPERAXNTheme.COLORS['accent_primary'],
                width=1,
                state='hidden'
            )
            self.animation_items.append(('ring', ring, i))

    def _start_animations(self) -> None:
        """Schedule all splash animation callbacks."""
        self.animation_running = True

        # Define animation schedules
        animations = [
            (30, self._animate_progress),
            (500, self._animate_loading_dots),
            (2000, self._animate_rings),
            (50, self._animate_rotation)
        ]

        for delay, callback in animations:
            self._schedule_animation(delay, callback)

    def _schedule_animation(self, delay: int, callback: Callable) -> None:
        """Queue a callback on the splash window if still alive."""
        if self.animation_running and self.splash and self.splash.winfo_exists():
            self.splash.after(delay, callback)

    def _animate_progress(self) -> None:
        """Advance the progress bar fill by one step."""
        if not self.animation_running or not self.splash.winfo_exists():
            return

        current = self.progress_var.get()
        if current < 100:
            current += 3
            self.progress_var.set(min(current, 100))

            # Update bar
            progress_width = (current / 100) * self.ui_elements['bar_width']
            x1 = self.ui_elements['bar_center_x'] - self.ui_elements['bar_width'] // 2
            x2 = x1 + progress_width

            coords = self.canvas.coords(self.ui_elements['progress_rect'])
            if coords:
                self.canvas.coords(
                    self.ui_elements['progress_rect'],
                    coords[0], coords[1], x2, coords[3]
                )

                # Update glow
                if current > 0:
                    self.canvas.itemconfig(self.ui_elements['progress_glow'], state='normal')
                    self.canvas.coords(
                        self.ui_elements['progress_glow'],
                        x2 - 2, coords[1] - 2, x2 + 2, coords[3] + 2
                    )

        self._schedule_animation(30, self._animate_progress)

    def _animate_loading_dots(self) -> None:
        """Cycle the loading-dot indicator text."""
        if not self.animation_running or not self.splash.winfo_exists():
            return

        current = self.canvas.itemcget(self.ui_elements['loading_dots'], 'text')
        new_text = "" if len(current) >= 3 else current + "•"
        self.canvas.itemconfig(self.ui_elements['loading_dots'], text=new_text)

        self._schedule_animation(500, self._animate_loading_dots)

    def _animate_rings(self) -> None:
        """Flash diffraction ring ovals in sequence."""
        if not self.animation_running or not self.splash.winfo_exists():
            return

        for item_type, item_id, index in self.animation_items:
            if item_type == 'ring':
                state = self.canvas.itemcget(item_id, 'state')
                if state == 'hidden':
                    self.canvas.itemconfig(item_id, state='normal')
                    self.splash.after(
                        100 * index,
                        lambda id=item_id: self.canvas.itemconfig(id, state='hidden')
                        if self.splash.winfo_exists() else None
                    )

        self._schedule_animation(2000, self._animate_rings)

    def _animate_rotation(self) -> None:
        """Rotate the inner hexagon by one angular step."""
        if not self.animation_running or not self.splash.winfo_exists():
            return

        center_x = self.width // 2
        center_y = 80

        for i, (item_type, item_id, angle) in enumerate(self.animation_items):
            if item_type == 'rotate':
                new_angle = (angle + 2) % 360
                self.animation_items[i] = (item_type, item_id, new_angle)

                # Calculate new points
                points = []
                for j in range(6):
                    rot_angle = math.pi / 3 * j + math.radians(new_angle)
                    x = center_x + 20 * math.cos(rot_angle)
                    y = center_y + 20 * math.sin(rot_angle)
                    points.extend([x, y])

                self.canvas.coords(item_id, *points)

        self._schedule_animation(50, self._animate_rotation)

    def update_status(self, message: str, progress: Optional[float] = None) -> None:
        """Set the status label text and optionally the progress value."""
        if self.splash and self.splash.winfo_exists():
            self.canvas.itemconfig(self.ui_elements['status_label'], text=message)
            if progress is not None:
                self.progress_var.set(progress)

    def close(self) -> None:
        """Fade out and destroy the splash window."""
        self.animation_running = False
        if self.splash and self.splash.winfo_exists():
            # Fade out effect
            try:
                for i in range(20, 0, -1):
                    self.splash.attributes('-alpha', i / 20)
                    self.splash.update()
                    time.sleep(0.01)
            except (tk.TclError, RuntimeError):
                pass

            self.splash.destroy()


# ============================================================================
# Application Manager
# ============================================================================

class ApplicationManager:
    """Orchestrates startup, splash, and main-window lifecycle."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise application manager."""
        self.args = args
        self.logger = setup_logging(args.debug or DEBUG_MODE)
        self.root = None
        self.app = None
        self.splash_manager = None
        self.initialization_complete = False

    def run(self) -> int:
        """Launch the GUI event loop; return 0 on success, 1 on error."""
        try:
            # Validate environment
            if not self._validate_environment():
                return 1

            # Initialise UI
            self._initialise_ui()

            # Schedule application initialization
            self.root.after(SPLASH_DURATION_MS - 200, self._initialise_application)
            self.root.after(SPLASH_DURATION_MS, self._show_main_window)

            # Start main loop
            self.logger.info("Starting main event loop")
            self.root.mainloop()

            self.logger.info("Application closed normally")
            return 0

        except Exception as e:
            self.logger.exception("Fatal error occurred")
            self._show_error(f"An unexpected error occurred:\n\n{str(e)}")
            return 1

    def _validate_environment(self) -> bool:
        """Check required dependencies; return False if any are missing."""
        missing_required, missing_optional = check_dependencies()

        if missing_required:
            self._show_error(
                "Missing required dependencies:\n\n" +
                "\n".join(f"• {dep}" for dep in missing_required) +
                "\n\nPlease install using:\npip install -r requirements.txt"
            )
            return False

        if missing_optional and (self.args.debug or DEBUG_MODE):
            self.logger.warning("Optional dependencies not installed: %s",
                                ", ".join(missing_optional))

        if sys.platform == 'darwin' and tk.TkVersion < 8.6:
            self._show_error(
                f"Tcl/Tk {tk.TkVersion} detected.\n\n"
                "OperaXN requires Tcl/Tk 8.6 or newer on macOS. "
                "The system Python does not include a compatible version.\n\n"
                "Fix: install Python from https://www.python.org, "
                "create a virtual environment with it, and reinstall OperaXN."
            )
            return False

        return True

    def _initialise_ui(self) -> None:
        """Create the root Tk window and show the splash screen."""
        # Create root window
        self.root = tk.Tk()
        self.root.withdraw()  # Hide initially

        # Configure root
        self.root.title(f"{APP_NAME} v{__version__}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Set icon if available
        self._set_window_icon()

        # Show splash screen
        self._show_splash()

    def _show_splash(self) -> None:
        """Display the splash screen and schedule progress status updates."""
        self.splash_manager = EnhancedSplashScreen(self.root)
        self.splash_manager.show()

        # Schedule status updates
        updates = [
            (100, "Checking dependencies...", 15),
            (700, "Initialising GUI components...", 50),
            (1000, "Loading visualisation engine...", 70),
            (1300, "Preparing workspace...", 85),
            (1600, "Ready to launch!", 100)
        ]

        for delay, message, progress in updates:
            self.root.after(delay, lambda m=message, p=progress:
                            self.splash_manager.update_status(m, p))

    def _initialise_application(self) -> None:
        """Create the OPERAXN GUI instance."""
        if self.initialization_complete:
            return

        self.logger.info("Initialising %s v%s", APP_NAME, __version__)

        if self.splash_manager:
            self.splash_manager.update_status("Creating application interface...", 95)

        try:
            self.app = OPERAXN(master=self.root)
            self.initialization_complete = True
            self.logger.info("Application initialised successfully")
        except Exception as e:
            self.logger.exception("Failed to initialise application")
            self._show_error(f"Failed to initialise:\n\n{str(e)}")
            raise

    def _show_main_window(self) -> None:
        """Close the splash and reveal the main application window."""
        if not self.initialization_complete:
            self._initialise_application()

        if self.splash_manager:
            self.splash_manager.close()

        # Position and show
        WindowManager.center_window(self.root)
        self.root.deiconify()
        self.root.update()
        self.root.lift()
        self.root.focus_force()

        # macOS: force the app to the foreground
        if sys.platform == 'darwin':
            try:
                from AppKit import NSApplication
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except ImportError:
                subprocess.Popen(
                    ['osascript', '-e',
                     'tell application "System Events" to set frontmost of '
                     f'every process whose unix id is {os.getpid()} to true'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

    def _set_window_icon(self) -> None:
        """Load and apply the window icon if the asset exists."""
        try:
            icon_path = Path(__file__).parent / 'assets' / 'icon.png'
            if icon_path.exists():
                photo = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, photo)
        except (tk.TclError, FileNotFoundError, OSError):
            pass  # Icon optional

    def _on_closing(self) -> None:
        """Clear caches and destroy the root window."""
        try:
            if self.app:
                # Clear caches
                from .output import clear_plot_cache
                from .input import clear_global_cache, cleanup_session_cache
                clear_plot_cache()
                clear_global_cache()
                cleanup_session_cache()
        except Exception as e:
            self.logger.debug("Error clearing caches on close: %s", e)

        self.root.quit()
        self.root.destroy()

    def _show_error(self, message: str) -> None:
        """Display a fatal-error messagebox, falling back to stderr."""
        try:
            if not self.root:
                self.root = tk.Tk()
                self.root.withdraw()
            messagebox.showerror("Fatal Error", message)
        except Exception:
            print(f"FATAL ERROR: {message}", file=sys.stderr)


# ============================================================================
# Information Display Functions
# ============================================================================

def display_dependency_check() -> int:
    """Print dependency status to stdout; return 1 if required deps are missing."""
    missing_required, missing_optional = check_dependencies()

    print("\n" + "=" * 60)
    print(f"{APP_NAME} Dependency Check")
    print("=" * 60 + "\n")

    if missing_required:
        print("❌ Missing required dependencies:")
        for dep in missing_required:
            print(f"   - {dep}")
        print("\nPlease install using:")
        print("   pip install -r requirements.txt")
        return 1
    else:
        print("✓ All required dependencies installed")

    if missing_optional:
        print("\n⚠ Optional dependencies not installed:")
        for dep in missing_optional:
            print(f"   - {dep}")
    else:
        print("✓ All optional dependencies installed")

    print("\n" + "=" * 60)
    return 0


# ============================================================================
# Command Line Interface
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} - OPERAndo X-ray and Neutron data visualisation tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Launch GUI
  %(prog)s --debug                   # Launch with debug logging
  %(prog)s --check-deps              # Check dependencies and exit
        """
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'{APP_NAME} {__version__}'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )

    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )

    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main(args: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and launch OperaXN; return exit code."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Handle info commands
    if parsed_args.check_deps:
        return display_dependency_check()

    # Run application
    app_manager = ApplicationManager(parsed_args)
    return app_manager.run()


if __name__ == "__main__":
    sys.exit(main())
