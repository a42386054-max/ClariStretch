"""
ClariStretch
============
A linear-workflow astrophotography image processor: FITS/OSC-camera ingestion,
background extraction, wavelet denoising with a star-protecting mask, an arcsinh
midtone stretch, RGB channel alignment, and 8/16-bit export - all backed by
QThread workers so the UI never blocks on full-resolution processing.
"""

import sys
import os
import gc
import json
import re
import multiprocessing
import warnings

# ClariStretch never uses astropy.samp (SAMP messaging) - only astropy.io.fits - so this
# warning can never indicate anything we need to act on. It mainly shows up as PyInstaller
# build-log noise (see README: PyInstaller's astropy hook eagerly imports every astropy
# submodule, including samp, which unconditionally warns on import). Filtering it here
# covers the rare case it also fires during a normal (non-frozen) run.
warnings.filterwarnings("ignore", message=r".*astropy\.samp was deprecated.*")

# --------------------------------------------------------------------------------------
# Dependency imports. Every import that can plausibly be missing from a user's Python
# environment is wrapped here so the failure is captured (as a pip package name + the
# underlying exception) instead of raised immediately. sys.exit()'s message only ever
# reaches a console, and a PyInstaller --windowed build has no console - so left
# unhandled, a missing dependency would make the app silently fail to launch with no
# visible sign anything went wrong. Once we reach __main__ and a QApplication exists,
# every failure collected in _MISSING_DEPENDENCIES is reported in one QMessageBox dialog
# instead. A sentinel (None) is bound for each failed import's name so later code in this
# module - including class bodies evaluated at *module* import time, like
# ClariStretch._BAYER_CV_CODE below - can be written defensively against it rather than
# crashing with a NameError before __main__ is ever reached.
# --------------------------------------------------------------------------------------
_MISSING_DEPENDENCIES = []  # list of (pip_package, ImportError) tuples, in import order

try:
    import cv2
except ImportError as e:
    cv2 = None
    _MISSING_DEPENDENCIES.append(("opencv-python-headless", e))

try:
    import numpy as np
except ImportError as e:
    np = None
    _MISSING_DEPENDENCIES.append(("numpy", e))

try:
    from skimage.restoration import denoise_wavelet
except ImportError as e:
    denoise_wavelet = None
    _MISSING_DEPENDENCIES.append(("scikit-image PyWavelets", e))

try:
    import threadpoolctl
    _HAS_THREADPOOLCTL = True
except ImportError:
    # threadpoolctl lets us dynamically cap native BLAS/OpenMP thread pools (used
    # internally by numpy/scikit-image) at runtime. Without it, the CPU-cores slider
    # still works for OpenCV's own operations via cv2.setNumThreads, just not for
    # numpy's underlying linear-algebra threads. Genuinely optional - never added to
    # _MISSING_DEPENDENCIES or treated as fatal.
    _HAS_THREADPOOLCTL = False

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError as e:
    Figure = None
    FigureCanvasQTAgg = None
    _MISSING_DEPENDENCIES.append(("matplotlib", e))

# PyQt6 itself is the one dependency this file cannot work around: without it there is no
# GUI toolkit available to draw a QMessageBox with, so a missing PyQt6 has to stay a
# console/stderr failure (sys.exit below) rather than a dialog - there's nothing left to
# show a dialog with.
try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
    from PyQt6.QtGui import QImage, QPixmap, QIcon
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
        QPushButton, QLabel, QCheckBox, QComboBox, QSlider, QFileDialog, QScrollArea,
        QSizePolicy, QProgressBar, QMessageBox, QMenu,
    )
except ImportError as e:
    sys.exit(
        "ClariStretch requires PyQt6, but it isn't installed in this Python environment.\n"
        "Fix with:\n"
        "    pip install PyQt6\n"
        f"(underlying import error: {e})"
    )


def resource_path(relative_path):
    """Resolve a bundled resource's path, working both in a normal dev run and
    inside a frozen PyInstaller .exe (which unpacks data into sys._MEIPASS)."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def _load_app_icon():
    """Load the ClariStretch app/window icon from assets/icon.png.

    Returns an empty QIcon (Qt's harmless no-op icon) if the file can't be found or
    read, rather than raising - a missing icon should never be a reason the app fails
    to start. When packaging with PyInstaller, assets/icon.png must be bundled via
    --add-data (see README) so resource_path() can find it at sys._MEIPASS at runtime;
    --icon alone only sets the .exe file's own icon, not the running window's.
    """
    icon_path = resource_path(os.path.join("assets", "icon.png"))
    if not os.path.isfile(icon_path):
        return QIcon()
    return QIcon(icon_path)


# --------------------------------------------------------------------------------------
# Worker threads. Each one does pure numpy/cv2 computation and NEVER touches a Qt widget
# directly - Qt widgets may only be touched from the main (GUI) thread. Results travel
# back to the main thread via pyqtSignal, which Qt automatically delivers through a
# thread-safe queued connection when the emitting thread differs from the receiving one.
# This is the direct equivalent of the old Tkinter version's queue.Queue + root.after()
# polling loop, just using Qt's native mechanism for it.
# --------------------------------------------------------------------------------------

class BackgroundExtractionWorker(QThread):
    finished_ok = pyqtSignal(object, bool, str)  # gradient_removed array, did_extract, filename
    failed = pyqtSignal(str)

    def __init__(self, compute_fn, raw_image, do_extract, filename, star_mask_params=None):
        super().__init__()
        self._compute_fn = compute_fn
        self._raw_image = raw_image
        self._do_extract = do_extract
        self._filename = filename
        self._star_mask_params = star_mask_params

    def run(self):
        try:
            result = self._compute_fn(self._raw_image, self._do_extract, self._star_mask_params)
            self.finished_ok.emit(result, self._do_extract, self._filename)
        except Exception as e:
            self.failed.emit(str(e))


class ExportWorker(QThread):
    finished_ok = pyqtSignal(str, bool)
    failed = pyqtSignal(str)

    def __init__(self, apply_fn, gradient_removed, slider_values, output_bit_depth, out_path):
        super().__init__()
        self._apply_fn = apply_fn
        self._gradient_removed = gradient_removed
        self._slider_values = slider_values
        self._output_bit_depth = output_bit_depth
        self._out_path = out_path

    def run(self):
        try:
            # Re-run the pipeline at full resolution here rather than reusing the
            # preview's processed_image, which comes from the downsampled preview_source.
            full_res_output = self._apply_fn(
                self._gradient_removed, shift_scale=1.0, output_bit_depth=self._output_bit_depth,
                **self._slider_values,
            )
            success = bool(cv2.imwrite(self._out_path, full_res_output))
            self.finished_ok.emit(self._out_path, success)
        except Exception as e:
            self.failed.emit(str(e))


class DenoiseWorker(QThread):
    """Runs the full-resolution denoise + star-mask pipeline off the main thread.

    CPU throttling (point 8 of the spec) happens here: cv2.setNumThreads caps
    OpenCV's own internal threading, and threadpoolctl.threadpool_limits (when
    available) caps the native BLAS/OpenMP thread pools numpy and scikit-image's
    wavelet transform use underneath. Both are scoped to just this run() call.
    """
    finished_ok = pyqtSignal(object, object)  # denoised_linear_image, star_mask
    failed = pyqtSignal(str)

    def __init__(self, apply_fn, linear_image, params, cores):
        super().__init__()
        self._apply_fn = apply_fn
        self._linear_image = linear_image
        self._params = params
        self._cores = max(1, int(cores))

    def run(self):
        try:
            cv2.setNumThreads(self._cores)
            if _HAS_THREADPOOLCTL:
                with threadpoolctl.threadpool_limits(limits=self._cores):
                    result, mask = self._apply_fn(self._linear_image, **self._params)
            else:
                result, mask = self._apply_fn(self._linear_image, **self._params)
            self.finished_ok.emit(result, mask)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            # Explicit cleanup of this run's large temporary arrays (LAB planes, wavelet
            # intermediates, the 8-bit mask proxy) now that they're out of scope.
            gc.collect()


class ClickableImageLabel(QLabel):
    """A QLabel that reports click coordinates (in its own widget space) for the
    click-to-preview-crop feature."""
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        pos = event.position()
        self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


class ScaledSlider(QSlider):
    """QSlider is integer-only; this exposes a float get()/set() (like ttk.Scale had)
    by scaling to/from an internal integer range via a fixed divisor."""

    def __init__(self, minimum, maximum, divisor=1, default=None, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._divisor = divisor
        self.setMinimum(round(minimum * divisor))
        self.setMaximum(round(maximum * divisor))
        self.setValue(round((default if default is not None else minimum) * divisor))

    def get(self):
        return self.value() / self._divisor

    def set(self, value):
        self.setValue(round(value * self._divisor))

    @property
    def min_value(self):
        return self.minimum() / self._divisor

    @property
    def max_value(self):
        return self.maximum() / self._divisor


class ClariStretch(QMainWindow):
    # Maps standard CFA pattern names (as written by SharpCap, N.I.N.A., ASCOM/INDI,
    # ZWO/QHY/Player One drivers, etc.) to the matching OpenCV Bayer conversion code.
    # OpenCV names its Bayer codes one pixel offset from the usual RGGB-style naming
    # (its "BayerBG" pattern is the 2x2 tile B G / G R), hence the crossed-over mapping.
    #
    # Built defensively: this class body runs at *module* import time (when Python first
    # parses ClariStretch), which is before __main__ gets a chance to check whether cv2
    # actually imported. If cv2 is missing, cv2 is None (see the dependency-import block
    # above) and we fall back to an empty placeholder dict - it's never actually read,
    # because __main__ shows the missing-dependency dialog and exits before a ClariStretch
    # instance (and therefore any Bayer debayering) is ever created.
    _BAYER_CV_CODE = {
        "RGGB": cv2.COLOR_BayerBG2BGR,
        "BGGR": cv2.COLOR_BayerRG2BGR,
        "GRBG": cv2.COLOR_BayerGB2BGR,
        "GBRG": cv2.COLOR_BayerGR2BGR,
    } if cv2 is not None else {}

    # Interactive stretch/alignment/histogram all run against a downsampled copy of the
    # image rather than the full-resolution data - see _build_preview_source.
    MAX_PREVIEW_DIM = 1200

    # RGB slider range, shared by the UI construction and the auto-align clamp logic.
    ALIGN_SLIDER_RANGE = (-15, 15)

    # Target background brightness (0-1) for auto-stretch, and the robust shadow-clip
    # multiplier (in units of sigma below the median) - both follow the same conventions
    # PixInsight's STF ("AutoStretch") uses for a typical, non-inverted linear image.
    AUTO_STRETCH_TARGET_BG = 0.25
    AUTO_STRETCH_SHADOW_CLIP_SIGMA = 2.8

    # Denoising engine tuning. Luminance and chrominance strength sliders (0-100) map
    # onto a wavelet sigma via these caps - chrominance gets a notably higher ceiling
    # since color-noise splotches tolerate much more aggressive smoothing than luminance
    # detail (faint nebulosity) does before looking mushy.
    MAX_LUMINANCE_SIGMA = 0.08
    MAX_CHROMINANCE_SIGMA = 0.30

    # Star-mask morphology: top-hat kernel size (pixels) used to isolate small bright
    # features from smooth local background, independent of the sensitivity slider.
    STAR_MASK_TOPHAT_KERNEL = 15

    # Side length of the real-time click-to-preview crop used for instant slider feedback.
    PREVIEW_CROP_SIZE = 300

    # File-picker filter and drag-and-drop both need to agree on exactly which
    # extensions ClariStretch can open - kept in one place so they can't drift apart.
    SUPPORTED_LOAD_EXTS = (".tif", ".tiff", ".fits", ".fit", ".png", ".jpg", ".jpeg")
    LOAD_FILE_FILTER = "All Astro Formats (*.tif *.tiff *.fits *.fit *.png *.jpg)"

    # Recent-files list cap - small enough to stay a quick-glance list, generous
    # enough to cover a typical multi-target imaging session.
    MAX_RECENT_FILES = 10

    # Common FITS FILTER/EXTNAME header spellings for each LRGB channel, used to spot
    # a single-filter sub/stack loaded via "Load Stacked Image" (e.g. a Luminance-only
    # frame) so a hint can point toward LRGB Combination instead - narrowband filters
    # (Ha/OIII/SII etc.) deliberately aren't in here, since those aren't part of the
    # LRGB workflow and a mono narrowband frame is a normal, intentional thing to load
    # and process on its own.
    LRGB_FILTER_ALIASES = {
        "l": "L", "lum": "L", "luminance": "L", "clear": "L", "clr": "L", "pan": "L",
        "r": "R", "red": "R",
        "g": "G", "green": "G",
        "b": "B", "blue": "B",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClariStretch")
        self.resize(1250, 850)
        self.setWindowIcon(_load_app_icon())
        self.setAcceptDrops(True)  # drag-and-drop a FITS/image file straight onto the window

        # Image arrays
        self.raw_image = None         # Untouched loaded file
        self.gradient_removed = None  # Base image after background extraction
        self.processed_image = None   # Final display image (stretched + aligned), used for the on-screen preview
        self.preview_source = None    # Downsampled copy of gradient_removed used for interactive editing
        self.preview_scale = 1.0      # preview_source size / full-resolution size
        self.file_path = None

        # Before/after comparison: a frozen copy of preview_source taken the moment a
        # load finishes - before any denoise/stretch/alignment is applied - so the
        # comparison view has a stable "before" to show no matter how much the user
        # goes on to denoise, undo, or re-stretch afterward.
        self._original_preview_source = None

        # Recent files: persisted via QSettings (survives app restarts, unlike an
        # in-memory list), so a repeat session on the same target is one click away.
        self._settings = QSettings("ClariStretch", "ClariStretch")
        stored_recent = self._settings.value("recent_files", [])
        if isinstance(stored_recent, str):  # QSettings collapses a 1-item list to a bare str
            stored_recent = [stored_recent] if stored_recent else []
        self._recent_files = [p for p in (stored_recent or []) if isinstance(p, str)]

        # LRGB combination workflow: separately-loaded L/R/G/B masters, each stored as a
        # float [0,1] mono array (already percentile-normalized, see load_lrgb_channel),
        # independent of self.raw_image until Combine LRGB is clicked. Keeping these
        # loaded (rather than clearing them after a combine) lets the luminance blend
        # strength be retuned and recombined without re-picking files.
        self._lrgb_sources = {"L": None, "R": None, "G": None, "B": None}

        # Set by _load_fits when "Load Stacked Image" loads a genuinely mono FITS
        # frame whose FILTER/EXTNAME header names a single LRGB channel (e.g. a
        # Luminance-only sub) - _on_background_done turns this into an on-screen hint
        # pointing toward LRGB Combination. Reset on every load so a stale hint from
        # an earlier file never survives into the next, unrelated one.
        self._last_single_filter_channel = None

        # Linear denoising workflow state. linear_image is the float32 [0,1] array the
        # denoise engine reads and writes; it stays perfectly linear (no STF/arcsinh
        # stretch applied) and is pushed back into gradient_removed after each change
        # so the existing stretch/alignment/export pipeline sees the denoised result.
        self.linear_image = None
        self.is_denoised = False
        self.pre_denoise_backup = None   # single-slot undo history - never more than one copy
        self.star_mask_full = None
        self._preview_crop_source = None       # current 300x300-ish linear crop for instant feedback
        self._last_preview_pixmap_rect = None  # (x, y, w, h) of the displayed pixmap within preview_label

        # Worker thread handles are kept on self so they aren't garbage-collected while running.
        self._bg_worker = None
        self._export_worker = None
        self._denoise_worker = None
        self._busy = False

        self._setup_ui()

    # ------------------------------------------------------------------ UI construction

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        main_layout.addWidget(self._build_control_panel(), stretch=0)
        main_layout.addWidget(self._build_viz_panel(), stretch=1)

    def _build_control_panel(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # --- File Control ---
        self.load_btn = QPushButton("Load Stacked Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)

        self.recent_files_btn = QPushButton("Recent Files ▾")
        self.recent_files_menu = QMenu(self)
        self.recent_files_btn.setMenu(self.recent_files_menu)
        control_layout.addWidget(self.recent_files_btn)
        self._refresh_recent_files_menu()

        drop_hint = QLabel("Tip: drag and drop a FITS/image file anywhere on this window to load it.")
        drop_hint.setWordWrap(True)
        drop_hint.setStyleSheet("color: #666666;")
        control_layout.addWidget(drop_hint)

        self.info_label = QLabel("No image loaded")
        self.info_label.setWordWrap(True)
        control_layout.addWidget(self.info_label)

        # --- Section 0b: LRGB Combination (optional alternative to Load Stacked Image) ---
        lrgb_group = QGroupBox("LRGB Combination (Optional)")
        lrgb_layout = QVBoxLayout(lrgb_group)
        lrgb_help = QLabel(
            "Load separate L/R/G/B master files and combine them into one color image: "
            "R, G and B provide the color, and a loaded L master replaces just the "
            "luminance/detail channel (the standard LRGB technique) rather than being "
            "layered on top in pixel space. L is optional - R, G and B alone combine "
            "into a plain RGB color image."
        )
        lrgb_help.setWordWrap(True)
        lrgb_help.setStyleSheet("color: #666666;")
        lrgb_layout.addWidget(lrgb_help)

        self._lrgb_channel_labels = {}
        for channel in ("L", "R", "G", "B"):
            row = QHBoxLayout()
            btn = QPushButton(f"Load {channel}" + (" (optional)" if channel == "L" else ""))
            btn.clicked.connect(lambda _=None, c=channel: self.load_lrgb_channel(c))
            status = QLabel("Not loaded")
            status.setStyleSheet("color: #888888;")
            status.setWordWrap(True)
            row.addWidget(btn)
            row.addWidget(status, stretch=1)
            lrgb_layout.addLayout(row)
            self._lrgb_channel_labels[channel] = status

        self.lum_blend_slider = ScaledSlider(0, 100, divisor=1, default=100)
        self.lum_blend_value_label = QLabel(str(self.lum_blend_slider.get()))
        lrgb_layout.addLayout(self._labeled_row("Luminance Blend Strength:", self.lum_blend_value_label))
        lrgb_layout.addWidget(self.lum_blend_slider)
        self.lum_blend_slider.valueChanged.connect(
            lambda _=None: self.lum_blend_value_label.setText(f"{self.lum_blend_slider.get():.0f}"))

        self.combine_lrgb_btn = QPushButton("Combine LRGB")
        self.combine_lrgb_btn.setEnabled(False)  # needs R, G and B loaded first
        self.combine_lrgb_btn.clicked.connect(self.combine_lrgb_clicked)
        lrgb_layout.addWidget(self.combine_lrgb_btn)

        control_layout.addWidget(lrgb_group)

        # --- Section 1: Gradient Removal (Light Pollution) ---
        gradient_group = QGroupBox("Gradient Removal (Light Pollution)")
        gradient_layout = QVBoxLayout(gradient_group)
        self.bg_chk = QCheckBox("Enable Automatic Background Extraction")
        self.bg_chk.stateChanged.connect(self.process_background_and_update)
        gradient_layout.addWidget(self.bg_chk)
        control_layout.addWidget(gradient_group)

        # --- Section 1b: Manual Debayer Override (for FITS with no/nonstandard CFA header) ---
        osc_group = QGroupBox("Color Camera (OSC) Debayering - FITS only")
        osc_layout = QVBoxLayout(osc_group)
        osc_layout.addWidget(QLabel("Bayer Pattern:"))
        self.bayer_combo = QComboBox()
        self.bayer_combo.addItems(["Auto-detect", "None (Mono)", "RGGB", "BGGR", "GRBG", "GBRG"])
        self.bayer_combo.currentTextChanged.connect(self.on_bayer_pattern_change)
        osc_layout.addWidget(self.bayer_combo)
        osc_help = QLabel(
            "Use this if a color camera's FITS file has no BAYERPAT header "
            "(auto-detect finds nothing and the image loads as mono)."
        )
        osc_help.setWordWrap(True)
        osc_help.setStyleSheet("color: #666666;")
        osc_layout.addWidget(osc_help)
        control_layout.addWidget(osc_group)

        # --- Section 1c: Denoising Engine (Linear Workflow) ---
        # Placed before the stretch section deliberately: professional workflow is
        # denoise-then-stretch, and the stretch controls below start disabled until
        # Apply Denoise completes (see _set_stretch_controls_enabled).
        denoise_group = QGroupBox("Denoising Engine (Linear Workflow)")
        denoise_layout = QVBoxLayout(denoise_group)

        self.luminance_slider = ScaledSlider(0, 100, divisor=1, default=20)
        self.luminance_value_label = QLabel(str(self.luminance_slider.get()))
        denoise_layout.addLayout(self._labeled_row("Luminance Strength:", self.luminance_value_label))
        denoise_layout.addWidget(self.luminance_slider)
        self.luminance_slider.valueChanged.connect(self._update_zoom_preview)
        self.luminance_slider.valueChanged.connect(
            lambda _=None: self.luminance_value_label.setText(f"{self.luminance_slider.get():.0f}"))

        self.chrominance_slider = ScaledSlider(0, 100, divisor=1, default=40)
        self.chrominance_value_label = QLabel(str(self.chrominance_slider.get()))
        denoise_layout.addLayout(self._labeled_row("Chrominance Strength:", self.chrominance_value_label))
        denoise_layout.addWidget(self.chrominance_slider)
        self.chrominance_slider.valueChanged.connect(self._update_zoom_preview)
        self.chrominance_slider.valueChanged.connect(
            lambda _=None: self.chrominance_value_label.setText(f"{self.chrominance_slider.get():.0f}"))

        self.min_star_sensitivity_slider = ScaledSlider(0, 100, divisor=1, default=50)
        self.min_star_value_label = QLabel(str(self.min_star_sensitivity_slider.get()))
        denoise_layout.addLayout(self._labeled_row("Min Star Sensitivity:", self.min_star_value_label))
        denoise_layout.addWidget(self.min_star_sensitivity_slider)
        self.min_star_sensitivity_slider.valueChanged.connect(self._update_zoom_preview)
        self.min_star_sensitivity_slider.valueChanged.connect(
            lambda _=None: self.min_star_value_label.setText(f"{self.min_star_sensitivity_slider.get():.0f}"))

        self.star_mask_expand_slider = ScaledSlider(0, 10, divisor=1, default=2)
        self.star_mask_expand_value_label = QLabel(str(self.star_mask_expand_slider.get()))
        denoise_layout.addLayout(self._labeled_row("Star Mask Expand:", self.star_mask_expand_value_label))
        denoise_layout.addWidget(self.star_mask_expand_slider)
        self.star_mask_expand_slider.valueChanged.connect(self._update_zoom_preview)
        self.star_mask_expand_slider.valueChanged.connect(
            lambda _=None: self.star_mask_expand_value_label.setText(f"{self.star_mask_expand_slider.get():.0f}"))

        self.star_mask_blur_slider = ScaledSlider(0, 15, divisor=1, default=3)
        self.star_mask_blur_value_label = QLabel(str(self.star_mask_blur_slider.get()))
        denoise_layout.addLayout(self._labeled_row("Star Mask Blur:", self.star_mask_blur_value_label))
        denoise_layout.addWidget(self.star_mask_blur_slider)
        self.star_mask_blur_slider.valueChanged.connect(self._update_zoom_preview)
        self.star_mask_blur_slider.valueChanged.connect(
            lambda _=None: self.star_mask_blur_value_label.setText(f"{self.star_mask_blur_slider.get():.0f}"))

        mask_toggle_row = QHBoxLayout()
        self.view_mask_chk = QCheckBox("View Mask")
        self.invert_mask_chk = QCheckBox("Invert Mask")
        self.view_mask_chk.stateChanged.connect(self._update_zoom_preview)
        self.invert_mask_chk.stateChanged.connect(self._update_zoom_preview)
        mask_toggle_row.addWidget(self.view_mask_chk)
        mask_toggle_row.addWidget(self.invert_mask_chk)
        denoise_layout.addLayout(mask_toggle_row)

        denoise_layout.addWidget(QLabel("Click the Live Preview to pick a 300x300 zoom spot:"))
        self.zoom_preview_label = QLabel("No crop selected")
        self.zoom_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_preview_label.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa;")
        self.zoom_preview_label.setFixedSize(300, 300)
        denoise_layout.addWidget(self.zoom_preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        cores_row = QHBoxLayout()
        self.cpu_cores_slider = QSlider(Qt.Orientation.Horizontal)
        self.cpu_cores_slider.setMinimum(1)
        self.cpu_cores_slider.setMaximum(max(1, multiprocessing.cpu_count()))
        self.cpu_cores_slider.setValue(max(1, multiprocessing.cpu_count()))
        self.cpu_cores_value_label = QLabel(str(self.cpu_cores_slider.value()))
        self.cpu_cores_slider.valueChanged.connect(lambda v: self.cpu_cores_value_label.setText(str(v)))
        denoise_layout.addLayout(self._labeled_row("Max CPU Cores:", self.cpu_cores_value_label))
        denoise_layout.addWidget(self.cpu_cores_slider)

        self.apply_denoise_btn = QPushButton("Apply Denoise")
        self.apply_denoise_btn.clicked.connect(self.apply_denoise_clicked)
        denoise_layout.addWidget(self.apply_denoise_btn)

        self.undo_denoise_btn = QPushButton("Undo Denoise")
        self.undo_denoise_btn.setEnabled(False)
        self.undo_denoise_btn.clicked.connect(self.undo_denoise_clicked)
        denoise_layout.addWidget(self.undo_denoise_btn)

        self.denoise_progress_bar = QProgressBar()
        self.denoise_progress_bar.setRange(0, 1)
        self.denoise_progress_bar.setValue(0)
        denoise_layout.addWidget(self.denoise_progress_bar)

        control_layout.addWidget(denoise_group)

        # --- Section 2: ArcSinh Stretching ---
        stretch_group = QGroupBox("ArcSinh Midtone Transformation")
        stretch_layout = QVBoxLayout(stretch_group)

        self.asinh_slider = ScaledSlider(1.0, 50.0, divisor=10, default=1.0)
        self.asinh_value_label = QLabel()
        stretch_layout.addLayout(self._labeled_row("Asinh Stretch Factor (Faint Details):", self.asinh_value_label))
        stretch_layout.addWidget(self.asinh_slider)
        self.asinh_slider.valueChanged.connect(self.update_image)
        self.asinh_slider.valueChanged.connect(lambda _=None: self.asinh_value_label.setText(f"{self.asinh_slider.get():.1f}"))

        self.black_slider = ScaledSlider(0.0, 0.3, divisor=1000, default=0.0)
        self.black_value_label = QLabel()
        stretch_layout.addLayout(self._labeled_row("Black Point Clipping (Background Sky):", self.black_value_label))
        stretch_layout.addWidget(self.black_slider)
        self.black_slider.valueChanged.connect(self.update_image)
        self.black_slider.valueChanged.connect(lambda _=None: self.black_value_label.setText(f"{self.black_slider.get():.3f}"))

        self.auto_stretch_btn = QPushButton("Auto-Stretch (from Image Statistics)")
        self.auto_stretch_btn.clicked.connect(self.auto_stretch)
        stretch_layout.addWidget(self.auto_stretch_btn)

        control_layout.addWidget(stretch_group)
        self.asinh_value_label.setText(f"{self.asinh_slider.get():.1f}")
        self.black_value_label.setText(f"{self.black_slider.get():.3f}")

        # --- Section 3: RGB Channel Alignment ---
        rgb_group = QGroupBox("RGB Channel Alignment (Fringe Fix)")
        rgb_layout = QVBoxLayout(rgb_group)

        self.r_x_slider = ScaledSlider(*self.ALIGN_SLIDER_RANGE, divisor=1, default=0)
        self.r_y_slider = ScaledSlider(*self.ALIGN_SLIDER_RANGE, divisor=1, default=0)
        self.b_x_slider = ScaledSlider(*self.ALIGN_SLIDER_RANGE, divisor=1, default=0)
        self.b_y_slider = ScaledSlider(*self.ALIGN_SLIDER_RANGE, divisor=1, default=0)

        self.r_x_value_label = QLabel(str(self.r_x_slider.get()))
        self.r_y_value_label = QLabel(str(self.r_y_slider.get()))
        self.b_x_value_label = QLabel(str(self.b_x_slider.get()))
        self.b_y_value_label = QLabel(str(self.b_y_slider.get()))

        for label_text, slider, value_label, color in [
            ("Red Channel Shift X:", self.r_x_slider, self.r_x_value_label, "red"),
            ("Red Channel Shift Y:", self.r_y_slider, self.r_y_value_label, "red"),
            ("Blue Channel Shift X:", self.b_x_slider, self.b_x_value_label, "blue"),
            ("Blue Channel Shift Y:", self.b_y_slider, self.b_y_value_label, "blue"),
        ]:
            row = self._labeled_row(label_text, value_label, label_color=color)
            rgb_layout.addLayout(row)
            rgb_layout.addWidget(slider)
            slider.valueChanged.connect(self.update_image)
            slider.valueChanged.connect(lambda _=None, s=slider, lbl=value_label: lbl.setText(f"{s.get():.0f}"))

        auto_align_btn = QPushButton("Auto-Align (Cross-Correlate)")
        auto_align_btn.clicked.connect(self.auto_align)
        rgb_layout.addWidget(auto_align_btn)

        reset_alignment_btn = QPushButton("Reset Alignment")
        reset_alignment_btn.clicked.connect(self.reset_alignment)
        rgb_layout.addWidget(reset_alignment_btn)

        control_layout.addWidget(rgb_group)

        # --- Section 3b: Session / Recipe Persistence ---
        recipe_group = QGroupBox("Session / Recipe")
        recipe_layout = QVBoxLayout(recipe_group)
        recipe_help = QLabel(
            "Save every slider above (denoise, stretch, alignment) to a small JSON "
            "file, then reload it later to reprocess the same target, or apply a "
            "known-good setting to a new session, without re-tuning from scratch."
        )
        recipe_help.setWordWrap(True)
        recipe_help.setStyleSheet("color: #666666;")
        recipe_layout.addWidget(recipe_help)

        save_settings_btn = QPushButton("Save Settings...")
        save_settings_btn.clicked.connect(self.save_settings_clicked)
        recipe_layout.addWidget(save_settings_btn)

        load_settings_btn = QPushButton("Load Settings...")
        load_settings_btn.clicked.connect(self.load_settings_clicked)
        recipe_layout.addWidget(load_settings_btn)

        control_layout.addWidget(recipe_group)

        # --- Export Layout ---
        self.export_16bit_chk = QCheckBox("Export as 16-bit (PNG/TIFF only)")
        control_layout.addWidget(self.export_16bit_chk)

        self.save_btn = QPushButton("Export Image")
        self.save_btn.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_btn)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("color: #0066cc;")
        control_layout.addWidget(self.progress_label)

        control_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(control_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _labeled_row(label_text, value_widget, label_color=None):
        """A QHBoxLayout pairing a description label with a live numeric value label."""
        row = QHBoxLayout()
        text_label = QLabel(label_text)
        if label_color:
            text_label.setStyleSheet(f"color: {label_color};")
        row.addWidget(text_label)
        row.addStretch(1)
        row.addWidget(value_widget)
        return row

    def _build_viz_panel(self):
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        preview_header = QHBoxLayout()
        preview_title = QLabel("Live Preview")
        preview_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(QLabel("View:"))
        self.compare_mode_combo = QComboBox()
        self.compare_mode_combo.addItems(["After (Processed)", "Before (Original)", "Split View"])
        self.compare_mode_combo.currentTextChanged.connect(lambda _=None: self.update_preview_display())
        preview_header.addWidget(self.compare_mode_combo)
        viz_layout.addLayout(preview_header)

        # Embedded live preview panel - the direct replacement for the original
        # cv2.imshow/waitKey popup, which ran its own native window outside of the
        # UI toolkit's event loop and was never cleaned up when the app closed.
        self.preview_label = ClickableImageLabel("No image loaded")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa;")
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.clicked.connect(self.on_preview_clicked)
        viz_layout.addWidget(self.preview_label, stretch=3)

        hist_title = QLabel("Live Dynamic Histogram")
        hist_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        viz_layout.addWidget(hist_title)

        self.fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        viz_layout.addWidget(self.canvas, stretch=1)

        return viz_widget

    # ------------------------------------------------------------------ Qt-specific plumbing

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render the (already-computed) preview at the new panel size on window resize.
        if getattr(self, "processed_image", None) is not None:
            self.update_preview_display()

    def closeEvent(self, event):
        for worker in (self._bg_worker, self._export_worker):
            if worker is not None and worker.isRunning():
                worker.wait(2000)
        event.accept()

    def _set_busy(self, busy):
        self._busy = busy
        self.load_btn.setEnabled(not busy)
        self.bg_chk.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)
        self.apply_denoise_btn.setEnabled(not busy)
        self.undo_denoise_btn.setEnabled(not busy and self.pre_denoise_backup is not None)

    @staticmethod
    def _set_slider_silently(slider, value):
        """Set a slider's value without firing update_image for every intermediate change -
        used when several sliders are set together (auto-align, auto-stretch, reset)."""
        slider.blockSignals(True)
        slider.set(value)
        slider.blockSignals(False)

    # ------------------------------------------------------------------ File loading

    def load_image(self):
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Stacked Image", "", self.LOAD_FILE_FILTER,
        )
        if not file_path:
            return
        self._load_file_path(file_path)

    def _load_file_path(self, file_path):
        """Shared tail-end of every way a file can be loaded (the file dialog, a drag-
        and-drop drop, or picking a Recent Files entry): set self.file_path, load the
        raw image, record it as a recent file, and kick off background processing."""
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        self.file_path = file_path
        if not self._load_raw_image_from_current_path():
            return

        self._add_recent_file(file_path)
        self.process_background_and_update()

    # ------------------------------------------------------------------ Drag-and-drop

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(os.path.splitext(u.toLocalFile())[1].lower() in self.SUPPORTED_LOAD_EXTS
               for u in urls if u.isLocalFile()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        valid = [p for p in paths if os.path.splitext(p)[1].lower() in self.SUPPORTED_LOAD_EXTS]
        if not valid:
            event.ignore()
            return
        event.acceptProposedAction()
        # Only one stacked image is ever loaded at a time (same as the Load Stacked
        # Image dialog, which is also single-select) - if several files were dropped
        # together, take the first rather than guessing which one was intended.
        self._load_file_path(valid[0])

    # ------------------------------------------------------------------ Recent files

    def _add_recent_file(self, file_path):
        path = os.path.abspath(file_path)
        self._recent_files = [path] + [p for p in self._recent_files if p != path]
        self._recent_files = self._recent_files[: self.MAX_RECENT_FILES]
        self._settings.setValue("recent_files", self._recent_files)
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self._recent_files:
            placeholder = self.recent_files_menu.addAction("(no recent files)")
            placeholder.setEnabled(False)
            return

        for path in self._recent_files:
            action = self.recent_files_menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.triggered.connect(lambda _=None, p=path: self._open_recent_file(p))

        self.recent_files_menu.addSeparator()
        clear_action = self.recent_files_menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self._clear_recent_files)

    def _open_recent_file(self, path):
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return
        if not os.path.isfile(path):
            self.info_label.setText(f"Recent file no longer exists: {path}")
            self._recent_files = [p for p in self._recent_files if p != path]
            self._settings.setValue("recent_files", self._recent_files)
            self._refresh_recent_files_menu()
            return
        self._load_file_path(path)

    def _clear_recent_files(self):
        self._recent_files = []
        self._settings.setValue("recent_files", [])
        self._refresh_recent_files_menu()

    def on_bayer_pattern_change(self, _text=None):
        """Re-read the current FITS file when the manual debayer override changes."""
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return
        if not self.file_path:
            return
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in (".fits", ".fit"):
            return  # override only applies to FITS files
        if self._load_raw_image_from_current_path():
            self.process_background_and_update()

    def _load_raw_image_from_current_path(self):
        """Load self.file_path into self.raw_image. Returns True on success."""
        filename = os.path.basename(self.file_path)
        ext = os.path.splitext(self.file_path)[1].lower()
        self._last_single_filter_channel = None  # reset each load; _load_fits sets it when relevant

        if ext in (".fits", ".fit"):
            # cv2.imread has no FITS codec - it will always return None for these,
            # so FITS files need to be read through astropy instead.
            try:
                self.raw_image = self._load_fits(self.file_path, forced_pattern=self.bayer_combo.currentText())
            except ImportError:
                self.info_label.setText("FITS support requires astropy. Install with: pip install astropy")
                self.raw_image = None
                return False
            except Exception as e:
                self.info_label.setText(f"Error loading FITS file: {e}")
                self.raw_image = None
                return False
        else:
            self.raw_image = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)

        if self.raw_image is None:
            self.info_label.setText("Error loading file.")
            return False

        self.info_label.setText(f"Loaded: {filename}")
        return True

    # ------------------------------------------------------------------ LRGB combination
    # (Alternative way to populate self.raw_image: instead of one pre-stacked color file,
    # combine separately-loaded single-filter masters. Once combined, the result feeds
    # into process_background_and_update() exactly like a normal Load Stacked Image, so
    # denoising, stretching, alignment and export all work on it unchanged.)

    def load_lrgb_channel(self, channel):
        """Load a single L/R/G/B master file for the LRGB combination workflow.

        Each master is stored independently as a float [0,1] mono array (robust
        percentile-normalized, the same treatment _load_fits already gives a mono FITS
        frame) rather than being combined immediately - this lets masters be loaded in
        any order, replaced individually, and recombined with a different luminance
        blend strength without re-picking files.
        """
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Load {channel} Master", "", self.LOAD_FILE_FILTER,
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in (".fits", ".fit"):
                entries = self._extract_fits_entries(file_path)
                mono_entries = [e for e in entries if e[0] == 2]
                if not mono_entries:
                    raise ValueError("No 2-D image data found in FITS file.")
                # An LRGB master is expected to be a single mono frame; if the file
                # somehow has several 2-D layers, just use the first rather than
                # guessing which one is the intended master.
                data = mono_entries[0][3]
                array = self._normalize_channel(data)
            else:
                raw = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise ValueError("Error loading file.")
                if raw.ndim == 3:
                    # A color file used as an L/R/G/B master is unusual but not invalid -
                    # collapse it to one channel via standard luma weighting rather than
                    # silently picking a single color channel.
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                array = self._normalize_channel(raw.astype(np.float64))
        except Exception as e:
            self._lrgb_channel_labels[channel].setText(f"Error: {e}")
            self._lrgb_channel_labels[channel].setStyleSheet("color: #cc3333;")
            return

        self._lrgb_sources[channel] = array
        h, w = array.shape
        self._lrgb_channel_labels[channel].setText(f"{os.path.basename(file_path)} ({w}x{h})")
        self._lrgb_channel_labels[channel].setStyleSheet("color: #2a8a2a;")

        have_rgb = all(self._lrgb_sources[c] is not None for c in ("R", "G", "B"))
        self.combine_lrgb_btn.setEnabled(have_rgb)

    def combine_lrgb_clicked(self):
        """Combine the loaded R/G/B masters into a color image, optionally blending in
        the loaded L master as the luminance/detail channel, then feed the result into
        the normal load pipeline (background extraction, linear workflow lock, etc.)."""
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        r, g, b = self._lrgb_sources["R"], self._lrgb_sources["G"], self._lrgb_sources["B"]
        if r is None or g is None or b is None:
            self.info_label.setText("Load R, G and B masters before combining (L is optional).")
            return

        l = self._lrgb_sources["L"]
        lum_blend_strength = self.lum_blend_slider.get() / 100.0

        try:
            combined = self._compute_lrgb_combine(r, g, b, l, lum_blend_strength)
        except Exception as e:
            self.info_label.setText(f"LRGB combine failed: {e}")
            return

        self.raw_image = combined
        self._last_single_filter_channel = None  # the result is a combined color image, never a hint candidate
        # No single source file represents a combined image, but process_background_and_
        # update()'s own "Loaded: ..." message (set once the background worker finishes)
        # derives its filename from self.file_path - a descriptive pseudo-path here means
        # that message reads "Loaded: LRGB Combination (L+R+G+B)" instead of being
        # overwritten with a vague "Image processed." the moment that worker completes.
        used = "+".join(c for c in ("L", "R", "G", "B") if self._lrgb_sources[c] is not None)
        self.file_path = f"LRGB Combination ({used})"

        self.process_background_and_update()

    def _compute_lrgb_combine(self, r, g, b, l, lum_blend_strength):
        """Combine normalized [0,1] mono R/G/B channels into a BGR color image.

        LRGB masters routinely differ in resolution (a full-resolution L master shot
        against 2x2-binned RGB is common), so every channel is resized to a common
        target resolution first: the L master's own resolution if one was loaded
        (conventionally the reference frame in an LRGB set), otherwise the largest of
        R/G/B.

        If an L master was loaded, it's blended into the combined image's luminance via
        LAB color space - L carries structural detail (usually the sharpest,
        highest-SNR data in an LRGB set) while R/G/B contribute color (chrominance).
        This is the standard LRGB combination technique: L is substituted for the
        luminance channel, not layered on top of the RGB image in plain pixel space.

        lum_blend_strength (0.0-1.0) controls how much of the result's luminance comes
        from the loaded L master vs. the luminance the R/G/B combine already implies -
        1.0 fully replaces it, 0.0 leaves the RGB-derived luminance untouched.
        """
        if l is not None:
            target_h, target_w = l.shape
        else:
            target_h, target_w = max((c.shape for c in (r, g, b)), key=lambda s: s[0] * s[1])

        def to_target(channel):
            if channel.shape == (target_h, target_w):
                return channel.astype(np.float32)
            is_upscale = channel.shape[0] * channel.shape[1] < target_h * target_w
            interp = cv2.INTER_CUBIC if is_upscale else cv2.INTER_AREA
            return cv2.resize(channel.astype(np.float32), (target_w, target_h), interpolation=interp)

        r_rs, g_rs, b_rs = to_target(r), to_target(g), to_target(b)
        bgr = np.clip(np.stack([b_rs, g_rs, r_rs], axis=-1), 0.0, 1.0).astype(np.float32)

        if l is not None and lum_blend_strength > 0:
            l_rs = np.clip(to_target(l), 0.0, 1.0)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            rgb_luminance = lab[:, :, 0] / 100.0  # [0,1], the luminance R/G/B alone implies
            blended = (1.0 - lum_blend_strength) * rgb_luminance + lum_blend_strength * l_rs
            lab[:, :, 0] = np.clip(blended, 0.0, 1.0) * 100.0
            bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)

        return (np.clip(bgr, 0.0, 1.0) * 65535.0).astype(np.uint16)

    # ------------------------------------------------------------------ FITS handling
    # (Pure computation below - unchanged from the Tkinter version, and still safe to
    # call from a worker thread since none of it touches a widget.)

    @staticmethod
    def _extract_fits_entries(path):
        """Open a FITS file and pull every image HDU's data out into plain numpy arrays.

        Returns a list of (ndim, shape, label, array_copy, bayer_pattern_or_None) tuples -
        _load_fits does the actual color/mono/Bayer interpretation on top of this.

        Modern astropy.io.fits usage: restrict to actual image layers via
        isinstance(ImageHDU / PrimaryHDU) rather than assuming every HDU with data is an
        image (a BinTableHDU also has non-None .data, but it's a FITS_rec, not a plain
        image array), and copy each array out with .data.copy() while the file is still
        open. Every hdu.data / hdu.header access must happen inside the `with` block -
        astropy backs HDU data with a memory map by default, and touching .data again
        after the file closes raises "I/O operation on closed file".

        Tries memmap=True first since it's the memory-efficient path for large stacks,
        but some FITS files - notably ones with BZERO/BSCALE (scaled integer data, used
        to store unsigned values in a signed integer type) and/or a BLANK keyword
        (integer "no data" sentinel) - can't be memory-mapped at all; astropy raises a
        ValueError naming exactly that ("Cannot load a memory-mapped image:
        BZERO/BSCALE/BLANK header keywords present. Set memmap=False.") rather than
        silently handling it. That's not a corrupt file or a bug in this app - it's a
        real limitation of memory-mapping scaled/blanked FITS data - so on that specific
        error this transparently retries the same file with memmap=False (reads the
        whole file into memory instead, which is the only thing memmap=False actually
        changes) rather than surfacing it to the user as a load failure.
        """
        from astropy.io import fits  # imported lazily so astropy is only required for FITS files

        def _read(use_memmap):
            collected = []
            with fits.open(path, memmap=use_memmap) as hdul:
                for hdu in hdul:
                    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                        continue
                    if hdu.data is None:
                        continue
                    arr = hdu.data.copy().astype(np.float64)  # detach before the file closes
                    header = hdu.header
                    label = str(
                        header.get("FILTER", "") or header.get("EXTNAME", "") or getattr(hdu, "name", "") or ""
                    ).strip().lower()

                    bayer = None
                    for key in ("BAYERPAT", "BAYERPTN", "BAYPAT"):
                        val = header.get(key)
                        if val:
                            val_up = str(val).strip().upper()
                            if val_up in ClariStretch._BAYER_CV_CODE:
                                bayer = val_up
                                break

                    collected.append((arr.ndim, arr.shape, label, arr, bayer))
            return collected

        try:
            return _read(use_memmap=True)
        except ValueError as e:
            if "memory-mapped" in str(e).lower() or "BZERO" in str(e) or "BSCALE" in str(e):
                return _read(use_memmap=False)
            raise

    def _load_fits(self, path, forced_pattern="Auto-detect"):
        """Load a FITS file into a normalized uint16 array (BGR order for color data).

        Handles four common layouts:
          - A single 3-D color cube (channels, height, width), RGB order.
          - Separate R/G/B stored as distinct extensions/HDUs (common output from
            per-filter stacking), matched up by FILTER/EXTNAME header or file order.
          - A single 2-D raw frame from a one-shot-color (OSC) camera, still in its
            unprocessed Bayer mosaic. Debayered into BGR using either an auto-detected
            BAYERPAT/BAYERPTN/BAYPAT header keyword, or forced_pattern if the caller
            supplies one (needed since many capture pipelines don't preserve any CFA
            header at all).
          - A single 2-D mono image (no color info at all).

        forced_pattern: "Auto-detect" (default, use the header if present), "None (Mono)"
        (never debayer even if a header pattern is found), or one of "RGGB"/"BGGR"/
        "GRBG"/"GBRG" to force that pattern regardless of the header.

        Per-filter subs are normalized independently, since they're often captured with
        very different exposure times/gains - a single shared normalization range can
        crush two channels toward black or blow one out, making the image look
        essentially monochrome even though the data is genuinely color. A debayered OSC
        frame, by contrast, gets one combined normalization across all three channels,
        since its R/G/B balance comes from the sensor itself and is part of the actual
        color signal - stretching each channel independently there would flatten real
        color balance rather than reveal it.
        """
        entries = self._extract_fits_entries(path)  # list of (ndim, shape, label, array_copy, bayer_or_None)

        if not entries:
            raise ValueError("No image data found in FITS file.")

        two_d = [e for e in entries if e[0] == 2]
        three_d = [e for e in entries if e[0] == 3]

        color_planes = None  # list of 2-D arrays in B, G, R order, if we can identify a triplet

        if len(two_d) >= 3:
            # Group same-shaped 2-D extensions together - the real R/G/B trio should be
            # the largest group sharing a shape (calibration/mask frames are usually alone).
            shape_groups = {}
            for e in two_d:
                shape_groups.setdefault(e[1], []).append(e)
            candidate_shapes = [s for s, es in shape_groups.items() if len(es) >= 3]

            if candidate_shapes:
                shape = max(candidate_shapes, key=lambda s: s[0] * s[1])
                channel_entries = shape_groups[shape]

                r_arr = g_arr = b_arr = None
                for _, _, label, arr, _bayer in channel_entries:
                    if r_arr is None and label in ("r", "red"):
                        r_arr = arr
                    elif g_arr is None and label in ("g", "green"):
                        g_arr = arr
                    elif b_arr is None and label in ("b", "blue"):
                        b_arr = arr

                if not (r_arr is not None and g_arr is not None and b_arr is not None):
                    # Couldn't identify channels from headers - fall back to file order,
                    # assuming the common R, G, B convention.
                    r_arr, g_arr, b_arr = channel_entries[0][3], channel_entries[1][3], channel_entries[2][3]

                color_planes = [b_arr, g_arr, r_arr]

        if color_planes is not None:
            normalized = [self._normalize_channel(p) for p in color_planes]
            return (np.stack(normalized, axis=-1) * 65535.0).astype(np.uint16)

        if three_d:
            # FITS color cubes are conventionally stored as (channels, height, width) in RGB order.
            data = three_d[0][3]
            data = np.transpose(data, (1, 2, 0))
            data = data[:, :, ::-1]  # RGB -> BGR to match the rest of the pipeline (cv2 convention)
            normalized = [self._normalize_channel(data[:, :, i]) for i in range(data.shape[2])]
            return (np.stack(normalized, axis=-1) * 65535.0).astype(np.uint16)

        if two_d:
            _, _, label, data, header_bayer = two_d[0]

            forced = str(forced_pattern or "").strip().upper()
            if forced in self._BAYER_CV_CODE:
                effective_bayer = forced          # user forced a specific pattern
            elif forced in ("NONE (MONO)", "NONE", "MONO"):
                effective_bayer = None            # user forced mono, ignore any header pattern
            else:
                effective_bayer = header_bayer    # "Auto-detect" or anything unrecognized

            if effective_bayer:
                return self._debayer(data, effective_bayer)

            # A genuinely mono frame (no Bayer pattern applied) whose FILTER/EXTNAME
            # header names a single LRGB channel is very likely one sub/stack out of
            # an LRGB set (e.g. a Luminance-only frame), not a finished color image -
            # _load_raw_image_from_current_path surfaces this as a hint once loading
            # completes, pointing toward LRGB Combination instead.
            self._last_single_filter_channel = self._detect_lrgb_filter_label(label)
            return (self._normalize_channel(data) * 65535.0).astype(np.uint16)

        raise ValueError("Unsupported FITS data layout.")

    def _debayer(self, mosaic, bayer_pattern):
        """Turn a raw one-shot-color sensor frame (single-channel CFA mosaic) into BGR."""
        # Normalize the whole mosaic together first - relative pixel values within the
        # tile are what the debayer interpolation actually depends on, and a shared
        # 16-bit range keeps the interpolation well-behaved.
        mosaic_u16 = (self._normalize_channel(mosaic) * 65535.0).astype(np.uint16)
        cv2_code = self._BAYER_CV_CODE[bayer_pattern]
        return cv2.cvtColor(mosaic_u16, cv2_code)  # -> uint16 3-channel BGR

    def _normalize_channel(self, data):
        """Robust 0-1 percentile stretch for a single 2-D channel's raw ADU values."""
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size == 0:
            raise ValueError("FITS data contains no finite pixel values.")

        lo, hi = np.percentile(finite_vals, 0.1), np.percentile(finite_vals, 99.9)
        if hi <= lo:
            lo, hi = float(finite_vals.min()), float(finite_vals.max())
        if hi <= lo:
            hi = lo + 1.0

        return np.clip((data - lo) / (hi - lo), 0, 1)

    @classmethod
    def _detect_lrgb_filter_label(cls, label):
        """Match a FITS FILTER/EXTNAME header value against the common L/R/G/B
        aliases (e.g. "L", "Luminance", "Red") used to tag a single-filter
        calibrated sub/stack. Returns the matching LRGB channel letter, or None if
        the label doesn't look like one of those four - so a narrowband label
        (e.g. "Ha") or anything unrecognized never triggers the hint."""
        if not label:
            return None
        match = re.match(r"[a-zA-Z]+", label.strip())
        if not match:
            return None
        return cls.LRGB_FILTER_ALIASES.get(match.group(0).lower())

    # ------------------------------------------------------------------ Background extraction (threaded)

    def _compute_background_removal(self, raw_image, do_extract, star_mask_params=None):
        """Pure computation, safe to run on a worker thread (touches no Qt widgets).

        star_mask_params, when given, enables star-aware gradient fitting: stars are
        small, very bright, and carry no information about the smooth light-pollution
        gradient this step models, so left in, they bias the downsampled local average
        toward "brighter" wherever a star happens to sit - on star-dense fields that
        shows up as a faint dark halo around stars once the (slightly too-bright)
        background model gets subtracted back out. This excludes star pixels from
        that local average (reusing the same top-hat star-mask logic the denoiser
        already uses, via _compute_star_mask) via a star-confidence-weighted resize,
        rather than an unweighted one, so each downsampled cell reads the surrounding
        sky even where a star sits on top of it.
        """
        if not do_extract:
            return raw_image.copy()

        img_float = raw_image.astype(np.float32) / (65535.0 if raw_image.dtype == np.uint16 else 255.0)
        h, w = img_float.shape[:2]
        ds_h, ds_w = max(16, h // 32), max(16, w // 32)

        if star_mask_params is not None:
            # Star detection runs at full resolution (same cost as the denoiser's own
            # mask); everything after it is just a couple of extra resize calls, so
            # this stays cheap regardless of how large the source image is.
            gray_for_mask = img_float.mean(axis=2) if img_float.ndim == 3 else img_float
            star_mask_full = self._compute_star_mask(gray_for_mask, **star_mask_params)  # 1.0 = star
            keep_weight = (1.0 - star_mask_full).astype(np.float32)  # 1.0 = sky, 0.0 = star

            weight_small = cv2.resize(keep_weight, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            weighted_source = img_float * (keep_weight[:, :, None] if img_float.ndim == 3 else keep_weight)
            weighted_small = cv2.resize(weighted_source, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

            # Dividing the weighted sum by the weight recovers a local average over
            # just the non-star pixels in each downsampled cell - a weighted-resize
            # equivalent of "mask it out, then average what's left". A cell that's
            # almost entirely star (weight ~0, e.g. a very bright/large star) has
            # nothing reliable left to divide by, so it falls back to the plain
            # (unweighted) average for that cell alone rather than dividing by ~0.
            safe_weight = np.where(weight_small > 1e-3, weight_small, 1.0)
            divisor = safe_weight[:, :, None] if img_float.ndim == 3 else safe_weight
            small = weighted_small / divisor

            low_weight = weight_small <= 1e-3
            if low_weight.any():
                plain_small = cv2.resize(img_float, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
                small[low_weight] = plain_small[low_weight]
        else:
            small = cv2.resize(img_float, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

        background_small = cv2.medianBlur(small, 5)

        # A real light-pollution gradient varies slowly across the whole frame, so the
        # background model should too - but median blur is edge-preserving, not
        # smoothing: on this very coarse grid (one cell per ~32px) it can leave sharp
        # plateau-to-plateau steps between cells wherever the downsampled image has a
        # real hard edge in it, most commonly a saturated/clipped bright core (a
        # star's center, or an overexposed galaxy/nebula core) sitting next to
        # unsaturated surroundings. cv2.resize's cubic interpolation doesn't smooth
        # those steps away - cubic splines can overshoot right at a hard edge - so
        # they survive the upsample to full resolution as visible rectangular blocks
        # once this background is subtracted back out of the image. A small Gaussian
        # blur here, while the array is still tiny (tens of pixels across, so this
        # stays cheap regardless of the source image's real resolution), erases those
        # steps before the expensive part - the cubic upsample - ever sees them.
        small_blur_sigma = max(1.0, min(ds_h, ds_w) * 0.08)
        background_small = cv2.GaussianBlur(background_small, (0, 0), small_blur_sigma)

        background = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_CUBIC)

        if len(img_float.shape) == 3:
            subtracted = img_float - background + np.mean(background, axis=(0, 1))
        else:
            subtracted = img_float - background + np.mean(background)

        subtracted = np.clip(subtracted, 0, 1)
        if raw_image.dtype == np.uint16:
            return (subtracted * 65535.0).astype(np.uint16)
        else:
            return (subtracted * 255.0).astype(np.uint8)

    def process_background_and_update(self, _=None):
        """Kick off background extraction (if enabled) on a QThread.

        Background extraction involves a resize + medianBlur + resize over the full
        image, which is slow enough on a large stack to freeze the UI for several
        seconds if run on the main thread. It now runs on a QThread; the result comes
        back via a pyqtSignal and is applied by _on_background_done/_on_background_error.
        """
        if self.raw_image is None:
            return
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        self._set_busy(True)
        do_extract = self.bg_chk.isChecked()
        if do_extract:
            self.info_label.setText("Extracting background gradient... please wait.")
        filename = os.path.basename(self.file_path) if self.file_path else ""

        # Reuse the denoiser's own star-mask sliders for the gradient fit's star
        # exclusion too - one set of "how sensitive is star detection" controls,
        # consistently applied, rather than a second duplicate set of sliders.
        star_mask_params = None
        if do_extract:
            star_mask_params = dict(
                min_sensitivity=self.min_star_sensitivity_slider.get(),
                expand_iters=self.star_mask_expand_slider.get(),
                blur_radius=self.star_mask_blur_slider.get(),
            )

        self._bg_worker = BackgroundExtractionWorker(
            self._compute_background_removal, self.raw_image, do_extract, filename, star_mask_params
        )
        self._bg_worker.finished_ok.connect(self._on_background_done)
        self._bg_worker.failed.connect(self._on_background_error)
        self._bg_worker.start()

    def _on_background_done(self, gradient_removed, did_extract, filename):
        self.gradient_removed = gradient_removed
        suffix = " (Gradient Removed)" if did_extract else ""
        base_msg = f"Loaded: {filename}{suffix}" if filename else "Image processed."

        # Single-filter LRGB sub hint (point: "do this please" - flagging a mono L/R/G/B
        # frame loaded via Load Stacked Image instead of LRGB Combination) - appended
        # here, after background extraction, since this is the last thing that sets
        # info_label for a normal load and so the one message guaranteed not to get
        # overwritten a moment later.
        hint = ""
        if self._last_single_filter_channel:
            hint = (
                f"  This looks like a single {self._last_single_filter_channel}-filter "
                f"sub, not a combined color image - use LRGB Combination (below) to "
                f"combine it with the other channels instead."
            )
        self.info_label.setText(base_msg + hint)
        self._build_preview_source()
        # Frozen "before" snapshot for the comparison view - taken right after
        # background extraction (if any) but before denoise/stretch/alignment, and
        # never touched again for this load, even if the user later denoises, undoes,
        # or re-stretches.
        self._original_preview_source = None if self.preview_source is None else self.preview_source.copy()

        # Linear workflow reset: every fresh load starts over at the pre-denoise, pre-stretch
        # stage. Stretching controls lock until Apply Denoise completes (see point 5 of spec).
        self.linear_image = self._normalize_to_unit_float(self.gradient_removed)
        self.is_denoised = False
        self.pre_denoise_backup = None
        self.star_mask_full = None
        self._set_stretch_controls_enabled(False)
        self.undo_denoise_btn.setEnabled(False)

        # Seed the click-to-preview crop at the image center so the zoom panel isn't
        # empty before the user clicks anywhere.
        h, w = self.linear_image.shape[:2]
        self._extract_preview_crop(w / 2.0, h / 2.0)
        self._update_zoom_preview()

        self.update_image()
        self._set_busy(False)

    def _on_background_error(self, message):
        self.info_label.setText(f"Error during background extraction: {message}")
        self._set_busy(False)

    def _build_preview_source(self):
        """Downsample gradient_removed for interactive use; store the scale used.

        The live preview panel is only a few hundred pixels across, and every slider
        drag re-runs the stretch+alignment pipeline, so doing that at, say, 6000x4000
        on every single tick would allocate well over a gigabyte of temporary arrays
        many times per second. This cap only affects the interactive preview -
        save_image() re-runs the pipeline at full resolution once, on export.
        """
        if self.gradient_removed is None:
            self.preview_source = None
            self.preview_scale = 1.0
            return

        h, w = self.gradient_removed.shape[:2]
        longest = max(h, w)

        if longest <= self.MAX_PREVIEW_DIM:
            self.preview_source = self.gradient_removed
            self.preview_scale = 1.0
            return

        self.preview_scale = self.MAX_PREVIEW_DIM / float(longest)
        new_w, new_h = max(1, int(round(w * self.preview_scale))), max(1, int(round(h * self.preview_scale)))
        self.preview_source = cv2.resize(self.gradient_removed, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ Denoising engine (linear workflow)
    #
    # All of this operates on self.linear_image: a float32 array in [0, 1], kept
    # perfectly linear (no STF/arcsinh stretch) so denoising happens on the same data
    # the sensor actually captured, not on a display-stretched version of it. The STF
    # preview stretch applied elsewhere (update_image / _apply_stretch_and_alignment)
    # is purely a rendering step onto the QLabel - it never writes back into this array.

    def _normalize_to_unit_float(self, array):
        """Rescale a loaded array (uint8/uint16/already-float) to float32 in [0, 1]."""
        if array.dtype == np.uint16:
            max_val = 65535.0
        elif array.dtype == np.uint8:
            max_val = 255.0
        else:
            return np.clip(array.astype(np.float32), 0.0, 1.0)
        return (array.astype(np.float32) / max_val)

    def _set_stretch_controls_enabled(self, enabled):
        """Linear-workflow lock: stretching is only meaningful, and only allowed, once
        denoising has been applied to the linear data (point 5 of the spec)."""
        self.asinh_slider.setEnabled(enabled)
        self.black_slider.setEnabled(enabled)
        self.auto_stretch_btn.setEnabled(enabled)

    def _sync_linear_image_into_pipeline(self):
        """Push self.linear_image back into the native-dtype pipeline (gradient_removed /
        preview_source) that stretch, alignment and export already operate on."""
        if self.linear_image is None:
            return
        out_dtype = self.gradient_removed.dtype if self.gradient_removed is not None else np.uint16
        max_val = 65535.0 if out_dtype == np.uint16 else 255.0
        self.gradient_removed = np.clip(self.linear_image * max_val, 0, max_val).astype(out_dtype)
        self._build_preview_source()
        self.update_image()

    def _compute_star_mask(self, linear_image, min_sensitivity, expand_iters, blur_radius):
        """Build a 0-1 float star mask so stars can be protected from the denoiser.

        OpenCV's morphology ops don't support float64/arbitrary-depth input well (and
        are far slower on it), so this creates a lightweight uint8 proxy via NumPy
        min/max normalization strictly for the mask math, then converts the final
        result back to a high-bitrate float32 mask at the end.
        """
        gray_linear = linear_image.mean(axis=2) if linear_image.ndim == 3 else linear_image

        lo, hi = float(gray_linear.min()), float(gray_linear.max())
        if hi <= lo:
            hi = lo + 1e-6
        img_u8 = np.clip((gray_linear - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        # Top-hat isolates small bright features (stars) from smoothly-varying local
        # background/nebulosity - a multiscale alternative to a single global threshold.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.STAR_MASK_TOPHAT_KERNEL, self.STAR_MASK_TOPHAT_KERNEL)
        )
        tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel)

        # min_sensitivity (0-100): higher = catch fainter stars. Inverted onto the
        # threshold so the slider reads intuitively (higher = more permissive).
        threshold = max(1, int(round(255 * (1.0 - min_sensitivity / 100.0) * 0.5)))
        _, mask_u8 = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)

        if expand_iters > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = cv2.dilate(mask_u8, dilate_kernel, iterations=int(expand_iters))

        if blur_radius > 0:
            k = int(blur_radius) * 2 + 1  # GaussianBlur kernel size must be odd
            mask_u8 = cv2.GaussianBlur(mask_u8, (k, k), 0)

        return (mask_u8.astype(np.float32) / 255.0)

    def _denoise_core(self, linear_image, luminance_strength, chrominance_strength):
        """Wavelet-denoise a linear [0,1] image via a decoupled luminance/chrominance space.

        Converts to LAB (structural L channel vs. color a/b channels), denoises each
        independently with scikit-image's modern denoise_wavelet API - channel_axis=-1
        for the 2-channel chrominance stack rather than the removed `multichannel` flag,
        and rescale_sigma=True so user-set sigmas are handled correctly - then recombines.
        """
        img = linear_image.astype(np.float32)

        if img.ndim == 2:
            # Mono data has no chrominance to separate - denoise directly as luminance.
            if luminance_strength <= 0:
                return img.copy()
            sigma_l = (luminance_strength / 100.0) * self.MAX_LUMINANCE_SIGMA
            denoised = denoise_wavelet(img, method="BayesShrink", mode="soft", sigma=sigma_l, rescale_sigma=True)
            return np.clip(denoised, 0.0, 1.0).astype(np.float32)

        # OpenCV's float32 BGR<->LAB conversion expects BGR in [0,1] and returns L in
        # [0,100], a/b roughly in [-127,127]; rescale both into clean 0-1-ish ranges
        # before handing them to the wavelet denoiser.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:3] / 127.0

        if luminance_strength > 0:
            sigma_l = (luminance_strength / 100.0) * self.MAX_LUMINANCE_SIGMA
            L = denoise_wavelet(L, method="BayesShrink", mode="soft", sigma=sigma_l, rescale_sigma=True)

        if chrominance_strength > 0:
            # Chrominance gets a separate, more aggressive filter (point 3 of the spec) -
            # background color-noise splotches tolerate much heavier smoothing than
            # luminance detail before the image starts looking soft.
            sigma_c = (chrominance_strength / 100.0) * self.MAX_CHROMINANCE_SIGMA
            ab = denoise_wavelet(ab, method="BayesShrink", mode="soft", sigma=sigma_c, rescale_sigma=True, channel_axis=-1)

        lab_denoised = np.empty_like(lab)
        lab_denoised[:, :, 0] = np.clip(L, 0.0, 1.0) * 100.0
        lab_denoised[:, :, 1:3] = np.clip(ab, -1.0, 1.0) * 127.0

        bgr_denoised = cv2.cvtColor(lab_denoised.astype(np.float32), cv2.COLOR_LAB2BGR)
        return np.clip(bgr_denoised, 0.0, 1.0).astype(np.float32)

    def _apply_denoise_and_mask(self, linear_image, luminance_strength, chrominance_strength,
                                 min_sensitivity, expand, blur):
        """Full denoise pipeline: build the star mask, denoise, then alpha-blend the
        pure original pixels back in under the mask so stars stay crisp points rather
        than being smoothed into blurry blobs. Safe to call from a worker thread."""
        star_mask = self._compute_star_mask(linear_image, min_sensitivity, expand, blur)
        denoised = self._denoise_core(linear_image, luminance_strength, chrominance_strength)

        mask_for_blend = star_mask[:, :, None] if linear_image.ndim == 3 else star_mask
        blended = mask_for_blend * linear_image + (1.0 - mask_for_blend) * denoised
        return np.clip(blended, 0.0, 1.0).astype(np.float32), star_mask

    def _current_denoise_params(self):
        return dict(
            luminance_strength=self.luminance_slider.get(),
            chrominance_strength=self.chrominance_slider.get(),
            min_sensitivity=self.min_star_sensitivity_slider.get(),
            expand=self.star_mask_expand_slider.get(),
            blur=self.star_mask_blur_slider.get(),
        )

    def apply_denoise_clicked(self):
        """Run the full-resolution denoise on a QThread (point 6: never on the main thread)."""
        if self.linear_image is None:
            self.info_label.setText("Load an image before denoising.")
            return
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return

        self._set_busy(True)
        self.denoise_progress_bar.setRange(0, 0)  # indeterminate pulse while the worker runs
        self.info_label.setText("Denoising full-resolution image... please wait.")

        # Single-slot undo history (point 7): this overwrites any prior backup, so RAM
        # never holds more than the current image plus exactly one earlier state.
        self.pre_denoise_backup = self.linear_image.copy()

        params = self._current_denoise_params()
        cores = self.cpu_cores_slider.value()

        self._denoise_worker = DenoiseWorker(self._apply_denoise_and_mask, self.linear_image, params, cores)
        self._denoise_worker.finished_ok.connect(self._on_denoise_done)
        self._denoise_worker.failed.connect(self._on_denoise_error)
        self._denoise_worker.start(QThread.Priority.LowPriority)

    def _on_denoise_done(self, denoised_linear, star_mask):
        self.linear_image = denoised_linear
        self.star_mask_full = star_mask
        self.is_denoised = True
        self.denoise_progress_bar.setRange(0, 1)
        self.denoise_progress_bar.setValue(0)
        self._set_stretch_controls_enabled(True)
        self.info_label.setText("Denoising complete - stretching controls are now unlocked.")
        self._sync_linear_image_into_pipeline()
        self._set_busy(False)

    def _on_denoise_error(self, message):
        self.pre_denoise_backup = None  # the attempted run never produced a valid state to undo to
        self.denoise_progress_bar.setRange(0, 1)
        self.denoise_progress_bar.setValue(0)
        self.info_label.setText(f"Denoise failed: {message}")
        self._set_busy(False)

    def undo_denoise_clicked(self):
        """Restore the exact pre-denoise linear image and re-lock the stretch controls."""
        if self.pre_denoise_backup is None:
            self.info_label.setText("Nothing to undo.")
            return

        self.linear_image = self.pre_denoise_backup
        self.pre_denoise_backup = None  # single-slot history: clear immediately after use
        self.is_denoised = False
        self.star_mask_full = None
        self._set_stretch_controls_enabled(False)
        self.undo_denoise_btn.setEnabled(False)
        self._sync_linear_image_into_pipeline()
        self.info_label.setText("Denoise undone - stretching controls re-locked.")
        gc.collect()

    # ------------------------------------------------------------------ Click-to-preview crop

    def on_preview_clicked(self, label_x, label_y):
        """Map a click on the main viewport back to full-resolution coordinates and
        refresh the 300x300 real-time denoise preview centered there."""
        if self.linear_image is None or self.preview_source is None or self._last_preview_pixmap_rect is None:
            return

        off_x, off_y, pw, ph = self._last_preview_pixmap_rect
        if pw <= 0 or ph <= 0:
            return
        px, py = label_x - off_x, label_y - off_y
        if not (0 <= px <= pw and 0 <= py <= ph):
            return  # click landed on the letterboxed padding, not the image itself

        preview_h, preview_w = self.preview_source.shape[:2]
        preview_x = px / pw * preview_w
        preview_y = py / ph * preview_h

        scale = self.preview_scale if self.preview_scale else 1.0
        full_x = preview_x / scale
        full_y = preview_y / scale

        self._extract_preview_crop(full_x, full_y)
        self._update_zoom_preview()

    def _extract_preview_crop(self, full_x, full_y):
        """Slice a lightweight 300x300 (or smaller, if the image is tinier) crop out of
        the full-resolution linear image, centered as close to (full_x, full_y) as the
        image bounds allow. Overwrites the previous crop reference rather than
        accumulating - exactly one crop is ever kept alive at a time (point 7)."""
        h, w = self.linear_image.shape[:2]
        size = min(self.PREVIEW_CROP_SIZE, h, w)
        half = size // 2
        cx, cy = int(round(full_x)), int(round(full_y))
        x0 = max(0, min(w - size, cx - half))
        y0 = max(0, min(h - size, cy - half))
        self._preview_crop_source = self.linear_image[y0:y0 + size, x0:x0 + size].copy()

    def _update_zoom_preview(self, _=None):
        """Instant, main-thread, crop-only preview of the denoise+mask settings. Only
        ever touches a small (<=300x300) slice, so it stays sub-second even while a
        slider is being dragged continuously."""
        if self._preview_crop_source is None:
            return

        blended, mask = self._apply_denoise_and_mask(self._preview_crop_source, **self._current_denoise_params())

        if self.view_mask_chk.isChecked():
            display = (1.0 - mask) if self.invert_mask_chk.isChecked() else mask
            display_u8 = np.clip(display * 255.0, 0, 255).astype(np.uint8)
            display_bgr = cv2.cvtColor(display_u8, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        self._render_small_preview(display_bgr)

        # This preview's intermediates are only needed for the single frame just
        # rendered - let them go rather than holding onto extra array references.
        del blended, mask

    def _render_small_preview(self, bgr_u8):
        img_rgb = np.ascontiguousarray(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB))
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.zoom_preview_label.width(), self.zoom_preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.zoom_preview_label.setPixmap(scaled)

    # ------------------------------------------------------------------ Alignment

    def shift_channel(self, channel, dx, dy):
        if dx == 0 and dy == 0:
            return channel
        rows, cols = channel.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(channel, M, (cols, rows))

    def reset_alignment(self):
        self._set_slider_silently(self.r_x_slider, 0)
        self._set_slider_silently(self.r_y_slider, 0)
        self._set_slider_silently(self.b_x_slider, 0)
        self._set_slider_silently(self.b_y_slider, 0)
        self._refresh_rgb_labels()
        self.update_image()

    def _refresh_alignment_labels(self):
        # blockSignals() during _set_slider_silently means the value-label lambdas
        # didn't fire, so refresh them explicitly after a batch slider update.
        self.asinh_value_label.setText(f"{self.asinh_slider.get():.1f}")
        self.black_value_label.setText(f"{self.black_slider.get():.3f}")

    def auto_align(self):
        """Cross-correlate R and B against G (cv2.phaseCorrelate) and set the alignment sliders.

        Runs against the downsampled preview_source rather than the full-resolution
        image - phase correlation only needs enough resolution to find the offset, and
        this keeps it fast even on a large stack. The resulting shift is in preview
        pixels, so it's divided by preview_scale to get the equivalent full-resolution
        pixel offset the sliders actually represent.
        """
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return
        if self.preview_source is None or self.preview_source.ndim != 3 or self.preview_source.shape[2] != 3:
            self.info_label.setText("Auto-align needs a loaded color image.")
            return

        b_chan, g_chan, r_chan = cv2.split(self.preview_source.astype(np.float32))
        h, w = g_chan.shape
        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)

        # cv2.phaseCorrelate multiplies its inputs by the window IN PLACE, so pass copies -
        # g_chan in particular is reused for both calls and must stay unmodified between them.
        shift_r, response_r = cv2.phaseCorrelate(g_chan.copy(), r_chan.copy(), hann)
        shift_b, response_b = cv2.phaseCorrelate(g_chan.copy(), b_chan.copy(), hann)

        scale = self.preview_scale if self.preview_scale else 1.0
        lo, hi = self.ALIGN_SLIDER_RANGE
        clamped = False

        def to_slider_value(delta):
            nonlocal clamped
            full_res_delta = -delta / scale
            if full_res_delta < lo or full_res_delta > hi:
                clamped = True
            return max(lo, min(hi, full_res_delta))

        rx = to_slider_value(shift_r[0])
        ry = to_slider_value(shift_r[1])
        bx = to_slider_value(shift_b[0])
        by = to_slider_value(shift_b[1])

        self._set_slider_silently(self.r_x_slider, rx)
        self._set_slider_silently(self.r_y_slider, ry)
        self._set_slider_silently(self.b_x_slider, bx)
        self._set_slider_silently(self.b_y_slider, by)
        self._refresh_rgb_labels()
        self.update_image()

        msg = (f"Auto-aligned: R=({rx:.1f}, {ry:.1f})  B=({bx:.1f}, {by:.1f})  "
               f"[confidence R={response_r:.2f}, B={response_b:.2f}]")
        if clamped:
            msg += " - needed shift exceeded the ±15px slider range and was clamped."
        elif min(response_r, response_b) < 0.1:
            msg += " - low correlation confidence; check the result visually."
        self.info_label.setText(msg)

    def _refresh_rgb_labels(self):
        # Companion to _refresh_alignment_labels, for the RGB shift value labels.
        self.r_x_value_label.setText(f"{self.r_x_slider.get():.0f}")
        self.r_y_value_label.setText(f"{self.r_y_slider.get():.0f}")
        self.b_x_value_label.setText(f"{self.b_x_slider.get():.0f}")
        self.b_y_value_label.setText(f"{self.b_y_slider.get():.0f}")

    # ------------------------------------------------------------------ Stretch pipeline

    def _apply_stretch_and_alignment(self, source, stretch_factor, black_point, rx, ry, bx, by,
                                      shift_scale=1.0, output_bit_depth=8):
        """Run the arcsinh stretch + RGB channel alignment on `source`.

        Takes all slider values as plain numbers rather than reading the widgets
        directly, so this method is safe to call from a background thread (Qt widgets
        must only ever be touched from the main thread).

        shift_scale converts the sliders' full-resolution pixel offsets to whatever
        resolution `source` actually is (1.0 for full-res export, <1.0 for the
        downsampled interactive preview) so a "5px" alignment looks the same size on
        screen as it will in the final exported image.

        output_bit_depth: 8 (default, used for the on-screen preview) or 16, for
        export when finer tonal precision matters (avoids 8-bit banding in smooth
        sky/nebulosity gradients).
        """
        max_val = 65535.0 if source.dtype == np.uint16 else 255.0
        img_norm = source.astype(np.float32) / max_val
        img_norm = np.maximum(0, img_norm - black_point)

        # ArcSinh Stretching Formula
        stretched = np.arcsinh(img_norm * stretch_factor) / np.arcsinh(stretch_factor)

        if output_bit_depth == 16:
            out_dtype, out_max = np.uint16, 65535.0
        else:
            out_dtype, out_max = np.uint8, 255.0
        final_preview = np.clip(stretched * out_max, 0, out_max).astype(out_dtype)

        # Process RGB Alignment Shift
        if len(final_preview.shape) == 3 and final_preview.shape[2] == 3:
            b_chan, g_chan, r_chan = cv2.split(final_preview)

            rx = int(round(rx * shift_scale))
            ry = int(round(ry * shift_scale))
            bx = int(round(bx * shift_scale))
            by = int(round(by * shift_scale))

            r_aligned = self.shift_channel(r_chan, rx, ry)
            b_aligned = self.shift_channel(b_chan, bx, by)

            return cv2.merge([b_aligned, g_chan, r_aligned])
        else:
            return final_preview

    def _current_slider_values(self):
        """Read every relevant slider on the main thread, for handing off to a worker."""
        return dict(
            stretch_factor=self.asinh_slider.get(),
            black_point=self.black_slider.get(),
            rx=self.r_x_slider.get(),
            ry=self.r_y_slider.get(),
            bx=self.b_x_slider.get(),
            by=self.b_y_slider.get(),
        )

    def update_image(self, _=None):
        if self.preview_source is None:
            return

        self.processed_image = self._apply_stretch_and_alignment(
            self.preview_source, shift_scale=self.preview_scale, output_bit_depth=8,
            **self._current_slider_values(),
        )

        # Render dynamic visual displays
        self.render_histogram()
        self.update_preview_display()

    def auto_stretch(self):
        """Set the asinh stretch factor and black point from the image's own statistics.

        Uses the median and median absolute deviation (MAD) of the (downsampled)
        image as a robust estimate of the background level and noise, the same
        general approach PixInsight's Screen Transfer Function auto-stretch uses:
        clip the black point a few sigma below the background median (so the noise
        floor isn't destroyed, just the extreme low tail), then pick a stretch
        strength that lifts what's left of the background up to a reasonable midtone
        brightness rather than leaving the image looking flat and unstretched.
        """
        if self._busy:
            self.info_label.setText("Still working - please wait for the current operation to finish.")
            return
        if self.preview_source is None:
            self.info_label.setText("Load an image before using Auto-Stretch.")
            return

        source = self.preview_source
        max_val = 65535.0 if source.dtype == np.uint16 else 255.0
        norm = source.astype(np.float32) / max_val
        gray = norm.mean(axis=2) if norm.ndim == 3 else norm

        median = float(np.median(gray))
        mad = float(np.median(np.abs(gray - median)))
        sigma = 1.4826 * mad  # normal-distribution-equivalent standard deviation from MAD

        black_min, black_max = self.black_slider.min_value, self.black_slider.max_value
        black_point = float(np.clip(median - self.AUTO_STRETCH_SHADOW_CLIP_SIGMA * sigma, black_min, black_max))

        residual_median = max(0.0, median - black_point)
        stretch_min, stretch_max = self.asinh_slider.min_value, self.asinh_slider.max_value
        stretch_factor = self._solve_stretch_factor(
            residual_median, target=self.AUTO_STRETCH_TARGET_BG, lo=stretch_min, hi=stretch_max
        )

        self._set_slider_silently(self.black_slider, black_point)
        self._set_slider_silently(self.asinh_slider, stretch_factor)
        self._refresh_alignment_labels()
        self.update_image()

        self.info_label.setText(
            f"Auto-stretched: black point={black_point:.3f}, asinh factor={stretch_factor:.1f} "
            f"(background median was {median:.3f})"
        )

    @staticmethod
    def _solve_stretch_factor(residual_median, target, lo=1.0, hi=50.0, iterations=40):
        """Find s in [lo, hi] so arcsinh(residual_median * s) / arcsinh(s) ~= target.

        f(s) is monotonically increasing in s for residual_median in (0, 1), so a
        simple bisection is sufficient - no external optimizer dependency needed.
        """
        if residual_median <= 0:
            return lo

        def f(s):
            return np.arcsinh(residual_median * s) / np.arcsinh(s)

        if f(lo) >= target:
            return lo
        if f(hi) <= target:
            return hi

        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------ Preview + histogram rendering

    def _render_before_image(self, image):
        """Quick percentile-based 0-1 stretch so the pre-denoise/stretch 'before'
        image is actually visible in the comparison view (the real linear data looks
        almost black unstretched). Purely a display rendering - this never feeds back
        into the real linear/gradient_removed pipeline the way the arcsinh stretch does.
        """
        arr = image.astype(np.float32) / (65535.0 if image.dtype == np.uint16 else 255.0)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = np.percentile(finite, 0.5), np.percentile(finite, 99.5)
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1e-6
        stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
        return (stretched * 255.0).astype(np.uint8)

    def _compose_comparison_image(self):
        """Pick/build the image actually drawn into the preview, per the before/after/
        split comparison mode (point 2 of the latest feature request): flip between
        seeing the original loaded image and the current processed result, rather than
        only ever seeing the current state."""
        after_img = self.processed_image
        mode = self.compare_mode_combo.currentText() if hasattr(self, "compare_mode_combo") else "After (Processed)"

        if mode == "After (Processed)" or self._original_preview_source is None:
            return after_img

        before_img = self._render_before_image(self._original_preview_source)

        def to_bgr(img):
            return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        before_bgr, after_bgr = to_bgr(before_img), to_bgr(after_img)
        if before_bgr.shape[:2] != after_bgr.shape[:2]:
            before_bgr = cv2.resize(
                before_bgr, (after_bgr.shape[1], after_bgr.shape[0]), interpolation=cv2.INTER_AREA
            )

        if mode == "Before (Original)":
            return before_bgr

        # Split View: original on the left half, processed on the right, with a thin
        # amber divider line so the seam is easy to find even on a busy starfield.
        h, w = after_bgr.shape[:2]
        split_x = w // 2
        composite = after_bgr.copy()
        composite[:, :split_x] = before_bgr[:, :split_x]
        composite[:, max(0, split_x - 1):split_x + 1] = (0, 215, 255)  # BGR amber
        return composite

    def update_preview_display(self):
        """Draw the processed (or before/split-comparison) image into the embedded
        live-preview QLabel."""
        if self.processed_image is None:
            return

        img = self._compose_comparison_image()
        if img is None:
            return
        if img.ndim == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        # .copy() detaches the QImage from the numpy buffer, which is about to go out of scope.
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        label_w = max(self.preview_label.width(), 100)
        label_h = max(self.preview_label.height(), 100)
        scaled = pixmap.scaled(
            label_w, label_h,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

        # Record where the pixmap actually lands within the label (it's centered and
        # letterboxed by KeepAspectRatio) so on_preview_clicked can map a click back
        # to full-resolution image coordinates.
        off_x = (label_w - scaled.width()) / 2.0
        off_y = (label_h - scaled.height()) / 2.0
        self._last_preview_pixmap_rect = (off_x, off_y, scaled.width(), scaled.height())

    def render_histogram(self):
        self.ax.clear()
        if len(self.processed_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                self.ax.plot(hist, color=col)
        else:
            hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            self.ax.plot(hist, color='black')

        self.ax.set_xlim([0, 256])
        self.canvas.draw()

    # ------------------------------------------------------------------ Session / recipe persistence

    def _collect_recipe_settings(self):
        """Snapshot every tunable control into a plain JSON-serializable dict."""
        return {
            "luminance_strength": self.luminance_slider.get(),
            "chrominance_strength": self.chrominance_slider.get(),
            "min_star_sensitivity": self.min_star_sensitivity_slider.get(),
            "star_mask_expand": self.star_mask_expand_slider.get(),
            "star_mask_blur": self.star_mask_blur_slider.get(),
            "lum_blend_strength": self.lum_blend_slider.get(),
            "asinh_stretch_factor": self.asinh_slider.get(),
            "black_point": self.black_slider.get(),
            "r_shift_x": self.r_x_slider.get(),
            "r_shift_y": self.r_y_slider.get(),
            "b_shift_x": self.b_x_slider.get(),
            "b_shift_y": self.b_y_slider.get(),
            "cpu_cores": self.cpu_cores_slider.value(),
            "background_extraction_enabled": self.bg_chk.isChecked(),
            "export_16bit": self.export_16bit_chk.isChecked(),
            "bayer_pattern": self.bayer_combo.currentText(),
        }

    def _apply_recipe_settings(self, settings):
        """Apply a previously-saved (or hand-edited) settings dict to every control.

        Unknown/missing keys are simply skipped rather than treated as an error, so a
        recipe saved by an older version of ClariStretch (fewer sliders) still loads
        cleanly, and a partial hand-edited file only touches the keys it names.
        Sliders are set with blockSignals (via _set_slider_silently) so applying a
        whole recipe doesn't re-run the processing pipeline once per slider - the
        labels and a single pipeline refresh happen explicitly at the end instead.
        """
        slider_keys = {
            "luminance_strength": self.luminance_slider,
            "chrominance_strength": self.chrominance_slider,
            "min_star_sensitivity": self.min_star_sensitivity_slider,
            "star_mask_expand": self.star_mask_expand_slider,
            "star_mask_blur": self.star_mask_blur_slider,
            "lum_blend_strength": self.lum_blend_slider,
            "asinh_stretch_factor": self.asinh_slider,
            "black_point": self.black_slider,
            "r_shift_x": self.r_x_slider,
            "r_shift_y": self.r_y_slider,
            "b_shift_x": self.b_x_slider,
            "b_shift_y": self.b_y_slider,
        }
        for key, slider in slider_keys.items():
            if key in settings:
                try:
                    self._set_slider_silently(slider, float(settings[key]))
                except (TypeError, ValueError):
                    pass  # a malformed value for this one key shouldn't abort the rest

        if "cpu_cores" in settings:
            try:
                clamped = max(self.cpu_cores_slider.minimum(),
                               min(self.cpu_cores_slider.maximum(), int(settings["cpu_cores"])))
                self.cpu_cores_slider.blockSignals(True)
                self.cpu_cores_slider.setValue(clamped)
                self.cpu_cores_slider.blockSignals(False)
            except (TypeError, ValueError):
                pass

        if "background_extraction_enabled" in settings:
            self.bg_chk.blockSignals(True)
            self.bg_chk.setChecked(bool(settings["background_extraction_enabled"]))
            self.bg_chk.blockSignals(False)

        if "export_16bit" in settings:
            self.export_16bit_chk.setChecked(bool(settings["export_16bit"]))

        if "bayer_pattern" in settings:
            idx = self.bayer_combo.findText(str(settings["bayer_pattern"]))
            if idx >= 0:
                self.bayer_combo.blockSignals(True)
                self.bayer_combo.setCurrentIndex(idx)
                self.bayer_combo.blockSignals(False)

        # Refresh every value label the blockSignals() calls above skipped, then
        # re-render once with the newly-applied values instead of once per slider.
        self.luminance_value_label.setText(f"{self.luminance_slider.get():.0f}")
        self.chrominance_value_label.setText(f"{self.chrominance_slider.get():.0f}")
        self.min_star_value_label.setText(f"{self.min_star_sensitivity_slider.get():.0f}")
        self.star_mask_expand_value_label.setText(f"{self.star_mask_expand_slider.get():.0f}")
        self.star_mask_blur_value_label.setText(f"{self.star_mask_blur_slider.get():.0f}")
        self.lum_blend_value_label.setText(f"{self.lum_blend_slider.get():.0f}")
        self.cpu_cores_value_label.setText(str(self.cpu_cores_slider.value()))
        self._refresh_alignment_labels()
        self._refresh_rgb_labels()

        self._update_zoom_preview()
        self.update_image()

    def save_settings_clicked(self):
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings (Recipe)", "", "ClariStretch Recipe (*.json)",
        )
        if not out_path:
            return
        if not os.path.splitext(out_path)[1]:
            out_path += ".json"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_recipe_settings(), f, indent=2)
        except OSError as e:
            self.progress_label.setText(f"Error saving settings: {e}")
            return
        self.progress_label.setText(f"Settings saved: {os.path.basename(out_path)}")

    def load_settings_clicked(self):
        if self._busy:
            self.progress_label.setText("Still working - please wait for the current operation to finish.")
            return

        in_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings (Recipe)", "", "ClariStretch Recipe (*.json)",
        )
        if not in_path:
            return

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.progress_label.setText(f"Error loading settings: {e}")
            return

        if not isinstance(settings, dict):
            self.progress_label.setText("Settings file is not a valid recipe (expected a JSON object).")
            return

        self._apply_recipe_settings(settings)
        self.progress_label.setText(f"Settings loaded: {os.path.basename(in_path)}")

    # ------------------------------------------------------------------ Export (threaded)

    def save_image(self):
        if self.gradient_removed is None:
            self.progress_label.setText("No processed image to export yet.")
            return
        if self._busy:
            self.progress_label.setText("Still working - please wait for the current operation to finish.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)",
        )
        if not out_path:
            return
        if not os.path.splitext(out_path)[1]:
            out_path += ".png"  # QFileDialog doesn't always append the filter's extension

        want_16bit = self.export_16bit_chk.isChecked()
        ext = os.path.splitext(out_path)[1].lower()
        if want_16bit and ext in (".jpg", ".jpeg"):
            # JPEG has no 16-bit mode - fall back rather than fail the export outright.
            want_16bit = False
            self.progress_label.setText("JPEG doesn't support 16-bit - exporting as 8-bit instead.")
        else:
            self.progress_label.setText("Rendering full-resolution export...")

        self._set_busy(True)

        # Read slider values and snapshot the source image here, on the main thread -
        # the worker thread must not touch any Qt widgets directly.
        slider_values = self._current_slider_values()
        output_bit_depth = 16 if want_16bit else 8

        self._export_worker = ExportWorker(
            self._apply_stretch_and_alignment, self.gradient_removed, slider_values, output_bit_depth, out_path
        )
        self._export_worker.finished_ok.connect(self._on_export_done)
        self._export_worker.failed.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_done(self, out_path, success):
        if success:
            self.progress_label.setText(f"Exported: {os.path.basename(out_path)}")
        else:
            self.progress_label.setText("Error exporting image (unsupported path or format).")
        self._set_busy(False)

    def _on_export_error(self, message):
        self.progress_label.setText(f"Export failed: {message}")
        self._set_busy(False)


def main():
    """Entry point used both by `python claristretch.py` (see __main__ below) and by
    the `claristretch` console-script installed from pyproject.toml, so packaging and
    a direct script run always go through the exact same startup path."""
    multiprocessing.freeze_support()  # required so a frozen (.exe) build doesn't re-spawn the app
    app = QApplication(sys.argv)
    # Set at the QApplication level (not just on the main window) so the icon also
    # shows up in places the OS pulls it from independently of any one window - the
    # taskbar/dock entry on some Linux desktops, and the missing-dependency dialog
    # below, in addition to QMainWindow's own setWindowIcon() call in __init__.
    app.setWindowIcon(_load_app_icon())

    # Report any missing required dependency through a visible dialog instead of a
    # console message, since a PyInstaller --windowed build has no console for the user
    # to ever see that message in. This has to happen here, after QApplication exists
    # but before ClariStretch() is constructed, because QMessageBox needs a running
    # QApplication to draw itself, and the main window's __init__ assumes cv2, numpy,
    # matplotlib and denoise_wavelet are all already importable.
    if _MISSING_DEPENDENCIES:
        package_list = "\n".join(f"    pip install {pkg}" for pkg, _ in _MISSING_DEPENDENCIES)
        error_detail = "\n".join(f"  - {pkg}: {err}" for pkg, err in _MISSING_DEPENDENCIES)
        QMessageBox.critical(
            None,
            "ClariStretch - Missing Dependencies",
            "ClariStretch can't start because one or more required packages aren't "
            "installed in this Python environment.\n\n"
            "Fix with:\n"
            f"{package_list}\n\n"
            "Underlying import errors:\n"
            f"{error_detail}",
        )
        sys.exit(1)

    window = ClariStretch()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
