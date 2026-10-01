# ClariStretch
<<<<<<< HEAD

ClariStretch is a desktop astrophotography image processor built with PyQt6. It
handles the full linear-to-stretched workflow for stacked subs: loading (including
FITS, multi-extension per-filter stacks, and one-shot-color/Bayer sensor data),
background (light-pollution gradient) extraction, wavelet-based denoising with a
star-protecting mask, an arcsinh midtone stretch, RGB channel alignment, and 8/16-bit
export - all without freezing the UI, since every heavy operation runs on a
background `QThread`.

## Features

- **FITS ingestion**: single color cubes, separate R/G/B extensions, one-shot-color
  (Bayer mosaic) frames with auto-detected or manually-forced CFA pattern, and plain
  mono files. Opens with `astropy.io.fits` using `memmap=True` for memory efficiency.
- **Background extraction**: removes light-pollution gradients via a downsampled
  median-blur model.
- **Linear-workflow denoising**: wavelet denoising (`scikit-image`) in a decoupled
  luminance/chrominance space, with an automatic star mask (OpenCV morphology) so
  stars stay sharp points instead of being smoothed into blobs. Stretching is locked
  until denoising has been applied, matching standard astrophotography practice.
- **Auto-Stretch**: a PixInsight-STF-style automatic arcsinh stretch computed from
  the image's own background statistics.
- **Auto-Align**: cross-correlation-based RGB channel alignment (`cv2.phaseCorrelate`)
  to fix chromatic fringing in one click.
- **Real-time local preview**: click anywhere on the main viewport to instantly
  preview denoise/mask settings on a small crop, without reprocessing the whole image.
- **8-bit/16-bit export**, with automatic fallback when a chosen format (e.g. JPEG)
  doesn't support 16-bit.
- **CPU throttling**: a Max CPU Cores slider caps both OpenCV's and (when
  `threadpoolctl` is installed) NumPy/scikit-image's native thread pools during
  the denoise pass.

## Requirements

- Python 3.10+
- PyQt6
- numpy
- opencv-python-headless (use the **headless** build - the regular `opencv-python`
  bundles its own Qt platform plugins, which conflict with PyQt6's and will crash
  the app on startup)
- matplotlib
- astropy
- scikit-image
- PyWavelets (required by `scikit-image.restoration.denoise_wavelet`)
- threadpoolctl (optional - enables finer-grained CPU throttling; the app still
  works without it, just with coarser throttling via OpenCV alone)

Install everything with:

```bash
pip install PyQt6 numpy opencv-python-headless matplotlib astropy scikit-image PyWavelets threadpoolctl
```

Or save that same list as `requirements.txt`:

```text
PyQt6
numpy
opencv-python-headless
matplotlib
astropy
scikit-image
PyWavelets
threadpoolctl
```

and install with `pip install -r requirements.txt`.

## Running

```bash
python claristretch.py
```

If a required package is genuinely missing, ClariStretch exits with a clear
message naming the package and the exact `pip install` command to fix it,
instead of a raw traceback.

## Troubleshooting

**`ModuleNotFoundError: No module named 'skimage'` (or any other package) even
after running `pip install`.** This almost always means `pip` installed the
package into a *different* Python environment than the one running
`claristretch.py` - a very common issue when a system has multiple Python
installs, or when a virtual environment exists but isn't activated. Fixes, in
order of how often they solve it:

1. Install with the exact same interpreter you use to run the script:
   ```bash
   python -m pip install -r requirements.txt
   python claristretch.py
   ```
   (use whichever of `python` / `python3` / `py` you actually use to launch the
   app - just keep it identical across both commands).
2. If you're using a virtual environment, make sure it's **activated** in the
   current terminal before installing or running anything:
   ```bash
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```
3. Confirm which interpreter and site-packages are actually in play:
   ```bash
   python -c "import sys; print(sys.executable)"
   python -m pip show scikit-image
   ```
   If the second command says "not found" while the first shows an interpreter
   you don't recognize, that mismatch is the issue - reinstall with that exact
   interpreter.
4. If you're getting an error pointing at an old line number or an old-looking
   message, you may be running a stale copy of `claristretch.py` - re-save the
   latest version before retrying.

**App crashes immediately with a Qt platform plugin error (e.g.
`Could not load the Qt platform plugin "xcb"`).** You likely have the
non-headless `opencv-python` installed alongside PyQt6; its bundled Qt plugins
conflict with PyQt6's own. Fix:
```bash
pip uninstall opencv-python
pip install --force-reinstall opencv-python-headless
```

## App icon

`assets/icon.png` is the window/taskbar icon (loaded at runtime via
`QMainWindow.setWindowIcon()` and `QApplication.setWindowIcon()`), and
`assets/icon.ico` is a multi-resolution Windows icon for PyInstaller's
`--icon` flag, which sets the icon on the built `.exe` file itself (Explorer,
taskbar pins, shortcuts). They're generated from the same source image, so
they always match. If `assets/icon.png` is ever missing or unreadable, the
app falls back to Qt's default empty icon rather than failing to start -
nothing in the app treats the icon as required.

## Building a standalone Windows executable

The entry point already includes `multiprocessing.freeze_support()` and a
`resource_path()` helper for locating bundled resources under PyInstaller's
`sys._MEIPASS` extraction directory, so it's ready to build with:

```bash
pyinstaller --onefile --windowed --name ClariStretch ^
    --icon assets/icon.ico ^
    --add-data "assets/icon.png;assets" ^
    --exclude-module astropy.samp --exclude-module pyvo ^
    claristretch.py
```

(On macOS/Linux, replace the `--add-data "assets/icon.png;assets"` separator
with a colon: `--add-data "assets/icon.png:assets"`, and drop the trailing
`^` line-continuations, or just run it as one line on either platform.)

`--icon` and `--add-data` do two different jobs and you need both: `--icon`
only stamps `assets/icon.ico` onto the compiled `.exe` file itself (what
Explorer and shortcuts show before the app is even running); `--add-data`
bundles `assets/icon.png` into the frozen app so `resource_path()` can find
it at `sys._MEIPASS` once the app is actually running and sets its own
window/taskbar icon. Skip `--add-data` and the `.exe` file will look right
in Explorer, but the running window will fall back to Qt's blank default
icon.

**About the `AstropyDeprecationWarning: astropy.samp was deprecated...` build log
line**: this comes from PyInstaller's own dependency-analysis step, not from
ClariStretch itself. PyInstaller's astropy hook eagerly imports every astropy
submodule (to decide what to bundle), including `astropy.samp` - and that
submodule unconditionally fires a deprecation warning the instant it's imported,
by anyone. ClariStretch only ever uses `astropy.io.fits` and never touches SAMP,
so the `--exclude-module astropy.samp` flag above removes it from the build
entirely (slightly smaller executable too) and the warning stops appearing. If
you're on an older PyInstaller/astropy combination where that flag alone doesn't
fully suppress it, set `PYTHONWARNINGS=ignore` in the environment you run
`pyinstaller` from - that silences it at the build-process level regardless of
which hook triggers it.

## Project status

This is an actively developed internal tool; interfaces and defaults (denoise
sigma scaling, star-mask thresholds, preview crop size, etc.) may still change.
=======
A PyQt6 astrophotography image processor — FITS/OSC ingestion, wavelet denoising with star protection, arcsinh stretch, and RGB alignment.
>>>>>>> 4e3e95392ae908c7193bc6830411efde803b63c1
