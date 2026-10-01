# ClariStretch

ClariStretch is a free, open-source desktop astrophotography image processor,
licensed under the [GNU GPL v3](LICENSE) - the full source is in this repo, anyone
can read it, modify it, and redistribute their own changes under the same license.
It's a small project built and maintained without a company or a budget behind it,
which is also why the Windows build below isn't code-signed (see
[Code signing and Windows SmartScreen](#code-signing-and-windows-smartscreen)
for what that means for you and how to get past the warning it causes).

ClariStretch
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
- **LRGB combination**: load separate Luminance/Red/Green/Blue master frames and
  combine them into one color image. R/G/B supply color (chrominance) while L -
  typically the sharpest, highest-SNR frame of the set - substitutes for the
  luminance channel via an LAB color-space blend, with an adjustable blend strength
  slider. Masters of different resolutions are automatically resized to match
  (defaulting to L's resolution when L is loaded).
- **Real-time local preview**: click anywhere on the main viewport to instantly
  preview denoise/mask settings on a small crop, without reprocessing the whole image.
- **8-bit/16-bit export**, with automatic fallback when a chosen format (e.g. JPEG)
  doesn't support 16-bit.
- **CPU throttling**: a Max CPU Cores slider caps both OpenCV's and (when
  `threadpoolctl` is installed) NumPy/scikit-image's native thread pools during
  the denoise pass.
- **Star-aware background extraction**: the light-pollution gradient model excludes
  star pixels (reusing the denoiser's own star-mask logic) before fitting, instead
  of letting bright stars bias the local gradient estimate - this avoids the faint
  dark halos that otherwise appear around stars in star-dense fields once the
  background model is subtracted back out.
- **Before/after comparison**: a "View" selector above the live preview flips
  between the original loaded image, the current processed result, or a side-by-side
  split view with a divider line - so you can always check what your edits actually
  changed, not just the current state.
- **Drag-and-drop loading + recent files**: drop a FITS/TIFF/PNG/JPEG file anywhere
  on the window to load it, or reopen one of your last 10 files from the Recent
  Files menu next to the load button (persisted across app restarts).
- **Session/recipe persistence**: save every slider (denoise strengths, stretch
  factor, alignment offsets, star-mask settings, and more) to a small `.json` file
  via "Save Settings...", and reload it later with "Load Settings..." - useful for
  reprocessing the same target in a future session, or applying a known-good
  setting to a new one, without re-tuning from scratch.

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

## Code signing and Windows SmartScreen

ClariStretch is an open-source project with no company behind it, and the
`.exe` built from the instructions above is **not code-signed**. Code signing
requires buying a certificate from a certificate authority (typically on the
order of $100-500/year) and re-signing every release, which isn't something
this project currently does. An unsigned `.exe` is completely normal for a
small open-source tool - it does not mean the file is unsafe - but it does
mean Windows has no cryptographic way to vouch for who built it, so it treats
it with suspicion by default.

Because of that, the first time you (or anyone) runs `ClariStretch.exe` on
Windows, **Windows Defender SmartScreen** will very likely block it with a
blue "Windows protected your PC" screen. This is Microsoft's standard
reputation check for any executable that isn't signed by a well-known
publisher or that few people have run yet - it isn't specific to
ClariStretch, and it isn't a virus/malware detection (that's a separate
thing - Windows Defender antivirus - and if that flags the file instead,
treat it as a real signal and don't bypass it).

To run the app anyway, once you've built it yourself (or downloaded it from
this project's own GitHub Releases page) and you're confident of where it
came from:

1. When the blue SmartScreen screen appears, click **"More info"** (a small
   link, easy to miss, in the body of the dialog).
2. A **"Run anyway"** button will appear - click it.
3. The app launches normally, and Windows remembers this choice for this
   specific file going forward.

If you don't see a SmartScreen prompt at all but the file still won't open,
right-click the `.exe` → **Properties** → check for an **"Unblock"** checkbox
near the bottom of the **General** tab (Windows adds this to files downloaded
from the internet) → check it → **Apply**.

Only do this for a copy of `ClariStretch.exe` you built yourself from this
repo's source, or downloaded directly from this project's own GitHub page -
never for a copy someone sent you another way, since SmartScreen's warning
is also the normal first line of defense against genuinely malicious
software, and "it's just unsigned, bypass the warning" is exactly what a
trojan would also want you to believe.

## Project status

This is an actively developed internal tool; interfaces and defaults (denoise
sigma scaling, star-mask thresholds, preview crop size, etc.) may still change.
