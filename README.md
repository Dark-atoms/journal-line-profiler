# Journal Line Profiler

[![tests](https://github.com/Dark-atoms/journal-line-profiler/actions/workflows/tests.yml/badge.svg)](https://github.com/Dark-atoms/journal-line-profiler/actions/workflows/tests.yml)

A small desktop tool for pulling **intensity line profiles out of images** (microscopy, simulation snapshots, spin-wave maps, camera images) and turning them into **publication-style plots** in the familiar Origin look: four-sided frame, inward ticks, bold labels.

Click two points on the image and you get the profile plot, ready to export as CSV or as a high-DPI figure.

## Features

- Open PNG, JPG, TIFF, BMP, GIF or WebP images
- Click two points to define the line; the profile updates immediately
- **Line-width averaging** across the line (odd widths, 1–101 px) to reduce noise
- Channel choice: luminance, red, green or blue
- Physical units: set an origin and a scale (unit per pixel)
- Optional normalisation (0–1) and Savitzky–Golay smoothing
- Plot styling: font, label and tick sizes, tick direction, minor ticks, axis ranges, tick spacing, line colour/style/width, markers
- Export: **CSV** of the profile, and figures as **PNG / PDF / SVG / EPS / TIFF** at 150–1200 DPI

## Install and run

Requires Python 3.9+ with Tk (included in the python.org installers for Windows and macOS; on Linux install `python3-tk`).

```bash
git clone https://github.com/Dark-atoms/journal-line-profiler
cd journal-line-profiler
pip install -r requirements.txt
python journal_line_profiler.py
```

Shortcuts: **Ctrl+O** open image, **Ctrl+S** export CSV, **Ctrl+E** export figure, **Esc** clear the line.

## Use the extraction in your own code

The core function works without the GUI:

```python
import numpy as np
from PIL import Image
from journal_line_profiler import extract_line_profile

img = np.array(Image.open("snapshot.png"))
distance_px, intensity = extract_line_profile(img, x0=10, y0=50, x1=300, y1=50,
                                              line_width=5, channel="luminance")
```

Importing the module needs Tk installed (it defines the GUI), but no display is needed to call `extract_line_profile`.

## Tests

```bash
pip install pytest
pytest
```

## License

MIT. See [LICENSE](LICENSE).

*"Origin" and "OriginPro" are products of OriginLab Corporation. This project is independent and only imitates the plot style.*
