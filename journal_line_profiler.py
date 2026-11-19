"""
Journal Line Profiler  –  Origin-Style Output
==============================================
Extract line profiles from microscopy / simulation images and produce
publication-quality graphs styled like OriginLab / OriginPro.

Features:
  • Load any image (PNG, JPG, TIFF, BMP …)
  • Click two points to define the profile line
  • Extract intensity with line-width averaging
  • Tune origin, physical scale, axis range, tick interval
  • Origin-style graph: 4-sided frame, inward ticks, bold labels,
    Arial/Times fonts, clean white background
  • Line style, colour, width, marker controls
  • Optional smoothing (Savitzky–Golay)
  • Export as CSV and high-DPI figure (PNG / PDF / SVG / EPS / TIFF)

Usage:
    python journal_line_profiler.py

Dependencies:
    pip install numpy matplotlib Pillow scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
import os, csv


# ──────────────────────────────────────────────────────────────
#  Core extraction
# ──────────────────────────────────────────────────────────────

def extract_line_profile(image_array, x0, y0, x1, y1,
                         num_points=None, line_width=1, channel="luminance"):
    img = np.asarray(image_array, dtype=np.float64)
    if img.ndim == 3:
        ch = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
        if channel in ch and ch[channel] < img.shape[2]:
            img = img[:, :, ch[channel]]
        else:
            if img.shape[2] >= 3:
                img = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
            else:
                img = img[:,:,0]

    length = np.hypot(x1 - x0, y1 - y0)
    if num_points is None:
        num_points = max(int(np.ceil(length)), 2)

    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    px, py = -dy, dx

    t = np.linspace(0, 1, num_points)
    distances = t * length

    line_width = max(1, int(line_width))
    if line_width % 2 == 0:
        line_width += 1
    offsets = np.arange(line_width) - line_width // 2

    values = np.zeros(num_points)
    for off in offsets:
        rc = y0 + t*(y1-y0) + off*py
        cc = x0 + t*(x1-x0) + off*px
        values += map_coordinates(img, np.vstack([rc, cc]), order=1, mode='nearest')
    values /= len(offsets)
    return distances, values


# ──────────────────────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────────────────────

class JournalLineProfiler(tk.Tk):

    # Dark-theme palette for the app chrome (NOT the graph)
    BG   = "#1a1b2e"
    BG2  = "#232440"
    BG3  = "#2e2f50"
    FG   = "#d0d4f0"
    ACC  = "#7ca6f8"
    ACC2 = "#78d6a0"
    WARN = "#f0a878"
    BDR  = "#404060"

    def __init__(self):
        super().__init__()
        self.title("Journal Line Profiler  ·  Origin-Style")
        self.configure(bg=self.BG)
        self.geometry("1500x900")
        self.minsize(1200, 750)

        # -- state --
        self.image_array = None
        self.display_image = None
        self.tk_image = None
        self.scale_factor = 1.0
        self.points = []
        self.line_id = None
        self.marker_ids = []
        self.distances = None
        self.values = None
        self.image_path = None
        self.line_color = "#000000"   # for the exported graph line

        self._build_styles()
        self._build_ui()
        self._bind_shortcuts()

    # ── ttk styles ────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("D.TFrame",       background=self.BG)
        s.configure("Card.TFrame",    background=self.BG2)
        s.configure("D.TLabel",       background=self.BG,  foreground=self.FG,
                    font=("Segoe UI", 10))
        s.configure("H.TLabel",       background=self.BG,  foreground=self.ACC,
                    font=("Segoe UI", 13, "bold"))
        s.configure("C.TLabel",       background=self.BG2, foreground=self.FG,
                    font=("Segoe UI", 10))
        s.configure("Sec.TLabel",     background=self.BG2, foreground=self.ACC,
                    font=("Segoe UI", 10, "bold"))
        for name, bg_c in [("Acc", self.ACC), ("Wrn", self.WARN), ("Grn", self.ACC2)]:
            s.configure(f"{name}.TButton", background=bg_c, foreground="#1a1b2e",
                        font=("Segoe UI", 10, "bold"), padding=(10, 5))
            s.map(f"{name}.TButton", background=[("active", bg_c), ("disabled", self.BG3)])
        s.configure("D.TLabelframe",       background=self.BG2, foreground=self.ACC,
                    font=("Segoe UI", 10, "bold"))
        s.configure("D.TLabelframe.Label", background=self.BG2, foreground=self.ACC,
                    font=("Segoe UI", 10, "bold"))
        s.configure("D.TCheckbutton", background=self.BG2, foreground=self.FG,
                    font=("Segoe UI", 10))
        s.configure("D.TCombobox", fieldbackground=self.BG3, background=self.BG3,
                    foreground=self.FG)

    # ── layout ────────────────────────────────────────────────

    def _build_ui(self):
        # --- top bar ---
        top = ttk.Frame(self, style="D.TFrame")
        top.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(top, text="📐 Journal Line Profiler", style="H.TLabel").pack(side="left")

        bf = ttk.Frame(top, style="D.TFrame")
        bf.pack(side="right")
        for txt, sty, cmd in [
            ("📂 Open Image",   "Acc.TButton", self._open_image),
            ("🔄 Reset Line",   "Wrn.TButton", self._reset_line),
            ("💾 Export CSV",    "Grn.TButton", self._export_csv),
            ("🖼 Export Figure", "Grn.TButton", self._export_figure),
        ]:
            ttk.Button(bf, text=txt, style=sty, command=cmd).pack(side="left", padx=3)

        # --- body ---
        body = ttk.Frame(self, style="D.TFrame")
        body.pack(fill="both", expand=True, padx=10, pady=6)

        # LEFT: image canvas
        left = ttk.Frame(body, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(left, bg="#0e0f1a", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)

        self.status_var = tk.StringVar(value="Open an image to begin  (Ctrl+O)")
        tk.Label(left, textvariable=self.status_var, bg=self.BG2, fg=self.ACC,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=4, pady=(0,4))

        # RIGHT: settings + preview
        right = ttk.Frame(body, style="D.TFrame", width=440)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # scrollable settings area
        settings_canvas = tk.Canvas(right, bg=self.BG, highlightthickness=0, width=420)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=settings_canvas.yview)
        settings_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        settings_canvas.pack(side="left", fill="both", expand=True)

        self.settings_frame = ttk.Frame(settings_canvas, style="D.TFrame")
        settings_canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        self.settings_frame.bind("<Configure>",
            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            settings_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        settings_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_extraction_settings()
        self._build_graph_settings()
        self._build_plot_preview()

    # ── extraction settings ───────────────────────────────────

    def _build_extraction_settings(self):
        sf = ttk.LabelFrame(self.settings_frame, text="  Line Extraction",
                            style="D.TLabelframe", padding=8)
        sf.pack(fill="x", padx=4, pady=(0, 4))

        fields = [
            ("Origin X (px)",   "origin_x_var",   0.0),
            ("Origin Y (px)",   "origin_y_var",   0.0),
            ("Scale (unit/px)", "scale_var",       1.0),
        ]
        for r, (label, attr, default) in enumerate(fields):
            ttk.Label(sf, text=label+":", style="C.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, attr, var)
            tk.Entry(sf, textvariable=var, width=10, bg=self.BG3, fg=self.FG,
                     insertbackground=self.FG, font=("Consolas", 10),
                     relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)

        r = len(fields)
        ttk.Label(sf, text="Unit label:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.unit_var = tk.StringVar(value="µm")
        tk.Entry(sf, textvariable=self.unit_var, width=10, bg=self.BG3, fg=self.FG,
                 insertbackground=self.FG, font=("Consolas", 10),
                 relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        ttk.Label(sf, text="Line width (px):", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.linewidth_var = tk.IntVar(value=1)
        tk.Spinbox(sf, from_=1, to=101, increment=2, textvariable=self.linewidth_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        ttk.Label(sf, text="Channel:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.channel_var = tk.StringVar(value="luminance")
        ttk.Combobox(sf, textvariable=self.channel_var, width=12,
                     values=["luminance", "red", "green", "blue"],
                     state="readonly", style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        self.normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text="Normalize (0–1)", variable=self.normalize_var,
                        style="D.TCheckbutton").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1

        # Smoothing
        ttk.Label(sf, text="Smooth window:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.smooth_var = tk.IntVar(value=0)
        tk.Spinbox(sf, from_=0, to=201, increment=2, textvariable=self.smooth_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)

        sf.columnconfigure(1, weight=1)

    # ── graph style settings (Origin-like) ────────────────────

    def _build_graph_settings(self):
        gf = ttk.LabelFrame(self.settings_frame, text="  Origin-Style Graph",
                            style="D.TLabelframe", padding=8)
        gf.pack(fill="x", padx=4, pady=(0, 4))

        r = 0
        # Font
        ttk.Label(gf, text="Font:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.font_var = tk.StringVar(value="Arial")
        ttk.Combobox(gf, textvariable=self.font_var, width=14,
                     values=["Arial", "Times New Roman", "Helvetica",
                             "Calibri", "CMU Serif", "DejaVu Sans"],
                     state="readonly", style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Font size
        ttk.Label(gf, text="Label font size:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.fontsize_var = tk.IntVar(value=14)
        tk.Spinbox(gf, from_=8, to=28, textvariable=self.fontsize_var, width=6,
                   bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Tick size
        ttk.Label(gf, text="Tick font size:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.ticksize_var = tk.IntVar(value=12)
        tk.Spinbox(gf, from_=6, to=24, textvariable=self.ticksize_var, width=6,
                   bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Bold labels
        self.bold_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="Bold axis labels", variable=self.bold_labels_var,
                        style="D.TCheckbutton").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1

        # X-axis label
        ttk.Label(gf, text="X label:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.xlabel_var = tk.StringVar(value="Distance (µm)")
        tk.Entry(gf, textvariable=self.xlabel_var, width=18, bg=self.BG3, fg=self.FG,
                 insertbackground=self.FG, font=("Consolas", 10),
                 relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Y-axis label
        ttk.Label(gf, text="Y label:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.ylabel_var = tk.StringVar(value="Intensity (a.u.)")
        tk.Entry(gf, textvariable=self.ylabel_var, width=18, bg=self.BG3, fg=self.FG,
                 insertbackground=self.FG, font=("Consolas", 10),
                 relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Title
        ttk.Label(gf, text="Title:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.title_var = tk.StringVar(value="")
        tk.Entry(gf, textvariable=self.title_var, width=18, bg=self.BG3, fg=self.FG,
                 insertbackground=self.FG, font=("Consolas", 10),
                 relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # --- Axis ranges ---
        ttk.Label(gf, text="── Axis Range ──", style="Sec.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6,2))
        r += 1

        self.auto_range_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="Auto range", variable=self.auto_range_var,
                        style="D.TCheckbutton").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1

        for label, attr in [("X min", "xmin_var"), ("X max", "xmax_var"),
                            ("Y min", "ymin_var"), ("Y max", "ymax_var")]:
            ttk.Label(gf, text=label+":", style="C.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="")
            setattr(self, attr, var)
            tk.Entry(gf, textvariable=var, width=10, bg=self.BG3, fg=self.FG,
                     insertbackground=self.FG, font=("Consolas", 10),
                     relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            r += 1

        # Tick interval
        ttk.Label(gf, text="── Ticks ──", style="Sec.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6,2))
        r += 1

        for label, attr in [("X major tick Δ", "xtick_var"),
                            ("Y major tick Δ", "ytick_var")]:
            ttk.Label(gf, text=label+":", style="C.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="")
            setattr(self, attr, var)
            tk.Entry(gf, textvariable=var, width=10, bg=self.BG3, fg=self.FG,
                     insertbackground=self.FG, font=("Consolas", 10),
                     relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            r += 1

        ttk.Label(gf, text="Minor ticks /Δ:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.minor_ticks_var = tk.IntVar(value=5)
        tk.Spinbox(gf, from_=0, to=10, textvariable=self.minor_ticks_var, width=6,
                   bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Tick direction
        ttk.Label(gf, text="Tick direction:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.tick_dir_var = tk.StringVar(value="in")
        ttk.Combobox(gf, textvariable=self.tick_dir_var, width=8,
                     values=["in", "out", "inout"],
                     state="readonly", style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # --- Line style ---
        ttk.Label(gf, text="── Line Style ──", style="Sec.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6,2))
        r += 1

        # Line color
        ttk.Label(gf, text="Line colour:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.color_btn = tk.Button(gf, text="  ■  ", bg=self.line_color,
                                   fg="white", relief="flat",
                                   font=("Consolas", 10),
                                   command=self._pick_color)
        self.color_btn.grid(row=r, column=1, sticky="w", padx=4, pady=2)
        r += 1

        # Line width
        ttk.Label(gf, text="Line width:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.plot_lw_var = tk.DoubleVar(value=1.5)
        tk.Spinbox(gf, from_=0.5, to=6, increment=0.5, textvariable=self.plot_lw_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Line style
        ttk.Label(gf, text="Line style:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.linestyle_var = tk.StringVar(value="—  solid")
        ttk.Combobox(gf, textvariable=self.linestyle_var, width=14,
                     values=["—  solid", "-- dashed", "·· dotted",
                             "-· dashdot", "none"],
                     state="readonly", style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Marker
        ttk.Label(gf, text="Marker:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.marker_var = tk.StringVar(value="none")
        ttk.Combobox(gf, textvariable=self.marker_var, width=14,
                     values=["none", "○  circle", "□  square",
                             "△  triangle", "◇  diamond", "+  plus", "×  cross"],
                     state="readonly", style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Marker every N
        ttk.Label(gf, text="Marker every N:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.markevery_var = tk.IntVar(value=10)
        tk.Spinbox(gf, from_=1, to=200, textvariable=self.markevery_var, width=6,
                   bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # --- Export size ---
        ttk.Label(gf, text="── Export ──", style="Sec.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6,2))
        r += 1

        ttk.Label(gf, text="Fig width (in):", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.fig_w_var = tk.DoubleVar(value=6.0)
        tk.Spinbox(gf, from_=3, to=12, increment=0.5, textvariable=self.fig_w_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        ttk.Label(gf, text="Fig height (in):", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.fig_h_var = tk.DoubleVar(value=4.5)
        tk.Spinbox(gf, from_=2, to=10, increment=0.5, textvariable=self.fig_h_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        ttk.Label(gf, text="DPI:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.dpi_var = tk.IntVar(value=300)
        ttk.Combobox(gf, textvariable=self.dpi_var, width=8,
                     values=[150, 300, 600, 1200],
                     style="D.TCombobox").grid(
            row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # Frame linewidth
        ttk.Label(gf, text="Frame width:", style="C.TLabel").grid(
            row=r, column=0, sticky="w", pady=2)
        self.frame_lw_var = tk.DoubleVar(value=1.5)
        tk.Spinbox(gf, from_=0.5, to=4, increment=0.5, textvariable=self.frame_lw_var,
                   width=6, bg=self.BG3, fg=self.FG, font=("Consolas", 10),
                   relief="flat").grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        r += 1

        # UPDATE button
        ttk.Button(gf, text="🔃  Update Profile", style="Acc.TButton",
                   command=self._update_profile).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        gf.columnconfigure(1, weight=1)

    # ── preview plot ──────────────────────────────────────────

    def _build_plot_preview(self):
        pf = ttk.LabelFrame(self.settings_frame, text="  Preview",
                            style="D.TLabelframe", padding=4)
        pf.pack(fill="x", padx=4, pady=(0, 4))

        self.fig = Figure(figsize=(4.2, 3.0), dpi=96, facecolor="white")
        self.ax = self.fig.add_subplot(111)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._origin_style_axes(self.ax, preview=True)

        toolbar_f = ttk.Frame(pf, style="Card.TFrame")
        toolbar_f.pack(fill="x")
        NavigationToolbar2Tk(self.plot_canvas, toolbar_f)

    # ── Origin-style axis formatting ──────────────────────────

    def _origin_style_axes(self, ax, preview=False):
        """Apply OriginLab-like formatting to an axes object."""
        font = self.font_var.get() if hasattr(self, 'font_var') else "Arial"
        label_size = self.fontsize_var.get() if hasattr(self, 'fontsize_var') else 14
        tick_size  = self.ticksize_var.get() if hasattr(self, 'ticksize_var') else 12
        bold  = self.bold_labels_var.get() if hasattr(self, 'bold_labels_var') else True
        tick_dir = self.tick_dir_var.get() if hasattr(self, 'tick_dir_var') else "in"
        minor_n  = self.minor_ticks_var.get() if hasattr(self, 'minor_ticks_var') else 5
        frame_lw = self.frame_lw_var.get() if hasattr(self, 'frame_lw_var') else 1.5

        weight = "bold" if bold else "normal"

        ax.set_facecolor("white")

        # 4-sided border (Origin style)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(frame_lw)
            spine.set_color("black")

        # Tick styling
        major_len = 6 if not preview else 4
        minor_len = 3 if not preview else 2
        ax.tick_params(axis='both', which='major', direction=tick_dir,
                       length=major_len, width=frame_lw, labelsize=tick_size,
                       top=True, right=True, labeltop=False, labelright=False,
                       color="black", labelcolor="black")
        ax.tick_params(axis='both', which='minor', direction=tick_dir,
                       length=minor_len, width=frame_lw * 0.7,
                       top=True, right=True, color="black")

        # Font
        ax.xaxis.label.set_fontfamily(font)
        ax.yaxis.label.set_fontfamily(font)
        ax.xaxis.label.set_fontweight(weight)
        ax.yaxis.label.set_fontweight(weight)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_fontfamily(font)
        ax.title.set_fontweight(weight)
        ax.title.set_fontsize(label_size + 1)
        ax.title.set_color("black")

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(font)
            lbl.set_color("black")

        # Minor ticks
        if minor_n > 0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_n))
            ax.yaxis.set_minor_locator(AutoMinorLocator(minor_n))

    def _get_linestyle(self):
        ls_map = {
            "—  solid": "-", "-- dashed": "--", "·· dotted": ":",
            "-· dashdot": "-.", "none": "None"
        }
        return ls_map.get(self.linestyle_var.get(), "-")

    def _get_marker(self):
        m_map = {
            "none": None, "○  circle": "o", "□  square": "s",
            "△  triangle": "^", "◇  diamond": "D",
            "+  plus": "+", "×  cross": "x"
        }
        return m_map.get(self.marker_var.get(), None)

    # ── colour picker ─────────────────────────────────────────

    def _pick_color(self):
        c = colorchooser.askcolor(initialcolor=self.line_color, title="Line Colour")
        if c and c[1]:
            self.line_color = c[1]
            self.color_btn.configure(bg=self.line_color)

    # ── keyboard shortcuts ────────────────────────────────────

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._open_image())
        self.bind("<Control-s>", lambda e: self._export_csv())
        self.bind("<Escape>",    lambda e: self._reset_line())
        self.bind("<Control-e>", lambda e: self._export_figure())

    # ── open image ────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files",
                        "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            pil_img = Image.open(path)
            # Convert palette / RGBA-on-demand images so numpy is happy
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGB")
            self.image_array = np.array(pil_img)
            self.image_path = path
            self._reset_line()
            self._render_image()
            h, w = self.image_array.shape[:2]
            self.status_var.set(
                f"Loaded: {os.path.basename(path)}  |  {w}×{h} px  |  "
                "Click two points to define a line")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open image:\n{exc}")

    # ── render image ──────────────────────────────────────────

    def _render_image(self):
        if self.image_array is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        ih, iw = self.image_array.shape[:2]
        self.scale_factor = min(cw/iw, ch/ih, 1.0)
        nw = max(int(iw * self.scale_factor), 1)
        nh = max(int(ih * self.scale_factor), 1)
        pil = Image.fromarray(self.image_array).resize((nw, nh), Image.LANCZOS)
        self.display_image = pil
        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor="center", image=self.tk_image)
        self._img_ox = (cw - nw) / 2
        self._img_oy = (ch - nh) / 2
        self._draw_markers()

    def _on_resize(self, event):
        self._render_image()

    # ── canvas click ──────────────────────────────────────────

    def _on_click(self, event):
        if self.image_array is None:
            return
        ox = (event.x - self._img_ox) / self.scale_factor
        oy = (event.y - self._img_oy) / self.scale_factor
        ih, iw = self.image_array.shape[:2]
        if ox < 0 or oy < 0 or ox >= iw or oy >= ih:
            return
        if len(self.points) >= 2:
            self.points = []
        self.points.append((ox, oy))
        self._draw_markers()
        if len(self.points) == 1:
            self.status_var.set(f"Start: ({ox:.0f}, {oy:.0f})  →  Click second point")
        elif len(self.points) == 2:
            x0,y0 = self.points[0]; x1,y1 = self.points[1]
            self.status_var.set(
                f"Line: ({x0:.0f},{y0:.0f})→({x1:.0f},{y1:.0f})  |  "
                f"Length: {np.hypot(x1-x0,y1-y0):.1f} px")
            self._update_profile()

    # ── draw markers ──────────────────────────────────────────

    def _draw_markers(self):
        for mid in self.marker_ids:
            self.canvas.delete(mid)
        self.marker_ids.clear()
        if self.line_id:
            self.canvas.delete(self.line_id)
            self.line_id = None
        if not self.points:
            return
        def tc(px, py):
            return (px*self.scale_factor + self._img_ox,
                    py*self.scale_factor + self._img_oy)
        for px, py in self.points:
            cx, cy = tc(px, py)
            r = 5
            ids = [
                self.canvas.create_oval(cx-r,cy-r,cx+r,cy+r,
                    outline="#ff3333", fill="", width=2),
                self.canvas.create_line(cx-r-3,cy,cx+r+3,cy, fill="#ff3333", width=1),
                self.canvas.create_line(cx,cy-r-3,cx,cy+r+3, fill="#ff3333", width=1),
            ]
            self.marker_ids.extend(ids)
        if len(self.points) == 2:
            c0, c1 = tc(*self.points[0]), tc(*self.points[1])
            self.line_id = self.canvas.create_line(
                c0[0],c0[1],c1[0],c1[1], fill="#00ffaa", width=2, dash=(6,3))
            # width band
            lw = self.linewidth_var.get()
            if lw > 1:
                x0,y0 = self.points[0]; x1,y1 = self.points[1]
                L = np.hypot(x1-x0, y1-y0)
                if L > 0:
                    dx,dy = (x1-x0)/L, (y1-y0)/L
                    px,py = -dy, dx
                    h = lw/2
                    corners = [(x0+h*px,y0+h*py),(x1+h*px,y1+h*py),
                               (x1-h*px,y1-h*py),(x0-h*px,y0-h*py)]
                    flat = [c for pt in corners for c in tc(*pt)]
                    self.marker_ids.append(
                        self.canvas.create_polygon(*flat, outline="#00ffaa",
                            fill="", width=1, dash=(3,3)))

    # ── compute + plot ────────────────────────────────────────

    def _update_profile(self):
        if self.image_array is None or len(self.points) < 2:
            return
        x0,y0 = self.points[0]; x1,y1 = self.points[1]

        try:
            distances, values = extract_line_profile(
                self.image_array, x0, y0, x1, y1,
                line_width=self.linewidth_var.get(),
                channel=self.channel_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc)); return

        # origin shift
        L = np.hypot(x1-x0, y1-y0)
        if L > 0:
            dx,dy = (x1-x0)/L, (y1-y0)/L
            proj = (x0 - self.origin_x_var.get())*dx + (y0 - self.origin_y_var.get())*dy
        else:
            proj = 0
        distances = (distances + proj) * self.scale_var.get()

        # normalise
        if self.normalize_var.get():
            mn, mx = values.min(), values.max()
            if mx > mn:
                values = (values - mn) / (mx - mn)

        # smoothing (Savitzky–Golay)
        sw = self.smooth_var.get()
        if sw >= 5 and len(values) > sw:
            if sw % 2 == 0:
                sw += 1
            values = savgol_filter(values, sw, min(3, sw-1))

        self.distances = distances
        self.values = values

        # update preview
        self.ax.clear()
        self._plot_on_axes(self.ax)
        self._origin_style_axes(self.ax, preview=True)
        self.fig.tight_layout()
        self.plot_canvas.draw()
        self._draw_markers()

    def _plot_on_axes(self, ax):
        """Plot the current data on a given axes with the chosen styles."""
        ls = self._get_linestyle()
        mk = self._get_marker()
        me = self.markevery_var.get()

        kwargs = dict(color=self.line_color,
                      linewidth=self.plot_lw_var.get(),
                      linestyle=ls if ls != "None" else "None",
                      markersize=5)
        if mk:
            kwargs["marker"] = mk
            kwargs["markevery"] = me
            kwargs["markerfacecolor"] = "white"
            kwargs["markeredgecolor"] = self.line_color
            kwargs["markeredgewidth"] = 1.2

        ax.plot(self.distances, self.values, **kwargs)

        # axis labels
        ax.set_xlabel(self.xlabel_var.get())
        ax.set_ylabel(self.ylabel_var.get())
        t = self.title_var.get()
        if t:
            ax.set_title(t)

        # axis range
        if not self.auto_range_var.get():
            try:
                xmn = float(self.xmin_var.get()); xmx = float(self.xmax_var.get())
                ax.set_xlim(xmn, xmx)
            except ValueError:
                pass
            try:
                ymn = float(self.ymin_var.get()); ymx = float(self.ymax_var.get())
                ax.set_ylim(ymn, ymx)
            except ValueError:
                pass

        # tick intervals
        try:
            xt = float(self.xtick_var.get())
            if xt > 0:
                ax.xaxis.set_major_locator(MultipleLocator(xt))
        except ValueError:
            pass
        try:
            yt = float(self.ytick_var.get())
            if yt > 0:
                ax.yaxis.set_major_locator(MultipleLocator(yt))
        except ValueError:
            pass

    # ── export CSV ────────────────────────────────────────────

    def _export_csv(self):
        if self.distances is None:
            messagebox.showinfo("No data", "Extract a profile first."); return
        base = os.path.splitext(os.path.basename(self.image_path or "profile"))[0]
        path = filedialog.asksaveasfilename(
            title="Save CSV", defaultextension=".csv",
            initialfile=f"{base}_profile.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.xlabel_var.get(), self.ylabel_var.get()])
            for d, v in zip(self.distances, self.values):
                w.writerow([f"{d:.6g}", f"{v:.6g}"])
        self.status_var.set(f"Saved CSV → {os.path.basename(path)}")

    # ── export figure (Origin quality) ────────────────────────

    def _export_figure(self):
        if self.distances is None:
            messagebox.showinfo("No data", "Extract a profile first."); return
        base = os.path.splitext(os.path.basename(self.image_path or "profile"))[0]
        path = filedialog.asksaveasfilename(
            title="Save Figure", defaultextension=".png",
            initialfile=f"{base}_origin_plot.png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg"),
                       ("EPS","*.eps"),("TIFF","*.tiff"),("All","*.*")])
        if not path:
            return

        fw = self.fig_w_var.get()
        fh = self.fig_h_var.get()
        dpi = int(self.dpi_var.get())

        fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi, facecolor="white")
        self._plot_on_axes(ax)
        self._origin_style_axes(ax, preview=False)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white",
                    edgecolor="none")
        plt.close(fig)
        self.status_var.set(f"Saved figure → {os.path.basename(path)}")

    # ── reset ─────────────────────────────────────────────────

    def _reset_line(self):
        self.points.clear()
        self.distances = None
        self.values = None
        if self.line_id:
            self.canvas.delete(self.line_id)
            self.line_id = None
        for mid in self.marker_ids:
            self.canvas.delete(mid)
        self.marker_ids.clear()
        self.ax.clear()
        self._origin_style_axes(self.ax, preview=True)
        self.fig.tight_layout()
        self.plot_canvas.draw()
        if self.image_array is not None:
            self.status_var.set("Line cleared. Click two points for a new line.")
        else:
            self.status_var.set("Open an image to begin  (Ctrl+O)")


if __name__ == "__main__":
    app = JournalLineProfiler()
    app.mainloop()
