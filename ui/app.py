"""
TraceID TraceID — Multi-Object Re-Identification
UI: production-grade dark theme, video-first layout.
"""

from __future__ import annotations

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reid_engine import ReIDEngine
from core.embedding_extractor import EmbeddingExtractor
from core.pipeline import ReIDPipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# ── Identity palette: 12 vivid, perceptually-distinct colours ─────────────────
# Each has sufficient contrast on both a dark background (UI) and on video frames.
ID_PALETTE = [
    "#00D4FF",  # electric cyan
    "#00E676",  # vivid green
    "#FF6D00",  # amber
    "#E040FB",  # violet
    "#FF4081",  # rose
    "#76FF03",  # lime
    "#FFEA00",  # yellow
    "#40C4FF",  # sky blue
    "#FF6E40",  # deep orange
    "#64FFDA",  # teal
    "#B388FF",  # lavender
    "#F48FB1",  # pink
]

def _id_hex(gid: str) -> str:
    try:
        return ID_PALETTE[int(gid.split("-")[1], 16) % len(ID_PALETTE)]
    except Exception:
        return ID_PALETTE[0]

def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    """Convert #RRGGBB → (B, G, R) for OpenCV."""
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)

def _darken_bgr(bgr: tuple, t: float = 0.18) -> tuple[int, int, int]:
    return int(bgr[0] * t), int(bgr[1] * t), int(bgr[2] * t)


# ── Colour tokens ─────────────────────────────────────────────────────────────
# Backgrounds — four levels of depth
BG0 = "#070B0F"   # deepest: canvas letterbox
BG1 = "#0D1117"   # base: window background
BG2 = "#161B22"   # surface: sidebar, header, toolbar
BG3 = "#1C2333"   # elevated: cards, input fields, text boxes


BTN_TEXT = "#000000"

# Borders
BORDER_DIM    = "#21262D"   # hairline dividers
BORDER_MID    = "#30363D"   # panel borders, input outlines
BORDER_FOCUS  = "#388BFD"   # active input highlight

# Accents
A_BLUE   = "#388BFD"   # primary — links, focus rings, active accent
A_GREEN  = "#3FB950"   # success — Start, LIVE, NEW badge
A_RED    = "#F85149"   # danger  — Stop
A_AMBER  = "#D29922"   # warning — alerts
A_PURPLE = "#8957E5"   # reset / secondary action
A_CYAN   = "#39D3F2"   # HUD / threshold display

# Text — four levels of weight
T1 = "#E6EDF3"   # primary:   headings, values, important labels
T2 = "#8B949E"   # secondary: descriptions, sub-labels
T3 = "#484F58"   # muted:     placeholder, disabled, separator text
T4 = "#21262D"   # ghost:     borders used as "invisible" text

# Semantic status colours (also used in video overlay)
S_NEW  = "#3FB950"   # bright green — first-time identity
S_SEEN = "#58A6FF"   # bright blue  — known identity

# Fonts
FF = "Segoe UI"
FM = "Consolas"
F_XS      = (FF, 7)
F_SM      = (FF, 8)
F_BASE    = (FF, 9)
F_BASE_B  = (FF, 9, "bold")
F_MED     = (FF, 10)
F_MED_B   = (FF, 10, "bold")
F_LG_B    = (FF, 13, "bold")
F_XL_B    = (FF, 16, "bold")
F_MONO    = (FM, 8)
F_MONO_B  = (FM, 8, "bold")
F_MONO_MD = (FM, 9)


# ═══════════════════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class _Button(tk.Button):
    """
    Flat button with explicit normal / hover / disabled visual states.
    Disabled state uses a clearly different background so the UI never
    looks broken — just inactive.
    """
    _DARKEN = 40   # how much to darken on hover (0-255 per channel)

    def __init__(self, parent, text: str, bg: str, fg: str = BTN_TEXT,
                 disabled_bg: str = "#E5E7EB", disabled_fg: str = BTN_TEXT, **kw):
        self._bg_normal   = bg
        self._fg_normal   = fg
        self._bg_disabled = disabled_bg
        self._fg_disabled = disabled_fg
        self._bg_hover    = self._make_hover(bg)
        super().__init__(
            parent, text=text,
            bg=bg, fg=fg,
            font=F_BASE_B,
            relief=tk.FLAT, bd=0,
            padx=16, pady=8,
            cursor="hand2",
            activebackground=self._bg_hover,
            activeforeground=fg,
            **kw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)

    @staticmethod
    def _make_hover(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        # brighten slightly
        r = min(255, r + 25)
        g = min(255, g + 25)
        b = min(255, b + 25)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _enter(self, _=None):
        if str(self["state"]) != "disabled":
            self.config(bg=self._bg_hover)

    def _leave(self, _=None):
        if str(self["state"]) != "disabled":
            self.config(bg=self._bg_normal)

    def enable(self):
        self.config(state=tk.NORMAL, bg=self._bg_normal,
                    fg=self._fg_normal, cursor="hand2")

    def disable(self):
        self.config(state=tk.DISABLED, bg=self._bg_disabled,
                    fg=self._fg_disabled, cursor="arrow")


class _StatChip(tk.Frame):
    """
    Inline stat chip.  VALUE is shown in accent colour; LABEL in muted text.
    Example:  12  IDs
    """
    def __init__(self, parent, label: str, accent: str = A_BLUE, **kw):
        super().__init__(parent, bg=BG3,
                         highlightthickness=1,
                         highlightbackground=BORDER_DIM, **kw)
        inner = tk.Frame(self, bg=BG3)
        inner.pack(padx=12, pady=6)
        self._val = tk.Label(inner, text="—", bg=BG3,
                              fg=accent, font=F_MED_B)
        self._val.pack(side=tk.LEFT)
        tk.Label(inner, text=f"  {label}", bg=BG3,
                 fg=T3, font=F_SM).pack(side=tk.LEFT, pady=(1, 0))

    def set(self, v):
        self._val.config(text=str(v))


def _hline(parent, px=0, py=0):
    """Horizontal 1-px hairline divider."""
    tk.Frame(parent, bg=BORDER_DIM, height=1).pack(
        fill=tk.X, padx=px, pady=py)


def _vline(parent, px=0, py=0):
    """Vertical 1-px hairline divider."""
    tk.Frame(parent, bg=BORDER_DIM, width=1).pack(
        side=tk.LEFT, fill=tk.Y, padx=px, pady=py)


def _section_label(parent, text: str, pad_top: int = 14):
    """
    Compact ALL-CAPS section label with a 3-px left accent bar.
    Visual weight is intentionally low — it organises without competing.
    """
    row = tk.Frame(parent, bg=parent["bg"])
    row.pack(fill=tk.X, padx=16, pady=(pad_top, 6))
    tk.Frame(row, bg=A_BLUE, width=3).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    tk.Label(row, text=text.upper(), bg=parent["bg"],
             fg=T3, font=F_XS).pack(side=tk.LEFT, pady=2)


def _field_label(parent, text: str, padx: int = 16):
    tk.Label(parent, text=text, bg=parent["bg"],
             fg=T3, font=F_SM, anchor="w").pack(
        fill=tk.X, padx=padx, pady=(0, 3))


def _scrolled_text(parent, height=None, fg=T1) -> tk.Text:
    """Text widget + scrollbar, styled for the dark theme."""
    frame = tk.Frame(parent, bg=parent["bg"])
    frame.pack(fill=tk.BOTH, expand=(height is None))

    sb = tk.Scrollbar(frame, width=4,
                       bg=BG2, troughcolor=BG1, relief=tk.FLAT)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    kw = dict(
        bg=BG3, fg=fg,
        font=F_MONO,
        relief=tk.FLAT, bd=0,
        state=tk.DISABLED,
        wrap=tk.WORD,
        cursor="arrow",
        yscrollcommand=sb.set,
        selectbackground=BORDER_MID,
        inactiveselectbackground=BORDER_MID,
        spacing1=3, spacing3=3,
        padx=10, pady=8,
    )
    if height is not None:
        kw["height"] = height
    t = tk.Text(frame, **kw)
    t.pack(fill=tk.BOTH, expand=(height is None))
    sb.config(command=t.yview)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TraceID(tk.Tk):

    APP_NAME = "TraceID"
    APP_SUB  = "Multi-Object Re-Identification"

    SIDEBAR_W = 292
    CANVAS_W  = 960
    CANVAS_H  = 540
    FPS_CAP   = 30

    def __init__(self):
        super().__init__()
        self.title(f"{self.APP_NAME}  ·  {self.APP_SUB}")
        self.configure(bg=BG1)
        self.minsize(1160, 700)

        # runtime state
        self._running     = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._pipeline: Optional[ReIDPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._photo       = None          # must stay alive (Tk GC)
        self._fps         = 0.0
        self._frame_count = 0
        self._alert_shown = 0

        self._reid_engine = ReIDEngine(similarity_threshold=0.65)
        self._extractor   = EmbeddingExtractor()

        self._setup_ttk()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── TTK theme overrides ───────────────────────────────────────────────────

    def _setup_ttk(self):
        s = ttk.Style()
        s.theme_use("default")
        # Combobox — dark field, readable foreground, no white flash on selection
        s.configure("V.TCombobox",
                    fieldbackground=BG3,
                    background=BG3,
                    foreground=T1,
                    selectbackground=BORDER_MID,
                    selectforeground=T1,
                    bordercolor=BORDER_MID,
                    arrowcolor=T2,
                    darkcolor=BG1,
                    lightcolor=BG1,
                    relief="flat",
                    padding=6)
        s.map("V.TCombobox",
              fieldbackground=[("readonly", BG3), ("disabled", BG2)],
              foreground=[("readonly", T1), ("disabled", T3)],
              selectbackground=[("readonly", BORDER_MID)])

    # ── Top-level skeleton ────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_titlebar()
        _hline(self)

        body = tk.Frame(self, bg=BG1)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(body)
        _vline(body)                          # 1-px gutter between sidebar / main
        self._build_main_area(body)

        _hline(self)
        self._build_statusbar()

    # ══════════════════════════════════════════════════════════════════════════
    #  TITLE BAR
    # ══════════════════════════════════════════════════════════════════════════

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=BG2)
        bar.pack(fill=tk.X)

        inner = tk.Frame(bar, bg=BG2)
        inner.pack(fill=tk.X, padx=20, pady=12)

        # ── Logo area ──
        logo = tk.Frame(inner, bg=BG2)
        logo.pack(side=tk.LEFT)

        # Vertical accent stripe (3 px, full height of logo row)
        tk.Frame(logo, bg=A_BLUE, width=3).pack(
            side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        name_block = tk.Frame(logo, bg=BG2)
        name_block.pack(side=tk.LEFT)
        tk.Label(name_block, text=self.APP_NAME,
                 bg=BG2, fg=T1, font=F_XL_B).pack(anchor="w")
        tk.Label(name_block, text=self.APP_SUB,
                 bg=BG2, fg=T3, font=F_SM).pack(anchor="w")

        # ── Right: stat chips + live indicator ──
        right = tk.Frame(inner, bg=BG2)
        right.pack(side=tk.RIGHT)

        self._chip_ids    = _StatChip(right, "Identities", A_CYAN)
        self._chip_ids.pack(side=tk.LEFT, padx=(0, 6))

        self._chip_alerts = _StatChip(right, "Alerts",     A_AMBER)
        self._chip_alerts.pack(side=tk.LEFT, padx=(0, 6))

        self._chip_fps    = _StatChip(right, "FPS",        A_GREEN)
        self._chip_fps.pack(side=tk.LEFT, padx=(0, 6))

        self._chip_frames = _StatChip(right, "Frames",     T2)
        self._chip_frames.pack(side=tk.LEFT, padx=(0, 12))

        # LIVE dot — colour changes when running
        self._live_dot = tk.Label(right, text="●  OFFLINE",
                                   bg=BG2, fg=T3,
                                   font=F_BASE_B)
        self._live_dot.pack(side=tk.LEFT)

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=self.SIDEBAR_W)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        # ── Session controls ──────────────────────────────────────────────────
        _section_label(sb, "Session", pad_top=16)

        btn_row = tk.Frame(sb, bg=BG2)
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 6))

        self._btn_start = _Button(
            btn_row, "▶  Start", A_GREEN,
            disabled_bg=BG3, disabled_fg=T3, command=self._start)
        self._btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self._btn_stop = _Button(
            btn_row, "■  Stop", A_RED,
            disabled_bg=BG3, disabled_fg=T3, command=self._stop)
        self._btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._btn_stop.disable()

        self._btn_reset = _Button(
            sb, "↺  Reset All Identities", A_PURPLE,
            disabled_bg=BG3, disabled_fg=T3, command=self._reset)
        self._btn_reset.pack(fill=tk.X, padx=16, pady=(0, 4))

        _hline(sb, px=16, py=10)

        # ── Video source ──────────────────────────────────────────────────────
        _section_label(sb, "Video Source")

        _field_label(sb, "Camera index or file path")

        src_row = tk.Frame(sb, bg=BG2)
        src_row.pack(fill=tk.X, padx=16, pady=(0, 8))

        self._src_var = tk.StringVar(value="0")
        self._src_entry = tk.Entry(
            src_row, textvariable=self._src_var,
            bg=BG3, fg=T1,
            insertbackground=A_BLUE,
            relief=tk.FLAT, font=F_BASE, bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_MID,
            highlightcolor=BORDER_FOCUS)
        self._src_entry.pack(side=tk.LEFT, fill=tk.X,
                              expand=True, ipady=7, padx=(0, 6))

        browse = _Button(src_row, "Browse",  "#F3F4F6", fg=BTN_TEXT,
                         disabled_bg="#E5E7EB", disabled_fg=BTN_TEXT)
        browse.config(padx=10, pady=7)
        browse.config(command=self._browse)
        browse.pack(side=tk.LEFT)

        _field_label(sb, "Detection model")
        self._yolo_var = tk.StringVar(value="yolov8n.pt")
        self._yolo_cb = ttk.Combobox(
            sb, textvariable=self._yolo_var,
            style="V.TCombobox",
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                    "yolo11n.pt", "yolo11s.pt"],
            state="readonly", font=F_BASE)
        self._yolo_cb.pack(fill=tk.X, padx=16, pady=(0, 4))

        _hline(sb, px=16, py=10)

        # ── ReID sensitivity ──────────────────────────────────────────────────
        _section_label(sb, "ReID Sensitivity")

        thr_outer = tk.Frame(sb, bg=BG2)
        thr_outer.pack(fill=tk.X, padx=16, pady=(0, 4))

        self._thresh_var = tk.DoubleVar(value=0.65)

        # Large, readable threshold value on the right
        self._thresh_lbl = tk.Label(
            thr_outer, text="0.65",
            bg=BG2, fg=A_CYAN,
            font=(FF, 20, "bold"), width=5, anchor="e")
        self._thresh_lbl.pack(side=tk.RIGHT)

        tk.Scale(
            thr_outer,
            from_=0.30, to=0.95, resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._thresh_var,
            bg=BG2, fg=T1,
            troughcolor=BG3,
            activebackground=A_BLUE,
            highlightthickness=0,
            showvalue=False,
            command=self._on_threshold_change,
        ).pack(fill=tk.X, expand=True, side=tk.LEFT, pady=(4, 0))

        hint = tk.Frame(sb, bg=BG2)
        hint.pack(fill=tk.X, padx=16, pady=(0, 4))
        tk.Label(hint, text="← Stricter", bg=BG2, fg=T3,
                 font=F_XS).pack(side=tk.LEFT)
        tk.Label(hint, text="Looser →",  bg=BG2, fg=T3,
                 font=F_XS).pack(side=tk.RIGHT)

        _hline(sb, px=16, py=10)

        # ── Known identities (grows to fill remaining height) ─────────────────
        _section_label(sb, "Known Identities")

        id_wrap = tk.Frame(sb, bg=BG2)
        id_wrap.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 0))

        self._id_text = _scrolled_text(id_wrap, fg=T1)

        # Colour tags for the identity list
        self._id_text.tag_configure(
            "id_val", foreground=A_CYAN, font=F_MONO_B)
        self._id_text.tag_configure(
            "cls",    foreground=T1,    font=(FM, 8, "bold"))
        self._id_text.tag_configure(
            "new",    foreground=S_NEW, font=F_MONO_B)
        self._id_text.tag_configure(
            "seen",   foreground=S_SEEN)
        self._id_text.tag_configure(
            "meta",   foreground=T2)
        self._id_text.tag_configure(
            "sep",    foreground=BORDER_MID)
        self._id_text.tag_configure(
            "empty",  foreground=T3)

        _hline(sb, px=16, py=8)

        # ── Alert history (fixed height) ─────────────────────────────────────
        _section_label(sb, "Alert History", pad_top=0)

        al_wrap = tk.Frame(sb, bg=BG2)
        al_wrap.pack(fill=tk.X, padx=16, pady=(0, 16))

        self._al_text = _scrolled_text(al_wrap, height=6, fg=A_AMBER)
        self._al_text.tag_configure("ts",    foreground=T3)
        self._al_text.tag_configure("msg",   foreground=A_AMBER, font=F_MONO_B)
        self._al_text.tag_configure("empty", foreground=T3)

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN AREA  (toolbar + canvas)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_main_area(self, parent):
        main = tk.Frame(parent, bg=BG1)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_toolbar(main)
        _hline(main)
        self._build_canvas(main)

    def _build_toolbar(self, parent):
        tb = tk.Frame(parent, bg=BG2)
        tb.pack(fill=tk.X)

        inner = tk.Frame(tb, bg=BG2)
        inner.pack(fill=tk.X, padx=16, pady=10)

        # Left side: status pill + model info
        self._status_pill = tk.Label(
            inner,
            text="  ●  OFFLINE  ",
            bg=BG3, fg=T3,
            font=F_BASE_B,
            padx=6, pady=4,
            relief=tk.FLAT)
        self._status_pill.pack(side=tk.LEFT, padx=(0, 12))

        self._model_lbl = tk.Label(
            inner, text="No model loaded",
            bg=BG2, fg=T3, font=F_BASE)
        self._model_lbl.pack(side=tk.LEFT)

        # Right side: threshold readout + source path
        right = tk.Frame(inner, bg=BG2)
        right.pack(side=tk.RIGHT)

        tk.Label(right, text="Threshold:", bg=BG2,
                 fg=T3, font=F_SM).pack(side=tk.LEFT, padx=(0, 4))
        self._tb_thresh = tk.Label(right, text="0.65",
                                    bg=BG2, fg=A_CYAN, font=F_BASE_B)
        self._tb_thresh.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(right, text="Source:", bg=BG2,
                 fg=T3, font=F_SM).pack(side=tk.LEFT, padx=(0, 4))
        self._source_lbl = tk.Label(right, text="—",
                                     bg=BG2, fg=T2, font=F_BASE)
        self._source_lbl.pack(side=tk.LEFT)

    def _build_canvas(self, parent):
        wrap = tk.Frame(parent, bg=BG0)
        wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self._canvas = tk.Canvas(
            wrap,
            width=self.CANVAS_W, height=self.CANVAS_H,
            bg=BG0,
            highlightthickness=1,
            highlightbackground=BORDER_MID)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder()

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG2)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        inner = tk.Frame(bar, bg=BG2)
        inner.pack(fill=tk.X, padx=16, pady=5)

        self._status_var = tk.StringVar(
            value="Ready  ·  Configure a source and press  ▶ Start")
        tk.Label(inner, textvariable=self._status_var,
                 bg=BG2, fg=T3, font=F_SM, anchor="w").pack(side=tk.LEFT)

        tk.Label(inner, text="TraceID  v2.0",
                 bg=BG2, fg=T3, font=F_SM).pack(side=tk.RIGHT)

    # ══════════════════════════════════════════════════════════════════════════
    #  CONTROLS
    # ══════════════════════════════════════════════════════════════════════════

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if p:
            self._src_var.set(p)

    def _on_threshold_change(self, _=None):
        v = self._thresh_var.get()
        self._thresh_lbl.config(text=f"{v:.2f}")
        self._tb_thresh.config(text=f"{v:.2f}")
        self._reid_engine.threshold = v

    def _start(self):
        if self._running:
            return
        model_name = self._yolo_var.get()
        self._set_status(f"Loading {model_name} …")
        self.update()

        try:
            from ultralytics import YOLO
            yolo = YOLO(model_name)
        except Exception as e:
            messagebox.showerror("Model Error", f"Cannot load {model_name}:\n{e}")
            self._set_status("Model load failed.")
            return

        src_str = self._src_var.get().strip()
        src = int(src_str) if src_str.isdigit() else src_str
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            messagebox.showerror("Source Error", f"Cannot open: {src}")
            self._set_status("Cannot open video source.")
            return

        self._pipeline = ReIDPipeline(
            yolo_model=yolo,
            reid_engine=self._reid_engine,
            extractor=self._extractor,
        )

        self._running = True
        self._btn_start.disable()
        self._btn_stop.enable()
        self._yolo_cb.config(state=tk.DISABLED)
        self._src_entry.config(state=tk.DISABLED)
        self._set_live(True, model_name, str(src))
        self._thread = threading.Thread(target=self._video_loop, daemon=True)
        self._thread.start()
        self._set_status(f"Running  ·  {model_name}  ·  source: {src}")

    def _stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_start.enable()
        self._btn_stop.disable()
        self._yolo_cb.config(state="readonly")
        self._src_entry.config(state=tk.NORMAL)
        self._set_live(False)
        self._draw_placeholder()
        self._set_status("Stopped.")

    def _reset(self):
        if self._running:
            self._stop()
        (self._pipeline or self._reid_engine).reset() \
            if self._pipeline else self._reid_engine.reset()
        self._alert_shown = 0
        self._frame_count = 0
        self._fps = 0.0
        self._refresh_chips()
        self._refresh_identities()
        self._refresh_alerts()
        self._set_status("All identities cleared.")

    def _on_close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self.destroy()

    def _set_live(self, online: bool, model: str = "", source: str = ""):
        if online:
            self._status_pill.config(
                text="  ●  LIVE  ", fg=A_GREEN, bg="#0C2016")
            self._live_dot.config(text="●  LIVE", fg=A_GREEN)
            self._model_lbl.config(text=model, fg=T2)
            self._source_lbl.config(text=source, fg=T2)
        else:
            self._status_pill.config(
                text="  ●  OFFLINE  ", fg=T3, bg=BG3)
            self._live_dot.config(text="●  OFFLINE", fg=T3)
            self._model_lbl.config(text="No model loaded", fg=T3)
            self._source_lbl.config(text="—", fg=T3)

    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO LOOP  (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _video_loop(self):
        interval = 1.0 / self.FPS_CAP
        while self._running:
            t0 = time.perf_counter()

            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    break

            dets      = self._pipeline.process_frame(frame)
            annotated = self._annotate(frame, dets)
            self.after(0, self._update_canvas, annotated)
            self.after(0, self._refresh_all)

            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
            self._fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
            self._frame_count += 1

        self.after(0, self._stop)

    # ══════════════════════════════════════════════════════════════════════════
    #  FRAME ANNOTATION
    # ══════════════════════════════════════════════════════════════════════════

    def _annotate(self, frame: np.ndarray, detections) -> np.ndarray:
        """
        Draw professional-looking tracker annotations on a copy of the frame.

        Design goals
        ────────────
        • Corner-bracket box style — minimal, modern, uncluttered.
        • Frosted-glass label chip — dark semi-transparent background with a
          thin coloured left edge; text always readable regardless of scene.
        • Two-line label: (1) CLASS  ID in bright white; (2) status + track
          in a softer secondary colour.
        • Status dot: vivid green = new, vivid blue = seen before.
        • HUD in the top-right corner: FPS and active ID count.
        """
        out = frame.copy()
        fh, fw = out.shape[:2]

        for d in detections:
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
            hex_c  = _id_hex(d.global_id)
            bgr_c  = _hex_to_bgr(hex_c)
            dark_c = _darken_bgr(bgr_c, 0.15)
            is_new = d.status == "New"

            # ── Corner-bracket bounding box ───────────────────────────────────
            # Only the four corners are drawn, not the full rectangle.
            # This avoids visual noise while still clearly bounding the subject.
            arm    = max(10, min(20, (x2 - x1) // 5, (y2 - y1) // 5))
            thick  = 2

            for (px, py, dx, dy) in [
                (x1, y1, +1, +1),   # top-left
                (x2, y1, -1, +1),   # top-right
                (x1, y2, +1, -1),   # bottom-left
                (x2, y2, -1, -1),   # bottom-right
            ]:
                cv2.line(out, (px, py), (px + dx * arm, py),       bgr_c, thick)
                cv2.line(out, (px, py), (px,            py + dy * arm), bgr_c, thick)

            # Very subtle tinted fill inside the box (12 % opacity)
            ov = out.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), dark_c, -1)
            cv2.addWeighted(ov, 0.12, out, 0.88, 0, out)

            # ── Label chip ────────────────────────────────────────────────────
            font_main = cv2.FONT_HERSHEY_DUPLEX
            fs1  = 0.42   # class + ID line
            fs2  = 0.36   # status line
            lh   = 16     # line height in pixels
            px_  = 8      # left / right padding inside chip
            py_  = 5      # top / bottom padding

            line1 = f"{d.label.upper()}  {d.global_id}"
            line2 = ("NEW" if is_new else "SEEN") + f"  |  Track {d.track_id}"
            if d.similarity > 0:
                line2 += f"  |  Sim {d.similarity:.2f}"

            (w1, h1), _ = cv2.getTextSize(line1, font_main, fs1, 1)
            (w2,  _), _ = cv2.getTextSize(line2, font_main, fs2, 1)
            chip_w = max(w1, w2) + px_ * 2
            chip_h = lh * 2 + py_ * 2

            # Clamp chip above the box (or below if no room)
            if y1 - chip_h - 4 >= 0:
                cy1 = y1 - chip_h - 4
            else:
                cy1 = y2 + 4
            cy2 = cy1 + chip_h
            cx1 = min(x1, fw - chip_w)   # don't overflow frame right edge

            # Background: very dark semi-transparent fill
            chip_ov = out.copy()
            cv2.rectangle(chip_ov, (cx1, cy1), (cx1 + chip_w, cy2),
                          (8, 12, 18), -1)
            cv2.addWeighted(chip_ov, 0.82, out, 0.18, 0, out)

            # Coloured left accent stripe (3 px)
            cv2.rectangle(out, (cx1, cy1), (cx1 + 3, cy2), bgr_c, -1)

            # Status indicator dot (right edge of chip)
            dot_bgr = (55, 185, 55) if is_new else (64, 166, 255)
            cv2.circle(out,
                       (cx1 + chip_w - 8, cy1 + chip_h // 2),
                       4, dot_bgr, -1)

            # Line 1: CLASS  ID — bright, high-contrast white
            cv2.putText(out, line1,
                        (cx1 + px_, cy1 + py_ + h1),
                        font_main, fs1, (236, 242, 250), 1, cv2.LINE_AA)

            # Line 2: status  ·  track  — softer blue-grey
            cv2.putText(out, line2,
                        (cx1 + px_, cy1 + py_ + h1 + lh),
                        font_main, fs2, (110, 145, 175), 1, cv2.LINE_AA)

        # ── HUD: top-right corner ─────────────────────────────────────────────
        # Minimal — only FPS and active identity count.
        # Dark pill background so it's readable over any scene.
        hud = [
            f"FPS  {self._fps:.0f}",
            f"IDs  {len(self._reid_engine.identities)}",
        ]
        hf = cv2.FONT_HERSHEY_DUPLEX
        hfs = 0.38
        line_h = 14
        hud_sizes = [cv2.getTextSize(l, hf, hfs, 1)[0] for l in hud]
        hud_w = max(s[0] for s in hud_sizes) + 16
        hud_h = len(hud) * line_h + 10

        hx1 = fw - hud_w - 8
        hy1 = 8
        hx2 = fw - 8
        hy2 = hy1 + hud_h

        hud_ov = out.copy()
        cv2.rectangle(hud_ov, (hx1, hy1), (hx2, hy2), (8, 12, 18), -1)
        cv2.addWeighted(hud_ov, 0.75, out, 0.25, 0, out)

        for i, (line, sz) in enumerate(zip(hud, hud_sizes)):
            yt = hy1 + 5 + sz[1] + i * line_h
            cv2.putText(out, line, (hx1 + 8, yt),
                        hf, hfs, (100, 130, 155), 1, cv2.LINE_AA)

        return out

    # ══════════════════════════════════════════════════════════════════════════
    #  CANVAS  (main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_canvas(self, frame: np.ndarray):
        """
        Scale the annotated frame to fill the fixed canvas, centred.
        The canvas widget size never changes — this eliminates jitter.
        """
        from PIL import Image, ImageTk

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 8 or ch < 8:
            cw, ch = self.CANVAS_W, self.CANVAS_H

        fh, fw = frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw     = int(fw * scale)
        nh     = int(fh * scale)
        ox     = (cw - nw) // 2
        oy     = (ch - nh) // 2

        rgb         = cv2.cvtColor(
            cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, cw, ch, fill=BG0, outline="")
        self._canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

    def _draw_placeholder(self):
        self._canvas.delete("all")
        cw = self._canvas.winfo_width()  or self.CANVAS_W
        ch = self._canvas.winfo_height() or self.CANVAS_H
        cx, cy = cw // 2, ch // 2

        self._canvas.create_rectangle(0, 0, cw, ch, fill=BG0, outline="")

        # Dashed border inset
        m = 32
        self._canvas.create_rectangle(
            m, m, cw - m, ch - m,
            outline=BORDER_DIM, dash=(6, 8))

        # Corner marks (decorative, like a viewfinder)
        arm = 20
        for (px, py, dx, dy) in [
            (m, m, 1, 1), (cw - m, m, -1, 1),
            (m, ch - m, 1, -1), (cw - m, ch - m, -1, -1)
        ]:
            self._canvas.create_line(px, py, px + dx * arm, py,
                                      fill=BORDER_MID, width=2)
            self._canvas.create_line(px, py, px, py + dy * arm,
                                      fill=BORDER_MID, width=2)


        # Recalculate center based on current canvas size
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()

        if cw < 100:
            cw = self.CANVAS_W
        if ch < 100:
            ch = self.CANVAS_H

        cx = cw // 2
        cy = ch // 2

        # Crosshair
        gap = 14
        for x1, y1, x2, y2 in [
            (cx - 40, cy, cx - gap, cy),
            (cx + gap, cy, cx + 40, cy),
            (cx, cy - 40, cx, cy - gap),
            (cx, cy + gap, cx, cy + 40),
        ]:
            self._canvas.create_line(x1, y1, x2, y2,
                                    fill=BORDER_MID, width=1)

        # Centre circle
        r = 12
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                outline=BORDER_MID, width=1)

        self._canvas.create_text(
            cx, cy + 52,
            text="No video source active",
            fill="#9CA3AF",
            font=(FF, 14, "bold"),
            anchor="center",
            width=int(cw * 0.7)
        )

        self._canvas.create_text(
            cx, cy + 82,
            text="Configure source in the sidebar, then press Start",
            fill="#6B7280",
            font=(FF, 10),
            anchor="center",
            width=int(cw * 0.7)
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL REFRESH  (main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_all(self):
        self._refresh_chips()
        self._refresh_identities()
        self._refresh_alerts()

    def _refresh_chips(self):
        self._chip_ids.set(len(self._reid_engine.identities))
        self._chip_alerts.set(len(self._reid_engine.alerts))
        self._chip_fps.set(f"{self._fps:.0f}")
        self._chip_frames.set(self._frame_count)

    def _refresh_identities(self):
        ids = self._reid_engine.get_identities()
        t   = self._id_text
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not ids:
            t.insert(tk.END,
                     "No identities registered yet.\n"
                     "Start the system to begin tracking.\n",
                     "empty")
        else:
            for ident in ids:
                first = time.strftime("%H:%M:%S",
                                      time.localtime(ident.first_seen))
                last  = time.strftime("%H:%M:%S",
                                      time.localtime(ident.last_seen))

                # ID + class on one line
                t.insert(tk.END, f"{ident.global_id}", "id_val")
                t.insert(tk.END, f"  {ident.label}\n", "cls")

                # Status indicator
                if ident.times_seen == 1:
                    t.insert(tk.END, "  ● NEW\n", "new")
                else:
                    t.insert(tk.END,
                             f"  ✓ Seen {ident.times_seen}×\n", "seen")

                # Timestamps
                t.insert(tk.END,
                         f"  First  {first}\n"
                         f"  Last   {last}\n", "meta")

                # Subtle separator
                t.insert(tk.END, "  " + "╌" * 26 + "\n", "sep")

        t.config(state=tk.DISABLED)

    def _refresh_alerts(self):
        alerts = self._reid_engine.get_alerts()
        if len(alerts) == self._alert_shown:
            return
        self._alert_shown = len(alerts)

        t = self._al_text
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not alerts:
            t.insert(tk.END, "No alerts yet.\n", "empty")
        else:
            for a in reversed(alerts):
                ts = time.strftime("%H:%M:%S", time.localtime(a["time"]))
                t.insert(tk.END, f"[{ts}]  ", "ts")
                t.insert(tk.END, f"{a['message']}\n", "msg")

        t.config(state=tk.DISABLED)
        t.see("1.0")

    def _set_status(self, msg: str):
        n_ids  = len(self._reid_engine.identities)
        n_alts = len(self._reid_engine.alerts)
        self._status_var.set(
            f"{msg}   ·   {n_ids} identities"
            f"   ·   {n_alts} alerts"
            f"   ·   {self._frame_count} frames")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        from PIL import Image, ImageTk  # noqa: F401
    except ImportError:
        print("Pillow is required:  pip install Pillow")
        sys.exit(1)
    TraceID().mainloop()


if __name__ == "__main__":
    main()
