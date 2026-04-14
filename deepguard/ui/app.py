"""
DeepGuard Desktop App
======================
Accurate, visually rich deepfake detection GUI.
Tabs: Detect | History | Models | Help
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

# PIL for image display
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

from inference.engine import DeepGuardEngine
from inference.text_engine import analyse_text

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0b0f19",      # Sleek, extremely dark blue-slate
    "surface":   "#151e2e",      # Elevated dark base
    "surface2":  "#1e293b",      # Slightly lifted layer (glass-like depth)
    "border":    "#334155",      # Subtle but clear divider
    "accent":    "#00f0ff",      # Radiant cyan for main actions
    "accent2":   "#bb86fc",      # Soft vibrant violet for alternative accents
    "success":   "#00e676",      # Bright neon green
    "danger":    "#ff5252",      # Alerting vivid crimson
    "warning":   "#ffb74d",      # Warm golden orange
    "text":      "#f8fafc",      # Crisp high-contrast white
    "muted":     "#94a3b8",      # Smooth readable silver
    "dim":       "#475569",      # Dimmed placeholder/inactive
}

F_TITLE  = ("Segoe UI", 28, "bold")
F_HEAD   = ("Segoe UI Semibold", 13)
F_BODY   = ("Segoe UI", 12)
F_SMALL  = ("Segoe UI", 11)
F_MONO   = ("Consolas", 12)
F_BIG    = ("Segoe UI", 42, "bold")
F_MED    = ("Segoe UI Semibold", 18)


# ── Custom Widgets ────────────────────────────────────────────────────────────
class AnimBar(tk.Canvas):
    """Thin animated progress bar."""
    def __init__(self, parent, w=500, h=5, **kw):
        super().__init__(parent, width=w, height=h,
                         bg=C["border"], highlightthickness=0, **kw)
        self._width = w; self._height = h; self._v = 0

    def set(self, v):
        self._v = max(0, min(100, v))
        self.delete("all")
        fw = int(self._width * self._v / 100)
        if fw > 0:
            self.create_rectangle(0, 0, fw, self._height, fill=C["accent2"], outline="")
            self.create_rectangle(0, 0, max(1, fw // 3), self._height,
                                  fill=C["accent"], outline="")


class PulsingDot(tk.Label):
    """Pulsing indicator dot."""
    def __init__(self, parent, **kw):
        super().__init__(parent, text="●", font=("Segoe UI", 12),
                         bg=C["surface"], fg=C["dim"], **kw)
        self._colors = [C["accent"], C["accent2"], C["success"]]
        self._i = 0; self._pulsing = False

    def start(self):
        self._pulsing = True; self._pulse()

    def stop(self, color=None):
        self._pulsing = False
        self.configure(fg=color or C["dim"])

    def _pulse(self):
        if not self._pulsing: return
        self.configure(fg=self._colors[self._i % len(self._colors)])
        self._i += 1
        self.after(400, self._pulse)


class Card(tk.Frame):
    """Styled card with optional title bar."""
    def __init__(self, parent, title="", accent=None, **kw):
        super().__init__(parent, bg=C["surface"], **kw)
        if title:
            bar = tk.Frame(self, bg=C["surface2"])
            bar.pack(fill="x")
            col = accent or C["accent"]
            tk.Frame(bar, bg=col, width=4).pack(side="left", fill="y")
            tk.Label(bar, text=f"  {title}", font=F_HEAD,
                     bg=C["surface2"], fg=C["text"], pady=9).pack(side="left")
            tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    def body(self, padx=12, pady=10):
        f = tk.Frame(self, bg=C["surface"])
        f.pack(fill="both", expand=True, padx=padx, pady=pady)
        return f


class GlowBtn(tk.Canvas):
    """Glowing button with hover effect."""
    def __init__(self, parent, text, cmd=None, w=160, h=36,
                 color=None, **kw):
        self._c = color or C["accent"]
        super().__init__(parent, width=w, height=h,
                         bg=C["surface"], highlightthickness=0, **kw)
        self._width = w; self._height = h; self._text = text; self._cmd = cmd
        self._hov = False
        self._draw()
        self.bind("<Enter>",    lambda e: self._hover(True))
        self.bind("<Leave>",    lambda e: self._hover(False))
        self.bind("<Button-1>", lambda e: cmd() if cmd else None)

    def _hover(self, on):
        self._hov = on; self._draw()

    def _draw(self):
        self.delete("all")
        r = 8; w = self._width; h = self._height
        c = self._c if not self._hov else "#a855f7" if self._c == C["accent2"] else "#40ffff"
        for x1, y1, x2, y2, s, e in [
            (0,0,r*2,r*2,90,90),(w-r*2,0,w,r*2,0,90),
            (0,h-r*2,r*2,h,180,90),(w-r*2,h-r*2,w,h,270,90)
        ]:
            self.create_arc(x1,y1,x2,y2,start=s,extent=e,fill=c,outline=c)
        self.create_rectangle(r,0,w-r,h,fill=c,outline=c)
        self.create_rectangle(0,r,w,h-r,fill=c,outline=c)
        self.create_text(w//2,h//2,text=self._text,fill=C["bg"],
                         font=("Segoe UI",11,"bold"))

    def configure_color(self, color):
        self._c = color; self._draw()


class ImagePanel(tk.Label):
    """Label that displays a PIL image, auto-scaled to fit."""
    def __init__(self, parent, placeholder="", w=460, h=260, **kw):
        super().__init__(parent, bg=C["surface2"],
                         text=placeholder, font=F_SMALL,
                         fg=C["muted"], width=w//8, **kw)
        self._pw = w; self._ph = h
        self.configure(height=h//14)
        self._photo = None

    def show(self, pil_img: "Image.Image"):
        if not PIL_OK: return
        pil_img = pil_img.resize((self._pw, self._ph), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.configure(image=self._photo, text="")

    def clear(self, msg=""):
        self._photo = None
        self.configure(image="", text=msg)


# ── Main App ──────────────────────────────────────────────────────────────────
class DeepGuardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepGuard  ·  AI Deepfake Detector")
        self.geometry("1060x820")
        self.minsize(960, 720)
        self.configure(bg=C["bg"])

        self._file_path  = None
        self._result     = None
        self._history    = []
        self._engine     = None
        self._analyzing  = False

        self._style()
        self._build()
        self.after(100, self._boot_engine)

    # ── ttk style ─────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",    background=C["bg"],      borderwidth=0)
        s.configure("TNotebook.Tab",background=C["surface2"],
                     foreground=C["muted"], font=F_BODY, padding=[16,8])
        s.map("TNotebook.Tab",
              background=[("selected", C["surface"])],
              foreground=[("selected", C["accent"])])
        s.configure("Treeview", background=C["surface"],
                     foreground=C["text"], fieldbackground=C["surface"],
                     rowheight=26, font=F_SMALL)
        s.configure("Treeview.Heading", background=C["surface2"],
                     foreground=C["accent"], font=("Segoe UI Semibold",10), relief="flat")
        s.map("Treeview", background=[("selected", C["accent2"])])
        s.configure("TCombobox", fieldbackground=C["surface2"],
                     background=C["surface2"], foreground=C["text"],
                     selectbackground=C["accent2"])

    # ── Top header ────────────────────────────────────────────────────────
    def _build(self):
        hdr = tk.Frame(self, bg=C["surface"])
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=C["accent"], width=4).pack(side="left", fill="y")
        inf = tk.Frame(hdr, bg=C["surface"])
        inf.pack(side="left", padx=16, pady=12)
        tk.Label(inf, text="DEEPGUARD", font=F_TITLE,
                 bg=C["surface"], fg=C["accent"]).pack(anchor="w")
        tk.Label(inf, text="Multi-Modal Deepfake Detector  ·  Video · Image · Audio",
                 font=F_SMALL, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

        # Right-side status
        sr = tk.Frame(hdr, bg=C["surface"])
        sr.pack(side="right", padx=16)
        self._dot = PulsingDot(sr)
        self._dot.pack(side="right", padx=(6,0))
        self._status_lbl = tk.Label(sr, text="Loading...", font=F_SMALL,
                                     bg=C["surface"], fg=C["muted"])
        self._status_lbl.pack(side="right")

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self._nb = nb

        self._tab_detect  = tk.Frame(nb, bg=C["bg"])
        self._tab_text    = tk.Frame(nb, bg=C["bg"])
        self._tab_history = tk.Frame(nb, bg=C["bg"])
        self._tab_models  = tk.Frame(nb, bg=C["bg"])
        self._tab_help    = tk.Frame(nb, bg=C["bg"])

        nb.add(self._tab_detect,  text="  🔍  DETECT  ")
        nb.add(self._tab_text,    text="  📝  TEXT  ")
        nb.add(self._tab_history, text="  📋  HISTORY  ")
        nb.add(self._tab_models,  text="  🧠  MODELS  ")
        nb.add(self._tab_help,    text="  ❓  HELP    ")

        self._build_detect()
        self._build_text()
        self._build_history()
        self._build_models()
        self._build_help()

    # ═════════════════════════════════════════════════════════════════════
    #  DETECT TAB
    # ═════════════════════════════════════════════════════════════════════
    def _build_detect(self):
        root = self._tab_detect

        # ── Top: upload + progress (left) | verdict (right) ──────────────
        top = tk.Frame(root, bg=C["bg"])
        top.pack(fill="x", padx=12, pady=(10,6))

        # LEFT column
        left = tk.Frame(top, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0,6))

        # Upload card
        uc = Card(left, "UPLOAD  FILE")
        uc.pack(fill="x", pady=(0,8))
        ub = uc.body()

        # Drop zone
        dz = tk.Frame(ub, bg=C["surface2"], height=80, cursor="hand2")
        dz.pack(fill="x", pady=(0,8))
        dz.pack_propagate(False)
        tk.Label(dz, text="⬆  Click to browse  (Video · Image · Audio)",
                 font=F_BODY, bg=C["surface2"], fg=C["muted"]).place(
                     relx=0.5, rely=0.5, anchor="center")
        dz.bind("<Button-1>", lambda e: self._browse())

        # File name row
        frow = tk.Frame(ub, bg=C["surface"])
        frow.pack(fill="x")
        tk.Label(frow, text="FILE:", font=F_SMALL,
                 bg=C["surface"], fg=C["muted"]).pack(side="left")
        self._fn_lbl = tk.Label(frow, text="— none selected —",
                                 font=F_MONO, bg=C["surface"], fg=C["text"])
        self._fn_lbl.pack(side="left", padx=6)

        # Format chips
        chips = tk.Frame(ub, bg=C["surface"])
        chips.pack(fill="x", pady=(6,0))
        for icon, fmts in [("🎥","MP4 AVI MOV"),("🖼","JPG PNG BMP"),("🎵","WAV MP3 FLAC")]:
            c = tk.Frame(chips, bg=C["border"], padx=7, pady=3)
            c.pack(side="left", padx=3)
            tk.Label(c, text=f"{icon} {fmts}", font=F_SMALL,
                     bg=C["border"], fg=C["muted"]).pack()

        # Analyse button
        abf = tk.Frame(ub, bg=C["surface"])
        abf.pack(pady=(10,4))
        self._analyze_btn = GlowBtn(abf, "▶  ANALYZE", cmd=self._analyze,
                                     w=180, h=38)
        self._analyze_btn.pack()

        # Progress card
        pc = Card(left, "PROGRESS")
        pc.pack(fill="x", pady=(0,8))
        pb = pc.body(pady=8)
        self._pbar = AnimBar(pb, w=480)
        self._pbar.pack(fill="x")
        prow = tk.Frame(pb, bg=C["surface"])
        prow.pack(fill="x", pady=(4,0))
        self._ppct = tk.Label(prow, text="0%", font=F_MONO,
                               bg=C["surface"], fg=C["accent"])
        self._ppct.pack(side="left")
        self._pmsg = tk.Label(prow, text="Ready", font=F_SMALL,
                               bg=C["surface"], fg=C["muted"])
        self._pmsg.pack(side="right")

        # RIGHT column — verdict panel
        right = tk.Frame(top, bg=C["bg"], width=310)
        right.pack(side="right", fill="y", padx=(6,0))
        right.pack_propagate(False)

        vc = Card(right, "VERDICT")
        vc.pack(fill="x")
        vb = vc.body(pady=16)

        self._verdict_lbl = tk.Label(vb, text="—", font=F_BIG,
                                      bg=C["surface"], fg=C["muted"])
        self._verdict_lbl.pack()
        self._conf_lbl = tk.Label(vb, text="—", font=F_MED,
                                   bg=C["surface"], fg=C["muted"])
        self._conf_lbl.pack(pady=(4,0))
        self._risk_lbl = tk.Label(vb, text="", font=F_SMALL,
                                   bg=C["surface"], fg=C["muted"],
                                   padx=10, pady=4)
        self._risk_lbl.pack(pady=(6,0))
        self._expl_lbl = tk.Label(vb, text="", font=F_SMALL,
                                   bg=C["surface"], fg=C["muted"],
                                   wraplength=270, justify="center")
        self._expl_lbl.pack(pady=(8,0))

        # Metric grid
        mc = Card(right, "METRICS")
        mc.pack(fill="x", pady=(8,0))
        mb = mc.body(pady=8)
        self._metric_vars = {}
        for i, (label, key) in enumerate([
            ("Media",    "media_type"),
            ("File",     "file_name"),
            ("Time",     "inference_time"),
            ("Score",    "raw_score"),
        ]):
            r = i // 2; co = (i % 2) * 2
            tk.Label(mb, text=f"{label}:", font=F_SMALL,
                     bg=C["surface"], fg=C["muted"], width=8, anchor="w"
                     ).grid(row=r, column=co, sticky="w", pady=2)
            v = tk.StringVar(value="—")
            self._metric_vars[key] = v
            tk.Label(mb, textvariable=v, font=F_MONO,
                     bg=C["surface"], fg=C["text"], anchor="w"
                     ).grid(row=r, column=co+1, sticky="w", padx=(4,16), pady=2)

        # Export
        ef = tk.Frame(right, bg=C["bg"])
        ef.pack(pady=6)
        GlowBtn(ef, "💾  EXPORT JSON", cmd=self._export,
                w=170, h=32, color=C["accent2"]).pack()

        # ── Bottom: visual analysis panels ───────────────────────────────
        bot = tk.Frame(root, bg=C["bg"])
        bot.pack(fill="both", expand=True, padx=12, pady=(0,10))

        # Visual output notebook (sub-tabs)
        self._vis_nb = ttk.Notebook(bot)
        self._vis_nb.pack(fill="both", expand=True)

        self._vis_gradcam  = tk.Frame(self._vis_nb, bg=C["bg"])
        self._vis_frames   = tk.Frame(self._vis_nb, bg=C["bg"])
        self._vis_spec     = tk.Frame(self._vis_nb, bg=C["bg"])
        self._vis_log      = tk.Frame(self._vis_nb, bg=C["bg"])

        self._vis_nb.add(self._vis_gradcam, text="  🔥  HEATMAP  ")
        self._vis_nb.add(self._vis_frames,  text="  📊  FRAME SCORES  ")
        self._vis_nb.add(self._vis_spec,    text="  🎵  SPECTROGRAM  ")
        self._vis_nb.add(self._vis_log,     text="  📝  LOG  ")

        # Heatmap panel
        hc = Card(self._vis_gradcam, "GRAD-CAM  FACE  ANALYSIS",
                  accent=C["danger"])
        hc.pack(fill="both", expand=True, padx=6, pady=6)
        hb = hc.body()
        self._gradcam_panel = ImagePanel(hb,
            placeholder="Run analysis on an image or video to see Grad-CAM heatmap",
            w=680, h=210)
        self._gradcam_panel.pack(fill="both", expand=True)

        # Frame scores panel
        fc_card = Card(self._vis_frames, "PER-FRAME  FAKE  PROBABILITY",
                       accent=C["warning"])
        fc_card.pack(fill="both", expand=True, padx=6, pady=6)
        fb = fc_card.body()
        self._frames_panel = ImagePanel(fb,
            placeholder="Run analysis on a video to see per-frame scores",
            w=680, h=210)
        self._frames_panel.pack(fill="both", expand=True)

        # Spectrogram panel
        sc_card = Card(self._vis_spec, "MEL  SPECTROGRAM  +  SEGMENT  SCORES",
                       accent=C["accent2"])
        sc_card.pack(fill="both", expand=True, padx=6, pady=6)
        sb = sc_card.body()
        self._spec_panel = ImagePanel(sb,
            placeholder="Run analysis on an audio file to see Mel spectrogram",
            w=680, h=210)
        self._spec_panel.pack(fill="both", expand=True)

        # Log panel
        lc = Card(self._vis_log, "SYSTEM  LOG")
        lc.pack(fill="both", expand=True, padx=6, pady=6)
        lb = lc.body()
        self._log = tk.Text(lb, font=F_SMALL, bg=C["bg"],
                             fg=C["muted"], relief="flat", bd=0,
                             state="disabled", wrap="word",
                             insertbackground=C["accent"])
        ls = tk.Scrollbar(lb, command=self._log.yview,
                          bg=C["border"], troughcolor=C["bg"])
        self._log.configure(yscrollcommand=ls.set)
        ls.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True)

        # Color tags
        self._log.tag_configure("info",    foreground=C["muted"])
        self._log.tag_configure("ok",      foreground=C["success"])
        self._log.tag_configure("warn",    foreground=C["warning"])
        self._log.tag_configure("error",   foreground=C["danger"])
        self._log.tag_configure("accent",  foreground=C["accent"])


    # ═════════════════════════════════════════════════════════════════════
    #  TEXT ANALYSIS TAB
    # ═════════════════════════════════════════════════════════════════════
    def _build_text(self):
        root = self._tab_text

        # ── Left: input + controls ────────────────────────────────────────
        paned = tk.PanedWindow(root, orient="horizontal", bg=C["bg"],
                               sashwidth=6, sashrelief="flat",
                               sashpad=0, relief="flat", bd=0)
        paned.pack(fill="both", expand=True, padx=12, pady=10)

        left_f = tk.Frame(paned, bg=C["bg"])
        right_f = tk.Frame(paned, bg=C["bg"])
        paned.add(left_f, minsize=380)
        paned.add(right_f, minsize=420)

        # Input card
        ic = Card(left_f, "INPUT  TEXT", accent=C["accent"])
        ic.pack(fill="both", expand=True, pady=(0, 8))
        ib = ic.body(pady=8)

        # Mode selector
        mode_row = tk.Frame(ib, bg=C["surface"])
        mode_row.pack(fill="x", pady=(0, 8))
        tk.Label(mode_row, text="Analysis Mode:", font=F_SMALL,
                 bg=C["surface"], fg=C["muted"]).pack(side="left")
        self._text_mode = tk.StringVar(value="all")
        for val, lbl in [("all","All Three"),("ai","AI Detection"),
                          ("fake_news","Fake News"),("grammar","Grammar")]:
            tk.Radiobutton(mode_row, text=lbl, variable=self._text_mode,
                           value=val, font=F_SMALL, bg=C["surface"],
                           fg=C["text"], selectcolor=C["surface2"],
                           activebackground=C["surface"],
                           activeforeground=C["accent"]).pack(side="left", padx=6)

        # Sample buttons
        samp_row = tk.Frame(ib, bg=C["surface"])
        samp_row.pack(fill="x", pady=(0, 6))
        tk.Label(samp_row, text="Load sample:", font=F_SMALL,
                 bg=C["surface"], fg=C["muted"]).pack(side="left")
        for lbl, mode in [("AI text", "ai_sample"),
                           ("Fake news", "fake_sample"),
                           ("Bad grammar", "grammar_sample")]:
            tk.Button(samp_row, text=lbl, font=F_SMALL,
                      bg=C["surface2"], fg=C["accent"],
                      relief="flat", bd=0, padx=8, pady=2, cursor="hand2",
                      command=lambda m=mode: self._load_sample(m)
                      ).pack(side="left", padx=4)

        # Text input box
        self._text_input = tk.Text(ib, height=14, font=F_BODY,
                                    bg=C["surface2"], fg=C["text"],
                                    insertbackground=C["accent"],
                                    relief="flat", bd=0, wrap="word",
                                    padx=8, pady=8)
        ts_in = tk.Scrollbar(ib, command=self._text_input.yview,
                              bg=C["border"], troughcolor=C["bg"])
        self._text_input.configure(yscrollcommand=ts_in.set)
        ts_in.pack(side="right", fill="y")
        self._text_input.pack(fill="both", expand=True)

        # Word count label
        wc_row = tk.Frame(ib, bg=C["surface"])
        wc_row.pack(fill="x", pady=(4, 0))
        self._wc_lbl = tk.Label(wc_row, text="0 words", font=F_SMALL,
                                 bg=C["surface"], fg=C["muted"])
        self._wc_lbl.pack(side="right")
        self._text_input.bind("<KeyRelease>", self._update_wc)

        # Analyse button
        bf = tk.Frame(ic, bg=C["surface"])
        bf.pack(padx=12, pady=(4, 12))
        GlowBtn(bf, "🔍  ANALYSE TEXT", cmd=self._run_text_analysis,
                w=200, h=38).pack(side="left", padx=6)
        GlowBtn(bf, "✕  CLEAR", cmd=self._clear_text,
                w=90, h=38, color=C["muted"]).pack(side="left")

        # ── Right: results sub-notebook ──────────────────────────────────
        self._text_nb = ttk.Notebook(right_f)
        self._text_nb.pack(fill="both", expand=True)

        self._tab_ai       = tk.Frame(self._text_nb, bg=C["bg"])
        self._tab_fakenews = tk.Frame(self._text_nb, bg=C["bg"])
        self._tab_grammar  = tk.Frame(self._text_nb, bg=C["bg"])
        self._text_nb.add(self._tab_ai,       text="  🤖  AI DETECTION  ")
        self._text_nb.add(self._tab_fakenews, text="  📰  FAKE NEWS  ")
        self._text_nb.add(self._tab_grammar,  text="  ✏  GRAMMAR  ")

        self._build_ai_tab()
        self._build_fakenews_tab()
        self._build_grammar_tab()

    # ── AI Detection result tab ───────────────────────────────────────────
    def _build_ai_tab(self):
        root = self._tab_ai

        vc = Card(root, "AI-GENERATED TEXT DETECTOR", accent=C["accent2"])
        vc.pack(fill="x", padx=6, pady=(6, 4))
        vb = vc.body(pady=14)

        self._ai_verdict = tk.Label(vb, text="—", font=("Segoe UI",28,"bold"),
                                     bg=C["surface"], fg=C["muted"])
        self._ai_verdict.pack()
        self._ai_conf = tk.Label(vb, text="", font=F_MED,
                                  bg=C["surface"], fg=C["muted"])
        self._ai_conf.pack(pady=(4, 0))
        self._ai_expl = tk.Label(vb, text="Paste text and click Analyse",
                                  font=F_SMALL, bg=C["surface"], fg=C["muted"],
                                  wraplength=360, justify="center")
        self._ai_expl.pack(pady=(8, 0))

        # Score bar
        sb_c = Card(root, "AI SCORE", accent=C["accent2"])
        sb_c.pack(fill="x", padx=6, pady=(0, 4))
        sb_b = sb_c.body(pady=8)
        self._ai_bar = AnimBar(sb_b, w=380)
        self._ai_bar.pack(fill="x")
        self._ai_bar_lbl = tk.Label(sb_b, text="0%", font=F_MONO,
                                     bg=C["surface"], fg=C["accent2"])
        self._ai_bar_lbl.pack(anchor="e", pady=(3, 0))

        # Feature table
        fc = Card(root, "LINGUISTIC FEATURES", accent=C["accent2"])
        fc.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        fb = fc.body(pady=8)

        feat_cols = ("Feature", "Value", "Signal")
        self._ai_tree = ttk.Treeview(fb, columns=feat_cols,
                                      show="headings", height=7)
        for col, w in zip(feat_cols, [160, 90, 120]):
            self._ai_tree.heading(col, text=col)
            self._ai_tree.column(col, width=w, anchor="center")
        self._ai_tree.tag_configure("ai",    foreground=C["accent2"])
        self._ai_tree.tag_configure("human", foreground=C["success"])
        self._ai_tree.tag_configure("neutral",foreground=C["muted"])
        self._ai_tree.pack(fill="both", expand=True)

    # ── Fake news result tab ──────────────────────────────────────────────
    def _build_fakenews_tab(self):
        root = self._tab_fakenews

        vc = Card(root, "FAKE NEWS / MISINFORMATION DETECTOR", accent=C["danger"])
        vc.pack(fill="x", padx=6, pady=(6, 4))
        vb = vc.body(pady=14)

        self._fn_verdict = tk.Label(vb, text="—", font=("Segoe UI",28,"bold"),
                                     bg=C["surface"], fg=C["muted"])
        self._fn_verdict.pack()
        self._fn_conf = tk.Label(vb, text="", font=F_MED,
                                  bg=C["surface"], fg=C["muted"])
        self._fn_conf.pack(pady=(4, 0))
        self._fn_expl = tk.Label(vb, text="Paste text and click Analyse",
                                  font=F_SMALL, bg=C["surface"], fg=C["muted"],
                                  wraplength=360, justify="center")
        self._fn_expl.pack(pady=(8, 0))

        # Score bar
        sb_c = Card(root, "CREDIBILITY SCORE  (lower = more credible)", accent=C["danger"])
        sb_c.pack(fill="x", padx=6, pady=(0, 4))
        sb_b = sb_c.body(pady=8)
        self._fn_bar = AnimBar(sb_b, w=380)
        self._fn_bar.pack(fill="x")
        self._fn_bar_lbl = tk.Label(sb_b, text="0%", font=F_MONO,
                                     bg=C["surface"], fg=C["danger"])
        self._fn_bar_lbl.pack(anchor="e", pady=(3, 0))

        # Triggered patterns
        pc = Card(root, "DETECTED SIGNALS", accent=C["danger"])
        pc.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        pb = pc.body(pady=8)
        self._fn_signals = tk.Text(pb, height=8, font=F_SMALL,
                                    bg=C["surface2"], fg=C["text"],
                                    relief="flat", bd=0, state="disabled",
                                    wrap="word", padx=8, pady=6)
        self._fn_signals.tag_configure("hit",    foreground=C["danger"])
        self._fn_signals.tag_configure("ok",     foreground=C["success"])
        self._fn_signals.tag_configure("header", foreground=C["accent"],
                                        font=("Segoe UI Semibold", 10))
        self._fn_signals.pack(fill="both", expand=True)

    # ── Grammar correction tab ────────────────────────────────────────────
    def _build_grammar_tab(self):
        root = self._tab_grammar

        # Stats row
        sc = Card(root, "GRAMMAR CORRECTION SUMMARY", accent=C["success"])
        sc.pack(fill="x", padx=6, pady=(6, 4))
        sb = sc.body(pady=10)

        stat_row = tk.Frame(sb, bg=C["surface"])
        stat_row.pack(fill="x")
        self._gram_vars = {}
        for i, (label, key) in enumerate([
            ("Changes Made", "change_count"),
            ("Quality Before", "quality_before"),
            ("Quality After",  "quality_after"),
        ]):
            col_f = tk.Frame(stat_row, bg=C["surface2"], padx=12, pady=8)
            col_f.pack(side="left", padx=6, expand=True)
            tk.Label(col_f, text=label, font=F_SMALL,
                     bg=C["surface2"], fg=C["muted"]).pack()
            v = tk.StringVar(value="—")
            self._gram_vars[key] = v
            tk.Label(col_f, textvariable=v, font=("Segoe UI",18,"bold"),
                     bg=C["surface2"], fg=C["accent"]).pack(pady=(4,0))

        # Corrected text output
        cc = Card(root, "CORRECTED  TEXT", accent=C["success"])
        cc.pack(fill="x", padx=6, pady=(0, 4))
        cb = cc.body(pady=8)
        self._gram_out = tk.Text(cb, height=5, font=F_BODY,
                                  bg=C["surface2"], fg=C["text"],
                                  relief="flat", bd=0, state="disabled",
                                  wrap="word", padx=8, pady=6)
        self._gram_out.tag_configure("added",   background="#052e16",
                                      foreground="#4ade80")
        self._gram_out.tag_configure("removed", background="#450a0a",
                                      foreground="#f87171",
                                      overstrike=True)
        self._gram_out.pack(fill="x")

        # Changes list
        chc = Card(root, "CHANGES  APPLIED", accent=C["success"])
        chc.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        chb = chc.body(pady=8)
        chg_cols = ("#", "Rule Applied", "Original snippet", "Suggestion")
        self._chg_tree = ttk.Treeview(chb, columns=chg_cols,
                                       show="headings", height=6)
        for col, w in zip(chg_cols, [40, 180, 180, 180]):
            self._chg_tree.heading(col, text=col)
            self._chg_tree.column(col, width=w, anchor="w")
        self._chg_tree.tag_configure("change", foreground=C["success"])
        sv = ttk.Scrollbar(chb, orient="vertical", command=self._chg_tree.yview)
        self._chg_tree.configure(yscrollcommand=sv.set)
        sv.pack(side="right", fill="y")
        self._chg_tree.pack(fill="both", expand=True)

    # ═════════════════════════════════════════════════════════════════════
    #  Text Analysis Logic
    # ═════════════════════════════════════════════════════════════════════
    def _update_wc(self, event=None):
        text = self._text_input.get("1.0", "end-1c")
        wc   = len(text.split()) if text.strip() else 0
        self._wc_lbl.configure(text=f"{wc} word{'s' if wc!=1 else ''}")

    def _clear_text(self):
        self._text_input.delete("1.0", "end")
        self._wc_lbl.configure(text="0 words")

    def _load_sample(self, kind: str):
        samples = {
            "ai_sample": (
                "In today's rapidly evolving landscape, it is important to note that "
                "artificial intelligence represents a multifaceted and comprehensive "
                "paradigm shift. Furthermore, it is worth noting that these transformative "
                "technologies leverage nuanced approaches to foster robust solutions. "
                "Nevertheless, it is crucial to ensure that stakeholders navigate these "
                "challenges in a pivotal and collaborative manner. To summarize, the "
                "implications of this domain are both essential and paramount to our "
                "collective understanding going forward."
            ),
            "fake_sample": (
                "BREAKING: Scientists BAFFLED as Government EXPOSED for HIDING "
                "the TRUTH about miracle cure!!! They don't want you to know what "
                "BIG PHARMA is hiding!!! Share BEFORE they delete this!!! "
                "100% PROVEN cure that doctors HATE — number 7 will SHOCK you. "
                "Wake up SHEEPLE! The regime is DESTROYING our future!!!"
            ),
            "grammar_sample": (
                "i went to the store yesterday and buyed a apple. "
                "Their was a lot of people their , and the cashier say "
                "that they recieve new stock tommorow. "
                "Me and my friend seen a intresting movie last night , "
                "it was more better then the first one. "
                "We should of went earlier because it dont start till late."
            ),
        }
        text = samples.get(kind, "")
        self._text_input.delete("1.0", "end")
        self._text_input.insert("1.0", text)
        self._update_wc()
        # Auto-select matching mode
        mode_map = {"ai_sample":"ai","fake_sample":"fake_news","grammar_sample":"grammar"}
        self._text_mode.set(mode_map.get(kind, "all"))

    def _run_text_analysis(self):
        text = self._text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty", "Please enter some text to analyse.")
            return

        mode = self._text_mode.get()

        def run():
            try:
                result = analyse_text(text, mode)
                self.after(0, lambda: self._display_text_results(result, mode))
            except Exception as e:
                err = str(e)
                self.after(0, lambda err=err: messagebox.showerror(
                    "Analysis Error", f"Text analysis failed:\n{err}"))
        threading.Thread(target=run, daemon=True).start()

    def _display_text_results(self, result: dict, mode: str):
        # ── AI Detection ──────────────────────────────────────────────
        if "ai_detection" in result:
            ai = result["ai_detection"]
            label = ai["label"]
            conf  = ai["confidence"]
            score = ai["score"]

            color_map = {
                "AI-GENERATED": C["danger"],
                "LIKELY AI":    C["warning"],
                "UNCERTAIN":    C["muted"],
                "HUMAN":        C["success"],
            }
            lc = color_map.get(label, C["muted"])

            self._ai_verdict.configure(text=label, fg=lc)
            self._ai_conf.configure(text=f"{conf:.1f}% confidence  ·  score {score:.1f}/100",
                                     fg=lc)
            self._ai_expl.configure(text=ai["explanation"], fg=C["muted"])
            self._ai_bar.set(score)
            self._ai_bar_lbl.configure(text=f"{score:.1f}%")

            # Feature table
            for row in self._ai_tree.get_children():
                self._ai_tree.delete(row)

            feat = ai.get("features", {})
            FEATURE_INFO = [
                ("Burstiness",          "burstiness",
                 lambda v: ("↓ AI-like" if v < 0 else "↑ Human-like", "ai" if v < 0 else "human")),
                ("Lexical Diversity",   "lexical_diversity",
                 lambda v: ("Mid AI range" if 0.44<=v<=0.62 else "Human range",
                             "ai" if 0.44<=v<=0.62 else "human")),
                ("AI Vocab Ratio",      "ai_vocab_ratio",
                 lambda v: ("High — AI marker" if v>0.05 else "Low — OK",
                             "ai" if v>0.05 else "human")),
                ("Contraction Use",     "contraction_ratio",
                 lambda v: ("Low — AI marker" if v<0.01 else "Present — human",
                             "ai" if v<0.01 else "human")),
                ("Punct. Variety",      "punct_variety",
                 lambda v: ("Low — AI-like" if v<0.05 else "Good",
                             "ai" if v<0.05 else "human")),
                ("Avg Sentence Length", "avg_sentence_len",
                 lambda v: ("AI range (18–26)" if 17<=v<=27 else "Human range",
                             "ai" if 17<=v<=27 else "human")),
                ("Sentence Count",      "sentence_count",
                 lambda v: (str(int(v)), "neutral")),
                ("Word Count",          "word_count",
                 lambda v: (str(int(v)), "neutral")),
            ]
            for fname, fkey, interp in FEATURE_INFO:
                val = feat.get(fkey, 0)
                signal, tag = interp(val)
                self._ai_tree.insert("", "end",
                                     values=(fname, f"{val}", signal),
                                     tags=(tag,))

            # Switch to AI tab
            self._text_nb.select(0)

        # ── Fake News ─────────────────────────────────────────────────
        if "fake_news" in result:
            fn    = result["fake_news"]
            label = fn["label"]
            conf  = fn["confidence"]
            score = fn["score"]

            fn_colors = {
                "LIKELY FAKE":    C["danger"],
                "SUSPICIOUS":     C["warning"],
                "MOSTLY CREDIBLE":C["success"],
                "CREDIBLE":       C["success"],
                "UNCERTAIN":      C["muted"],
            }
            fnc = fn_colors.get(label, C["muted"])

            self._fn_verdict.configure(text=label, fg=fnc)
            self._fn_conf.configure(text=f"{conf:.1f}% confidence  ·  score {score:.1f}/100",
                                     fg=fnc)
            self._fn_expl.configure(text=fn["explanation"], fg=C["muted"])
            self._fn_bar.set(score)
            self._fn_bar_lbl.configure(text=f"{score:.1f}%")

            # Signals text
            self._fn_signals.configure(state="normal")
            self._fn_signals.delete("1.0", "end")

            triggered = fn.get("triggered_patterns", [])
            sigs      = fn.get("signals", {})

            self._fn_signals.insert("end", "TRIGGERED PATTERNS\n", "header")
            if triggered:
                for pat in triggered:
                    self._fn_signals.insert("end", f"  \u26a0  {pat}\n", "hit")
            else:
                self._fn_signals.insert("end", "  \u2713  No suspicious patterns detected\n", "ok")

            self._fn_signals.insert("end", "\nSIGNAL COUNTS\n", "header")
            for k, v in sigs.items():
                tag = "hit" if (isinstance(v, (int,float)) and v > 0
                                and k != "credibility_hits" and k != "word_count") else "ok"
                self._fn_signals.insert("end", f"  {k:<22}: {v}\n", tag)

            self._fn_signals.configure(state="disabled")

            if "fake_news" in result and "ai_detection" not in result:
                self._text_nb.select(1)

        # ── Grammar ───────────────────────────────────────────────────
        if "grammar" in result:
            gr = result["grammar"]
            n  = gr["change_count"]
            qb = gr["quality_before"]
            qa = gr["quality_after"]

            self._gram_vars["change_count"].set(str(n))
            self._gram_vars["quality_before"].set(f"{qb:.0f}/100")
            self._gram_vars["quality_after"].set(f"{qa:.0f}/100")

            # Corrected text with diff highlighting
            self._gram_out.configure(state="normal")
            self._gram_out.delete("1.0", "end")

            diff = gr.get("diff", [])
            if diff:
                for token in diff:
                    t    = token["text"]
                    typ  = token["type"]
                    tag  = {"same": "", "removed": "removed", "added": "added"}.get(typ, "")
                    self._gram_out.insert("end", t + " ", tag)
            else:
                self._gram_out.insert("end", gr["corrected"])

            self._gram_out.configure(state="disabled")

            # Changes list
            for row in self._chg_tree.get_children():
                self._chg_tree.delete(row)
            for i, change in enumerate(gr["changes"], 1):
                self._chg_tree.insert("", "end",
                                      values=(i, change["rule"], change.get("before","")[:40], change.get("after","")[:40]),
                                      tags=("change",))

            if n == 0:
                self._chg_tree.insert("", "end",
                                      values=("✓", "No errors found", "Text looks good!", ""),
                                      tags=("change",))

            if "grammar" in result and "ai_detection" not in result and "fake_news" not in result:
                self._text_nb.select(2)

    # ═════════════════════════════════════════════════════════════════════
    #  HISTORY TAB
    # ═════════════════════════════════════════════════════════════════════
    def _build_history(self):
        hc = Card(self._tab_history, "DETECTION  HISTORY")
        hc.pack(fill="both", expand=True, padx=12, pady=12)
        hb = hc.body()

        cols = ("Timestamp","File","Type","Verdict","Confidence","Risk","Score")
        self._tree = ttk.Treeview(hb, columns=cols, show="headings")
        widths = [145,180,70,80,95,75,80]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")
        self._tree.tag_configure("fake", foreground=C["danger"])
        self._tree.tag_configure("real", foreground=C["success"])

        vsb = ttk.Scrollbar(hb, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        bf = tk.Frame(hc, bg=C["surface"])
        bf.pack(padx=12, pady=(0,10))
        GlowBtn(bf, "🗑  CLEAR",   cmd=self._clear_history, w=120, h=30,
                color=C["danger"]).pack(side="left", padx=4)
        GlowBtn(bf, "💾  EXPORT CSV", cmd=self._export_csv, w=140, h=30,
                color=C["accent2"]).pack(side="left", padx=4)

    # ═════════════════════════════════════════════════════════════════════
    #  MODELS TAB
    # ═════════════════════════════════════════════════════════════════════
    def _build_models(self):
        canvas = tk.Canvas(self._tab_models, bg=C["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(self._tab_models, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True, padx=12, pady=12)

        models = [
            ("🎥  VIDEO  DETECTOR", C["warning"], [
                ("Backbone",     "EfficientNet-B7  (ImageNet pretrained via timm)"),
                ("Temporal",     "Bidirectional LSTM  ·  attention mechanism"),
                ("Preprocessing","Eulerian Video Magnification (EVM)"),
                ("Input",        "20 frames  ·  380×380 px  ·  MP4/AVI/MOV"),
                ("Target Acc.",  "97.76%  (FaceForensics++)"),
                ("Fine-tune",    "Place video_head.pt in checkpoints/pretrained/"),
            ]),
            ("🖼  IMAGE  DETECTOR", C["danger"], [
                ("Backbone",     "EfficientNet-B7  (ImageNet pretrained via timm)"),
                ("Explainability","Grad-CAM heatmap on last conv block"),
                ("Face Detect",  "OpenCV Haar Cascade + 25% padding"),
                ("Input",        "Any resolution  ·  380×380 crop  ·  JPG/PNG"),
                ("Target Acc.",  "93.64%  (FaceForensics++ / 140k faces)"),
                ("Fine-tune",    "Place image_head.pt in checkpoints/pretrained/"),
            ]),
            ("🎵  AUDIO  DETECTOR", C["accent2"], [
                ("Architecture", "Light CNN (LCNN)  ·  Max-Feature-Map activations"),
                ("Features",     "128-band log-Mel spectrogram  ·  16kHz  ·  4s chunks"),
                ("Aggregation",  "Weighted max-pool over overlapping segments"),
                ("Input",        "WAV / MP3 / FLAC  ·  any duration"),
                ("Target Acc.",  "99.13%  (ASVspoof2019 LA track)"),
                ("Fine-tune",    "Place audio_lcnn.pt in checkpoints/pretrained/"),
            ]),
            ("📦  USING  FINE-TUNED  WEIGHTS", C["accent"], [
                ("Step 1",  "Download weights from HuggingFace (see Help tab)"),
                ("Step 2",  "Rename to video_head.pt / image_head.pt / audio_lcnn.pt"),
                ("Step 3",  "Place in:  deepguard/checkpoints/pretrained/"),
                ("Step 4",  "Click 'Reload Models' below — no restart needed"),
                ("Note",    "Without fine-tuned weights, EfficientNet-B7 ImageNet"),
                ("",        "features still provide useful deepfake signal"),
            ]),
        ]

        for title, accent, rows in models:
            c = Card(inner, title, accent=accent)
            c.pack(fill="x", pady=(0,8))
            g = c.body(pady=10)
            for k, v in rows:
                r = tk.Frame(g, bg=C["surface"])
                r.pack(fill="x", pady=1)
                tk.Label(r, text=f"{k}:", font=F_SMALL, bg=C["surface"],
                         fg=C["muted"], width=14, anchor="w").pack(side="left")
                tk.Label(r, text=v, font=F_MONO, bg=C["surface"],
                         fg=C["text"], anchor="w").pack(side="left", padx=6)

        # Reload button
        rf = tk.Frame(inner, bg=C["bg"])
        rf.pack(pady=8)
        GlowBtn(rf, "🔄  RELOAD  MODELS", cmd=self._reload_models,
                w=200, h=36).pack()

    # ═════════════════════════════════════════════════════════════════════
    #  HELP TAB
    # ═════════════════════════════════════════════════════════════════════
    def _build_help(self):
        canvas = tk.Canvas(self._tab_help, bg=C["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(self._tab_help, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True, padx=12, pady=12)

        sections = [
            ("QUICK  START", C["accent"], [
                ("1. Install",
                 "pip install torch torchvision timm opencv-python pillow librosa soundfile matplotlib"),
                ("2. Launch",   "python main.py"),
                ("3. Detect",   "Click 'Upload File' → select media → 'Analyze'"),
                ("4. Results",  "Verdict + confidence + heatmap / frame chart / spectrogram"),
            ]),
            ("PRETRAINED  WEIGHTS  (RECOMMENDED)", C["warning"], [
                ("Image/Video",
                 "github.com/ondyari/FaceForensics  (Xception / EfficientNet checkpoints)"),
                ("Audio",
                 "github.com/asvspoof-challenge/2019  (RawNet2 / LCNN weights)"),
                ("HuggingFace",
                 "huggingface.co/spaces/dima806/deepfake_vs_real_image_detection"),
                ("After download",
                 "Rename → place in checkpoints/pretrained/ → Reload Models"),
            ]),
            ("READING  RESULTS", C["success"], [
                ("FAKE  HIGH",   "Strong AI-generated indicators. Treat as deepfake."),
                ("FAKE  MEDIUM", "Multiple artifacts found. Likely fake; manual review advised."),
                ("FAKE  LOW",    "Weak signal. Could be real with post-processing applied."),
                ("REAL  HIGH",   "No deepfake markers. Highly likely to be authentic."),
                ("REAL  MEDIUM", "Mostly clean. Minor anomalies — could be compression."),
                ("HEATMAP",      "Red/hot regions = areas the model flagged as suspicious."),
                ("FRAME  CHART", "Bars above 0.5 line = frames classified as fake."),
                ("SPECTROGRAM",  "Red bars = audio segments with synthetic speech artifacts."),
            ]),
            ("LIMITATIONS", C["danger"], [
                ("No fine-tune",  "Without deepfake-specific weights, accuracy is lower."),
                ("Heavy makeup",  "Unusual makeup can cause false positives."),
                ("Compression",   "Heavy JPEG/video compression may reduce accuracy."),
                ("Resolution",    "Very low-resolution inputs reduce reliability."),
                ("Novel GANs",    "Brand-new generation methods may evade detection."),
            ]),
            ("SYSTEM  REQUIREMENTS", C["muted"], [
                ("Python",    "3.8 or higher"),
                ("PyTorch",   "pip install torch torchvision   (CUDA optional but faster)"),
                ("timm",      "pip install timm   (EfficientNet backbone)"),
                ("Audio",     "pip install librosa soundfile"),
                ("Vision",    "pip install opencv-python pillow matplotlib"),
                ("RAM",       "8 GB minimum   ·   16 GB recommended"),
                ("GPU",       "Optional — CUDA/MPS auto-detected for 10–30× speedup"),
            ]),
        ]

        for title, accent, rows in sections:
            c = Card(inner, title, accent=accent)
            c.pack(fill="x", pady=(0,8))
            g = c.body(pady=10)
            for k, v in rows:
                r = tk.Frame(g, bg=C["surface"])
                r.pack(fill="x", pady=2)
                if k:
                    tk.Label(r, text=f"{k}:", font=("Courier New",9,"bold"),
                             bg=C["surface"], fg=accent,
                             width=14, anchor="w").pack(side="left")
                else:
                    tk.Label(r, text=" "*16, bg=C["surface"]).pack(side="left")
                tk.Label(r, text=v, font=F_SMALL, bg=C["surface"],
                         fg=C["text"], anchor="w", wraplength=680,
                         justify="left").pack(side="left", padx=4)

    # ═════════════════════════════════════════════════════════════════════
    #  Logic
    # ═════════════════════════════════════════════════════════════════════
    def _boot_engine(self):
        def load():
            try:
                self._log_msg("Initialising DeepGuard engine...", "accent")
                self._engine = DeepGuardEngine()
                
                self.after(0, lambda: (
                    self._set_status("Ready", True),
                    self._log_msg("Engine ready. Select a file and click Analyze.", "ok")
                ))
            except Exception as e:
                self.after(0, lambda: (
                    self._set_status("Error", False),
                    self._log_msg(f"Engine error: {e}", "error")
                ))
        threading.Thread(target=load, daemon=True).start()

    def _set_status(self, msg, ok):
        if ok:
            self._dot.stop(C["success"])
        else:
            self._dot.stop(C["danger"])
        self._status_lbl.configure(text=msg)

    def _log_msg(self, msg, tag="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        def _write():
            self._log.configure(state="normal")
            self._log.insert("end", line, tag)
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _write)

    def _set_progress(self, val, msg):
        def _u():
            self._pbar.set(val)
            self._ppct.configure(text=f"{val}%")
            self._pmsg.configure(text=msg)
        self.after(0, _u)

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[
            ("All media", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png "
                          "*.bmp *.webp *.wav *.mp3 *.flac *.ogg"),
            ("Video",  "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Image",  "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("Audio",  "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("All",    "*.*"),
        ])
        if path:
            self._file_path = path
            name = Path(path).name
            self._fn_lbl.configure(text=name, fg=C["accent"])
            self._log_msg(f"Selected: {name}", "accent")
            # Clear old visuals
            self._gradcam_panel.clear("Run analysis to see heatmap")
            self._frames_panel.clear("Run analysis to see frame scores")
            self._spec_panel.clear("Run analysis to see spectrogram")

    def _analyze(self):
        if self._analyzing:
            return
        if not self._file_path:
            messagebox.showwarning("No File", "Please select a media file first.")
            return
        if self._engine is None:
            messagebox.showerror("Not Ready", "Engine still loading. Please wait a moment.")
            return

        self._analyzing = True
        self._dot.start()
        self._analyze_btn.configure_color(C["muted"])
        self._set_progress(5, "Loading specific AI model (This may take up to 10s)...")
        self._log_msg(f"Analysing: {Path(self._file_path).name}", "accent")

        def run():
            try:
                result = self._engine.predict(
                    self._file_path,
                    progress_cb=lambda v: self._set_progress(max(10, v), f"Processing... {v}%")
                )
                self.after(0, lambda: self._show_result(result))
            except Exception as e:
                self._log_msg(f"Analysis failed: {e}", "error")
                self.after(0, lambda: self._set_progress(0, "Error"))
            finally:
                self._analyzing = False
                self.after(0, lambda: (
                    self._dot.stop(C["success"]),
                    self._analyze_btn.configure_color(C["accent"])
                ))

        threading.Thread(target=run, daemon=True).start()

    def _show_result(self, r: dict):
        if r.get("label") == "ERROR":
            self._log_msg(f"Error: {r.get('error','Unknown')}", "error")
            return

        label = r["label"]
        conf  = r["confidence"]
        risk  = r["risk_level"]
        expl  = r.get("explanation", "")
        mtype = r.get("media_type","—").upper()
        fname = r.get("file_name","—")
        t     = r.get("inference_time", 0)
        raw   = r.get("raw_score", 0)

        # Verdict colors
        lc = C["danger"] if label == "FAKE" else C["success"]
        rc = {
            "HIGH":   C["danger"] if label == "FAKE" else C["success"],
            "MEDIUM": C["warning"],
            "LOW":    C["muted"],
        }.get(risk, C["muted"])

        self._verdict_lbl.configure(
            text="⚠  DEEPFAKE" if label == "FAKE" else "✓  AUTHENTIC",
            fg=lc
        )
        self._conf_lbl.configure(text=f"{conf:.1f}%  confidence", fg=lc)
        self._risk_lbl.configure(
            text=f"  RISK: {risk}  ", bg=rc,
            fg=C["bg"] if risk in ("HIGH","MEDIUM") else C["text"]
        )
        self._expl_lbl.configure(text=expl, fg=C["muted"])

        fn_short = fname[:24] + "…" if len(fname) > 24 else fname
        self._metric_vars["media_type"].set(mtype)
        self._metric_vars["file_name"].set(fn_short)
        self._metric_vars["inference_time"].set(f"{t:.2f}s")
        self._metric_vars["raw_score"].set(f"{raw:.4f}")

        self._set_progress(100, "Complete")
        self._log_msg(
            f"Result: {'⚠ DEEPFAKE' if label=='FAKE' else '✓ AUTHENTIC'}  "
            f"| Confidence: {conf:.1f}%  | Risk: {risk}  | Time: {t:.2f}s",
            "warn" if label == "FAKE" else "ok"
        )

        # ── Visual panels ──────────────────────────────────────────────
        self._render_visuals(r)

        # History
        self._add_history(r)
        self._result = r

    def _render_visuals(self, r: dict):
        """Render all visual panels in a background thread."""
        def render():
            try:
                from utils.visualize import (
                    render_gradcam, render_frame_scores,
                    render_spectrogram, MPL_OK
                )
                if not MPL_OK:
                    self._log_msg("matplotlib not installed — visuals unavailable", "warn")
                    return

                mtype = r.get("media_type", "")
                label = r["label"]
                conf  = r["confidence"]

                if mtype == "image":
                    cam  = r.get("gradcam")
                    crop = r.get("face_crop")
                    if cam is not None and crop is not None:
                        img = render_gradcam(crop, cam, label, conf)
                        self.after(0, lambda i=img: (
                            self._gradcam_panel.show(i),
                            self._vis_nb.select(0)    # switch to heatmap tab
                        ))
                    self._frames_panel.clear("Frame scores: N/A for images")
                    self._spec_panel.clear("Spectrogram: N/A for images")

                elif mtype == "video":
                    fsc = r.get("frame_scores")
                    if fsc:
                        img = render_frame_scores(fsc, label, conf)
                        self.after(0, lambda i=img: (
                            self._frames_panel.show(i),
                            self._vis_nb.select(1)    # frame scores tab
                        ))
                    self._gradcam_panel.clear("Grad-CAM: run on first key frame")
                    self._spec_panel.clear("Spectrogram: N/A for video")

                elif mtype == "audio":
                    mel  = r.get("full_mel")
                    ssc  = r.get("segment_scores", [])
                    spos = r.get("segment_positions", [])
                    dur  = r.get("duration", 0)
                    if mel is not None:
                        img = render_spectrogram(mel, ssc, spos, label, conf, dur)
                        self.after(0, lambda i=img: (
                            self._spec_panel.show(i),
                            self._vis_nb.select(2)    # spectrogram tab
                        ))
                    self._gradcam_panel.clear("Grad-CAM: N/A for audio")
                    self._frames_panel.clear("Frame scores: N/A for audio")

            except Exception as e:
                self._log_msg(f"Visualization error: {e}", "warn")

        threading.Thread(target=render, daemon=True).start()

    def _add_history(self, r: dict):
        e = {
            "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file":       r.get("file_name","—"),
            "type":       r.get("media_type","—").upper(),
            "verdict":    r["label"],
            "confidence": f"{r['confidence']:.1f}%",
            "risk":       r.get("risk_level","—"),
            "score":      f"{r.get('raw_score',0):.4f}",
        }
        self._history.append(e)
        tag = "fake" if e["verdict"] == "FAKE" else "real"
        self.after(0, lambda: self._tree.insert(
            "", 0,
            values=(e["time"],e["file"],e["type"],
                    e["verdict"],e["confidence"],e["risk"],e["score"]),
            tags=(tag,)
        ))

    def _export(self):
        if not self._result:
            messagebox.showinfo("No Result", "Run detection first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON","*.json")],
            initialfile=f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if path:
            SKIP = {"gradcam", "face_crop", "full_mel",
                    "analysis_report", "quick_summary", "frame_report",
                    "forensic_report", "scene_report"}
            meta = {}
            for k, v in self._result.items():
                if k in SKIP or v is None:
                    continue
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                except Exception:
                    pass
                meta[k] = v

            export_data = {
                "deepguard_version": "3.0",
                "analysis_metadata": meta,
                "quick_summary":     self._result.get("quick_summary", {}),
                "analysis_report":   self._result.get("analysis_report", {}),
                "frame_report":      self._result.get("frame_report"),   # list for video, null otherwise
            }
            with open(path, "w") as f:
                json.dump(export_data, f, indent=2)
            self._log_msg(f"Exported full report: {Path(path).name}", "ok")
            messagebox.showinfo("Saved", f"Full DeepGuard report saved to:\n{path}")

    def _export_csv(self):
        if not self._history:
            messagebox.showinfo("Empty", "No history to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialfile=f"deepguard_history_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        if path:
            import csv
            keys = ["time","file","type","verdict","confidence","risk","score"]
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._history)
            messagebox.showinfo("Saved", f"CSV saved to:\n{path}")

    def _clear_history(self):
        if messagebox.askyesno("Clear", "Clear all history?"):
            for i in self._tree.get_children():
                self._tree.delete(i)
            self._history.clear()

    def _reload_models(self):
        if self._engine:
            for m in ("video","image","audio"):
                self._engine.reload_model(m)
            self._log_msg("Models reloaded from checkpoints/pretrained/", "ok")
            messagebox.showinfo("Reloaded", "Models reloaded successfully.")
