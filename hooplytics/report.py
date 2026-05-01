"""PDF report builder for Hooplytics.

Renders a printable, brand-styled analytics report covering the active
roster: model quality, edge board, per-player projections, and (optionally)
AI-written prose context. The builder is pure: it returns ``bytes`` so it
has no Streamlit dependencies and is easy to test or invoke from a CLI.

Design goals:

* Energetic but accurate — every number is sourced from the bundle / edge
  frames the caller passes in. AI prose is treated as flavor only.
* Print-ready typography with a dense, info-rich layout.
* Multiple chart types for visual variety: lollipop, diverging bar,
  edge distribution histogram, per-player projection-vs-line bars.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect
from reportlab.graphics.shapes import String as _RLString
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ── Font registration ──────────────────────────────────────────────────────
# Standard PDF Type-1 fonts (Helvetica) only cover WinAnsi 8-bit chars, which
# turns characters like >, ć, and certain dashes into "tofu" boxes. We try a
# few common system TrueType fonts that ship with broad Unicode coverage so
# names like "Nurkić", arrows, and curly punctuation render cleanly. If none
# is available we silently fall back to Helvetica + an ASCII substitution
# pass in :func:`_safe_text`.
_TTF_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # macOS — Helvetica is .ttc and reportlab can't load it directly. Prefer
    # Arial Unicode for broad glyph coverage, then fall back to Arial / DejaVu.
    ("HoopArial", "Arial Unicode.ttf", "Arial Unicode.ttf"),
    ("HoopArial", "Arial.ttf", "Arial Bold.ttf"),
    ("HoopArial", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    # Linux — DejaVu is the default on Debian/Ubuntu.
    ("HoopArial", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    # Windows
    ("HoopArial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
)
_FONT_SEARCH_DIRS = (
    "/Library/Fonts",
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
)


def _resolve_font_path(name: str) -> str | None:
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for d in _FONT_SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _register_unicode_font() -> tuple[str, str]:
    """Register a Unicode TrueType font; return (regular, bold) face names."""
    for family, reg_name, bold_name in _TTF_CANDIDATES:
        reg_path = _resolve_font_path(reg_name)
        bold_path = _resolve_font_path(bold_name)
        if not reg_path:
            continue
        try:
            pdfmetrics.registerFont(TTFont(family, reg_path))
            bold_face = family
            if bold_path and bold_path != reg_path:
                bold_face = f"{family}-Bold"
                pdfmetrics.registerFont(TTFont(bold_face, bold_path))
            return family, bold_face
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


_BODY_FONT, _BOLD_FONT = _register_unicode_font()


# ── Unicode safety net ─────────────────────────────────────────────────────
# Some macOS / Linux fonts render specific Unicode glyphs as tofu boxes even
# though the TTF claims to support them. To guarantee a beautiful report on
# every machine we strip the most fragile characters down to ASCII at render
# time, regardless of which font registered.
_ASCII_SUBSTITUTIONS: dict[str, str] = {
    "\u2014": "-",     # — em dash
    "\u2013": "-",     # – en dash
    "\u2192": ">",     # → rightwards arrow
    "\u2190": "<",     # ← leftwards arrow
    "\u2191": "^",     # ↑ upwards arrow
    "\u2193": "v",     # ↓ downwards arrow
    "\u201c": '"',     # “ left double quote
    "\u201d": '"',     # ” right double quote
    "\u2018": "'",     # ‘ left single quote
    "\u2019": "'",     # ’ right single quote
    "\u2026": "...",   # … horizontal ellipsis
    "\u2022": "-",     # • bullet
    "\u2502": "|",     # │ box-drawings light vertical
    "\u2500": "-",     # ─ box-drawings light horizontal
    "\u203a": ">",     # › single right angle quote
    "\u2039": "<",     # ‹ single left angle quote
    "\u00b7": "|",     # · middle dot (renders as tofu in some Arial builds)
    "\u2264": "<=",    # ≤ less-than-or-equal
    "\u25a0": "",      # ■ black square (defensive)
    "\u25aa": "",      # ▪ black small square
    # Latin Extended-A diacritics — Arial WinAnsi lacks these.
    "\u0107": "c", "\u0106": "C",   # ć Ć
    "\u010d": "c", "\u010c": "C",   # č Č
    "\u0161": "s", "\u0160": "S",   # š Š
    "\u017e": "z", "\u017d": "Z",   # ž Ž
    "\u0111": "d", "\u0110": "D",   # đ Đ
    "\u00f1": "n", "\u00d1": "N",   # ñ Ñ
}


def _safe_text(s: Any) -> str:
    """Return ``s`` with fragile glyphs swapped to ASCII so the PDF never
    shows tofu/■ boxes regardless of which font ReportLab loaded."""
    text = str(s if s is not None else "")
    for src, dst in _ASCII_SUBSTITUTIONS.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def String(x, y, text="", **kwargs):
    """Drop-in replacement for ``reportlab.graphics.shapes.String`` that runs
    label text through :func:`_safe_text` so chart glyphs match the body."""
    return _RLString(x, y, _safe_text(text), **kwargs)


# ── Brand palette (light/printable variant of the Streamlit theme) ──────────
BRAND_ORANGE = colors.HexColor("#ff7a18")
BRAND_ORANGE_DEEP = colors.HexColor("#cc5a00")
BRAND_ORANGE_SOFT = colors.HexColor("#fff1e6")
INK_DARK = colors.HexColor("#11151c")
INK_BODY = colors.HexColor("#2b3340")
INK_MUTED = colors.HexColor("#6b7686")
INK_FAINT = colors.HexColor("#9aa3b2")
PANEL_BG = colors.HexColor("#f7f8fb")
PANEL_BORDER = colors.HexColor("#e1e4ea")
POS_GREEN = colors.HexColor("#1f9d6c")
NEG_RED = colors.HexColor("#d24545")
NEUTRAL_BLUE = colors.HexColor("#3a6ea5")
GOLD = colors.HexColor("#d4a017")
WHITE = colors.white


# ── Sportsbook abbreviations for compact rendering in tables/panels. ────────
_BOOK_ABBREV: dict[str, str] = {
    "draftkings": "DK",
    "fanduel": "FD",
    "betmgm": "MGM",
    "caesars": "CZR",
    "williamhill_us": "CZR",
    "betrivers": "BR",
    "espnbet": "ESPN",
    "espn bet": "ESPN",
    "hardrockbet": "HR",
    "hard rock bet": "HR",
    "fanatics": "FAN",
    "bovada": "BOV",
    "pinnacle": "PIN",
}


def _abbrev_books(book_names: Any, *, max_show: int | None = None) -> str:
    """Render a list/string of book titles as a compact comma-joined ticker.

    When ``max_show`` is set, anything past that many distinct books is
    collapsed to a "+N more" tail so a long list (e.g. 8 books) doesn't
    overflow tight table columns in the report.
    """
    if book_names is None:
        return "—"
    # Pandas float NaN sneaks through as a non-None scalar in DataFrame iteration.
    try:
        if not isinstance(book_names, (list, tuple, set)) and pd.isna(book_names):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(book_names, (list, tuple, set)):
        names = [str(n) for n in book_names]
    else:
        s = str(book_names).strip()
        if not s:
            return "—"
        names = [n.strip() for n in s.split(",") if n.strip()]
    if not names:
        return "—"
    seen: list[str] = []
    for n in names:
        key = n.lower()
        abbr = _BOOK_ABBREV.get(key)
        if abbr is None:
            # Fallback: take initials or first 3 letters.
            parts = [p for p in n.replace(".", " ").split() if p]
            if len(parts) >= 2:
                abbr = "".join(p[0].upper() for p in parts[:3])
            else:
                abbr = n[:3].upper()
        if abbr not in seen:
            seen.append(abbr)
    if max_show is not None and len(seen) > max_show:
        head = ", ".join(seen[:max_show])
        return f"{head}, +{len(seen) - max_show} more"
    return ", ".join(seen)


# ── Bookmark / TOC plumbing ─────────────────────────────────────────────────
class _AnchorFlowable(Flowable):
    """Zero-height flowable that registers a clickable anchor + outline entry.

    Each section header drops one of these so the cover-page TOC links can
    jump directly to that page, and the PDF outline pane mirrors the same
    structure for sidebar navigation.
    """

    def __init__(self, key: str, label: str, *, level: int = 0) -> None:
        super().__init__()
        self.key = key
        self.label = label
        self.level = level
        self.width = 0
        self.height = 0

    def draw(self) -> None:  # pragma: no cover - rendering side effect
        c = self.canv
        c.bookmarkPage(self.key)
        try:
            c.addOutlineEntry(self.label, self.key, self.level, closed=False)
        except Exception:
            # Outline entry duplicates would raise; we already bookmarked the
            # page so navigation still works.
            pass


class _ActivePlayerMarker(Flowable):
    """Zero-height marker that stamps the active player on the canvas.

    Each per-player block drops one of these as its first flowable. The page
    chrome callback reads ``canvas._hl_active_player`` to render the player's
    name in the header band, so a reader on page 7 always knows whose
    analytics they are looking at without having to scroll back.

    Flowables draw in order, so within a single page the LAST marker drawn
    wins — which is the correct behaviour when a page contains the tail of
    one player's profile and the start of another (the reader sees the
    incoming player's name, matching where the bulk of the page belongs).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.width = 0
        self.height = 0

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        return (0, 0)

    def draw(self) -> None:  # pragma: no cover - rendering side effect
        try:
            self.canv._hl_active_player = self.name
        except Exception:
            pass


def _short(text: Any, limit: int = 18) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[: limit - 1] + "..."


# ── Styles ──────────────────────────────────────────────────────────────────
def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body_font = _BODY_FONT
    bold_font = _BOLD_FONT

    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base, fontName=bold_font, fontSize=46,
            leading=52, textColor=INK_DARK, alignment=TA_LEFT,
        ),
        "cover_eyebrow": ParagraphStyle(
            "cover_eyebrow", parent=base, fontName=bold_font, fontSize=10,
            leading=14, textColor=BRAND_ORANGE, alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base, fontName=body_font, fontSize=13,
            leading=19, textColor=INK_MUTED, alignment=TA_LEFT, spaceBefore=16,
        ),
        "cover_meta_label": ParagraphStyle(
            "cover_meta_label", parent=base, fontName=bold_font, fontSize=8,
            leading=10, textColor=BRAND_ORANGE_DEEP, alignment=TA_LEFT,
        ),
        "cover_meta_value": ParagraphStyle(
            "cover_meta_value", parent=base, fontName=bold_font, fontSize=18,
            leading=22, textColor=INK_DARK, alignment=TA_LEFT,
        ),
        "cover_tag": ParagraphStyle(
            "cover_tag", parent=base, fontName=body_font, fontSize=9,
            leading=12, textColor=INK_MUTED, alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base, fontName=bold_font, fontSize=20, leading=24,
            textColor=INK_DARK, spaceAfter=10, spaceBefore=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base, fontName=bold_font, fontSize=15, leading=19,
            textColor=INK_DARK, spaceAfter=6, spaceBefore=12,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base, fontName=bold_font, fontSize=10.5, leading=14,
            textColor=BRAND_ORANGE_DEEP, spaceAfter=4, spaceBefore=8,
        ),
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base, fontName=bold_font, fontSize=8.5,
            leading=12, textColor=BRAND_ORANGE, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body", parent=base, fontName=body_font, fontSize=10, leading=14.5,
            textColor=INK_BODY, spaceAfter=6,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base, fontName=body_font, fontSize=10.5,
            leading=15, textColor=INK_DARK, spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "muted", parent=base, fontName=body_font, fontSize=9, leading=12.5,
            textColor=INK_MUTED, spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base, fontName=body_font, fontSize=7.5,
            leading=10, textColor=INK_MUTED, alignment=TA_CENTER,
        ),
    }


# ── Page chrome (header / footer) ───────────────────────────────────────────
@dataclass
class _ReportMeta:
    generated_at: str
    roster_count: int
    has_ai: bool


def _draw_page_chrome(canvas, doc, meta: _ReportMeta) -> None:
    """Draw a thin branded header band and footer on every non-cover page."""
    canvas.saveState()
    width, height = LETTER

    # Top band — thin orange strip with a darker keyline beneath
    canvas.setFillColor(BRAND_ORANGE)
    canvas.rect(0, height - 0.16 * inch, width, 0.16 * inch, stroke=0, fill=1)
    canvas.setFillColor(BRAND_ORANGE_DEEP)
    canvas.rect(0, height - 0.18 * inch, width, 0.02 * inch, stroke=0, fill=1)

    # Header text
    canvas.setFont(_BOLD_FONT, 9)
    canvas.setFillColor(INK_DARK)
    canvas.drawString(0.6 * inch, height - 0.42 * inch, "HOOPLYTICS")
    canvas.setFont(_BODY_FONT, 8)
    canvas.setFillColor(INK_MUTED)
    title_x = 0.6 * inch + 1.1 * inch
    canvas.drawString(
        title_x,
        height - 0.42 * inch,
        "Roster Analytics Report",
    )

    # Active player tag — set by ``_ActivePlayerMarker`` flowables. Rendered
    # in brand orange next to the report title so the reader always knows
    # whose section they are reading without having to scroll back to the
    # player profile header.
    active_player = getattr(canvas, "_hl_active_player", None)
    if active_player:
        title_w = canvas.stringWidth("Roster Analytics Report", _BODY_FONT, 8)
        sep_x = title_x + title_w + 6
        canvas.setFillColor(INK_FAINT)
        canvas.drawString(sep_x, height - 0.42 * inch, "·")
        canvas.setFont(_BOLD_FONT, 8.5)
        canvas.setFillColor(BRAND_ORANGE_DEEP)
        canvas.drawString(sep_x + 8, height - 0.42 * inch, str(active_player))

    canvas.setFont(_BODY_FONT, 8)
    canvas.setFillColor(INK_MUTED)
    canvas.drawRightString(
        width - 0.6 * inch,
        height - 0.42 * inch,
        meta.generated_at,
    )

    # Footer
    canvas.setStrokeColor(PANEL_BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(0.6 * inch, 0.6 * inch, width - 0.6 * inch, 0.6 * inch)
    canvas.setFont(_BODY_FONT, 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawCentredString(
        width / 2,
        0.42 * inch,
        "Hooplytics is for statistical analysis and entertainment purposes only.",
    )
    # Streamlit app attribution (left) and page number (right).
    app_url = "https://hooplytics.streamlit.app/"
    canvas.setFillColor(BRAND_ORANGE_DEEP)
    canvas.drawString(0.6 * inch, 0.28 * inch, app_url)
    url_w = canvas.stringWidth(app_url, _BODY_FONT, 7.5)
    canvas.linkURL(
        app_url,
        (0.6 * inch, 0.22 * inch, 0.6 * inch + url_w, 0.36 * inch),
        relative=0,
    )
    canvas.setFillColor(INK_MUTED)
    canvas.drawRightString(
        width - 0.6 * inch,
        0.28 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _draw_cover_chrome(canvas, doc) -> None:
    canvas.saveState()
    width, height = LETTER
    # Wide orange band on the left edge
    canvas.setFillColor(BRAND_ORANGE)
    canvas.rect(0, 0, 0.55 * inch, height, stroke=0, fill=1)
    # Subtle inner shadow line
    canvas.setFillColor(BRAND_ORANGE_DEEP)
    canvas.rect(0.55 * inch, 0, 0.04 * inch, height, stroke=0, fill=1)
    # Horizontal divider near the top
    canvas.setStrokeColor(PANEL_BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(
        0.95 * inch, height - 1.2 * inch,
        width - 0.6 * inch, height - 1.2 * inch,
    )
    # Footer
    canvas.setFont(_BODY_FONT, 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawCentredString(
        width / 2, 0.55 * inch,
        "Hooplytics  |  Roster Analytics Report",
    )
    app_url = "https://hooplytics.streamlit.app/"
    canvas.setFillColor(BRAND_ORANGE_DEEP)
    canvas.drawCentredString(width / 2, 0.38 * inch, app_url)
    url_w = canvas.stringWidth(app_url, _BODY_FONT, 7.5)
    canvas.linkURL(
        app_url,
        (
            width / 2 - url_w / 2,
            0.32 * inch,
            width / 2 + url_w / 2,
            0.46 * inch,
        ),
        relative=0,
    )
    canvas.restoreState()


# ── Helpers ─────────────────────────────────────────────────────────────────
def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(f):
        return "—"
    return f"{f:.{digits}f}"


def _fmt_signed(v: Any, digits: int = 2) -> str:
    s = _fmt(v, digits)
    if s in ("—", "nan"):
        return s
    try:
        return f"+{s}" if float(v) > 0 else s
    except (TypeError, ValueError):
        return s


def _para(text: str, style: ParagraphStyle) -> Paragraph:
    """Wrap untrusted text into a Paragraph after escaping XML metacharacters."""
    safe = (
        _safe_text(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(safe, style)


# ── Probabilistic decision helpers ─────────────────────────────────────────
# These power the bettor-facing additions (hit %, confidence score,
# distribution bands, volatility / role chips, slip builder). All math is
# deterministic and dependency-free so the report stays drop-in renderable.

def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf — no scipy needed."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_inv(p: float) -> float:
    """Approximate inverse standard normal (Acklam) for distribution bands."""
    import math
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _metric_sigma_lookup(metrics: pd.DataFrame | None) -> dict[str, float]:
    """Return ``{model_name: rmse}`` for use as the predictive σ."""
    out: dict[str, float] = {}
    if metrics is None or not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return out
    name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
    rmse_col = next((c for c in ("RMSE", "rmse") if c in metrics.columns), None)
    if not name_col or not rmse_col:
        return out
    for _, r in metrics.iterrows():
        try:
            v = float(r[rmse_col])
        except (TypeError, ValueError):
            continue
        if pd.isna(v) or v <= 0:
            continue
        out[str(r[name_col])] = v
    return out


def _hit_probability(
    projection: Any, line: Any, side: str, sigma: float | None,
) -> float | None:
    """P(prop clears) given a normal residual model.

    Side MORE/OVER → P(actual > line); LESS/UNDER → P(actual < line).
    Returns None when inputs are missing or σ is unknown.
    """
    if sigma is None or sigma <= 0:
        return None
    try:
        proj_f = float(projection)
        line_f = float(line)
    except (TypeError, ValueError):
        return None
    if pd.isna(proj_f) or pd.isna(line_f):
        return None
    z = (line_f - proj_f) / sigma
    cdf_below = _normal_cdf(z)
    s = (side or "").upper().strip().split()[0] if side else ""
    if s in ("LESS", "UNDER"):
        return float(cdf_below)
    if s in ("MORE", "OVER"):
        return float(1.0 - cdf_below)
    return float(max(cdf_below, 1.0 - cdf_below))


def _confidence_score(
    edge: Any, sigma: float | None, books: Any,
) -> int | None:
    """0-100 composite combining standardized edge and market depth.

    edge_pts: |edge|/σ scaled (caps at 2.5σ → 80 pts).
    depth_pts: log-scaled book count (caps at 8 books → 20 pts).
    """
    import math
    try:
        ev = abs(float(edge))
    except (TypeError, ValueError):
        return None
    if sigma is None or sigma <= 0:
        return None
    z = min(ev / sigma, 2.5)
    edge_pts = (z / 2.5) * 80.0
    try:
        b = float(books)
    except (TypeError, ValueError):
        b = 0.0
    if pd.isna(b) or b <= 0:
        depth_pts = 0.0
    else:
        depth_pts = (math.log(min(b, 8) + 1) / math.log(9)) * 20.0
    return int(round(max(0.0, min(100.0, edge_pts + depth_pts))))


def _side_display(side: Any) -> str:
    """Map internal MORE/OVER → ABOVE and LESS/UNDER → BELOW for display.

    Internal data still uses MORE/LESS (PrizePicks lingo) and OVER/UNDER
    (sportsbook lingo) so all comparison logic stays untouched. This helper
    only flips the *displayed* label so the report reads as projection
    analytics rather than sportsbook copy.
    """
    s = str(side or "").upper().strip().split()[0] if side else ""
    if s in ("MORE", "OVER"):
        return "ABOVE"
    if s in ("LESS", "UNDER"):
        return "BELOW"
    return s or "—"


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p * 100:.0f}%"


def _volatility_label(
    games: pd.DataFrame | None, target_col: str = "pts", n: int = 10,
) -> tuple[str, "colors.Color"] | None:
    """Classify a player's recent variance via coefficient of variation."""
    if games is None or not isinstance(games, pd.DataFrame) or games.empty:
        return None
    if target_col not in games.columns:
        return None
    s = pd.to_numeric(games[target_col], errors="coerce").dropna().tail(max(n, 5))
    if len(s) < 5:
        return None
    mean = float(s.mean())
    if mean <= 0:
        return None
    cv = float(s.std(ddof=1)) / mean
    if cv < 0.25:
        return ("LOW VOL", POS_GREEN)
    if cv < 0.40:
        return ("MED VOL", GOLD)
    return ("HIGH VOL", NEG_RED)


def _role_stability_label(
    games: pd.DataFrame | None, n: int = 10,
) -> tuple[str, "colors.Color"] | None:
    """High = stable minutes + shot diet; low = swing role.

    Uses minutes std and FGA std as proxies for usage volatility because
    explicit usage rate isn't always present in the games frame.
    """
    if games is None or not isinstance(games, pd.DataFrame) or games.empty:
        return None
    score = 0.0
    parts = 0
    if "min" in games.columns:
        m = pd.to_numeric(games["min"], errors="coerce").dropna().tail(n)
        if len(m) >= 5 and m.mean() > 0:
            cv = float(m.std(ddof=1)) / float(m.mean())
            score += max(0.0, 1.0 - min(cv / 0.30, 1.0))
            parts += 1
    if "fga" in games.columns:
        f = pd.to_numeric(games["fga"], errors="coerce").dropna().tail(n)
        if len(f) >= 5 and f.mean() > 0:
            cv = float(f.std(ddof=1)) / float(f.mean())
            score += max(0.0, 1.0 - min(cv / 0.40, 1.0))
            parts += 1
    if parts == 0:
        return None
    norm = score / parts
    if norm >= 0.70:
        return ("ROLE: HIGH", POS_GREEN)
    if norm >= 0.45:
        return ("ROLE: MED", GOLD)
    return ("ROLE: LOW", NEG_RED)


def _minutes_projection(
    games: pd.DataFrame | None, n: int = 10,
) -> tuple[float, float, float] | None:
    """Forward-looking minutes estimate: weighted L3/L5/L10 mean + ±1σ band."""
    if games is None or not isinstance(games, pd.DataFrame) or games.empty:
        return None
    if "min" not in games.columns:
        return None
    m = pd.to_numeric(games["min"], errors="coerce").dropna()
    if len(m) < 3:
        return None
    l3 = float(m.tail(3).mean())
    l5 = float(m.tail(5).mean()) if len(m) >= 5 else l3
    l10 = float(m.tail(min(len(m), n)).mean())
    proj = 0.45 * l5 + 0.30 * l3 + 0.25 * l10
    sd = float(m.tail(min(len(m), n)).std(ddof=1)) if len(m) >= 4 else 0.0
    lo = max(0.0, proj - sd)
    hi = proj + sd
    return (proj, lo, hi)


def _distribution_bands(
    projection: Any, sigma: float | None,
) -> dict[str, float] | None:
    """{p25, p50, p75} assuming Normal(projection, σ²)."""
    if sigma is None or sigma <= 0:
        return None
    try:
        mu = float(projection)
    except (TypeError, ValueError):
        return None
    if pd.isna(mu):
        return None
    z25 = _normal_inv(0.25)
    z75 = _normal_inv(0.75)
    return {"p25": mu + z25 * sigma, "p50": mu, "p75": mu + z75 * sigma}


def _correlation_clusters(edge_df: pd.DataFrame | None) -> list[dict]:
    """Find same-game / same-side prop clusters that hurt parlay independence.

    Fires when ≥2 calls share a matchup AND a directional side (LESS/UNDER
    together or MORE/OVER together) — these are tempo-correlated and real
    diversification disappears.
    """
    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return []
    if "matchup" not in edge_df.columns:
        return []
    df = edge_df.copy()
    sides = df.get("call", df.get("side"))
    if sides is None:
        return []
    df["_side"] = sides.astype(str).str.upper().str.split().str[0]
    df["_dir"] = df["_side"].map(
        lambda s: "UNDER" if s in ("LESS", "UNDER") else (
            "OVER" if s in ("MORE", "OVER") else None
        )
    )
    df = df.dropna(subset=["_dir", "matchup"])
    if df.empty:
        return []
    clusters: list[dict] = []
    for (matchup, direction), grp in df.groupby(["matchup", "_dir"]):
        # Dedupe players (one player can have multiple props in the same game)
        # while preserving original order so the warning lists distinct names.
        raw_players = [str(p) for p in grp.get("player", []).tolist()]
        unique_players = list(dict.fromkeys(raw_players))
        if len(unique_players) < 2:
            continue
        clusters.append({
            "matchup": str(matchup),
            "direction": str(direction),
            "players": unique_players,
            "count": int(len(grp)),
        })
    clusters.sort(key=lambda c: -c["count"])
    return clusters


def _section_header(
    title: str,
    eyebrow: str,
    styles: dict,
    *,
    anchor: str | None = None,
) -> list:
    """Return the section header flow list.

    The eyebrow + heading pair is wrapped in a ``KeepTogether`` so it can
    never be orphaned at the bottom of a page (a common formatting wart in
    earlier renders where "SECTION 04" would dangle on its own).
    """
    head: list = []
    if anchor:
        head.append(_AnchorFlowable(anchor, title, level=0))
    # Require ~0.9" of vertical space so the eyebrow + title can't dangle alone
    # at the bottom of a page, but stay tight enough that a fresh-page section
    # header doesn't strand the entire body of the section to the next page
    # (the cause of the "blank page with just a title" rendering wart).
    head.append(CondPageBreak(0.9 * inch))
    block = KeepTogether([
        _para(eyebrow.upper(), styles["eyebrow"]),
        _para(title, styles["h1"]),
        Spacer(1, 4),
    ])
    head.append(block)
    return head


def _toc_flowables(
    items: list[tuple],
    styles: dict,
) -> list:
    """Render a glossy clickable table of contents.

    ``items`` accepts:
      * ``(label, anchor)`` \u2014 top-level section, no description
      * ``(label, anchor, description)`` \u2014 top-level with subtitle line
      * ``(label, anchor, "sub")`` \u2014 indented chevron sub-row (no number)
    """
    if not items:
        return []

    # Hero panel for the front matter \u2014 a magazine-style banner that frames
    # the TOC and signals "this is a real report" before any data appears.
    hero = Table(
        [[
            Paragraph(
                "<font size='8.5' color='#cc5a00'><b>INSIDE THIS REPORT</b></font>"
                "<br/><br/>"
                "<font size='22' color='#11151c'><b>Contents</b></font>"
                "<br/>"
                "<font size='10' color='#6b7686'>"
                "Tap any row to jump straight to that section. "
                "Skim the <b>Signal summary</b> page if you only have a minute."
                "</font>",
                styles["body"],
            )
        ]],
        colWidths=[7.0 * inch],
    )
    hero.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_ORANGE_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BRAND_ORANGE_DEEP),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, PANEL_BORDER),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))

    flow: list = [hero, Spacer(1, 16)]

    rows: list[list[Any]] = []
    is_sub_flags: list[bool] = []
    section_idx = 0
    for entry in items:
        is_sub = len(entry) >= 3 and entry[2] == "sub"
        label, anchor = entry[0], entry[1]
        description = ""
        if not is_sub and len(entry) >= 3 and entry[2] != "sub":
            description = str(entry[2] or "")

        if is_sub:
            # Indented chevron sub-row \u2014 quiet visual nesting under its parent.
            left = Paragraph(
                f'<link href="#{anchor}" color="#6b7686">'
                f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                f'<font size="9" color="#ff7a18">></font>'
                f'&nbsp;&nbsp;<font size="9.5" color="#3a4250">{label}</font>'
                f'</link>',
                styles["body"],
            )
            right = Paragraph("", styles["body"])
        else:
            section_idx += 1
            num = f"{section_idx:02d}"
            if description:
                left = Paragraph(
                    f'<link href="#{anchor}" color="#11151c">'
                    f'<font size="8" color="#ffffff" backColor="#ff7a18">'
                    f'&nbsp;<b>&nbsp;{num}&nbsp;</b>&nbsp;</font>'
                    f'&nbsp;&nbsp;<font size="11.5" color="#11151c"><b>{label}</b></font>'
                    f'<br/>'
                    f'<font size="8.5" color="#6b7686">'
                    f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{description}</font>'
                    f'</link>',
                    styles["body"],
                )
            else:
                left = Paragraph(
                    f'<link href="#{anchor}" color="#11151c">'
                    f'<font size="8" color="#ffffff" backColor="#ff7a18">'
                    f'&nbsp;<b>&nbsp;{num}&nbsp;</b>&nbsp;</font>'
                    f'&nbsp;&nbsp;<font size="11.5" color="#11151c"><b>{label}</b></font>'
                    f'</link>',
                    styles["body"],
                )
            right = Paragraph(
                f'<link href="#{anchor}" color="#cc5a00">'
                f'<font size="11" color="#cc5a00"><b>></b></font>'
                f'</link>',
                ParagraphStyle(
                    "toc_jump", parent=styles["body"],
                    alignment=2,  # right
                ),
            )
        rows.append([left, right])
        is_sub_flags.append(is_sub)

    table = Table(rows, colWidths=[6.4 * inch, 0.6 * inch])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]
    # Per-row backgrounds: white for main sections, faint panel for sub-items
    # so the indented player rows visually nest under their parent section.
    for ri, sub in enumerate(is_sub_flags):
        if sub:
            style.append(("BACKGROUND", (0, ri), (-1, ri), PANEL_BG))
            style.append(("TOPPADDING", (0, ri), (-1, ri), 4))
            style.append(("BOTTOMPADDING", (0, ri), (-1, ri), 4))
        else:
            style.append(("BACKGROUND", (0, ri), (-1, ri), colors.white))
        # Hairline separator between rows (skip last row).
        if ri < len(is_sub_flags) - 1:
            style.append(("LINEBELOW", (0, ri), (-1, ri), 0.4, PANEL_BORDER))
    style.append(("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER))
    table.setStyle(TableStyle(style))
    flow.append(table)
    return flow


# ── KPI strip ───────────────────────────────────────────────────────────────
def _kpi_card_flowable(
    value: str,
    label: str,
    *,
    accent: colors.Color,
    styles: dict,
) -> Table:
    """A single KPI tile with a colored accent bar across the top."""
    body = Paragraph(
        f"<font size='20' color='#11151c'><b>{value}</b></font><br/>"
        f"<font size='7.5' color='#6b7686'>{label.upper()}</font>",
        ParagraphStyle(
            "kpi", parent=styles["body"], alignment=TA_CENTER,
            leading=14, textColor=INK_DARK,
        ),
    )
    accent_bar = Paragraph(
        " ",
        ParagraphStyle(
            "kpi_accent", parent=styles["body"],
            backColor=accent, leading=4, fontSize=2,
        ),
    )
    inner = Table(
        [[accent_bar], [body]],
        colWidths=[1.18 * inch],
        rowHeights=[0.06 * inch, None],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), accent),
        ("BACKGROUND", (0, 1), (0, 1), PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
        ("TOPPADDING", (0, 1), (0, 1), 9),
        ("BOTTOMPADDING", (0, 1), (0, 1), 9),
        ("LEFTPADDING", (0, 1), (0, 1), 4),
        ("RIGHTPADDING", (0, 1), (0, 1), 4),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
    ]))
    return inner


def _kpi_strip_flowables(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    styles: dict,
) -> list:
    """Render a KPI strip with accent-bar tiles."""
    median_r2 = float("nan")
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        if r2_col is not None:
            median_r2 = float(pd.to_numeric(metrics[r2_col], errors="coerce").median())

    edge_count = int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0
    strong_count = 0
    mean_abs_edge = float("nan")
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "edge" in edge_df.columns:
        abs_edge = pd.to_numeric(edge_df["edge"], errors="coerce").abs()
        strong_count = int((abs_edge >= 2.0).sum())
        mean_abs_edge = float(abs_edge.mean())

    proj_rows = 0
    for frame in (projections or {}).values():
        if isinstance(frame, pd.DataFrame):
            proj_rows += int(len(frame))

    cards_spec = [
        ("Players", str(len(roster)), BRAND_ORANGE),
        ("Model rows", str(proj_rows), NEUTRAL_BLUE),
        ("Live signals", str(edge_count), GOLD),
        ("Strong signals", str(strong_count), POS_GREEN if strong_count else INK_FAINT),
        ("Avg |edge|", _fmt(mean_abs_edge, 2) if not pd.isna(mean_abs_edge) else "—", BRAND_ORANGE_DEEP),
        ("Median R²", _fmt(median_r2, 2) if not pd.isna(median_r2) else "—", NEUTRAL_BLUE),
    ]
    cards = [_kpi_card_flowable(v, lbl, accent=ac, styles=styles) for lbl, v, ac in cards_spec]
    strip = Table([cards], colWidths=[1.18 * inch] * len(cards))
    strip.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [strip, Spacer(1, 12)]


# ── Deterministic prose ─────────────────────────────────────────────────────
def _slate_lean_label(edge_df: pd.DataFrame | None) -> tuple[str, int, int]:
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty or "edge" not in edge_df.columns:
        return "neutral", 0, 0
    edges = pd.to_numeric(edge_df["edge"], errors="coerce").dropna()
    if edges.empty:
        return "neutral", 0, 0
    pos = int((edges > 0).sum())
    neg = int((edges < 0).sum())
    if pos > neg * 1.4:
        return "ABOVE-leaning", pos, neg
    if neg > pos * 1.4:
        return "BELOW-leaning", pos, neg
    return "balanced", pos, neg


def _deterministic_summary_text(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
) -> str:
    players_n = len(roster)
    edge_n = int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0
    lean, pos, neg = _slate_lean_label(edge_df)

    median_r2_txt = "—"
    best_target = ""
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
        if r2_col is not None:
            r2 = pd.to_numeric(metrics[r2_col], errors="coerce")
            median_r2 = float(r2.median())
            if not pd.isna(median_r2):
                median_r2_txt = f"{median_r2:.2f}"
            if name_col is not None and not r2.dropna().empty:
                idx = r2.idxmax()
                best_target = f" Best-fit model: {metrics.loc[idx, name_col]} (R²={r2.max():.2f})."

    top_line = "No live signal rows are available."
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "edge" in edge_df.columns:
        df = edge_df.copy()
        df["abs_edge"] = pd.to_numeric(df["edge"], errors="coerce").abs()
        df = df.sort_values("abs_edge", ascending=False)
        if not df.empty:
            r = df.iloc[0]
            side = str(r.get("call") or r.get("side") or "").upper() or "—"
            top_line = (
                f"Loudest signal: {_short(r.get('player', 'Unknown'), 24)} "
                f"{r.get('model', 'metric')} ({side}) at edge "
                f"{_fmt_signed(r.get('edge'), 2)}."
            )

    lean_phrase = ""
    if edge_n:
        lean_phrase = (
            f" Slate posture: {lean} ({pos} ABOVE / {neg} BELOW across {edge_n} mapped lines)."
        )

    return (
        f"Tracking {players_n} player{'s' if players_n != 1 else ''} with "
        f"{edge_n} live signal row{'s' if edge_n != 1 else ''} and a median "
        f"model R² of {median_r2_txt}.{best_target}{lean_phrase} {top_line}"
    )


# ── Charts ──────────────────────────────────────────────────────────────────
def _r2_lollipop_chart(metrics: pd.DataFrame) -> Drawing | None:
    """Lollipop-style R² chart with a colored stem per model."""
    if metrics.empty:
        return None
    r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
    name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
    rmse_col = next((c for c in ("RMSE", "rmse") if c in metrics.columns), None)
    if r2_col is None or name_col is None:
        return None

    df = metrics[[name_col, r2_col] + ([rmse_col] if rmse_col else [])].copy()
    df[r2_col] = pd.to_numeric(df[r2_col], errors="coerce")
    df = df.dropna(subset=[r2_col]).sort_values(r2_col, ascending=False).head(10)
    if df.empty:
        return None

    width = 7.0 * inch
    height = 2.55 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 14, "Model R²  |  accuracy lollipop",
        fontName=_BOLD_FONT, fontSize=10.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 28, "Higher = the model explains more game-to-game variance.",
        fontName=_BODY_FONT, fontSize=8, fillColor=INK_MUTED,
    ))

    plot_x = 0.45 * inch
    plot_y = 0.4 * inch
    plot_w = width - plot_x - 0.2 * inch
    plot_h = height - 1.0 * inch

    drawing.add(Rect(
        plot_x, plot_y, plot_w, plot_h,
        fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
    ))

    n = len(df)
    max_r2 = max(0.6, float(df[r2_col].max()) * 1.15)
    for tick in (0.2, 0.4, 0.6):
        if tick > max_r2:
            continue
        ty = plot_y + (tick / max_r2) * plot_h
        drawing.add(Line(
            plot_x, ty, plot_x + plot_w, ty,
            strokeColor=PANEL_BORDER, strokeWidth=0.3, strokeDashArray=[1, 2],
        ))
        drawing.add(String(
            plot_x - 4, ty - 3, f"{tick:.1f}",
            fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_FAINT,
            textAnchor="end",
        ))

    slot = plot_w / max(n, 1)
    for i, (_, row) in enumerate(df.iterrows()):
        cx = plot_x + slot * (i + 0.5)
        r2v = float(row[r2_col])
        cy = plot_y + (r2v / max_r2) * plot_h
        if r2v >= 0.45:
            color = POS_GREEN
        elif r2v >= 0.30:
            color = BRAND_ORANGE
        elif r2v >= 0.18:
            color = GOLD
        else:
            color = NEG_RED
        drawing.add(Line(cx, plot_y, cx, cy, strokeColor=color, strokeWidth=2))
        drawing.add(Circle(cx, cy, 4.2, fillColor=color, strokeColor=WHITE, strokeWidth=1))
        drawing.add(String(
            cx, cy + 7, f"{r2v:.2f}",
            fontName=_BOLD_FONT, fontSize=7.5, fillColor=INK_DARK,
            textAnchor="middle",
        ))
        drawing.add(String(
            cx, plot_y - 11, _short(row[name_col], 11),
            fontName=_BODY_FONT, fontSize=7, fillColor=INK_BODY,
            textAnchor="middle",
        ))
        if rmse_col is not None and pd.notna(row.get(rmse_col)):
            drawing.add(String(
                cx, plot_y - 20, f"RMSE {float(row[rmse_col]):.2f}",
                fontName=_BODY_FONT, fontSize=6, fillColor=INK_FAINT,
                textAnchor="middle",
            ))

    return drawing


def _diverging_edge_chart(edge_df: pd.DataFrame) -> Drawing | None:
    """Diverging horizontal bar chart with a zero reference line."""
    if edge_df.empty or "edge" not in edge_df.columns:
        return None
    df = edge_df.copy()
    df["edge_num"] = pd.to_numeric(df["edge"], errors="coerce")
    df = df.dropna(subset=["edge_num"])
    if df.empty:
        return None
    if "abs_edge" not in df.columns:
        df["abs_edge"] = df["edge_num"].abs()
    df = df.sort_values("abs_edge", ascending=False).head(10).iloc[::-1]

    labels = [
        f"{_short(r.get('player', ''), 14)}  |  {_short(r.get('model', ''), 9)}"
        for _, r in df.iterrows()
    ]
    values = [float(v) for v in df["edge_num"]]
    span = max(abs(min(values)), abs(max(values)), 0.5) * 1.15

    width = 7.0 * inch
    height = 2.85 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 14, "Signal magnitudes  |  model vs line",
        fontName=_BOLD_FONT, fontSize=10.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 28, "Bars to the right = ABOVE-line signal    Bars to the left = BELOW-line signal.",
        fontName=_BODY_FONT, fontSize=8, fillColor=INK_MUTED,
    ))

    plot_x = 1.95 * inch
    plot_y = 0.32 * inch
    plot_w = width - plot_x - 0.25 * inch
    plot_h = height - 0.85 * inch

    drawing.add(Rect(
        plot_x, plot_y, plot_w, plot_h,
        fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
    ))

    zero_x = plot_x + plot_w / 2.0
    drawing.add(Line(
        zero_x, plot_y, zero_x, plot_y + plot_h,
        strokeColor=INK_MUTED, strokeWidth=0.6,
    ))
    for frac in (-1.0, -0.5, 0.5, 1.0):
        tx = zero_x + (plot_w / 2) * frac
        drawing.add(Line(
            tx, plot_y, tx, plot_y + plot_h,
            strokeColor=PANEL_BORDER, strokeWidth=0.3, strokeDashArray=[1, 2],
        ))
        drawing.add(String(
            tx, plot_y - 10, f"{span * frac:+.1f}",
            fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_FAINT,
            textAnchor="middle",
        ))

    n = len(values)
    row_h = plot_h / max(n, 1)
    bar_h = max(7, row_h * 0.55)
    for i, (lbl, v) in enumerate(zip(labels, values)):
        cy = plot_y + row_h * (i + 0.5)
        bar_len = (abs(v) / span) * (plot_w / 2)
        if v >= 0:
            color = POS_GREEN
            x0 = zero_x
            x1 = zero_x + bar_len
        else:
            color = NEG_RED
            x0 = zero_x - bar_len
            x1 = zero_x
        drawing.add(Rect(
            x0, cy - bar_h / 2, max(x1 - x0, 1), bar_h,
            fillColor=color, strokeColor=WHITE, strokeWidth=0.6,
        ))
        if v >= 0:
            drawing.add(String(
                x1 + 4, cy - 3, _fmt_signed(v, 2),
                fontName=_BOLD_FONT, fontSize=7, fillColor=color,
            ))
        else:
            drawing.add(String(
                x0 - 4, cy - 3, _fmt_signed(v, 2),
                fontName=_BOLD_FONT, fontSize=7, fillColor=color,
                textAnchor="end",
            ))
        drawing.add(String(
            plot_x - 6, cy - 3, lbl,
            fontName=_BODY_FONT, fontSize=7.5, fillColor=INK_BODY,
            textAnchor="end",
        ))

    return drawing


def _edge_distribution_chart(edge_df: pd.DataFrame) -> Drawing | None:
    """Histogram of all edges, color-coded by sign."""
    if edge_df.empty or "edge" not in edge_df.columns:
        return None
    edges = pd.to_numeric(edge_df["edge"], errors="coerce").dropna().tolist()
    if not edges:
        return None

    span = max(abs(min(edges)), abs(max(edges)), 1.0)
    n_bins = 7
    bin_w = (2 * span) / n_bins
    counts = [0] * n_bins
    for v in edges:
        idx = min(int((v + span) / bin_w), n_bins - 1)
        counts[idx] += 1

    width = 3.4 * inch
    height = 2.15 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 14, "Signal distribution",
        fontName=_BOLD_FONT, fontSize=10, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 26, "How concentrated are the model-vs-line gaps?",
        fontName=_BODY_FONT, fontSize=7.5, fillColor=INK_MUTED,
    ))

    plot_x = 0.25 * inch
    plot_y = 0.32 * inch
    plot_w = width - plot_x - 0.1 * inch
    plot_h = height - 0.85 * inch

    drawing.add(Rect(
        plot_x, plot_y, plot_w, plot_h,
        fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
    ))

    max_c = max(counts) or 1
    bar_w = plot_w / n_bins
    for i, c in enumerate(counts):
        center = -span + (i + 0.5) * bin_w
        color = POS_GREEN if center > 0.05 else (NEG_RED if center < -0.05 else INK_FAINT)
        bx = plot_x + i * bar_w + 1
        bh = (c / max_c) * (plot_h - 14)
        drawing.add(Rect(
            bx, plot_y, max(bar_w - 2, 1), bh,
            fillColor=color, strokeColor=WHITE, strokeWidth=0.5,
        ))
        if c:
            drawing.add(String(
                bx + (bar_w - 2) / 2, plot_y + bh + 2, str(c),
                fontName=_BOLD_FONT, fontSize=7, fillColor=INK_DARK,
                textAnchor="middle",
            ))

    drawing.add(String(
        plot_x, plot_y - 10, f"{-span:+.1f}",
        fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_FAINT,
    ))
    drawing.add(String(
        plot_x + plot_w / 2, plot_y - 10, "0",
        fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_FAINT,
        textAnchor="middle",
    ))
    drawing.add(String(
        plot_x + plot_w, plot_y - 10, f"{span:+.1f}",
        fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_FAINT,
        textAnchor="end",
    ))

    return drawing


def _slate_summary_panel(
    edge_df: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    styles: dict,
) -> Table:
    """Compact text panel of derived slate stats next to the histogram."""
    lean, pos, neg = _slate_lean_label(edge_df)
    edge_n = int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0
    avg_books = float("nan")
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "books" in edge_df.columns:
        avg_books = float(pd.to_numeric(edge_df["books"], errors="coerce").mean())

    best_r2_line = "—"
    worst_r2_line = "—"
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
        if r2_col is not None and name_col is not None:
            r2 = pd.to_numeric(metrics[r2_col], errors="coerce")
            if not r2.dropna().empty:
                best_r2_line = f"{metrics.loc[r2.idxmax(), name_col]} ({r2.max():.2f})"
                worst_r2_line = f"{metrics.loc[r2.idxmin(), name_col]} ({r2.min():.2f})"

    rows = [
        ("Slate posture", lean.upper()),
        ("ABOVE / BELOW leans", f"{pos} / {neg}"),
        ("Mapped signal rows", str(edge_n)),
        ("Avg books per row", _fmt(avg_books, 1) if not pd.isna(avg_books) else "—"),
        ("Most reliable model", best_r2_line),
        ("Noisiest model", worst_r2_line),
    ]

    cells = []
    for label, val in rows:
        cells.append([
            Paragraph(
                f"<font size='7' color='#6b7686'>{label.upper()}</font>",
                styles["muted"],
            ),
            Paragraph(
                f"<font size='9.5' color='#11151c'><b>{val}</b></font>",
                styles["body"],
            ),
        ])
    t = Table(cells, colWidths=[1.45 * inch, 2.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, PANEL_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _player_minichart(
    projection: pd.DataFrame,
    edge_df: pd.DataFrame | None,
    player: str,
) -> Drawing | None:
    """Side-by-side bars per market: model projection vs posted line."""
    if not isinstance(projection, pd.DataFrame) or projection.empty:
        return None
    proj = projection.copy()
    edge_lookup: dict[str, dict] = {}
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "player" in edge_df.columns:
        sub = edge_df[edge_df["player"] == player]
        for _, r in sub.iterrows():
            edge_lookup[str(r["model"])] = {
                "line": r.get("posted line", r.get("line")),
                "edge": r.get("edge"),
            }
    rows: list[tuple[str, float, float, float]] = []
    for _, r in proj.iterrows():
        name = str(r.get("model", ""))
        ed = edge_lookup.get(name)
        if not ed:
            continue
        line_v = ed.get("line")
        try:
            line_f = float(line_v)
            pred_f = float(r["prediction"])
            edge_f = float(ed.get("edge")) if ed.get("edge") is not None else (pred_f - line_f)
        except (TypeError, ValueError):
            continue
        if pd.isna(line_f) or pd.isna(pred_f):
            continue
        rows.append((name, pred_f, line_f, edge_f))
    if not rows:
        return None

    width = 7.0 * inch
    height = 1.55 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 13, "Model projection vs posted line",
        fontName=_BOLD_FONT, fontSize=9.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 24, "Orange = model    Grey = market line.",
        fontName=_BODY_FONT, fontSize=7.5, fillColor=INK_MUTED,
    ))

    plot_x = 0.2 * inch
    plot_y = 0.3 * inch
    plot_w = width - plot_x - 0.2 * inch
    plot_h = height - 0.7 * inch

    drawing.add(Rect(
        plot_x, plot_y, plot_w, plot_h,
        fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
    ))

    n = len(rows)
    slot = plot_w / max(n, 1)
    max_v = max(max(p, l) for _, p, l, _ in rows) * 1.15 or 1.0
    bar_w = min(14, slot * 0.30)
    for i, (name, pred, line_v, edge_v) in enumerate(rows):
        cx = plot_x + slot * (i + 0.5)
        h_pred = (pred / max_v) * plot_h
        h_line = (line_v / max_v) * plot_h
        drawing.add(Rect(
            cx - bar_w - 1, plot_y, bar_w, h_pred,
            fillColor=BRAND_ORANGE, strokeColor=WHITE, strokeWidth=0.5,
        ))
        drawing.add(Rect(
            cx + 1, plot_y, bar_w, h_line,
            fillColor=INK_FAINT, strokeColor=WHITE, strokeWidth=0.5,
        ))
        edge_color = POS_GREEN if edge_v > 0 else (NEG_RED if edge_v < 0 else INK_BODY)
        drawing.add(String(
            cx, plot_y + max(h_pred, h_line) + 3, _fmt_signed(edge_v, 2),
            fontName=_BOLD_FONT, fontSize=7, fillColor=edge_color,
            textAnchor="middle",
        ))
        drawing.add(String(
            cx, plot_y - 10, _short(name, 10),
            fontName=_BODY_FONT, fontSize=6.8, fillColor=INK_BODY,
            textAnchor="middle",
        ))

    return drawing


# ── Section assemblers ──────────────────────────────────────────────────────
def _conviction_leaderboard(
    edge_df: pd.DataFrame,
    styles: dict,
    *,
    top_n: int = 10,
) -> list | None:
    """Replacement for the old edge-vs-books scatter.

    A ranked leaderboard of the highest-conviction signals. Each row shows:
    a numbered rank chip, the player + market, a horizontal bar whose length
    encodes ``|edge|``, the signed edge value, and a "books" pill badge.
    Sorted by a composite conviction score ``|edge| * sqrt(books)`` so that
    deep-market signals out-rank shallow ones at the same edge.
    """
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None
    if not {"edge", "books"}.issubset(edge_df.columns):
        return None
    df = edge_df.copy()
    df["edge_n"] = pd.to_numeric(df["edge"], errors="coerce")
    df["books_n"] = pd.to_numeric(df["books"], errors="coerce")
    df = df.dropna(subset=["edge_n", "books_n"])
    if df.empty:
        return None
    df["abs_edge"] = df["edge_n"].abs()
    df["conviction"] = df["abs_edge"] * df["books_n"].clip(lower=1).pow(0.5)
    df = df.sort_values("conviction", ascending=False).head(top_n).reset_index(drop=True)

    side_col = "side" if "side" in df.columns else ("call" if "call" in df.columns else None)
    max_abs_edge = max(float(df["abs_edge"].max()), 0.5)
    max_books = max(int(df["books_n"].max()), 1)
    bar_full_w = 2.7 * inch  # max bar length in points

    title = Paragraph(
        "<font size='10.5' color='#11151c'><b>Conviction leaderboard</b></font><br/>"
        "<font size='8' color='#6b7686'>"
        "Sorted by |edge| weighted by market depth (deeper books = stronger conviction)."
        "</font>",
        styles["body"],
    )

    rows: list[list[Any]] = [[
        Paragraph("<font size='7' color='#6b7686'><b>RANK</b></font>", styles["body"]),
        Paragraph("<font size='7' color='#6b7686'><b>SIGNAL</b></font>", styles["body"]),
        Paragraph(
            "<para alignment='center'>"
            "<font size='7' color='#6b7686'><b>"
            "&lt;&nbsp; BELOW&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;ABOVE &nbsp;&gt;"
            "</b></font></para>",
            styles["body"],
        ),
        Paragraph("<font size='7' color='#6b7686'><b>EDGE</b></font>", styles["body"]),
        Paragraph("<font size='7' color='#6b7686'><b>BOOKS</b></font>", styles["body"]),
    ]]

    extra: list = []
    for i, r in df.iterrows():
        idx = int(i) + 1
        edge_val = float(r["edge_n"])
        abs_edge = float(r["abs_edge"])
        books_n = int(r["books_n"])
        side = ""
        if side_col and pd.notna(r.get(side_col)):
            side = str(r[side_col]).upper().split()[0]
        if side in ("MORE", "OVER"):
            color = POS_GREEN
            side_label = "MORE"
        elif side in ("LESS", "UNDER"):
            color = NEG_RED
            side_label = "LESS"
        else:
            color = NEUTRAL_BLUE
            side_label = "—"
        hex_color = color.hexval()[2:]

        # Rank chip — numbered tile with conviction-tier shading.
        if idx <= 3:
            chip_bg = "#11151c"
        elif idx <= 6:
            chip_bg = "#2b3340"
        else:
            chip_bg = "#6b7686"
        rank_cell = Paragraph(
            f"<font size='10' color='#ffffff' backColor='{chip_bg}'>"
            f"&nbsp;&nbsp;<b>{idx:>2}</b>&nbsp;&nbsp;</font>",
            styles["body"],
        )

        # Signal label: player on top, market+side below.
        name = _short(str(r.get("player", "")) or "—", 28)
        market = str(r.get("model", "")) or ""
        signal_cell = Paragraph(
            f"<font size='9.5' color='#11151c'><b>{name}</b></font><br/>"
            f"<font size='7.5' color='#6b7686'>{market.upper()}  "
            f"<font color='#{hex_color}'><b>{side_label}</b></font></font>",
            styles["body"],
        )

        # Horizontal bar — drawn inside a tiny Drawing.
        # Center-anchored: a vertical mid-axis splits MORE (right, green)
        # from LESS (left, red), so direction reads at a glance.
        bar_h = 9.0
        half_w = bar_full_w / 2.0
        fill_w = max(6.0, half_w * (abs_edge / max_abs_edge))
        bar_drawing = Drawing(bar_full_w, bar_h + 4)
        # Track (faint background spans full width)
        bar_drawing.add(Rect(
            0, 2, bar_full_w, bar_h,
            fillColor=PANEL_BG, strokeColor=None,
        ))
        # Filled bar grows from the midline outward in the side's direction.
        if edge_val >= 0:
            bar_drawing.add(Rect(
                half_w, 2, fill_w, bar_h,
                fillColor=color, strokeColor=None,
            ))
        else:
            bar_drawing.add(Rect(
                half_w - fill_w, 2, fill_w, bar_h,
                fillColor=color, strokeColor=None,
            ))
        # Center axis tick — thin dark line marking the zero point.
        bar_drawing.add(Rect(
            half_w - 0.4, 0, 0.8, bar_h + 4,
            fillColor=INK_DARK, strokeColor=None,
        ))

        edge_cell = Paragraph(
            f"<font size='10' color='#{hex_color}'><b>{_fmt_signed(edge_val, 2)}</b></font>",
            styles["body"],
        )

        # Books pill — always solid INK_DARK chip, opacity-tier in the body
        # rather than the background, so the badge stays visible against
        # zebra-striped rows.
        if books_n >= max(6, max_books - 1):
            pill_bg = "#11151c"
            pill_fg = "#ffffff"
        elif books_n >= 4:
            pill_bg = "#3a4250"
            pill_fg = "#ffffff"
        else:
            pill_bg = "#a8b0bd"
            pill_fg = "#ffffff"
        books_cell = Paragraph(
            f"<font size='8' color='{pill_fg}' backColor='{pill_bg}'>"
            f"&nbsp;&nbsp;<b>{books_n}</b>&nbsp;&nbsp;</font>",
            styles["body"],
        )

        rows.append([rank_cell, signal_cell, bar_drawing, edge_cell, books_cell])

    table = Table(
        rows,
        colWidths=[
            0.55 * inch,
            2.10 * inch,
            bar_full_w,
            0.55 * inch,
            0.55 * inch,
        ],
    )
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("ALIGN", (4, 0), (4, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        # Header row
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, PANEL_BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
    ]
    # Zebra striping for body rows
    for ri in range(1, len(rows)):
        if ri % 2 == 0:
            style.append(("BACKGROUND", (0, ri), (-1, ri), PANEL_BG))
    style.extend(extra)
    table.setStyle(TableStyle(style))
    return [title, Spacer(1, 6), table]


def _edge_confidence_scatter(
    edge_df: pd.DataFrame,
) -> tuple[Drawing, list[tuple[int, str, colors.Color]]] | None:
    """Scatter of |edge| vs market depth (book count).

    Returns the Drawing plus a numbered legend ``(idx, label, color)`` so a
    caller can render the player names below the plot — the chart itself only
    shows numbered markers, which keeps every label collision-free no matter
    how many props share the same coordinates.
    """
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None
    if not {"edge", "books"}.issubset(edge_df.columns):
        return None
    df = edge_df.copy()
    df["edge_n"] = pd.to_numeric(df["edge"], errors="coerce")
    df["books_n"] = pd.to_numeric(df["books"], errors="coerce")
    df = df.dropna(subset=["edge_n", "books_n"])
    if df.empty:
        return None
    df["abs_edge"] = df["edge_n"].abs()
    df = df.sort_values("abs_edge", ascending=False).reset_index(drop=True)

    width = 7.0 * inch
    height = 2.4 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 13, "Edge size vs market depth",
        fontName=_BOLD_FONT, fontSize=10, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 25,
        "Numbered markers map to the legend below — top-right = the highest-conviction zone.",
        fontName=_BODY_FONT, fontSize=8, fillColor=INK_MUTED,
    ))

    plot_x = 0.6 * inch
    plot_y = 0.7 * inch
    plot_w = width - plot_x - 0.25 * inch
    plot_h = height - 1.2 * inch

    drawing.add(Rect(
        plot_x, plot_y, plot_w, plot_h,
        fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
    ))

    max_edge = max(float(df["abs_edge"].max()), 0.5)
    max_books = max(float(df["books_n"].max()), 3.0)

    # Quadrant guides at the midpoints
    mid_x = plot_x + plot_w * 0.5
    mid_y = plot_y + plot_h * 0.5
    drawing.add(Line(
        mid_x, plot_y, mid_x, plot_y + plot_h,
        strokeColor=INK_FAINT, strokeWidth=0.4, strokeDashArray=[2, 3],
    ))
    drawing.add(Line(
        plot_x, mid_y, plot_x + plot_w, mid_y,
        strokeColor=INK_FAINT, strokeWidth=0.4, strokeDashArray=[2, 3],
    ))
    drawing.add(String(
        plot_x + 6, plot_y + plot_h - 11, "HIGH CONVICTION ZONE >",
        fontName=_BOLD_FONT, fontSize=6.5, fillColor=INK_FAINT,
        textAnchor="start",
    ))

    # Axis labels
    drawing.add(String(
        plot_x + plot_w / 2, 8, "|EDGE|  >",
        fontName=_BOLD_FONT, fontSize=7, fillColor=INK_MUTED,
        textAnchor="middle",
    ))
    drawing.add(String(
        10, plot_y + plot_h / 2, "BOOKS",
        fontName=_BOLD_FONT, fontSize=7, fillColor=INK_MUTED,
        textAnchor="middle",
    ))

    # Axis ticks (a couple of reference values)
    for tick_frac in (0.0, 0.5, 1.0):
        tx = plot_x + plot_w * tick_frac
        drawing.add(String(
            tx, plot_y - 10, f"{max_edge * tick_frac:.1f}",
            fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_MUTED,
            textAnchor="middle",
        ))
        ty = plot_y + plot_h * tick_frac
        drawing.add(String(
            plot_x - 4, ty - 2, f"{int(round(max_books * tick_frac))}",
            fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_MUTED,
            textAnchor="end",
        ))

    side_col = "side" if "side" in df.columns else ("call" if "call" in df.columns else None)
    legend: list[tuple[int, str, colors.Color]] = []

    # Plot every row as a numbered marker; only the top 8 get listed in the
    # legend so we don't blow the page out.
    legend_limit = 8
    placed: list[tuple[float, float, float]] = []  # (x, y, radius)
    for i, r in df.iterrows():
        idx = i + 1
        ax = plot_x + (float(r["abs_edge"]) / max_edge) * plot_w
        ay = plot_y + (float(r["books_n"]) / max_books) * plot_h
        side = ""
        if side_col and pd.notna(r.get(side_col)):
            side = str(r[side_col]).upper().split()[0]
        if side in ("MORE", "OVER"):
            color = POS_GREEN
        elif side in ("LESS", "UNDER"):
            color = NEG_RED
        else:
            color = NEUTRAL_BLUE

        radius = 7.5 if idx <= legend_limit else 4.0

        # Anti-collision: nudge marker if it would overlap a placed one.
        # Spiral outward in a small jitter pattern until clear or capped.
        nudge_steps = 0
        max_nudges = 24
        spiral = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (-1, 1), (-1, -1), (1, -1),
        ]
        while nudge_steps < max_nudges:
            collides = False
            for (px, py, pr) in placed:
                min_d = pr + radius + 1.5
                if (ax - px) ** 2 + (ay - py) ** 2 < min_d * min_d:
                    collides = True
                    break
            if not collides:
                break
            dx, dy = spiral[nudge_steps % len(spiral)]
            step = (radius * 1.6) * (1 + nudge_steps // len(spiral))
            ax = max(plot_x + radius, min(plot_x + plot_w - radius, ax + dx * step))
            ay = max(plot_y + radius, min(plot_y + plot_h - radius, ay + dy * step))
            nudge_steps += 1
        placed.append((ax, ay, radius))

        drawing.add(Circle(
            ax, ay, radius,
            fillColor=color, strokeColor=WHITE, strokeWidth=0.8,
        ))
        if idx <= legend_limit:
            drawing.add(String(
                ax, ay - 2.6, str(idx),
                fontName=_BOLD_FONT, fontSize=7.5, fillColor=WHITE,
                textAnchor="middle",
            ))
            name = str(r.get("player", "")) or "—"
            market = str(r.get("model", "")) or ""
            label = f"{name}  |  {market}" if market else name
            legend.append((idx, label, color))

    return drawing, legend


def _edge_confidence_legend(
    legend: list[tuple[int, str, colors.Color]],
    styles: dict,
) -> Table | None:
    """Render the numbered scatter legend as a clean two-column grid."""
    if not legend:
        return None
    # Lay out as two side-by-side columns of items.
    half = (len(legend) + 1) // 2
    col_a = legend[:half]
    col_b = legend[half:]

    def _row(item: tuple[int, str, colors.Color]) -> Paragraph:
        idx, label, color = item
        hex_str = color.hexval()[2:] if hasattr(color, "hexval") else "1f9d6c"
        return Paragraph(
            f"<font size='8' color='#ffffff' backColor='#{hex_str}'>"
            f"&nbsp;<b>{idx:>2}</b>&nbsp;</font>"
            f"&nbsp;&nbsp;<font size='8.5' color='#11151c'>{label}</font>",
            styles["body"],
        )

    n_rows = max(len(col_a), len(col_b))
    rows: list[list[Any]] = []
    for i in range(n_rows):
        left = _row(col_a[i]) if i < len(col_a) else Paragraph("", styles["body"])
        right = _row(col_b[i]) if i < len(col_b) else Paragraph("", styles["body"])
        rows.append([left, right])

    table = Table(rows, colWidths=[3.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _how_to_read_strip(styles: dict) -> Table:
    """Inline legend explaining the visual grammar of the report."""
    items = [
        (POS_GREEN, "ABOVE-LINE signal"),
        (NEG_RED, "BELOW-LINE signal"),
        (BRAND_ORANGE, "Model projection"),
        (INK_FAINT, "Posted market line"),
        (NEUTRAL_BLUE, "Neutral signal"),
    ]
    cells: list = []
    for color, label in items:
        swatch = Table([[Paragraph(
            f"<font size='8' color='#11151c'>{label}</font>", styles["body"],
        )]], colWidths=[1.35 * inch])
        swatch.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
            ("LINEBEFORE", (0, 0), (0, -1), 3, color),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        cells.append(swatch)
    head = Paragraph(
        "<font size='7' color='#6b7686'><b>HOW TO READ THIS REPORT</b></font>",
        styles["muted"],
    )
    legend = Table([cells], colWidths=[1.4 * inch] * len(items))
    legend.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return Table([[head], [legend]], colWidths=[7.0 * inch])


def _analytics_visuals_flowables(
    *,
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Analytics visuals", "Section 05", styles, anchor="sec-analytics")

    rendered = False
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        chart = _r2_lollipop_chart(metrics)
        if chart is not None:
            flow.append(chart)
            flow.append(Spacer(1, 10))
            rendered = True

    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
        leaderboard = _conviction_leaderboard(edge_df, styles)
        if leaderboard:
            flow.extend(leaderboard)
            flow.append(Spacer(1, 12))
            rendered = True

        chart = _diverging_edge_chart(edge_df)
        if chart is not None:
            flow.append(chart)
            flow.append(Spacer(1, 10))
            rendered = True

        hist = _edge_distribution_chart(edge_df)
        panel = _slate_summary_panel(edge_df, metrics, styles)
        if hist is not None:
            row = Table(
                [[hist, panel]],
                colWidths=[3.5 * inch, 3.55 * inch],
            )
            row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            flow.append(row)
            flow.append(Spacer(1, 6))
            rendered = True

    if not rendered:
        flow.append(_para("Insufficient data to render charts.", styles["muted"]))
    return flow


# ── Per-player visual building blocks ───────────────────────────────────────
def _player_form_sparklines(
    games: pd.DataFrame | None,
    *,
    last_n: int = 12,
    width: float = 7.0 * inch,
    height: float = 1.35 * inch,
) -> Drawing | None:
    """Three small sparklines (PTS / REB / AST) of the player's recent games.

    ``games`` is expected to have columns ``game_date``, ``pts``, ``reb``,
    ``ast``. Missing columns are skipped. Returns ``None`` if there is not
    enough data to plot.
    """
    if not isinstance(games, pd.DataFrame) or games.empty:
        return None

    df = games.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"]).sort_values("game_date")
    df = df.tail(last_n)
    if len(df) < 3:
        return None

    panels: list[tuple[str, str, colors.Color]] = [
        ("PTS", "pts", BRAND_ORANGE),
        ("REB", "reb", NEUTRAL_BLUE),
        ("AST", "ast", POS_GREEN),
    ]
    panels = [p for p in panels if p[1] in df.columns]
    if not panels:
        return None

    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 12, "Recent form  |  last games",
        fontName=_BOLD_FONT, fontSize=9.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 24, "Trend lines    The dot marks the most recent game.",
        fontName=_BODY_FONT, fontSize=7.5, fillColor=INK_MUTED,
    ))

    panel_w = (width - (len(panels) - 1) * 10) / len(panels)
    plot_y = 0.18 * inch
    plot_h = height - 0.85 * inch

    for i, (label, col, color) in enumerate(panels):
        series = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        if len(series) < 2:
            continue
        x0 = i * (panel_w + 10)
        # Panel background
        drawing.add(Rect(
            x0, plot_y, panel_w, plot_h,
            fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4,
        ))
        # Label + latest value
        drawing.add(String(
            x0 + 6, plot_y + plot_h - 11,
            f"{label}",
            fontName=_BOLD_FONT, fontSize=8, fillColor=INK_MUTED,
        ))
        avg = sum(series) / len(series)
        last = series[-1]
        drawing.add(String(
            x0 + panel_w - 6, plot_y + plot_h - 11,
            f"avg {avg:.1f}    last {last:.1f}",
            fontName=_BODY_FONT, fontSize=7, fillColor=INK_MUTED,
            textAnchor="end",
        ))

        plot_x0 = x0 + 8
        plot_x1 = x0 + panel_w - 8
        plot_y0 = plot_y + 6
        plot_y1 = plot_y + plot_h - 18
        lo = min(series)
        hi = max(series)
        rng = (hi - lo) or 1.0
        n = len(series)
        pts = []
        for j, v in enumerate(series):
            px = plot_x0 + (plot_x1 - plot_x0) * (j / max(n - 1, 1))
            py = plot_y0 + (plot_y1 - plot_y0) * ((v - lo) / rng)
            pts.append((px, py))

        # Average reference line
        avg_y = plot_y0 + (plot_y1 - plot_y0) * ((avg - lo) / rng)
        drawing.add(Line(
            plot_x0, avg_y, plot_x1, avg_y,
            strokeColor=INK_FAINT, strokeWidth=0.4, strokeDashArray=[1.5, 2],
        ))

        # Trend line
        for j in range(1, len(pts)):
            drawing.add(Line(
                pts[j - 1][0], pts[j - 1][1], pts[j][0], pts[j][1],
                strokeColor=color, strokeWidth=1.4,
            ))
        # Dot on most recent point
        drawing.add(Circle(
            pts[-1][0], pts[-1][1], 2.6,
            fillColor=color, strokeColor=WHITE, strokeWidth=0.8,
        ))

    return drawing


def _player_hit_rate_donut(
    history_df: pd.DataFrame | None,
    *,
    diameter: float = 1.25 * inch,
) -> Drawing | None:
    """Donut chart showing Over / Under / Push split from resolved history."""
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return None
    if "result" not in history_df.columns:
        return None
    counts = (
        history_df["result"].astype(str).str.lower().value_counts()
        if not history_df.empty else pd.Series(dtype=int)
    )
    overs = int(counts.get("over", 0))
    unders = int(counts.get("under", 0))
    pushes = int(counts.get("push", 0))
    total = overs + unders + pushes
    if total == 0:
        return None

    width = 2.75 * inch
    height = 1.5 * inch
    drawing = Drawing(width, height)
    cx = 0.78 * inch
    cy = height / 2
    r_outer = diameter / 2
    r_inner = r_outer * 0.58

    # Render donut as concentric arcs using thin Lines (ReportLab Wedge would
    # also work but Drawing keeps deps minimal).
    import math
    segments = [
        (overs, POS_GREEN),
        (unders, NEG_RED),
        (pushes, INK_FAINT),
    ]
    angle = -math.pi / 2  # start at top
    step_n = 64
    for count, color in segments:
        if count <= 0:
            continue
        sweep = (count / total) * 2 * math.pi
        a = angle
        for s in range(step_n):
            t0 = a + (sweep * s / step_n)
            t1 = a + (sweep * (s + 1) / step_n)
            x0o = cx + r_outer * math.cos(t0)
            y0o = cy + r_outer * math.sin(t0)
            x1o = cx + r_outer * math.cos(t1)
            y1o = cy + r_outer * math.sin(t1)
            x0i = cx + r_inner * math.cos(t0)
            y0i = cy + r_inner * math.sin(t0)
            x1i = cx + r_inner * math.cos(t1)
            y1i = cy + r_inner * math.sin(t1)
            from reportlab.graphics.shapes import Polygon
            drawing.add(Polygon(
                points=[x0o, y0o, x1o, y1o, x1i, y1i, x0i, y0i],
                fillColor=color, strokeColor=color, strokeWidth=0.2,
            ))
        angle += sweep

    # Center text — Over hit rate
    over_pct = overs / max(overs + unders, 1) * 100
    drawing.add(String(
        cx, cy + 2, f"{over_pct:.0f}%",
        fontName=_BOLD_FONT, fontSize=14, fillColor=INK_DARK,
        textAnchor="middle",
    ))
    drawing.add(String(
        cx, cy - 12, "OVER RATE",
        fontName=_BOLD_FONT, fontSize=6.5, fillColor=INK_MUTED,
        textAnchor="middle",
    ))

    # Legend
    legend_x = cx + r_outer + 0.22 * inch
    items = [
        (POS_GREEN, f"Over  {overs}"),
        (NEG_RED, f"Under  {unders}"),
    ]
    if pushes:
        items.append((INK_FAINT, f"Push  {pushes}"))
    for j, (clr, lbl) in enumerate(items):
        ly = cy + 16 - j * 14
        drawing.add(Rect(
            legend_x, ly - 4, 8, 8,
            fillColor=clr, strokeColor=clr,
        ))
        drawing.add(String(
            legend_x + 13, ly - 1, lbl,
            fontName=_BODY_FONT, fontSize=8, fillColor=INK_BODY,
        ))
    return drawing


def _risk_chips(
    *,
    player: str,
    edge_df: pd.DataFrame | None,
    games: pd.DataFrame | None,
    recent_form: dict[str, float] | None,
    history_df: pd.DataFrame | None,
) -> list[tuple[str, colors.Color]]:
    """Compute small risk / signal chips shown above the per-player rationale."""
    chips: list[tuple[str, colors.Color]] = []

    # Back-to-back: most recent game within <=1 day of the prior.
    if isinstance(games, pd.DataFrame) and "game_date" in games.columns and len(games) >= 2:
        gd = pd.to_datetime(games["game_date"], errors="coerce").dropna().sort_values()
        if len(gd) >= 2:
            gap = (gd.iloc[-1] - gd.iloc[-2]).days
            if 0 <= gap <= 1:
                chips.append(("BACK-TO-BACK", NEG_RED))
            elif gap >= 4:
                chips.append((f"{int(gap)}-DAY REST", POS_GREEN))

    # Books depth check — thin markets are riskier.
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "books" in edge_df.columns:
        sub = edge_df[edge_df.get("player") == player]
        if not sub.empty:
            books = pd.to_numeric(sub["books"], errors="coerce").dropna()
            if not books.empty:
                if books.max() <= 3:
                    chips.append(("THIN MARKET", GOLD))
                elif books.min() >= 6:
                    chips.append(("DEEP MARKET", NEUTRAL_BLUE))

    # Minutes volatility / availability proxy.
    if isinstance(games, pd.DataFrame) and "min" in games.columns:
        m = pd.to_numeric(games["min"], errors="coerce").dropna().tail(5)
        if not m.empty and m.mean() < 22:
            chips.append(("LIMITED MIN", GOLD))

    # Form vs season: recent form spike or slump on points.
    if isinstance(games, pd.DataFrame) and "pts" in games.columns and recent_form:
        season = pd.to_numeric(games["pts"], errors="coerce").dropna()
        recent = recent_form.get("pts")
        if recent is not None and len(season) >= 10:
            base = float(season.tail(min(len(season), 30)).mean())
            if base > 0:
                delta = recent - base
                if delta >= max(2.0, base * 0.10):
                    chips.append(("HOT FORM", POS_GREEN))
                elif delta <= -max(2.0, base * 0.10):
                    chips.append(("COLD FORM", NEG_RED))

    # Historical Over/Under bias.
    if isinstance(history_df, pd.DataFrame) and not history_df.empty and "result" in history_df.columns:
        cnt = history_df["result"].astype(str).str.lower().value_counts()
        overs = int(cnt.get("over", 0))
        unders = int(cnt.get("under", 0))
        total = overs + unders
        if total >= 4:
            rate = overs / total
            if rate >= 0.65:
                chips.append((f"OVER-BIASED  {rate*100:.0f}%", POS_GREEN))
            elif rate <= 0.35:
                chips.append((f"UNDER-BIASED  {(1-rate)*100:.0f}%", NEG_RED))

    # Volatility + role stability — variance and usage-swing posture.
    vol = _volatility_label(games)
    if vol is not None:
        chips.append(vol)
    role = _role_stability_label(games)
    if role is not None:
        chips.append(role)

    return chips


def _risk_chip_strip(chips: list[tuple[str, colors.Color]], styles: dict) -> Table | None:
    if not chips:
        return None
    # Chips render as quiet pills: a colored leading dot + dark label on a
    # faint panel background. Far less visually loud than the previous
    # filled buttons that competed with the orange hero band above them.
    cells: list = []
    for label, color in chips[:6]:
        hex_str = color.hexval()[2:] if hasattr(color, "hexval") else "11151c"
        para = Paragraph(
            f"<font size='9' color='#{hex_str}'>●</font>"
            f"&nbsp;&nbsp;<font size='7' color='#11151c'><b>{label}</b></font>",
            ParagraphStyle(
                "chip", parent=styles["body"], leading=10, alignment=TA_CENTER,
            ),
        )
        inner = Table([[para]], colWidths=[1.05 * inch])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        cells.append(inner)
    while len(cells) < 6:
        cells.append(Paragraph("", styles["body"]))
    strip = Table([cells], colWidths=[1.13 * inch] * 6)
    strip.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return strip


def _signal_summary_panel(
    *,
    player: str,
    edge_df: pd.DataFrame | None,
    recent_form: dict[str, float] | None,
    history_df: pd.DataFrame | None,
    styles: dict,
    width: float | None = None,
) -> Table | None:
    """Compact 'Why this signal' panel: top edge + form anchor + hit-rate.

    ``width`` (in points) lets callers expand the panel to full content
    width when there's no donut paired beside it; defaults to the original
    3.95" two-column width.
    """
    if not (isinstance(edge_df, pd.DataFrame) and not edge_df.empty):
        return None
    sub = edge_df[edge_df.get("player") == player].copy()
    if sub.empty or "edge" not in sub.columns:
        return None
    sub["abs_edge"] = pd.to_numeric(sub["edge"], errors="coerce").abs()
    sub = sub.sort_values("abs_edge", ascending=False)
    r = sub.iloc[0]

    edge_val = pd.to_numeric(r.get("edge"), errors="coerce")
    side = str(r.get("call") or r.get("side") or "").upper().split()[0] if (r.get("call") or r.get("side")) else (
        "MORE" if pd.notna(edge_val) and edge_val > 0 else "LESS"
    )
    grade, grade_color = _signal_grade(abs(float(edge_val)) if pd.notna(edge_val) else 0.0)
    side_color = POS_GREEN if side in ("MORE", "OVER") else NEG_RED

    market = str(r.get("model", "—"))
    line_v = r.get("posted line", r.get("line"))
    proj = r.get("model prediction", r.get("projection"))
    books = r.get("books")

    # Form anchor for the relevant stat
    stat_key = {
        "points": "pts", "rebounds": "reb", "assists": "ast",
        "pra": "pra", "threepm": "fg3m", "fantasy_score": "fantasy_score",
    }.get(market.lower(), None)
    form_val = (recent_form or {}).get(stat_key) if stat_key else None

    # Recent O/U record on this market specifically
    over_rec = "—"
    if isinstance(history_df, pd.DataFrame) and not history_df.empty and "metric" in history_df.columns:
        m = history_df[history_df["metric"].astype(str).str.lower() == market.lower()]
        if not m.empty and "result" in m.columns:
            cnt = m["result"].astype(str).str.lower().value_counts()
            o = int(cnt.get("over", 0))
            u = int(cnt.get("under", 0))
            if o + u > 0:
                over_rec = f"{o}-{u}"

    grade_hex = grade_color.hexval()[2:] if hasattr(grade_color, "hexval") else "ff7a18"
    side_hex = side_color.hexval()[2:] if hasattr(side_color, "hexval") else "1f9d6c"

    head = Paragraph(
        f"<font size='7.5' color='#cc5a00'><b>WHY THIS SIGNAL "
        f"&nbsp;|&nbsp; {player.upper()}</b></font><br/>"
        f"<font size='13' color='#11151c'><b>{_short(market.title(), 24)}</b></font> "
        f"<font size='9' color='#6b7686'>&nbsp;&nbsp;grade </font>"
        f"<font size='9' color='#{grade_hex}'><b>{grade.upper()}</b></font>",
        styles["body"],
    )
    book_names_val = r.get("book_names") if "book_names" in r else None
    books_label = _abbrev_books(book_names_val)
    if books_label == "—" and pd.notna(books):
        books_label = f"{int(books)} books"
    rows = [
        ("Signal", f"<font color='#{side_hex}'><b>{_side_display(side)}</b></font>"),
        ("Edge", f"<b>{_fmt_signed(edge_val, 2)}</b>"),
        ("Posted line", _fmt(line_v, 1)),
        ("Model proj.", _fmt(proj, 2)),
        ("Recent form", _fmt(form_val, 1) if form_val is not None else "—"),
        ("Books", books_label),
        (f"O/U on {market.lower()}", over_rec),
    ]
    body_cells: list[list] = [[
        Paragraph(
            f"<font size='7' color='#6b7686'>{lbl.upper()}</font>",
            styles["muted"],
        ),
        Paragraph(
            f"<font size='9.5' color='#11151c'>{val}</font>",
            styles["body"],
        ),
    ] for lbl, val in rows]
    body = Table(body_cells, colWidths=[1.45 * inch, 2.45 * inch])
    body.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, PANEL_BORDER),
        ("BOX", (0, 0), (-1, -1), 0.25, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    panel_w = width if width is not None else (3.95 * inch)
    # Re-flow the inner body table to match the outer panel width.
    if width is not None:
        label_w = 1.6 * inch
        value_w = panel_w - label_w - (2 * 8)  # match LEFT/RIGHTPADDING
        body = Table(body_cells, colWidths=[label_w, value_w])
        body.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, PANEL_BORDER),
            ("BOX", (0, 0), (-1, -1), 0.25, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

    panel = Table([[head], [body]], colWidths=[panel_w])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BRAND_ORANGE_SOFT),
        ("LINEBELOW", (0, 0), (0, 0), 1.2, BRAND_ORANGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, 0), (0, 0), 6),
        ("TOPPADDING", (0, 1), (-1, 1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
    ]))
    return panel


def _player_history_table(
    player: str,
    history_df: pd.DataFrame | None,
    styles: dict,
    *,
    max_rows: int = 10,
) -> list:
    """Render a 'historical lines vs actual outcomes' table for a player.

    Expected ``history_df`` columns: ``game_date``, ``metric`` (e.g. 'points'),
    ``line``, ``actual``, ``result`` (Over/Under/Push), ``margin``. Only the
    most recent ``max_rows`` games (across markets) are shown to keep the
    PDF dense without overflowing.
    """
    header_block = [
        Spacer(1, 6),
        Paragraph(
            f"<font size='7.5' color='#cc5a00'><b>"
            f"{player.upper()} &nbsp;|&nbsp; HISTORICAL LINES vs OUTCOMES"
            f"</b></font>",
            styles["body"],
        ),
    ]

    if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return header_block + [
            Paragraph(
                "<font size='8' color='#6b7280'><i>No resolved historical "
                "lines for this player in the cached window. Backfill more "
                "dates in the Odds tab to populate.</i></font>",
                styles["body"],
            ),
        ]

    df = history_df.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("game_date", ascending=False)
    df = df.head(max_rows)
    if df.empty:
        return header_block + [
            Paragraph(
                "<font size='8' color='#6b7280'><i>No resolved historical "
                "lines for this player in the cached window.</i></font>",
                styles["body"],
            ),
        ]

    rows: list[list[Any]] = [
        ["Date", "Market", "Line", "Actual", "Margin", "Result"]
    ]
    style_extra: list = []
    hits = 0
    losses = 0
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        gd = r.get("game_date")
        date_str = gd.strftime("%b %d") if pd.notna(gd) else "—"
        metric = str(r.get("metric", "—")).title()
        line = _fmt(r.get("line"), 1)
        actual = _fmt(r.get("actual"), 1)
        margin = _fmt_signed(r.get("margin"), 1)
        result = str(r.get("result", "")).title() or "—"
        rows.append([date_str, metric, line, actual, margin, result])
        if result.lower() == "over":
            hits += 1
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
            style_extra.append(("FONTNAME", (5, i), (5, i), _BOLD_FONT))
            style_extra.append(("TEXTCOLOR", (4, i), (4, i), POS_GREEN))
        elif result.lower() == "under":
            losses += 1
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
            style_extra.append(("FONTNAME", (5, i), (5, i), _BOLD_FONT))
            style_extra.append(("TEXTCOLOR", (4, i), (4, i), NEG_RED))
        else:
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), INK_MUTED))

    table = _styled_table(
        rows,
        col_widths=[0.7 * inch, 1.1 * inch, 0.7 * inch, 0.75 * inch,
                    0.85 * inch, 0.7 * inch],
        align_right_cols=[2, 3, 4],
        header_bg=NEUTRAL_BLUE,
    )
    table.setStyle(TableStyle(style_extra))

    total_resolved = hits + losses
    summary_bits: list[str] = []
    if total_resolved:
        rate = hits / total_resolved * 100.0
        summary_bits.append(
            f"Recent Over/Under record: <b>{hits}-{losses}</b> "
            f"({rate:.0f}% Over)"
        )
    if "margin" in df.columns:
        margins = pd.to_numeric(df["margin"], errors="coerce").dropna()
        if not margins.empty:
            summary_bits.append(
                f"Avg margin vs line: <b>{margins.mean():+.2f}</b>"
            )
    summary = "   |   ".join(summary_bits) if summary_bits else (
        "Resolved outcomes unavailable for this window."
    )

    return header_block + [
        Paragraph(summary, styles["body"]),
        Spacer(1, 3),
        table,
    ]


def _styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    *,
    header_bg=BRAND_ORANGE,
    header_fg=WHITE,
    zebra: bool = True,
    align_right_cols: list[int] | None = None,
) -> Table:
    align_right_cols = align_right_cols or []
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmd = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_fg),
        ("FONTNAME", (0, 0), (-1, 0), _BOLD_FONT),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), _BODY_FONT),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK_BODY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BRAND_ORANGE_DEEP),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, PANEL_BORDER),
    ]
    for col in align_right_cols:
        cmd.append(("ALIGN", (col, 1), (col, -1), "RIGHT"))
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                cmd.append(("BACKGROUND", (0, r), (-1, r), PANEL_BG))
    t.setStyle(TableStyle(cmd))
    return t


# ── Section: cover ──────────────────────────────────────────────────────────
def _cover_meta_tile(label: str, value: str, styles: dict) -> Table:
    cell = Table(
        [
            [_para(label.upper(), styles["cover_meta_label"])],
            [_para(value, styles["cover_meta_value"])],
        ],
        colWidths=[1.55 * inch],
    )
    cell.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
        ("LINEBELOW", (0, 1), (0, 1), 1.4, BRAND_ORANGE),
    ]))
    return cell


def _cover_flowables(
    *,
    roster: dict[str, list[str]],
    meta: _ReportMeta,
    edge_df: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    styles: dict,
) -> list:
    seasons_set = sorted({s for ss in roster.values() for s in ss})
    seasons_label = ", ".join(seasons_set) if seasons_set else "—"

    edge_n = int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0
    median_r2_txt = "—"
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        if r2_col is not None:
            v = float(pd.to_numeric(metrics[r2_col], errors="coerce").median())
            if not pd.isna(v):
                median_r2_txt = f"{v:.2f}"

    tiles = Table(
        [[
            _cover_meta_tile("Players", str(meta.roster_count), styles),
            _cover_meta_tile("Live signals", str(edge_n), styles),
            _cover_meta_tile("Median R²", median_r2_txt, styles),
        ]],
        colWidths=[1.6 * inch, 1.6 * inch, 1.6 * inch],
    )
    tiles.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    flow: list = [
        Spacer(1, 1.4 * inch),
        _para("HOOPLYTICS  |  ROSTER ANALYTICS", styles["cover_eyebrow"]),
        _para("Tonight's Scouting Report.", styles["cover_title"]),
        _para(
            "Projection gaps, model leans, and the loudest signals from your "
            "tracked roster — straight off the Hooplytics engine.",
            styles["cover_sub"],
        ),
        Spacer(1, 0.5 * inch),
        tiles,
        Spacer(1, 0.35 * inch),
        _para(f"Generated  |  {meta.generated_at}", styles["cover_tag"]),
        _para(f"Seasons  |  {seasons_label}", styles["cover_tag"]),
        _para(
            "Prose  |  " + ("AI-augmented (OpenAI)" if meta.has_ai else "data-only"),
            styles["cover_tag"],
        ),
    ]
    return flow


# ── Section: executive summary + spotlight ──────────────────────────────────
def _callout_box(text: str, styles: dict, *, accent: colors.Color = BRAND_ORANGE) -> Table:
    body = _para(text, styles["callout"])
    t = Table([[body]], colWidths=[7.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_ORANGE_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _bottom_line_flowables(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    styles: dict,
) -> list:
    """Bottom-line-up-front (BLUF) panel.

    A scannable, hero-style verdict block that lands BEFORE the executive
    summary so a reader knows the headline takeaway in three seconds:
    - the loudest signal (top conviction play),
    - which way the slate is leaning,
    - how much to trust the models tonight,
    - and a watch-out flagging the noisiest market.
    """
    # ── Compute the four pillar facts ─────────────────────────────────────
    top_label = "No live signals yet"
    top_detail = "Connect odds + run the model to populate the leaderboard."
    top_color = INK_FAINT
    top_edge_str = "—"
    top_line: Any = None
    top_proj: Any = None
    top_market = ""

    lean_label = "BALANCED"
    lean_detail = "No directional pressure across the slate."
    lean_color = NEUTRAL_BLUE

    conf_label = "—"
    conf_detail = "Model metrics unavailable."
    conf_color = INK_FAINT

    risk_label: str | None = None
    risk_detail = ""
    risk_color = NEG_RED

    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "edge" in edge_df.columns:
        df = edge_df.copy()
        df["edge_n"] = pd.to_numeric(df["edge"], errors="coerce")
        df["books_n"] = pd.to_numeric(df.get("books"), errors="coerce").fillna(1)
        df = df.dropna(subset=["edge_n"])
        if not df.empty:
            df["abs_edge"] = df["edge_n"].abs()
            df["conviction"] = df["abs_edge"] * df["books_n"].clip(lower=1).pow(0.5)
            top = df.sort_values("conviction", ascending=False).iloc[0]
            top_edge_val = float(top["edge_n"])
            top_side = str(top.get("side") or top.get("call") or "").upper().split()[0]
            if not top_side:
                top_side = "MORE" if top_edge_val > 0 else "LESS"
            top_color = POS_GREEN if top_side in ("MORE", "OVER") else NEG_RED
            # Use the full player name; the headline Paragraph wraps inside
            # the 5" left block and auto-shrinks for very long names so we
            # never render a "SHAI GILGEOUS-ALEXAND..." truncation.
            top_label = str(top.get("player", "—")).upper()
            top_detail = (
                f"{str(top.get('model', '')).upper()} {_side_display(top_side)} "
                f"{_fmt_signed(top_edge_val, 2)} across "
                f"{int(top['books_n'])} books."
            )
            top_edge_str = _fmt_signed(top_edge_val, 2)
            top_line = top.get("posted line", top.get("line"))
            top_proj = top.get("model prediction", top.get("projection"))
            top_market = str(top.get("model", "—")).upper()

            # Slate posture
            label, pos, neg = _slate_lean_label(edge_df)
            total = max(pos + neg, 1)
            if "MORE" in label:
                lean_label = "ABOVE-LEANING"
                lean_color = POS_GREEN
                lean_detail = f"{pos} of {total} live signals point ABOVE the line."
            elif "LESS" in label:
                lean_label = "BELOW-LEANING"
                lean_color = NEG_RED
                lean_detail = f"{neg} of {total} live signals point BELOW the line."
            else:
                lean_detail = f"{pos} ABOVE vs {neg} BELOW — no clear directional signal."

            # Risk: flag the loudest contrarian edge with thin market depth.
            shallow = df[df["books_n"] <= 4].sort_values("abs_edge", ascending=False)
            if not shallow.empty:
                r = shallow.iloc[0]
                risk_label = "WATCH OUT"
                risk_detail = (
                    f"{_short(str(r.get('player', '—')), 22)} "
                    f"{str(r.get('model', '')).upper()} edge "
                    f"{_fmt_signed(float(r['edge_n']), 2)} sits on only "
                    f"{int(r['books_n'])} book(s) — thin liquidity, "
                    f"treat as a tell, not a thesis."
                )

    # Model confidence
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        if r2_col is not None:
            r2 = pd.to_numeric(metrics[r2_col], errors="coerce").dropna()
            if not r2.empty:
                med = float(r2.median())
                if med >= 0.50:
                    conf_label = f"STRONG | {med:.2f}"
                    conf_color = POS_GREEN
                    conf_detail = "Model fit is solid — trust the projections."
                elif med >= 0.30:
                    conf_label = f"MIXED | {med:.2f}"
                    conf_color = GOLD
                    conf_detail = "Moderate fit — combine with context, not blindly."
                else:
                    conf_label = f"NOISY | {med:.2f}"
                    conf_color = NEG_RED
                    conf_detail = "Low fit tonight — small edges may be noise."

    # ── Headline strip (dark band) ────────────────────────────────────────
    eyebrow = Paragraph(
        "<font size='8' color='#ff7a18'><b>BOTTOM LINE UP FRONT</b></font>",
        styles["body"],
    )
    # Auto-shrink the headline so long player names (e.g. "SHAI
    # GILGEOUS-ALEXANDER") fit on a single line inside the 5" left block
    # instead of getting truncated with an ellipsis.
    title_size = 16 if len(top_label) <= 18 else (14 if len(top_label) <= 24 else 12)
    title_para = Paragraph(
        f"<font size='{title_size}' color='#ffffff'><b>{top_label}</b></font>",
        styles["body"],
    )
    detail_para = Paragraph(
        f"<font size='10' color='#e1e4ea'>{top_detail}</font>",
        styles["body"],
    )
    left_block = Table(
        [[eyebrow], [title_para], [detail_para]],
        colWidths=[5.0 * inch],
    )
    left_block.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 4),
        ("BOTTOMPADDING", (0, 1), (0, 1), 4),
        ("BOTTOMPADDING", (0, 2), (0, 2), 0),
    ]))

    chip_eyebrow = Paragraph(
        "<font size='8' color='#9aa3b2'><b>TOP SIGNAL</b></font>",
        styles["body"],
    )
    chip_value_color = top_color.hexval()[2:] if top_edge_str != "—" else "9aa3b2"
    chip_value = Paragraph(
        f"<font size='24' color='#{chip_value_color}'><b>{top_edge_str}</b></font>",
        styles["body"],
    )
    edge_chip = Table(
        [[chip_eyebrow], [chip_value]],
        colWidths=[2.0 * inch],
    )
    edge_chip.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 6),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
    ]))

    headline = Table(
        [[left_block, edge_chip]],
        colWidths=[5.0 * inch, 2.0 * inch],
    )
    headline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))

    # ── Three insight tiles ───────────────────────────────────────────────
    def _tile(eyebrow_text: str, label: str, detail: str, accent: colors.Color) -> Table:
        accent_hex = accent.hexval()[2:]
        body = Paragraph(
            f"<font size='7' color='#9aa3b2'><b>{eyebrow_text}</b></font><br/>"
            f"<font size='12' color='#{accent_hex}'><b>{label}</b></font><br/>"
            f"<font size='8' color='#2b3340'>{detail}</font>",
            styles["body"],
        )
        t = Table([[body]], colWidths=[2.30 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LINEABOVE", (0, 0), (-1, 0), 3, accent),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return t

    # MODEL VS LINE — the actionable comparison the headline implies but
    # never spells out. Replaces the redundant "THE PLAY" tile that just
    # echoed the headline word-for-word.
    if top_line is not None and top_proj is not None and top_edge_str != "—":
        try:
            line_str = _fmt(top_line, 1)
            proj_str = _fmt(top_proj, 2)
            mvl_label = f"{proj_str}  vs  {line_str}"
            mvl_detail = (
                f"Model projects {proj_str} on the {top_market.lower()} line of "
                f"{line_str}."
            )
        except Exception:
            mvl_label = top_edge_str
            mvl_detail = "Model vs market gap on the loudest edge."
    else:
        mvl_label = "—"
        mvl_detail = "No mapped line available for the loudest edge."
    tile_play = _tile("MODEL  vs  LINE", mvl_label, mvl_detail, top_color)
    tile_lean = _tile("SLATE POSTURE", lean_label, lean_detail, lean_color)
    tile_conf = _tile("MODEL CONFIDENCE", conf_label, conf_detail, conf_color)

    tiles = Table(
        [[tile_play, tile_lean, tile_conf]],
        colWidths=[2.34 * inch, 2.34 * inch, 2.34 * inch],
    )
    tiles.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    flow: list = [headline, Spacer(1, 8), tiles]

    # ── Watch-out risk strip (only when we have a thin-market signal) ─────
    if risk_label:
        risk = Table([[Paragraph(
            f"<font size='7.5' color='#d24545'><b>{risk_label}</b></font>&nbsp;&nbsp;"
            f"<font size='9' color='#2b3340'>{risk_detail}</font>",
            styles["body"],
        )]], colWidths=[7.0 * inch])
        risk.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, risk_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.extend([Spacer(1, 8), risk])

    flow.append(Spacer(1, 14))
    return flow


def _quick_calls_flowables(
    *,
    edge_df: pd.DataFrame | None,
    recent_form: dict[str, dict[str, float]] | None,
    styles: dict,
    sigma_lookup: dict[str, float] | None = None,
) -> list:
    """At-a-glance call sheet: every more/less call with a one-line why.

    Designed to be the section a skimming reader lands on first. Renders a
    compact card grid (one card per edge) sorted by absolute edge, each
    showing player + market + side + posted line + model projection +
    a deterministic one-line rationale (book depth + form delta vs the line).
    """
    flow: list = _section_header(
        "Signal summary  |  top 5", "Section 01", styles, anchor="sec-quick-calls",
    )
    flow.append(_para(
        "Skim-friendly summary \u2014 the top 5 above/below-line signals from the live "
        "board, sorted by signal strength.",
        styles["muted"],
    ))
    flow.append(Spacer(1, 6))

    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        flow.append(_para(
            "No live signals yet. Add an Odds API key and refresh lines to "
            "populate this section.",
            styles["muted"],
        ))
        return flow

    df = edge_df.copy()
    if "edge" not in df.columns:
        flow.append(_para("No edge values available.", styles["muted"]))
        return flow
    df["abs_edge"] = pd.to_numeric(df["edge"], errors="coerce").abs()
    df = df.sort_values("abs_edge", ascending=False).head(5)
    if df.empty:
        flow.append(_para("No callable edges in the current snapshot.", styles["muted"]))
        return flow

    recent_form = recent_form or {}

    def _one_line_why(row: pd.Series) -> str:
        """Deterministic one-liner combining edge size + form-vs-line delta."""
        try:
            edge_val = float(row.get("edge"))
        except (TypeError, ValueError):
            edge_val = 0.0
        try:
            line_val = float(row.get("posted line", row.get("line")))
        except (TypeError, ValueError):
            line_val = float("nan")
        side = str(row.get("call") or row.get("side") or "").upper().split()[0]
        side = side or ("MORE" if edge_val > 0 else "LESS")
        market = str(row.get("model", "")).lower()
        form_map = recent_form.get(str(row.get("player", "")), {}) or {}
        # Map market \u2192 recent-form key. Fall back gracefully.
        stat_key = {
            "points": "pts", "rebounds": "reb", "assists": "ast",
            "pra": "pra", "threepm": "fg3m",
            "stl_blk": "stl_blk", "turnovers": "tov",
            "fantasy_score": "fantasy_score",
        }.get(market)
        form_val = None
        if stat_key:
            form_val = form_map.get(stat_key)
            if form_val is None:
                form_val = form_map.get(stat_key + "_l5") or form_map.get(stat_key + "_l10")
        try:
            books = int(pd.to_numeric(row.get("books"), errors="coerce"))
        except (TypeError, ValueError):
            books = 0
        depth = "deep" if books >= 7 else ("moderate" if books >= 4 else "thin")
        bits: list[str] = [f"|edge| {abs(edge_val):.2f} on {depth} market ({books} books)"]
        if form_val is not None and not pd.isna(form_val) and not pd.isna(line_val):
            try:
                form_f = float(form_val)
                gap = form_f - line_val
                direction = "above" if gap > 0 else "below"
                bits.append(f"recent form {form_f:.1f} runs {abs(gap):.1f} {direction} the {line_val:.1f} line")
            except (TypeError, ValueError):
                pass
        return " \u2022 ".join(bits)

    cards: list = []
    for _, row in df.iterrows():
        edge_val = pd.to_numeric(row.get("edge"), errors="coerce")
        if pd.isna(edge_val):
            continue
        side = str(row.get("call") or row.get("side") or "").upper().split()[0]
        side = side or ("MORE" if edge_val > 0 else "LESS")
        side_color = "#1f9d6c" if side in ("MORE", "OVER") else "#d24545"
        side_bg = "#e9f7f0" if side in ("MORE", "OVER") else "#fbecec"

        pill = Paragraph(
            f"<font size='8.5' color='{side_color}'><b>{_side_display(side)}</b></font>",
            ParagraphStyle("qc_side", parent=styles["body"], alignment=1),
        )
        pill_tbl = Table([[pill]], colWidths=[0.65 * inch])
        pill_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(side_bg)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(side_color)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

        line_str = _fmt(row.get("posted line", row.get("line")), 1)
        proj_str = _fmt(row.get("model prediction", row.get("projection")), 2)
        edge_str = _fmt_signed(edge_val, 2)

        title = Paragraph(
            f"<font size='10.5' color='#11151c'><b>"
            f"{_short(row.get('player', '—'), 26)}</b></font>"
            f"&nbsp;&nbsp;<font size='9' color='#6b7686'>"
            f"{_short(row.get('model', ''), 14)}</font>",
            styles["body"],
        )
        line_block = Paragraph(
            f"<font size='8' color='#6b7686'>LINE</font> "
            f"<font size='10' color='#11151c'><b>{line_str}</b></font>"
            f"&nbsp;&nbsp;<font size='8' color='#6b7686'>MODEL</font> "
            f"<font size='10' color='#11151c'><b>{proj_str}</b></font>"
            f"&nbsp;&nbsp;<font size='8' color='#6b7686'>EDGE</font> "
            f"<font size='10' color='{side_color}'><b>{edge_str}</b></font>",
            styles["body"],
        )

        # Probabilistic stats line: HIT% (clears the line) + CONF (composite).
        sigma = (sigma_lookup or {}).get(str(row.get("model", "")))
        hit_p = _hit_probability(
            row.get("model prediction", row.get("projection")),
            row.get("posted line", row.get("line")),
            side, sigma,
        )
        conf = _confidence_score(edge_val, sigma, row.get("books"))
        if hit_p is not None or conf is not None:
            hit_str = _fmt_pct(hit_p)
            conf_str = f"{conf}/100" if conf is not None else "—"
            stats_block = Paragraph(
                f"<font size='8' color='#6b7686'>HIT%</font> "
                f"<font size='10' color='{side_color}'><b>{hit_str}</b></font>"
                f"&nbsp;&nbsp;<font size='8' color='#6b7686'>CONF</font> "
                f"<font size='10' color='#11151c'><b>{conf_str}</b></font>",
                styles["body"],
            )
        else:
            stats_block = None

        why = Paragraph(
            f"<font size='8.5' color='#3a4250'>"
            f"{_safe_text(_one_line_why(row))}</font>",
            styles["body"],
        )

        body_rows: list = [[title], [line_block]]
        if stats_block is not None:
            body_rows.append([stats_block])
        body_rows.append([why])
        body_tbl = Table(
            body_rows,
            colWidths=[5.5 * inch],
        )
        body_tbl.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        card = Table(
            [[pill_tbl, body_tbl]],
            colWidths=[0.85 * inch, 5.95 * inch],
        )
        card.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(side_color)),
        ]))
        cards.append([card])

    if cards:
        wrap = Table(cards, colWidths=[6.8 * inch])
        wrap.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        flow.append(wrap)
    flow.append(Spacer(1, 10))
    return flow


def _executive_summary_flowables(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    ai_sections: dict[str, Any] | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Slate brief", "Section 02", styles, anchor="sec-exec")
    flow.append(_callout_box(
        _deterministic_summary_text(roster=roster, metrics=metrics, edge_df=edge_df),
        styles,
    ))
    flow.append(Spacer(1, 6))

    ai_text = str((ai_sections or {}).get("executive_summary", "")).strip()
    outlook = str((ai_sections or {}).get("slate_outlook", "")).strip()
    if ai_text or outlook:
        # AI prose lives in a styled accent panel so the reader can tell at a
        # glance which prose came from the model vs the deterministic summary
        # callout above. A left rule + AI eyebrow badge keeps it on-brand.
        prose_html_parts: list[str] = []
        if ai_text:
            prose_html_parts.append(
                f"<font size='10' color='#2b3340'>{_safe_text(ai_text)}</font>"
            )
        if outlook:
            prose_html_parts.append(
                f"<font size='10' color='#2b3340'>{_safe_text(outlook)}</font>"
            )
        prose_html = "<br/><br/>".join(prose_html_parts)
        ai_panel = Table(
            [[Paragraph(
                "<font size='8' color='#cc5a00'><b>AI&nbsp;SCOUT</b></font>"
                "&nbsp;&nbsp;"
                "<font size='8' color='#6b7686'><b>CONTEXT NARRATIVE</b></font>"
                f"<br/><br/>{prose_html}",
                styles["body"],
            )]],
            colWidths=[7.0 * inch],
        )
        ai_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8f1")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_ORANGE),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        flow.append(ai_panel)
    return flow


def _ai_picks_flowables(
    *,
    roster: dict[str, list[str]],
    edge_df: pd.DataFrame | None,
    ai_sections: dict[str, Any] | None,
    sigma_lookup: dict[str, float] | None,
    styles: dict,
) -> list:
    """Aggregate the per-player AI scout output into a single readable board.

    Renders one card per rostered player with:
      * the AI's concrete "more / less / no play" prediction line
      * the loudest model-vs-line gap pulled from ``edge_df`` for context
      * the AI's news + rationale prose

    The deterministic edge data anchors each card so the reader can see *why*
    the AI is leaning a particular direction without flipping back to other
    sections. AI prose lives in tinted accent panels with an "AI Scout" badge
    so it never gets confused with deterministic content.
    """
    flow: list = _section_header(
        "AI scout picks  |  top 3", "Section 03", styles, anchor="sec-ai-picks",
    )

    ai_players: dict[str, Any] = (ai_sections or {}).get("players") or {}
    if not roster or not ai_players:
        flow.append(_para(
            "Connect an OpenAI key in the sidebar to populate the AI scout "
            "section. Every other section in this report still works without "
            "it — only the AI prose is gated.",
            styles["muted"],
        ))
        return flow

    # Pre-compute "loudest signal" per player so each card can show the
    # primary edge the AI is reasoning about, not just unanchored prose.
    loudest_by_player: dict[str, dict] = {}
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
        df = edge_df.copy()
        if "abs_edge" not in df.columns and "edge" in df.columns:
            df["abs_edge"] = df["edge"].abs()
        df = df.sort_values("abs_edge", ascending=False)
        for player_name in roster.keys():
            sub = df[df["player"] == player_name]
            if sub.empty:
                continue
            top = sub.iloc[0]
            loudest_by_player[player_name] = {
                "model": str(top.get("model", "")),
                "line": top.get("posted line", top.get("line")),
                "proj": top.get("model prediction", top.get("projection")),
                "edge": top.get("edge"),
                "abs_edge": abs(float(top.get("edge"))) if top.get("edge") is not None else 0.0,
                "side": str(top.get("call") or top.get("side") or "").upper(),
            }

    # Rank players by loudest absolute edge and keep only the top 5 so the
    # section stays skim-friendly. Players without any edge fall to the end
    # and are dropped past the cap.
    ranked_players = sorted(
        roster.keys(),
        key=lambda p: loudest_by_player.get(p, {}).get("abs_edge", -1.0),
        reverse=True,
    )
    top_players = [p for p in ranked_players if p in loudest_by_player][:3]
    # If nothing has an edge, fall back to first 3 roster players so the
    # section isn't empty when AI prose still has value.
    if not top_players:
        top_players = list(roster.keys())[:3]

    intro = _callout_box(
        "Hooplytics Scout's three loudest more/less calls for tonight, ranked "
        "by model-vs-line edge. Each card pairs the AI's concrete pick with "
        "the deterministic signal that anchors it. The full ranked board "
        "lives in Section 07.",
        styles,
    )
    flow.append(intro)
    flow.append(Spacer(1, 8))

    for player_name in top_players:
        ai_entry = ai_players.get(player_name, "")
        if isinstance(ai_entry, dict):
            news = str(ai_entry.get("news", "")).strip()
            prediction = str(ai_entry.get("prediction", "")).strip()
            rationale = str(ai_entry.get("rationale", "")).strip()
        else:
            news = prediction = ""
            rationale = str(ai_entry or "").strip()

        if not (news or prediction or rationale):
            continue

        loud = loudest_by_player.get(player_name)
        # Header row: player name on the left, AI pick chip on the right.
        edge_chip_html = ""
        side_color = INK_MUTED
        if loud is not None:
            try:
                ev = float(loud.get("edge")) if loud.get("edge") is not None else None
            except (TypeError, ValueError):
                ev = None
            side = loud.get("side", "")
            if side in ("MORE", "OVER"):
                side_color = POS_GREEN
                side_label = "ABOVE"
            elif side in ("LESS", "UNDER"):
                side_color = NEG_RED
                side_label = "BELOW"
            else:
                side_label = "NEUTRAL"
            edge_str = _fmt_signed(ev, 2) if ev is not None else "—"
            edge_chip_html = (
                f"<font size='7.5' color='#6b7686'><b>LOUDEST SIGNAL</b></font>"
                f"<br/>"
                f"<font size='9' color='#{side_color.hexval()[2:]}'>"
                f"<b>{loud.get('model','').upper()} {side_label} {edge_str}</b>"
                f"</font><br/>"
                f"<font size='7.5' color='#6b7686'>"
                f"line {_fmt(loud.get('line'), 1)} &middot; "
                f"model {_fmt(loud.get('proj'), 2)}"
                f"</font>"
            )
        else:
            edge_chip_html = (
                "<font size='7.5' color='#6b7686'><b>LOUDEST SIGNAL</b></font>"
                "<br/><font size='8.5' color='#9aa3b2'>No live signal</font>"
            )

        header_left = Paragraph(
            "<font size='7.5' color='#cc5a00'><b>AI&nbsp;SCOUT&nbsp;PICK</b></font>"
            f"<br/><font size='13' color='#11151c'><b>{_safe_text(player_name)}</b></font>",
            styles["body"],
        )
        header_right = Paragraph(edge_chip_html, ParagraphStyle(
            "ai_pick_right", parent=styles["body"], alignment=TA_LEFT,
        ))
        header_row = Table(
            [[header_left, header_right]],
            colWidths=[3.6 * inch, 3.4 * inch],
        )
        header_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        # Pick line — the concrete more/less call.
        pick_html = (
            f"<font size='7.5' color='#cc5a00'><b>THE PICK</b></font>"
            f"<br/><font size='11' color='#11151c'><b>"
            f"{_safe_text(prediction or 'No play — see analyst notes')}"
            f"</b></font>"
        )
        pick_para = Paragraph(pick_html, styles["body"])

        # Prose body — Latest context + Reasoning. We removed the matchup /
        # usage-trend chip strip because it was a hallucination magnet
        # whenever extras.today_matchups didn't cover the player; the news
        # paragraph now carries any opponent angle the AI is allowed to cite.
        prose_parts: list[str] = []
        if news:
            prose_parts.append(
                "<font size='7.5' color='#cc5a00'><b>LATEST CONTEXT</b></font>"
                f"<br/><font size='9.5' color='#2b3340'>{_safe_text(news)}</font>"
            )
        if rationale:
            prose_parts.append(
                "<font size='7.5' color='#cc5a00'><b>REASONING</b></font>"
                f"<br/><font size='9.5' color='#2b3340'>{_safe_text(rationale)}</font>"
            )
        prose_html = "<br/><br/>".join(prose_parts) or (
            "<font size='9' color='#9aa3b2'>"
            "No additional context provided.</font>"
        )
        prose_para = Paragraph(prose_html, styles["body"])

        card = Table(
            [
                [header_row],
                [pick_para],
                [prose_para],
            ],
            colWidths=[7.0 * inch],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8f1")),
            ("LINEBEFORE", (0, 0), (0, -1), 4, BRAND_ORANGE),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("LINEBELOW", (0, 0), (0, 0), 0.4, PANEL_BORDER),
            ("LINEBELOW", (0, 1), (0, 1), 0.4, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        flow.append(KeepTogether([card, Spacer(1, 8)]))

    return flow


def _signal_grade(abs_edge: float) -> tuple[str, colors.Color]:
    if abs_edge >= 3.0:
        return "A-tier", POS_GREEN
    if abs_edge >= 2.0:
        return "B-tier", BRAND_ORANGE
    if abs_edge >= 1.0:
        return "C-tier", GOLD
    return "Watchlist", INK_FAINT


def _spotlight_card_flowable(
    rank: int, row: pd.Series, styles: dict,
    sigma_lookup: dict[str, float] | None = None,
) -> Table:
    edge_val = float(pd.to_numeric(row.get("edge"), errors="coerce"))
    side = str(row.get("call") or row.get("side") or "").upper() or (
        "MORE" if edge_val > 0 else "LESS"
    )
    abs_edge = abs(edge_val)
    grade, grade_color = _signal_grade(abs_edge)
    side_color = POS_GREEN if side in ("MORE", "OVER") else NEG_RED
    accent_hex = "#1f9d6c" if side in ("MORE", "OVER") else "#d24545"

    sigma = (sigma_lookup or {}).get(str(row.get("model", "")))
    hit_p = _hit_probability(
        row.get("model prediction", row.get("projection")),
        row.get("posted line", row.get("line")), side, sigma,
    )
    conf = _confidence_score(edge_val, sigma, row.get("books"))
    hit_html = ""
    if hit_p is not None or conf is not None:
        hit_str = _fmt_pct(hit_p) if hit_p is not None else "—"
        conf_str = f"{conf}/100" if conf is not None else "—"
        hit_html = (
            f"<br/><font size='8' color='#6b7686'>HIT%</font> "
            f"<font size='10' color='{accent_hex}'><b>{hit_str}</b></font>"
            f"  |  <font size='8' color='#6b7686'>CONF</font> "
            f"<font size='10' color='#11151c'><b>{conf_str}</b></font>"
        )

    badge_w = 0.35 * inch
    badge = Drawing(badge_w, 0.35 * inch)
    badge.add(Circle(badge_w / 2, badge_w / 2, badge_w / 2,
                     fillColor=grade_color, strokeColor=WHITE, strokeWidth=1))
    badge.add(String(
        badge_w / 2, badge_w / 2 - 4, f"#{rank}",
        fontName=_BOLD_FONT, fontSize=10, fillColor=WHITE,
        textAnchor="middle",
    ))

    text = Paragraph(
        (
            f"<font size='7.5' color='#cc5a00'><b>{grade.upper()} SIGNAL</b></font><br/>"
            f"<font size='11' color='#11151c'><b>{_short(row.get('player', 'Unknown'), 22)}</b></font><br/>"
            f"<font size='8.5' color='#6b7686'>{_short(row.get('model', 'metric'), 14)}</font><br/>"
            f"<br/>"
            f"<font size='8.5' color='#6b7686'>SIGNAL</font> "
            f"<font size='10' color='{accent_hex}'><b>{_side_display(side)}</b></font>"
            f"  |  <font size='8.5' color='#6b7686'>EDGE</font> "
            f"<font size='10' color='{accent_hex}'><b>{_fmt_signed(edge_val, 2)}</b></font><br/>"
            f"<font size='8' color='#6b7686'>"
            f"Line {_fmt(row.get('posted line', row.get('line')), 1)}  |  "
            f"Model {_fmt(row.get('model prediction', row.get('projection')), 2)}"
            f"</font>"
            f"{hit_html}"
        ),
        ParagraphStyle(
            f"spot_{rank}",
            parent=styles["body"],
            fontSize=9.3,
            leading=13,
            textColor=INK_BODY,
        ),
    )
    inner = Table(
        [[badge, text]],
        colWidths=[0.45 * inch, 1.85 * inch],
    )
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3, side_color),
        ("LEFTPADDING", (0, 0), (0, -1), 4),
        ("LEFTPADDING", (1, 0), (1, -1), 8),
        ("RIGHTPADDING", (1, 0), (1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return inner


def _spotlight_flowables(
    edge_df: pd.DataFrame | None, styles: dict,
    *, sigma_lookup: dict[str, float] | None = None,
) -> list:
    flow: list = _section_header("Signal spotlight  |  top 3", "Section 04", styles, anchor="sec-spotlight")
    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        flow.append(_para("No live signals to spotlight yet.", styles["muted"]))
        return flow

    df = edge_df.copy()
    if "edge" not in df.columns:
        flow.append(_para("No edge values available for spotlight cards.", styles["muted"]))
        return flow
    if "abs_edge" not in df.columns:
        df["abs_edge"] = pd.to_numeric(df["edge"], errors="coerce").abs()
    top = df.sort_values("abs_edge", ascending=False).head(3)
    if top.empty:
        flow.append(_para("No edge values available for spotlight cards.", styles["muted"]))
        return flow

    cards: list = [
        _spotlight_card_flowable(i + 1, row, styles, sigma_lookup=sigma_lookup)
        for i, (_, row) in enumerate(top.iterrows())
    ]
    while len(cards) < 3:
        cards.append(Paragraph("", styles["body"]))

    spot = Table([cards], colWidths=[2.4 * inch] * 3)
    spot.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(spot)
    flow.append(Spacer(1, 8))
    return flow


# ── Section: model quality ──────────────────────────────────────────────────
def _model_quality_flowables(metrics: pd.DataFrame | None, styles: dict) -> list:
    flow: list = _section_header("Model quality", "Section 06", styles, anchor="sec-model-quality")
    if metrics is None or not isinstance(metrics, pd.DataFrame) or metrics.empty:
        flow.append(_para("Model metrics unavailable.", styles["muted"]))
        return flow

    rows: list[list[Any]] = [["Model", "Target", "R²", "RMSE", "Tier"]]
    df = metrics.copy()
    name_col = next((c for c in ("model", "name") if c in df.columns), None)
    target_col = next((c for c in ("target", "stat") if c in df.columns), None)
    r2_col = next((c for c in ("R²", "r2", "R2") if c in df.columns), None)
    rmse_col = next((c for c in ("RMSE", "rmse") if c in df.columns), None)

    tier_indices: list[tuple[int, colors.Color]] = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        try:
            r2v = float(r[r2_col]) if r2_col else float("nan")
        except (TypeError, ValueError):
            r2v = float("nan")
        if pd.isna(r2v):
            tier_label, tier_color = "—", INK_FAINT
        elif r2v >= 0.45:
            tier_label, tier_color = "Strong", POS_GREEN
        elif r2v >= 0.30:
            tier_label, tier_color = "Solid", BRAND_ORANGE
        elif r2v >= 0.18:
            tier_label, tier_color = "Light", GOLD
        else:
            tier_label, tier_color = "Noisy", NEG_RED
        rows.append([
            str(r[name_col]) if name_col else "—",
            str(r[target_col]) if target_col else "—",
            _fmt(r[r2_col], 2) if r2_col else "—",
            _fmt(r[rmse_col], 2) if rmse_col else "—",
            tier_label,
        ])
        tier_indices.append((i, tier_color))

    table = _styled_table(
        rows,
        col_widths=[2.0 * inch, 1.4 * inch, 0.85 * inch, 0.85 * inch, 1.1 * inch],
        align_right_cols=[2, 3],
    )
    extra: list = []
    for idx, color in tier_indices:
        extra.append(("TEXTCOLOR", (4, idx), (4, idx), color))
        extra.append(("FONTNAME", (4, idx), (4, idx), _BOLD_FONT))
    table.setStyle(TableStyle(extra))
    flow.append(table)
    return flow


# ── Section: edge board ─────────────────────────────────────────────────────
def _edge_board_flowables(
    edge_df: pd.DataFrame | None,
    styles: dict,
    *,
    top_n: int = 14,
    sigma_lookup: dict[str, float] | None = None,
) -> list:
    flow: list = _section_header("Model vs line gaps", "Section 07", styles, anchor="sec-edges")
    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        flow.append(_para(
            "No live signals available. Add an Odds API key and fetch lines "
            "to populate this section.",
            styles["muted"],
        ))
        return flow

    df = edge_df.copy()
    if "abs_edge" not in df.columns and "edge" in df.columns:
        df["abs_edge"] = df["edge"].abs()
    df = df.sort_values("abs_edge", ascending=False).head(top_n)

    rows: list[list[Any]] = [
        ["Player", "Market", "Line", "Proj.", "Edge", "Hit%", "Conf", "Side", "Books"]
    ]
    for _, r in df.iterrows():
        edge_val = r.get("edge")
        side = str(r.get("call") or r.get("side") or "").upper()
        sigma = (sigma_lookup or {}).get(str(r.get("model", "")))
        hit_p = _hit_probability(
            r.get("model prediction", r.get("projection")),
            r.get("posted line", r.get("line")), side, sigma,
        )
        conf = _confidence_score(edge_val, sigma, r.get("books"))
        # Books column is tight on width, so render a small numeric chip
        # ("8 books") instead of a wrapping ticker — the per-player block
        # already shows the full abbreviated book list when it matters.
        n = r.get("books")
        if pd.notna(n):
            try:
                books_label = f"{int(n)}"
            except (TypeError, ValueError):
                books_label = "—"
        else:
            books_label = "—"
        rows.append([
            str(r.get("player", "—")),
            str(r.get("model", "—")),
            _fmt(r.get("posted line", r.get("line")), 1),
            _fmt(r.get("model prediction", r.get("projection")), 2),
            _fmt_signed(edge_val, 2),
            _fmt_pct(hit_p),
            f"{conf}" if conf is not None else "—",
            _side_display(side) if side else "—",
            books_label,
        ])

    table = _styled_table(
        rows,
        # Player column gets the extra space freed up by collapsing Books
        # to a numeric chip; every other column inherits the previous
        # widths so the existing per-cell colorings still line up.
        col_widths=[1.85 * inch, 1.05 * inch, 0.55 * inch, 0.65 * inch,
                    0.65 * inch, 0.55 * inch, 0.5 * inch, 0.55 * inch, 0.55 * inch],
        align_right_cols=[2, 3, 4, 5, 6, 8],
    )
    extra: list = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        try:
            ev = float(r.get("edge"))
        except (TypeError, ValueError):
            ev = 0.0
        color = POS_GREEN if ev > 0 else (NEG_RED if ev < 0 else INK_BODY)
        extra.append(("TEXTCOLOR", (4, i), (4, i), color))
        extra.append(("FONTNAME", (4, i), (4, i), _BOLD_FONT))
        side = str(r.get("call") or r.get("side") or "").upper()
        if side in ("MORE", "OVER"):
            extra.append(("TEXTCOLOR", (7, i), (7, i), POS_GREEN))
            extra.append(("FONTNAME", (7, i), (7, i), _BOLD_FONT))
            extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
        elif side in ("LESS", "UNDER"):
            extra.append(("TEXTCOLOR", (7, i), (7, i), NEG_RED))
            extra.append(("FONTNAME", (7, i), (7, i), _BOLD_FONT))
            extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
    table.setStyle(TableStyle(extra))
    flow.append(table)
    return flow


# ── Section: per-player breakdown ───────────────────────────────────────────
def _player_hero_band(player: str, recent_form: dict[str, float] | None, styles: dict) -> Table:
    bits = []
    for label, key in (
        ("PTS", "pts"), ("REB", "reb"), ("AST", "ast"),
        ("PRA", "pra"), ("FAN", "fantasy_score"), ("MIN", "min"),
    ):
        if recent_form and recent_form.get(key) is not None:
            bits.append(
                f"<font size='7' color='#fff1e6'><b>{label}</b></font> "
                f"<font size='10' color='#ffffff'><b>{_fmt(recent_form[key], 1)}</b></font>"
            )
    form_html = "   |   ".join(bits) if bits else (
        "<font size='9' color='#fff1e6'>No recent-form snapshot.</font>"
    )

    # Left accent strip (deep orange) creates a duotone hero feel.
    accent_tab = Paragraph("&nbsp;", styles["body"])
    name_para = Paragraph(
        f"<font size='15' color='#ffffff'><b>{_short(player, 32)}</b></font><br/>"
        f"<font size='7.5' color='#fff1e6'><b>RECENT FORM (LAST GAMES)</b></font>",
        styles["body"],
    )
    form_para = Paragraph(form_html, styles["body"])

    band = Table(
        [[accent_tab, name_para, form_para]],
        colWidths=[0.18 * inch, 2.92 * inch, 3.9 * inch],
    )
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BRAND_ORANGE_DEEP),
        ("BACKGROUND", (1, 0), (-1, -1), BRAND_ORANGE),
        ("LINEBELOW", (0, 0), (-1, -1), 1.5, BRAND_ORANGE_DEEP),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (-1, -1), 12),
        ("RIGHTPADDING", (1, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return band


def _player_block(
    player: str,
    *,
    edge_df: pd.DataFrame | None,
    projection: pd.DataFrame | None,
    recent_form: dict[str, float] | None,
    rationale: str,
    news: str = "",
    prediction: str = "",
    history: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
    sigma_lookup: dict[str, float] | None = None,
    styles: dict,
) -> list:
    slug = "player-" + "".join(
        ch.lower() if ch.isalnum() else "-" for ch in player
    ).strip("-")

    # Header cluster: anchor + orange hero band + (optional) chip strip.
    # Wrapped in KeepTogether so the bright hero band can never be orphaned
    # at the bottom of a page with the chips/sparklines starting on the
    # next page (a common eyesore in earlier renders).
    chips = _risk_chips(
        player=player, edge_df=edge_df, games=games,
        recent_form=recent_form, history_df=history,
    )
    chip_strip = _risk_chip_strip(chips, styles)
    header_cluster: list = [
        _AnchorFlowable(slug, player, level=1),
        _player_hero_band(player, recent_form, styles),
        Spacer(1, 6),
    ]
    if chip_strip is not None:
        header_cluster.append(chip_strip)
        header_cluster.append(Spacer(1, 6))
    flow: list = [KeepTogether(header_cluster)]

    # Visual stack: full-width sparklines first (so all three stats breathe),
    # then a clean two-column row with the structured "Why this signal" panel
    # on the left and the Over/Under hit-rate donut on the right.
    spark = _player_form_sparklines(games)
    donut = _player_hit_rate_donut(history)
    signal = _signal_summary_panel(
        player=player, edge_df=edge_df, recent_form=recent_form,
        history_df=history, styles=styles,
    )

    if spark is not None:
        flow.append(spark)
        flow.append(Spacer(1, 6))

    if signal is not None and donut is not None:
        row = Table(
            [[signal, donut]],
            colWidths=[4.2 * inch, 2.8 * inch],
        )
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        flow.append(row)
        flow.append(Spacer(1, 6))
    elif signal is not None:
        # No donut to pair with — re-render the panel at full content width
        # so it doesn't float as a narrow column at left.
        wide_signal = _signal_summary_panel(
            player=player, edge_df=edge_df, recent_form=recent_form,
            history_df=history, styles=styles, width=7.0 * inch,
        )
        flow.append(wide_signal if wide_signal is not None else signal)
        flow.append(Spacer(1, 6))
    elif donut is not None:
        flow.append(donut)
        flow.append(Spacer(1, 6))

    chart = _player_minichart(projection, edge_df, player) if projection is not None else None
    if chart is not None:
        flow.append(chart)
        flow.append(Spacer(1, 4))

    if (
        projection is not None
        and isinstance(projection, pd.DataFrame)
        and not projection.empty
    ):
        proj = projection.copy()
        edge_lookup: dict[str, dict] = {}
        if (
            edge_df is not None
            and isinstance(edge_df, pd.DataFrame)
            and not edge_df.empty
        ):
            sub = edge_df[edge_df["player"] == player]
            for _, r in sub.iterrows():
                edge_lookup[str(r["model"])] = {
                    "line": r.get("posted line", r.get("line")),
                    "edge": r.get("edge"),
                    "side": r.get("call", r.get("side", "")),
                }

        rows: list[list[Any]] = [
            ["Model", "Target", "Projection", "Line", "Edge", "Hit%", "Side"]
        ]
        sigma_lookup = sigma_lookup or {}
        for _, r in proj.iterrows():
            name = str(r["model"])
            ed = edge_lookup.get(name, {})
            side = str(ed.get("side", "")).upper().split()[0] if ed.get("side") else "—"
            sig = sigma_lookup.get(name)
            hit_p = _hit_probability(
                r.get("prediction"), ed.get("line"), ed.get("side", ""), sig,
            ) if ed else None
            rows.append([
                name,
                str(r.get("target", "—")),
                _fmt(r["prediction"], 2),
                _fmt(ed.get("line"), 1) if ed else "—",
                _fmt_signed(ed.get("edge"), 2) if ed else "—",
                _fmt_pct(hit_p),
                _side_display(side) if side and side != "—" else "—",
            ])

        table = _styled_table(
            rows,
            col_widths=[1.6 * inch, 1.05 * inch, 0.9 * inch, 0.65 * inch,
                        0.8 * inch, 0.6 * inch, 0.55 * inch],
            align_right_cols=[2, 3, 4, 5],
        )
        extra: list = []
        for i, (_, r) in enumerate(proj.iterrows(), start=1):
            name = str(r["model"])
            ed = edge_lookup.get(name, {})
            try:
                ev = float(ed.get("edge")) if ed.get("edge") is not None else None
            except (TypeError, ValueError):
                ev = None
            if ev is not None:
                color = POS_GREEN if ev > 0 else (NEG_RED if ev < 0 else INK_BODY)
                extra.append(("TEXTCOLOR", (4, i), (4, i), color))
                extra.append(("FONTNAME", (4, i), (4, i), _BOLD_FONT))
            side = str(ed.get("side", "")).upper().split()[0] if ed.get("side") else ""
            if side in ("MORE", "OVER"):
                extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
                extra.append(("FONTNAME", (5, i), (5, i), _BOLD_FONT))
                extra.append(("TEXTCOLOR", (6, i), (6, i), POS_GREEN))
                extra.append(("FONTNAME", (6, i), (6, i), _BOLD_FONT))
            elif side in ("LESS", "UNDER"):
                extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
                extra.append(("FONTNAME", (5, i), (5, i), _BOLD_FONT))
                extra.append(("TEXTCOLOR", (6, i), (6, i), NEG_RED))
                extra.append(("FONTNAME", (6, i), (6, i), _BOLD_FONT))
        table.setStyle(TableStyle(extra))

        # Sticky breadcrumb above the projections table — survives a page
        # break so the reader always knows which player the rows belong to.
        breadcrumb = Paragraph(
            f"<font size='7.5' color='#cc5a00'><b>"
            f"{player.upper()} &nbsp;|&nbsp; MODEL PROJECTIONS"
            f"</b></font>",
            styles["body"],
        )
        flow.append(KeepTogether([breadcrumb, Spacer(1, 4), table]))

        # Outlook panel: distribution bands (top model) + minutes projection.
        # Lets the reader see realistic 25/50/75 outcomes and a forward-looking
        # minutes window without leaving the per-player section.
        try:
            top_row = proj.iloc[0]
            top_name = str(top_row["model"])
            top_sigma = (sigma_lookup or {}).get(top_name)
            bands = _distribution_bands(top_row.get("prediction"), top_sigma)
        except Exception:
            bands = None
            top_name = ""
        mins = _minutes_projection(games)
        if bands is not None or mins is not None:
            left_html = ""
            if bands is not None:
                left_html = (
                    f"<font size='7.5' color='#cc5a00'><b>OUTCOME BAND "
                    f"&middot; {top_name.upper()}</b></font><br/>"
                    f"<font size='9' color='#11151c'>"
                    f"P25 <b>{bands['p25']:.1f}</b> &nbsp;"
                    f"P50 <b>{bands['p50']:.1f}</b> &nbsp;"
                    f"P75 <b>{bands['p75']:.1f}</b>"
                    f"</font><br/>"
                    f"<font size='6.5' color='#6b7480'>"
                    f"Middle 50% of modeled outcomes</font>"
                )
            else:
                left_html = (
                    "<font size='7.5' color='#6b7480'><b>OUTCOME BAND</b>"
                    "</font><br/><font size='8' color='#6b7480'>Insufficient "
                    "variance estimate.</font>"
                )
            if mins is not None:
                m_proj, m_lo, m_hi = mins
                right_html = (
                    f"<font size='7.5' color='#cc5a00'><b>MINUTES PROJECTION"
                    f"</b></font><br/>"
                    f"<font size='9' color='#11151c'>"
                    f"<b>{m_proj:.1f}</b> min &nbsp;"
                    f"<font size='7' color='#6b7480'>range "
                    f"{m_lo:.1f}\u2013{m_hi:.1f}</font></font><br/>"
                    f"<font size='6.5' color='#6b7480'>"
                    f"Weighted L3/L5/L10 &middot; \u00b11\u03c3 band</font>"
                )
            else:
                right_html = (
                    "<font size='7.5' color='#6b7480'><b>MINUTES PROJECTION"
                    "</b></font><br/><font size='8' color='#6b7480'>"
                    "No minutes history.</font>"
                )
            left_p = Paragraph(left_html, styles["body"])
            right_p = Paragraph(right_html, styles["body"])
            outlook = Table(
                [[left_p, right_p]],
                colWidths=[3.5 * inch, 3.5 * inch],
            )
            outlook.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
                ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
                ("LINEAFTER", (0, 0), (0, -1), 0.4, PANEL_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            flow.append(Spacer(1, 4))
            flow.append(outlook)
    else:
        flow.append(_para("No model projections available.", styles["muted"]))

    flow.append(Spacer(1, 4))

    # Hooplytics prediction panel — concrete pick + confidence pulled from
    # the AI rationale. Rendered as a high-contrast band so a skimmer can
    # land on the call without reading the analyst notes.
    if prediction and prediction.strip():
        pred_para = Paragraph(
            f"<font size='8' color='#cc5a00'><b>HOOPLYTICS PREDICTION</b></font>"
            f"<br/><font size='10.5' color='#11151c'><b>"
            f"{_safe_text(prediction.strip())}</b></font>",
            styles["body"],
        )
        pred_tbl = Table([[pred_para]], colWidths=[7.0 * inch])
        pred_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4ea")),
            ("BOX", (0, 0), (-1, -1), 0.5, BRAND_ORANGE),
            ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_ORANGE_DEEP),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(KeepTogether([pred_tbl, Spacer(1, 6)]))

    # AI scout prose — plain inline subheadings so the per-player block reads
    # like the reference report (no card chrome, no chip strip). The "loud"
    # AI scout card lives in Section 03; here the prose just supports the
    # numbers above. The block is clearly attributed to the AI scout via the
    # eyebrow heading so readers don't confuse it with deterministic output.
    if (news and news.strip()) or (rationale and rationale.strip()):
        flow.append(Spacer(1, 8))
        flow.append(Paragraph(
            "<font size='8' color='#cc5a00'><b>AI&nbsp;SCOUT&nbsp;REPORT</b></font>"
            "&nbsp;&nbsp;<font size='7.5' color='#9aa3b2'>"
            "Generated prose &middot; not a deterministic signal"
            "</font>",
            styles["body"],
        ))
        flow.append(Spacer(1, 2))
        if news and news.strip():
            flow.append(Paragraph(
                "<font size='7.5' color='#cc5a00'><b>LATEST CONTEXT</b></font>",
                styles["body"],
            ))
            flow.append(Paragraph(
                f"<font size='9.5' color='#2b3340'>{_safe_text(news.strip())}</font>",
                styles["body"],
            ))
            flow.append(Spacer(1, 4))
        if rationale and rationale.strip():
            flow.append(Paragraph(
                "<font size='7.5' color='#cc5a00'><b>ANALYST NOTES</b></font>",
                styles["body"],
            ))
            flow.append(Paragraph(
                f"<font size='9.5' color='#2b3340'>{_safe_text(rationale.strip())}</font>",
                styles["body"],
            ))

    history_block = _player_history_table(player, history, styles)
    if history_block:
        flow.extend(history_block)

    flow.append(Spacer(1, 12))
    return flow


def _slip_builder_flowables(
    *,
    edge_df: pd.DataFrame | None,
    sigma_lookup: dict[str, float] | None,
    player_games: dict[str, pd.DataFrame] | None,
    styles: dict,
) -> list:
    """Anchor / Differentiator / Secondary picks plus correlation warnings.

    Designed so a bettor can build a defensible 2-3 leg slip without flipping
    back through the per-player section. All ranking is deterministic so the
    same edge_df produces the same slip every render.
    """
    flow: list = _section_header(
        "Signal stack", "Section 08", styles, anchor="sec-slip",
    )
    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        flow.append(_para(
            "No signals available to build a stack from.", styles["muted"],
        ))
        return flow

    sigma_lookup = sigma_lookup or {}
    player_games = player_games or {}
    df = edge_df.copy()

    # Score every row: confidence (primary) + |edge| as tie-breaker.
    rows: list[dict] = []
    for _, r in df.iterrows():
        sig = sigma_lookup.get(str(r.get("model", "")))
        conf = _confidence_score(r.get("edge"), sig, r.get("books"))
        try:
            ev = float(r.get("edge")) if r.get("edge") is not None else 0.0
        except (TypeError, ValueError):
            ev = 0.0
        side_raw = str(r.get("call", r.get("side", ""))).upper().split()
        side = side_raw[0] if side_raw else ""
        direction = (
            "OVER" if side in ("MORE", "OVER")
            else ("UNDER" if side in ("LESS", "UNDER") else "")
        )
        player = str(r.get("player", ""))
        vol = _volatility_label(player_games.get(player))
        vol_label = vol[0] if vol else ""
        rows.append({
            "player": player,
            "market": str(r.get("market", r.get("target", ""))),
            "side": side or "—",
            "line": r.get("posted line", r.get("line")),
            "edge": ev,
            "abs_edge": abs(ev),
            "conf": conf if conf is not None else 0,
            "direction": direction,
            "vol": vol_label,
        })

    if not rows:
        flow.append(_para(
            "Signal stack requires at least one ranked signal.", styles["muted"],
        ))
        return flow

    rows.sort(key=lambda d: (-d["conf"], -d["abs_edge"]))

    # Anchor: highest confidence with non-HIGH volatility (fallback to first).
    anchor = next(
        (d for d in rows if d["vol"] != "HIGH VOL"), rows[0],
    )
    # Differentiator: highest |edge| in the OPPOSITE direction (or any other
    # player) so we don't double down on the same tempo bucket.
    diff_candidates = [
        d for d in rows
        if d["player"] != anchor["player"]
        and (not anchor["direction"] or d["direction"] != anchor["direction"])
    ]
    if not diff_candidates:
        diff_candidates = [d for d in rows if d["player"] != anchor["player"]]
    differentiator = max(diff_candidates, key=lambda d: d["abs_edge"]) if diff_candidates else None

    # Secondary: next-best confidence not already chosen and not same player.
    chosen_players = {anchor["player"]}
    if differentiator is not None:
        chosen_players.add(differentiator["player"])
    secondary = next(
        (d for d in rows if d["player"] not in chosen_players), None,
    )

    def _slip_card(label: str, pick: dict | None, accent_hex: str) -> Table:
        if pick is None:
            body_html = (
                "<font size='8' color='#6b7480'>No qualifying pick.</font>"
            )
        else:
            edge_color = "#0a8f3a" if pick["edge"] > 0 else (
                "#c0392b" if pick["edge"] < 0 else "#11151c"
            )
            line_str = _fmt(pick["line"], 1) if pick["line"] is not None else "—"
            vol_chip = (
                f" &nbsp;<font size='6.5' color='#6b7480'>&middot; "
                f"{pick['vol']}</font>"
                if pick["vol"] else ""
            )
            body_html = (
                f"<font size='10' color='#11151c'><b>{_safe_text(pick['player'])}</b>"
                f"</font><br/>"
                f"<font size='8.5' color='#11151c'>{_safe_text(pick['market'])} "
                f"<b>{_side_display(pick['side'])}</b> {line_str}</font><br/>"
                f"<font size='7.5' color='#6b7480'>EDGE "
                f"<font color='{edge_color}'><b>"
                f"{_fmt_signed(pick['edge'], 2)}</b></font> &nbsp;"
                f"&middot; CONF <b>{pick['conf']}/100</b>{vol_chip}</font>"
            )
        body = Paragraph(body_html, styles["body"])
        eyebrow = Paragraph(
            f"<font size='7' color='{accent_hex}'><b>{label}</b></font>",
            styles["body"],
        )
        card = Table(
            [[eyebrow], [body]],
            colWidths=[2.25 * inch],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9f2")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(accent_hex)),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent_hex)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return card

    cards_row = Table(
        [[
            _slip_card("BEST ANCHOR", anchor, "#cc5a00"),
            _slip_card("BEST DIFFERENTIATOR", differentiator, "#0a8f3a"),
            _slip_card("SECONDARY ADD", secondary, "#3b6fb3"),
        ]],
        colWidths=[2.33 * inch, 2.33 * inch, 2.33 * inch],
    )
    cards_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(cards_row)
    flow.append(Spacer(1, 10))

    # Correlation warnings — clusters of same-game / same-direction picks.
    clusters = _correlation_clusters(edge_df)
    if clusters:
        # Wrap long cells in Paragraphs so they wrap inside the column instead
        # of overflowing into adjacent cells (the cause of the previous
        # "Phoenix Suns" overlapping "UNDER" rendering wart).
        cell_style = ParagraphStyle(
            "warn_cell", parent=styles["body"], fontSize=8.5, leading=10.5,
        )
        warn_rows: list[list[Any]] = [["Matchup", "Direction", "Players", "Legs"]]
        for c in clusters[:5]:
            players_txt = ", ".join(c["players"][:4]) + (
                "\u2026" if len(c["players"]) > 4 else ""
            )
            warn_rows.append([
                Paragraph(_safe_text(c["matchup"]), cell_style),
                c["direction"],
                Paragraph(_safe_text(players_txt), cell_style),
                str(c["count"]),
            ])
        warn_table = _styled_table(
            warn_rows,
            col_widths=[2.35 * inch, 0.75 * inch, 3.35 * inch, 0.55 * inch],
            align_right_cols=[3],
        )
        warn_eyebrow = Paragraph(
            "<font size='7.5' color='#c0392b'><b>AVOID STACKING &middot; "
            "TEMPO-CORRELATED LEGS</b></font>",
            styles["body"],
        )
        warn_note = Paragraph(
            "<font size='8' color='#6b7480'>These calls share a game and a "
            "direction \u2014 they\u2019ll mostly hit or miss together, which "
            "breaks parlay independence. Treat them as a single thesis, not "
            "separate legs.</font>",
            styles["body"],
        )
        flow.append(KeepTogether([
            warn_eyebrow, Spacer(1, 4), warn_table, Spacer(1, 4), warn_note,
        ]))
    else:
        ok = Paragraph(
            "<font size='8' color='#6b7480'>No correlated clusters detected "
            "\u2014 your top signals are spread across independent games.</font>",
            styles["body"],
        )
        flow.append(ok)

    flow.append(Spacer(1, 8))
    return flow


def _per_player_flowables(
    roster: dict[str, list[str]],
    *,
    edge_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    recent_form: dict[str, dict[str, float]] | None,
    ai_sections: dict[str, Any] | None,
    player_history: dict[str, pd.DataFrame] | None = None,
    player_games: dict[str, pd.DataFrame] | None = None,
    sigma_lookup: dict[str, float] | None = None,
    styles: dict,
) -> list:
    flow: list = _section_header("Per-player breakdown", "Section 09", styles, anchor="sec-players")
    if not roster:
        flow.append(_para("Roster is empty.", styles["muted"]))
        return flow

    ai_players = (ai_sections or {}).get("players") or {}
    projections = projections or {}
    recent_form = recent_form or {}
    player_history = player_history or {}
    player_games = player_games or {}

    for player in roster.keys():
        ai_entry = ai_players.get(player, "")
        if isinstance(ai_entry, dict):
            p_rationale = str(ai_entry.get("rationale", "")).strip()
            p_news = str(ai_entry.get("news", "")).strip()
            p_prediction = str(ai_entry.get("prediction", "")).strip()
        else:
            p_rationale = str(ai_entry or "").strip()
            p_news = ""
            p_prediction = ""
        block = _player_block(
            player,
            edge_df=edge_df,
            projection=projections.get(player),
            recent_form=recent_form.get(player),
            rationale=p_rationale,
            news=p_news,
            prediction=p_prediction,
            history=player_history.get(player),
            games=player_games.get(player),
            sigma_lookup=sigma_lookup,
            styles=styles,
        )
        # Each player block already KeepTogether's its own hero band; flowing
        # the rest freely lets the first player start directly under the
        # section header instead of forcing it onto a fresh page (the cause
        # of the \"Section 07: Per-player breakdown\" lone-title page).
        flow.extend(block)

    return flow


# ═══════════════════════════════════════════════════════════════════════════
# REPORT V2 — Editorial cover + dashboard interior redesign
# ═══════════════════════════════════════════════════════════════════════════
# Everything below is the new visual system. It reuses the existing
# palette / fonts / safe-text helpers above so the file stays self-contained.
# Old V1 helpers are intentionally left in place but unreferenced — git is
# the rollback. ``build_pdf_report`` (further below) wires only the V2 flow.

# ── V2 palette extensions ──────────────────────────────────────────────────
V2_INK_HERO = colors.HexColor("#0a0e14")           # near-black for hero serif
V2_ACCENT_TEAL = colors.HexColor("#1f7a8c")        # cool counterpoint
V2_HAIRLINE = colors.HexColor("#d6dae2")           # softer than PANEL_BORDER
V2_BG_CREAM = colors.HexColor("#faf6f0")           # warm panel for editorial pages
V2_BG_INK = colors.HexColor("#11151c")             # hero band background
V2_BG_INK_SOFT = colors.HexColor("#1c2230")        # secondary dark band

# 7-stop diverging ramp for confidence heatmap (BELOW red ↔ ABOVE green).
V2_HEAT_NEG_3 = colors.HexColor("#a32020")
V2_HEAT_NEG_2 = colors.HexColor("#d24545")
V2_HEAT_NEG_1 = colors.HexColor("#f0b3b3")
V2_HEAT_NEUTRAL = colors.HexColor("#eef0f4")
V2_HEAT_POS_1 = colors.HexColor("#aedcc6")
V2_HEAT_POS_2 = colors.HexColor("#1f9d6c")
V2_HEAT_POS_3 = colors.HexColor("#0f6e4a")

# Reliability gauge tier ramp (red → gold → green).
V2_TIER_LOW = colors.HexColor("#d24545")
V2_TIER_MID = colors.HexColor("#d4a017")
V2_TIER_HIGH = colors.HexColor("#1f9d6c")


# ── V2 serif font (best-effort, silent fallback to existing sans bold) ─────
_V2_SERIF_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # macOS (Georgia ships preinstalled).
    ("HoopSerif", "Georgia.ttf", "Georgia Bold.ttf"),
    # macOS supplemental.
    ("HoopSerif", "Charter.ttc", "Charter.ttc"),
    # Linux DejaVu Serif.
    ("HoopSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ("HoopSerif", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    # Windows Times New Roman.
    ("HoopSerif", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
)


def _v2_register_serif() -> tuple[str, str]:
    """Best-effort serif registration. Returns (regular_face, bold_face).

    ``.ttc`` collection files are skipped because reportlab's ``TTFont`` can't
    load them directly. Falls back to the existing sans bold so the layout
    still reads cleanly even without a serif installed.
    """
    for fam, reg, bold in _V2_SERIF_CANDIDATES:
        rp = _resolve_font_path(reg)
        bp = _resolve_font_path(bold)
        if not rp or rp.lower().endswith(".ttc"):
            continue
        try:
            pdfmetrics.registerFont(TTFont(fam, rp))
            bold_face = fam
            if bp and bp != rp and not bp.lower().endswith(".ttc"):
                bold_face = f"{fam}-Bold"
                try:
                    pdfmetrics.registerFont(TTFont(bold_face, bp))
                except Exception:
                    bold_face = fam
            return fam, bold_face
        except Exception:
            continue
    return _BODY_FONT, _BOLD_FONT


_V2_SERIF, _V2_SERIF_BOLD = _v2_register_serif()


# ── V2 styles helper ───────────────────────────────────────────────────────
def _v2_styles(styles: dict[str, ParagraphStyle]) -> dict[str, ParagraphStyle]:
    """Augment the V1 style dict with editorial styles. Idempotent."""
    if "v2_cover_title" in styles:
        return styles
    base = styles["body"]
    styles["v2_cover_title"] = ParagraphStyle(
        "v2_cover_title", parent=base, fontName=_V2_SERIF_BOLD,
        fontSize=58, leading=62, textColor=V2_INK_HERO, alignment=TA_LEFT,
    )
    styles["v2_cover_eyebrow"] = ParagraphStyle(
        "v2_cover_eyebrow", parent=base, fontName=_BOLD_FONT, fontSize=9,
        leading=12, textColor=BRAND_ORANGE_DEEP, alignment=TA_LEFT,
    )
    styles["v2_cover_dek"] = ParagraphStyle(
        "v2_cover_dek", parent=base, fontName=_V2_SERIF, fontSize=15,
        leading=21, textColor=INK_BODY, alignment=TA_LEFT,
    )
    styles["v2_section_eyebrow"] = ParagraphStyle(
        "v2_section_eyebrow", parent=base, fontName=_BOLD_FONT, fontSize=8,
        leading=11, textColor=BRAND_ORANGE_DEEP, alignment=TA_LEFT,
    )
    styles["v2_section_title"] = ParagraphStyle(
        "v2_section_title", parent=base, fontName=_V2_SERIF_BOLD, fontSize=26,
        leading=30, textColor=V2_INK_HERO, alignment=TA_LEFT, spaceAfter=4,
    )
    styles["v2_section_dek"] = ParagraphStyle(
        "v2_section_dek", parent=base, fontName=_V2_SERIF, fontSize=11,
        leading=15, textColor=INK_MUTED, alignment=TA_LEFT, spaceAfter=8,
    )
    styles["v2_kpi_value_xl"] = ParagraphStyle(
        "v2_kpi_value_xl", parent=base, fontName=_V2_SERIF_BOLD, fontSize=34,
        leading=36, textColor=V2_INK_HERO, alignment=TA_LEFT,
    )
    styles["v2_kpi_label"] = ParagraphStyle(
        "v2_kpi_label", parent=base, fontName=_BOLD_FONT, fontSize=7.5,
        leading=10, textColor=INK_MUTED, alignment=TA_LEFT,
    )
    styles["v2_dashboard_h"] = ParagraphStyle(
        "v2_dashboard_h", parent=base, fontName=_BOLD_FONT, fontSize=10,
        leading=13, textColor=V2_INK_HERO, alignment=TA_LEFT,
    )
    styles["v2_player_name"] = ParagraphStyle(
        "v2_player_name", parent=base, fontName=_V2_SERIF_BOLD, fontSize=30,
        leading=34, textColor=WHITE, alignment=TA_LEFT,
    )
    styles["v2_pill"] = ParagraphStyle(
        "v2_pill", parent=base, fontName=_BOLD_FONT, fontSize=8.5,
        leading=11, textColor=WHITE, alignment=TA_CENTER,
    )
    styles["v2_body"] = ParagraphStyle(
        "v2_body", parent=base, fontName=_BODY_FONT, fontSize=9.5,
        leading=13.5, textColor=INK_BODY, spaceAfter=4,
    )
    styles["v2_micro"] = ParagraphStyle(
        "v2_micro", parent=base, fontName=_BODY_FONT, fontSize=7.5,
        leading=10, textColor=INK_MUTED, alignment=TA_LEFT,
    )
    return styles


# ── V2 internal utilities ──────────────────────────────────────────────────
def _v2_heat_color(value: float, max_abs: float) -> colors.Color:
    """Map a signed edge to a 7-stop diverging green↔red ramp."""
    if max_abs <= 0:
        return V2_HEAT_NEUTRAL
    t = max(-1.0, min(1.0, value / max_abs))
    if t <= -0.66: return V2_HEAT_NEG_3
    if t <= -0.33: return V2_HEAT_NEG_2
    if t < -0.05:  return V2_HEAT_NEG_1
    if t <= 0.05:  return V2_HEAT_NEUTRAL
    if t < 0.33:   return V2_HEAT_POS_1
    if t < 0.66:   return V2_HEAT_POS_2
    return V2_HEAT_POS_3


def _v2_heat_text_color(value: float, max_abs: float) -> colors.Color:
    """Pick a legible text color for a heatmap cell based on saturation."""
    if max_abs <= 0:
        return INK_BODY
    t = abs(value) / max_abs
    return WHITE if t > 0.55 else INK_DARK


def _v2_top_signal(edge_df: pd.DataFrame | None) -> dict | None:
    """Return the loudest |edge| row as a dict, or None when no edges."""
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None
    if "edge" not in edge_df.columns:
        return None
    df = edge_df.copy()
    df["_abs"] = pd.to_numeric(df["edge"], errors="coerce").abs()
    df = df.dropna(subset=["_abs"]).sort_values("_abs", ascending=False)
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "player": str(r.get("player", "")),
        "model": str(r.get("model", "")),
        "edge": float(r["edge"]) if pd.notna(r.get("edge")) else 0.0,
        "side": str(r.get("call") or r.get("side") or ""),
        "line": r.get("line") or r.get("posted line"),
        "projection": r.get("projection") or r.get("prediction"),
        "books": r.get("books"),
        "matchup": r.get("matchup", ""),
    }


def _v2_get_line(row) -> Any:
    """Edge frames stamped by app.py use ``posted line``. Standalone fixtures
    use ``line``. Try both before giving up."""
    try:
        v = row.get("posted line") if hasattr(row, "get") else None
    except Exception:
        v = None
    if v is None or (isinstance(v, float) and pd.isna(v)):
        try:
            v = row.get("line") if hasattr(row, "get") else None
        except Exception:
            v = None
    return v


def _v2_get_proj(row) -> Any:
    """Same idea as ``_v2_get_line`` but for the model output column."""
    try:
        v = row.get("model prediction") if hasattr(row, "get") else None
    except Exception:
        v = None
    if v is None or (isinstance(v, float) and pd.isna(v)):
        for k in ("projection", "prediction"):
            try:
                v = row.get(k) if hasattr(row, "get") else None
            except Exception:
                v = None
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                break
    return v


def _v2_market_universe(edge_df: pd.DataFrame | None) -> list[str]:
    """Ordered list of distinct markets present in edge_df.

    Uses a fixed canonical order so the heatmap columns are stable across
    runs even when only some markets are populated tonight.
    """
    canonical = ["points", "rebounds", "assists", "threepm", "pra",
                 "fantasy_score", "stl_blk", "turnovers"]
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty or "model" not in edge_df.columns:
        return []
    present = set(str(m).lower() for m in edge_df["model"].dropna().unique())
    return [m for m in canonical if m in present]


def _v2_pill_para(text: str, fill: colors.Color, *, fg=WHITE, font_size: int = 8) -> Paragraph:
    """Tiny rounded-rect-style pill via a styled background."""
    return Paragraph(
        f"<font size='{font_size}' color='{fg.hexval()[2:].rjust(6, '0').lower()}'>"
        f"<b>&nbsp;{_safe_text(text)}&nbsp;</b></font>",
        ParagraphStyle(
            "v2_pill_inline", fontName=_BOLD_FONT, fontSize=font_size,
            leading=font_size + 3, textColor=fg, alignment=TA_CENTER,
            backColor=fill, borderPadding=2,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# V2 CHART PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Slate radar (or skyline fallback for thin slates) ───────────────────
def _v2_slate_radar(
    edge_df: pd.DataFrame | None,
    *,
    width: float = 6.0 * inch,
    height: float = 4.6 * inch,
) -> Drawing | None:
    """Editorial divergent edge chart for the cover.

    Each row is a player×market signal, drawn as a horizontal bar from a
    center axis. Length encodes |edge|, color encodes side (green = ABOVE,
    red = BELOW). Sorted with the loudest mispricing on top.
    """
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty or "edge" not in edge_df.columns:
        return None
    df = edge_df.copy()
    df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
    df = df.dropna(subset=["_e"])
    if df.empty:
        return None

    df["_abs"] = df["_e"].abs()
    df = df.sort_values("_abs", ascending=False).head(10).reset_index(drop=True)
    n = len(df)
    max_abs = float(df["_abs"].max()) or 1.0

    d = Drawing(width, height)
    pad_l = 2.3 * inch     # space for player + market labels (wider names)
    pad_r = 0.55 * inch    # space for edge value
    pad_t = 38             # space for ABOVE/BELOW headers
    pad_b = 26             # space for caption
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    cx = pad_l + plot_w / 2.0
    if plot_h <= 0 or plot_w <= 0:
        return d

    market_short = {
        "points": "PTS", "rebounds": "REB", "assists": "AST",
        "threepm": "3PM", "pra": "PRA", "fantasy_score": "FAN",
        "stl_blk": "S+B", "turnovers": "TOV",
    }

    # Direction headers above the plot. Use small filled triangles so the
    # glyph renders cleanly in any embedded font.
    from reportlab.graphics.shapes import Polygon as _Polygon
    header_y = pad_b + plot_h + 16
    tri_l_x = cx - 8
    d.add(_Polygon(points=[tri_l_x, header_y + 2,
                           tri_l_x + 6, header_y - 2,
                           tri_l_x + 6, header_y + 6],
                   strokeColor=None, fillColor=NEG_RED))
    d.add(String(tri_l_x - 4, header_y - 1, "BELOW",
                 fontName=_BOLD_FONT, fontSize=8.5,
                 fillColor=NEG_RED, textAnchor="end"))
    tri_r_x = cx + 8
    d.add(_Polygon(points=[tri_r_x, header_y + 2,
                           tri_r_x - 6, header_y - 2,
                           tri_r_x - 6, header_y + 6],
                   strokeColor=None, fillColor=POS_GREEN))
    d.add(String(tri_r_x + 4, header_y - 1, "ABOVE",
                 fontName=_BOLD_FONT, fontSize=8.5,
                 fillColor=POS_GREEN, textAnchor="start"))

    # Faint quarter-scale guide lines
    for frac in (0.5, 1.0):
        for sign in (-1, 1):
            xg = cx + sign * (plot_w / 2.0) * frac
            d.add(Line(xg, pad_b, xg, pad_b + plot_h,
                       strokeColor=V2_HAIRLINE, strokeWidth=0.4,
                       strokeDashArray=[1, 3]))

    # Solid center axis.
    d.add(Line(cx, pad_b - 4, cx, pad_b + plot_h + 4,
               strokeColor=INK_FAINT, strokeWidth=0.8))

    # Bars (top-down: rank 1 on top).
    row_h = plot_h / max(n, 1)
    bar_h = max(8.0, min(18.0, row_h * 0.62))
    for i, r in df.iterrows():
        e = float(r["_e"])
        # Top-down ordering: index 0 (loudest) at top of plot.
        y_center = pad_b + plot_h - (i + 0.5) * row_h
        y = y_center - bar_h / 2.0
        bw = (plot_w / 2.0) * (abs(e) / max_abs)
        col = POS_GREEN if e > 0 else NEG_RED
        bg_col = colors.HexColor("#e8f5ee") if e > 0 else colors.HexColor("#fbecec")
        # Light track on the active half so short bars still register visually.
        half_w = plot_w / 2.0
        if e >= 0:
            d.add(Rect(cx, y + bar_h * 0.25, half_w, bar_h * 0.5,
                       strokeColor=None, fillColor=bg_col))
            d.add(Rect(cx, y, bw, bar_h, strokeColor=None, fillColor=col))
            tip_x = cx + bw
        else:
            d.add(Rect(cx - half_w, y + bar_h * 0.25, half_w, bar_h * 0.5,
                       strokeColor=None, fillColor=bg_col))
            d.add(Rect(cx - bw, y, bw, bar_h, strokeColor=None, fillColor=col))
            tip_x = cx - bw
        # Bar tip dot for newspaper feel.
        d.add(Circle(tip_x, y + bar_h / 2.0, 2.6,
                     strokeColor=None, fillColor=col))

        # Player name (bold serif) on the far left.
        player = str(r.get("player", ""))
        market = str(r.get("model", "")).lower()
        ms = market_short.get(market, market[:3].upper())
        d.add(String(pad_l - 12, y_center - 3,
                     _short(player, 22),
                     fontName=_V2_SERIF_BOLD, fontSize=9.5,
                     fillColor=INK_DARK, textAnchor="end"))
        # Market label (small caps) just inside, below or beside name.
        d.add(String(pad_l - 12, y_center - 12,
                     ms,
                     fontName=_BOLD_FONT, fontSize=6.5,
                     fillColor=INK_MUTED, textAnchor="end"))
        # Edge value placement: when the bar is wide enough, draw the value
        # INSIDE the bar near the tip (white) so it never overlaps with the
        # player name label on the left margin. When the bar is narrow, fall
        # back to the previous outside-the-tip placement.
        edge_text = _fmt_signed(e, 2)
        # Approximate text width in points (Helvetica ~ 0.55 em-width at 9pt).
        text_w = max(18, int(len(edge_text) * 5.4))
        # Inside-bar placement is feasible only if the bar is wider than the
        # text plus a little padding.
        if abs(bw) >= text_w + 8:
            inside = True
        else:
            inside = False
        if inside:
            if e >= 0:
                lx = tip_x - 4
                anchor = "end"
            else:
                lx = tip_x + 4
                anchor = "start"
            d.add(String(lx, y_center - 3,
                         edge_text,
                         fontName=_V2_SERIF_BOLD, fontSize=9,
                         fillColor=colors.white, textAnchor=anchor))
        else:
            side_x = tip_x + 4 if e >= 0 else tip_x - 4
            anchor = "start" if e >= 0 else "end"
            d.add(String(side_x, y_center - 3,
                         edge_text,
                         fontName=_V2_SERIF_BOLD, fontSize=9,
                         fillColor=col, textAnchor=anchor))

    # Caption strip below.
    d.add(String(cx, 8,
                 "EVERY LIVE EDGE  ·  bar length = |edge|, color = side",
                 fontName=_BODY_FONT, fontSize=7,
                 fillColor=INK_MUTED, textAnchor="middle"))
    return d


def _v2_edge_skyline(
    df: pd.DataFrame,
    *,
    width: float = 6.0 * inch,
    height: float = 4.6 * inch,
) -> Drawing:
    """Fallback horizontal bar chart when the slate is too thin for a radar."""
    df = df.sort_values("_e", key=lambda s: s.abs(), ascending=True).reset_index(drop=True)
    n = len(df)
    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 1.4 * inch, 0.4 * inch, 28, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    if n == 0 or plot_h <= 0:
        return d
    max_abs = float(df["_e"].abs().max()) or 1.0
    bar_h = max(8.0, plot_h / max(n, 1) * 0.7)
    row_h = plot_h / max(n, 1)
    cx = pad_l + plot_w / 2.0
    # Center axis.
    d.add(Line(cx, pad_b, cx, pad_b + plot_h,
               strokeColor=V2_HAIRLINE, strokeWidth=0.6))
    for i, r in df.iterrows():
        e = float(r["_e"])
        y = pad_b + i * row_h + (row_h - bar_h) / 2.0
        bw = (plot_w / 2.0) * (abs(e) / max_abs)
        col = POS_GREEN if e > 0 else NEG_RED
        if e >= 0:
            d.add(Rect(cx, y, bw, bar_h, strokeColor=None, fillColor=col))
        else:
            d.add(Rect(cx - bw, y, bw, bar_h, strokeColor=None, fillColor=col))
        # Label on left
        player = str(r.get("player", ""))
        market = str(r.get("model", ""))
        d.add(String(pad_l - 6, y + bar_h / 2 - 3,
                     _short(f"{player} | {market}", 28),
                     fontName=_BOLD_FONT, fontSize=8,
                     fillColor=INK_DARK, textAnchor="end"))
        # Edge value
        side_x = cx + bw + 4 if e >= 0 else cx - bw - 4
        anchor = "start" if e >= 0 else "end"
        d.add(String(side_x, y + bar_h / 2 - 3,
                     _fmt_signed(e, 2),
                     fontName=_BOLD_FONT, fontSize=8,
                     fillColor=col, textAnchor=anchor))
    d.add(String(cx, height - 14, "EDGE SKYLINE",
                 fontName=_BOLD_FONT, fontSize=8.5,
                 fillColor=INK_DARK, textAnchor="middle"))
    return d


# ── 2. Player × Market confidence heatmap ──────────────────────────────────
def _v2_confidence_heatmap(
    edge_df: pd.DataFrame | None,
    *,
    width: float = 7.3 * inch,
    height: float = 3.0 * inch,
) -> Drawing | None:
    """Grid: rows = players, cols = markets, cells colored by signed edge."""
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None
    if "edge" not in edge_df.columns or "player" not in edge_df.columns or "model" not in edge_df.columns:
        return None

    markets = _v2_market_universe(edge_df)
    if not markets:
        return None

    # Player order: by max |edge|, descending — loudest player on top.
    df = edge_df.copy()
    df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
    df["_abs"] = df["_e"].abs()
    player_order = (
        df.dropna(subset=["_e"])
          .groupby("player")["_abs"].max()
          .sort_values(ascending=False)
          .index.tolist()
    )
    if not player_order:
        return None

    max_abs = float(df["_abs"].max()) or 1.0

    # Build a quick lookup: (player, market_lower) -> edge.
    lookup: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        p = str(r.get("player", ""))
        m = str(r.get("model", "")).lower()
        e = r.get("_e")
        if pd.notna(e):
            lookup[(p, m)] = float(e)

    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 1.55 * inch, 0.15 * inch, 24, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_rows = len(player_order)
    n_cols = len(markets)
    if n_rows == 0 or n_cols == 0:
        return d

    cell_w = plot_w / n_cols
    cell_h = plot_h / n_rows

    # Column headers
    market_label = {
        "points": "PTS", "rebounds": "REB", "assists": "AST",
        "threepm": "3PM", "pra": "PRA", "fantasy_score": "FAN",
        "stl_blk": "S+B", "turnovers": "TOV",
    }
    for j, m in enumerate(markets):
        x = pad_l + j * cell_w + cell_w / 2.0
        d.add(String(x, height - pad_t + 6,
                     market_label.get(m, m[:3].upper()),
                     fontName=_BOLD_FONT, fontSize=8,
                     fillColor=INK_DARK, textAnchor="middle"))

    for i, player in enumerate(player_order):
        # Row label — single line, full name (truncated if needed) so it
        # never visually competes with the cell numbers.
        row_y_top = height - pad_t - i * cell_h
        row_y_mid = row_y_top - cell_h / 2.0
        d.add(String(pad_l - 8, row_y_mid - 2, _short(player, 22),
                     fontName=_BOLD_FONT, fontSize=8.5,
                     fillColor=INK_DARK, textAnchor="end"))
        for j, m in enumerate(markets):
            cx = pad_l + j * cell_w
            cy = row_y_top - cell_h
            edge = lookup.get((player, m))
            if edge is None:
                # Empty cell.
                d.add(Rect(cx + 1, cy + 1, cell_w - 2, cell_h - 2,
                           strokeColor=V2_HAIRLINE, strokeWidth=0.5,
                           fillColor=V2_HEAT_NEUTRAL))
                d.add(String(cx + cell_w / 2, cy + cell_h / 2 - 3, "—",
                             fontName=_BODY_FONT, fontSize=8,
                             fillColor=INK_FAINT, textAnchor="middle"))
                continue
            fill = _v2_heat_color(edge, max_abs)
            text_col = _v2_heat_text_color(edge, max_abs)
            d.add(Rect(cx + 1, cy + 1, cell_w - 2, cell_h - 2,
                       strokeColor=WHITE, strokeWidth=0.8, fillColor=fill))
            d.add(String(cx + cell_w / 2, cy + cell_h / 2 - 3,
                         _fmt_signed(edge, 2),
                         fontName=_BOLD_FONT, fontSize=8.5,
                         fillColor=text_col, textAnchor="middle"))

    # Caption
    d.add(String(pad_l, 8, "CONFIDENCE GRID",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=BRAND_ORANGE_DEEP, textAnchor="start"))
    d.add(String(width - pad_r, 8,
                 "cell color = signed edge   |   blank = no live line",
                 fontName=_BODY_FONT, fontSize=7,
                 fillColor=INK_MUTED, textAnchor="end"))
    return d


# Per-player history-table status strings. These mirror the values the app
# layer emits in ``_player_history_lines_vs_outcomes`` and pick which
# placeholder we render when a player has no resolved lines.
_PLAYER_HISTORY_STATUS_HAS_LINES = "has_lines"
_PLAYER_HISTORY_STATUS_NO_API_COVERAGE = "no_api_coverage"
_PLAYER_HISTORY_STATUS_NO_GAME_OVERLAP = "no_game_overlap"
_PLAYER_HISTORY_STATUS_EMPTY_CACHE = "empty_cache"


# Generational suffixes that should never be displayed as a player's
# "last name" on their own — for "Jabari Smith Jr." we want "Smith Jr."
# in the legend, not the bare "Jr." that ``split()[-1]`` produces.
_NAME_SUFFIXES: frozenset[str] = frozenset({
    "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v",
})


def _legend_last_name(full_name: str) -> str:
    """Last token of ``full_name``, with a generational suffix glued back on.

    "Jabari Smith Jr."  → "Smith Jr."
    "LeBron James"       → "James"
    "Karl-Anthony Towns" → "Towns"
    Empty / single-token names fall back to whatever was passed in.
    """
    if not full_name:
        return "?"
    tokens = full_name.split()
    if not tokens:
        return full_name
    if len(tokens) == 1:
        return tokens[0]
    last = tokens[-1]
    if last.lower().rstrip(".") in {s.rstrip(".") for s in _NAME_SUFFIXES}:
        return f"{tokens[-2]} {last}"
    return last


# ── 3. Edge × Confidence quadrant ──────────────────────────────────────────
def _v2_edge_confidence_quadrant(
    edge_df: pd.DataFrame | None,
    sigma_lookup: dict[str, float],
    *,
    width: float = 6.4 * inch,
    height: float = 4.4 * inch,
) -> tuple[Drawing | None, list[dict]]:
    """Scatter of |edge| (y) vs book depth (x) with quadrant labels.

    Returns (drawing, legend_rows). Each legend row is a dict with keys
    ``rank``, ``player``, ``market``, ``edge``, ``side`` so the call site
    can render a richly formatted index panel.
    """
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None, []
    if "edge" not in edge_df.columns:
        return None, []
    df = edge_df.copy()
    df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
    df["_abs"] = df["_e"].abs()
    if "books" in df.columns:
        df["_books"] = pd.to_numeric(df["books"], errors="coerce")
    else:
        df["_books"] = float("nan")
    df = df.dropna(subset=["_e"])
    if df.empty:
        return None, []
    df["_books"] = df["_books"].fillna(1.0).clip(lower=1.0)
    # Confidence proxy via sigma for marker coloring.
    def _conf_for(row) -> int | None:
        sig = sigma_lookup.get(str(row.get("model", "")).lower())
        return _confidence_score(row.get("edge"), sig, row.get("books"))
    df["_conf"] = df.apply(_conf_for, axis=1)
    df = df.sort_values("_abs", ascending=False).reset_index(drop=True)
    df = df.head(7)

    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 0.65 * inch, 0.25 * inch, 28, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_abs = max(0.5, float(df["_abs"].max()))
    max_books = max(8.0, float(df["_books"].max()))

    # Plot background
    d.add(Rect(pad_l, pad_b, plot_w, plot_h,
               strokeColor=V2_HAIRLINE, strokeWidth=0.6,
               fillColor=V2_BG_CREAM))

    # Quadrant midpoints (x = books mid, y = edge mid)
    x_mid = pad_l + plot_w * 0.5
    y_mid = pad_b + plot_h * 0.4
    d.add(Line(x_mid, pad_b, x_mid, pad_b + plot_h,
               strokeColor=V2_HAIRLINE, strokeWidth=0.5,
               strokeDashArray=[2, 2]))
    d.add(Line(pad_l, y_mid, pad_l + plot_w, y_mid,
               strokeColor=V2_HAIRLINE, strokeWidth=0.5,
               strokeDashArray=[2, 2]))

    # Quadrant tags rendered OUTSIDE the plot rect so dots never collide
    # with the labels. Top tags above the rect; bottom tags below.
    top_label_y = pad_b + plot_h + 10
    d.add(String(pad_l + 4, top_label_y, "SLEEPER",
                 fontName=_V2_SERIF_BOLD, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="start"))
    d.add(String(pad_l + plot_w - 4, top_label_y, "HEADLINE",
                 fontName=_V2_SERIF_BOLD, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="end"))
    bot_label_y = pad_b - 14
    d.add(String(pad_l + 4, bot_label_y, "SKIP",
                 fontName=_V2_SERIF_BOLD, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="start"))
    d.add(String(pad_l + plot_w - 4, bot_label_y, "CROWD PLAY",
                 fontName=_V2_SERIF_BOLD, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="end"))

    # Axis labels
    d.add(String(pad_l + plot_w / 2, 12, "MARKET DEPTH (books)",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="middle"))
    d.add(String(pad_l - 30, pad_b + plot_h / 2,
                 "|EDGE|",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="middle"))

    # Markers with directly attached labels (last name + market). The label
    # placement alternates left/right of the dot to dodge collisions; no
    # separate numeric legend is needed.
    market_short = {
        "points": "PTS", "rebounds": "REB", "assists": "AST",
        "threepm": "3PM", "pra": "PRA", "fantasy_score": "FAN",
        "stl_blk": "S+B", "turnovers": "TOV",
    }
    legend: list[dict] = []

    # ── Compute initial dot positions, then iteratively relax to dodge ──
    # collisions. Marker radius is 8 so we need ≥ 18 px center-to-center
    # spacing for a clean visual gap.
    MIN_DIST = 19.0
    BOUND_L = pad_l + 12
    BOUND_R = pad_l + plot_w - 12
    BOUND_B = pad_b + 12
    BOUND_T = pad_b + plot_h - 12

    raw: list[list[float]] = []
    rows_data: list = []
    for _, row in df.iterrows():
        e = float(row["_e"])
        books = float(row["_books"])
        x0 = pad_l + plot_w * (books / max_books)
        y0 = pad_b + plot_h * (abs(e) / max_abs) * 0.92 + plot_h * 0.06
        raw.append([x0, y0])
        rows_data.append((e, row))

    # Force-directed relaxation (~30 iterations is more than enough for ≤7 dots).
    import math as _math
    n = len(raw)
    for _ in range(40):
        moved = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = raw[j][0] - raw[i][0]
                dy = raw[j][1] - raw[i][1]
                dist = _math.hypot(dx, dy)
                if dist < MIN_DIST:
                    if dist < 0.01:
                        # Identical points: nudge in opposing diagonals
                        dx, dy, dist = 1.0, 1.0, 1.4142
                    overlap = (MIN_DIST - dist) / 2.0 + 0.5
                    ux = dx / dist
                    uy = dy / dist
                    raw[i][0] -= ux * overlap
                    raw[i][1] -= uy * overlap
                    raw[j][0] += ux * overlap
                    raw[j][1] += uy * overlap
                    moved += overlap
        # Clamp into plot rect after each pass
        for i in range(n):
            raw[i][0] = max(BOUND_L, min(BOUND_R, raw[i][0]))
            raw[i][1] = max(BOUND_B, min(BOUND_T, raw[i][1]))
        if moved < 0.5:
            break

    for rank, ((x, y), (e, row)) in enumerate(zip(raw, rows_data), start=1):
        col = POS_GREEN if e > 0 else NEG_RED
        # Numbered marker: filled disk, white inner, rank numeral.
        d.add(Circle(x, y, 8.0, strokeColor=None, fillColor=col))
        d.add(Circle(x, y, 6.5, strokeColor=None, fillColor=WHITE))
        d.add(String(x, y - 2.5, str(rank),
                     fontName=_BOLD_FONT, fontSize=7.5,
                     fillColor=col, textAnchor="middle"))

        player = str(row.get("player", ""))
        last_name = _legend_last_name(player)
        market = market_short.get(str(row.get("model", "")).lower(),
                                  str(row.get("model", ""))[:3].upper())
        legend.append({
            "rank": rank,
            "player": _short(last_name, 14),
            "full_player": player,
            "market": market,
            "edge": e,
            "side": "OVER" if e > 0 else "UNDER",
        })

    return d, legend


# ── 4. Forecast fan / cone chart ──────────────────────────────────────────
def _v2_forecast_fan(
    *,
    projection: float,
    sigma: float | None,
    line: float | None,
    label: str = "",
    width: float = 4.8 * inch,
    height: float = 2.4 * inch,
) -> Drawing | None:
    """Single-stat forecast fan with three shaded confidence bands.

    Center vertical line = model projection.
    Shaded bands at ±0.5σ, ±1σ, ±1.5σ (light → dark orange).
    Posted line drawn as a vertical dashed marker.
    """
    if sigma is None or sigma <= 0:
        return None
    try:
        mu = float(projection)
    except (TypeError, ValueError):
        return None
    if pd.isna(mu):
        return None

    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 0.45 * inch, 0.45 * inch, 22, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # X-axis range: mu ± 2σ; widen if line falls outside.
    span = 2.2 * sigma
    x_lo = mu - span
    x_hi = mu + span
    if line is not None and pd.notna(line):
        try:
            lf = float(line)
            if lf < x_lo: x_lo = lf - 0.5 * sigma
            if lf > x_hi: x_hi = lf + 0.5 * sigma
        except (TypeError, ValueError):
            pass
    x_range = max(x_hi - x_lo, 1e-6)

    def x_pos(v: float) -> float:
        return pad_l + plot_w * (v - x_lo) / x_range

    # Bands: 3 nested rectangles, lightest on outside.
    band_specs = [
        (1.5, colors.HexColor("#ffeed8")),
        (1.0, colors.HexColor("#ffd6a8")),
        (0.5, colors.HexColor("#ffb878")),
    ]
    for k, fill in band_specs:
        x1 = x_pos(mu - k * sigma)
        x2 = x_pos(mu + k * sigma)
        d.add(Rect(x1, pad_b, x2 - x1, plot_h,
                   strokeColor=None, fillColor=fill))

    # Center projection line
    d.add(Line(x_pos(mu), pad_b, x_pos(mu), pad_b + plot_h,
               strokeColor=BRAND_ORANGE_DEEP, strokeWidth=1.6))
    d.add(String(x_pos(mu), pad_b + plot_h + 4, _fmt(mu, 2),
                 fontName=_BOLD_FONT, fontSize=8.5,
                 fillColor=BRAND_ORANGE_DEEP, textAnchor="middle"))

    # Posted line marker
    if line is not None and pd.notna(line):
        try:
            lf = float(line)
            d.add(Line(x_pos(lf), pad_b, x_pos(lf), pad_b + plot_h,
                       strokeColor=INK_DARK, strokeWidth=1.2,
                       strokeDashArray=[3, 2]))
            d.add(String(x_pos(lf), pad_b - 12, f"line {_fmt(lf, 1)}",
                         fontName=_BOLD_FONT, fontSize=7.5,
                         fillColor=INK_DARK, textAnchor="middle"))
        except (TypeError, ValueError):
            pass

    # Bottom axis ticks at ±1σ, ±2σ
    for k in (-2, -1, 0, 1, 2):
        v = mu + k * sigma
        if x_lo <= v <= x_hi:
            xv = x_pos(v)
            d.add(Line(xv, pad_b, xv, pad_b - 3,
                       strokeColor=INK_FAINT, strokeWidth=0.4))
            if k != 0:
                d.add(String(xv, pad_b - 22, f"{'+' if k > 0 else ''}{k}σ",
                             fontName=_BODY_FONT, fontSize=6.5,
                             fillColor=INK_FAINT, textAnchor="middle"))

    # Title strip
    d.add(String(pad_l, height - 12, "FORECAST DISTRIBUTION",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=BRAND_ORANGE_DEEP, textAnchor="start"))
    if label:
        d.add(String(width - pad_r, height - 12, label.upper(),
                     fontName=_BOLD_FONT, fontSize=7.5,
                     fillColor=INK_MUTED, textAnchor="end"))
    # Legend bottom-right
    d.add(String(width - pad_r, 4, "BANDS: ±0.5σ  ±1σ  ±1.5σ",
                 fontName=_BODY_FONT, fontSize=6.5,
                 fillColor=INK_FAINT, textAnchor="end"))
    return d


# ── 5. Hit-rate streak strip ───────────────────────────────────────────────
def _v2_hitrate_streak(
    history_df: pd.DataFrame | None,
    *,
    metric: str | None = None,
    width: float = 2.6 * inch,
    height: float = 1.0 * inch,
) -> Drawing | None:
    """Last-10 results as colored chips (green=Over, red=Under, grey=Push).

    When ``metric`` is provided, only that market's history is used; otherwise
    all metrics are mixed (most-recent-first).
    """
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return None
    if "result" not in history_df.columns:
        return None
    df = history_df.copy()
    if metric and "metric" in df.columns:
        df = df[df["metric"].astype(str).str.lower() == str(metric).lower()]
    if df.empty:
        return None
    if "game_date" in df.columns:
        df = df.sort_values("game_date", ascending=False)
    df = df.head(10)

    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 4, 4, 18, 14
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(df)
    if n == 0:
        return d
    chip_gap = 4
    chip_w = (plot_w - chip_gap * (n - 1)) / max(n, 1)
    chip_w = max(8.0, min(chip_w, 26.0))

    overs = 0
    chips = list(df["result"].astype(str).tolist())
    for i, res in enumerate(chips):
        x = pad_l + i * (chip_w + chip_gap)
        y = pad_b + 4
        rl = res.strip().lower()
        if "over" in rl or rl == "win":
            col = POS_GREEN; symb = "O"; overs += 1
        elif "under" in rl or rl == "loss":
            col = NEG_RED; symb = "U"
        else:
            col = INK_FAINT; symb = "P"
        d.add(Rect(x, y, chip_w, plot_h - 8,
                   strokeColor=None, fillColor=col))
        d.add(String(x + chip_w / 2, y + (plot_h - 8) / 2 - 3, symb,
                     fontName=_BOLD_FONT, fontSize=7.5,
                     fillColor=WHITE, textAnchor="middle"))

    # Title + summary line
    over_pct = (overs / n) * 100 if n else 0
    title = f"LAST {n}"
    if metric:
        title = f"LAST {n} · {str(metric).upper()}"
    d.add(String(pad_l, height - 6, title,
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=BRAND_ORANGE_DEEP, textAnchor="start"))
    d.add(String(width - pad_r, height - 6, f"{over_pct:.0f}% Over",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=POS_GREEN if over_pct >= 50 else NEG_RED,
                 textAnchor="end"))
    return d


# ── 6. Reliability gauge + small multiples ─────────────────────────────────
def _v2_reliability_gauge(
    metrics: pd.DataFrame | None,
    *,
    width: float = 3.4 * inch,
    height: float = 2.4 * inch,
    label: str = "TONIGHT'S MODEL TRUST",
) -> Drawing | None:
    """Composite gauge with median R² needle and tier-colored arc."""
    import math
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return None
    r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
    if not r2_col:
        return None
    r2 = pd.to_numeric(metrics[r2_col], errors="coerce").dropna()
    if r2.empty:
        return None
    median_r2 = float(r2.median())

    d = Drawing(width, height)
    cx = width / 2.0
    # Geometry: keep the arc fully inside the canvas. Arc spans from cy to
    # cy+r_outer (top), so we anchor cy near the bottom and cap r_outer to
    # available headroom (minus a small margin for the 0.0/1.0 ticks).
    r_outer = min(width * 0.42, (height - 60) * 0.95)
    r_outer = max(r_outer, 40.0)
    cy = 36.0  # bottom margin reserves space for the readout block
    r_inner = r_outer * 0.62

    # Arc segments from 180° to 0° (left → right), 60 micro-segments.
    n_seg = 60
    for s in range(n_seg):
        t0 = math.pi - math.pi * (s / n_seg)
        t1 = math.pi - math.pi * ((s + 1) / n_seg)
        frac = (s + 0.5) / n_seg
        if frac < 0.35: col = V2_TIER_LOW
        elif frac < 0.65: col = V2_TIER_MID
        else: col = V2_TIER_HIGH
        # Approximate arc segment as a thin polygon (4 points).
        from reportlab.graphics.shapes import Polygon
        x0o = cx + r_outer * math.cos(t0)
        y0o = cy + r_outer * math.sin(t0)
        x1o = cx + r_outer * math.cos(t1)
        y1o = cy + r_outer * math.sin(t1)
        x0i = cx + r_inner * math.cos(t0)
        y0i = cy + r_inner * math.sin(t0)
        x1i = cx + r_inner * math.cos(t1)
        y1i = cy + r_inner * math.sin(t1)
        d.add(Polygon(points=[x0o, y0o, x1o, y1o, x1i, y1i, x0i, y0i],
                      strokeColor=None, fillColor=col))

    # Tier labels under arc
    d.add(String(cx + r_outer * math.cos(math.pi) * 0.95,
                 cy + r_outer * math.sin(math.pi) * 0.0 - 14,
                 "0.0", fontName=_BODY_FONT, fontSize=7,
                 fillColor=INK_FAINT, textAnchor="end"))
    d.add(String(cx + r_outer * 0.95, cy - 14, "1.0",
                 fontName=_BODY_FONT, fontSize=7,
                 fillColor=INK_FAINT, textAnchor="start"))

    # Needle
    needle_t = math.pi - math.pi * max(0.0, min(1.0, median_r2))
    nx = cx + (r_outer - 4) * math.cos(needle_t)
    ny = cy + (r_outer - 4) * math.sin(needle_t)
    d.add(Line(cx, cy, nx, ny, strokeColor=V2_INK_HERO, strokeWidth=2.6))
    d.add(Circle(cx, cy, 5.0, strokeColor=V2_INK_HERO, strokeWidth=0.6,
                 fillColor=BRAND_ORANGE))

    # Center readout (below the dial center)
    d.add(String(cx, cy - 14, f"{median_r2:.2f}",
                 fontName=_V2_SERIF_BOLD, fontSize=22,
                 fillColor=V2_INK_HERO, textAnchor="middle"))
    d.add(String(cx, cy - 28, "MEDIAN R²",
                 fontName=_BOLD_FONT, fontSize=7.5,
                 fillColor=INK_MUTED, textAnchor="middle"))

    # Optional caption above the arc — only render when there is room so it
    # never overlaps the colored ring.
    if height - (cy + r_outer) >= 14:
        d.add(String(cx, height - 8, label,
                     fontName=_BOLD_FONT, fontSize=8,
                     fillColor=BRAND_ORANGE_DEEP, textAnchor="middle"))
    return d


def _v2_reliability_multiples(
    metrics: pd.DataFrame | None,
    *,
    width: float = 7.3 * inch,
    height: float = 1.9 * inch,
) -> Drawing | None:
    """4×2 grid of mini half-gauges, one per model."""
    import math
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return None
    name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
    r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
    if not name_col or not r2_col:
        return None

    df = metrics.copy()
    df["_r2"] = pd.to_numeric(df[r2_col], errors="coerce")
    df = df.dropna(subset=["_r2"]).sort_values("_r2", ascending=False).head(8)
    if df.empty:
        return None

    d = Drawing(width, height)
    cols = 4
    rows = 2
    cell_w = width / cols
    cell_h = height / rows

    for idx, (_, row) in enumerate(df.iterrows()):
        ci = idx % cols
        ri = idx // cols
        cx = ci * cell_w + cell_w / 2.0
        cy = (rows - ri - 1) * cell_h + cell_h * 0.35
        r2v = float(row["_r2"])
        name = str(row[name_col])
        r_outer = min(cell_w, cell_h) * 0.36
        r_inner = r_outer * 0.62
        n_seg = 30
        for s in range(n_seg):
            t0 = math.pi - math.pi * (s / n_seg)
            t1 = math.pi - math.pi * ((s + 1) / n_seg)
            frac = (s + 0.5) / n_seg
            seg_filled = frac <= max(0.0, min(1.0, r2v))
            if not seg_filled:
                col = V2_HEAT_NEUTRAL
            elif r2v < 0.35:
                col = V2_TIER_LOW
            elif r2v < 0.55:
                col = V2_TIER_MID
            else:
                col = V2_TIER_HIGH
            from reportlab.graphics.shapes import Polygon
            x0o = cx + r_outer * math.cos(t0)
            y0o = cy + r_outer * math.sin(t0)
            x1o = cx + r_outer * math.cos(t1)
            y1o = cy + r_outer * math.sin(t1)
            x0i = cx + r_inner * math.cos(t0)
            y0i = cy + r_inner * math.sin(t0)
            x1i = cx + r_inner * math.cos(t1)
            y1i = cy + r_inner * math.sin(t1)
            d.add(Polygon(points=[x0o, y0o, x1o, y1o, x1i, y1i, x0i, y0i],
                          strokeColor=None, fillColor=col))
        # Readout
        d.add(String(cx, cy - 4, f"{r2v:.2f}",
                     fontName=_V2_SERIF_BOLD, fontSize=14,
                     fillColor=V2_INK_HERO, textAnchor="middle"))
        d.add(String(cx, cy - 18, _short(name, 14).upper(),
                     fontName=_BOLD_FONT, fontSize=7,
                     fillColor=INK_MUTED, textAnchor="middle"))
    return d


# ═══════════════════════════════════════════════════════════════════════════
# V2 SECTION ASSEMBLERS
# ═══════════════════════════════════════════════════════════════════════════

# ── Page 1 — Editorial Cover ───────────────────────────────────────────────
def _v2_cover_flowables(
    *,
    roster: dict[str, list[str]],
    meta: _ReportMeta,
    edge_df: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    styles: dict,
) -> list:
    flow: list = []
    eyebrow_text = (
        f"HOOPLYTICS  ·  ROSTER ANALYTICS  ·  {meta.generated_at.upper()}"
    )
    flow.append(_para(eyebrow_text, styles["v2_cover_eyebrow"]))
    flow.append(Spacer(1, 16))
    flow.append(_para("Tonight's Slate.", styles["v2_cover_title"]))
    flow.append(Spacer(1, 8))

    top = _v2_top_signal(edge_df)
    if top:
        side = _side_display(top["side"])
        dek = (
            f"{top['player']} {top['model']} {side} {_fmt_signed(top['edge'], 2)} "
            f"is the loudest mispricing on the board tonight."
        )
    else:
        dek = "No live mispricings on the board right now — model output only."
    flow.append(_para(dek, styles["v2_cover_dek"]))
    flow.append(Spacer(1, 22))

    radar = _v2_slate_radar(edge_df, width=6.3 * inch, height=4.0 * inch)
    if radar is not None:
        flow.append(radar)
        flow.append(Spacer(1, 18))
    else:
        empty = Table([[Paragraph(
            "<font size='10' color='#6b7686'>"
            "No live edges available — start by entering an Odds API key in "
            "the Streamlit sidebar to populate tonight's board."
            "</font>",
            styles["body"],
        )]], colWidths=[6.3 * inch], rowHeights=[1.6 * inch])
        empty.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_ORANGE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 18),
            ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ]))
        flow.append(empty)
        flow.append(Spacer(1, 18))

    median_r2 = float("nan")
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        if r2_col:
            median_r2 = float(pd.to_numeric(metrics[r2_col], errors="coerce").median())
    edge_count = int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0
    tiles_data = [
        ("PLAYERS", str(len(roster or {}))),
        ("LIVE SIGNALS", str(edge_count)),
        ("MEDIAN R²", _fmt(median_r2, 2) if not pd.isna(median_r2) else "—"),
    ]

    def _tile(label: str, value: str) -> Table:
        v = Paragraph(
            f"<font size='34' color='#0a0e14' name='{_V2_SERIF_BOLD}'><b>{value}</b></font>",
            ParagraphStyle(
                "v2_kpi_inner", fontName=_V2_SERIF_BOLD, fontSize=34,
                leading=38, textColor=V2_INK_HERO, alignment=TA_LEFT,
            ),
        )
        l = Paragraph(
            f"<font size='7.5' color='#6b7686' name='{_BOLD_FONT}'><b>{label}</b></font>",
            ParagraphStyle(
                "v2_kpi_label_inner", fontName=_BOLD_FONT, fontSize=7.5,
                leading=10, textColor=INK_MUTED, alignment=TA_LEFT,
            ),
        )
        t = Table([[l], [v]], colWidths=[2.0 * inch],
                  rowHeights=[14, 42])
        t.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 1.0, BRAND_ORANGE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (0, 0), 8),
            ("BOTTOMPADDING", (0, 0), (0, 0), 0),
            ("TOPPADDING", (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (0, 1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    strip = Table(
        [[_tile(lbl, val) for lbl, val in tiles_data]],
        colWidths=[2.1 * inch] * 3,
    )
    strip.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(strip)
    return flow


# ── Page 2 — Tonight's Setup ──────────────────────────────────────────────
def _v2_tonight_setup_flowables(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    ai_sections: dict | None,
    styles: dict,
) -> list:
    from reportlab.lib.enums import TA_RIGHT
    flow: list = []
    flow.append(_AnchorFlowable("v2-setup", "Tonight's Setup", level=0))
    flow.append(_para("PAGE ONE  ·  THE BLUF", styles["v2_section_eyebrow"]))
    flow.append(_para("Tonight's Setup.", styles["v2_section_title"]))
    flow.append(_para(
        "The loudest mispricing, the slate's directional posture, and how "
        "much trust to place in the model tonight — all on one page.",
        styles["v2_section_dek"],
    ))
    flow.append(Spacer(1, 8))

    top = _v2_top_signal(edge_df)
    if top is not None:
        side_disp = _side_display(top["side"])
        side_col = POS_GREEN if top["edge"] > 0 else NEG_RED
        edge_chip = Paragraph(
            f"<font size='32' name='{_V2_SERIF_BOLD}' color='#{side_col.hexval()[2:].lower()}'>"
            f"<b>{_fmt_signed(top['edge'], 2)}</b></font>",
            ParagraphStyle(
                "v2_edge_chip", fontName=_V2_SERIF_BOLD, fontSize=32,
                leading=36, textColor=side_col, alignment=TA_RIGHT,
            ),
        )
        line_str = _fmt(top['line'], 1)
        proj_str = _fmt(top['projection'], 2)
        meta_bits = [_safe_text(top['model'].upper()), side_disp]
        if line_str != "\u2014":
            meta_bits.append(f"line {line_str}")
        if proj_str != "\u2014":
            meta_bits.append(f"proj {proj_str}")
        meta_line = "  /  ".join(meta_bits)
        left_block = Paragraph(
            f"<font size='8' name='{_BOLD_FONT}' color='#ff7a18'>"
            f"<b>TOP SIGNAL</b></font><br/><br/>"
            f"<font size='22' name='{_V2_SERIF_BOLD}' color='#ffffff'>"
            f"<b>{_safe_text(top['player'])}</b></font><br/>"
            f"<font size='10' name='{_BOLD_FONT}' color='#9aa3b2'>"
            f"<b>{meta_line}</b>"
            f"</font>",
            ParagraphStyle(
                "v2_top_left", fontName=_BODY_FONT, fontSize=10,
                leading=15, textColor=WHITE, alignment=TA_LEFT,
            ),
        )
        hero = Table(
            [[left_block, edge_chip]],
            colWidths=[5.0 * inch, 2.3 * inch],
        )
        hero.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_INK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 22),
            ("RIGHTPADDING", (0, 0), (-1, -1), 22),
            ("TOPPADDING", (0, 0), (-1, -1), 20),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
            ("LINEBELOW", (0, 0), (-1, -1), 3, BRAND_ORANGE),
        ]))
        flow.append(hero)
    flow.append(Spacer(1, 14))

    lean, pos, neg = _slate_lean_label(edge_df)
    median_r2 = float("nan")
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        if r2_col:
            median_r2 = float(pd.to_numeric(metrics[r2_col], errors="coerce").median())

    def _insight_tile(eyebrow: str, headline: str, sub: str,
                      accent: colors.Color) -> Table:
        body = Paragraph(
            f"<font size='8' name='{_BOLD_FONT}' color='#cc5a00'><b>{eyebrow}</b></font>"
            f"<br/><br/>"
            f"<font size='17' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
            f"<b>{_safe_text(headline)}</b></font>"
            f"<br/>"
            f"<font size='8.5' color='#6b7686'>{_safe_text(sub)}</font>",
            ParagraphStyle(
                "v2_insight_inner", fontName=_BODY_FONT, fontSize=10,
                leading=14, textColor=INK_BODY,
            ),
        )
        t = Table([[body]], colWidths=[2.32 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEABOVE", (0, 0), (-1, 0), 2.0, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    if top is not None:
        proj_str = _fmt(top['projection'], 2)
        line_str = _fmt(top['line'], 1)
        if proj_str == "—" and line_str == "—":
            proj_vs_line = "—"
        elif proj_str == "—":
            proj_vs_line = f"line {line_str}"
        elif line_str == "—":
            proj_vs_line = f"proj {proj_str}"
        else:
            proj_vs_line = f"{proj_str} vs {line_str}"
        proj_sub = f"Edge of {_fmt_signed(top['edge'], 2)} on {top['model']}."
    else:
        proj_vs_line = "—"
        proj_sub = "No live lines."
    posture_sub = (
        f"{pos} ABOVE / {neg} BELOW across "
        f"{int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0} mapped lines."
    )
    if not pd.isna(median_r2):
        if median_r2 >= 0.55:
            trust_headline = "Strong trust"
        elif median_r2 >= 0.40:
            trust_headline = "Solid trust"
        else:
            trust_headline = "Caution mode"
        trust_sub = f"Median R² of {median_r2:.2f} across the bundle."
    else:
        trust_headline = "Unknown"
        trust_sub = "No metrics available."

    tiles = Table(
        [[
            _insight_tile("MODEL VS LINE", proj_vs_line, proj_sub, BRAND_ORANGE),
            _insight_tile("SLATE POSTURE", lean.title(), posture_sub, V2_ACCENT_TEAL),
            _insight_tile("MODEL TRUST", trust_headline, trust_sub, V2_TIER_HIGH),
        ]],
        colWidths=[2.4 * inch, 2.4 * inch, 2.5 * inch],
    )
    tiles.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(tiles)
    flow.append(Spacer(1, 16))

    brief_text = _deterministic_summary_text(
        roster=roster, metrics=metrics, edge_df=edge_df,
    )
    brief_block = [
        _para("THE BRIEF", styles["v2_section_eyebrow"]),
        Spacer(1, 4),
        _para(brief_text, styles["v2_body"]),
    ]
    if ai_sections and ai_sections.get("executive_summary"):
        ai_text = str(ai_sections["executive_summary"]).strip()
        if ai_text:
            brief_block.append(Spacer(1, 6))
            brief_block.append(_para("AI SCOUT  ·  CONTEXT", styles["v2_section_eyebrow"]))
            brief_block.append(Spacer(1, 4))
            brief_block.append(_para(ai_text, styles["v2_body"]))

    panel = Table([[brief_block]], colWidths=[7.3 * inch])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_ORANGE_DEEP),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(panel)
    return flow


# ── Page 2.5 — Tonight's Matchups ────────────────────────────────────────
def _v2_matchups_flowables(
    *,
    matchup_predictions: list[dict] | None,
    ai_sections: dict | None,
    styles: dict,
) -> list:
    """Render one card per game on the slate with model-projected scores,
    home win probability, and (when present) the AI Scout matchup narrative.

    ``matchup_predictions`` is the JSON-friendly list produced by
    :func:`hooplytics.matchups.to_grounding_payload` so the renderer never has
    to reach back into the dataclass.
    """
    flow: list = []
    if not matchup_predictions:
        return flow

    flow.append(PageBreak())
    flow.append(_AnchorFlowable("v2-matchups", "Tonight's Matchups", level=0))
    flow.append(_para("THE MATCHUPS  ·  TEAM-VS-TEAM FORECAST", styles["v2_section_eyebrow"]))
    flow.append(_para("Tonight's Matchups.", styles["v2_section_title"]))
    flow.append(_para(
        "Only the games on tonight's slate where you have rostered players. "
        "Each card pairs the model's read with the consensus market line — "
        "and falls back to market-anchored numbers when rotation coverage is "
        "too thin to produce a reliable team rollup.",
        styles["v2_section_dek"],
    ))
    flow.append(Spacer(1, 6))

    ai_matchups = (ai_sections or {}).get("matchups") or {}

    for entry in matchup_predictions:
        if not isinstance(entry, dict):
            continue
        flow.append(KeepTogether(_v2_matchup_card(entry, ai_matchups, styles)))
        flow.append(Spacer(1, 10))

    return flow


def _v2_matchup_card(
    entry: dict,
    ai_matchups: dict,
    styles: dict,
) -> list:
    """Single matchup card. Rendered as a vertical stack of:

    1. Team header strip — projected scores when rotation coverage is solid,
       or just team names when coverage is thin.
    2. Win-probability bar (only when we trust the rollup).
    3. Three metric tiles — switches between model-derived numbers (good
       coverage) and market-anchored numbers (thin coverage) so we never
       print misleading totals like "Toronto 0.0".
    4. Optional AI Scout narrative panel.
    """
    from reportlab.lib.enums import TA_RIGHT
    home = str(entry.get("home_team", "") or "")
    away = str(entry.get("away_team", "") or "")
    matchup_label = str(entry.get("matchup") or f"{away} @ {home}")
    home_pts = float(entry.get("model_home_pts") or 0.0)
    away_pts = float(entry.get("model_away_pts") or 0.0)
    spread = float(entry.get("model_spread") or 0.0)
    total = float(entry.get("model_total") or 0.0)
    p_home = float(entry.get("model_home_win_prob") or 0.5)
    p_away = float(entry.get("model_away_win_prob") or 1.0 - p_home)
    confidence = str(entry.get("confidence") or "medium").lower()
    market_spread = entry.get("market_home_spread")
    market_total = entry.get("market_total")
    market_p_home = entry.get("market_home_win_prob")
    spread_edge = entry.get("spread_edge_vs_market")
    total_edge = entry.get("total_edge_vs_market")
    upset = bool(entry.get("upset_flag"))

    # When rotation coverage is too thin, the model team totals are dominated
    # by whichever side happens to be in the user's modeling frame. In that
    # case we pivot to a market-anchored card: header without scores, no
    # win-prob bar (unless market moneylines are available), and tiles that
    # surface market data + rostered players instead of nonsense rollups.
    rollup_trustworthy = confidence in {"high", "medium", "low"}

    # Win-prob source: model when rollup is trusted; market moneyline (already
    # de-vigged in matchups.attach_market_lines) when not. Fall back to None
    # (suppress the bar entirely) when neither is available.
    if rollup_trustworthy:
        wp_home = p_home
        wp_source = "model"
    elif isinstance(market_p_home, (int, float)):
        wp_home = float(market_p_home)
        wp_source = "market"
    else:
        wp_home = None
        wp_source = ""
    wp_away = (1.0 - wp_home) if wp_home is not None else None
    favored_home = (wp_home or 0.5) >= 0.5

    # ── Header strip: AWAY  vs  HOME ──
    # Show projected scores only when the rollup is trustworthy. Otherwise
    # just team names (no fake scores).
    def _team_block(eyebrow: str, name: str, score: float | None) -> Table:
        rows: list[list] = [
            [Paragraph(
                f"<font size='8' name='{_BOLD_FONT}' color='#9aa3b2'>"
                f"<b>{eyebrow}</b></font>",
                ParagraphStyle("v2_match_eyebrow", fontName=_BOLD_FONT,
                               fontSize=8, leading=10, textColor=INK_MUTED,
                               alignment=TA_LEFT),
            )],
            [Paragraph(
                f"<font size='14' name='{_V2_SERIF_BOLD}' color='#ffffff'>"
                f"<b>{_safe_text(name)}</b></font>",
                ParagraphStyle("v2_match_team", fontName=_V2_SERIF_BOLD,
                               fontSize=14, leading=18, textColor=WHITE,
                               alignment=TA_LEFT),
            )],
        ]
        if score is not None:
            rows.append([Paragraph(
                f"<font size='30' name='{_V2_SERIF_BOLD}' color='#ffffff'>"
                f"<b>{score:.1f}</b></font>",
                ParagraphStyle("v2_match_score", fontName=_V2_SERIF_BOLD,
                               fontSize=30, leading=34, textColor=WHITE,
                               alignment=TA_LEFT),
            )])
        t = Table(rows, colWidths=[3.0 * inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return t

    score_away = away_pts if rollup_trustworthy else None
    score_home = home_pts if rollup_trustworthy else None
    away_block = _team_block("AWAY", away, score_away)
    home_block = _team_block("HOME", home, score_home)
    sep = Paragraph(
        f"<font size='14' name='{_BOLD_FONT}' color='#ff7a18'><b>vs</b></font>",
        ParagraphStyle("v2_match_vs", fontName=_BOLD_FONT, fontSize=14,
                       leading=16, textColor=BRAND_ORANGE, alignment=TA_CENTER,
                       wordWrap=None),
    )
    header = Table(
        [[away_block, sep, home_block]],
        colWidths=[3.0 * inch, 0.9 * inch, 3.0 * inch],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), V2_BG_INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))

    # ── Win-prob bar (rendered only when we have a trustworthy probability) ──
    # Team labels live in a separate row ABOVE the bar so a narrow segment
    # never forces the team name to wrap one character per line.
    bar_flow: list = []
    if wp_home is not None and wp_away is not None:
        wp_label_total = 7.3 * inch  # full inner width of the card
        labels = Table(
            [[
                Paragraph(
                    f"<font size='8' name='{_BOLD_FONT}' color='#6b7686'>"
                    f"<b>{_safe_text(away)} · {wp_away*100:.0f}%</b></font>",
                    ParagraphStyle("v2_bar_label_away", fontName=_BOLD_FONT,
                                   fontSize=8, leading=11, textColor=INK_MUTED,
                                   alignment=TA_LEFT),
                ),
                Paragraph(
                    f"<font size='7.5' name='{_BOLD_FONT}' color='#cc5a00'>"
                    f"<b>WIN PROBABILITY · {wp_source.upper()}</b></font>",
                    ParagraphStyle("v2_bar_eyebrow", fontName=_BOLD_FONT,
                                   fontSize=7.5, leading=10,
                                   textColor=BRAND_ORANGE_DEEP, alignment=TA_CENTER),
                ),
                Paragraph(
                    f"<font size='8' name='{_BOLD_FONT}' color='#6b7686'>"
                    f"<b>{wp_home*100:.0f}% · {_safe_text(home)}</b></font>",
                    ParagraphStyle("v2_bar_label_home", fontName=_BOLD_FONT,
                                   fontSize=8, leading=11, textColor=INK_MUTED,
                                   alignment=TA_RIGHT),
                ),
            ]],
            colWidths=[wp_label_total * 0.4, wp_label_total * 0.2, wp_label_total * 0.4],
        )
        labels.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        bar_left_w = max(0.02, min(0.98, wp_away))
        bar_right_w = 1.0 - bar_left_w
        away_bar_color = V2_TIER_LOW if not favored_home else INK_MUTED
        home_bar_color = V2_TIER_HIGH if favored_home else INK_MUTED
        bar = Table(
            [[Paragraph("", styles["body"]), Paragraph("", styles["body"])]],
            colWidths=[bar_left_w * wp_label_total, bar_right_w * wp_label_total],
            rowHeights=[0.18 * inch],
        )
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), away_bar_color),
            ("BACKGROUND", (1, 0), (1, 0), home_bar_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        bar_flow = [Spacer(1, 8), labels, bar]

    # ── Metric tiles row ──
    fav_team = home if favored_home else away
    if rollup_trustworthy:
        # Model-derived favorite tile.
        fav_pct = max(p_home, p_away) * 100.0
        spread_str = f"{abs(spread):.1f}"
        spread_sign_team = home if spread >= 0 else away
        if confidence == "low":
            fav_eyebrow = "LEAN"
        elif upset:
            fav_eyebrow = "UPSET WATCH"
        else:
            fav_eyebrow = "MODEL FAVORS"
        fav_headline = _safe_text(fav_team)
        fav_sub = f"Model: {spread_sign_team} by {spread_str} · win prob {fav_pct:.0f}%"
        if market_spread is not None:
            fav_sub += f" · market {_v2_format_market_spread(market_spread, home)}"
        if isinstance(spread_edge, (int, float)):
            fav_sub += f" · edge {spread_edge:+.1f}"

        total_str = f"{total:.1f}"
        if isinstance(market_total, (int, float)):
            total_sub = f"Market {market_total:.1f}"
            if isinstance(total_edge, (int, float)):
                total_sub += f" · edge {total_edge:+.1f}"
        else:
            total_sub = "No market line — model-only"
    else:
        # Thin coverage: fall back to market-anchored framing.
        if market_spread is not None:
            market_fav_team = home if market_spread < 0 else away
            fav_eyebrow = "MARKET FAVORS"
            fav_headline = _safe_text(market_fav_team)
            # Render the spread from the favorite's perspective — quoting
            # "Orlando Magic +4.0" while the headline says "Detroit Pistons"
            # is technically right but reads as a contradiction.
            fav_spread_signed = market_spread if market_spread < 0 else -market_spread
            fav_sub = f"Spread {market_fav_team} {fav_spread_signed:+.1f}"
            if isinstance(market_p_home, (int, float)):
                pct = (float(market_p_home) if market_spread < 0 else 1.0 - float(market_p_home)) * 100.0
                fav_sub += f" · win prob {pct:.0f}%"
        else:
            fav_eyebrow = "MARKET LINE"
            fav_headline = "Pending"
            fav_sub = "No consensus line in cache yet."

        if isinstance(market_total, (int, float)):
            total_str = f"{market_total:.1f}"
            total_sub = "Market consensus"
        else:
            total_str = "—"
            total_sub = "No market total cached."

    # Engines tile: list rostered players in this game with their projections.
    # Falls back to the recent-form leaders only when the user has no rostered
    # players in the matchup (which won't happen on the report path now that
    # roster_only=True filters those games out, but kept for safety).
    rostered_home = entry.get("rostered_players_home") or []
    rostered_away = entry.get("rostered_players_away") or []
    rostered_set = set(rostered_home) | set(rostered_away)
    contributors = [
        *(entry.get("top_contributors_away") or []),
        *(entry.get("top_contributors_home") or []),
    ]
    if rostered_set:
        engine_rows = [c for c in contributors if c.get("player") in rostered_set]
        engines_eyebrow = "YOUR PLAYERS"
    else:
        engine_rows = contributors[:2]
        engines_eyebrow = "TOP PROJECTED"

    engines_lines: list[str] = []
    for top in engine_rows[:3]:
        player = str(top.get("player") or "").strip()
        if not player:
            continue
        pts = float(top.get("pts_proj") or 0.0)
        # Source tag clarifies whether this is a model projection or a
        # season-average backfill, since the team rollup mixes both.
        src = str(top.get("source") or "").strip()
        src_label = "model" if src == "model" else "L10 avg"
        engines_lines.append(
            f"<font size='10.5' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
            f"<b>{_safe_text(player)}</b></font>  "
            f"<font size='9' name='{_BOLD_FONT}' color='#cc5a00'>"
            f"<b>{pts:.1f} pts</b></font>  "
            f"<font size='7.5' color='#9aa3b2'>· {src_label}</font>"
        )
    engines_body = (
        "<br/>".join(engines_lines) if engines_lines
        else "<font size='9' color='#9aa3b2'>No rotation coverage</font>"
    )
    rostered_count = len(rostered_set)
    rostered_label = (
        f"{rostered_count} rostered in this game" if rostered_count
        else "No rostered players in this game"
    )
    engines_para = Paragraph(
        f"<font size='8' name='{_BOLD_FONT}' color='#cc5a00'><b>{engines_eyebrow}</b></font>"
        f"<br/><br/>{engines_body}"
        f"<br/><br/><font size='8' color='#6b7686'>{_safe_text(rostered_label)}</font>",
        ParagraphStyle("v2_match_engines", fontName=_BODY_FONT, fontSize=10,
                       leading=14, textColor=INK_BODY),
    )
    engines_tile = Table([[engines_para]], colWidths=[2.5 * inch])
    engines_tile.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
        ("LINEABOVE", (0, 0), (-1, 0), 2.0, BRAND_ORANGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    def _tile(eyebrow: str, headline: str, sub: str, accent: colors.Color) -> Table:
        body = Paragraph(
            f"<font size='8' name='{_BOLD_FONT}' color='#cc5a00'><b>{_safe_text(eyebrow)}</b></font>"
            f"<br/><br/>"
            f"<font size='15' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
            f"<b>{_safe_text(headline)}</b></font>"
            f"<br/>"
            f"<font size='8' color='#6b7686'>{_safe_text(sub)}</font>",
            ParagraphStyle("v2_match_tile", fontName=_BODY_FONT, fontSize=10,
                           leading=13.5, textColor=INK_BODY),
        )
        t = Table([[body]], colWidths=[2.32 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEABOVE", (0, 0), (-1, 0), 2.0, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    win_accent = NEG_RED if (rollup_trustworthy and upset) else (
        V2_TIER_HIGH if confidence in {"high", "medium"} else INK_MUTED
    )
    total_eyebrow = "PROJECTED TOTAL" if rollup_trustworthy else "MARKET TOTAL"
    tiles = Table(
        [[
            _tile(fav_eyebrow, fav_headline, fav_sub, win_accent),
            _tile(total_eyebrow, total_str, total_sub, V2_ACCENT_TEAL),
            engines_tile,
        ]],
        colWidths=[2.4 * inch, 2.4 * inch, 2.5 * inch],
    )
    tiles.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    # ── AI prose panel ──
    ai_block = _v2_match_lookup_ai(matchup_label, away, home, ai_matchups)
    ai_flow: list = []
    if ai_block:
        headline = str(ai_block.get("headline", "")).strip()
        narrative = str(ai_block.get("narrative", "")).strip()
        if headline or narrative:
            inner_parts: list[str] = []
            if headline:
                inner_parts.append(
                    f"<font size='14' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
                    f"<b>{_safe_text(headline)}</b></font>"
                )
            if narrative:
                inner_parts.append(
                    f"<font size='9.5' color='#2b3340'>{_safe_text(narrative)}</font>"
                )
            ai_para = Paragraph(
                "<font size='8' name='{font}' color='#cc5a00'><b>AI&nbsp;SCOUT&nbsp;·&nbsp;NARRATIVE</b></font>"
                "<br/><br/>".format(font=_BOLD_FONT)
                + "<br/><br/>".join(inner_parts),
                ParagraphStyle("v2_match_ai", fontName=_BODY_FONT, fontSize=10,
                               leading=14, textColor=INK_BODY),
            )
            ai_panel = Table([[ai_para]], colWidths=[7.3 * inch])
            ai_panel.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
                ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_ORANGE_DEEP),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            ai_flow = [Spacer(1, 8), ai_panel]

    return [header, *bar_flow, Spacer(1, 8), tiles, *ai_flow]


def _v2_format_market_spread(market_spread: float, home_team: str) -> str:
    """Render an Odds-API-style spread (signed, home-team-relative).

    ``market_spread`` is the home team's posted spread: a value of ``-3.5``
    means the home team is favored by 3.5; ``+3.5`` means the home team is
    the underdog. Returns a string like ``Detroit Pistons -7.5``.
    """
    if not isinstance(market_spread, (int, float)) or not math.isfinite(float(market_spread)):
        return ""
    sign = "-" if market_spread < 0 else "+"
    return f"{home_team} {sign}{abs(float(market_spread)):.1f}"


def _v2_match_lookup_ai(
    matchup_label: str,
    away: str,
    home: str,
    ai_matchups: dict,
) -> dict | None:
    """Tolerant lookup for the AI matchup entry — the AI might key by
    ``"<away> @ <home>"``, ``"<home> vs <away>"``, or just team names.
    """
    if not isinstance(ai_matchups, dict) or not ai_matchups:
        return None
    candidates = [
        matchup_label,
        f"{away} @ {home}",
        f"{home} vs {away}",
        f"{away} at {home}",
        home,
        away,
    ]
    norm_index = {str(k).strip().lower(): v for k, v in ai_matchups.items()}
    for cand in candidates:
        v = norm_index.get(cand.strip().lower())
        if isinstance(v, dict):
            return v
    # Last-resort partial match
    for k, v in norm_index.items():
        if isinstance(v, dict) and away.lower() in k and home.lower() in k:
            return v
    return None


# ── Page 3 — The Signal Board ─────────────────────────────────────────────
def _v2_signal_board_flowables(
    *,
    edge_df: pd.DataFrame | None,
    sigma_lookup: dict[str, float],
    styles: dict,
) -> list:
    from reportlab.lib.enums import TA_RIGHT
    flow: list = []
    flow.append(PageBreak())
    flow.append(_AnchorFlowable("v2-signal-board", "The Signal Board", level=0))
    flow.append(_para("PAGE TWO  ·  EVERY EDGE AT A GLANCE", styles["v2_section_eyebrow"]))
    flow.append(_para("The Signal Board.", styles["v2_section_title"]))
    flow.append(_para(
        "Every player × market combination, color-coded by signed edge. "
        "Loudest signals on top.",
        styles["v2_section_dek"],
    ))
    flow.append(Spacer(1, 6))

    heat = _v2_confidence_heatmap(edge_df, width=7.3 * inch, height=2.8 * inch)
    if heat is not None:
        flow.append(heat)
    else:
        flow.append(_para("No live edges available to map.", styles["muted"]))
    flow.append(Spacer(1, 14))

    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "edge" in edge_df.columns:
        df = edge_df.copy()
        df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
        df["_abs"] = df["_e"].abs()
        df = df.dropna(subset=["_e"]).sort_values("_abs", ascending=False)
        if len(df) >= 1:
            anchor_row = df.iloc[0]
            anchor_dir = "ABOVE" if float(anchor_row["_e"]) > 0 else "BELOW"
            anchor_player = str(anchor_row.get("player", ""))
            used_idx: set = {anchor_row.name}
            used_players: set = {anchor_player}

            # Differentiator: prefer opposite-direction signal from a DIFFERENT player.
            opp_pool = df[df["_e"].apply(lambda x: (x < 0) if anchor_dir == "ABOVE" else (x > 0))]
            diff_row = None
            for _, cand in opp_pool.iterrows():
                if str(cand.get("player", "")) not in used_players:
                    diff_row = cand
                    break
            if diff_row is None and len(opp_pool) > 0:
                diff_row = opp_pool.iloc[0]
            if diff_row is None and len(df) > 1:
                # No opposite signal exists — fall back to next-largest by abs edge,
                # preferring a different player.
                for _, cand in df.iloc[1:].iterrows():
                    if str(cand.get("player", "")) not in used_players:
                        diff_row = cand
                        break
                if diff_row is None:
                    diff_row = df.iloc[1]
            if diff_row is not None:
                used_idx.add(diff_row.name)
                used_players.add(str(diff_row.get("player", "")))

            # Secondary: prefer a third distinct player; else any unused row.
            sec_row = None
            for _, cand in df.iterrows():
                if cand.name in used_idx:
                    continue
                if str(cand.get("player", "")) not in used_players:
                    sec_row = cand
                    break
            if sec_row is None:
                for _, cand in df.iterrows():
                    if cand.name in used_idx:
                        continue
                    sec_row = cand
                    break

            def _stack_card(label: str, accent: colors.Color, row) -> Table:
                if row is None:
                    body = Paragraph(
                        f"<font size='8' name='{_BOLD_FONT}' color='#{accent.hexval()[2:].lower()}'>"
                        f"<b>{label}</b></font><br/><br/>"
                        f"<font size='10' color='#9aa3b2'>—</font>",
                        styles["body"],
                    )
                else:
                    e = float(row["_e"])
                    side_col = POS_GREEN if e > 0 else NEG_RED
                    line_str = _fmt(_v2_get_line(row), 1)
                    line_part = f" line {line_str}" if line_str != "\u2014" else ""
                    body = Paragraph(
                        f"<font size='8' name='{_BOLD_FONT}' color='#{accent.hexval()[2:].lower()}'>"
                        f"<b>{label}</b></font><br/><br/>"
                        f"<font size='14' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
                        f"<b>{_safe_text(row.get('player', ''))}</b></font><br/>"
                        f"<font size='9' color='#6b7686'>"
                        f"{_safe_text(row.get('model', ''))} {_side_display(row.get('call') or row.get('side'))}"
                        f"{line_part}</font><br/><br/>"
                        f"<font size='14' name='{_V2_SERIF_BOLD}' "
                        f"color='#{side_col.hexval()[2:].lower()}'>"
                        f"<b>{_fmt_signed(e, 2)}</b></font>",
                        styles["body"],
                    )
                t = Table([[body]], colWidths=[2.32 * inch])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("LINEABOVE", (0, 0), (-1, 0), 2.0, accent),
                    ("BOX", (0, 0), (-1, -1), 0.5, V2_HAIRLINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                return t

            card_row = Table(
                [[
                    _stack_card("BEST ANCHOR", BRAND_ORANGE_DEEP, anchor_row),
                    _stack_card("BEST DIFFERENTIATOR", V2_ACCENT_TEAL, diff_row),
                    _stack_card("SECONDARY ADD", NEUTRAL_BLUE, sec_row),
                ]],
                colWidths=[2.4 * inch, 2.4 * inch, 2.5 * inch],
            )
            card_row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.append(card_row)
            flow.append(Spacer(1, 12))

    top_eye = _para("TOP 4 SIGNALS", styles["v2_section_eyebrow"])
    top_section: list = [top_eye, Spacer(1, 4)]
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "edge" in edge_df.columns:
        df = edge_df.copy()
        df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
        df["_abs"] = df["_e"].abs()
        df = df.dropna(subset=["_e"]).sort_values("_abs", ascending=False).head(4)
        rows: list[list] = []
        for _, r in df.iterrows():
            e = float(r["_e"])
            side_col = POS_GREEN if e > 0 else NEG_RED
            side_disp = _side_display(r.get("call") or r.get("side"))
            sig = sigma_lookup.get(str(r.get("model", "")).lower())
            conf = _confidence_score(e, sig, r.get("books"))
            hit = _hit_probability(_v2_get_proj(r),
                                   _v2_get_line(r), r.get("call") or r.get("side"), sig)
            rank_chip = Paragraph(
                f"<font size='8' name='{_BOLD_FONT}' color='#ffffff'><b>"
                f"&nbsp;&nbsp;{len(rows)+1}&nbsp;&nbsp;</b></font>",
                ParagraphStyle("rk", fontName=_BOLD_FONT, fontSize=8,
                               textColor=WHITE, alignment=TA_CENTER,
                               backColor=V2_INK_HERO, borderPadding=4),
            )
            label = Paragraph(
                f"<font size='10.5' name='{_BOLD_FONT}' color='#0a0e14'>"
                f"<b>{_safe_text(r.get('player', ''))}</b></font>"
                f"&nbsp;&nbsp;<font size='9' color='#6b7686'>"
                f"{_safe_text(r.get('model', ''))}</font>",
                styles["body"],
            )
            line_v = _v2_get_line(r)
            proj_v = _v2_get_proj(r)
            line_bit = f"line <b>{_fmt(line_v, 1)}</b>" if line_v is not None else ""
            proj_bit = f"proj <b>{_fmt(proj_v, 2)}</b>" if proj_v is not None else ""
            hit_bit = f"hit <b>{_fmt_pct(hit)}</b>"
            conf_bit = f"conf <b>{conf if conf is not None else '—'}</b>"
            stat_bits = [b for b in (line_bit, proj_bit, hit_bit, conf_bit) if b]
            stats = Paragraph(
                f"<font size='8' color='#6b7686'>"
                + "  ·  ".join(stat_bits)
                + "</font>",
                styles["body"],
            )
            edge_para = Paragraph(
                f"<font size='12' name='{_V2_SERIF_BOLD}' "
                f"color='#{side_col.hexval()[2:].lower()}'>"
                f"<b>{side_disp} {_fmt_signed(e, 2)}</b></font>",
                ParagraphStyle("edge_para", fontName=_V2_SERIF_BOLD,
                               fontSize=12, textColor=side_col,
                               alignment=TA_RIGHT),
            )
            rows.append([rank_chip, [label, stats], edge_para])

        if rows:
            tbl = Table(rows, colWidths=[0.42 * inch, 4.7 * inch, 2.18 * inch])
            style = [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, V2_HAIRLINE),
            ]
            for ri in range(len(rows)):
                if ri % 2 == 1:
                    style.append(("BACKGROUND", (0, ri), (-1, ri), V2_BG_CREAM))
                if ri < len(rows) - 1:
                    style.append(("LINEBELOW", (0, ri), (-1, ri), 0.4, V2_HAIRLINE))
            tbl.setStyle(TableStyle(style))
            top_section.append(tbl)
    flow.append(KeepTogether(top_section))
    return flow


# ── Page 4 — Conviction Map ───────────────────────────────────────────────
def _v2_conviction_map_flowables(
    *,
    edge_df: pd.DataFrame | None,
    ai_sections: dict | None,
    sigma_lookup: dict[str, float],
    styles: dict,
) -> list:
    flow: list = []
    flow.append(PageBreak())
    flow.append(_AnchorFlowable("v2-conviction", "Conviction Map", level=0))
    # Build the section title block as flowables; we'll wrap it together with
    # the chart layout below in a single KeepTogether so the heading never
    # gets orphaned on its own page above the actual conviction quadrant.
    title_block: list = [
        _para("PAGE THREE  ·  WHERE TO PLAY, WHERE TO PASS", styles["v2_section_eyebrow"]),
        _para("Conviction Map.", styles["v2_section_title"]),
        _para(
            "Each signal plotted by its edge size (vertical) and market depth "
            "(horizontal). Top-right wins.",
            styles["v2_section_dek"],
        ),
        Spacer(1, 6),
    ]

    quad, legend = _v2_edge_confidence_quadrant(
        edge_df, sigma_lookup, width=4.6 * inch, height=4.0 * inch,
    )

    ai_cards: list = []
    if ai_sections and ai_sections.get("players") and isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
        df = edge_df.copy()
        df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
        df["_abs"] = df["_e"].abs()
        df = df.dropna(subset=["_e"]).sort_values("_abs", ascending=False)
        seen_players: set[str] = set()
        for _, r in df.iterrows():
            p = str(r.get("player", ""))
            if p in seen_players:
                continue
            seen_players.add(p)
            ai_block = (ai_sections.get("players") or {}).get(p)
            if not isinstance(ai_block, dict):
                continue
            rationale = str(ai_block.get("rationale") or ai_block.get("prediction") or "").strip()
            if not rationale:
                continue
            e = float(r["_e"])
            side_col = POS_GREEN if e > 0 else NEG_RED
            side_disp = _side_display(r.get("call") or r.get("side"))
            card_para = Paragraph(
                f"<font size='7' name='{_BOLD_FONT}' color='#cc5a00'><b>AI SCOUT PICK</b></font>"
                f"<br/>"
                f"<font size='12' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
                f"<b>{_safe_text(p)}</b></font><br/>"
                f"<font size='9' name='{_BOLD_FONT}' "
                f"color='#{side_col.hexval()[2:].lower()}'>"
                f"<b>{_safe_text(str(r.get('model', '')).upper())} "
                f"{side_disp} {_fmt(_v2_get_line(r), 1)}</b></font>"
                f"<br/>"
                f"<font size='8' color='#6b7686'>"
                f"{_safe_text(rationale)}"
                f"</font>",
                styles["body"],
            )
            inner = Table([[card_para]], colWidths=[2.55 * inch])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
                ("LINEBEFORE", (0, 0), (0, -1), 2, side_col),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            ai_cards.append(inner)
            ai_cards.append(Spacer(1, 6))
            if len(ai_cards) >= 4:
                break

    if not ai_cards and isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
        df = edge_df.copy()
        df["_e"] = pd.to_numeric(df["edge"], errors="coerce")
        df["_abs"] = df["_e"].abs()
        df = df.dropna(subset=["_e"]).sort_values("_abs", ascending=False).head(2)
        for _, r in df.iterrows():
            e = float(r["_e"])
            side_col = POS_GREEN if e > 0 else NEG_RED
            side_disp = _side_display(r.get("call") or r.get("side"))
            books_n = int(r.get("books") or 0) if pd.notna(r.get("books")) else 0
            card_para = Paragraph(
                f"<font size='7' name='{_BOLD_FONT}' color='#cc5a00'><b>TOP CALL</b></font>"
                f"<br/>"
                f"<font size='12' name='{_V2_SERIF_BOLD}' color='#0a0e14'>"
                f"<b>{_safe_text(r.get('player', ''))}</b></font><br/>"
                f"<font size='9' name='{_BOLD_FONT}' "
                f"color='#{side_col.hexval()[2:].lower()}'>"
                f"<b>{_safe_text(str(r.get('model', '')).upper())} "
                f"{side_disp} {_fmt(_v2_get_line(r), 1)}</b></font>"
                f"<br/>"
                f"<font size='8' color='#6b7686'>"
                f"Model {_fmt(_v2_get_proj(r), 2)} on "
                f"line {_fmt(_v2_get_line(r), 1)} "
                f"({_fmt_signed(e, 2)} edge across {books_n} books)."
                f"</font>",
                styles["body"],
            )
            inner = Table([[card_para]], colWidths=[2.55 * inch])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
                ("LINEBEFORE", (0, 0), (0, -1), 2, side_col),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            ai_cards.append(inner)
            ai_cards.append(Spacer(1, 6))

    right_block: list = ai_cards if ai_cards else [
        _para("AI scout context unavailable — render report with an OpenAI key for picks.",
              styles["muted"]),
    ]

    if quad is not None:
        # ── Build a beautifully formatted Signal Index ────────────────────
        # Each row: [#chip] [Surname]  [MARKET]  [+edge]
        # Two columns; thin rules between rows for an editorial table feel.

        def _chip(rank: int, side: str) -> Drawing:
            col = POS_GREEN if side == "OVER" else NEG_RED
            chip = Drawing(16, 14)
            chip.add(Circle(8, 7, 6.5, strokeColor=col, strokeWidth=0.9,
                            fillColor=V2_BG_CREAM))
            chip.add(String(8, 4.5, str(rank),
                            fontName=_BOLD_FONT, fontSize=7.0,
                            fillColor=col, textAnchor="middle"))
            return chip

        def _row_cells(item: dict) -> list:
            edge_col = "#1f7a3d" if item["side"] == "OVER" else "#b8392b"
            name_para = Paragraph(
                f"<font name='{_V2_SERIF_BOLD}' size='9' color='#0a0e14'>"
                f"{_safe_text(item['player'])}</font>"
                f" &nbsp;<font name='{_BOLD_FONT}' size='6.5' color='#6b7686'>"
                f"{_safe_text(item['market'])}</font>",
                styles["body"],
            )
            edge_para = Paragraph(
                f"<para align='right'>"
                f"<font name='{_V2_SERIF_BOLD}' size='9' color='{edge_col}'>"
                f"{_fmt_signed(item['edge'], 2)}</font></para>",
                styles["body"],
            )
            return [_chip(item["rank"], item["side"]), name_para, edge_para]

        legend_rows: list[list] = []
        if legend:
            half = (len(legend) + 1) // 2
            left_items = legend[:half]
            right_items = legend[half:]
            for idx in range(half):
                left = _row_cells(left_items[idx]) if idx < len(left_items) else ["", "", ""]
                right = _row_cells(right_items[idx]) if idx < len(right_items) else ["", "", ""]
                legend_rows.append(left + [""] + right)

        # Header row: title + inline color/axis legend.
        header = Paragraph(
            f"<font size='7' name='{_BOLD_FONT}' color='#cc5a00'>"
            f"<b>SIGNAL INDEX</b></font>"
            f" &nbsp;<font size='6.5' name='{_BOLD_FONT}' color='#a3a8b1'>"
            f"NUMBERED MARKERS"
            f"</font>",
            styles["body"],
        )
        cues = Paragraph(
            f"<para align='right'>"
            f"<font size='7' name='{_BOLD_FONT}' color='#1f7a3d'>\u25CF OVER</font>"
            f" &nbsp;"
            f"<font size='7' name='{_BOLD_FONT}' color='#b8392b'>\u25CF UNDER</font>"
            f" &nbsp;"
            f"<font size='6.5' name='{_BOLD_FONT}' color='#a3a8b1'>"
            f"y=|edge|  x=books"
            f"</font></para>",
            styles["body"],
        )

        # Column widths: chip + name + edge | gutter | chip + name + edge
        col_widths = [
            0.30 * inch, 1.30 * inch, 0.65 * inch,
            0.06 * inch,
            0.30 * inch, 1.30 * inch, 0.65 * inch,
        ]
        all_rows: list[list] = [
            [header, "", "", "", cues, "", ""],
        ] + legend_rows
        cap_table = Table(all_rows, colWidths=col_widths)

        style_cmds: list = [
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEABOVE", (0, 0), (-1, 0), 0.5, V2_HAIRLINE),
            # Header spans
            ("SPAN", (0, 0), (3, 0)),
            ("SPAN", (4, 0), (6, 0)),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (0, -1), 12),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, V2_HAIRLINE),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            # Gutter column gets extra breathing room and a faint vertical rule
            ("LINEAFTER", (3, 1), (3, -1), 0.4, V2_HAIRLINE),
        ]
        # Hairline between body rows (skip the header).
        for r_idx in range(1, len(all_rows) - 1):
            style_cmds.append(
                ("LINEBELOW", (0, r_idx), (2, r_idx), 0.25,
                 colors.HexColor("#e6e2da"))
            )
            style_cmds.append(
                ("LINEBELOW", (4, r_idx), (6, r_idx), 0.25,
                 colors.HexColor("#e6e2da"))
            )
        cap_table.setStyle(TableStyle(style_cmds))
        left_block: list = [quad, Spacer(1, 6), cap_table]
    else:
        left_block = [_para("Not enough live edges for the conviction quadrant.",
                            styles["muted"])]

    layout = Table(
        [[left_block, right_block]],
        colWidths=[4.65 * inch, 2.65 * inch],
    )
    layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    # Bundle title + layout so the section heading never orphans onto a
    # page above the chart it introduces.
    flow.append(KeepTogether(title_block + [layout]))
    flow.append(Spacer(1, 10))

    # Legend intentionally omitted — the numbered points map 1:1 to the
    # ranked rows on the Signal Board (page 2), so a separate index would
    # only duplicate that information and tends to orphan onto a new page.
    _ = legend  # kept for the function signature
    return flow


# ── Page 5 — Model Quality v2 ─────────────────────────────────────────────
def _v2_model_quality_flowables(
    *,
    metrics: pd.DataFrame | None,
    styles: dict,
) -> list:
    flow: list = []
    flow.append(PageBreak())
    flow.append(_AnchorFlowable("v2-models", "Model Quality", level=0))
    title_block: list = [
        _para("PAGE FOUR  ·  HOW MUCH TO TRUST THE BOARD", styles["v2_section_eyebrow"]),
        _para("Model Quality.", styles["v2_section_title"]),
        _para(
            "Composite trust on the left; per-target reliability on the right. "
            "Higher R² explains more game-to-game variance.",
            styles["v2_section_dek"],
        ),
        Spacer(1, 8),
    ]

    gauge = _v2_reliability_gauge(metrics, width=3.4 * inch, height=2.5 * inch)
    multiples = _v2_reliability_multiples(metrics, width=3.8 * inch, height=2.5 * inch)
    if gauge is not None or multiples is not None:
        layout = Table(
            [[gauge if gauge is not None else _para("No metrics", styles["muted"]),
              multiples if multiples is not None else _para("No multiples", styles["muted"])]],
            colWidths=[3.5 * inch, 3.8 * inch],
        )
        layout.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        # Title and chart land on the same page — never orphan the heading.
        flow.append(KeepTogether(title_block + [layout]))
        flow.append(Spacer(1, 14))
    else:
        flow.extend(title_block)

    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        name_col = next((c for c in ("model", "name") if c in metrics.columns), None)
        target_col = next((c for c in ("target", "Target") if c in metrics.columns), None)
        r2_col = next((c for c in ("R²", "r2", "R2") if c in metrics.columns), None)
        rmse_col = next((c for c in ("RMSE", "rmse") if c in metrics.columns), None)
        if name_col and r2_col and rmse_col:
            df = metrics.copy()
            df["_r2"] = pd.to_numeric(df[r2_col], errors="coerce")
            df["_rmse"] = pd.to_numeric(df[rmse_col], errors="coerce")
            df = df.sort_values("_r2", ascending=False)
            rows = [["MODEL", "TARGET", "R²", "RMSE", "TIER"]]
            tier_styles: list[tuple] = []
            for ri, (_, r) in enumerate(df.iterrows(), start=1):
                r2v = float(r["_r2"]) if pd.notna(r["_r2"]) else float("nan")
                rmse_v = float(r["_rmse"]) if pd.notna(r["_rmse"]) else float("nan")
                if not pd.isna(r2v) and r2v >= 0.55:
                    tier_label, tier_col = "Strong", V2_TIER_HIGH
                elif not pd.isna(r2v) and r2v >= 0.40:
                    tier_label, tier_col = "Solid", V2_TIER_MID
                else:
                    tier_label, tier_col = "Light", V2_TIER_LOW
                rows.append([
                    str(r[name_col]),
                    str(r[target_col]) if target_col else "—",
                    _fmt(r2v, 2),
                    _fmt(rmse_v, 2),
                    tier_label,
                ])
                tier_styles.append(("TEXTCOLOR", (4, ri), (4, ri), tier_col))
            tbl = Table(rows, colWidths=[1.6 * inch, 1.6 * inch, 0.9 * inch,
                                         0.9 * inch, 1.0 * inch])
            base_style = [
                ("FONT", (0, 0), (-1, 0), _BOLD_FONT, 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK_MUTED),
                ("BACKGROUND", (0, 0), (-1, 0), V2_BG_CREAM),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, V2_HAIRLINE),
                ("FONT", (0, 1), (-1, -1), _BODY_FONT, 9),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK_BODY),
                ("FONT", (4, 1), (4, -1), _BOLD_FONT, 9),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
            for ri in range(1, len(rows)):
                if ri % 2 == 0:
                    base_style.append(("BACKGROUND", (0, ri), (-1, ri), V2_BG_CREAM))
                base_style.append(("LINEBELOW", (0, ri), (-1, ri), 0.3, V2_HAIRLINE))
            tbl.setStyle(TableStyle(base_style + tier_styles))
            flow.append(tbl)
    return flow


# ── Pages 6+ — Player Profile ─────────────────────────────────────────────
def _v2_player_profile_block(
    *,
    player: str,
    edge_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    recent_form: dict[str, dict[str, float]] | None,
    ai_sections: dict | None,
    player_history: dict[str, pd.DataFrame] | None,
    player_games: dict[str, pd.DataFrame] | None,
    sigma_lookup: dict[str, float],
    styles: dict,
    player_history_status: dict[str, str] | None = None,
) -> list:
    from reportlab.lib.enums import TA_RIGHT
    flow: list = []
    flow.append(PageBreak())
    slug = "v2-player-" + "".join(
        ch.lower() if ch.isalnum() else "-" for ch in player
    ).strip("-")
    flow.append(_AnchorFlowable(slug, player, level=1))
    # Stamp the active player so the page chrome renders the player's name
    # in the header band on every page of this section.
    flow.append(_ActivePlayerMarker(player))

    pdf_player = None
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "player" in edge_df.columns:
        pdf_player = edge_df[edge_df["player"].astype(str) == player].copy()
        if "edge" in pdf_player.columns and not pdf_player.empty:
            pdf_player["_abs"] = pd.to_numeric(pdf_player["edge"], errors="coerce").abs()
            pdf_player = pdf_player.sort_values("_abs", ascending=False)

    top_row = pdf_player.iloc[0] if pdf_player is not None and not pdf_player.empty else None

    if top_row is not None:
        e = float(top_row.get("edge", 0))
        side_disp = _side_display(top_row.get("call") or top_row.get("side"))
        sig = sigma_lookup.get(str(top_row.get("model", "")).lower())
        conf = _confidence_score(e, sig, top_row.get("books"))
        if conf is not None and conf >= 60:
            conf_word = "high confidence"
        elif conf is not None and conf >= 35:
            conf_word = "moderate confidence"
        else:
            conf_word = "low confidence"
        line_disp = _fmt(_v2_get_line(top_row), 1)
        # Build with explicit pieces so the divider stays a real bullet
        # (``_safe_text`` rewrites · → "|" — we only sanitise the model name).
        if line_disp != "—":
            call_text_html = (
                f"{_safe_text(str(top_row.get('model', '')).upper())} "
                f"{side_disp} {line_disp} · {conf_word}"
            )
        else:
            call_text_html = (
                f"{_safe_text(str(top_row.get('model', '')).upper())} "
                f"{side_disp} · {conf_word}"
            )
        matchup = str(top_row.get("matchup", "")) or ""
    else:
        call_text_html = "No live call available"
        matchup = ""

    rf = (recent_form or {}).get(player, {}) or {}
    form_strip = "  ·  ".join(
        f"{k.upper()} {_fmt(rf.get(k), 1)}"
        for k in ("pts", "reb", "ast", "pra", "min")
        if rf.get(k) is not None
    ) or "—"

    # Adapt the display font for very long names so they always render on
    # one line. "Shai Gilgeous-Alexander" / "Victor Wembanyama" overflow the
    # 4.6 inch hero column at size 30, which produces awkward two-line
    # layouts and pushes the matchup/form rows down.
    safe_player = _safe_text(player)
    if len(player) > 22:
        name_size = 22
    elif len(player) > 18:
        name_size = 26
    else:
        name_size = 30
    hero_left = Paragraph(
        f"<font size='{name_size}' name='{_V2_SERIF_BOLD}' color='#ffffff'>"
        f"<b>{safe_player}</b></font><br/>"
        f"<font size='9' name='{_BOLD_FONT}' color='#9aa3b2'>"
        f"<b>{_safe_text(matchup) if matchup else 'TONIGHT'}</b></font><br/>"
        f"<font size='8' color='#9aa3b2'>RECENT FORM  ·  {_safe_text(form_strip)}</font>",
        styles["body"],
    )
    side_col = POS_GREEN if (top_row is not None and float(top_row.get("edge", 0)) > 0) else (
        NEG_RED if top_row is not None else BRAND_ORANGE)
    hero_right = Paragraph(
        f"<font size='8' name='{_BOLD_FONT}' color='#ff7a18'>"
        f"<b>TONIGHT'S CALL</b></font><br/>"
        f"<font size='13' name='{_V2_SERIF_BOLD}' "
        f"color='#{side_col.hexval()[2:].lower()}'>"
        f"<b>{call_text_html}</b></font>",
        ParagraphStyle("ph_call", fontName=_V2_SERIF_BOLD,
                       fontSize=13, leading=17,
                       textColor=side_col, alignment=TA_RIGHT),
    )
    hero = Table([[hero_left, hero_right]],
                 colWidths=[4.6 * inch, 2.7 * inch])
    hero.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), V2_BG_INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 22),
        ("RIGHTPADDING", (0, 0), (-1, -1), 22),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LINEBELOW", (0, 0), (-1, -1), 3, BRAND_ORANGE),
    ]))
    flow.append(hero)
    flow.append(Spacer(1, 12))

    games = (player_games or {}).get(player)
    left_blocks: list = []
    if top_row is not None:
        market = str(top_row.get("model", "")).lower()
        sig = sigma_lookup.get(market)
        proj = _v2_get_proj(top_row)
        line = _v2_get_line(top_row)
        try:
            proj_f = float(proj) if proj is not None and pd.notna(proj) else None
        except (TypeError, ValueError):
            proj_f = None
        if proj_f is not None and sig:
            fan = _v2_forecast_fan(
                projection=proj_f, sigma=sig, line=line, label=market,
                width=4.4 * inch, height=2.0 * inch,
            )
            if fan is not None:
                left_blocks.append(fan)
                left_blocks.append(Spacer(1, 6))

    if isinstance(games, pd.DataFrame) and not games.empty:
        try:
            spark = _player_form_sparklines(games, width=4.4 * inch, height=1.2 * inch)
            if spark is not None:
                left_blocks.append(spark)
                left_blocks.append(Spacer(1, 6))
        except Exception:
            pass

    # Resolved-lines history goes directly under the sparklines on the left
    # so it lives on the same page as the rest of the player block.
    history = (player_history or {}).get(player)
    if isinstance(history, pd.DataFrame) and not history.empty:
        df = history.copy()
        if "game_date" in df.columns:
            df = df.sort_values("game_date", ascending=False)
        df = df.head(4)
        rows = [["DATE", "MARKET", "LINE", "ACTUAL", "RESULT"]]
        for _, r in df.iterrows():
            rows.append([
                str(r.get("game_date", ""))[:10] if r.get("game_date") is not None else "—",
                str(r.get("metric", "—")),
                _fmt(r.get("line"), 1),
                _fmt(r.get("actual"), 1),
                str(r.get("result", "")),
            ])
        htbl = Table(rows, colWidths=[0.85 * inch, 1.05 * inch, 0.6 * inch,
                                      0.7 * inch, 0.7 * inch])
        hstyle = [
            ("FONT", (0, 0), (-1, 0), _BOLD_FONT, 7),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK_MUTED),
            ("BACKGROUND", (0, 0), (-1, 0), V2_BG_CREAM),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, V2_HAIRLINE),
            ("FONT", (0, 1), (-1, -1), _BODY_FONT, 8),
            ("ALIGN", (2, 0), (3, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for ri in range(1, len(rows)):
            res = rows[ri][4].lower()
            if "over" in res:
                hstyle.append(("TEXTCOLOR", (4, ri), (4, ri), POS_GREEN))
                hstyle.append(("FONT", (4, ri), (4, ri), _BOLD_FONT, 8))
            elif "under" in res:
                hstyle.append(("TEXTCOLOR", (4, ri), (4, ri), NEG_RED))
                hstyle.append(("FONT", (4, ri), (4, ri), _BOLD_FONT, 8))
            if ri % 2 == 0:
                hstyle.append(("BACKGROUND", (0, ri), (-1, ri), V2_BG_CREAM))
        htbl.setStyle(TableStyle(hstyle))
        left_blocks.append(Spacer(1, 4))
        left_blocks.append(_para("LAST 4 RESOLVED LINES", styles["v2_section_eyebrow"]))
        left_blocks.append(Spacer(1, 4))
        left_blocks.append(htbl)
    else:
        # No resolved-line history yet for this player. Pick a placeholder
        # message based on WHY the table is empty so the reader can tell
        # whether more data is reachable (refresh game log / run backfill)
        # or whether the bookmaker simply isn't posting props for this
        # player in the cached window (nothing actionable on our side).
        status = (player_history_status or {}).get(player, "")
        if status == _PLAYER_HISTORY_STATUS_NO_API_COVERAGE:
            placeholder_text = (
                f"No pregame prop lines have been posted for "
                f"{_safe_text(player)} on any cached date. The bookmaker "
                f"simply isn't listing this player right now — most often "
                f"because their team is off the slate the API has been "
                f"tracking (e.g. a non-playoff team during the postseason). "
                f"This panel will populate automatically once props are "
                f"posted and the game is played."
            )
        elif status == _PLAYER_HISTORY_STATUS_NO_GAME_OVERLAP:
            placeholder_text = (
                f"Pregame prop lines exist for {_safe_text(player)} but "
                f"none of those dates align with the cached game log. This "
                f"usually means the game-log cache hasn't ingested recent "
                f"games yet — the next refresh will repopulate, or you can "
                f"toggle “Backfill historical odds” on the report config "
                f"card to widen the window."
            )
        elif status == _PLAYER_HISTORY_STATUS_EMPTY_CACHE:
            placeholder_text = (
                f"No historical odds are cached locally yet. Toggle "
                f"“Backfill historical odds” on the report config card "
                f"to fetch them, or wait for daily live snapshots to "
                f"accumulate as you visit the app on game days."
            )
        else:
            placeholder_text = (
                f"No resolved lines on file for {_safe_text(player)} yet. "
                f"Lines fill in once a cached pregame prop matches a "
                f"played game in their log."
            )
        left_blocks.append(Spacer(1, 4))
        left_blocks.append(_para("LAST 4 RESOLVED LINES", styles["v2_section_eyebrow"]))
        left_blocks.append(Spacer(1, 4))
        empty_para = Paragraph(
            f"<font size='8' color='#6b7686'>{placeholder_text}</font>",
            styles["body"],
        )
        empty_panel = Table([[empty_para]], colWidths=[3.9 * inch])
        empty_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEBEFORE", (0, 0), (0, -1), 2, V2_HAIRLINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        left_blocks.append(empty_panel)

    proj_player = (projections or {}).get(player)
    if isinstance(proj_player, pd.DataFrame) and not proj_player.empty and pdf_player is not None and not pdf_player.empty:
        try:
            mini = _player_minichart(proj_player, pdf_player,
                                     width=4.4 * inch, height=1.5 * inch)
            if mini is not None:
                left_blocks.append(mini)
        except Exception:
            pass

    if not left_blocks:
        left_blocks = [_para("Forecast visuals unavailable for this player.",
                             styles["muted"])]

    right_blocks: list = []

    streak_metric = None
    if top_row is not None:
        market_to_label = {
            "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
            "threepm": "Threepm",
        }
        streak_metric = market_to_label.get(str(top_row.get("model", "")).lower())
    streak = _v2_hitrate_streak(history, metric=streak_metric,
                                width=2.55 * inch, height=0.95 * inch)
    if streak is None:
        streak = _v2_hitrate_streak(history, metric=None,
                                    width=2.55 * inch, height=0.95 * inch)
    if streak is not None:
        right_blocks.append(streak)
        right_blocks.append(Spacer(1, 6))

    ai_player = (ai_sections or {}).get("players", {}).get(player) if ai_sections else None
    if isinstance(ai_player, dict):
        news = str(ai_player.get("news") or ai_player.get("prediction") or "").strip()
        rationale = str(ai_player.get("rationale") or "").strip()
    elif isinstance(ai_player, str):
        news = ""
        rationale = ai_player.strip()
    else:
        news = rationale = ""

    if news or rationale:
        body_html = ""
        if news:
            body_html += (
                f"<font size='7' name='{_BOLD_FONT}' color='#cc5a00'>"
                f"<b>LATEST CONTEXT</b></font><br/>"
                f"<font size='8' color='#2b3340'>{_safe_text(news)}</font><br/><br/>"
            )
        if rationale:
            body_html += (
                f"<font size='7' name='{_BOLD_FONT}' color='#cc5a00'>"
                f"<b>ANALYST NOTES</b></font><br/>"
                f"<font size='8' color='#2b3340'>{_safe_text(rationale)}</font>"
            )
        ai_panel = Table(
            [[Paragraph(body_html, styles["body"])]],
            colWidths=[2.55 * inch],
        )
        ai_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), V2_BG_CREAM),
            ("LINEBEFORE", (0, 0), (0, -1), 2, BRAND_ORANGE_DEEP),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        right_blocks.append(ai_panel)
        right_blocks.append(Spacer(1, 6))

    if isinstance(proj_player, pd.DataFrame) and not proj_player.empty:
        df = proj_player.copy()
        model_col = next((c for c in ("model", "Model") if c in df.columns), None)
        pred_col = next((c for c in ("prediction", "Prediction", "projection") if c in df.columns), None)
        if model_col and pred_col:
            df = df.head(4)
            rows = [["MODEL", "PROJ", "LINE", "EDGE"]]
            for _, r in df.iterrows():
                m = str(r[model_col])
                proj = r[pred_col]
                line = "—"
                edge_v = "—"
                if pdf_player is not None and not pdf_player.empty:
                    match = pdf_player[pdf_player["model"].astype(str) == m]
                    if not match.empty:
                        line = _fmt(_v2_get_line(match.iloc[0]), 1)
                        try:
                            ev = float(match.iloc[0].get("edge"))
                            edge_v = _fmt_signed(ev, 2)
                        except (TypeError, ValueError):
                            pass
                rows.append([m, _fmt(proj, 2), line, edge_v])
            ptbl = Table(rows, colWidths=[0.95 * inch, 0.55 * inch, 0.5 * inch, 0.55 * inch])
            pstyle = [
                ("FONT", (0, 0), (-1, 0), _BOLD_FONT, 7),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK_MUTED),
                ("BACKGROUND", (0, 0), (-1, 0), V2_BG_CREAM),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, V2_HAIRLINE),
                ("FONT", (0, 1), (-1, -1), _BODY_FONT, 8),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK_BODY),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
            for ri in range(1, len(rows)):
                if ri % 2 == 0:
                    pstyle.append(("BACKGROUND", (0, ri), (-1, ri), V2_BG_CREAM))
                txt = str(rows[ri][3])
                if txt.startswith("+"):
                    pstyle.append(("TEXTCOLOR", (3, ri), (3, ri), POS_GREEN))
                    pstyle.append(("FONT", (3, ri), (3, ri), _BOLD_FONT, 8))
                elif txt.startswith("-"):
                    pstyle.append(("TEXTCOLOR", (3, ri), (3, ri), NEG_RED))
                    pstyle.append(("FONT", (3, ri), (3, ri), _BOLD_FONT, 8))
            ptbl.setStyle(TableStyle(pstyle))
            right_blocks.append(ptbl)

    if not right_blocks:
        right_blocks = [_para("History and projections unavailable.", styles["muted"])]

    body = Table([[left_blocks, right_blocks]],
                 colWidths=[4.5 * inch, 2.8 * inch])
    body.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    flow.append(body)
    flow.append(Spacer(1, 10))

    return flow


# ── Main entrypoint ─────────────────────────────────────────────────────────
def build_pdf_report(
    *,
    roster: dict[str, list[str]],
    bundle_metrics: pd.DataFrame | None = None,
    edge_df: pd.DataFrame | None = None,
    projections: dict[str, pd.DataFrame] | None = None,
    recent_form: dict[str, dict[str, float]] | None = None,
    ai_sections: dict[str, Any] | None = None,
    player_history: dict[str, pd.DataFrame] | None = None,
    player_history_status: dict[str, str] | None = None,
    player_games: dict[str, pd.DataFrame] | None = None,
    matchup_predictions: list[dict] | None = None,
) -> bytes:
    """Render the report and return the PDF as raw bytes.

    Parameters
    ----------
    roster
        ``{player: [season, ...]}`` from the active app state.
    bundle_metrics
        DataFrame of per-model metrics with columns like ``model``, ``target``,
        ``R²``, ``RMSE``.
    edge_df
        Output of the edge-board builder. Optional.
    projections
        ``{player: DataFrame}`` with one row per model and a ``prediction``
        column. Optional.
    recent_form
        ``{player: {stat_key: value}}`` snapshot for the player block header.
    ai_sections
        Dict with keys ``executive_summary``, ``slate_outlook``, ``players``.
        Pass ``None`` to render a data-only report.
    player_history
        ``{player: DataFrame}`` with columns ``game_date``, ``metric``,
        ``line``, ``actual``, ``margin``, ``result`` for the player's recent
        games where a historical line was available. Used to render a
        "Historical lines vs outcomes" panel inside each player block.
    player_history_status
        Optional ``{player: str}`` describing why each player's history is
        empty. Recognised values: ``"has_lines"`` (table populates),
        ``"no_api_coverage"`` (bookmaker hasn't posted props for them in the
        cached window), ``"no_game_overlap"`` (props exist but the game log
        doesn't cover those dates), and ``"empty_cache"`` (no historical
        odds cached at all). Drives the per-player placeholder text so the
        reader knows whether more data is reachable.
    """
    styles = _build_styles()

    # Sanitize every user-supplied string so fragile glyphs (·, ć, │, ...) can
    # never reach the PDF as tofu boxes regardless of which font loaded.
    def _clean_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return df
        out = df.copy()
        for col in out.columns:
            kind = out[col].dtype.kind
            # Cover legacy object columns AND pandas 2.x StringDtype/Unicode (kind 'O' or 'U').
            if kind in ("O", "U") or str(out[col].dtype) in ("string", "str"):
                out[col] = out[col].map(lambda v: _safe_text(v) if isinstance(v, str) else v)
        return out

    if roster:
        roster = {
            _safe_text(group): [_safe_text(p) for p in players]
            for group, players in roster.items()
        }
    edge_df = _clean_df(edge_df)
    bundle_metrics = _clean_df(bundle_metrics)
    if projections:
        projections = {_safe_text(k): _clean_df(v) for k, v in projections.items()}
    if recent_form:
        recent_form = {_safe_text(k): v for k, v in recent_form.items()}
    if ai_sections:
        ai_sections = {
            **ai_sections,
            "players": {_safe_text(k): v for k, v in (ai_sections.get("players") or {}).items()},
        }
    if player_history:
        player_history = {_safe_text(k): _clean_df(v) for k, v in player_history.items()}
    if player_history_status:
        player_history_status = {
            _safe_text(k): str(v) for k, v in player_history_status.items()
        }
    if player_games:
        player_games = {_safe_text(k): _clean_df(v) for k, v in player_games.items()}

    meta = _ReportMeta(
        generated_at=datetime.now().strftime("%b %d, %Y  |  %H:%M"),
        roster_count=len(roster or {}),
        has_ai=bool(ai_sections and (
            ai_sections.get("executive_summary")
            or ai_sections.get("players")
        )),
    )

    buf = BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Hooplytics Roster Analytics Report",
        author="Hooplytics",
    )

    cover_frame = Frame(
        0.95 * inch, 0.6 * inch,
        LETTER[0] - 1.55 * inch, LETTER[1] - 1.2 * inch,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="cover",
    )
    body_frame = Frame(
        doc.leftMargin, doc.bottomMargin + 0.15 * inch,
        doc.width, doc.height - 0.4 * inch,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="body",
    )
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=_draw_cover_chrome),
        PageTemplate(
            id="body",
            frames=[body_frame],
            # ``onPageEnd`` fires AFTER all flowables on the page have been
            # placed, so the chrome reads the latest ``_hl_active_player``
            # set by ``_ActivePlayerMarker`` flowables on this page. Drawing
            # chrome at the end is purely a sequencing change — the chrome
            # uses absolute coordinates, so it still appears at the top of
            # the page visually.
            onPageEnd=lambda c, d: _draw_page_chrome(c, d, meta),
        ),
    ])

    # V2 styles bolt the editorial type system on top of the existing palette.
    styles = _v2_styles(styles)

    sigma_lookup = _metric_sigma_lookup(bundle_metrics)

    flow: list = []
    # ── Page 1 — Editorial Cover ──
    flow.extend(_v2_cover_flowables(
        roster=roster, meta=meta,
        edge_df=edge_df, metrics=bundle_metrics,
        styles=styles,
    ))
    flow.append(NextPageTemplate("body"))

    # ── Page 2 — Tonight's Setup ──
    flow.append(PageBreak())
    flow.extend(_v2_tonight_setup_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        ai_sections=ai_sections,
        styles=styles,
    ))

    # ── Page 2.5 — Tonight's Matchups (only when we have slate predictions) ──
    if matchup_predictions:
        flow.extend(_v2_matchups_flowables(
            matchup_predictions=matchup_predictions,
            ai_sections=ai_sections,
            styles=styles,
        ))

    # ── Page 3 — The Signal Board ──
    flow.extend(_v2_signal_board_flowables(
        edge_df=edge_df,
        sigma_lookup=sigma_lookup,
        styles=styles,
    ))

    # ── Page 4 — Conviction Map ──
    flow.extend(_v2_conviction_map_flowables(
        edge_df=edge_df,
        ai_sections=ai_sections,
        sigma_lookup=sigma_lookup,
        styles=styles,
    ))

    # ── Page 5 — Model Quality ──
    flow.extend(_v2_model_quality_flowables(
        metrics=bundle_metrics,
        styles=styles,
    ))

    # ── Pages 6+ — Per-player editorials ──
    for player_name in (roster or {}).keys():
        flow.extend(_v2_player_profile_block(
            player=player_name,
            edge_df=edge_df,
            projections=projections,
            recent_form=recent_form,
            ai_sections=ai_sections,
            player_history=player_history,
            player_history_status=player_history_status,
            player_games=player_games,
            sigma_lookup=sigma_lookup,
            styles=styles,
        ))

    doc.build(flow)
    return buf.getvalue()


__all__ = ["build_pdf_report"]
