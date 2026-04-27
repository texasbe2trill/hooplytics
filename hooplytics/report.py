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
    canvas.drawString(
        0.6 * inch + 1.1 * inch,
        height - 0.42 * inch,
        "Roster Analytics Report",
    )
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
    flow: list = _section_header("Analytics visuals", "Section 04", styles, anchor="sec-analytics")

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
    if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return []

    df = history_df.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("game_date", ascending=False)
    df = df.head(max_rows)
    if df.empty:
        return []

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

    return [
        Spacer(1, 6),
        Paragraph(
            f"<font size='7.5' color='#cc5a00'><b>"
            f"{player.upper()} &nbsp;|&nbsp; HISTORICAL LINES vs OUTCOMES"
            f"</b></font>",
            styles["body"],
        ),
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
        "Signal summary", "Section 01", styles, anchor="sec-quick-calls",
    )
    flow.append(_para(
        "Skim-friendly summary \u2014 every above/below-line signal from the live "
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
    df = df.sort_values("abs_edge", ascending=False).head(12)
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
        flow.append(_para("Context narrative (AI)", styles["h3"]))
        if ai_text:
            flow.append(_para(ai_text, styles["body"]))
        if outlook:
            flow.append(_para(outlook, styles["body"]))
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
    flow: list = _section_header("Signal spotlight  |  top 3", "Section 03", styles, anchor="sec-spotlight")
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
    flow: list = _section_header("Model quality", "Section 05", styles, anchor="sec-model-quality")
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
    flow: list = _section_header("Model vs line gaps", "Section 06", styles, anchor="sec-edges")
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
        books_label = (
            _abbrev_books(r.get("book_names"), max_show=3)
            if "book_names" in r else None
        )
        if not books_label or books_label == "—":
            n = r.get("books")
            books_label = f"{int(n)} books" if pd.notna(n) else "—"
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
        col_widths=[1.35 * inch, 0.95 * inch, 0.55 * inch, 0.65 * inch,
                    0.6 * inch, 0.55 * inch, 0.5 * inch, 0.5 * inch, 1.05 * inch],
        align_right_cols=[2, 3, 4, 5, 6],
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

    if news and news.strip():
        flow.append(_para("Latest context", styles["h3"]))
        flow.append(_para(news.strip(), styles["body"]))

    if rationale and rationale.strip():
        flow.append(_para("Analyst notes", styles["h3"]))
        flow.append(_para(rationale.strip(), styles["body"]))

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
        "Signal stack", "Section 07", styles, anchor="sec-slip",
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
    flow: list = _section_header("Per-player breakdown", "Section 08", styles, anchor="sec-players")
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
    player_games: dict[str, pd.DataFrame] | None = None,
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
            onPage=lambda c, d: _draw_page_chrome(c, d, meta),
        ),
    ])

    flow: list = []
    flow.extend(_cover_flowables(
        roster=roster, meta=meta,
        edge_df=edge_df, metrics=bundle_metrics,
        styles=styles,
    ))
    flow.append(NextPageTemplate("body"))
    flow.append(PageBreak())

    # Clickable Table of Contents — links jump to bookmarked section anchors,
    # and the same anchors register as PDF outline entries for sidebar nav.
    # Each top-level row carries a short description so the TOC reads like a
    # glossy magazine front matter rather than a bare hyperlink list.
    toc_items: list[tuple] = [
        ("Signal summary", "sec-quick-calls",
         "Skim-friendly summary \u2014 every above/below-line signal ranked."),
        ("Slate brief", "sec-exec",
         "Loudest signal, slate posture, and AI-augmented context."),
        ("Signal spotlight  |  top 3", "sec-spotlight",
         "The three biggest model-vs-line gaps tonight."),
        ("Analytics visuals", "sec-analytics",
         "R\u00b2 lollipop, conviction leaderboard, and signal distribution."),
        ("Model quality", "sec-model-quality",
         "Per-model R\u00b2 and RMSE with confidence tiers."),
        ("Model vs line gaps", "sec-edges",
         "Full ranked table of every model-vs-line signal."),
        ("Signal stack", "sec-slip",
         "Anchor, differentiator, and avoid-stack guidance."),
        ("Per-player breakdown", "sec-players",
         "Recent form, predictions, news, and analyst notes."),
    ]
    for player_name in (roster or {}).keys():
        slug = "player-" + "".join(
            ch.lower() if ch.isalnum() else "-" for ch in player_name
        ).strip("-")
        toc_items.append((player_name, slug, "sub"))
    flow.append(_AnchorFlowable("toc", "Contents", level=0))
    flow.extend(_toc_flowables(toc_items, styles))
    flow.append(PageBreak())

    flow.extend(_kpi_strip_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        projections=projections,
        styles=styles,
    ))
    flow.append(Spacer(1, 6))
    flow.append(_how_to_read_strip(styles))
    flow.append(Spacer(1, 14))
    flow.extend(_bottom_line_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        styles=styles,
    ))
    sigma_lookup = _metric_sigma_lookup(bundle_metrics)
    flow.extend(_quick_calls_flowables(
        edge_df=edge_df,
        recent_form=recent_form,
        sigma_lookup=sigma_lookup,
        styles=styles,
    ))
    flow.extend(_executive_summary_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        ai_sections=ai_sections,
        styles=styles,
    ))
    flow.extend(_spotlight_flowables(edge_df, styles, sigma_lookup=sigma_lookup))
    flow.extend(_analytics_visuals_flowables(
        metrics=bundle_metrics,
        edge_df=edge_df,
        styles=styles,
    ))
    flow.extend(_model_quality_flowables(bundle_metrics, styles))
    flow.extend(_edge_board_flowables(edge_df, styles, sigma_lookup=sigma_lookup))
    flow.extend(_slip_builder_flowables(
        edge_df=edge_df,
        sigma_lookup=sigma_lookup,
        player_games=player_games,
        styles=styles,
    ))
    flow.extend(_per_player_flowables(
        roster,
        edge_df=edge_df,
        projections=projections,
        recent_form=recent_form,
        ai_sections=ai_sections,
        player_history=player_history,
        player_games=player_games,
        sigma_lookup=sigma_lookup,
        styles=styles,
    ))

    doc.build(flow)
    return buf.getvalue()


__all__ = ["build_pdf_report"]
