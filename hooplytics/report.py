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

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
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


def _short(text: Any, limit: int = 18) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── Styles ──────────────────────────────────────────────────────────────────
def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"

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
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(INK_DARK)
    canvas.drawString(0.6 * inch, height - 0.42 * inch, "HOOPLYTICS")
    canvas.setFont("Helvetica", 8)
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
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawCentredString(
        width / 2,
        0.42 * inch,
        "Hooplytics is for statistical analysis and entertainment. "
        "Not financial or betting advice.",
    )
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
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawCentredString(
        width / 2, 0.45 * inch,
        "Hooplytics · Roster Analytics Report",
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
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(safe, style)


def _section_header(title: str, eyebrow: str, styles: dict) -> list:
    return [
        _para(eyebrow.upper(), styles["eyebrow"]),
        _para(title, styles["h1"]),
        Spacer(1, 4),
    ]


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
        ("Live edges", str(edge_count), GOLD),
        ("Strong edges", str(strong_count), POS_GREEN if strong_count else INK_FAINT),
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
        return "MORE-leaning", pos, neg
    if neg > pos * 1.4:
        return "LESS-leaning", pos, neg
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

    top_line = "No live edge rows are available."
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
            f" Slate posture: {lean} ({pos} MORE / {neg} LESS across {edge_n} mapped lines)."
        )

    return (
        f"Tracking {players_n} player{'s' if players_n != 1 else ''} with "
        f"{edge_n} live edge row{'s' if edge_n != 1 else ''} and a median "
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
        0, height - 14, "Model R² · accuracy lollipop",
        fontName="Helvetica-Bold", fontSize=10.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 28, "Higher = the model explains more game-to-game variance.",
        fontName="Helvetica", fontSize=8, fillColor=INK_MUTED,
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
            fontName="Helvetica", fontSize=6.5, fillColor=INK_FAINT,
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
            fontName="Helvetica-Bold", fontSize=7.5, fillColor=INK_DARK,
            textAnchor="middle",
        ))
        drawing.add(String(
            cx, plot_y - 11, _short(row[name_col], 11),
            fontName="Helvetica", fontSize=7, fillColor=INK_BODY,
            textAnchor="middle",
        ))
        if rmse_col is not None and pd.notna(row.get(rmse_col)):
            drawing.add(String(
                cx, plot_y - 20, f"RMSE {float(row[rmse_col]):.2f}",
                fontName="Helvetica", fontSize=6, fillColor=INK_FAINT,
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
        f"{_short(r.get('player', ''), 14)} · {_short(r.get('model', ''), 9)}"
        for _, r in df.iterrows()
    ]
    values = [float(v) for v in df["edge_num"]]
    span = max(abs(min(values)), abs(max(values)), 0.5) * 1.15

    width = 7.0 * inch
    height = 2.85 * inch
    drawing = Drawing(width, height)
    drawing.add(String(
        0, height - 14, "Edge magnitudes · model vs market",
        fontName="Helvetica-Bold", fontSize=10.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 28, "Bars to the right = MORE lean · bars to the left = LESS lean.",
        fontName="Helvetica", fontSize=8, fillColor=INK_MUTED,
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
            fontName="Helvetica", fontSize=6.5, fillColor=INK_FAINT,
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
                fontName="Helvetica-Bold", fontSize=7, fillColor=color,
            ))
        else:
            drawing.add(String(
                x0 - 4, cy - 3, _fmt_signed(v, 2),
                fontName="Helvetica-Bold", fontSize=7, fillColor=color,
                textAnchor="end",
            ))
        drawing.add(String(
            plot_x - 6, cy - 3, lbl,
            fontName="Helvetica", fontSize=7.5, fillColor=INK_BODY,
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
        0, height - 14, "Edge distribution",
        fontName="Helvetica-Bold", fontSize=10, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 26, "How concentrated are the model-vs-line gaps?",
        fontName="Helvetica", fontSize=7.5, fillColor=INK_MUTED,
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
                fontName="Helvetica-Bold", fontSize=7, fillColor=INK_DARK,
                textAnchor="middle",
            ))

    drawing.add(String(
        plot_x, plot_y - 10, f"{-span:+.1f}",
        fontName="Helvetica", fontSize=6.5, fillColor=INK_FAINT,
    ))
    drawing.add(String(
        plot_x + plot_w / 2, plot_y - 10, "0",
        fontName="Helvetica", fontSize=6.5, fillColor=INK_FAINT,
        textAnchor="middle",
    ))
    drawing.add(String(
        plot_x + plot_w, plot_y - 10, f"{span:+.1f}",
        fontName="Helvetica", fontSize=6.5, fillColor=INK_FAINT,
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
        ("MORE / LESS leans", f"{pos} / {neg}"),
        ("Mapped edge rows", str(edge_n)),
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
        fontName="Helvetica-Bold", fontSize=9.5, fillColor=INK_DARK,
    ))
    drawing.add(String(
        0, height - 24, "Orange = model · Grey = market line.",
        fontName="Helvetica", fontSize=7.5, fillColor=INK_MUTED,
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
            fontName="Helvetica-Bold", fontSize=7, fillColor=edge_color,
            textAnchor="middle",
        ))
        drawing.add(String(
            cx, plot_y - 10, _short(name, 10),
            fontName="Helvetica", fontSize=6.8, fillColor=INK_BODY,
            textAnchor="middle",
        ))

    return drawing


# ── Section assemblers ──────────────────────────────────────────────────────
def _analytics_visuals_flowables(
    *,
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Analytics visuals", "Section 02A", styles)

    rendered = False
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        chart = _r2_lollipop_chart(metrics)
        if chart is not None:
            flow.append(chart)
            flow.append(Spacer(1, 10))
            rendered = True

    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
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
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
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
            _cover_meta_tile("Live edges", str(edge_n), styles),
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
        _para("HOOPLYTICS · ROSTER ANALYTICS", styles["cover_eyebrow"]),
        _para("Tonight's Scouting Report.", styles["cover_title"]),
        _para(
            "Decoded edges, model leans, and the loudest signals from your "
            "tracked roster — straight off the Hooplytics engine.",
            styles["cover_sub"],
        ),
        Spacer(1, 0.5 * inch),
        tiles,
        Spacer(1, 0.35 * inch),
        _para(f"Generated · {meta.generated_at}", styles["cover_tag"]),
        _para(f"Seasons · {seasons_label}", styles["cover_tag"]),
        _para(
            "Prose · " + ("AI-augmented (OpenAI)" if meta.has_ai else "data-only"),
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


def _executive_summary_flowables(
    *,
    roster: dict[str, list[str]],
    metrics: pd.DataFrame | None,
    edge_df: pd.DataFrame | None,
    ai_sections: dict[str, Any] | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Executive summary", "Section 01", styles)
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
) -> Table:
    edge_val = float(pd.to_numeric(row.get("edge"), errors="coerce"))
    side = str(row.get("call") or row.get("side") or "").upper() or (
        "MORE" if edge_val > 0 else "LESS"
    )
    abs_edge = abs(edge_val)
    grade, grade_color = _signal_grade(abs_edge)
    side_color = POS_GREEN if side in ("MORE", "OVER") else NEG_RED
    accent_hex = "#1f9d6c" if side in ("MORE", "OVER") else "#d24545"

    badge_w = 0.35 * inch
    badge = Drawing(badge_w, 0.35 * inch)
    badge.add(Circle(badge_w / 2, badge_w / 2, badge_w / 2,
                     fillColor=grade_color, strokeColor=WHITE, strokeWidth=1))
    badge.add(String(
        badge_w / 2, badge_w / 2 - 4, f"#{rank}",
        fontName="Helvetica-Bold", fontSize=10, fillColor=WHITE,
        textAnchor="middle",
    ))

    text = Paragraph(
        (
            f"<font size='7.5' color='#cc5a00'><b>{grade.upper()} SIGNAL</b></font><br/>"
            f"<font size='11' color='#11151c'><b>{_short(row.get('player', 'Unknown'), 22)}</b></font><br/>"
            f"<font size='8.5' color='#6b7686'>{_short(row.get('model', 'metric'), 14)}</font><br/>"
            f"<br/>"
            f"<font size='8.5' color='#6b7686'>LEAN</font> "
            f"<font size='10' color='{accent_hex}'><b>{side}</b></font>"
            f"  ·  <font size='8.5' color='#6b7686'>EDGE</font> "
            f"<font size='10' color='{accent_hex}'><b>{_fmt_signed(edge_val, 2)}</b></font><br/>"
            f"<font size='8' color='#6b7686'>"
            f"Line {_fmt(row.get('posted line', row.get('line')), 1)}  ·  "
            f"Model {_fmt(row.get('model prediction', row.get('projection')), 2)}"
            f"</font>"
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


def _spotlight_flowables(edge_df: pd.DataFrame | None, styles: dict) -> list:
    flow: list = _section_header("Signal spotlight · top 3", "Section 01A", styles)
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
        _spotlight_card_flowable(i + 1, row, styles)
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
    flow: list = _section_header("Model quality", "Section 02", styles)
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
        extra.append(("FONTNAME", (4, idx), (4, idx), "Helvetica-Bold"))
    table.setStyle(TableStyle(extra))
    flow.append(table)
    return flow


# ── Section: edge board ─────────────────────────────────────────────────────
def _edge_board_flowables(
    edge_df: pd.DataFrame | None,
    styles: dict,
    *,
    top_n: int = 14,
) -> list:
    flow: list = _section_header("Top edges — model vs market", "Section 03", styles)
    if edge_df is None or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        flow.append(_para(
            "No live edges available. Add an Odds API key and fetch lines "
            "to populate this section.",
            styles["muted"],
        ))
        return flow

    df = edge_df.copy()
    if "abs_edge" not in df.columns and "edge" in df.columns:
        df["abs_edge"] = df["edge"].abs()
    df = df.sort_values("abs_edge", ascending=False).head(top_n)

    rows: list[list[Any]] = [
        ["Player", "Market", "Line", "Projection", "Edge", "Side", "Books"]
    ]
    for _, r in df.iterrows():
        edge_val = r.get("edge")
        side = str(r.get("call") or r.get("side") or "").upper()
        rows.append([
            str(r.get("player", "—")),
            str(r.get("model", "—")),
            _fmt(r.get("posted line", r.get("line")), 1),
            _fmt(r.get("model prediction", r.get("projection")), 2),
            _fmt_signed(edge_val, 2),
            side,
            str(int(r["books"])) if "books" in r and pd.notna(r["books"]) else "—",
        ])

    table = _styled_table(
        rows,
        col_widths=[1.7 * inch, 1.3 * inch, 0.7 * inch, 0.95 * inch,
                    0.75 * inch, 0.65 * inch, 0.6 * inch],
        align_right_cols=[2, 3, 4, 6],
    )
    extra: list = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        try:
            ev = float(r.get("edge"))
        except (TypeError, ValueError):
            ev = 0.0
        color = POS_GREEN if ev > 0 else (NEG_RED if ev < 0 else INK_BODY)
        extra.append(("TEXTCOLOR", (4, i), (4, i), color))
        extra.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
        side = str(r.get("call") or r.get("side") or "").upper()
        if side in ("MORE", "OVER"):
            extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
            extra.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
        elif side in ("LESS", "UNDER"):
            extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
            extra.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
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
    form_html = "  ·  ".join(bits) if bits else (
        "<font size='9' color='#fff1e6'>No recent-form snapshot.</font>"
    )

    name_para = Paragraph(
        f"<font size='15' color='#ffffff'><b>{_short(player, 32)}</b></font><br/>"
        f"<font size='7.5' color='#fff1e6'><b>RECENT FORM (LAST GAMES)</b></font>",
        styles["body"],
    )
    form_para = Paragraph(form_html, styles["body"])

    band = Table(
        [[name_para, form_para]],
        colWidths=[2.6 * inch, 4.4 * inch],
    )
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_ORANGE),
        ("LINEBELOW", (0, 0), (-1, -1), 1.5, BRAND_ORANGE_DEEP),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
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
    styles: dict,
) -> list:
    flow: list = [
        _player_hero_band(player, recent_form, styles),
        Spacer(1, 6),
    ]

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
            ["Model", "Target", "Projection", "Line", "Edge", "Side"]
        ]
        for _, r in proj.iterrows():
            name = str(r["model"])
            ed = edge_lookup.get(name, {})
            side = str(ed.get("side", "")).upper().split()[0] if ed.get("side") else "—"
            rows.append([
                name,
                str(r.get("target", "—")),
                _fmt(r["prediction"], 2),
                _fmt(ed.get("line"), 1) if ed else "—",
                _fmt_signed(ed.get("edge"), 2) if ed else "—",
                side,
            ])

        table = _styled_table(
            rows,
            col_widths=[1.8 * inch, 1.2 * inch, 0.95 * inch, 0.7 * inch,
                        0.85 * inch, 0.65 * inch],
            align_right_cols=[2, 3, 4],
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
                extra.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
            side = str(ed.get("side", "")).upper().split()[0] if ed.get("side") else ""
            if side in ("MORE", "OVER"):
                extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
                extra.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
            elif side in ("LESS", "UNDER"):
                extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
                extra.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
        table.setStyle(TableStyle(extra))
        flow.append(table)
    else:
        flow.append(_para("No model projections available.", styles["muted"]))

    flow.append(Spacer(1, 4))
    flow.append(_para("Data rationale", styles["h3"]))
    if (
        projection is not None
        and isinstance(projection, pd.DataFrame)
        and not projection.empty
    ):
        top_text = "No market-mapped model edge was available for this player."
        if edge_df is not None and isinstance(edge_df, pd.DataFrame) and not edge_df.empty:
            sub = edge_df[edge_df["player"] == player].copy()
            if not sub.empty and "edge" in sub.columns:
                sub["abs_edge"] = pd.to_numeric(sub["edge"], errors="coerce").abs()
                sub = sub.sort_values("abs_edge", ascending=False)
                r = sub.iloc[0]
                top_text = (
                    f"Largest gap is {r.get('model', 'metric')} at edge "
                    f"{_fmt_signed(r.get('edge'), 2)} with posted line "
                    f"{_fmt(r.get('posted line', r.get('line')), 1)} and model "
                    f"projection {_fmt(r.get('model prediction', r.get('projection')), 2)}."
                )
        flow.append(_para(top_text, styles["body"]))
    else:
        flow.append(_para(
            "Model projections were unavailable for this player in this run.",
            styles["muted"],
        ))

    if rationale and rationale.strip():
        flow.append(Spacer(1, 4))
        flow.append(_para("Context notes (AI)", styles["h3"]))
        flow.append(_para(rationale.strip(), styles["body"]))
    flow.append(Spacer(1, 12))
    return flow


def _per_player_flowables(
    roster: dict[str, list[str]],
    *,
    edge_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    recent_form: dict[str, dict[str, float]] | None,
    ai_sections: dict[str, Any] | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Per-player breakdown", "Section 04", styles)
    if not roster:
        flow.append(_para("Roster is empty.", styles["muted"]))
        return flow

    ai_players = (ai_sections or {}).get("players") or {}
    projections = projections or {}
    recent_form = recent_form or {}

    for player in roster.keys():
        block = _player_block(
            player,
            edge_df=edge_df,
            projection=projections.get(player),
            recent_form=recent_form.get(player),
            rationale=str(ai_players.get(player, "")),
            styles=styles,
        )
        # Keep player hero band + first chart together so we don't orphan a name.
        head = block[:2] if len(block) >= 2 else block
        flow.append(KeepTogether(head))
        flow.extend(block[len(head):])

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
    """
    styles = _build_styles()
    meta = _ReportMeta(
        generated_at=datetime.now().strftime("%b %d, %Y · %H:%M"),
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

    flow.extend(_kpi_strip_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        projections=projections,
        styles=styles,
    ))
    flow.extend(_executive_summary_flowables(
        roster=roster,
        metrics=bundle_metrics,
        edge_df=edge_df,
        ai_sections=ai_sections,
        styles=styles,
    ))
    flow.extend(_spotlight_flowables(edge_df, styles))
    flow.extend(_analytics_visuals_flowables(
        metrics=bundle_metrics,
        edge_df=edge_df,
        styles=styles,
    ))
    flow.extend(_model_quality_flowables(bundle_metrics, styles))
    flow.extend(_edge_board_flowables(edge_df, styles))
    flow.extend(_per_player_flowables(
        roster,
        edge_df=edge_df,
        projections=projections,
        recent_form=recent_form,
        ai_sections=ai_sections,
        styles=styles,
    ))

    doc.build(flow)
    return buf.getvalue()


__all__ = ["build_pdf_report"]
