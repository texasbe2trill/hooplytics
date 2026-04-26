"""PDF report builder for Hooplytics.

Generates a printable, brand-styled analytics report covering the active
roster: model quality, edge board, per-player projections, and (optionally)
AI-written prose rationale supplied via :mod:`hooplytics.openai_agent`.

The builder is pure: it accepts plain dataframes and dicts and returns the
PDF as ``bytes``, so it has no Streamlit dependencies and is easy to test or
invoke from a CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
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
INK_DARK = colors.HexColor("#11151c")
INK_BODY = colors.HexColor("#2b3340")
INK_MUTED = colors.HexColor("#6b7686")
PANEL_BG = colors.HexColor("#f5f6f8")
PANEL_BORDER = colors.HexColor("#e1e4ea")
POS_GREEN = colors.HexColor("#1f9d6c")
NEG_RED = colors.HexColor("#d24545")
WHITE = colors.white


# ── Styles ──────────────────────────────────────────────────────────────────
def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"

    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base, fontName=bold_font, fontSize=44,
            leading=50, textColor=INK_DARK, alignment=TA_LEFT,
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
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base, fontName=body_font, fontSize=10,
            leading=14, textColor=INK_MUTED, alignment=TA_LEFT, spaceBefore=4,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base, fontName=bold_font, fontSize=20, leading=24,
            textColor=INK_DARK, spaceAfter=10, spaceBefore=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base, fontName=bold_font, fontSize=14, leading=18,
            textColor=INK_DARK, spaceAfter=6, spaceBefore=14,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base, fontName=bold_font, fontSize=11, leading=15,
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

    # Top band
    canvas.setFillColor(BRAND_ORANGE)
    canvas.rect(0, height - 0.18 * inch, width, 0.18 * inch, stroke=0, fill=1)

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
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawCentredString(
        width / 2,
        0.45 * inch,
        "Hooplytics is for statistical analysis and entertainment. "
        "Not financial or betting advice.",
    )
    canvas.drawRightString(
        width - 0.6 * inch,
        0.32 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _draw_cover_chrome(canvas, doc) -> None:
    canvas.saveState()
    width, height = LETTER
    # Big orange band on the left edge
    canvas.setFillColor(BRAND_ORANGE)
    canvas.rect(0, 0, 0.42 * inch, height, stroke=0, fill=1)
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


# ── Section builders ────────────────────────────────────────────────────────
def _cover_flowables(
    *,
    roster: dict[str, list[str]],
    meta: _ReportMeta,
    styles: dict,
) -> list:
    seasons_set = sorted({s for ss in roster.values() for s in ss})
    seasons_label = ", ".join(seasons_set) if seasons_set else "—"
    flow: list = [
        Spacer(1, 1.6 * inch),
        _para("HOOPLYTICS", styles["cover_eyebrow"]),
        _para("Roster Analytics Report", styles["cover_title"]),
        _para(
            "Model projections, market edges, and contextual rationale for "
            "your tracked roster.",
            styles["cover_sub"],
        ),
        Spacer(1, 0.6 * inch),
        _para(f"Generated · {meta.generated_at}", styles["cover_meta"]),
        _para(f"Players · {meta.roster_count}", styles["cover_meta"]),
        _para(f"Seasons · {seasons_label}", styles["cover_meta"]),
        _para(
            "Prose · " + ("AI-augmented (OpenAI)" if meta.has_ai else "data-only"),
            styles["cover_meta"],
        ),
    ]
    return flow


def _executive_summary_flowables(
    ai_sections: dict[str, Any] | None,
    styles: dict,
) -> list:
    flow: list = _section_header("Executive summary", "Section 01", styles)
    text = ""
    if ai_sections:
        text = str(ai_sections.get("executive_summary", "")).strip()
    if not text:
        text = (
            "AI-generated prose was not requested or unavailable. The data "
            "tables in this report represent the active model output and "
            "current market consensus."
        )
    flow.append(_para(text, styles["body"]))

    outlook = ""
    if ai_sections:
        outlook = str(ai_sections.get("slate_outlook", "")).strip()
    if outlook:
        flow.append(Spacer(1, 4))
        flow.append(_para("Slate outlook", styles["h3"]))
        flow.append(_para(outlook, styles["body"]))
    return flow


def _model_quality_flowables(metrics: pd.DataFrame | None, styles: dict) -> list:
    flow: list = _section_header("Model quality", "Section 02", styles)
    if metrics is None or not isinstance(metrics, pd.DataFrame) or metrics.empty:
        flow.append(_para("Model metrics unavailable.", styles["muted"]))
        return flow

    rows: list[list[Any]] = [["Model", "Target", "R²", "RMSE"]]
    df = metrics.copy()
    name_col = next((c for c in ("model", "name") if c in df.columns), None)
    target_col = next((c for c in ("target", "stat") if c in df.columns), None)
    r2_col = next((c for c in ("R²", "r2", "R2") if c in df.columns), None)
    rmse_col = next((c for c in ("RMSE", "rmse") if c in df.columns), None)

    for _, r in df.iterrows():
        rows.append([
            str(r[name_col]) if name_col else "—",
            str(r[target_col]) if target_col else "—",
            _fmt(r[r2_col], 2) if r2_col else "—",
            _fmt(r[rmse_col], 2) if rmse_col else "—",
        ])

    flow.append(_styled_table(
        rows,
        col_widths=[2.4 * inch, 1.6 * inch, 0.9 * inch, 0.9 * inch],
        align_right_cols=[2, 3],
    ))
    return flow


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
    # Color edge cells & side cells for readability.
    style_extra: list = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        edge_val = r.get("edge")
        try:
            ev = float(edge_val)
        except (TypeError, ValueError):
            ev = 0.0
        color = POS_GREEN if ev > 0 else (NEG_RED if ev < 0 else INK_BODY)
        style_extra.append(("TEXTCOLOR", (4, i), (4, i), color))
        style_extra.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
        side = str(r.get("call") or r.get("side") or "").upper()
        if side in ("MORE", "OVER"):
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
        elif side in ("LESS", "UNDER"):
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
    table.setStyle(TableStyle(style_extra))
    flow.append(table)
    return flow


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
        _para("PLAYER", styles["eyebrow"]),
        _para(player, styles["h2"]),
    ]

    # Recent form snapshot row
    if recent_form:
        bits = []
        for label, key in (
            ("PTS", "pts"), ("REB", "reb"), ("AST", "ast"),
            ("PRA", "pra"), ("FAN", "fantasy_score"), ("MIN", "min"),
        ):
            val = recent_form.get(key)
            if val is None:
                continue
            bits.append(f"<b>{label}</b> {_fmt(val, 1)}")
        if bits:
            flow.append(Paragraph(
                "  ·  ".join(bits),
                ParagraphStyle(
                    "rf", parent=styles["muted"], textColor=INK_MUTED,
                    fontSize=9, leading=13,
                ),
            ))
            flow.append(Spacer(1, 2))

    # Projections vs market table
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
        style_extra: list = []
        for i, (_, r) in enumerate(proj.iterrows(), start=1):
            name = str(r["model"])
            ed = edge_lookup.get(name, {})
            try:
                ev = float(ed.get("edge")) if ed.get("edge") is not None else None
            except (TypeError, ValueError):
                ev = None
            if ev is not None:
                color = POS_GREEN if ev > 0 else (NEG_RED if ev < 0 else INK_BODY)
                style_extra.append(("TEXTCOLOR", (4, i), (4, i), color))
                style_extra.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
            side = str(ed.get("side", "")).upper().split()[0] if ed.get("side") else ""
            if side in ("MORE", "OVER"):
                style_extra.append(("TEXTCOLOR", (5, i), (5, i), POS_GREEN))
            elif side in ("LESS", "UNDER"):
                style_extra.append(("TEXTCOLOR", (5, i), (5, i), NEG_RED))
        table.setStyle(TableStyle(style_extra))
        flow.append(table)
    else:
        flow.append(_para("No model projections available.", styles["muted"]))

    # AI rationale
    flow.append(Spacer(1, 4))
    if rationale and rationale.strip():
        flow.append(_para("Rationale", styles["h3"]))
        flow.append(_para(rationale.strip(), styles["body"]))
    flow.append(Spacer(1, 10))
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
        # Keep player header + first table together so we don't orphan a name.
        flow.append(KeepTogether(block[:3]))
        flow.extend(block[3:])

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
        bottomMargin=0.65 * inch,
        title="Hooplytics Roster Analytics Report",
        author="Hooplytics",
    )

    cover_frame = Frame(
        0.85 * inch, 0.6 * inch,
        LETTER[0] - 1.4 * inch, LETTER[1] - 1.2 * inch,
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
    flow.extend(_cover_flowables(roster=roster, meta=meta, styles=styles))
    from reportlab.platypus import NextPageTemplate
    flow.append(NextPageTemplate("body"))
    flow.append(PageBreak())

    flow.extend(_executive_summary_flowables(ai_sections, styles))
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
