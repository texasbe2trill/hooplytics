"""Player Performance Analytics PDF report.

A second printable report focused on player-development analytics for
coaching staffs. Strictly performance-oriented: no betting edges, no
projection-vs-line content. Each player gets a profile page with KPI
scorecards, trend sparklines, a shooting & efficiency profile, a
strengths/weaknesses radar, consistency analysis, role/usage trends,
hot/cold streak detection, and an optional AI-written coaching note.

The builder shares brand chrome (fonts, colors, header/footer, callouts)
with :mod:`hooplytics.report` so both reports look like part of the same
publication. It is purposely independent of :func:`build_pdf_report` so the
existing projections report cannot regress.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Polygon,
    PolyLine,
    Rect,
    Wedge,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
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

# ── Shared brand primitives reused from the projections report ────────────
from hooplytics.report import (
    BRAND_ORANGE,
    BRAND_ORANGE_DEEP,
    BRAND_ORANGE_SOFT,
    GOLD,
    INK_BODY,
    INK_DARK,
    INK_FAINT,
    INK_MUTED,
    NEG_RED,
    NEUTRAL_BLUE,
    PANEL_BG,
    PANEL_BORDER,
    POS_GREEN,
    WHITE,
    String,
    _AnchorFlowable,
    _BODY_FONT,
    _BOLD_FONT,
    _ReportMeta,
    _build_styles,
    _callout_box,
    _draw_cover_chrome,
    _draw_page_chrome,
    _fmt,
    _fmt_signed,
    _para,
    _safe_text,
    _section_header,
    _styled_table,
    _v2_styles,
)


# ── Performance metric definitions ───────────────────────────────────────
# Shown on KPI scorecards and sparklines. Order is preserved.
_KPI_STATS: tuple[tuple[str, str], ...] = (
    ("pts", "PTS"),
    ("reb", "REB"),
    ("ast", "AST"),
    ("pra", "PRA"),
    ("stl", "STL"),
    ("blk", "BLK"),
    ("min", "MIN"),
    ("fantasy_score", "FAN"),
)

# Stats whose "higher is better" direction is intuitive (used for arrows).
_HIGHER_IS_BETTER: set[str] = {
    "pts", "reb", "ast", "pra", "stl", "blk", "min",
    "fantasy_score", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
}

# Radar axes — composite skill dimensions derived from the box score.
_RADAR_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Scoring",      ("pts",)),
    ("Playmaking",   ("ast",)),
    ("Rebounding",   ("reb",)),
    ("Defense",      ("stl", "blk")),
    ("Efficiency",   ("ts_pct",)),
    ("Volume",       ("min",)),
)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation helpers — pure functions over a player's game log frame
# ═══════════════════════════════════════════════════════════════════════════


def _series(games: pd.DataFrame, col: str) -> pd.Series:
    if col not in games.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(games[col], errors="coerce").dropna()


def _safe_mean(s: pd.Series) -> float:
    if s is None or s.empty:
        return float("nan")
    return float(s.mean())


def _safe_std(s: pd.Series) -> float:
    if s is None or len(s) < 2:
        return float("nan")
    return float(s.std(ddof=0))


def _shooting_profile(games: pd.DataFrame) -> dict[str, float]:
    """Compute season FG/3P/FT/TS percentages from raw makes/attempts."""
    out: dict[str, float] = {}
    fgm = _series(games, "fgm").sum()
    fga = _series(games, "fga").sum()
    fg3m = _series(games, "fg3m").sum()
    fg3a = _series(games, "fg3a").sum()
    ftm = _series(games, "ftm").sum()
    fta = _series(games, "fta").sum()
    pts = _series(games, "pts").sum()

    out["fg_pct"] = float(fgm / fga) if fga > 0 else float("nan")
    out["fg3_pct"] = float(fg3m / fg3a) if fg3a > 0 else float("nan")
    out["ft_pct"] = float(ftm / fta) if fta > 0 else float("nan")
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    denom = 2.0 * (fga + 0.44 * fta)
    out["ts_pct"] = float(pts / denom) if denom > 0 else float("nan")
    return out


def _usage_proxy_series(games: pd.DataFrame) -> pd.Series:
    """Per-game usage proxy: (FGA + 0.44*FTA + TOV) / MIN."""
    needed = {"fga", "fta", "tov", "min"}
    if not needed.issubset(games.columns):
        return pd.Series(dtype=float)
    minutes = pd.to_numeric(games["min"], errors="coerce").replace(0, np.nan)
    usg = (
        pd.to_numeric(games["fga"], errors="coerce")
        + 0.44 * pd.to_numeric(games["fta"], errors="coerce")
        + pd.to_numeric(games["tov"], errors="coerce")
    ) / minutes
    return usg.dropna()


def _percentile(s: pd.Series, q: float) -> float:
    if s is None or s.empty:
        return float("nan")
    return float(np.nanpercentile(s.to_numpy(), q))


def _hot_cold_zscore(s: pd.Series, recent_n: int = 5) -> tuple[float, float, float]:
    """Return (recent_avg, season_avg, z_score) for the last ``recent_n`` games.

    z compares the recent window's mean against the season mean using the
    season's stdev as the noise floor. Magnitude > 1 indicates a meaningful
    streak; sign indicates direction.
    """
    if s is None or len(s) < recent_n + 2:
        return (float("nan"), float("nan"), float("nan"))
    recent = float(s.tail(recent_n).mean())
    season = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd <= 0 or math.isnan(sd):
        return (recent, season, 0.0)
    z = (recent - season) / (sd / math.sqrt(recent_n))
    return (recent, season, z)


def player_performance_summary(games: pd.DataFrame, recent_n: int = 10) -> dict[str, Any]:
    """Pre-compute every analytical anchor a single profile page needs.

    Pure function — safe to call from the Streamlit layer to feed both the
    deterministic charts and the AI grounding payload.
    """
    if games is None or games.empty:
        return {"games_played": 0}

    summary: dict[str, Any] = {"games_played": int(len(games))}

    # KPI averages: full season + last-N window.
    kpis: dict[str, dict[str, float]] = {}
    for col, _label in _KPI_STATS:
        s = _series(games, col)
        if s.empty:
            continue
        season = _safe_mean(s)
        recent = _safe_mean(s.tail(recent_n))
        kpis[col] = {
            "season_avg": season,
            "recent_avg": recent,
            "delta": recent - season,
            "std": _safe_std(s),
            "p10": _percentile(s, 10),
            "p50": _percentile(s, 50),
            "p90": _percentile(s, 90),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    summary["kpis"] = kpis

    # Shooting / efficiency profile.
    summary["shooting"] = _shooting_profile(games)

    # Usage proxy (per-game) season + recent.
    usg = _usage_proxy_series(games)
    if not usg.empty:
        summary["usage"] = {
            "season_avg": _safe_mean(usg),
            "recent_avg": _safe_mean(usg.tail(recent_n)),
            "std": _safe_std(usg),
        }

    # Hot/cold streak signal across primary scoring stats.
    streaks: dict[str, dict[str, float]] = {}
    for col, _label in _KPI_STATS:
        s = _series(games, col)
        if s.empty:
            continue
        recent, season, z = _hot_cold_zscore(s, recent_n=5)
        streaks[col] = {"recent_avg": recent, "season_avg": season, "z": z}
    summary["streaks"] = streaks

    return summary


def _radar_components(
    summary: dict[str, Any],
    roster_baseline: dict[str, dict[str, float]] | None,
) -> list[tuple[str, float]]:
    """Z-score each radar axis vs the roster baseline; clip to +/-2.5."""
    out: list[tuple[str, float]] = []
    kpis = summary.get("kpis", {}) if summary else {}
    shoot = summary.get("shooting", {}) if summary else {}

    def _val_for(col: str) -> float:
        if col == "ts_pct":
            return float(shoot.get("ts_pct", float("nan")))
        return float(kpis.get(col, {}).get("season_avg", float("nan")))

    for label, cols in _RADAR_AXES:
        # Player axis value = mean of constituent columns.
        vals = [v for v in (_val_for(c) for c in cols) if not math.isnan(v)]
        if not vals:
            out.append((label, 0.0))
            continue
        player_v = sum(vals) / len(vals)

        if not roster_baseline:
            # No baseline → degenerate to a flat 0 ring.
            out.append((label, 0.0))
            continue

        # Baseline: roster-wide mean and stdev for these columns.
        baseline_vals = []
        for col in cols:
            for _player, ssum in roster_baseline.items():
                if col == "ts_pct":
                    bv = ssum.get("shooting", {}).get("ts_pct")
                else:
                    bv = ssum.get("kpis", {}).get(col, {}).get("season_avg")
                if bv is not None and not (isinstance(bv, float) and math.isnan(bv)):
                    baseline_vals.append(float(bv))
        if len(baseline_vals) < 2:
            out.append((label, 0.0))
            continue
        mu = float(np.mean(baseline_vals))
        sd = float(np.std(baseline_vals, ddof=0)) or 1.0
        z = (player_v - mu) / sd
        out.append((label, max(-2.5, min(2.5, z))))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Visual primitives — chart drawings tailored for the performance report
# ═══════════════════════════════════════════════════════════════════════════


def _trend_sparkline(
    s: pd.Series,
    *,
    width: float,
    height: float,
    color: colors.Color = BRAND_ORANGE,
) -> Drawing:
    """Filled mini-line chart of the last N games with rolling-5 overlay."""
    d = Drawing(width, height)
    if s is None or s.empty:
        d.add(_RLString_safe(width / 2, height / 2, "no data", INK_FAINT))
        return d

    pad_l, pad_r, pad_t, pad_b = 4.0, 4.0, 6.0, 6.0
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    vals = s.to_numpy(dtype=float)
    n = len(vals)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0

    def _x(i: int) -> float:
        if n <= 1:
            return pad_l + inner_w / 2
        return pad_l + inner_w * i / (n - 1)

    def _y(v: float) -> float:
        return pad_b + inner_h * (v - vmin) / (vmax - vmin)

    # Baseline (mean) line.
    mean_v = float(np.nanmean(vals))
    by = _y(mean_v)
    d.add(Line(pad_l, by, pad_l + inner_w, by,
               strokeColor=PANEL_BORDER, strokeWidth=0.5, strokeDashArray=[2, 2]))

    # Filled area under the raw series.
    pts = [(_x(i), _y(vals[i])) for i in range(n)]
    poly = Polygon(
        [pad_l, pad_b] + [c for p in pts for c in p] + [pad_l + inner_w, pad_b],
        fillColor=colors.Color(color.red, color.green, color.blue, 0.18),
        strokeColor=None,
    )
    d.add(poly)

    # Raw line.
    for i in range(n - 1):
        d.add(Line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                   strokeColor=color, strokeWidth=1.1))

    # Rolling 5 overlay.
    if n >= 5:
        roll = pd.Series(vals).rolling(5, min_periods=1).mean().to_numpy()
        rpts = [(_x(i), _y(roll[i])) for i in range(n)]
        for i in range(n - 1):
            d.add(Line(rpts[i][0], rpts[i][1], rpts[i + 1][0], rpts[i + 1][1],
                       strokeColor=INK_DARK, strokeWidth=0.7))

    # Last-point marker.
    d.add(Circle(pts[-1][0], pts[-1][1], 1.8, fillColor=color, strokeColor=WHITE, strokeWidth=0.8))
    return d


def _RLString_safe(x: float, y: float, text: str, fill: colors.Color, font_size: int = 7) -> Any:
    s = String(x, y, text, fontName=_BODY_FONT, fontSize=font_size, fillColor=fill)
    s.textAnchor = "middle"
    return s


def _shooting_bars(
    shoot: dict[str, float],
    roster_shoot_baseline: dict[str, float] | None,
    *,
    width: float,
    height: float,
) -> Drawing:
    """Horizontal bars for FG%, 3P%, FT%, TS% with roster-median tick markers."""
    d = Drawing(width, height)
    rows = [
        ("FG%",  shoot.get("fg_pct"),  (roster_shoot_baseline or {}).get("fg_pct")),
        ("3P%",  shoot.get("fg3_pct"), (roster_shoot_baseline or {}).get("fg3_pct")),
        ("FT%",  shoot.get("ft_pct"),  (roster_shoot_baseline or {}).get("ft_pct")),
        ("TS%",  shoot.get("ts_pct"),  (roster_shoot_baseline or {}).get("ts_pct")),
    ]
    pad_l, pad_r, pad_t, pad_b = 38.0, 36.0, 4.0, 4.0
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    row_h = inner_h / max(len(rows), 1)
    bar_h = row_h * 0.55
    # x-axis: shooting % range capped at 0..0.7 for visual contrast.
    vmax = 0.7

    for i, (label, val, baseline) in enumerate(rows):
        cy = pad_b + (len(rows) - 1 - i) * row_h + row_h / 2
        # Track.
        d.add(Rect(pad_l, cy - bar_h / 2, inner_w, bar_h,
                   fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.4))
        # Bar.
        v = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else 0.0
        bar_w = inner_w * max(0.0, min(1.0, v / vmax))
        # Color: green if above baseline, red if below, orange if no baseline.
        if baseline is not None and not (isinstance(baseline, float) and math.isnan(baseline)):
            bar_color = POS_GREEN if v >= float(baseline) else NEG_RED
        else:
            bar_color = BRAND_ORANGE
        d.add(Rect(pad_l, cy - bar_h / 2, bar_w, bar_h,
                   fillColor=bar_color, strokeColor=None))
        # Baseline tick.
        if baseline is not None and not (isinstance(baseline, float) and math.isnan(baseline)):
            bx = pad_l + inner_w * max(0.0, min(1.0, float(baseline) / vmax))
            d.add(Line(bx, cy - bar_h / 2 - 1, bx, cy + bar_h / 2 + 1,
                       strokeColor=INK_DARK, strokeWidth=0.9))
        # Label (left).
        lbl = String(4, cy - 3, label, fontName=_BOLD_FONT, fontSize=8, fillColor=INK_DARK)
        d.add(lbl)
        # Value (right).
        val_text = f"{v * 100:.1f}%" if v > 0 else "—"
        vt = String(width - 4, cy - 3, val_text,
                    fontName=_BOLD_FONT, fontSize=8, fillColor=INK_BODY)
        vt.textAnchor = "end"
        d.add(vt)
    return d


def _radar_chart(
    components: list[tuple[str, float]],
    *,
    width: float,
    height: float,
) -> Drawing:
    """Radar/spider chart of skill z-scores. Range -2.5..+2.5, ring at 0."""
    d = Drawing(width, height)
    cx, cy = width / 2, height / 2 - 4
    radius = min(width, height) * 0.36
    n = len(components)
    if n < 3:
        d.add(_RLString_safe(cx, cy, "no data", INK_FAINT))
        return d

    # Concentric rings: -2, -1, 0, +1, +2 (5 rings).
    for r_step in (-2.0, -1.0, 0.0, 1.0, 2.0):
        rr = radius * (r_step + 2.5) / 5.0
        d.add(Circle(cx, cy, rr, fillColor=None,
                     strokeColor=PANEL_BORDER if r_step != 0 else INK_FAINT,
                     strokeWidth=0.6 if r_step != 0 else 0.9))

    # Label position slightly outside the outer ring with horizontal alignment
    # driven by the axis angle so labels can’t clip the drawing edges.
    angles = [(-math.pi / 2 + 2 * math.pi * i / n) for i in range(n)]
    for i, (label, _z) in enumerate(components):
        a = angles[i]
        x2 = cx + radius * math.cos(a)
        y2 = cy + radius * math.sin(a)
        d.add(Line(cx, cy, x2, y2, strokeColor=PANEL_BORDER, strokeWidth=0.4))
        cos_a, sin_a = math.cos(a), math.sin(a)
        lx = cx + (radius + 10) * cos_a
        ly = cy + (radius + 10) * sin_a - 3
        ls = String(lx, ly, label, fontName=_BOLD_FONT, fontSize=7.5, fillColor=INK_DARK)
        if cos_a > 0.35:
            ls.textAnchor = "start"
        elif cos_a < -0.35:
            ls.textAnchor = "end"
        else:
            ls.textAnchor = "middle"
        d.add(ls)

    # Player polygon.
    pts = []
    for i, (_label, z) in enumerate(components):
        a = angles[i]
        rr = radius * (z + 2.5) / 5.0
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    flat = [c for p in pts for c in p]
    d.add(Polygon(
        flat,
        fillColor=colors.Color(BRAND_ORANGE.red, BRAND_ORANGE.green, BRAND_ORANGE.blue, 0.30),
        strokeColor=BRAND_ORANGE_DEEP,
        strokeWidth=1.2,
    ))
    for x, y in pts:
        d.add(Circle(x, y, 2.0, fillColor=BRAND_ORANGE_DEEP, strokeColor=WHITE, strokeWidth=0.6))
    return d


def _comparison_radar(
    players: list[tuple[str, list[tuple[str, float]], colors.Color]],
    *,
    width: float,
    height: float,
) -> Drawing:
    """Overlay several players’ radar polygons on a single chart.

    ``players`` → ``[(label, [(axis_label, z), ...], color), ...]``.
    """
    d = Drawing(width, height)
    if not players:
        d.add(_RLString_safe(width / 2, height / 2, "no data", INK_FAINT))
        return d
    n_axes = len(players[0][1])
    if n_axes < 3:
        d.add(_RLString_safe(width / 2, height / 2, "no data", INK_FAINT))
        return d
    cx, cy = width / 2, height / 2 - 4
    radius = min(width, height) * 0.36
    for r_step in (-2.0, -1.0, 0.0, 1.0, 2.0):
        rr = radius * (r_step + 2.5) / 5.0
        d.add(Circle(cx, cy, rr, fillColor=None,
                     strokeColor=PANEL_BORDER if r_step != 0 else INK_FAINT,
                     strokeWidth=0.5 if r_step != 0 else 0.9))
    angles = [(-math.pi / 2 + 2 * math.pi * i / n_axes) for i in range(n_axes)]
    for i, (label, _z) in enumerate(players[0][1]):
        a = angles[i]
        x2 = cx + radius * math.cos(a)
        y2 = cy + radius * math.sin(a)
        d.add(Line(cx, cy, x2, y2, strokeColor=PANEL_BORDER, strokeWidth=0.4))
        cos_a = math.cos(a)
        lx = cx + (radius + 10) * cos_a
        ly = cy + (radius + 10) * math.sin(a) - 3
        ls = String(lx, ly, label, fontName=_BOLD_FONT, fontSize=7.5, fillColor=INK_DARK)
        if cos_a > 0.35:
            ls.textAnchor = "start"
        elif cos_a < -0.35:
            ls.textAnchor = "end"
        else:
            ls.textAnchor = "middle"
        d.add(ls)
    for _label, components, color in players:
        pts = []
        for i, (_axis, z) in enumerate(components):
            a = angles[i]
            rr = radius * (z + 2.5) / 5.0
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        flat = [c for p in pts for c in p]
        fill = colors.Color(color.red, color.green, color.blue, 0.20)
        d.add(Polygon(flat, fillColor=fill, strokeColor=color, strokeWidth=1.4))
        for x, y in pts:
            d.add(Circle(x, y, 1.8, fillColor=color, strokeColor=WHITE, strokeWidth=0.5))
    return d


def _distribution_strip(
    summary: dict[str, Any],
    *,
    width: float,
    height: float,
) -> Drawing:
    """P10 / P50 / P90 floor-ceiling strip across primary stats."""
    d = Drawing(width, height)
    kpis = summary.get("kpis", {}) if summary else {}
    rows: list[tuple[str, float, float, float]] = []
    for col, label in _KPI_STATS:
        if col not in kpis:
            continue
        info = kpis[col]
        p10, p50, p90 = info.get("p10"), info.get("p50"), info.get("p90")
        if p10 is None or p90 is None or math.isnan(p10) or math.isnan(p90):
            continue
        rows.append((label, float(p10), float(p50), float(p90)))
        if len(rows) >= 6:
            break
    if not rows:
        d.add(_RLString_safe(width / 2, height / 2, "no data", INK_FAINT))
        return d

    pad_l, pad_r, pad_t, pad_b = 46.0, 46.0, 4.0, 4.0
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    row_h = inner_h / len(rows)
    bar_h = row_h * 0.42

    # Per-row scaling so each strip uses its own range.
    for i, (label, p10, p50, p90) in enumerate(rows):
        cy = pad_b + (len(rows) - 1 - i) * row_h + row_h / 2
        rng = max(p90 - p10, 1e-6)
        # Track.
        d.add(Rect(pad_l, cy - bar_h / 2, inner_w, bar_h,
                   fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.3))
        # Floor → ceiling band.
        d.add(Rect(pad_l, cy - bar_h / 2, inner_w, bar_h,
                   fillColor=colors.Color(BRAND_ORANGE.red, BRAND_ORANGE.green, BRAND_ORANGE.blue, 0.12),
                   strokeColor=None))
        # Median tick.
        med_x = pad_l + inner_w * (p50 - p10) / rng
        d.add(Line(med_x, cy - bar_h / 2 - 1, med_x, cy + bar_h / 2 + 1,
                   strokeColor=BRAND_ORANGE_DEEP, strokeWidth=1.4))
        # End markers.
        d.add(Line(pad_l, cy - bar_h / 2 - 1, pad_l, cy + bar_h / 2 + 1,
                   strokeColor=INK_BODY, strokeWidth=0.7))
        d.add(Line(pad_l + inner_w, cy - bar_h / 2 - 1, pad_l + inner_w, cy + bar_h / 2 + 1,
                   strokeColor=INK_BODY, strokeWidth=0.7))
        # Label + endpoint values. Endpoint labels sit BELOW the bar so they
        # never overlap the median tick or the bar fill.
        d.add(String(4, cy - 3, label, fontName=_BOLD_FONT, fontSize=8, fillColor=INK_DARK))
        below_y = cy - bar_h / 2 - 7
        d.add(String(pad_l, below_y, f"{p10:.1f}",
                     fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_MUTED,
                     textAnchor="start"))
        d.add(String(med_x, below_y, f"{p50:.1f}",
                     fontName=_BOLD_FONT, fontSize=6.5, fillColor=BRAND_ORANGE_DEEP,
                     textAnchor="middle"))
        d.add(String(pad_l + inner_w, below_y, f"{p90:.1f}",
                     fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_MUTED,
                     textAnchor="end"))
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Garmin-style + ML primitives
# ═══════════════════════════════════════════════════════════════════════════


def _activity_rings(
    rings: list[tuple[str, float, str, colors.Color]],
    *,
    width: float,
    height: float,
) -> Drawing:
    """Garmin-style concentric activity rings.

    Backgrounds are stroked circles (true continuous rings, no seams). Active
    arcs are stroked polylines with rounded caps so each ring renders as a
    clean "track" with a coloured progress segment.
    """
    d = Drawing(width, height)
    if not rings:
        d.add(_RLString_safe(width / 2, height / 2, "no data", INK_FAINT))
        return d

    n = len(rings)
    # Rings occupy the left ~45% of the drawing; legend on the right.
    ring_box = min(width * 0.45, height) - 6
    cx = ring_box / 2 + 4
    cy = height / 2
    outer_r = ring_box / 2
    thickness = max(6.0, outer_r / (n * 2.1))
    gap = 4.5

    for i, (_label, frac, _value, color) in enumerate(rings):
        # Centerline radius for this ring (stroke is centered on this circle).
        r = outer_r - thickness / 2 - i * (thickness + gap)
        if r <= thickness / 2 + 1:
            break
        bg_color = colors.Color(color.red, color.green, color.blue, 0.16)
        # Background loop — stroked circle = perfect continuous ring.
        d.add(Circle(
            cx, cy, r,
            fillColor=None, strokeColor=bg_color, strokeWidth=thickness,
        ))
        # Active arc — polyline approximation, rounded caps.
        f = max(0.0, min(1.0, float(frac)))
        if f > 0.005:
            sweep = 360.0 * f
            n_seg = max(24, int(sweep * 0.6))
            pts: list[float] = []
            for j in range(n_seg + 1):
                t = j / n_seg
                ang = math.radians(90.0 - sweep * t)
                pts.append(cx + r * math.cos(ang))
                pts.append(cy + r * math.sin(ang))
            arc = PolyLine(
                pts,
                strokeColor=color,
                strokeWidth=thickness,
                strokeLineCap=1,
                strokeLineJoin=1,
            )
            d.add(arc)

    # Legend — label, value, percentage.
    legend_x = cx + outer_r + 22
    line_h = max(13.0, (height - 8) / max(n, 1))
    legend_top = cy + (n - 1) * line_h / 2 + line_h / 2 - 4
    for i, (label, frac, value, color) in enumerate(rings):
        ly = legend_top - i * line_h
        d.add(Rect(legend_x, ly - 1, 9, 9, fillColor=color, strokeColor=color))
        d.add(String(legend_x + 14, ly, f"{label}",
                     fontName=_BOLD_FONT, fontSize=8, fillColor=INK_DARK))
        d.add(String(legend_x + 14, ly - 10, f"{value}  ·  {int(round(frac * 100))}%",
                     fontName=_BODY_FONT, fontSize=7.5, fillColor=INK_MUTED))
    return d


def _form_arc(score: float, *, width: float, height: float, label: str = "FORM INDEX") -> Drawing:
    """Semicircle gauge (Garmin power-zone style) for a -2..+2 z-score input.

    A negative score sweeps the left half of the arc (cool blue), positive
    fills the right (brand orange). Big numeric centerpiece + label below.
    """
    d = Drawing(width, height)
    cx = width / 2
    cy = height * 0.42
    radius = min(width * 0.45, height * 0.85)
    thickness = max(6.0, radius * 0.18)

    # Background half-loop.
    bg = Wedge(cx, cy, radius, 180, 360, fillColor=PANEL_BORDER, strokeColor=None)
    bg.annular = True
    bg.radius1 = radius - thickness
    d.add(bg)

    # Fill: map score in [-2, +2] to angle in [180, 360].
    s = max(-2.0, min(2.0, float(score)))
    sweep_frac = (s + 2.0) / 4.0
    fill_end_angle = 180 + 180 * sweep_frac
    if abs(s) > 0.05:
        if s >= 0:
            color = BRAND_ORANGE
        else:
            color = NEUTRAL_BLUE
        arc = Wedge(cx, cy, radius, 180, fill_end_angle,
                    fillColor=color, strokeColor=None)
        arc.annular = True
        arc.radius1 = radius - thickness
        d.add(arc)

    # Centerline tick at neutral (270°, i.e. top of semicircle).
    tx = cx + 0
    d.add(Line(tx, cy + radius - thickness - 1, tx, cy + radius + 1,
               strokeColor=INK_DARK, strokeWidth=1.0))

    # Centerpiece value.
    sign = "+" if s >= 0 else "−"
    big = String(cx, cy + 2, f"{sign}{abs(s):.2f}",
                 fontName=_BOLD_FONT, fontSize=18, fillColor=INK_DARK)
    big.textAnchor = "middle"
    d.add(big)
    sub = String(cx, cy - 11, label,
                 fontName=_BOLD_FONT, fontSize=7.5, fillColor=BRAND_ORANGE_DEEP)
    sub.textAnchor = "middle"
    d.add(sub)
    return d


def _ml_next_game_forecast(
    games: pd.DataFrame,
    summary: dict[str, Any],
    *,
    cols: tuple[str, ...] = ("pts", "reb", "ast", "fantasy_score"),
    window: int = 20,
) -> dict[str, dict[str, float]]:
    """Linear-regression next-game projection per stat.

    Fits an OLS line to the last ``window`` games, predicts the next observation,
    and returns the value plus an 80% prediction interval derived from the
    residual standard error. Slope is also returned so callers can highlight
    statistically meaningful trends.
    """
    out: dict[str, dict[str, float]] = {}
    if not isinstance(games, pd.DataFrame) or games.empty:
        return out
    z80 = 1.2816
    for col in cols:
        s = _series(games, col)
        if s is None or s.empty:
            continue
        s = s.tail(window)
        n = int(len(s))
        if n < 5:
            continue
        x = np.arange(n, dtype=float)
        y = s.to_numpy(dtype=float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            continue
        resid = y - (slope * x + intercept)
        ddof = 2 if n > 2 else 0
        sigma = float(np.std(resid, ddof=ddof)) if n > 2 else float(np.std(resid))
        sigma = max(sigma, 1e-6)
        x_next = float(n)
        pred = float(slope * x_next + intercept)
        x_mean = float(x.mean())
        sxx = float(((x - x_mean) ** 2).sum()) or 1.0
        se_pred = sigma * math.sqrt(1.0 + 1.0 / n + (x_next - x_mean) ** 2 / sxx)
        pi = z80 * se_pred
        season = (summary.get("kpis", {}) or {}).get(col, {}).get("season_avg")
        recent = (summary.get("kpis", {}) or {}).get(col, {}).get("recent_avg")
        slope_se = sigma / math.sqrt(sxx) if sxx > 0 else float("inf")
        t_stat = float(slope / slope_se) if slope_se > 0 else 0.0

        def _f(v):
            if v is None:
                return float("nan")
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        out[col] = {
            "pred": max(0.0, pred),
            "lo": max(0.0, pred - pi),
            "hi": max(0.0, pred + pi),
            "slope": float(slope),
            "t_stat": t_stat,
            "sigma": sigma,
            "season": _f(season),
            "recent": _f(recent),
        }
    return out


def _ml_forecast_panel(
    forecast: dict[str, dict[str, float]],
    *,
    width: float,
    height: float,
) -> Drawing:
    """Bullet-style ML forecast chart for next-game predictions.

    Each row plots a faint range track, an 80% prediction interval band, a tick
    for the season average, and a bold dot for the predicted value. Labels show
    the prediction + interval and a trend arrow keyed to slope significance.
    """
    d = Drawing(width, height)
    label_map = {
        "pts": "PTS", "reb": "REB", "ast": "AST", "pra": "PRA",
        "stl": "STL", "blk": "BLK", "min": "MIN", "fantasy_score": "FAN",
    }
    if not forecast:
        d.add(_RLString_safe(width / 2, height / 2, "need 5+ games", INK_FAINT))
        return d
    rows = list(forecast.items())
    pad_l, pad_r, pad_t, pad_b = 50.0, 70.0, 6.0, 6.0
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    row_h = inner_h / len(rows)
    bar_h = row_h * 0.42
    for i, (col, info) in enumerate(rows):
        cy = pad_b + (len(rows) - 1 - i) * row_h + row_h / 2
        season = info["season"]
        pred = info["pred"]
        lo = info["lo"]
        hi = info["hi"]
        candidates = [v for v in (season, hi, info["recent"]) if not math.isnan(v)]
        ceiling = max(candidates + [pred]) * 1.05 if candidates else max(pred, 1.0) * 1.05
        ceiling = max(ceiling, 1e-6)

        def _x(v, _pl=pad_l, _iw=inner_w, _c=ceiling):
            return _pl + _iw * max(0.0, min(1.0, v / _c))

        d.add(Rect(pad_l, cy - bar_h / 2, inner_w, bar_h,
                   fillColor=PANEL_BG, strokeColor=PANEL_BORDER, strokeWidth=0.3))
        x_lo, x_hi = _x(lo), _x(hi)
        d.add(Rect(x_lo, cy - bar_h / 2, max(1.0, x_hi - x_lo), bar_h,
                   fillColor=colors.Color(BRAND_ORANGE.red, BRAND_ORANGE.green, BRAND_ORANGE.blue, 0.28),
                   strokeColor=None))
        if not math.isnan(season):
            sx = _x(season)
            d.add(Line(sx, cy - bar_h / 2 - 2, sx, cy + bar_h / 2 + 2,
                       strokeColor=INK_DARK, strokeWidth=1.0))
        px = _x(pred)
        d.add(Circle(px, cy, 3.5, fillColor=BRAND_ORANGE_DEEP, strokeColor=WHITE, strokeWidth=0.8))
        d.add(String(4, cy - 3, label_map.get(col, col.upper()),
                     fontName=_BOLD_FONT, fontSize=8.5, fillColor=INK_DARK))
        t = info.get("t_stat", 0.0)
        if t > 1.5:
            arrow, arrow_hex = "▲", "#1f9d6c"
        elif t < -1.5:
            arrow, arrow_hex = "▼", "#d24545"
        else:
            arrow, arrow_hex = "·", "#6b7686"
        d.add(String(pad_l + inner_w + 6, cy - 1, f"{pred:.1f}",
                     fontName=_BOLD_FONT, fontSize=10, fillColor=BRAND_ORANGE_DEEP))
        d.add(String(pad_l + inner_w + 36, cy - 1, arrow,
                     fontName=_BOLD_FONT, fontSize=8, fillColor=colors.HexColor(arrow_hex)))
        d.add(String(pad_l + inner_w + 6, cy - 11, f"{lo:.1f}–{hi:.1f}",
                     fontName=_BODY_FONT, fontSize=6.5, fillColor=INK_MUTED))
    return d


def _player_activity_rings(
    summary: dict[str, Any],
    roster_baseline: dict[str, dict[str, Any]],
    *,
    width: float,
    height: float,
) -> Drawing:
    """Three Garmin-style activity rings for SCORING / PLAYMAKING / EFFICIENCY.

    Each ring shows the player's value as a fraction of the roster leader in
    that metric, so coaches can see at a glance who anchors which area.
    """
    kpis = summary.get("kpis", {}) if summary else {}
    shoot = summary.get("shooting", {}) if summary else {}

    def _leader(col, kind="kpi"):
        best = 0.0
        for s in roster_baseline.values():
            if kind == "kpi":
                v = (s.get("kpis", {}) or {}).get(col, {}).get("season_avg")
            else:
                v = (s.get("shooting", {}) or {}).get(col)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                best = max(best, float(v))
        return best or 1.0

    pts_v = float((kpis.get("pts", {}) or {}).get("season_avg") or 0.0)
    reb_v = float((kpis.get("reb", {}) or {}).get("season_avg") or 0.0)
    ast_v = float((kpis.get("ast", {}) or {}).get("season_avg") or 0.0)
    ts_v = float(shoot.get("ts_pct") or 0.0)

    pts_lead = _leader("pts")
    reb_lead = _leader("reb")
    ast_lead = _leader("ast")
    ts_lead = max(_leader("ts_pct", "shoot"), 0.65)

    play_v = reb_v + ast_v
    play_lead = reb_lead + ast_lead

    rings = [
        ("SCORING", (pts_v / pts_lead) if pts_lead else 0.0, f"{pts_v:.1f} PTS", BRAND_ORANGE),
        ("PLAYMAKING", (play_v / play_lead) if play_lead else 0.0,
         f"{reb_v:.1f}R · {ast_v:.1f}A", NEUTRAL_BLUE),
        ("EFFICIENCY", (ts_v / ts_lead) if ts_lead else 0.0,
         f"{ts_v * 100:.1f}% TS", POS_GREEN),
    ]
    return _activity_rings(rings, width=width, height=height)


# ═══════════════════════════════════════════════════════════════════════════
# Flowable composers
# ═══════════════════════════════════════════════════════════════════════════

def _kpi_score_card(label: str, value: str, delta: str, accent: colors.Color, styles: dict) -> Table:
    """Compact KPI scorecard with season avg + recent delta."""
    body = Paragraph(
        f"<font size='18' color='#0a0e14' name='{_BOLD_FONT}'><b>{value}</b></font><br/>"
        f"<font size='7' color='#6b7686'>{label.upper()}</font><br/>"
        f"<font size='6.8' color='#2b3340'>L10 &nbsp;{delta}</font>",
        ParagraphStyle(
            "kpi_perf", parent=styles["body"], alignment=TA_CENTER,
            leading=13, textColor=INK_DARK,
        ),
    )
    bar = Paragraph(" ", ParagraphStyle("kpi_bar", parent=styles["body"], leading=4, fontSize=2))
    inner = Table(
        [[bar], [body]],
        colWidths=[0.94 * inch],
        rowHeights=[0.07 * inch, None],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), accent),
        ("BACKGROUND", (0, 1), (0, 1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.5, PANEL_BORDER),
        ("TOPPADDING", (0, 1), (0, 1), 10),
        ("BOTTOMPADDING", (0, 1), (0, 1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
    ]))
    return inner


def _kpi_strip_for_player(summary: dict[str, Any], styles: dict) -> Table | None:
    kpis = summary.get("kpis", {}) if summary else {}
    if not kpis:
        return None
    accent_cycle = [BRAND_ORANGE, NEUTRAL_BLUE, GOLD, POS_GREEN,
                    BRAND_ORANGE_DEEP, NEUTRAL_BLUE, INK_FAINT, GOLD]
    cards: list[Any] = []
    for i, (col, label) in enumerate(_KPI_STATS):
        if col not in kpis:
            continue
        info = kpis[col]
        season = info.get("season_avg", float("nan"))
        delta = info.get("delta", float("nan"))
        delta_sign = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "·")
        delta_color = "#1f9d6c" if delta > 0.05 else ("#d24545" if delta < -0.05 else "#6b7686")
        delta_text = (
            f'<font color="{delta_color}">{delta_sign} {abs(delta):.1f}</font>'
            if not (isinstance(delta, float) and math.isnan(delta))
            else "—"
        )
        value_text = "—" if math.isnan(season) else f"{season:.1f}"
        cards.append(_kpi_score_card(label, value_text, delta_text,
                                     accent_cycle[i % len(accent_cycle)], styles))
        if len(cards) >= 8:
            break
    if not cards:
        return None
    strip = Table([cards], colWidths=[0.94 * inch] * len(cards))
    strip.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return strip


def _trend_cell(games: pd.DataFrame, col: str, label: str, summary: dict[str, Any], styles: dict) -> Table | None:
    s = _series(games, col)
    if s.empty:
        return None
    info = (summary.get("kpis") or {}).get(col, {})
    season = info.get("season_avg")
    recent = info.get("recent_avg")
    delta = info.get("delta")
    season_t = "—" if season is None or math.isnan(season) else f"{season:.1f}"
    recent_t = "—" if recent is None or math.isnan(recent) else f"{recent:.1f}"
    if delta is None or math.isnan(delta):
        delta_html = ""
    else:
        sign = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "·")
        color = "#1f9d6c" if delta > 0.05 else ("#d24545" if delta < -0.05 else "#9aa3b2")
        delta_html = f" <font color='{color}'>{sign}&nbsp;{abs(delta):.1f}</font>"
    header = Paragraph(
        f"<font name='{_BOLD_FONT}' size='8.5' color='#cc5a00'><b>{label.upper()}</b></font>"
        f"&nbsp;&nbsp;<font size='7' color='#6b7686'>SEASON {season_t} · L10 {recent_t}{delta_html}</font>",
        ParagraphStyle("spark_head", parent=styles["body"], leading=11, textColor=INK_BODY),
    )
    chart = _trend_sparkline(s.tail(20), width=3.3 * inch, height=0.55 * inch)
    cell = Table([[header], [chart]], colWidths=[3.3 * inch], rowHeights=[0.18 * inch, None])
    cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 6),
    ]))
    return cell


def _trend_panel(games: pd.DataFrame, summary: dict[str, Any], styles: dict) -> Table | None:
    """Two-column grid of richer sparkline cards (last 20 games)."""
    cells: list[Any] = []
    for col, label in _KPI_STATS:
        c = _trend_cell(games, col, label, summary, styles)
        if c is not None:
            cells.append(c)
    if not cells:
        return None
    rows: list[list[Any]] = []
    for i in range(0, len(cells), 2):
        left = cells[i]
        right = cells[i + 1] if i + 1 < len(cells) else ""
        rows.append([left, right])
    t = Table(rows, colWidths=[3.45 * inch, 3.45 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _hot_cold_table(summary: dict[str, Any], styles: dict) -> Table | None:
    streaks = summary.get("streaks", {}) if summary else {}
    if not streaks:
        return None
    rows: list[list[Any]] = [["Stat", "L5", "Season", "Δ", "z", "Read"]]
    label_map = dict(_KPI_STATS)
    # Sort by absolute z so the most significant streaks lead.
    items = sorted(
        ((c, info) for c, info in streaks.items() if not math.isnan(info.get("z", float("nan")))),
        key=lambda kv: abs(kv[1].get("z", 0.0)),
        reverse=True,
    )
    for col, info in items[:5]:
        z = info.get("z", 0.0)
        recent = info.get("recent_avg", float("nan"))
        season = info.get("season_avg", float("nan"))
        delta = recent - season
        if z >= 1.5:
            read = "Hot"
        elif z <= -1.5:
            read = "Cold"
        elif abs(z) >= 0.75:
            read = "Trending"
        else:
            read = "Stable"
        rows.append([
            label_map.get(col, col.upper()),
            f"{recent:.1f}" if not math.isnan(recent) else "—",
            f"{season:.1f}" if not math.isnan(season) else "—",
            f"{delta:+.1f}" if not math.isnan(delta) else "—",
            f"{z:+.2f}",
            read,
        ])
    if len(rows) <= 1:
        return None
    t = _styled_table(
        rows,
        col_widths=[0.55 * inch, 0.42 * inch, 0.55 * inch, 0.45 * inch, 0.5 * inch, 0.73 * inch],
        align_right_cols=[1, 2, 3, 4],
    )
    return t


def _player_hero_band(
    player: str,
    summary: dict[str, Any],
    games: pd.DataFrame,
    styles: dict,
) -> Table:
    """Dark hero band with player name + summary chips. Uses paragraph styles
    with explicit leading + spaceBefore so the title and the eyebrow can never
    collide (the table-row approach mis-rendered when the body style had a
    custom leading).
    """
    gp = summary.get("games_played", 0) if summary else 0
    last_dt = ""
    if isinstance(games, pd.DataFrame) and not games.empty and "game_date" in games.columns:
        try:
            last_dt = pd.to_datetime(games["game_date"]).max().strftime("%b %d, %Y")
        except Exception:
            last_dt = ""

    eyebrow_style = ParagraphStyle(
        "hero_eyebrow", parent=styles["body"], fontName=_BOLD_FONT,
        fontSize=8, leading=11, textColor=BRAND_ORANGE_SOFT,
        spaceAfter=2, alignment=TA_LEFT,
    )
    name_style = ParagraphStyle(
        "hero_name", parent=styles["body"], fontName=_BOLD_FONT,
        fontSize=26, leading=30, textColor=WHITE,
        spaceBefore=0, spaceAfter=4, alignment=TA_LEFT,
    )
    sub_style = ParagraphStyle(
        "hero_sub", parent=styles["body"], fontName=_BODY_FONT,
        fontSize=8.5, leading=12, textColor=BRAND_ORANGE_SOFT,
        spaceBefore=0, alignment=TA_LEFT,
    )

    eyebrow = Paragraph("PLAYER PROFILE · HOOPLYTICS", eyebrow_style)
    name = Paragraph(_safe_text(player), name_style)
    sub_bits = [f"{gp} games sampled"]
    if last_dt:
        sub_bits.append(f"last log {last_dt}")
    # Add headline averages directly into the hero so the title block reads
    # like a mini scoreboard and stops being just decoration.
    kpis = summary.get("kpis", {}) if summary else {}
    head_bits: list[str] = []
    for col, lab in (("pts", "PTS"), ("reb", "REB"), ("ast", "AST")):
        v = (kpis.get(col) or {}).get("season_avg")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            head_bits.append(f"{v:.1f} {lab}")
    shoot = summary.get("shooting", {}) if summary else {}
    ts = shoot.get("ts_pct")
    if ts is not None and not (isinstance(ts, float) and math.isnan(ts)):
        head_bits.append(f"{ts*100:.1f}% TS")
    if head_bits:
        sub_bits.append(" · ".join(head_bits))
    sub = Paragraph("  ·  ".join(sub_bits), sub_style)

    inner = Table([[eyebrow], [name], [sub]], colWidths=[7.0 * inch])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK_DARK),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (0, 0), 18),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
        ("TOPPADDING", (0, 2), (0, 2), 2),
        ("BOTTOMPADDING", (0, 2), (0, 2), 18),
        ("LINEBELOW", (0, -1), (-1, -1), 2.5, BRAND_ORANGE),
    ]))
    return inner


def _coaching_hero_cards(
    summary: dict[str, Any],
    ai_section: dict[str, Any] | None,
    styles: dict,
    *,
    total_width: float = 7.0 * inch,
) -> Table | None:
    """Three-up coaching cards (Strengths · Growth · Focus).

    Each card is a small panel with a bold heading, a coloured top accent rule
    and a short body paragraph. Falls back to deterministic prose when no AI
    section is available so the layout never collapses.
    """
    strengths_text = ""
    growth_text = ""
    focus_text = ""
    if isinstance(ai_section, dict):
        strengths_text = _safe_text(str(ai_section.get("strengths", "")).strip())
        growth_text = _safe_text(str(ai_section.get("growth_areas", "")).strip())
        focus_text = _safe_text(str(ai_section.get("coaching_focus", "")).strip())

    if not (strengths_text or growth_text or focus_text):
        # Deterministic fallback derived from streaks + shooting.
        streaks = summary.get("streaks", {}) if summary else {}
        label_map = dict(_KPI_STATS)
        hot = sorted(streaks.items(), key=lambda kv: kv[1].get("z", 0.0), reverse=True)
        cold = sorted(streaks.items(), key=lambda kv: kv[1].get("z", 0.0))
        if hot and hot[0][1].get("z", 0.0) > 0.5:
            col, info = hot[0]
            strengths_text = (
                f"{label_map.get(col, col.upper())} running "
                f"{info.get('recent_avg', 0):.1f} over the last five vs "
                f"a {info.get('season_avg', 0):.1f} season baseline."
            )
        if cold and cold[0][1].get("z", 0.0) < -0.5:
            col, info = cold[0]
            growth_text = (
                f"{label_map.get(col, col.upper())} cooling to "
                f"{info.get('recent_avg', 0):.1f} (season "
                f"{info.get('season_avg', 0):.1f}). Check minutes load and matchups."
            )
        shoot = summary.get("shooting", {}) if summary else {}
        ts = shoot.get("ts_pct")
        if ts is not None and not (isinstance(ts, float) and math.isnan(ts)):
            if ts < 0.50:
                focus_text = (
                    f"True Shooting at {ts * 100:.1f}% — review shot diet, "
                    "emphasise rim pressure and clean catch-and-shoot looks."
                )
            else:
                focus_text = (
                    "Stay the course on rotation and shot mix; lean into "
                    "actions that already produce positive expected value."
                )

    if not (strengths_text or growth_text or focus_text):
        return None

    cards = [
        ("STRENGTHS", strengths_text or "—", colors.HexColor("#1f9d6c")),
        ("GROWTH AREAS", growth_text or "—", colors.HexColor("#cc5a00")),
        ("COACHING FOCUS", focus_text or "—", colors.HexColor("#3a6ea5")),
    ]
    # Layout: three card columns separated by two narrow gutter columns so
    # the cards never share a border and the body text never bleeds out of
    # its panel.
    gutter = 6.0
    col_w = (total_width - 2 * gutter) / 3.0
    # Inside each card we apply 8pt left/right padding via the outer table
    # style, so Paragraphs render into (col_w - 16) of usable width.
    body_style = ParagraphStyle(
        "coach_card_body", parent=styles["body"], fontName=_BODY_FONT,
        fontSize=8.2, leading=11.5, textColor=INK_BODY, alignment=TA_LEFT,
    )
    head_style = ParagraphStyle(
        "coach_card_head", parent=styles["body"], fontName=_BOLD_FONT,
        fontSize=8.0, leading=11, textColor=INK_BODY, alignment=TA_LEFT,
        spaceAfter=2,
    )

    def _build(label: str, body: str, accent: colors.Color) -> Table:
        accent_hex = "#" + accent.hexval()[2:]
        head = Paragraph(
            f"<font color='{accent_hex}'>{label}</font>", head_style,
        )
        body_para = Paragraph(body, body_style)
        card = Table([[head], [body_para]], colWidths=[col_w])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("LINEABOVE", (0, 0), (-1, 0), 2.0, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING", (0, 1), (-1, 1), 0),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return card

    built = [_build(lbl, body, accent) for (lbl, body, accent) in cards]
    row = Table(
        [[built[0], "", built[1], "", built[2]]],
        colWidths=[col_w, gutter, col_w, gutter, col_w],
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row


def _coaching_callout(
    player: str,
    summary: dict[str, Any],
    ai_section: dict[str, Any] | None,
    styles: dict,
) -> Table:
    """Coaching narrative box. Uses AI prose when available, else deterministic.

    The shared ``_callout_box`` helper from :mod:`hooplytics.report` runs the
    text through ``_para`` which XML-escapes ``<``, ``>`` and ``&`` for safety.
    Coaching narrative content is composed from author-controlled labels with
    real ReportLab inline markup (``<b>...</b>``), so we render it through
    :func:`_html_callout_box` which already sanitises the *user* portions via
    ``_safe_text`` before composing the markup.
    """
    if isinstance(ai_section, dict):
        strengths = _safe_text(str(ai_section.get("strengths", "")).strip())
        growth = _safe_text(str(ai_section.get("growth_areas", "")).strip())
        focus = _safe_text(str(ai_section.get("coaching_focus", "")).strip())
        matchup = _safe_text(str(ai_section.get("matchup_context", "")).strip())
        if any([strengths, growth, focus, matchup]):
            blocks: list[str] = []
            if strengths:
                blocks.append(f"<b>Strengths.</b> {strengths}")
            if growth:
                blocks.append(f"<b>Growth areas.</b> {growth}")
            if focus:
                blocks.append(f"<b>Coaching focus.</b> {focus}")
            # NB: ``matchup_context`` is intentionally not rendered here —
            # the AI tends to invent specific opponents/venues that don't
            # match real schedules, so we keep coaching prose to the
            # strengths/growth/focus columns that are grounded in the data.
            return _html_callout_box(blocks, styles, accent=BRAND_ORANGE)

    # Deterministic fallback — top strength + top growth area from streaks/KPIs.
    streaks = summary.get("streaks", {}) if summary else {}
    label_map = dict(_KPI_STATS)
    hot = sorted(streaks.items(), key=lambda kv: kv[1].get("z", 0.0), reverse=True)
    cold = sorted(streaks.items(), key=lambda kv: kv[1].get("z", 0.0))

    parts: list[str] = []
    if hot and hot[0][1].get("z", 0.0) > 0.5:
        col, info = hot[0]
        parts.append(
            f"<b>Trending up.</b> {label_map.get(col, col.upper())} is running "
            f"{info.get('recent_avg', 0):.1f} over the last five vs a "
            f"{info.get('season_avg', 0):.1f} season baseline — keep feeding the role."
        )
    if cold and cold[0][1].get("z", 0.0) < -0.5:
        col, info = cold[0]
        parts.append(
            f"<b>Cooling.</b> {label_map.get(col, col.upper())} has dropped to "
            f"{info.get('recent_avg', 0):.1f} (season {info.get('season_avg', 0):.1f}). "
            "Check minutes load, rotation fit, and opponent matchups."
        )
    shoot = summary.get("shooting", {}) if summary else {}
    ts = shoot.get("ts_pct")
    if ts is not None and not math.isnan(ts):
        if ts >= 0.58:
            parts.append(f"<b>Efficient scorer.</b> True Shooting at {ts*100:.1f}% — high-leverage usage.")
        elif ts < 0.50:
            parts.append(f"<b>Shot quality watch.</b> True Shooting at {ts*100:.1f}% — review shot diet and rim pressure.")
    if not parts:
        parts.append(
            f"<b>Stable profile.</b> {_safe_text(player)} is producing within normal bands across "
            "primary stats; coaching focus can stay on continuity rather than role changes."
        )
    return _html_callout_box(parts, styles, accent=BRAND_ORANGE)


def _html_callout_box(
    blocks: list[str],
    styles: dict,
    *,
    accent: colors.Color = BRAND_ORANGE,
) -> Table:
    """Callout box that renders ReportLab inline markup verbatim.

    ``blocks`` is a list of pre-composed strings; user-supplied substrings are
    expected to have already been passed through ``_safe_text``.
    """
    callout_style = styles["callout"]
    body_style = ParagraphStyle(
        "perf_callout_body",
        parent=callout_style,
        spaceAfter=6,
        leading=callout_style.leading,
    )
    paras = [Paragraph(b, body_style) for b in blocks]
    inner = Table([[p] for p in paras], colWidths=[7.0 * inch - 24])
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    t = Table([[inner]], colWidths=[7.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_ORANGE_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Page composers
# ═══════════════════════════════════════════════════════════════════════════


def _cover_flowables(
    *,
    roster: dict[str, list[str]],
    meta: _ReportMeta,
    summaries: dict[str, dict[str, Any]],
    styles: dict,
) -> list:
    seasons_set = sorted({s for ss in roster.values() for s in ss})
    seasons_label = ", ".join(seasons_set) if seasons_set else "—"
    total_games = sum(int(s.get("games_played", 0) or 0) for s in summaries.values())

    # Compute roster leader stats so the cover surfaces real, meaningful
    # numbers instead of generic counts like "sections per player".
    top_ts_player = "—"
    top_ts_value = "—"
    top_ts_raw = -1.0
    top_pts_player = "—"
    top_pts_value = "—"
    top_pts_raw = -1.0
    for player, ssum in summaries.items():
        ts = (ssum.get("shooting", {}) or {}).get("ts_pct")
        if ts is not None and not (isinstance(ts, float) and math.isnan(ts)):
            if float(ts) > top_ts_raw:
                top_ts_raw = float(ts)
                top_ts_player = player
                top_ts_value = f"{float(ts) * 100:.1f}%"
        pts = ((ssum.get("kpis", {}) or {}).get("pts") or {}).get("season_avg")
        if pts is not None and not (isinstance(pts, float) and math.isnan(pts)):
            if float(pts) > top_pts_raw:
                top_pts_raw = float(pts)
                top_pts_player = player
                top_pts_value = f"{float(pts):.1f}"

    flow: list = [
        Spacer(1, 1.1 * inch),
        _para("HOOPLYTICS  |  PLAYER PERFORMANCE", styles["cover_eyebrow"]),
        _para("Performance Analytics Report.", styles["cover_title"]),
        _para(
            "Per-player development analytics for coaching staffs — KPI "
            "scorecards, trend signals, shooting & efficiency, role "
            "trends, and skill-axis radars across the active roster.",
            styles["cover_sub"],
        ),
        Spacer(1, 0.35 * inch),
    ]

    def _tile(label: str, value: str, footnote: str | None = None) -> Table:
        rows: list[list[Any]] = [
            [_para(label.upper(), styles["cover_meta_label"])],
            [_para(value, styles["cover_meta_value"])],
        ]
        if footnote:
            rows.append([_para(footnote, ParagraphStyle(
                "cover_meta_foot", parent=styles["body"], fontName=_BODY_FONT,
                fontSize=7.0, leading=9, textColor=INK_MUTED, alignment=TA_LEFT,
                spaceBefore=2,
            ))])
        cell = Table(rows, colWidths=[1.55 * inch])
        sty = [
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
            ("LINEBELOW", (0, 1), (0, 1), 1.4, BRAND_ORANGE),
        ]
        cell.setStyle(TableStyle(sty))
        return cell

    tiles = Table(
        [[
            _tile("Players", str(meta.roster_count)),
            _tile("Game logs", str(total_games)),
            _tile("Top scorer", top_pts_value, _safe_text(top_pts_player) if top_pts_player != "—" else None),
            _tile("Top TS%", top_ts_value, _safe_text(top_ts_player) if top_ts_player != "—" else None),
        ]],
        colWidths=[1.55 * inch] * 4,
    )
    tiles.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(tiles)
    flow.append(Spacer(1, 0.6 * inch))

    # Cover hero — dark band that mirrors the player-page hero strip and
    # surfaces the roster headline averages so the cover doesn’t feel empty.
    headline_rows: list[list[Any]] = []
    for player, ssum in summaries.items():
        kpis = ssum.get("kpis", {})
        shoot = ssum.get("shooting", {})
        def _v(col: str) -> str:
            v = (kpis.get(col) or {}).get("season_avg")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.1f}"
        ts = shoot.get("ts_pct")
        ts_t = "—" if ts is None or (isinstance(ts, float) and math.isnan(ts)) else f"{ts*100:.1f}%"
        headline_rows.append([player, _v("pts"), _v("reb"), _v("ast"), ts_t])

    if headline_rows:
        hero_inner = [[
            Paragraph("PLAYER", ParagraphStyle("ch", parent=styles["body"], fontName=_BOLD_FONT, fontSize=7.5, textColor=BRAND_ORANGE_SOFT, alignment=TA_LEFT, leading=10)),
            Paragraph("PTS", ParagraphStyle("ch", parent=styles["body"], fontName=_BOLD_FONT, fontSize=7.5, textColor=BRAND_ORANGE_SOFT, alignment=TA_CENTER, leading=10)),
            Paragraph("REB", ParagraphStyle("ch", parent=styles["body"], fontName=_BOLD_FONT, fontSize=7.5, textColor=BRAND_ORANGE_SOFT, alignment=TA_CENTER, leading=10)),
            Paragraph("AST", ParagraphStyle("ch", parent=styles["body"], fontName=_BOLD_FONT, fontSize=7.5, textColor=BRAND_ORANGE_SOFT, alignment=TA_CENTER, leading=10)),
            Paragraph("TS%", ParagraphStyle("ch", parent=styles["body"], fontName=_BOLD_FONT, fontSize=7.5, textColor=BRAND_ORANGE_SOFT, alignment=TA_CENTER, leading=10)),
        ]]
        for player, p, r, a, t in headline_rows:
            anchor = _player_anchor_key(player)
            hero_inner.append([
                Paragraph(f"<a href='#{anchor}' color='#ffffff'><font color='#ffffff' name='{_BOLD_FONT}' size='11'>{_safe_text(player)}</font></a>", ParagraphStyle("hp", parent=styles["body"], leading=14)),
                Paragraph(f"<font color='#ffffff' name='{_BOLD_FONT}' size='13'>{p}</font>", ParagraphStyle("hv", parent=styles["body"], alignment=TA_CENTER, leading=15)),
                Paragraph(f"<font color='#ffffff' name='{_BOLD_FONT}' size='13'>{r}</font>", ParagraphStyle("hv", parent=styles["body"], alignment=TA_CENTER, leading=15)),
                Paragraph(f"<font color='#ffffff' name='{_BOLD_FONT}' size='13'>{a}</font>", ParagraphStyle("hv", parent=styles["body"], alignment=TA_CENTER, leading=15)),
                Paragraph(f"<font color='#ffffff' name='{_BOLD_FONT}' size='13'>{t}</font>", ParagraphStyle("hv", parent=styles["body"], alignment=TA_CENTER, leading=15)),
            ])
        hero = Table(hero_inner, colWidths=[2.6 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch, 1.05 * inch])
        hero_style: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, -1), INK_DARK),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, BRAND_ORANGE),
            ("LINEBELOW", (0, -1), (-1, -1), 2.0, BRAND_ORANGE),
            ("TOPPADDING", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]
        for ridx in range(1, len(hero_inner) - 1):
            hero_style.append(("LINEBELOW", (0, ridx), (-1, ridx), 0.4, colors.Color(1, 1, 1, 0.10)))
        hero.setStyle(TableStyle(hero_style))
        flow.append(hero)
        flow.append(Spacer(1, 0.35 * inch))

    flow.append(_para(f"Generated  |  {meta.generated_at}", styles["cover_tag"]))
    flow.append(_para(f"Seasons  |  {seasons_label}", styles["cover_tag"]))
    flow.append(_para(
        "Prose  |  " + ("AI-augmented coaching narrative" if meta.has_ai else "data-only"),
        styles["cover_tag"],
    ))
    return flow


def _roster_overview_flowables(
    *,
    roster: dict[str, list[str]],
    summaries: dict[str, dict[str, Any]],
    ai_overview: str,
    styles: dict,
) -> list:
    flow: list = []
    flow.extend(_section_header(
        "Roster overview",
        "01 · summary",
        styles,
        anchor="perf_overview",
    ))

    if ai_overview:
        flow.append(_html_callout_box([_safe_text(ai_overview)], styles, accent=BRAND_ORANGE))
        flow.append(Spacer(1, 8))

    # Snapshot table: per-player season averages across primary stats.
    rows: list[list[Any]] = [
        ["Player", "GP", "PTS", "REB", "AST", "PRA", "MIN", "FAN", "TS%"],
    ]
    for player in roster.keys():
        s = summaries.get(player) or {}
        kpis = s.get("kpis", {})
        shoot = s.get("shooting", {})

        def _val(col: str) -> str:
            v = kpis.get(col, {}).get("season_avg")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.1f}"

        ts = shoot.get("ts_pct")
        ts_text = "—" if ts is None or (isinstance(ts, float) and math.isnan(ts)) else f"{ts*100:.1f}%"
        rows.append([
            _safe_text(player),
            str(s.get("games_played", 0)),
            _val("pts"), _val("reb"), _val("ast"), _val("pra"),
            _val("min"), _val("fantasy_score"), ts_text,
        ])

    if len(rows) > 1:
        t = _styled_table(
            rows,
            col_widths=[1.7 * inch, 0.5 * inch, 0.55 * inch, 0.55 * inch,
                        0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.65 * inch],
            align_right_cols=[1, 2, 3, 4, 5, 6, 7, 8],
        )
        flow.append(t)
        flow.append(Spacer(1, 12))

    # Roster comparison radar — overlay every player on the same axes so a
    # coach can see the skill-shape contrast at a glance.
    # Distinct, high-contrast palette so overlaid polygons stay readable.
    palette = [
        BRAND_ORANGE,                       # vivid orange
        colors.HexColor("#3a6ea5"),         # cobalt
        colors.HexColor("#1f9d6c"),         # forest green
        colors.HexColor("#7e57c2"),         # violet
        colors.HexColor("#0d9488"),         # teal
        colors.HexColor("#d24545"),         # crimson
    ]
    overlay: list[tuple[str, list[tuple[str, float]], colors.Color]] = []
    for idx, (player, ssum) in enumerate(summaries.items()):
        comps = _radar_components(ssum, summaries)
        if comps:
            overlay.append((player, comps, palette[idx % len(palette)]))
    if len(overlay) >= 2:
        flow.append(_para("ROSTER SKILL OVERLAY", styles["eyebrow"]))
        flow.append(Spacer(1, 4))
        radar = _comparison_radar(overlay, width=4.0 * inch, height=2.4 * inch)
        # Build a colour legend.
        legend_cells: list[list[Any]] = []
        for player, _comps, color in overlay:
            swatch = Drawing(10, 10)
            swatch.add(Rect(0, 1, 10, 8, fillColor=color, strokeColor=color))
            legend_cells.append([
                swatch,
                Paragraph(f"<font name='{_BOLD_FONT}' size='8.5'>{_safe_text(player)}</font>",
                          ParagraphStyle("leg", parent=styles["body"], leading=11, textColor=INK_DARK)),
            ])
        legend = Table(legend_cells, colWidths=[0.18 * inch, 2.6 * inch])
        legend.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        compare = Table([[radar, legend]], colWidths=[4.2 * inch, 2.8 * inch])
        compare.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(compare)
        flow.append(Spacer(1, 8))

    flow.append(_para(
        "Each player gets a dedicated profile in the pages that follow with "
        "trend sparklines, shooting and efficiency, a skill-axis radar, "
        "consistency bands, role/usage trends, and hot/cold streak detection.",
        styles["muted"],
    ))
    return flow


def _player_anchor_key(name: str) -> str:
    """Stable PDF bookmark key for a player so the cover can deep-link to it."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return f"player_{slug or 'unknown'}"


def _player_profile_flowables(
    *,
    player: str,
    games: pd.DataFrame,
    summary: dict[str, Any],
    roster_baseline: dict[str, dict[str, Any]],
    ai_section: dict[str, Any] | None,
    styles: dict,
    section_index: int,
) -> list:
    flow: list = []
    flow.append(PageBreak())
    # Bookmark this page so the cover roster can deep-link to it.
    flow.append(_AnchorFlowable(_player_anchor_key(player), player, level=0))
    # The dark hero band is the player title — no separate section header so
    # we don’t double up on the player name.
    flow.append(_player_hero_band(player, summary, games, styles))
    flow.append(Spacer(1, 8))

    # KPI strip.
    kpi_strip = _kpi_strip_for_player(summary, styles)
    if kpi_strip is not None:
        flow.append(kpi_strip)
        flow.append(Spacer(1, 8))

    # Garmin-style activity rings + ML next-game forecast — side by side so
    # the ML projection lands on the same page as the player’s hero numbers.
    rings_drawing = _player_activity_rings(
        summary, roster_baseline, width=2.7 * inch, height=1.65 * inch,
    )
    rings_label = Paragraph(
        f"<font name='{_BOLD_FONT}' size='8.5' color='#cc5a00'><b>ACTIVITY RINGS</b></font>",
        ParagraphStyle("ring_head", parent=styles["body"], leading=11, textColor=INK_BODY),
    )
    rings_panel = Table(
        [[rings_label], [rings_drawing]],
        colWidths=[2.9 * inch],
        rowHeights=[0.22 * inch, None],
    )
    rings_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    forecast = _ml_next_game_forecast(games, summary)
    if forecast:
        forecast_drawing = _ml_forecast_panel(
            forecast, width=3.85 * inch, height=1.65 * inch,
        )
        forecast_label = Paragraph(
            f"<font name='{_BOLD_FONT}' size='8.5' color='#cc5a00'><b>ML NEXT-GAME PROJECTION</b></font>",
            ParagraphStyle("ml_head", parent=styles["body"], leading=11, textColor=INK_BODY),
        )
        forecast_panel = Table(
            [[forecast_label], [forecast_drawing]],
            colWidths=[4.0 * inch],
            rowHeights=[0.22 * inch, None],
        )
        forecast_panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (0, 0), 6),
            ("BOTTOMPADDING", (0, 0), (0, 0), 0),
            ("TOPPADDING", (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (0, 1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        side_row = Table([[rings_panel, forecast_panel]],
                         colWidths=[2.95 * inch, 4.05 * inch])
    else:
        side_row = Table([[rings_panel]], colWidths=[7.0 * inch])
    side_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(side_row)
    flow.append(Spacer(1, 10))

    # Trend sparklines.
    flow.append(_para("TREND · LAST 20 GAMES", styles["eyebrow"]))
    flow.append(Spacer(1, 2))
    trend = _trend_panel(games, summary, styles)
    if trend is not None:
        flow.append(trend)
    else:
        flow.append(_para("Not enough recent games to plot trends.", styles["muted"]))
    flow.append(Spacer(1, 10))

    # ─── PAGE 2: shooting + radar, consistency, role + hot/cold, coaching
    flow.append(PageBreak())

    # Roster baseline shooting (median of TS / FG / FT / 3P across roster).
    baseline_shoot: dict[str, float] = {}
    for key in ("fg_pct", "fg3_pct", "ft_pct", "ts_pct"):
        vals = []
        for s in roster_baseline.values():
            v = (s.get("shooting") or {}).get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
        if vals:
            baseline_shoot[key] = float(np.median(vals))

    shoot_drawing = _shooting_bars(
        summary.get("shooting", {}) if summary else {},
        baseline_shoot or None,
        width=3.4 * inch,
        height=1.6 * inch,
    )
    radar_drawing = _radar_chart(
        _radar_components(summary, roster_baseline),
        width=3.4 * inch,
        height=1.9 * inch,
    )
    side_by_side = Table(
        [[shoot_drawing, radar_drawing]],
        colWidths=[3.5 * inch, 3.5 * inch],
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBETWEEN", (0, 0), (0, -1), 0.4, PANEL_BORDER),
    ]))
    flow.append(KeepTogether([
        _para("SHOOTING & SKILL PROFILE", styles["eyebrow"]),
        Spacer(1, 2),
        side_by_side,
    ]))
    flow.append(Spacer(1, 6))

    # Consistency strip.
    dist = _distribution_strip(summary, width=7.0 * inch, height=1.55 * inch)
    dist_table = Table([[dist]], colWidths=[7.0 * inch])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.4, PANEL_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(KeepTogether([
        _para("CONSISTENCY · FLOOR / MEDIAN / CEILING", styles["eyebrow"]),
        dist_table,
    ]))
    flow.append(Spacer(1, 6))

    # Role / usage table + hot/cold table side by side to use full width.
    role_rows: list[list[Any]] = [["", "Season", "Last 10", "Δ"]]
    kpis = summary.get("kpis", {}) if summary else {}
    if "min" in kpis:
        info = kpis["min"]
        role_rows.append([
            "Minutes",
            _fmt(info.get("season_avg"), 1),
            _fmt(info.get("recent_avg"), 1),
            _fmt_signed(info.get("delta"), 1),
        ])
    usage = summary.get("usage") if summary else None
    if usage:
        role_rows.append([
            "Usage proxy",
            _fmt(usage.get("season_avg"), 3),
            _fmt(usage.get("recent_avg"), 3),
            _fmt_signed(
                (usage.get("recent_avg") or 0.0) - (usage.get("season_avg") or 0.0), 3
            ) if usage.get("season_avg") is not None else "—",
        ])
    if "fantasy_score" in kpis:
        info = kpis["fantasy_score"]
        role_rows.append([
            "Fantasy score",
            _fmt(info.get("season_avg"), 1),
            _fmt(info.get("recent_avg"), 1),
            _fmt_signed(info.get("delta"), 1),
        ])
    role_table = None
    if len(role_rows) > 1:
        role_table = _styled_table(
            role_rows,
            col_widths=[1.0 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch],
            align_right_cols=[1, 2, 3],
        )

    hot = _hot_cold_table(summary, styles)

    pair_cells: list[Any] = []
    if role_table is not None:
        pair_cells.append([
            _para("ROLE & USAGE", styles["eyebrow"]),
            Spacer(1, 2),
            role_table,
        ])
    if hot is not None:
        pair_cells.append([
            _para("HOT / COLD · LAST 5 vs SEASON", styles["eyebrow"]),
            Spacer(1, 2),
            hot,
        ])
    if pair_cells:
        if len(pair_cells) == 2:
            cell_l = Table([[x] for x in pair_cells[0]], colWidths=[3.4 * inch])
            cell_r = Table([[x] for x in pair_cells[1]], colWidths=[3.4 * inch])
            for c in (cell_l, cell_r):
                c.setStyle(TableStyle([
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]))
            pair = Table([[cell_l, cell_r]], colWidths=[3.5 * inch, 3.5 * inch])
            pair.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            flow.append(KeepTogether(pair))
            flow.append(Spacer(1, 6))
        else:
            for blk in pair_cells:
                for item in blk:
                    flow.append(item)
                flow.append(Spacer(1, 6))

    # Coaching narrative as three accent-topped hero cards (Strengths /
    # Growth / Focus). Falls back to a deterministic version when no AI
    # section is present. Appended inline (no KeepTogether wrap) so the
    # cards flow directly under the role/hot tables instead of getting
    # bumped to a fresh page.
    cards = _coaching_hero_cards(summary, ai_section, styles, total_width=7.0 * inch)
    if cards is not None:
        flow.append(_para("COACHING NOTE", styles["eyebrow"]))
        flow.append(Spacer(1, 3))
        flow.append(cards)
    return flow


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════


def build_player_performance_report(
    *,
    roster: dict[str, list[str]],
    player_games: dict[str, pd.DataFrame],
    bundle_metrics: pd.DataFrame | None = None,
    ai_sections: dict[str, Any] | None = None,
) -> bytes:
    """Render the Player Performance Analytics PDF and return raw bytes.

    Parameters
    ----------
    roster
        ``{player: [season, ...]}`` from the active app state.
    player_games
        ``{player: DataFrame}`` of per-game box-score logs (already in the
        modeling-frame schema: pts/reb/ast/min/fga/fgm/fg3a/fg3m/fta/ftm/...).
    bundle_metrics
        Optional model metrics frame — currently unused in the visuals but
        accepted to keep the signature parallel to ``build_pdf_report`` for
        future extension.
    ai_sections
        Optional dict from :func:`generate_performance_sections` with keys
        ``roster_overview`` (str) and ``players``
        (``{name: {strengths, growth_areas, coaching_focus, matchup_context}}``).
        Pass ``None`` to render the deterministic data-only version.
    """
    # ── Sanitize untrusted strings exactly like the projections report ──
    roster = {
        _safe_text(group): [_safe_text(p) for p in players]
        for group, players in (roster or {}).items()
    }
    cleaned_games: dict[str, pd.DataFrame] = {}
    for k, v in (player_games or {}).items():
        if isinstance(v, pd.DataFrame):
            cleaned_games[_safe_text(k)] = v
    player_games = cleaned_games

    # AI sections — accept legacy 'players' as dict[str, str] or dict[str, dict].
    safe_ai: dict[str, Any] = {"roster_overview": "", "players": {}}
    if isinstance(ai_sections, dict):
        safe_ai["roster_overview"] = _safe_text(str(ai_sections.get("roster_overview", "")).strip())
        raw_players = ai_sections.get("players") or {}
        if isinstance(raw_players, dict):
            for k, v in raw_players.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, dict):
                    safe_ai["players"][_safe_text(k)] = {
                        kk: _safe_text(str(vv or "").strip()) for kk, vv in v.items()
                    }
                elif isinstance(v, str):
                    safe_ai["players"][_safe_text(k)] = {
                        "coaching_focus": _safe_text(v.strip()),
                    }

    # ── Pre-compute summaries (used by overview, profiles, and radar baseline) ──
    summaries: dict[str, dict[str, Any]] = {}
    for player in roster.keys():
        games = player_games.get(player)
        summaries[player] = player_performance_summary(games) if isinstance(games, pd.DataFrame) else {"games_played": 0}

    meta = _ReportMeta(
        generated_at=datetime.now().strftime("%b %d, %Y  |  %H:%M"),
        roster_count=len(roster or {}),
        has_ai=bool(safe_ai["roster_overview"] or safe_ai["players"]),
    )

    # ── Document scaffolding (same chrome as projections report) ──
    buf = BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Hooplytics Player Performance Analytics",
        author="Hooplytics",
    )

    cover_frame = Frame(
        0.95 * inch, 0.6 * inch,
        LETTER[0] - 1.55 * inch, LETTER[1] - 1.2 * inch,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="perf_cover",
    )
    body_frame = Frame(
        doc.leftMargin, doc.bottomMargin + 0.15 * inch,
        doc.width, doc.height - 0.4 * inch,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="perf_body",
    )
    doc.addPageTemplates([
        PageTemplate(id="perf_cover", frames=[cover_frame], onPage=_draw_cover_chrome),
        PageTemplate(
            id="perf_body",
            frames=[body_frame],
            onPage=lambda c, d: _draw_page_chrome(c, d, meta),
        ),
    ])

    styles = _v2_styles(_build_styles())

    flow: list = []
    # Page 1 — cover.
    flow.extend(_cover_flowables(roster=roster, meta=meta, summaries=summaries, styles=styles))
    flow.append(NextPageTemplate("perf_body"))

    # Page 2 — roster overview.
    flow.append(PageBreak())
    flow.extend(_roster_overview_flowables(
        roster=roster,
        summaries=summaries,
        ai_overview=safe_ai["roster_overview"],
        styles=styles,
    ))

    # Pages 3+ — per-player profiles.
    for idx, player in enumerate(roster.keys(), start=2):
        flow.extend(_player_profile_flowables(
            player=player,
            games=player_games.get(player, pd.DataFrame()),
            summary=summaries.get(player) or {"games_played": 0},
            roster_baseline=summaries,
            ai_section=safe_ai["players"].get(player),
            styles=styles,
            section_index=idx,
        ))

    doc.build(flow)
    return buf.getvalue()


__all__ = [
    "build_player_performance_report",
    "player_performance_summary",
]
