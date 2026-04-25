"""Reusable Plotly chart factories — port + enhance the notebook visualizations."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hooplytics.web.styles import (
    COLOR_ACCENT,
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_LESS,
    COLOR_MORE,
    PLAYER_PALETTE,
    player_color_map,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(color=COLOR_AXIS, size=14),
    )
    fig.update_layout(
        height=240,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig


def _resolve_date_col(df: pd.DataFrame) -> str:
    if "game_date" in df.columns:
        return "game_date"
    if "GAME_DATE" in df.columns:
        return "GAME_DATE"
    raise KeyError("Expected a game_date or GAME_DATE column in the games DataFrame")


def _insert_offseason_gaps(df: pd.DataFrame, date_col: str = "game_date",
                           gap_days: int = 60) -> pd.DataFrame:
    """Insert NaN sentinel rows so Plotly draws gaps across the offseason."""
    if df.empty:
        return df
    df = df.sort_values(date_col).reset_index(drop=True)
    deltas = df[date_col].diff().dt.days.fillna(0)
    breaks = df.index[deltas > gap_days].tolist()
    if not breaks:
        return df
    pieces: list[pd.DataFrame] = []
    last = 0
    for b in breaks:
        pieces.append(df.iloc[last:b])
        gap_row = {c: np.nan for c in df.columns}
        gap_row[date_col] = df.loc[b, date_col] - pd.Timedelta(days=1)
        pieces.append(pd.DataFrame([gap_row]))
        last = b
    pieces.append(df.iloc[last:])
    return pd.concat(pieces, ignore_index=True)


# ── 1. Rolling form chart ────────────────────────────────────────────────────
def rolling_form_chart(
    games: pd.DataFrame,
    metric: str,
    *,
    window: int = 10,
    player_label: str | None = None,
    color: str = COLOR_ACCENT,
    season_avg: float | None = None,
    line_value: float | None = None,
) -> go.Figure:
    """Per-game scatter + rolling mean + season/line reference lines, with offseason gaps."""
    date_col = _resolve_date_col(games)
    cols = [date_col, metric]
    if "MATCHUP" in games.columns:
        cols.append("MATCHUP")
    df = games[cols].copy()
    if "MATCHUP" not in df.columns:
        df["MATCHUP"] = ""
    df = df.sort_values(date_col).reset_index(drop=True)
    df["rolling"] = df[metric].rolling(window, min_periods=3).mean()
    df = _insert_offseason_gaps(df, date_col=date_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[metric], mode="markers",
        name="Game",
        marker=dict(size=6, color=color, opacity=0.45,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
        customdata=df[["MATCHUP"]].fillna(""),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{customdata[0]}<br>"
                      f"{metric}: %{{y:.1f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df["rolling"], mode="lines",
        name=f"{window}-game avg",
        line=dict(width=3, color=color, shape="spline", smoothing=0.6),
        connectgaps=False,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
                      f"Rolling {metric}: %{{y:.1f}}<extra></extra>",
    ))
    if season_avg is not None and not np.isnan(season_avg):
        fig.add_hline(y=season_avg, line=dict(color=COLOR_AXIS, dash="dot", width=1.2),
                      annotation_text=f"season μ {season_avg:.1f}",
                      annotation_position="bottom right",
                      annotation_font_color=COLOR_AXIS)
    if line_value is not None:
        fig.add_hline(y=line_value, line=dict(color="#f0e442", dash="dash", width=1.6),
                      annotation_text=f"line {line_value:g}",
                      annotation_position="top right",
                      annotation_font_color="#f0e442")

    title = f"{player_label} — {metric}" if player_label else metric
    fig.update_layout(
        title=title, height=380,
        yaxis_title=metric, xaxis_title=None,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    return fig


# ── 2. Faceted distribution histograms ──────────────────────────────────────
def distribution_facets(
    games_by_player: dict[str, pd.DataFrame],
    metrics: list[str],
    *,
    cols: int = 4,
) -> go.Figure:
    """Per-metric histogram facets across players with mean lines."""
    cmap = player_color_map(list(games_by_player.keys()))
    n = len(metrics)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics,
                        horizontal_spacing=0.07, vertical_spacing=0.18)

    for idx, metric in enumerate(metrics):
        r, c = idx // cols + 1, idx % cols + 1
        for p_idx, (name, gdf) in enumerate(games_by_player.items()):
            if metric not in gdf.columns:
                continue
            vals = gdf[metric].dropna()
            if vals.empty:
                continue
            color = cmap[name]
            show_legend = idx == 0
            fig.add_trace(go.Histogram(
                x=vals, name=name, legendgroup=name, showlegend=show_legend,
                marker=dict(color=_hex_to_rgba(color, 0.55),
                            line=dict(color=color, width=1)),
                nbinsx=18, opacity=0.75,
                hovertemplate=f"<b>{name}</b><br>{metric}: %{{x}}<br>"
                              "games: %{y}<extra></extra>",
            ), row=r, col=c)
            mean_val = vals.mean()
            fig.add_vline(x=mean_val, line=dict(color=color, width=1.4, dash="dash"),
                          row=r, col=c)

    fig.update_layout(
        barmode="overlay",
        height=260 * rows + 80,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(t=60, b=80, l=40, r=20),
        title="Distribution by metric — recent games",
    )
    fig.update_annotations(font_size=12)
    return fig


# ── 3. Violin grid ──────────────────────────────────────────────────────────
def violin_grid(
    games_by_player: dict[str, pd.DataFrame],
    metrics: list[str],
    *,
    cols: int = 4,
) -> go.Figure:
    cmap = player_color_map(list(games_by_player.keys()))
    n = len(metrics)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics,
                        horizontal_spacing=0.06, vertical_spacing=0.20)

    for idx, metric in enumerate(metrics):
        r, c = idx // cols + 1, idx % cols + 1
        for p_idx, (name, gdf) in enumerate(games_by_player.items()):
            if metric not in gdf.columns:
                continue
            vals = gdf[metric].dropna()
            if vals.empty:
                continue
            color = cmap[name]
            show_legend = idx == 0
            short = name.split()[-1] if " " in name else name
            fig.add_trace(go.Violin(
                y=vals, name=name, legendgroup=name, showlegend=show_legend,
                x=[short] * len(vals),
                line=dict(color=color, width=1.2),
                fillcolor=_hex_to_rgba(color, 0.35),
                box=dict(visible=True, width=0.25,
                         fillcolor=_hex_to_rgba(color, 0.55),
                         line=dict(color="white", width=1)),
                meanline=dict(visible=True, color="white", width=1.5),
                points=False, spanmode="hard", scalemode="width",
                hovertemplate=f"<b>{name}</b><br>{metric}: %{{y:.1f}}<extra></extra>",
            ), row=r, col=c)

    fig.update_layout(
        height=300 * rows + 80,
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=60, b=80, l=40, r=20),
        title="Shape & spread by metric",
        showlegend=True,
    )
    fig.update_annotations(font_size=12)
    return fig


# ── 4. Normalized radar (player vs roster) ──────────────────────────────────
def normalized_radar(
    profiles: dict[str, dict[str, float]],
    *,
    title: str = "Skill profile (normalized 0–1)",
) -> go.Figure:
    """profiles: {player_name: {metric: normalized_value_0_1}}"""
    if not profiles:
        return go.Figure()
    cmap = player_color_map(list(profiles.keys()))
    cats = list(next(iter(profiles.values())).keys())
    fig = go.Figure()
    for name, vals_dict in profiles.items():
        color = cmap[name]
        vals = [vals_dict.get(c, 0.0) for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.22),
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=1)),
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title, height=480,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickvals=[0.25, 0.5, 0.75, 1.0],
                                   tickfont=dict(size=10))),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    )
    return fig


# ── 5. Predicted vs actual grid (port of notebook §4.2) ─────────────────────
def predicted_vs_actual_grid(
    panels: list[dict],
    *,
    cols: int = 4,
) -> go.Figure:
    """
    panels: list of dicts, each {
        'metric': str, 'r2': float,
        'points': [{'player','date','matchup','actual','pred','color'} ...]
    }
    """
    n = len(panels)
    if n == 0:
        return go.Figure()
    rows = (n + cols - 1) // cols
    titles = [f"<b>{p['metric']}</b>  R²={p['r2']:.2f}" for p in panels]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.20)

    seen_players: set[str] = set()
    for idx, panel in enumerate(panels):
        r, c = idx // cols + 1, idx % cols + 1
        pts = panel["points"]
        if not pts:
            continue
        # group points by player so legend entries collapse
        df = pd.DataFrame(pts)
        all_a = df["actual"].astype(float)
        all_p = df["pred"].astype(float)
        lo = float(min(all_a.min(), all_p.min()))
        hi = float(max(all_a.max(), all_p.max()))
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        # diagonal
        fig.add_trace(go.Scatter(
            x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=r, col=c)

        for player, sub in df.groupby("player"):
            show = player not in seen_players
            seen_players.add(player)
            color = sub["color"].iloc[0]
            customdata = np.column_stack([
                sub["date"].astype(str).values,
                sub["matchup"].astype(str).values,
                [player] * len(sub),
                [panel["metric"]] * len(sub),
            ])
            fig.add_trace(go.Scatter(
                x=sub["actual"], y=sub["pred"], mode="markers",
                name=player, legendgroup=player, showlegend=show,
                marker=dict(size=8, color=color, opacity=0.78,
                            line=dict(color="white", width=0.6)),
                customdata=customdata,
                hovertemplate=("<b>%{customdata[2]}</b> — %{customdata[3]}<br>"
                               "%{customdata[0]} · %{customdata[1]}<br>"
                               "actual=%{x:.1f}<br>pred=%{y:.1f}<extra></extra>"),
            ), row=r, col=c)
        fig.update_xaxes(title_text="actual", row=r, col=c, range=[lo - pad, hi + pad])
        fig.update_yaxes(title_text="pred", row=r, col=c, range=[lo - pad, hi + pad])

    fig.update_layout(
        height=340 * rows + 140,
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=50, r=30),
        title="Predicted vs actual — held-out games",
    )
    fig.update_annotations(font_size=12)
    return fig


# ── 6. Feature importance ───────────────────────────────────────────────────
def feature_importance_bar(importances: pd.Series, *, title: str,
                           top_n: int = 20) -> go.Figure:
    s = importances.sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker=dict(color=s.values, colorscale=[[0, "#0e1117"], [1, COLOR_ACCENT]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        hovertemplate="<b>%{y}</b><br>importance: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(360, 22 * len(s) + 100),
        xaxis_title="importance", yaxis_title=None,
        margin=dict(l=160, r=30, t=50, b=40),
    )
    return fig


# ── 7. Projection-gap bar (analytics board) ─────────────────────────────────
def edge_bar_chart(df: pd.DataFrame, *, top_n: int = 12) -> go.Figure:
    """df must have: player, model, posted line, model prediction, edge, side."""
    if df.empty:
        return _empty_figure("No projection gaps to display")
    d = df.assign(abs_edge=df["edge"].abs()).nlargest(top_n, "abs_edge").iloc[::-1]
    labels = (d["player"] + " · " + d["model"]).tolist()
    colors = [
        COLOR_MORE if str(side).upper() == "MORE" or edge > 0 else COLOR_LESS
        for side, edge in zip(d["side"], d["edge"])
    ]
    fig = go.Figure(go.Bar(
        x=d["edge"], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=0.6)),
        text=[
            f"line {line:g} · proj {projection:.1f}"
            for line, projection in zip(d["posted line"], d["model prediction"])
        ],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>projection gap: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.5)", width=1))
    fig.update_layout(
        title=f"Top {len(d)} projection gaps", height=max(360, 28 * len(d) + 120),
        xaxis_title="projection − line", yaxis_title=None,
        margin=dict(l=240, r=30, t=50, b=40),
    )
    return fig


# ── 8. Projection-gap distribution histogram ────────────────────────────────
def edge_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data to display")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["edge"], nbinsx=24,
        marker=dict(color=_hex_to_rgba(COLOR_ACCENT, 0.6),
                    line=dict(color=COLOR_ACCENT, width=1)),
        hovertemplate="gap bin: %{x}<br>count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(title="Projection gap distribution across the slate",
                      xaxis_title="projection − line", yaxis_title="count",
                      height=300)
    return fig


# ── 9. Consistency leaderboard (CV bar) ─────────────────────────────────────
def consistency_bars(df: pd.DataFrame, *, metric: str) -> go.Figure:
    """df indexed by player, columns include 'mean', 'std', 'cv'."""
    if df.empty:
        return go.Figure()
    d = df.sort_values("cv")
    fig = go.Figure(go.Bar(
        x=d["cv"], y=d.index, orientation="h",
        marker=dict(color=d["cv"],
                    colorscale=[[0, COLOR_MORE], [0.5, "#f0e442"], [1, COLOR_LESS]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"μ={m:.1f} · σ={s:.1f}" for m, s in zip(d["mean"], d["std"])],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>CV: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Consistency leaderboard — {metric} (lower CV = steadier)",
        xaxis_title="coefficient of variation (σ/μ)", yaxis_title=None,
        height=max(320, 28 * len(d) + 80),
        margin=dict(l=180, r=30, t=50, b=40),
    )
    return fig


# ── 10. R² gauge ────────────────────────────────────────────────────────────
def r2_gauge(r2: float, *, title: str) -> go.Figure:
    val = max(-1.0, min(1.0, float(r2)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(font=dict(size=28, color="#f5f5f5"), valueformat=".2f"),
        title=dict(text=title, font=dict(size=13, color=COLOR_AXIS)),
        gauge=dict(
            axis=dict(range=[-0.2, 1], tickwidth=1, tickcolor=COLOR_AXIS,
                      tickfont=dict(color=COLOR_AXIS, size=10)),
            bar=dict(color=COLOR_ACCENT, thickness=0.32),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=0,
            steps=[
                dict(range=[-0.2, 0.2], color="rgba(255,107,107,0.25)"),
                dict(range=[0.2, 0.5], color="rgba(240,228,66,0.20)"),
                dict(range=[0.5, 1.0], color="rgba(61,220,151,0.20)"),
            ],
        ),
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── 11. Single-metric distribution ─────────────────────────────────────────
def metric_distribution_chart(
    games: pd.DataFrame,
    metric: str,
    *,
    player_label: str | None = None,
    color: str = COLOR_ACCENT,
) -> go.Figure:
    """Histogram of a single metric for one player, with mean/median lines."""
    if games is None or games.empty or metric not in games.columns:
        return _empty_figure("No data for this metric")
    vals = pd.to_numeric(games[metric], errors="coerce").dropna()
    if vals.empty:
        return _empty_figure("No data for this metric")
    mean_val = float(vals.mean())
    median_val = float(vals.median())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=18,
        marker=dict(color=_hex_to_rgba(color, 0.55),
                    line=dict(color=color, width=1)),
        hovertemplate=f"{metric}: %{{x}}<br>games: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=mean_val, line=dict(color=color, width=1.6, dash="dash"),
                  annotation_text=f"μ {mean_val:.1f}",
                  annotation_position="top right",
                  annotation_font_color=color)
    fig.add_vline(x=median_val, line=dict(color=COLOR_AXIS, width=1.2, dash="dot"),
                  annotation_text=f"med {median_val:.1f}",
                  annotation_position="top left",
                  annotation_font_color=COLOR_AXIS)
    title = f"{player_label} — {metric} distribution" if player_label else f"{metric} distribution"
    fig.update_layout(title=title, height=300,
                      xaxis_title=metric, yaxis_title="games")
    return fig


# ── 12. Last N games bar chart ─────────────────────────────────────────────
def last_n_games_bar_chart(
    games: pd.DataFrame,
    metric: str,
    *,
    n: int = 10,
    line_value: float | None = None,
    color: str = COLOR_ACCENT,
) -> go.Figure:
    """Bar chart of a metric for the last N games, with optional reference line."""
    if games is None or games.empty or metric not in games.columns:
        return _empty_figure("No game data")
    date_col = "game_date" if "game_date" in games.columns else (
        "GAME_DATE" if "GAME_DATE" in games.columns else None
    )
    if date_col is None:
        return _empty_figure("No game_date column")
    keep = [date_col, metric] + (["MATCHUP"] if "MATCHUP" in games.columns else [])
    df = games[keep].dropna(subset=[metric]).copy()
    df = df.sort_values(date_col).tail(n)
    if df.empty:
        return _empty_figure("No recent games")
    avg = float(df[metric].mean())
    bar_colors = [color if v >= avg else _hex_to_rgba(color, 0.55) for v in df[metric]]
    matchups = df["MATCHUP"].fillna("").tolist() if "MATCHUP" in df.columns else [""] * len(df)
    fig = go.Figure(go.Bar(
        x=df[date_col], y=df[metric],
        marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        customdata=list(zip(matchups)),
        hovertemplate=("<b>%{x|%b %d, %Y}</b><br>%{customdata[0]}<br>"
                       f"{metric}: %{{y:.1f}}<extra></extra>"),
    ))
    fig.add_hline(y=avg, line=dict(color=COLOR_AXIS, dash="dash", width=1),
                  annotation_text=f"avg {avg:.1f}",
                  annotation_position="top right",
                  annotation_font_color=COLOR_AXIS)
    if line_value is not None:
        fig.add_hline(y=float(line_value), line=dict(color="#f0e442", dash="dot", width=1.4),
                      annotation_text=f"line {line_value:g}",
                      annotation_position="top left",
                      annotation_font_color="#f0e442")
    fig.update_layout(
        title=f"Last {len(df)} games — {metric}", height=320,
        xaxis_title=None, yaxis_title=metric, bargap=0.25,
    )
    return fig


# ── 13. Rolling metric comparison across players ───────────────────────────
def rolling_metric_compare(
    games_by_player: dict,
    metric: str,
    *,
    window: int = 5,
) -> go.Figure:
    """Per-player rolling mean of a metric over time."""
    if not games_by_player:
        return _empty_figure("Select players to compare")
    cmap = player_color_map(list(games_by_player.keys()))
    fig = go.Figure()
    plotted = 0
    for name, gdf in games_by_player.items():
        if gdf is None or gdf.empty or metric not in gdf.columns:
            continue
        date_col = "game_date" if "game_date" in gdf.columns else "GAME_DATE"
        if date_col not in gdf.columns:
            continue
        df = gdf[[date_col, metric]].dropna().sort_values(date_col)
        if df.empty:
            continue
        df["rolling"] = df[metric].rolling(window, min_periods=max(2, window // 2)).mean()
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df["rolling"], mode="lines",
            name=name, line=dict(width=2.5, color=cmap[name], shape="spline", smoothing=0.5),
            hovertemplate=(f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>"
                           f"{window}g {metric}: %{{y:.1f}}<extra></extra>"),
        ))
        plotted += 1
    if plotted == 0:
        return _empty_figure("No data for this metric")
    fig.update_layout(
        title=f"{window}-game rolling {metric}", height=380,
        xaxis_title=None, yaxis_title=metric,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    return fig


# ── 14. Signal direction donut ─────────────────────────────────────────────
def signal_direction_donut(df: pd.DataFrame) -> go.Figure:
    """Donut showing the split of Above-line vs Below-line model signals."""
    if df is None or df.empty or "side" not in df.columns:
        return _empty_figure("No signals to display")
    above = int((df["side"] == "MORE").sum())
    below = int((df["side"] == "LESS").sum())
    if above + below == 0:
        return _empty_figure("No signals to display")
    fig = go.Figure(go.Pie(
        labels=["Above line", "Below line"],
        values=[above, below],
        hole=0.62,
        marker=dict(colors=[COLOR_MORE, COLOR_LESS],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} signals (%{percent})<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        title="Signal direction", height=300,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# ── 15. Model signal count chart ───────────────────────────────────────────
def model_signal_count_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar of Above/Below counts grouped by model."""
    if df is None or df.empty or "model" not in df.columns or "side" not in df.columns:
        return _empty_figure("No signals to display")
    grp = (
        df.assign(direction=np.where(df["side"] == "MORE", "Above line", "Below line"))
          .groupby(["model", "direction"]).size().unstack(fill_value=0)
    )
    for col in ("Above line", "Below line"):
        if col not in grp.columns:
            grp[col] = 0
    grp = grp[["Above line", "Below line"]].sort_values("Above line", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=grp.index, x=grp["Above line"], name="Above line", orientation="h",
        marker=dict(color=COLOR_MORE),
        hovertemplate="<b>%{y}</b><br>Above line: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=grp.index, x=grp["Below line"], name="Below line", orientation="h",
        marker=dict(color=COLOR_LESS),
        hovertemplate="<b>%{y}</b><br>Below line: %{x}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack", title="Signals by model",
        height=max(280, 32 * len(grp) + 100),
        xaxis_title="signals", yaxis_title=None,
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    return fig


# ── 16. Player summary bar (recent form leaderboard) ───────────────────────
def player_summary_bar(df: pd.DataFrame, metric: str) -> go.Figure:
    """Horizontal bar of a metric per player, sorted descending."""
    if df is None or df.empty or metric not in df.columns:
        return _empty_figure("No data to compare")
    s = df[metric].dropna().sort_values(ascending=True)
    if s.empty:
        return _empty_figure("No data to compare")
    cmap = player_color_map(list(s.index))
    colors = [cmap[p] for p in s.index]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{v:.1f}" for v in s.values], textposition="outside",
        hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Recent form leaderboard — {metric}",
        height=max(260, 30 * len(s) + 100),
        xaxis_title=metric, yaxis_title=None,
        margin=dict(l=180, r=40, t=50, b=40),
    )
    return fig


# ── 17. Scenario output bar ────────────────────────────────────────────────
def scenario_output_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of model scenario predictions; expects columns model, prediction."""
    if df is None or df.empty or not {"model", "prediction"}.issubset(df.columns):
        return _empty_figure("No scenario predictions")
    d = df.sort_values("prediction", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["prediction"], y=d["model"], orientation="h",
        marker=dict(color=d["prediction"],
                    colorscale=[[0, "#0e1117"], [1, COLOR_ACCENT]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{v:.1f}" for v in d["prediction"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>predicted: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Scenario predictions by model",
        height=max(260, 32 * len(d) + 100),
        xaxis_title="predicted value", yaxis_title=None,
        margin=dict(l=160, r=40, t=50, b=40),
    )
    return fig


# ── 18. Model metric ranking ───────────────────────────────────────────────
def model_metric_ranking_chart(metrics_df: pd.DataFrame, *, metric: str = "R²") -> go.Figure:
    """Sorted horizontal bar of any model-evaluation metric across models."""
    if (metrics_df is None or metrics_df.empty
            or metric not in metrics_df.columns or "model" not in metrics_df.columns):
        return _empty_figure(f"No {metric} available")
    d = metrics_df[["model", metric]].dropna().copy()
    if d.empty:
        return _empty_figure(f"No {metric} available")
    higher_is_better = metric in ("R²",)
    d = d.sort_values(metric, ascending=higher_is_better)
    palette_lo, palette_hi = (COLOR_LESS, COLOR_MORE) if higher_is_better else (COLOR_MORE, COLOR_LESS)
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["model"], orientation="h",
        marker=dict(color=d[metric],
                    colorscale=[[0, palette_lo], [1, palette_hi]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{v:.3f}" for v in d[metric]], textposition="outside",
        hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Model ranking — {metric}",
        height=max(260, 30 * len(d) + 100),
        xaxis_title=metric, yaxis_title=None,
        margin=dict(l=160, r=60, t=50, b=40),
    )
    return fig


# ── 19. Residual distribution by model ─────────────────────────────────────
def residual_distribution_chart(panels: list) -> go.Figure:
    """Overlaid residual histograms (actual - predicted) per model.

    Reuses the same ``panels`` shape produced for ``predicted_vs_actual_grid``.
    """
    if not panels:
        return _empty_figure("No residuals to display")
    fig = go.Figure()
    palette = list(PLAYER_PALETTE)
    plotted = 0
    for i, panel in enumerate(panels):
        pts = panel.get("points", [])
        if not pts:
            continue
        df = pd.DataFrame(pts)
        if not {"actual", "pred"}.issubset(df.columns):
            continue
        residuals = (pd.to_numeric(df["actual"], errors="coerce")
                     - pd.to_numeric(df["pred"], errors="coerce")).dropna()
        if residuals.empty:
            continue
        color = palette[i % len(palette)]
        fig.add_trace(go.Histogram(
            x=residuals, name=panel.get("metric", f"model {i+1}"), nbinsx=22,
            marker=dict(color=_hex_to_rgba(color, 0.45),
                        line=dict(color=color, width=1)),
            opacity=0.75,
            hovertemplate=(f"<b>{panel.get('metric','')}</b><br>"
                           "residual: %{x:.2f}<br>games: %{y}<extra></extra>"),
        ))
        plotted += 1
    if plotted == 0:
        return _empty_figure("No residuals to display")
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"))
    fig.update_layout(
        title="Residual distribution by model (actual − predicted)",
        barmode="overlay", height=380,
        xaxis_title="residual", yaxis_title="games",
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    return fig


# ── 20-24. Analytics Dashboard chart factories ─────────────────────────────
# These functions operate on a *normalized* signal frame produced by
# ``_prepare_signal_frame`` in app.py. Required cols: player, metric, line,
# projection, gap, abs_gap, direction. Optional: books, matchup, signal_strength.

_REQ_SIGNAL_COLS = ("player", "metric", "line", "projection", "gap", "abs_gap", "direction")


def _has_signal_cols(df, *extra: str) -> bool:
    if df is None or df.empty:
        return False
    needed = set(_REQ_SIGNAL_COLS) | set(extra)
    return needed.issubset(df.columns)


def projection_gap_bar_chart(df: pd.DataFrame, *, top_n: int = 15) -> go.Figure:
    """Horizontal bar of the largest absolute projection gaps (player · metric)."""
    if not _has_signal_cols(df):
        return _empty_figure("No projection-gap data")
    d = df.copy()
    d = d.sort_values("abs_gap", ascending=False).head(top_n)
    if d.empty:
        return _empty_figure("No projection-gap data")
    d = d.iloc[::-1]  # largest at top after barh
    labels = [f"{p}  ·  {m}" for p, m in zip(d["player"], d["metric"])]
    colors = [COLOR_MORE if g >= 0 else COLOR_LESS for g in d["gap"]]
    fig = go.Figure(go.Bar(
        x=d["gap"], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{g:+.2f}" for g in d["gap"]], textposition="outside",
        customdata=np.stack([d["line"], d["projection"], d["direction"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Direction: %{customdata[2]}<br>"
            "Current line: %{customdata[0]:.2f}<br>"
            "Model projection: %{customdata[1]:.2f}<br>"
            "Projection gap: %{x:+.2f}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.35)", width=1))
    fig.update_layout(
        title=f"Top {len(d)} projection gaps",
        height=max(300, 28 * len(d) + 100),
        xaxis_title="projection gap (model − line)",
        yaxis_title=None,
        margin=dict(l=200, r=60, t=60, b=40),
        bargap=0.25,
    )
    return fig


def projection_gap_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of projection gaps across the slate, with quartile markers."""
    if df is None or df.empty or "gap" not in df.columns:
        return _empty_figure("No projection-gap data")
    vals = pd.to_numeric(df["gap"], errors="coerce").dropna()
    if vals.empty:
        return _empty_figure("No projection-gap data")
    p25, p50, p75 = (float(vals.quantile(q)) for q in (0.25, 0.5, 0.75))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=24,
        marker=dict(color=_hex_to_rgba(COLOR_ACCENT, 0.55),
                    line=dict(color=COLOR_ACCENT, width=1)),
        hovertemplate="gap range: %{x}<br>signals: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"),
                  annotation_text="line", annotation_position="top")
    for q, label, color in ((p25, "Q1", COLOR_AXIS), (p50, "median", COLOR_ACCENT),
                            (p75, "Q3", COLOR_AXIS)):
        fig.add_vline(x=q, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=f"{label} {q:+.1f}",
                      annotation_position="top",
                      annotation_font_color=color)
    fig.update_layout(
        title="Projection gap distribution",
        height=300,
        xaxis_title="projection gap (model − line)",
        yaxis_title="signals",
    )
    return fig


def direction_split_chart(df: pd.DataFrame) -> go.Figure:
    """Donut of Above-line vs Below-line signal counts (uses normalized direction)."""
    if df is None or df.empty or "direction" not in df.columns:
        return _empty_figure("No signals to split")
    above = int((df["direction"] == "Above line").sum())
    below = int((df["direction"] == "Below line").sum())
    if above + below == 0:
        return _empty_figure("No signals to split")
    fig = go.Figure(go.Pie(
        labels=["Above line", "Below line"],
        values=[above, below],
        hole=0.62,
        marker=dict(colors=[COLOR_MORE, COLOR_LESS],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} signals (%{percent})<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        title="Direction split", height=300, showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def metric_signal_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average projection gap by player × metric."""
    if not _has_signal_cols(df):
        return _empty_figure("Not enough data for heatmap")
    pivot = df.pivot_table(
        index="player", columns="metric", values="gap", aggfunc="mean",
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return _empty_figure("Need at least two players and two metrics")
    # sort rows by total absolute signal
    pivot = pivot.reindex(pivot.abs().sum(axis=1).sort_values(ascending=True).index)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0, COLOR_LESS], [0.5, "#1a1d24"], [1, COLOR_MORE]],
        zmid=0,
        colorbar=dict(title="gap", thickness=10, len=0.7),
        hovertemplate="<b>%{y}</b><br>%{x}<br>avg gap: %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Projection gap heatmap — player × metric",
        height=max(320, 28 * pivot.shape[0] + 100),
        xaxis_title=None, yaxis_title=None,
        margin=dict(l=180, r=40, t=60, b=40),
    )
    return fig


def projection_gap_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter of current line vs model projection, colored by metric."""
    if not _has_signal_cols(df):
        return _empty_figure("No projection-gap data")
    d = df.dropna(subset=["line", "projection"]).copy()
    if d.empty:
        return _empty_figure("No projection-gap data")
    metrics = sorted(d["metric"].unique())
    palette = list(PLAYER_PALETTE)
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(metrics)}
    fig = go.Figure()
    for m in metrics:
        sub = d[d["metric"] == m]
        fig.add_trace(go.Scatter(
            x=sub["line"], y=sub["projection"], mode="markers",
            name=m,
            marker=dict(size=10, color=cmap[m],
                        line=dict(color="rgba(255,255,255,0.25)", width=0.6),
                        opacity=0.85),
            customdata=np.stack([sub["player"], sub["gap"], sub["direction"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · " + m + "<br>"
                "Line: %{x:.2f}<br>"
                "Projection: %{y:.2f}<br>"
                "Gap: %{customdata[1]:+.2f} (%{customdata[2]})"
                "<extra></extra>"
            ),
        ))
    lo = float(min(d["line"].min(), d["projection"].min()))
    hi = float(max(d["line"].max(), d["projection"].max()))
    pad = max(0.5, (hi - lo) * 0.05)
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode="lines", name="line = projection",
        line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title="Model projection vs current line",
        height=420,
        xaxis_title="current line", yaxis_title="model projection",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    return fig


def player_coverage_bar(df: pd.DataFrame, *, top_n: int = 15) -> go.Figure:
    """Bar chart of players ranked by number of available lines."""
    if df is None or df.empty or "player" not in df.columns:
        return _empty_figure("No coverage data")
    counts = df["player"].value_counts().head(top_n).iloc[::-1]
    if counts.empty:
        return _empty_figure("No coverage data")
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=COLOR_ACCENT, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{v}" for v in counts.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} lines<extra></extra>",
    ))
    fig.update_layout(
        title=f"Player coverage — top {len(counts)}",
        height=max(280, 26 * len(counts) + 100),
        xaxis_title="lines available", yaxis_title=None,
        margin=dict(l=180, r=40, t=50, b=40),
    )
    return fig


def metric_coverage_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of how many lines each metric/category has."""
    if df is None or df.empty or "metric" not in df.columns:
        return _empty_figure("No coverage data")
    counts = df["metric"].value_counts().iloc[::-1]
    if counts.empty:
        return _empty_figure("No coverage data")
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=_hex_to_rgba(COLOR_ACCENT, 0.7),
                    line=dict(color=COLOR_ACCENT, width=0.5)),
        text=[f"{v}" for v in counts.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} lines<extra></extra>",
    ))
    fig.update_layout(
        title="Metric coverage",
        height=max(260, 26 * len(counts) + 100),
        xaxis_title="lines available", yaxis_title=None,
        margin=dict(l=140, r=40, t=50, b=40),
    )
    return fig


def avg_gap_by_metric_bar(df: pd.DataFrame) -> go.Figure:
    """Average absolute projection gap per metric/category."""
    if df is None or df.empty or "metric" not in df.columns or "abs_gap" not in df.columns:
        return _empty_figure("No gap data by metric")
    grp = df.groupby("metric")["abs_gap"].mean().sort_values(ascending=True)
    if grp.empty:
        return _empty_figure("No gap data by metric")
    fig = go.Figure(go.Bar(
        x=grp.values, y=grp.index, orientation="h",
        marker=dict(color=grp.values,
                    colorscale=[[0, "#1a1d24"], [1, COLOR_ACCENT]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        text=[f"{v:.2f}" for v in grp.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>avg |gap|: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Average projection gap by metric",
        height=max(260, 26 * len(grp) + 100),
        xaxis_title="average |projection gap|", yaxis_title=None,
        margin=dict(l=140, r=40, t=50, b=40),
    )
    return fig


def signal_strength_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked-bar count of Above/Below signals per signal-strength bucket."""
    if (df is None or df.empty
            or "signal_strength" not in df.columns or "direction" not in df.columns):
        return _empty_figure("No signal-strength data")
    order = ["Low", "Medium", "High", "Extreme"]
    grp = (df.groupby(["signal_strength", "direction"]).size()
             .unstack(fill_value=0))
    grp = grp.reindex(index=[b for b in order if b in grp.index])
    for col in ("Above line", "Below line"):
        if col not in grp.columns:
            grp[col] = 0
    if grp.empty:
        return _empty_figure("No signal-strength data")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp.index, y=grp["Above line"], name="Above line",
        marker=dict(color=COLOR_MORE),
        hovertemplate="<b>%{x}</b><br>Above line: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=grp.index, y=grp["Below line"], name="Below line",
        marker=dict(color=COLOR_LESS),
        hovertemplate="<b>%{x}</b><br>Below line: %{y}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        title="Signal strength distribution",
        height=300,
        xaxis_title=None, yaxis_title="signals",
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    return fig
