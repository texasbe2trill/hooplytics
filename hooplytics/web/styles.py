"""Visual system for the Hooplytics web app: Plotly template + CSS primitives."""

from __future__ import annotations

from typing import Dict, Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# Palette ─────────────────────────────────────────────────────────────────────
COURT_ORANGE = "#ff7a18"
COURT_ORANGE_SOFT = "#ffb47a"
COURT_ORANGE_DEEP = "#cc5a00"

INK = "#f5f5f5"
INK_MUTED = "#b2bdcc"
INK_QUIET = "#7c8da3"

BG_DEEP = "#0e1117"
BG_PANEL = "#161b25"
BG_PANEL_2 = "#1a1f2c"

POS = "#3ddc97"
NEG = "#ff6b6b"
WARN = "#f5b041"

GRID = "rgba(255,255,255,0.06)"
HAIRLINE = "rgba(255,255,255,0.08)"

PLAYER_PALETTE = (
    "#ff7a18", "#3ddc97", "#5cb8ff", "#c084fc",
    "#f5b041", "#ff6b6b", "#7ee787", "#facc15",
    "#22d3ee", "#fb7185", "#a78bfa", "#fbbf24",
)

# Aliases consumed by charts.py
COLOR_ACCENT = COURT_ORANGE
COLOR_AXIS = INK_QUIET
COLOR_GRID = GRID
COLOR_MORE = POS
COLOR_LESS = NEG


def player_color_map(players: Iterable[str]) -> Dict[str, str]:
    return {p: PLAYER_PALETTE[i % len(PLAYER_PALETTE)] for i, p in enumerate(players)}


# Plotly template ─────────────────────────────────────────────────────────────
def register_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
                  color=INK, size=13),
        title=dict(font=dict(color=INK, size=17), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(PLAYER_PALETTE),
        margin=dict(l=48, r=24, t=56, b=44),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=HAIRLINE,
                   tickcolor=HAIRLINE, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=HAIRLINE,
                   tickcolor=HAIRLINE, tickfont=dict(color=INK_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=HAIRLINE, borderwidth=1,
                    font=dict(color=INK_MUTED)),
        hoverlabel=dict(bgcolor=BG_PANEL_2, bordercolor=COURT_ORANGE,
                        font=dict(color=INK)),
    )
    pio.templates["hooplytics_dark"] = template
    pio.templates.default = "hooplytics_dark"


# CSS ─────────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
:root {{
  --hl-orange: {COURT_ORANGE};
  --hl-orange-soft: {COURT_ORANGE_SOFT};
  --hl-orange-deep: {COURT_ORANGE_DEEP};
  --hl-ink: {INK};
  --hl-ink-muted: {INK_MUTED};
  --hl-ink-quiet: {INK_QUIET};
  --hl-bg: {BG_DEEP};
  --hl-panel: {BG_PANEL};
  --hl-panel-2: {BG_PANEL_2};
  --hl-pos: {POS};
  --hl-neg: {NEG};
  --hl-hairline: {HAIRLINE};
}}

html, body, [class*="stApp"] {{
  background: radial-gradient(1200px 600px at 10% -10%, rgba(255,122,24,0.06), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(92,184,255,0.04), transparent 55%),
              var(--hl-bg);
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #1a1f2c 0%, #11151c 100%);
  border-right: 1px solid var(--hl-hairline);
}}
[data-testid="stSidebar"] .stRadio label {{
  font-weight: 500;
  color: var(--hl-ink-muted);
}}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] + div {{
  color: var(--hl-ink);
}}

/* Sidebar roster rows: keep player name vertically centered with the X button. */
[data-testid="stSidebar"] .hl-roster-row {{
  color: var(--hl-ink);
  font-size: 0.92rem;
  padding: 0.15rem 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

/* Headings */
h1, h2, h3, h4, h5 {{
  letter-spacing: -0.01em;
  color: var(--hl-ink);
}}

/* Section header */
.hl-section {{
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--hl-orange);
  margin: 1.4rem 0 0.4rem 0;
}}
.hl-section-sub {{
  color: var(--hl-ink-muted);
  font-size: 0.92rem;
  margin: 0 0 1rem 0;
}}

/* Subtle text */
.hl-subtle {{ color: var(--hl-ink-quiet); font-size: 0.86rem; }}
.hl-muted {{ color: var(--hl-ink-muted); }}
.hl-mono {{ font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }}

/* Hairline divider */
.hl-divider {{
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--hl-hairline) 20%, var(--hl-hairline) 80%, transparent);
  margin: 1.4rem 0;
}}

/* Brand row (Home only) */
.hl-brand-row {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.25rem 0 0.25rem 0;
}}
.hl-brand {{
  display: flex; align-items: center; gap: 0.7rem;
  font-weight: 800; font-size: 1.05rem; letter-spacing: 0.22em;
  color: var(--hl-ink);
}}
.hl-brand-mark {{
  width: 22px; height: 22px; border-radius: 6px;
  background: linear-gradient(135deg, var(--hl-orange) 0%, var(--hl-orange-deep) 100%);
  box-shadow: 0 0 0 1px rgba(255,122,24,0.35), 0 6px 16px rgba(255,122,24,0.18);
}}
.hl-status {{
  display: flex; align-items: center; gap: 0.45rem;
  font-size: 0.78rem; color: var(--hl-ink-muted); letter-spacing: 0.08em;
}}
.hl-status-dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--hl-pos);
  box-shadow: 0 0 0 3px rgba(61,220,151,0.18);
}}

/* Hero */
.hl-hero-wrap {{ margin: 1.6rem 0 0.6rem 0; }}
.hl-tagline {{
  font-size: clamp(2rem, 4.2vw, 3.1rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
  background: linear-gradient(120deg, #ffffff 0%, #ffd9b8 55%, var(--hl-orange) 100%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent;
  margin: 0;
}}
.hl-tagline-sub {{
  margin-top: 0.55rem;
  color: var(--hl-ink-muted);
  font-size: 1.02rem;
  max-width: 56ch;
}}

/* Generic page hero (non-Home) */
.hl-page-hero {{
  padding: 0.4rem 0 0.6rem 0;
  border-bottom: 1px solid var(--hl-hairline);
  margin-bottom: 1.4rem;
}}
.hl-page-hero h1 {{
  margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em;
}}
.hl-page-hero p {{
  margin: 0.35rem 0 0 0; color: var(--hl-ink-muted); font-size: 0.96rem;
}}

/* KPI metric tiles (st.metric) */
[data-testid="stMetric"] {{
  background: linear-gradient(160deg, rgba(255,122,24,0.07), rgba(255,255,255,0.015));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  transition: transform 120ms ease, border-color 120ms ease;
}}
[data-testid="stMetric"]:hover {{
  border-color: rgba(255,122,24,0.35);
  transform: translateY(-1px);
}}
[data-testid="stMetricLabel"] {{
  color: var(--hl-ink-quiet);
  font-size: 0.74rem; letter-spacing: 0.16em; text-transform: uppercase;
  font-weight: 600;
}}
[data-testid="stMetricValue"] {{
  color: var(--hl-ink); font-weight: 700;
  font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
}}

/* Generic card */
.hl-card {{
  background: linear-gradient(160deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
}}
.hl-card-title {{
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--hl-orange);
  margin: 0 0 0.85rem 0;
}}

/* Edge row (top edges on Home) */
.hl-edge {{
  display: grid;
  grid-template-columns: 28px 1fr auto;
  gap: 0.9rem;
  align-items: center;
  padding: 0.7rem 0;
  border-top: 1px solid var(--hl-hairline);
}}
.hl-edge:first-child {{ border-top: none; }}
.hl-edge-rank {{
  width: 26px; height: 26px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.78rem; color: var(--hl-ink);
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--hl-hairline);
}}
.hl-edge-rank.top {{
  background: linear-gradient(135deg, var(--hl-orange) 0%, var(--hl-orange-deep) 100%);
  border-color: transparent;
  box-shadow: 0 4px 12px rgba(255,122,24,0.25);
}}
.hl-edge-main {{ display: flex; flex-direction: column; gap: 0.2rem; min-width: 0; }}
.hl-edge-player {{
  font-weight: 600; color: var(--hl-ink); font-size: 0.98rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.hl-edge-meta {{
  font-size: 0.78rem; color: var(--hl-ink-quiet);
  letter-spacing: 0.04em;
}}
.hl-edge-meta strong {{ color: var(--hl-ink-muted); font-weight: 600; }}
.hl-edge-num {{
  text-align: right; font-feature-settings: "tnum";
  font-variant-numeric: tabular-nums;
}}
.hl-edge-edge {{
  font-size: 1.05rem; font-weight: 700;
}}
.hl-edge-side {{
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase;
}}
.hl-edge.more .hl-edge-edge,
.hl-edge.more .hl-edge-side {{ color: var(--hl-pos); }}
.hl-edge.less .hl-edge-edge,
.hl-edge.less .hl-edge-side {{ color: var(--hl-neg); }}

/* R² bar list (Home model quality) */
.hl-r2-row {{
  display: grid;
  grid-template-columns: 110px 1fr 56px;
  gap: 0.85rem;
  align-items: center;
  padding: 0.4rem 0;
}}
.hl-r2-name {{ color: var(--hl-ink-muted); font-size: 0.88rem; }}
.hl-r2-track {{
  height: 6px; border-radius: 999px;
  background: rgba(255,255,255,0.05);
  overflow: hidden;
}}
.hl-r2-fill {{
  height: 100%;
  background: linear-gradient(90deg, var(--hl-orange-deep), var(--hl-orange) 60%, var(--hl-orange-soft));
  border-radius: 999px;
}}
.hl-r2-fill.weak {{
  background: linear-gradient(90deg, #6b3a3a, var(--hl-neg));
}}
.hl-r2-val {{
  text-align: right; color: var(--hl-ink); font-weight: 600;
  font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
  font-size: 0.88rem;
}}

/* Roster cards */
.hl-roster-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.7rem;
}}
.hl-roster-card {{
  background: linear-gradient(160deg, rgba(255,255,255,0.025), rgba(255,255,255,0));
  border: 1px solid var(--hl-hairline);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  display: flex; flex-direction: column; gap: 0.25rem;
  transition: border-color 120ms ease;
}}
.hl-roster-card:hover {{ border-color: rgba(255,122,24,0.35); }}
.hl-roster-name {{ color: var(--hl-ink); font-weight: 600; font-size: 0.96rem; }}
.hl-roster-meta {{ color: var(--hl-ink-quiet); font-size: 0.78rem; letter-spacing: 0.03em; }}
.hl-roster-dot {{
  width: 6px; height: 6px; border-radius: 50%; display: inline-block;
  margin-right: 0.4rem; vertical-align: middle;
}}

/* Pills */
.hl-pill {{
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  border: 1px solid transparent;
}}
.hl-pill-more {{ background: rgba(61,220,151,0.12); color: var(--hl-pos); border-color: rgba(61,220,151,0.25); }}
.hl-pill-less {{ background: rgba(255,107,107,0.12); color: var(--hl-neg); border-color: rgba(255,107,107,0.25); }}
.hl-pill-live {{ background: rgba(92,184,255,0.12); color: #5cb8ff; border-color: rgba(92,184,255,0.25); }}
.hl-pill-warn {{ background: rgba(245,176,65,0.14); color: var(--hl-warn); border-color: rgba(245,176,65,0.28); }}
.hl-pill-quiet {{ background: rgba(255,255,255,0.04); color: var(--hl-ink-muted); border-color: var(--hl-hairline); }}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px;
  border: 1px solid var(--hl-hairline);
  background: rgba(255,255,255,0.02);
  color: var(--hl-ink);
  font-weight: 500;
  transition: border-color 120ms ease, background 120ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: rgba(255,122,24,0.45);
  background: rgba(255,122,24,0.06);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 0.25rem; border-bottom: 1px solid var(--hl-hairline);
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent; color: var(--hl-ink-quiet);
  font-weight: 500; padding: 0.5rem 0.9rem;
}}
.stTabs [aria-selected="true"] {{
  color: var(--hl-ink) !important;
  border-bottom: 2px solid var(--hl-orange) !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--hl-hairline);
  border-radius: 12px;
  overflow: hidden;
}}

/* Header bar tweak */
header[data-testid="stHeader"] {{ background: transparent; }}

/* Suppress the framework's "streamlitApp" tooltip from leaking through the
   main content iframe \u2014 strip the title attribute via pointer-events so
   browsers don't show the native tooltip when users hover empty page areas. */
iframe[title="streamlitApp"] {{ pointer-events: auto; }}
iframe[title="streamlitApp"]::before {{ content: ""; }}

/* Tighten the default Streamlit block container so cards align with hero edges. */
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; }}

/* Reduce visual weight of Streamlit's auto-generated heading anchor links. */
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {{ display: none; }}

/* Insight + chart cards */
.hl-insight-card {{
  background: linear-gradient(160deg, rgba(255,122,24,0.05), rgba(255,255,255,0.005));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 1rem 1.15rem;
  display: flex; gap: 0.85rem; align-items: flex-start;
  margin-bottom: 1rem;
}}
.hl-insight-icon {{
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, var(--hl-orange) 0%, var(--hl-orange-deep) 100%);
  display: flex; align-items: center; justify-content: center;
  color: white; font-weight: 700; font-size: 0.78rem;
  letter-spacing: 0.06em; flex-shrink: 0;
  box-shadow: 0 4px 10px rgba(255,122,24,0.25);
}}
.hl-insight-body {{ display: flex; flex-direction: column; gap: 0.2rem; }}
.hl-insight-title {{
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--hl-orange);
}}
.hl-insight-text {{ color: var(--hl-ink-muted); font-size: 0.92rem; line-height: 1.5; }}

.hl-chart-card {{
  background: linear-gradient(160deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 0.9rem 1rem 0.4rem 1rem;
  margin-bottom: 0.9rem;
}}

/* Mini KPI grid (inside cards) */
.hl-kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.6rem;
  margin: 0.4rem 0 0.2rem 0;
}}
.hl-mini-kpi {{
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--hl-hairline);
  border-radius: 10px;
  padding: 0.65rem 0.8rem;
  display: flex; flex-direction: column; gap: 0.1rem;
}}
.hl-mini-kpi-label {{
  font-size: 0.65rem; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--hl-ink-quiet);
}}
.hl-mini-kpi-value {{
  font-size: 1.15rem; font-weight: 700; color: var(--hl-ink);
  font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
}}
.hl-mini-kpi-sub {{
  font-size: 0.72rem; color: var(--hl-ink-quiet);
}}
.hl-mini-kpi-sub.up {{ color: var(--hl-pos); }}
.hl-mini-kpi-sub.down {{ color: var(--hl-neg); }}

/* Chips */
.hl-chip {{
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em;
  background: rgba(255,255,255,0.04);
  color: var(--hl-ink-muted);
  border: 1px solid var(--hl-hairline);
}}
.hl-chip.accent {{
  background: rgba(255,122,24,0.12);
  color: var(--hl-orange);
  border-color: rgba(255,122,24,0.28);
}}
.hl-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.6rem 0; }}

/* Disclaimer */
.hl-disclaimer {{
  background: linear-gradient(160deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
  border: 1px dashed var(--hl-hairline);
  border-radius: 12px;
  padding: 0.85rem 1.1rem;
  color: var(--hl-ink-quiet);
  font-size: 0.82rem;
  line-height: 1.55;
  margin: 1.2rem 0 0.6rem 0;
}}
.hl-disclaimer strong {{ color: var(--hl-ink-muted); font-weight: 600; }}

/* Empty state */
.hl-empty-state {{
  background: linear-gradient(160deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
  border: 1px dashed var(--hl-hairline);
  border-radius: 14px;
  padding: 2rem 1.5rem;
  text-align: center;
  color: var(--hl-ink-quiet);
}}
.hl-empty-title {{
  color: var(--hl-ink); font-weight: 600; font-size: 1rem;
  margin: 0 0 0.4rem 0;
}}
.hl-empty-body {{ font-size: 0.88rem; max-width: 48ch; margin: 0 auto; }}

/* ── Analytics Dashboard primitives ──────────────────────────────────────── */
.hl-dashboard-hero {{
  position: relative;
  background:
    radial-gradient(1200px 360px at 12% -20%, rgba(255,122,24,0.18), transparent 60%),
    linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--hl-hairline);
  border-radius: 18px;
  padding: 1.6rem 1.8rem;
  margin-bottom: 0.85rem;
  overflow: hidden;
}}
.hl-dashboard-hero::after {{
  content: "";
  position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(180deg, transparent 70%, rgba(0,0,0,0.18));
}}
.hl-hero-eyebrow {{
  display: inline-block;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--hl-orange);
  background: rgba(255,122,24,0.10);
  border: 1px solid rgba(255,122,24,0.28);
  border-radius: 999px;
  padding: 0.22rem 0.7rem;
  margin-bottom: 0.7rem;
}}
.hl-hero-title {{
  font-size: 2rem; font-weight: 700; letter-spacing: -0.01em;
  margin: 0 0 0.35rem 0; color: var(--hl-ink);
  line-height: 1.15;
}}
.hl-hero-subtitle {{
  font-size: 1rem; color: var(--hl-ink-muted);
  margin: 0 0 0.95rem 0;
}}
.hl-hero-copy {{
  font-size: 0.92rem; color: var(--hl-ink-quiet);
  line-height: 1.65;
}}
.hl-hero-copy strong {{ color: var(--hl-ink); font-weight: 600; }}

/* KPI strip */
.hl-kpi-strip {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  margin: 0 0 1.6rem 0;
}}
.hl-kpi-card {{
  background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  display: flex; flex-direction: column; gap: 0.3rem;
  min-width: 0;
  transition: border-color 0.18s ease, transform 0.18s ease;
}}
.hl-kpi-card:hover {{
  border-color: rgba(255,122,24,0.35);
  transform: translateY(-1px);
}}
.hl-kpi-label {{
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--hl-ink-quiet);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.hl-kpi-value {{
  font-size: clamp(1.25rem, 0.9vw + 1rem, 1.7rem);
  font-weight: 700; color: var(--hl-ink);
  line-height: 1.1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
}}
.hl-kpi-caption {{
  font-size: 0.78rem; color: var(--hl-ink-quiet); line-height: 1.4;
}}

/* Meta row (status + timestamp above hero strips) */
.hl-meta-row {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 0.85rem;
  font-size: 0.78rem; letter-spacing: 0.08em;
  color: var(--hl-ink-muted);
  margin: 0 0 0.85rem 0;
}}
.hl-meta-row .hl-meta-sep {{
  width: 1px; height: 14px; background: var(--hl-hairline);
}}

/* Section headers */
.hl-section-header {{
  display: flex; flex-direction: column; gap: 0.25rem;
  margin: 1.8rem 0 1rem 0;
  padding-bottom: 0.65rem;
  border-bottom: 1px solid var(--hl-hairline);
}}
.hl-section-eyebrow {{
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--hl-orange);
}}
.hl-section-title {{
  font-size: 1.25rem; font-weight: 700; color: var(--hl-ink);
  letter-spacing: -0.005em; margin: 0;
}}
.hl-section-copy {{
  font-size: 0.9rem; color: var(--hl-ink-muted); line-height: 1.55;
  max-width: 78ch; margin: 0.15rem 0 0 0;
}}

/* Signal grid + cards */
.hl-signal-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 0.85rem;
  margin: 0.6rem 0 0.4rem 0;
}}
.hl-signal-card {{
  position: relative;
  background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.005));
  border: 1px solid var(--hl-hairline);
  border-left: 3px solid var(--hl-orange);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  display: flex; flex-direction: column; gap: 0.55rem;
  transition: border-color 0.18s ease, transform 0.18s ease;
}}
.hl-signal-card.above {{ border-left-color: var(--hl-pos); }}
.hl-signal-card.below {{ border-left-color: var(--hl-neg); }}
.hl-signal-card:hover {{ transform: translateY(-1px); }}
.hl-signal-top {{
  display: flex; justify-content: space-between; align-items: flex-start; gap: 0.6rem;
}}
.hl-signal-player {{
  font-size: 1.02rem; font-weight: 700; color: var(--hl-ink); line-height: 1.2;
}}
.hl-signal-meta {{
  font-size: 0.74rem; color: var(--hl-ink-quiet);
  margin-top: 0.18rem; letter-spacing: 0.02em;
}}
.hl-signal-direction {{
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.22rem 0.6rem; border-radius: 999px;
  white-space: nowrap;
}}
.hl-signal-direction.above {{
  background: rgba(61,220,151,0.14); color: var(--hl-pos);
  border: 1px solid rgba(61,220,151,0.35);
}}
.hl-signal-direction.below {{
  background: rgba(255,107,107,0.14); color: var(--hl-neg);
  border: 1px solid rgba(255,107,107,0.35);
}}
.hl-signal-value-row {{
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0.4rem; margin-top: 0.1rem;
}}
.hl-signal-cell {{
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--hl-hairline);
  border-radius: 8px;
  padding: 0.45rem 0.55rem;
  display: flex; flex-direction: column; gap: 0.1rem;
}}
.hl-signal-cell-label {{
  font-size: 0.6rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--hl-ink-quiet);
}}
.hl-signal-cell-value {{
  font-size: 0.98rem; font-weight: 700; color: var(--hl-ink);
  font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
}}
.hl-signal-cell-value.gap-pos {{ color: var(--hl-pos); }}
.hl-signal-cell-value.gap-neg {{ color: var(--hl-neg); }}
.hl-signal-insight {{
  font-size: 0.84rem; color: var(--hl-ink-muted); line-height: 1.5;
  border-top: 1px solid var(--hl-hairline);
  padding-top: 0.55rem;
}}

/* Chart shell */
.hl-chart-shell {{
  background: linear-gradient(160deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 0.5rem 0.6rem 0.2rem 0.6rem;
  margin-bottom: 0.85rem;
}}

/* Strength chips */
.hl-chip-low {{
  background: rgba(255,255,255,0.05); color: var(--hl-ink-quiet);
  border: 1px solid var(--hl-hairline);
  padding: 0.16rem 0.55rem; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.hl-chip-medium {{
  background: rgba(245,176,65,0.12); color: var(--hl-warn);
  border: 1px solid rgba(245,176,65,0.35);
  padding: 0.16rem 0.55rem; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.hl-chip-high {{
  background: rgba(255,122,24,0.14); color: var(--hl-orange);
  border: 1px solid rgba(255,122,24,0.40);
  padding: 0.16rem 0.55rem; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.hl-chip-extreme {{
  background: linear-gradient(135deg, rgba(255,122,24,0.30), rgba(255,107,107,0.25));
  color: white;
  border: 1px solid rgba(255,107,107,0.55);
  padding: 0.16rem 0.55rem; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
}}

/* Disclaimer card variant */
.hl-disclaimer-card {{
  background: linear-gradient(160deg, rgba(255,255,255,0.025), rgba(255,255,255,0));
  border: 1px solid var(--hl-hairline);
  border-radius: 12px;
  padding: 0.95rem 1.15rem;
  color: var(--hl-ink-quiet);
  font-size: 0.84rem;
  line-height: 1.55;
  margin: 1.4rem 0 0.4rem 0;
}}
.hl-disclaimer-card strong {{ color: var(--hl-ink-muted); font-weight: 600; }}

/* ── Player Line Lab ────────────────────────────────────────────────────── */
.hl-lab-hero {{
  position: relative;
  background:
    radial-gradient(circle at 12% 0%, rgba(255,122,24,0.18), transparent 55%),
    radial-gradient(circle at 92% 110%, rgba(61,220,151,0.12), transparent 60%),
    linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--hl-hairline);
  border-radius: 18px;
  padding: 1.6rem 1.8rem;
  margin: 0.4rem 0 1.2rem 0;
  overflow: hidden;
}}
.hl-lab-hero h1 {{
  font-size: 1.65rem; font-weight: 700; margin: 0 0 0.35rem 0;
  letter-spacing: -0.01em;
}}
.hl-lab-hero p {{ color: var(--hl-ink-muted); margin: 0; font-size: 0.93rem; max-width: 760px; line-height: 1.55; }}

.hl-lab-controls {{
  background: rgba(255,255,255,0.018);
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 1.0rem 1.15rem 0.6rem;
  margin-bottom: 1.0rem;
}}
.hl-lab-controls .hl-controls-eyebrow {{
  font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--hl-orange); font-weight: 700; margin: 0 0 0.5rem 0;
}}

.hl-player-snapshot {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.7rem;
  margin: 0.6rem 0 1.0rem 0;
}}
.hl-snap-card {{
  background: rgba(255,255,255,0.022);
  border: 1px solid var(--hl-hairline);
  border-radius: 12px;
  padding: 0.85rem 0.95rem;
  transition: border-color 160ms ease, transform 160ms ease;
}}
.hl-snap-card:hover {{ border-color: rgba(255,122,24,0.35); transform: translateY(-1px); }}
.hl-snap-label {{
  font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--hl-ink-quiet); font-weight: 700; margin-bottom: 0.3rem;
}}
.hl-snap-value {{
  font-size: 1.35rem; font-weight: 700; color: var(--hl-ink);
  font-variant-numeric: tabular-nums; line-height: 1.1;
}}
.hl-snap-caption {{ font-size: 0.74rem; color: var(--hl-ink-muted); margin-top: 0.2rem; }}
.hl-snap-trend-up    {{ color: var(--hl-pos); }}
.hl-snap-trend-down  {{ color: var(--hl-neg); }}
.hl-snap-trend-flat  {{ color: var(--hl-ink-muted); }}

.hl-line-context-card,
.hl-outcome-card,
.hl-sensitivity-card,
.hl-methodology-card {{
  background: rgba(255,255,255,0.022);
  border: 1px solid var(--hl-hairline);
  border-radius: 14px;
  padding: 1.0rem 1.2rem;
  margin: 0.4rem 0 1.0rem 0;
}}
.hl-line-context-card .label,
.hl-outcome-card .label {{
  font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--hl-ink-quiet); font-weight: 700;
}}
.hl-line-context-card .value,
.hl-outcome-card .value {{
  font-size: 1.25rem; font-weight: 700; color: var(--hl-ink);
  font-variant-numeric: tabular-nums;
}}

.hl-threshold-chip {{
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.2rem 0.55rem; border-radius: 999px;
  background: rgba(255,122,24,0.14);
  color: var(--hl-orange); font-weight: 700;
  font-size: 0.72rem; letter-spacing: 0.06em;
  border: 1px solid rgba(255,122,24,0.3);
}}
.hl-threshold-chip.above {{ background: rgba(61,220,151,0.14); color: var(--hl-pos); border-color: rgba(61,220,151,0.32); }}
.hl-threshold-chip.below {{ background: rgba(255,107,107,0.14); color: var(--hl-neg); border-color: rgba(255,107,107,0.32); }}

.hl-signal-summary {{
  background: linear-gradient(155deg, rgba(255,122,24,0.10), rgba(255,255,255,0.012));
  border: 1px solid rgba(255,122,24,0.28);
  border-radius: 14px;
  padding: 1.05rem 1.25rem;
  margin: 0.4rem 0 0.9rem 0;
}}
.hl-signal-summary .eyebrow {{
  font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--hl-orange); font-weight: 700; margin: 0 0 0.35rem 0;
}}
.hl-signal-summary p {{ margin: 0; color: var(--hl-ink); font-size: 0.95rem; line-height: 1.6; }}

.hl-analyst-note {{
  background: rgba(255,255,255,0.022);
  border-left: 3px solid var(--hl-orange);
  border-radius: 0 12px 12px 0;
  padding: 0.95rem 1.15rem;
  margin: 0.6rem 0 1.0rem 0;
}}
.hl-analyst-note .eyebrow {{
  font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--hl-ink-quiet); font-weight: 700; margin: 0 0 0.3rem 0;
}}
.hl-analyst-note p {{ margin: 0; color: var(--hl-ink); font-size: 0.92rem; line-height: 1.6; }}

.hl-warning-soft {{
  background: rgba(245,176,65,0.08);
  border: 1px solid rgba(245,176,65,0.32);
  color: #f5b041;
  border-radius: 12px;
  padding: 0.7rem 1.0rem;
  font-size: 0.85rem;
  margin: 0.5rem 0;
}}
.hl-methodology-card p {{
  margin: 0; color: var(--hl-ink-muted); font-size: 0.83rem; line-height: 1.6;
}}

</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# Render helpers ──────────────────────────────────────────────────────────────
def page_hero(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="hl-page-hero"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str, *, icon: str = "i") -> None:
    """Render an explanatory header card above charts/sections."""
    st.markdown(
        f'<div class="hl-insight-card">'
        f'<div class="hl-insight-icon">{icon}</div>'
        f'<div class="hl-insight-body">'
        f'<div class="hl-insight-title">{title}</div>'
        f'<div class="hl-insight-text">{body}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="hl-disclaimer">{text}</div>',
                unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="hl-empty-state">'
        f'<p class="hl-empty-title">{title}</p>'
        f'<p class="hl-empty-body">{body}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def mini_kpi(label: str, value: str, sub: str | None = None,
             trend: str | None = None) -> str:
    """Return HTML for one mini-KPI tile (use inside hl-kpi-grid)."""
    sub_cls = f" {trend}" if trend in ("up", "down") else ""
    sub_html = f'<div class="hl-mini-kpi-sub{sub_cls}">{sub}</div>' if sub else ""
    return (
        f'<div class="hl-mini-kpi">'
        f'<div class="hl-mini-kpi-label">{label}</div>'
        f'<div class="hl-mini-kpi-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def kpi_grid(tiles_html: list[str]) -> None:
    st.markdown(f'<div class="hl-kpi-grid">{"".join(tiles_html)}</div>',
                unsafe_allow_html=True)


def meta_row(items: list[str]) -> None:
    """Render a thin meta row of small labels separated by hairlines."""
    if not items:
        return
    sep = '<span class="hl-meta-sep"></span>'
    body = sep.join(f"<span>{item}</span>" for item in items)
    st.markdown(f'<div class="hl-meta-row">{body}</div>',
                unsafe_allow_html=True)


def chip(text: str, *, accent: bool = False) -> str:
    cls = "hl-chip accent" if accent else "hl-chip"
    return f'<span class="{cls}">{text}</span>'


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<p class="hl-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<p class="hl-section">{title}</p>{sub}',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="hl-divider"></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "quiet") -> str:
    return f'<span class="hl-pill hl-pill-{kind}">{text}</span>'
