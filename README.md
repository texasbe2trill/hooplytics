<div align="center">

# 🏀 Hooplytics

### *An NBA player intelligence workbench.*

**Project the next game · Study any line · Pull live odds — straight from your terminal, notebook, or browser.**

<br>

[![CI](https://github.com/texasbe2trill/hooplytics/actions/workflows/ci.yml/badge.svg)](https://github.com/texasbe2trill/hooplytics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-FFD23F.svg?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Powered by The Odds API](https://img.shields.io/badge/lines-The%20Odds%20API-2ea44f.svg?style=flat-square)](https://the-odds-api.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/texasbe2trill/hooplytics/blob/main/hooplytics.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hooplytics.streamlit.app)

<br>

[**Quick Start**](#-tldr) ·
[**First 5 Minutes**](#-your-first-5-minutes-with-hooplytics) ·
[**Live Lines**](#-live-lines-made-analytical) ·
[**Dashboard**](#-app-preview) ·
[**CLI**](#-cli-walkthrough) ·
[**Use from Claude (MCP)**](#-use-from-claude-mcp) ·
[**Modeling**](#-analytics-approach) ·
[**Roadmap**](#-roadmap)

</div>

<br>

> **Hooplytics turns NBA game logs into a modern analytics workflow.** Model projections, recent-form trends, historical threshold studies, and live market context — designed to help you *explore, explain, and challenge* the numbers, not hand you a magic prediction.

<br>

<table>
<tr>
<td width="33%" valign="top">

### 🎛️ Streamlit dashboard
Click-driven analytics for the modern fan. Eight purpose-built pages, an AI scout (OpenAI **or** Claude), and printable PDF reports.

```bash
hooplytics-web
```

</td>
<td width="33%" valign="top">

### ⚡ Typer CLI
Rich-rendered tables, scriptable end to end, `--json` friendly.

```bash
hooplytics --help
```

</td>
<td width="33%" valign="top">

### 📓 Jupyter notebook
Reproducible, narrative-first exploration with charts inline.

```bash
jupyter lab hooplytics.ipynb
```

</td>
</tr>
</table>

> Lines from The Odds API are treated as **analytical thresholds** — comparison points for projections, distributions, and historical outcomes. **Hooplytics is analytics-first.**

---

## 🚀 TL;DR

Already know what you want? Here is the fast path.

```bash
git clone https://github.com/texasbe2trill/hooplytics.git && cd hooplytics
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

| I want to… | Command |
| :--- | :--- |
| 🎛️ &nbsp; Open the dashboard | `hooplytics-web` |
| 🎯 &nbsp; Project a player's next game | `hooplytics project "Jalen Brunson"` |
| 🔄 &nbsp; Detect a role / usage shift | `hooplytics role-shift "Jalen Brunson"` |
| 📈 &nbsp; Compare a projection vs. a live line | `hooplytics prop "Shai Gilgeous-Alexander" points` |
| 📊 &nbsp; See the live line board | `hooplytics lines --refresh` |
| 📑 &nbsp; Generate a printable scouting report | Open the dashboard → **Roster Report** → *Generate PDF* |
| 🏋️ &nbsp; Generate a coaching performance report | Open the dashboard → **Roster Report** → *Player Performance Analytics* |
| 🤖 &nbsp; Ask the AI scout a question | Open the dashboard → **Hooplytics Scout** |
| 📰 &nbsp; Read tonight's AI slate brief | Open the dashboard → **Home** (needs an AI key) |
| 🔍 &nbsp; Explain why a specific edge exists | Open the dashboard → **Analytics Dashboard** → *Edge Explainer* |
| ❓ &nbsp; See all CLI commands | `hooplytics --help` |
| 📓 &nbsp; Open the notebook | `jupyter lab hooplytics.ipynb` |

> 💡 **Tip:** set `ODDS_API_KEY` in `.env` and `prop` will auto-fetch the line for you — no `--line` needed.

---

## 🏀 Your First 5 Minutes With Hooplytics

New here? This is the smoothest path to **"oh, that's actually useful"** — and you don't even have to install anything.

<table>
<tr>
<td width="50px" align="center" valign="top"><h2>1️⃣</h2></td>
<td valign="top">
<strong>Open the live app — no install required</strong><br>
Go to <a href="https://hooplytics.streamlit.app/"><strong>hooplytics.streamlit.app</strong></a>. The hosted Streamlit app loads instantly with a pre-shipped roster of stars and a high-accuracy <strong>RACE</strong> model bundle already in memory — zero training, zero waiting. Multiple bundles ship in <code>bundles/</code> and you can switch between them in the sidebar (see <a href="#-choosing-a-model-bundle">Choosing a model bundle</a>).
<br><br>
<sub>Prefer to run it locally? <code>pip install -e . && hooplytics-web</code> opens the same app at <code>http://localhost:8501</code>.</sub>
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>2️⃣</h2></td>
<td valign="top">
<strong>Land on Home — see today's slate at a glance</strong><br>
You’ll see roster coverage, model-quality medians, and a roll-up of strong edges. This is your control tower.
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>3️⃣</h2></td>
<td valign="top">
<strong>(Optional) Paste your Odds API key in the sidebar</strong><br>
Get a free key at <a href="https://the-odds-api.com/">the-odds-api.com</a>, paste it into the sidebar password field, and the entire app lights up with live market lines and a real edge board. The key stays in session memory — never written to disk or sent anywhere except The Odds API.
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>4️⃣</h2></td>
<td valign="top">
<strong>Open <em>Analytics Dashboard</em> → read the slate</strong><br>
This is the fastest way to find tonight’s biggest projection-vs-line gaps. Sort by signed edge, filter by call (MORE / LESS), and use the <em>Strong</em> badge to surface the highest-conviction rows.
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>5️⃣</h2></td>
<td valign="top">
<strong>Drill into a player on <em>Player Projection</em></strong><br>
Pick a player from the sidebar to see their next-game projection across all 8 models, recent form trend, distribution context, and the model’s read versus the live line.
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>6️⃣</h2></td>
<td valign="top">
<strong>Generate a printable scouting PDF</strong><br>
Go to <em>Roster Report</em> → click <strong>Generate PDF</strong>. You get a branded, multi-page report with KPI tiles, a signal spotlight, R² lollipops, diverging edge charts, and per-player hero blocks — ready to share.
</td>
</tr>
<tr>
<td align="center" valign="top"><h2>7️⃣</h2></td>
<td valign="top">
<strong>(Optional) Talk to <em>Hooplytics Scout</em></strong><br>
Pick your AI provider in the sidebar (<strong>OpenAI</strong> or <strong>Anthropic Claude</strong>), paste a key, click <strong>Connect</strong>, and ask things like <em>“Give me a MORE/LESS read on the largest edge tonight, with confidence and risk factors.”</em> The Scout is grounded in your local data and structured for confidence + risk, not hot takes.
</td>
</tr>
</table>

> 🎯 **In a hurry?** The first three steps are the bare minimum. Steps 4–7 are where the real fun lives.

---

## ✨ What Makes Hooplytics Different

Most NBA tools either dump descriptive stats or hand you a single prediction with no context. **Hooplytics sits in the middle.**

<table>
<tr>
<td width="50%" valign="top">

#### 🎯 Projections, not promises
Eight model families, side-by-side, with held-out diagnostics in plain view.

#### 📡 Lines as analytical thresholds
The Odds API tells you what the market thinks; Hooplytics shows you how that compares to recent form, distributions, and history.

#### 🧭 Three surfaces, one mental model
Streamlit for visuals, CLI for speed, notebook for reproducibility.

</td>
<td width="50%" valign="top">

#### 🔍 No black boxes
Feature importance, residuals, calibration plots, and per-stat health summaries are first-class citizens.

#### 🔄 Role shift detection, built in
The **RoleShiftDetector** watches for sudden changes in assists, scoring, usage, and minutes across every player projection. When a signal fires, it surfaces a WARN or SUPPRESS flag — and the model respects it automatically.

#### 🛡️ Pregame-safe by design
Rolling features are computed from prior games only — no leakage, no surprises.

#### 🧪 Honest about noise
Lower-signal categories (steals + blocks, turnovers) are framed as such — always.

</td>
</tr>
</table>

---

## 📡 Live Lines, Made Analytical

Drop an `ODDS_API_KEY` in `.env` and the entire CLI lights up with live market data.

```bash
# Live line board for the tracked roster, sorted by projection gap
hooplytics lines --refresh

# Single-stat: projection vs. auto-fetched live line
hooplytics prop "Shai Gilgeous-Alexander" points

# Full 8-stat decision view with live lines folded in
hooplytics decisions "Victor Wembanyama"
```

**What you see in your terminal:**

| Column | Meaning |
| :--- | :--- |
| `model prediction` | The model's expected value for the next game |
| `posted line` | The market line pulled live from The Odds API |
| `5-game avg` | The player's recent-form baseline |
| `adj. threshold` | Vig-adjusted line used for the gap calculation |
| `edge` | Signed gap between the projection and the threshold |
| `call` | Directional signal (`MORE` / `LESS`) for analytical comparison |

> 🔓 **No key? No problem.** Every command still works — you just lose the live-line column.

---

## 🎬 See It In Motion

<div align="center">

![Streamlit walkthrough](docs/screenshots/streamlit-walkthrough.gif)

</div>

> If Hooplytics is useful to you, a ⭐ goes a long way — it helps others find the project.

---

## 🖼️ App Preview

The Streamlit app ships with **eight** purpose-built pages, each focused on a different analytics workflow.

<table>
<tr>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/home.png" alt="Home" />

#### 🏠 Home
Portfolio-style overview of roster coverage, model outputs, and high-level telemetry. With an AI key configured, an **AI Slate Brief** panel renders a one-paragraph daily read on the loudest mispricings — cached for the day so it costs one API call.

</td>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/analytics-dashboard.png" alt="Analytics Dashboard" />

#### 📊 Analytics Dashboard
Command center for projection gaps, signal quality, coverage, and live line telemetry. Pick any signal in the **Edge Explainer** dropdown for a 2–3 sentence AI breakdown of why that edge exists — line, projection, recent form, and matchup context, all from local data.

</td>
</tr>
<tr>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/player-projection.png" alt="Player Projection" />

#### 🎯 Player Projection
Next-game projection with recent form, distribution context, and supporting visuals.

</td>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/player-line-lab.png" alt="Player Line Lab" />

#### 🧪 Player Line Lab
Historical outcome study around a selected player, metric, and current threshold.

</td>
</tr>
<tr>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/compare-players.png" alt="Compare Players" />

#### ⚖️ Compare Players
Side-by-side form, distributions, profile shape, and game logs.

</td>
<td width="50%" valign="top" align="center">

<img src="docs/screenshots/model-diagnostics.png" alt="Model Diagnostics" />

#### 🔬 Model Diagnostics
Held-out model quality, ranking, residuals, and feature drivers.

</td>
</tr>
<tr>
<td colspan="2" valign="top">
<h4>🤖 Hooplytics Scout</h4>
<p>Bring-your-own-key chatbot grounded in your local roster, projections, model metrics, and live edge rows. Toggle <strong>OpenAI</strong> or <strong>Anthropic Claude</strong> in the sidebar — both providers run the same prompts and grounding payload, so picks read identically regardless of which one you bring. Hybrid mode by default lets the model layer in general NBA reasoning (explicitly labeled); flip <strong>Strict grounded</strong> to force local-only answers. Pick suggestions always include structured <strong>Confidence</strong> and <strong>Risk factors</strong>, never a guarantee.</p>
<p>🔑 <strong>Key setup:</strong> Paste an OpenAI or Claude key in the sidebar, or set <code>OPENAI_API_KEY</code> / <code>ANTHROPIC_API_KEY</code> in <code>.env</code> or Streamlit secrets.</p>
<p>🤝 <strong>Connect:</strong> Click <strong>Connect</strong> — available chat models are fetched from your key and the best GPT- or Claude-family model is auto-selected.</p>
<p>💬 <strong>Ask anything:</strong> <em>"Give me a MORE/LESS read on the largest edge tonight, with confidence and risk factors."</em></p>
<p>🔒 <strong>Grounding modes:</strong> <strong>Hybrid</strong> (default) allows labeled general NBA reasoning · <strong>Strict</strong> only cites local data.</p>
<p>🛡️ <strong>Privacy:</strong> Your key stays in session memory only — never written to disk or printed in logs.</p>
</td>
</tr>
<tr>
<td colspan="2" valign="top">
<h4>📑 Roster Report (PDF)</h4>
<p>One-click, print-ready editorial scouting report built with ReportLab — no headless browser required. Pulls directly from the live model bundle, edge board, and (optional) AI scout context (OpenAI or Claude — provider swap is invisible to the report). Designed to read like a magazine: serif display type, cream paper, hairline rules, and color-coded OVER / UNDER signals throughout. Every per-player page tags the active player in the page chrome so a reader can never get lost.</p>
<p>📰 <strong>Tonight's Slate (cover):</strong> Headline call-out for the loudest mispricing, divergent edge skyline of every live signal, and KPI rail (players, live signals, median R²).</p>
<p>🎯 <strong>Tonight's Setup:</strong> Anchor / Differentiator / Secondary cards with confidence chips, recent-form sparkline, and the ranked Top-4 signal cards.</p>
<p>📊 <strong>Signal Board:</strong> Full ranked board of every live edge with side, projection vs. line, hit %, confidence, and book counts.</p>
<p>🧭 <strong>Conviction Map:</strong> Numbered scatter (|edge| vs. book depth) with quadrant labels (SLEEPER / HEADLINE / SKIP / CROWD PLAY), a polished Signal Index legend mapping each marker to player · market · edge, and two AI Scout Picks with full untruncated rationale.</p>
<p>🔬 <strong>Model Quality:</strong> Composite trust meter on the left and per-target reliability lollipops on the right.</p>
<p>👤 <strong>Per-player profiles:</strong> Hero block with tonight's call · recent-form pills · last-4 resolved lines · sparklines · model projection vs. line table · full latest context and analyst notes from the AI scout.</p>
<p>Open the <strong>Roster Report</strong> page, click <em>Generate PDF</em>, and download.</p>
</td>
</tr>
<tr>
<td colspan="2" valign="top">
<h4>🏋️ Player Performance Analytics (PDF)</h4>
<p>A second printable report on the same Roster Report page — strictly performance-oriented (no betting edges, no projection-vs-line content). Designed for coaching staffs, player development, and anyone who wants the same magazine chrome focused on how a player is actually playing. Two pages per player; every page chrome tags the active player.</p>
<p>📰 <strong>Cover:</strong> Roster headline scoreboard with PTS / REB / AST / TS% per player, deep-linked names that jump straight to that player's profile page.</p>
<p>📋 <strong>Roster overview:</strong> Per-player snapshot table showing <strong>season + L10 stacked</strong> for PTS / REB / AST / PRA / MIN / FAN, plus a roster skill overlay radar so you can see every player's skill shape on a single chart.</p>
<p>📈 <strong>Per-player profile · Page A:</strong> Dark hero band with headline averages · KPI scorecard strip with L10 deltas · Garmin-style activity rings (SCORING / PLAYMAKING / EFFICIENCY vs. roster leader) · ML next-game projection panel (linear-regression forecast with 80% prediction interval and trend arrows) · trend sparklines for 6 primary stats over the last 20 games with rolling-5 overlay · <strong>Points by Game · Last 10</strong> tile strip (date, opponent, PTS color-coded vs season average, mini bar, REB / AST footer).</p>
<p>📊 <strong>Per-player profile · Page B:</strong> Shooting & efficiency bars (FG% / 3P% / FT% / TS%) with roster-median markers · skill-axis radar · floor / median / ceiling consistency strip · role & usage trends · hot/cold streak detection (z-scored vs. season baseline) · three accent-topped coaching cards (Strengths / Growth / Focus) with optional AI-augmented narrative (OpenAI or Claude).</p>
<p>Open the <strong>Roster Report</strong> page, switch the report-type toggle to <em>Player Performance Analytics</em>, and click <em>Generate performance report</em>.</p>
</td>
</tr>
</table>

<details>
<summary><strong>📓 Notebook gallery</strong> &nbsp;— earlier-era visualizations from the Jupyter workflow</summary>

<br>

<table>
<tr>
<td width="50%" align="center">

![Roster builder](docs/assets/roster-builder.png)
<sub>Roster setup with searchable player selection</sub>

</td>
<td width="50%" align="center">

![Rolling form chart](docs/assets/rolling-form-chart.png)
<sub>Recent-form view with hoverable game detail</sub>

</td>
</tr>
<tr>
<td width="50%" align="center">

![Predicted vs actual](docs/assets/predicted-vs-actual.png)
<sub>Held-out predicted-vs-actual diagnostics</sub>

</td>
<td width="50%" align="center">

![Feature importance](docs/assets/feature-importance.png)
<sub>Feature importance across model types</sub>

</td>
</tr>
</table>

</details>

---

## 🤔 Questions Hooplytics Helps You Answer

A player intelligence workbench is built to make data easier to *explore, explain, and challenge* — not to hand you a magic number.

- 🔮 What does this player's recent form actually look like?
- 📐 How does the model projection compare with tonight's posted line?
- 📈 Is the player trending above or below their season baseline?
- 🎚️ Which signals are stable, and which are noisy?
- 🗓️ How often has the player finished above similar thresholds historically?
- 🧭 Where do diagnostics suggest confidence — and where do they suggest caution?

---

## 🧠 Under the Hood

<table>
<tr>
<td valign="top">

- 🏀 Builds player-level datasets from NBA game logs via `nba_api`
- 🛡️ Engineers rolling and per-36 features for **pregame-safe** modeling
- 🤖 Trains eight projection models across core counting stats and fantasy score
- 📡 Pulls live line context from **The Odds API** for projection-vs-line comparison

</td>
<td valign="top">

- 📊 Visualizes recent form, distributions, profiles, residuals, and importance
- 🧪 Supports historical outcome studies and threshold sensitivity analysis
- 🧭 Same workflow through a notebook, a CLI, and a Streamlit dashboard
- 🔁 Reproducible end-to-end with cached datasets and pipelines

</td>
</tr>
</table>

---

## 🌟 Feature Highlights

| Area | Highlights |
| :--- | :--- |
| 🎛️ **Streamlit dashboard** | Eight purpose-built pages: Home, Player Projection, Analytics Dashboard, Compare Players, Player Line Lab, Model Diagnostics, Hooplytics Scout, Roster Report |
| 📑 **PDF Roster Report** | Editorial, magazine-style ReportLab PDF — Tonight's Slate cover, Tonight's Setup card stack, ranked Signal Board, Conviction Map with Signal Index legend and AI Scout Picks, Model Quality trust meter, and per-player profiles with latest context, sparklines, last-4 resolved lines, and active-player tag in the page chrome |
| 🏋️ **PDF Player Performance Analytics** | Second coach-focused PDF on the same page — 2 pages per player. KPI scorecards, Garmin-style activity rings, ML next-game projection with 80% prediction interval, trend sparklines, **Points by Game · Last 10** tile strip, shooting & efficiency bars, skill-axis radar, floor/median/ceiling consistency, role & usage trends, hot/cold streak z-scores, and three accent-topped coaching cards (Strengths / Growth / Focus) |
| 🤖 **Hooplytics Scout (AI)** | BYO-key chatbot — pick **OpenAI** or **Anthropic Claude** in the sidebar; both providers run the same grounded prompts. Hybrid or Strict grounded modes, structured Confidence + Risk factors |
| 📰 **AI Slate Brief** | One-paragraph daily read of tonight's loudest mispricings on the Home page. Cached per-day so it costs a single API call, regardless of how many times you refresh. |
| 🔍 **AI Edge Explainer** | Pick any signal in the Analytics Dashboard dropdown and get a 2–3 sentence breakdown grounded in line, projection, recent form, and matchup context. |
| 📡 **Live line context** | Auto-fetched lines from The Odds API across CLI and dashboard, with session-only BYO-key support in the web app |
| 🎯 **Edge board** | Slate-wide projection-vs-line gap analysis, signed edges, MORE/LESS calls, and book counts — feeds the dashboard, the AI scout, and the PDF report |
| 👤 **Player analysis** | Recent form, rolling trends, distributions, player profiles, season averages, and recent-window comparisons |
| 🔄 **Role Shift Detection** | **RoleShiftDetector** monitors 4 signals (assists σ, scoring σ, usage FGA%, minutes%) against per-signal thresholds. WARN flags add a confidence note; SUPPRESS flags flip projections to **NO_CALL** for the affected stats. Suppression map is empirically validated via a full backtest attribution cross-table (403 games, +0.064 overall directional-accuracy lift). Visible in the sidebar **Role Alerts** widget, the `role-shift` CLI command, and the `check_role_shift` MCP tool. |
| 🧠 **Modeling stack** | RACE blend (Ridge + kNN + Random Forest pipelines) across eight target stats, role and context features |
| 🎚️ **Market-anchored calibration** | Two-layer calibration applied at inference (Huber per-market `actual ≈ a + b·line` + per-player residual mean clipped to ±20%) blended with the model via per-market weights — corrects systematic bias without retraining. Built with `hooplytics-build-calibration` and shipped as `bundles/calibration_v1.json` |
| 📦 **Prebuilt RACE bundles** | Multiple ready-to-use bundles ship in `bundles/` (e.g. `race_fast.joblib`, `race_playoffs.joblib`). The Streamlit app auto-loads one on launch and lets you switch between them from the sidebar — zero cold-start training required |
| 🔬 **Diagnostics** | RMSE / MAE / R², predicted-vs-actual panels, residual views, feature importance, and per-stat health summaries |
| ⚡ **CLI workflows** | Single-player projection, prop comparison, scenario inputs, live line board, roster persistence, and prebuilt-bundle training |
| 🤖 **MCP server** | Use the full Hooplytics engine directly from Claude Desktop. Ten tools cover projections, prop analysis, role shift detection, scenario scoring, live line boards, scout reports, slate briefs, and roster management — see [HOOPLYTICS_MCP_SETUP.md](HOOPLYTICS_MCP_SETUP.md) |
| 📓 **Notebook workflow** | Rich exploratory narrative with tables, charts, code, and reproducible analysis in one place |

---

## ⚖️ Analytics first

Lines from The Odds API are treated as **analytical inputs** — thresholds to compare against projections, recent form, and historical outcomes. Hooplytics uses them to ask better questions:

- How far is the model projection from the current line?
- Is recent form above or below the season baseline?
- How volatile is the player around this threshold?
- How often has the player finished above similar thresholds?
- Does the model signal agree with historical performance?

> Hooplytics is a statistical analysis project for learning, exploration, and visualization.

---

## ⚡ CLI Walkthrough

Hooplytics ships with a Typer-based CLI that renders to **Rich** tables and panels in your terminal. Reproducible, scriptable, and `--json` friendly.

### Available commands

| Command | Purpose |
| :--- | :--- |
| `hooplytics project` | Project a player's next game across all 8 models |
| `hooplytics role-shift` | Detect assist / scoring / usage / minutes role shifts with WARN / SUPPRESS severity |
| `hooplytics prop` | Compare a player projection against a posted line for a single stat |
| `hooplytics decisions` | 8-stat projection summary with model-vs-line gap analysis |
| `hooplytics scenario` | Score a hypothetical box-score JSON payload |
| `hooplytics lines` | Live line board for the tracked roster, sorted by projection gap |
| `hooplytics train` | Pre-warm and cache the model bundle |
| `hooplytics-train-bundle` | Interactive prebuilt bundle trainer with progress bars and R2 validation gates |
| `hooplytics-build-calibration` | Fit the market-anchored calibration artifact (`bundles/calibration_v1.json`) from cached odds + game logs |
| `hooplytics roster list` | Show the tracked roster |
| `hooplytics roster add` | Add a player to the tracked roster |
| `hooplytics roster remove` | Remove a player from the tracked roster |

### Example CLI usage

```bash
# Next-game projection across all 8 models
hooplytics project "Victor Wembanyama"

# Role shift check — prints WARN/SUPPRESS panel; exits 0 (NONE), 1 (WARN), 2 (SUPPRESS)
hooplytics role-shift "Jalen Brunson"
hooplytics role-shift "Donovan Mitchell" --json   # machine-readable output

# Single-stat comparison — line auto-fetched from The Odds API
hooplytics prop "Shai Gilgeous-Alexander" points

# Same comparison with an explicit line override
hooplytics prop "Shai Gilgeous-Alexander" points --line 31.5

# 8-stat decision view, no live lines (offline mode)
hooplytics decisions "LeBron James" --no-live

# Live line board, freshly fetched
hooplytics lines --refresh

# Score a what-if box-score row
hooplytics scenario '{"fgm":8,"fga":15,"fg3m":3,"ftm":4,"min":34,"fg_pct":0.53,"ft_pct":1.0,"oreb":1,"dreb":5}'

# Roster + cache management
hooplytics roster add "Anthony Edwards"
hooplytics train

# Build and ship a prebuilt Streamlit bundle (defaults to bundles/race_fast.joblib)
hooplytics-train-bundle --mode exhaustive --players-source postseason-plus-anchors

# Fit the market-anchored calibration artifact from cached odds (used automatically by predict)
hooplytics-build-calibration build --season 2024-25 --season 2025-26 --verbose
```

> 🔑 `hooplytics lines` and live-enabled `prop` / `decisions` need `ODDS_API_KEY` (from `.env` or your shell). All commands support `--help`, and most reporting commands support `--json` for scripting.

---

## 🤖 Use from Claude (MCP)

Hooplytics ships a [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes the full projection + analytics engine to Claude Desktop (and any MCP-compatible client). Once connected, Claude can answer questions like *"project Anthony Edwards' next game and tell me which stat has the biggest edge"* or *"give me tonight's slate brief — which players have the loudest mispricings?"* by calling Hooplytics tools directly.

**What you get inside Claude:**

- **Next-game projections** across all 8 RACE models, plus single-stat MORE/LESS calls vs sportsbook lines.
- **Role shift detection** (`check_role_shift`) — ask Claude "is there a role shift concern for Jalen Brunson?" and get a full signal breakdown with WARN / SUPPRESS severity and per-stat NO_CALL flags.
- **Live line board** sorted by projection gap, with book counts and consensus medians.
- **Scenario scoring** for hypothetical box scores ("what if he plays 36 minutes and shoots 55%?").
- **AI Scout reports** and **AI Slate Briefs** — the same prose engine the Streamlit dashboard uses, grounded in projection data.
- **Player analytics** (game logs, season vs recent averages, volatility, trend deltas) and **roster management** (add / remove / list / reset).

**Setup:** see [HOOPLYTICS_MCP_SETUP.md](HOOPLYTICS_MCP_SETUP.md) for the Claude Desktop config block, prerequisites, and example prompts. Local stdio mode works out of the box; SSE mode is supported for remote clients.

> ℹ️ **Roster sharing.** The MCP server reads and writes the same `~/.hooplytics/roster.json` the CLI uses, so adding a player via Claude shows up in `hooplytics roster list` and vice versa. The Streamlit dashboard keeps its own session-only roster and does not sync with either.

---

## 📦 Installation

<table>
<tr>
<td width="33%" valign="top">

#### Base install

```bash
git clone https://github.com/texasbe2trill/hooplytics.git
cd hooplytics

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

</td>
<td width="33%" valign="top">

#### With notebook extras

```bash
pip install -e .[notebook]
```

Adds Jupyter, ipywidgets, and notebook-only visualization helpers.

</td>
<td width="33%" valign="top">

#### Web / Streamlit only

```bash
pip install -r requirements.txt
```

Minimal install for Streamlit Cloud or web-only deploys. Does not include notebook deps.

</td>
</tr>
</table>

> **Requires Python 3.11 or later.**

---

## 🔐 Configuration

Three optional API keys — none of them are required to launch the app.

| Variable | What it powers |
| :--- | :--- |
| `ODDS_API_KEY` | Live line context, edge board, prop comparisons (The Odds API) |
| `OPENAI_API_KEY` | Hooplytics Scout, AI Slate Brief, AI Edge Explainer, AI prose in both PDFs |
| `ANTHROPIC_API_KEY` | Same as above, but routed through Anthropic Claude when the sidebar provider is set to Claude |

Three safe ways to supply any of them:

| Method | How |
| :--- | :--- |
| 📄 **Local `.env`** | Copy `.env.example` → `.env`, set `ODDS_API_KEY=…`, `OPENAI_API_KEY=…`, `ANTHROPIC_API_KEY=…` |
| 🐚 **Shell session** | `export ODDS_API_KEY=…` (etc.) |
| 🌐 **Streamlit sidebar** | Paste your key into the sidebar password field — session-only, never stored |

```bash
cp .env.example .env
# then edit .env and set the keys you want to use
```

> 🔓 If no AI key is configured, the AI features (Scout, Slate Brief, Edge Explainer, PDF prose) cleanly disable and the rest of the app keeps working. If no Odds API key is set, you just lose the live-line column.

---

## 🛠️ Usage

### 🎛️ Streamlit dashboard — the recommended starting point

```bash
hooplytics-web
```

This launches the full multi-page dashboard at `http://localhost:8501`. A pre-trained **RACE** model bundle ships with the repo at `bundles/race_fast.joblib` and is auto-loaded — you get production-quality projections instantly, no training step required.

<table>
<tr>
<td valign="top" width="33%">
<strong>🔑 Sidebar setup (optional)</strong><br>
• Paste your <strong>Odds API key</strong> to enable live lines and the edge board.<br>
• Under <em>Hooplytics Scout</em>, pick <strong>OpenAI</strong> or <strong>Anthropic Claude</strong> and paste the matching key.<br>
• Keys are session-only — never written to disk.
</td>
<td valign="top" width="33%">
<strong>📍 Where to go first</strong><br>
• <strong>Home</strong> for the slate overview.<br>
• <strong>Analytics Dashboard</strong> for the live edge board.<br>
• <strong>Player Projection</strong> for a single-player deep dive.<br>
• <strong>Roster Report</strong> to export a branded PDF.<br>
• <strong>Hooplytics Scout</strong> to chat with your data.
</td>
<td valign="top" width="34%">
<strong>👥 Roster management</strong><br>
The sidebar lets you add or remove tracked players. Changes are persisted locally between sessions, so your roster is ready to go next time you launch the app.
</td>
</tr>
</table>

### ⚡ CLI

```bash
hooplytics --help                        # browse all commands
hooplytics roster list                   # see who's tracked
hooplytics project "Jalen Brunson" --last-n 10
hooplytics lines --refresh               # fresh live line board
```

### 🎚️ Choosing a model bundle

Hooplytics ships multiple pretrained bundles in [`bundles/`](bundles/) so you can switch model behavior without retraining. The default app launch uses `race_fast.joblib`.

**In the Streamlit sidebar:**

1. Make sure **Use prebuilt model bundle** is checked (it is by default).
2. A **Bundle** dropdown appears right below it listing every `.joblib` file found in `bundles/`.
3. Pick a bundle — the app reloads automatically. All projections, the edge board, the AI scout, and the PDF report immediately reflect the new bundle.

| Bundle | Best for |
| :--- | :--- |
| 🏃 **`race_fast.joblib`** | Default. Broad regular-season coverage trained on a large rolling window. Fast to load, balanced across all 8 stat targets. |
| 🏆 **`race_playoffs.joblib`** | Playoff-tuned variant. Weighted toward higher-stakes, lower-pace games — useful for postseason slates. |

> 💡 **Power users:** drop any additional `*.joblib` you trained with `hooplytics-train-bundle` into `bundles/` and it appears in the dropdown on next reload. To pin a non-default bundle headlessly, set `HOOPLYTICS_PRETRAINED_BUNDLE=/abs/path/to/your.joblib` in `.env` or your Streamlit secrets.

### 📓 Jupyter workflow

```bash
jupyter lab hooplytics.ipynb
```

Follow the notebook top-to-bottom for the full narrative analysis, or jump to a section.

---

## 📚 Data Sources

| Source | Used for |
| :--- | :--- |
| 🏀 `nba_api` | NBA player game logs and player metadata |
| 📡 The Odds API | Optional live line and book-count context |
| 💾 Local cache | Parquet/JSON caching for faster repeated workflows |

> Hooplytics does not redistribute NBA game data. All data is fetched at runtime.

---

## 🧪 Analytics Approach

Hooplytics emphasizes transparent, pregame-safe modeling rather than black-box outputs.

### Targets

| Model name | Target |
| :--- | :--- |
| `points` | `pts` |
| `rebounds` | `reb` |
| `assists` | `ast` |
| `pra` | `pts + reb + ast` |
| `threepm` | `fg3m` |
| `stl_blk` | `stl + blk` |
| `turnovers` | `tov` |
| `fantasy_score` | `fantasy_score` |

### Modeling principles

> 🛡️ **Pregame-safe.** Rolling windows are computed from prior games only — no leakage.
>
> 🧱 **Pipelines, not magic.** Scaling and feature handling stay inside scikit-learn pipelines, train/test boundaries intact.
>
> ⚖️ **Compare, don't hide.** Multiple model families are shown side by side instead of collapsing to one number.
>
> 🔬 **Diagnostics are first-class.** Calibration, residuals, and feature importance are surfaced everywhere, not buried.
>
> 🧪 **Honest about noise.** Steals + blocks and turnovers are framed as lower-signal categories — always.

---

## 📁 Project Structure

```text
hooplytics/
├── hooplytics.ipynb              # Narrative notebook workflow
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── index.html
│   ├── assets/                   # Notebook-era visualizations
│   └── screenshots/              # Streamlit dashboard captures
├── bundles/
│   ├── race_fast.joblib          # Default RACE bundle auto-loaded by the app
│   ├── race_playoffs.joblib      # Playoff-tuned RACE bundle (selectable in sidebar)
│   └── calibration_v1.json       # Market-anchored calibration artifact (auto-applied by predict)
├── hooplytics/
│   ├── cli.py                    # Typer CLI entry point (includes `role-shift` command)
│   ├── constants.py
│   ├── data.py                   # Game log ingestion + caching
│   ├── fantasy.py
│   ├── features_context.py       # Pace / matchup / opponent context
│   ├── features_market.py        # Market-aware features
│   ├── features_role.py          # Role / usage features
│   ├── models.py                 # 8-stat RACE model training
│   ├── odds.py                   # The Odds API client
│   ├── backtest.py               # Retro-projection accuracy backtest (pregame-safe)
│   ├── role_shift_detector.py    # RoleShiftDetector: 4-signal WARN/SUPPRESS engine
│   ├── role_shift_validate.py    # Empirical backtest validator (directional accuracy by severity)
│   ├── role_shift_attribute.py   # Attribution cross-table: (signal × stat) directional-accuracy lift
│   ├── role_shift_sweep.py       # Threshold sweep utility for signal calibration
│   ├── ai_agent.py               # Provider-agnostic AI dispatcher (OpenAI ↔ Claude)
│   ├── openai_agent.py           # OpenAI transport: chat, slate brief, edge explainer, PDF prose
│   ├── anthropic_agent.py        # Anthropic Claude transport (mirrors openai_agent's API)
│   ├── predict.py                # Projection + line comparison (auto-applies calibration)
│   ├── calibration.py            # Two-layer market-anchored calibration
│   ├── calibration_cli.py        # `hooplytics-build-calibration` entry point
│   ├── report.py                 # PDF Roster Report builder (ReportLab)
│   ├── report_performance.py     # PDF Player Performance Analytics builder (ReportLab)
│   ├── train_bundle.py           # Interactive prebuilt-bundle trainer
│   └── web/
│       ├── app.py                # Streamlit multi-page app
│       ├── charts.py
│       ├── launcher.py           # `hooplytics-web` entry point
│       ├── role_shift_widget.py  # Role Alerts sidebar widget (WARN/SUPPRESS chips + signal cards)
│       └── styles.py
└── tests/
```

---

## 🗺️ Roadmap

#### 🤖 AI & chat
- 🌊 **Streaming Scout responses** — render tokens as they arrive in Hooplytics Scout for a snappier read on long replies. Both the OpenAI and Anthropic SDKs already support `stream=True`; wire it through the dispatcher and surface via `st.write_stream`.
- 🔀 **Side-by-side provider comparison** — send the same prompt to OpenAI and Claude in parallel and diff the picks. The provider abstraction already supports this; just needs a UI toggle for second-opinion mode.
- 🗣️ **Conversational Line Lab** — natural-language "what-ifs" on the scenario explorer ("what changes if pace drops 4?", "show me a back-to-back scenario"). Mutates the scenario dict and re-projects in place.
- 📝 **Post-game recap memo** — daily auto-generated readout pairing yesterday's model calls with actual outcomes ("model said 24.3 PRA on a 21.5 line, player went 28; the OVER lean held"). Closes the trust loop and makes the model's tracking visible.
- 🧑‍🎨 **Custom system prompts** — power-user override for Scout / PDF prose (per-roster voice, risk tolerance, format preferences) without forking the codebase.

#### 📦 Data & analytics
- 🏷️ **Player archetype clustering** — auto-tag players ("primary creator", "stretch big", "off-ball wing") from role + shooting embeddings, surfaced in the grounding payload so AI prose can reason about role-shape directly.
- 👥 **Multi-roster profiles** — save and switch between named rosters from the sidebar (DFS lineup, season-long fantasy, futures slate) without losing each one's bundle / model state.
- 📡 Richer book-level line telemetry inside the Streamlit app — per-book vig, line movement, and fastest-mover attribution.
- 🔄 **Retrain RACE bundles on enriched odds data** — role shift detection lifts directional accuracy by +0.064 overall; the next step is retraining the core RACE models with cached sportsbook lines as features so the base projection absorbs market information before the suppression layer runs.

#### 🎨 UX & polish
- 🎬 Fresh Streamlit dashboard screenshots and rendered demos for the Roster Report, Performance Analytics, and Hooplytics Scout pages.
- 🎨 Saveable / shareable PDF report templates with custom branding (team colors, logo, footer).
- 📦 Reproducible demo datasets plus broader player and season presets for faster onboarding.

---

## ⚠️ Disclaimer

> Hooplytics is for **statistical analysis, education, and entertainment**.
>
> Line values are used as contextual inputs for comparing model projections, recent form, and historical outcomes. The project is **not an execution system and not a guarantee of future results**.

---

## 🤝 Contributing

Issues and pull requests are welcome, especially around:

- 🤖 model quality and calibration
- 🎨 UX improvements for the dashboard or CLI
- 📊 additional visualization layers
- 📚 documentation and reproducibility

---

## 📜 License & Acknowledgements

**MIT © 2026 [Chris Campbell](https://github.com/texasbe2trill)**

Hooplytics is the Python evolution of [hooplyticsR](https://github.com/texasbe2trill/hooplyticsR), with additional dashboarding, modeling, and CLI tooling.

**Built on the shoulders of:**

- 🏀 the [`nba_api`](https://github.com/swar/nba_api) project for accessible NBA stats endpoints
- 📡 [The Odds API](https://the-odds-api.com/) for optional live line context
- 📊 [Plotly](https://plotly.com/), [Streamlit](https://streamlit.io/), [Typer](https://typer.tiangolo.com/), [Rich](https://rich.readthedocs.io/), [pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/) for the core application stack

<div align="center">

<br>

*If Hooplytics helps you think more clearly about player performance, consider giving the repo a ⭐.*

</div>
