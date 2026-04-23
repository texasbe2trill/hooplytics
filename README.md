# 🏀 Hooplytics

> *Box scores in. Hot takes out.*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Hooplytics** turns NBA game logs into stats, visualizations, and machine-learning-powered More/Less calls. It's the Python port of [hooplyticsR](https://github.com/texasbe2trill/hooplyticsR) — same spirit, but reborn as an interactive Jupyter notebook with modern ML, slicker plots, and a one-knob configuration.

Type a player's name. Get a real-time analytics report.

---

## ⛹️ What you get

| Section | What it does |
|---|---|
| **Tale of the tape** | Per-player μ / σ / fantasy score. Color-graded tables. |
| **Consistency leaderboard** | Coefficient of variation — *who can you actually trust on a Tuesday?* |
| **Distributional vibes** | Faceted histograms with KDE overlays, side-by-side violin grid for every core stat. |
| **Rolling form chart** | Interactive 10-game rolling fantasy line. Hover, isolate, compare. |
| **Player profile radar** | Min-max normalized polar chart — at-a-glance archetypes. |
| **8 ML models** | scikit-learn `Pipeline`s tuned via `GridSearchCV`, evaluated on a 20% held-out split with RMSE / MAE / R². |
| **Predicted-vs-actual scatter** | Eyeball calibration per stat. |
| **Random-forest importances** | What did the model *actually* learn? |
| **More/Less engine** | Blends model predictions with sportsbook lines + auto-derived 5-game form into per-stat decisions. |
| **Try it yourself** | Three runnable recipes for hypothetical scenarios, next-game projections, and custom prop bets. |

---

## 🚀 Quickstart

```bash
git clone https://github.com/texasbe2trill/hooplytics.git
cd hooplytics

python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab hooplytics.ipynb
```

Run all cells (`⏵ Run All` / `Restart and Run All`). The first run pulls game logs from the NBA Stats API and caches them as Parquet under `data/cache/`; every subsequent run is instant.

> Headless / CI? `jupyter nbconvert --to notebook --execute hooplytics.ipynb`

---

## 🎛️ The whole interface is one dictionary

Edit **`ROSTER`** in §1.1 and you're done. The summaries, plots, models, and prop-bet calls all rebuild themselves.

```python
ROSTER = {
    "LeBron James":      {"seasons": CURRENT, "proj": {"points": 21.5, "fantasy_score": 41.5, "pra": 34.0}},
    "Kevin Durant":      {"seasons": CURRENT, "proj": {"points": 26.0, "fantasy_score": 42.0}},
    "Victor Wembanyama": {"seasons": CURRENT, "proj": {"points": 25.0, "fantasy_score": 53.0}},
    # Add a row, that's it 👇
    "Anthony Edwards":   {"seasons": nba_seasons(2023, 2025), "proj": {"points": 27.0}},
}
```

`seasons` accepts a list of NBA-style strings (`"2025-26"`); `nba_seasons(2023, 2025)` is a helper for ranges. `proj` is optional — if you skip it, the player's season average becomes the baseline. **5-game averages auto-derive from the data**, so you never maintain them by hand.

---

## 🎯 Three recipes for asking your own questions

### 1. Score a hypothetical stat line

> *"What does the model predict if KD goes 9-of-16 from the field, 3-of-7 from three, in 36 minutes?"*

```python
predict_scenario(dict(
    fgm=9, fga=16, fg3m=3, fg3a=7, ftm=4, fta=4,
    min=36, fg_pct=0.563, fg3_pct=0.429, ft_pct=1.000,
    oreb=1, dreb=5, pts=25, reb=6, ast=4, stl=1, blk=0, tov=2, plus_minus=8,
))
```

Returns a table with predictions for every model whose features are satisfied by your scenario.

### 2. Project a player's next game from recent form

> *"What's Wemby expected to do tonight given his last 10 games?"*

```python
project_next_game("Victor Wembanyama", last_n=10)
```

Uses a rolling median (robust to outliers) of the player's actual recent box scores to feed every model.

### 3. Run a sportsbook line through the decision engine

> *"Wemby's points line tonight is 24.5. Take the over?"*

```python
custom_prop(
    player="Victor Wembanyama",
    model_name="points",
    line=24.5,
    last_n=5,   # window for both the prediction features AND the recent average
)
# → {'model prediction': 38.13, '5-game avg': 34.6, 'edge': +10.07, 'call': 'MORE ✅'}
```

The engine inflates the threshold by a 10% confidence margin so it only fires on conviction.

---

## 📊 The models under the hood

| Target          | Model            | Predictors                                          |
|-----------------|------------------|-----------------------------------------------------|
| `pts`           | kNN (tuned)      | `fgm, fg3m, ftm, min, fg_pct, ft_pct`               |
| `reb`           | kNN (tuned)      | `oreb, dreb, min`                                   |
| `ast`           | RandomForest     | `min, pts, plus_minus, fga`                         |
| `pra`           | kNN (tuned)      | `pts, reb, ast, min, plus_minus`                    |
| `fg3m`          | kNN (tuned)      | `fg3a, min, fg3_pct`                                |
| `stl_blk`       | kNN (tuned)      | `min, plus_minus`                                   |
| `tov`           | kNN (tuned)      | `min, fga, ast`                                     |
| `fantasy_score` | RandomForest     | `pts, reb, ast, stl, blk, tov, min, plus_minus`     |

- **Pipelines** wrap `StandardScaler` + estimator so scaling is fit on training folds only (no leakage).
- **kNN** tuned over `n_neighbors ∈ [3, 21]` via `GridSearchCV` with **5-fold repeated CV** (2 repeats), mirroring the original R `caret` setup.
- **Random Forest** tunes `n_estimators`, `max_depth`, and `min_samples_leaf`.
- **80/20 train/test split**, seed `123` everywhere, scoring = `neg_root_mean_squared_error`.

**Fantasy scoring (DraftKings-ish):** `pts·1 + reb·1.2 + ast·1.5 + stl·3 + blk·3 − tov·1`.

---

## 🧠 Why a notebook (and not a script)?

The original project is an R Markdown narrative report — prose, tables, plots, and modeling output, all interleaved. A Jupyter notebook is the natural Python analog: it preserves the section-by-section storytelling, renders styled pandas tables and interactive Plotly charts inline, and lets you iterate on a single section without rerunning the whole pipeline.

A pure script would lose the narrative. A dashboard would lose the source-of-truth code. The notebook gets both.

---

## 🛠️ Requirements

- **Python 3.14+** (uses `typing.NotRequired` for the `ROSTER` schema)
- See [requirements.txt](requirements.txt)

---

## 📁 Project layout

```
hooplytics/
├── hooplytics.ipynb     # the whole report
├── README.md            # you are here
├── LICENSE
├── requirements.txt
├── .gitignore
└── data/cache/          # Parquet game-log cache, one file per player (gitignored)
```

---

## ⚖️ Disclaimer

Hooplytics is for analytical exploration and entertainment. The "MORE ✅ / LESS ❌" calls are not investment advice. Bet responsibly — or better yet, don't bet at all and just enjoy the math.

Game log data is fetched at runtime from the NBA Stats API and is **not redistributed** with this project. See the [NBA Terms of Use](https://www.nba.com/termsofuse).

---

## 📄 License

MIT © 2026 [Chris Campbell](https://github.com/texasbe2trill). See [LICENSE](LICENSE) for details.
