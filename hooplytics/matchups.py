"""Team-vs-team matchup predictions for the Roster Report.

Aggregates per-player point projections into team-level scores, derives a
projected spread / total / home win probability, and (optionally) joins live
sportsbook game lines so the report can render a "Tonight's Matchups" page
alongside the existing per-player edge content.

The roll-up is intentionally simple and transparent:

* For each team in tonight's slate we try to identify the **likely active
  rotation** from the modeling frame: top-N players by recent minutes across
  their last few games. This is just a recency heuristic — no manual rosters,
  no API calls.
* For every rotation player, we use the model's per-player point projection
  when the user has it (e.g. they're on the user's roster). When the user
  doesn't have a projection for that player, we fall back to that player's
  L10 (or season) PTS average from the modeling frame.
* Sum the projected points across the rotation → team projected score.
* Spread = home - away (signed; positive = home favored).
* Total = home + away.
* Home win probability uses a normal CDF on the projected margin with NBA
  σ ≈ 11.5 (historical std of game margins). ``math.erf`` is built-in so we
  don't pull in scipy.

When ``modeling_df`` doesn't cover one or both teams (e.g. the user has only
queried a few seasons), we emit a ``confidence`` of "low" and let the report
render a muted card. The AI prose layer reads ``confidence`` so it can hedge
appropriately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd


# NBA-margin std-dev used to convert a projected spread into a win probability.
# Historical seasons run ~11–12; 11.5 is a good middle ground that lines up
# with the implied probabilities of typical NBA spreads (e.g. -3 ≈ 60% fav).
_MARGIN_SIGMA: float = 11.5

# How many players per team make up the "rotation" we sum. NBA typical
# rotations land between 8 and 10; 9 keeps bench noise out without missing
# a clear sixth man. Players outside the top-N are ignored for the team total.
_ROTATION_SIZE: int = 9

# Recent-form window (in games) used to pick the rotation and to back-fill
# missing projections.
_RECENT_WINDOW: int = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm_team(name: str) -> str:
    """Lowercase + collapse whitespace for tolerant team-name matching."""
    return " ".join(str(name or "").lower().split())


def _normal_cdf(z: float) -> float:
    """CDF of the standard normal at z (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def margin_to_home_win_prob(margin: float, sigma: float = _MARGIN_SIGMA) -> float:
    """Convert a projected home-minus-away margin into P(home wins)."""
    if margin is None or not math.isfinite(margin):
        return 0.5
    return _normal_cdf(float(margin) / max(float(sigma), 1e-6))


def _player_points_projection(
    player: str,
    projections: dict[str, pd.DataFrame] | None,
) -> float | None:
    """Return the model's projected points for ``player`` from the user's
    projections dict, or ``None`` when the player isn't covered.
    """
    if not projections:
        return None
    frame = projections.get(player)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "model" not in frame.columns or "prediction" not in frame.columns:
        return None
    row = frame[frame["model"].astype(str).str.lower() == "points"]
    if row.empty:
        return None
    val = pd.to_numeric(row["prediction"], errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def _team_rotation(
    team_full: str,
    abbr: str,
    modeling_df: pd.DataFrame | None,
    *,
    rotation_size: int = _ROTATION_SIZE,
    recent_window: int = _RECENT_WINDOW,
) -> list[dict[str, Any]]:
    """Identify the likely active rotation for ``team_full`` from the modeling
    frame. Returns ``[{"player": str, "min_l": float, "pts_l": float,
    "games": int}, ...]`` sorted by recent minutes desc, capped at
    ``rotation_size``.

    Returns an empty list when the modeling frame doesn't cover this team —
    callers should treat that as a "low confidence" signal.
    """
    if modeling_df is None or modeling_df.empty:
        return []
    if "team_abbr" not in modeling_df.columns or "player" not in modeling_df.columns:
        return []
    if not abbr:
        return []

    df = modeling_df[modeling_df["team_abbr"].astype(str).str.upper() == abbr.upper()]
    if df.empty:
        return []

    # Sort by date so the recent window per player is the most recent games.
    if "game_date" in df.columns:
        df = df.sort_values("game_date")

    grouped = df.groupby("player", sort=False)
    rotation: list[dict[str, Any]] = []
    for player, sub in grouped:
        recent = sub.tail(recent_window)
        if recent.empty:
            continue
        min_l = float(pd.to_numeric(recent.get("min", pd.Series(dtype=float)), errors="coerce").mean()) if "min" in recent.columns else 0.0
        pts_l = float(pd.to_numeric(recent.get("pts", pd.Series(dtype=float)), errors="coerce").mean()) if "pts" in recent.columns else 0.0
        if not math.isfinite(min_l):
            min_l = 0.0
        if not math.isfinite(pts_l):
            pts_l = 0.0
        rotation.append({
            "player": str(player),
            "min_l": round(min_l, 2),
            "pts_l": round(pts_l, 2),
            "games": int(len(recent)),
        })
    rotation.sort(key=lambda r: (r["min_l"], r["pts_l"]), reverse=True)
    return rotation[:rotation_size]


# ── Public dataclass ────────────────────────────────────────────────────────

@dataclass
class MatchupPrediction:
    """Single team-vs-team prediction for tonight's slate.

    Field naming convention: every "spread"/"margin" field is signed
    ``home_minus_away`` so positive = home favored, negative = away favored.
    """

    matchup: str
    home_team: str
    away_team: str
    tipoff_iso: str = ""

    # Model output
    home_pts_proj: float = 0.0
    away_pts_proj: float = 0.0
    model_spread: float = 0.0  # home - away (positive = home favored)
    model_total: float = 0.0
    home_win_prob: float = 0.5
    away_win_prob: float = 0.5

    # Coverage / quality
    confidence: str = "medium"  # one of low / medium / high
    rotation_players_home: list[dict[str, Any]] = field(default_factory=list)
    rotation_players_away: list[dict[str, Any]] = field(default_factory=list)
    key_player_home: dict[str, Any] | None = None  # top contributor by points
    key_player_away: dict[str, Any] | None = None
    rostered_players_home: list[str] = field(default_factory=list)
    rostered_players_away: list[str] = field(default_factory=list)

    # Optional market-line attachments (filled by ``attach_market_lines``)
    market_spread: float | None = None  # home line; -3.5 means home -3.5
    market_total: float | None = None
    market_home_win_prob: float | None = None
    market_source: str = ""

    # Derived flags (only meaningful once market lines are attached)
    upset_flag: bool = False  # model picks underdog outright
    spread_edge: float | None = None  # model_spread - market_spread
    total_edge: float | None = None   # model_total - market_total

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        # Keep contained dataclass-friendly types
        return d


# ── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_team_score(
    team_full: str,
    abbr: str,
    modeling_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    roster_players: Iterable[str] | None,
) -> tuple[float, list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    """Return ``(team_pts_proj, rotation_with_pts, key_player, rostered_in_rotation)``.

    For each rotation player, prefer the model's per-player projection; fall
    back to the player's L10 PTS average when the user hasn't projected them.
    """
    rotation = _team_rotation(team_full=team_full, abbr=abbr, modeling_df=modeling_df)
    if not rotation:
        return 0.0, [], None, []

    roster_set = {str(p) for p in (roster_players or [])}

    total = 0.0
    rotation_with_pts: list[dict[str, Any]] = []
    rostered_in_rotation: list[str] = []
    key_player: dict[str, Any] | None = None

    for entry in rotation:
        player = entry["player"]
        proj = _player_points_projection(player, projections)
        source = "model" if proj is not None else "season_avg"
        if proj is None:
            proj = float(entry.get("pts_l", 0.0))
        total += float(proj)
        row = {
            "player": player,
            "pts_proj": round(float(proj), 2),
            "min_l": entry["min_l"],
            "pts_l": entry["pts_l"],
            "source": source,
        }
        rotation_with_pts.append(row)
        if player in roster_set:
            rostered_in_rotation.append(player)
        if key_player is None or row["pts_proj"] > key_player["pts_proj"]:
            key_player = row

    return round(total, 2), rotation_with_pts, key_player, rostered_in_rotation


def _confidence_label(
    home_rotation: list[dict[str, Any]],
    away_rotation: list[dict[str, Any]],
) -> str:
    """Heuristic confidence based on rotation coverage.

    * **high** when both rotations have ≥ 8 players with recent minutes.
    * **medium** when both rotations have ≥ 5 players.
    * **low** when both have ≥ 3 (the model rollup is partial but directional).
    * **thin** otherwise. Callers should suppress the model team totals when
      ``confidence == "thin"`` because the rollup is dominated by whichever
      side happens to be in the modeling frame and produces nonsense like
      "Toronto 0.0" when the user has no Raptors rostered.
    """
    h = sum(1 for r in home_rotation if r.get("min_l", 0) > 0)
    a = sum(1 for r in away_rotation if r.get("min_l", 0) > 0)
    if h >= 8 and a >= 8:
        return "high"
    if h >= 5 and a >= 5:
        return "medium"
    if h >= 3 and a >= 3:
        return "low"
    return "thin"


def project_matchup(
    *,
    home_team: str,
    away_team: str,
    matchup: str,
    tipoff_iso: str,
    abbr_to_full: dict[str, str],
    modeling_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    roster_players: Iterable[str] | None,
) -> MatchupPrediction:
    """Build a single :class:`MatchupPrediction` for one game on the slate."""
    full_to_abbr: dict[str, str] = {
        _norm_team(v): k for k, v in (abbr_to_full or {}).items()
    }
    home_abbr = full_to_abbr.get(_norm_team(home_team), "")
    away_abbr = full_to_abbr.get(_norm_team(away_team), "")

    home_pts, home_rotation, key_home, rostered_home = _aggregate_team_score(
        home_team, home_abbr, modeling_df, projections, roster_players,
    )
    away_pts, away_rotation, key_away, rostered_away = _aggregate_team_score(
        away_team, away_abbr, modeling_df, projections, roster_players,
    )

    spread = round(home_pts - away_pts, 2)
    total = round(home_pts + away_pts, 2)
    p_home = margin_to_home_win_prob(spread)
    p_away = 1.0 - p_home

    return MatchupPrediction(
        matchup=matchup,
        home_team=home_team,
        away_team=away_team,
        tipoff_iso=tipoff_iso,
        home_pts_proj=home_pts,
        away_pts_proj=away_pts,
        model_spread=spread,
        model_total=total,
        home_win_prob=round(p_home, 4),
        away_win_prob=round(p_away, 4),
        confidence=_confidence_label(home_rotation, away_rotation),
        rotation_players_home=home_rotation,
        rotation_players_away=away_rotation,
        key_player_home=key_home,
        key_player_away=key_away,
        rostered_players_home=rostered_home,
        rostered_players_away=rostered_away,
    )


def build_slate_predictions(
    *,
    slate: list[dict[str, str]] | None,
    abbr_to_full: dict[str, str],
    modeling_df: pd.DataFrame | None,
    projections: dict[str, pd.DataFrame] | None,
    roster_players: Iterable[str] | None,
    roster_only: bool = False,
) -> list[MatchupPrediction]:
    """Apply :func:`project_matchup` across the slate.

    Skips entries with no home_team / away_team. Always returns a list (empty
    when the slate is empty) so the caller doesn't need to guard.

    Parameters
    ----------
    roster_only
        When ``True``, only include matchups that have at least one rostered
        player on either team. Use this on the report path so the section
        doesn't list every NBA game on the slate when the user only cares
        about the games their rostered players are actually playing in.
    """
    out: list[MatchupPrediction] = []
    for ev in slate or []:
        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not home or not away:
            continue
        matchup = str(ev.get("matchup") or f"{away} @ {home}").strip()
        tipoff = str(ev.get("tipoff_iso") or "").strip()
        try:
            pred = project_matchup(
                home_team=home,
                away_team=away,
                matchup=matchup,
                tipoff_iso=tipoff,
                abbr_to_full=abbr_to_full,
                modeling_df=modeling_df,
                projections=projections,
                roster_players=roster_players,
            )
        except Exception:  # noqa: BLE001 — skip flaky games rather than fail the whole report
            continue
        if roster_only:
            has_rostered = bool(pred.rostered_players_home or pred.rostered_players_away)
            if not has_rostered:
                continue
        out.append(pred)
    return out


# ── Market line attachment ───────────────────────────────────────────────────

def _american_to_implied_prob(price: float | None) -> float | None:
    """Convert American odds to an implied probability (with vig)."""
    if price is None or not math.isfinite(float(price)):
        return None
    p = float(price)
    if p > 0:
        return 100.0 / (p + 100.0)
    return -p / (-p + 100.0)


def _devig_two_way(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    """Remove the bookmaker overround from a two-way market."""
    if p_a is None or p_b is None:
        return p_a, p_b
    s = p_a + p_b
    if s <= 0:
        return p_a, p_b
    return p_a / s, p_b / s


def attach_market_lines(
    predictions: list[MatchupPrediction],
    game_lines: list[dict[str, Any]] | None,
) -> list[MatchupPrediction]:
    """Merge market spreads/totals/moneylines onto each prediction in place.

    ``game_lines`` is the list returned by :func:`hooplytics.odds.fetch_game_lines`.
    Predictions whose home/away pair doesn't appear in the game-lines payload
    are left untouched. Returns the same list for chaining convenience.
    """
    if not predictions or not game_lines:
        return predictions

    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for line in game_lines:
        home = _norm_team(line.get("home_team", ""))
        away = _norm_team(line.get("away_team", ""))
        if home and away:
            by_pair[(home, away)] = line

    for pred in predictions:
        key = (_norm_team(pred.home_team), _norm_team(pred.away_team))
        line = by_pair.get(key)
        if line is None:
            continue

        market_spread = line.get("home_spread")
        market_total = line.get("total")
        home_ml = line.get("home_ml")
        away_ml = line.get("away_ml")

        if isinstance(market_spread, (int, float)) and math.isfinite(float(market_spread)):
            pred.market_spread = float(market_spread)
            pred.spread_edge = round(pred.model_spread - pred.market_spread, 2)
        if isinstance(market_total, (int, float)) and math.isfinite(float(market_total)):
            pred.market_total = float(market_total)
            pred.total_edge = round(pred.model_total - pred.market_total, 2)

        # Convert moneyline → de-vigged home win prob.
        ph = _american_to_implied_prob(home_ml)
        pa = _american_to_implied_prob(away_ml)
        ph, _pa = _devig_two_way(ph, pa)
        if ph is not None:
            pred.market_home_win_prob = round(float(ph), 4)

        # Upset flag: model picks the side the market has as the underdog.
        # Odds-API convention: ``market_home_spread`` is the line attached to
        # the home team — negative means home is favored. ``model_spread`` is
        # ``home_pts - away_pts`` — positive means model picks home to win.
        if pred.market_spread is not None:
            market_picks_home = pred.market_spread < 0
            model_picks_home = pred.model_spread > 0
            pred.upset_flag = market_picks_home != model_picks_home

        pred.market_source = str(line.get("source", "consensus"))

    return predictions


# ── Grounding payload helper ────────────────────────────────────────────────

def _display_summary(p: MatchupPrediction) -> str:
    """Pre-built one-line summary the AI must mirror verbatim.

    The Roster Report renders one fixed set of numbers per matchup card —
    the AI prose has to match those exact numbers or the reader sees a
    contradiction (e.g., the bar shows 60% home but the prose claims 55%
    away). Building the string here guarantees the AI has a single source
    of truth to copy from instead of having to choose between the model
    and market fields itself.
    """
    parts: list[str] = []
    # Win probability line — prefer market when available since that is what
    # the report displays for thin-coverage cards. For trustworthy rollups,
    # the report shows the model probability instead.
    if p.confidence in {"high", "medium", "low"}:
        wp_home = p.home_win_prob
        wp_source = "model"
    elif p.market_home_win_prob is not None:
        wp_home = p.market_home_win_prob
        wp_source = "market"
    else:
        wp_home = None
        wp_source = ""

    if wp_home is not None:
        wp_away = 1.0 - wp_home
        fav_team = p.home_team if wp_home >= 0.5 else p.away_team
        fav_pct = max(wp_home, wp_away) * 100.0
        parts.append(
            f"{wp_source.upper()} WP: {fav_team} {fav_pct:.0f}% "
            f"(home {p.home_team} {wp_home*100:.0f}% / away {p.away_team} {wp_away*100:.0f}%)"
        )

    if p.market_spread is not None:
        sign = "-" if p.market_spread < 0 else "+"
        parts.append(
            f"Market spread: {p.home_team} {sign}{abs(p.market_spread):.1f}"
        )
    if p.market_total is not None:
        parts.append(f"Market total: {p.market_total:.1f}")
    if p.confidence in {"high", "medium", "low"}:
        parts.append(
            f"Model score: {p.away_team} {p.away_pts_proj:.1f} @ "
            f"{p.home_team} {p.home_pts_proj:.1f}"
        )
    return " | ".join(parts)


def to_grounding_payload(
    predictions: list[MatchupPrediction],
    *,
    max_rotation_per_team: int = 5,
) -> list[dict[str, Any]]:
    """Compact, JSON-friendly view of predictions for the AI grounding block.

    Cuts each rotation list down to its top contributors so the prompt stays
    small even on a 12-game slate. Also includes a ``display_summary`` field
    — a pre-built string the AI prose must mirror verbatim so its win-prob
    and spread numbers can never contradict the card the report renders.
    """
    out: list[dict[str, Any]] = []
    for p in predictions or []:
        rot_home = list(p.rotation_players_home or [])[:max_rotation_per_team]
        rot_away = list(p.rotation_players_away or [])[:max_rotation_per_team]
        entry = {
            "matchup": p.matchup,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "tipoff_iso": p.tipoff_iso,
            "display_summary": _display_summary(p),
            "model_home_pts": p.home_pts_proj,
            "model_away_pts": p.away_pts_proj,
            "model_spread": p.model_spread,
            "model_total": p.model_total,
            "model_home_win_prob": round(p.home_win_prob, 3),
            "model_away_win_prob": round(p.away_win_prob, 3),
            "confidence": p.confidence,
            "rostered_players_home": list(p.rostered_players_home),
            "rostered_players_away": list(p.rostered_players_away),
            "top_contributors_home": rot_home,
            "top_contributors_away": rot_away,
        }
        if p.market_spread is not None:
            entry["market_home_spread"] = p.market_spread
        if p.market_total is not None:
            entry["market_total"] = p.market_total
        if p.market_home_win_prob is not None:
            entry["market_home_win_prob"] = p.market_home_win_prob
        if p.spread_edge is not None:
            entry["spread_edge_vs_market"] = p.spread_edge
        if p.total_edge is not None:
            entry["total_edge_vs_market"] = p.total_edge
        if p.upset_flag:
            entry["upset_flag"] = True
        out.append(entry)
    return out


__all__ = [
    "MatchupPrediction",
    "attach_market_lines",
    "build_slate_predictions",
    "margin_to_home_win_prob",
    "project_matchup",
    "to_grounding_payload",
]
