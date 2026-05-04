"""role_shift_widget.py
━━━━━━━━━━━━━━━━━━━━━
Streamlit rendering components for the RoleShiftDetector.

Public API
----------
render_role_shift_badge(result)            → coloured pill badge (inline)
render_role_shift_panel(result, expanded)  → expander with signal table
apply_stat_card_suppression(...)           → single stat card respecting severity
render_sidebar_role_shift_summary(...)     → sidebar list of all flagged players
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from hooplytics.role_shift_detector import RoleShiftDetector, RoleShiftResult, Severity


# ── Colour palette ────────────────────────────────────────────────────────────

_BADGE_OK = "background:#22c55e;color:#fff"
_BADGE_WARN = "background:#f59e0b;color:#fff"
_BADGE_SUPPRESS = "background:#ef4444;color:#fff"

_BORDER_OK = "#3b82f6"       # blue — normal
_BORDER_WARN = "#f59e0b"     # amber
_BORDER_SUPPRESS = "#ef4444" # red


# ── Badge ─────────────────────────────────────────────────────────────────────


def render_role_shift_badge(result: RoleShiftResult) -> None:
    """Render a small coloured pill badge inline via st.markdown."""
    if result.severity == Severity.SUPPRESS:
        style = _BADGE_SUPPRESS
        label = "🔴 SUPPRESS"
    elif result.severity == Severity.WARN:
        style = _BADGE_WARN
        label = "⚠ WARN"
    else:
        style = _BADGE_OK
        label = "✓ OK"

    st.markdown(
        f'<span style="'
        f"display:inline-block;padding:2px 10px;border-radius:9999px;"
        f"font-size:0.78rem;font-weight:700;letter-spacing:0.03em;"
        f'{style}'
        f'">{label}</span>',
        unsafe_allow_html=True,
    )


# ── Expander panel ────────────────────────────────────────────────────────────


def render_role_shift_panel(result: RoleShiftResult, expanded: bool = False) -> None:
    """Render a full role-shift detail panel inside a st.expander.

    Only renders when shift_detected is True.
    """
    if not result.shift_detected:
        return

    sev = result.severity
    if sev == Severity.SUPPRESS:
        banner_style = "background:#fef2f2;border-left:4px solid #ef4444;padding:0.6rem 1rem;border-radius:4px;"
        banner_text = f"🔴 <strong>SUPPRESS</strong> — {', '.join(s.value for s in result.shift_types)}"
    else:
        banner_style = "background:#fffbeb;border-left:4px solid #f59e0b;padding:0.6rem 1rem;border-radius:4px;"
        banner_text = f"⚠ <strong>WARN</strong> — {', '.join(s.value for s in result.shift_types)}"

    label = (
        f"🔴 Role shift: {result.severity.value}"
        if sev == Severity.SUPPRESS
        else f"⚠ Role shift: {result.severity.value}"
    )

    with st.expander(label, expanded=expanded):
        st.markdown(
            f'<div style="{banner_style}">{banner_text}</div>',
            unsafe_allow_html=True,
        )

        # Signal table
        st.markdown("**Signal breakdown**")
        _action_pill = {
            "OK": '<span style="color:#22c55e;font-weight:700">✓ OK</span>',
            "WARN": '<span style="color:#f59e0b;font-weight:700">⚠ WARN</span>',
            "SUPPRESS": '<span style="color:#ef4444;font-weight:700">🔴 SUPPRESS</span>',
        }

        rows_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.88rem'>"
            "<tr style='border-bottom:1px solid #e5e7eb'>"
            "<th style='text-align:left;padding:4px 8px'>Stat</th>"
            "<th style='text-align:right;padding:4px 8px'>L3</th>"
            "<th style='text-align:right;padding:4px 8px'>L30</th>"
            "<th style='text-align:right;padding:4px 8px'>σ</th>"
            "<th style='text-align:left;padding:4px 8px'>Action</th>"
            "</tr>"
        )
        for sig in result.signals:
            rows_html += (
                f"<tr style='border-bottom:1px solid #f3f4f6'>"
                f"<td style='padding:4px 8px'>{sig.stat}</td>"
                f"<td style='text-align:right;padding:4px 8px'>{sig.recent:.1f}</td>"
                f"<td style='text-align:right;padding:4px 8px'>{sig.baseline:.1f}</td>"
                f"<td style='text-align:right;padding:4px 8px'>{sig.z_score:+.2f}</td>"
                f"<td style='padding:4px 8px'>{_action_pill.get(sig.action, sig.action)}</td>"
                f"</tr>"
            )
        rows_html += "</table>"
        st.markdown(rows_html, unsafe_allow_html=True)

        if result.suppressed_stats:
            sup_str = ", ".join(f"**{s}**" for s in result.suppressed_stats)
            st.caption(
                f"NO_CALL applied to: {sup_str}. "
                f"These props are unreliable until the role transition stabilises. "
                f"Recommended last_n={result.recommended_last_n}."
            )
        elif result.severity == Severity.WARN:
            st.caption(
                f"Confidence reduced by {result.confidence_penalty:.0%} across all props. "
                f"Recommended last_n={result.recommended_last_n}."
            )


# ── Stat card with suppression awareness ─────────────────────────────────────


def apply_stat_card_suppression(
    stat: str,
    projection: float,
    line: float | None,
    call: str,
    result: RoleShiftResult,
) -> None:
    """Render a single stat card that respects role-shift severity.

    - SUPPRESS for this stat: red border, NO_CALL label, "role shift" sub-label.
    - WARN: amber border, -20% confidence note.
    - Clean: normal blue border with MORE/LESS call.
    """
    is_suppressed = (
        result.severity == Severity.SUPPRESS and stat in result.suppressed_stats
    )
    is_warned = result.severity == Severity.WARN

    if is_suppressed:
        border_color = _BORDER_SUPPRESS
        call_display = "NO_CALL"
        call_style = "color:#ef4444;font-weight:700"
        sub_label = '<span style="color:#ef4444;font-size:0.75rem">⚠ role shift</span>'
    elif is_warned:
        border_color = _BORDER_WARN
        call_display = call
        call_style = "color:#f59e0b;font-weight:700" if call not in ("MORE", "LESS") else (
            "color:#22c55e;font-weight:700" if call == "MORE" else "color:#ef4444;font-weight:700"
        )
        sub_label = '<span style="color:#f59e0b;font-size:0.75rem">−20% confidence</span>'
    else:
        border_color = _BORDER_OK
        call_display = call
        call_style = (
            "color:#22c55e;font-weight:700" if call == "MORE"
            else "color:#ef4444;font-weight:700" if call == "LESS"
            else "color:#6b7280;font-weight:700"
        )
        sub_label = ""

    line_str = f"line {line:g}" if line is not None else "no line"

    st.markdown(
        f'<div style="border:2px solid {border_color};border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem">'
        f'<div style="font-size:0.8rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em">'
        f'{stat}</div>'
        f'<div style="font-size:1.5rem;font-weight:700;margin:0.2rem 0">{projection:.1f}</div>'
        f'<div style="font-size:0.85rem;color:#9ca3af">{line_str}</div>'
        f'<div style="{call_style};font-size:0.9rem;margin-top:0.25rem">{call_display}</div>'
        f'{sub_label}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar summary ───────────────────────────────────────────────────────────


def _format_shift_types(result: RoleShiftResult) -> str:
    """Human-readable shift type list (excluding MULTI_ROLE_SHIFT meta-tag)."""
    names = [
        s.value.replace("_", " ").title()
        for s in result.shift_types
        if s.value != "MULTI_ROLE_SHIFT"
    ]
    return ", ".join(names) if names else "—"


def _peak_signal(result: RoleShiftResult):
    """Return the SignalResult with the largest |z|."""
    if not result.signals:
        return None
    return max(result.signals, key=lambda s: abs(s.z_score))


def render_sidebar_role_shift_summary(
    roster_results: dict[str, RoleShiftResult],
) -> None:
    """Enhanced sidebar widget surfacing role-shift state for the whole roster.

    Header: counts (✓ clear / ⚠ warn / 🔴 suppress)
    Body:   one card per flagged player with severity stripe, top signal σ,
            shift type, suppressed stats. Sorted SUPPRESS-first.
    Footer: "All clear" line when no flags fired.
    """
    if not roster_results:
        return

    n_total = len(roster_results)
    n_suppress = sum(
        1 for r in roster_results.values() if r.severity == Severity.SUPPRESS
    )
    n_warn = sum(1 for r in roster_results.values() if r.severity == Severity.WARN)
    n_clear = n_total - n_suppress - n_warn

    # Header with section title + count chips
    st.markdown(
        '<div style="margin-top:1rem;margin-bottom:0.4rem">'
        '<p class="hl-section" style="margin:0">Role alerts</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    chips = (
        f'<span class="hl-pill hl-pill-quiet" style="margin-right:4px">'
        f'{n_total} TRACKED</span>'
    )
    if n_suppress:
        chips += (
            f'<span class="hl-pill hl-pill-less" style="margin-right:4px">'
            f'🔴 {n_suppress} SUPPRESS</span>'
        )
    if n_warn:
        chips += (
            f'<span class="hl-pill hl-pill-warn" style="margin-right:4px">'
            f'⚠ {n_warn} WARN</span>'
        )
    if n_clear and not (n_suppress or n_warn):
        chips += (
            f'<span class="hl-pill hl-pill-more">'
            f'✓ {n_clear} CLEAR</span>'
        )
    st.markdown(
        f'<div style="margin-bottom:0.6rem;line-height:1.9">{chips}</div>',
        unsafe_allow_html=True,
    )

    flagged = {
        name: res
        for name, res in roster_results.items()
        if res.shift_detected
    }

    if not flagged:
        st.markdown(
            '<div style="font-size:0.82rem;color:#6b7280;'
            'padding:0.5rem 0.6rem;border-left:2px solid rgba(61,220,151,0.4);'
            'background:rgba(61,220,151,0.04);border-radius:3px">'
            '✓ No role transitions detected. RACE calls are valid across the roster.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Sort: SUPPRESS first (by max |z| desc), then WARN (by max |z| desc)
    def _sort_key(item: tuple[str, RoleShiftResult]) -> tuple[int, float]:
        _name, res = item
        sev_rank = 0 if res.severity == Severity.SUPPRESS else 1
        peak = _peak_signal(res)
        return (sev_rank, -abs(peak.z_score) if peak else 0.0)

    for name, res in sorted(flagged.items(), key=_sort_key):
        is_supp = res.severity == Severity.SUPPRESS
        accent = "#ff6b6b" if is_supp else "#f5b041"
        accent_bg = (
            "rgba(255,107,107,0.06)" if is_supp else "rgba(245,176,65,0.06)"
        )
        sev_label = "SUPPRESS" if is_supp else "WARN"
        icon = "🔴" if is_supp else "⚠"

        peak = _peak_signal(res)
        peak_line = ""
        if peak:
            sigma_color = (
                "#ff6b6b" if abs(peak.z_score) >= 2.0
                else "#f5b041" if abs(peak.z_score) >= 1.5
                else "#9ca3af"
            )
            peak_line = (
                f'<div style="font-size:0.72rem;color:#9ca3af;'
                f'margin-top:0.2rem;font-family:ui-monospace,monospace">'
                f'{peak.stat} · L3 {peak.recent:.1f} vs L30 {peak.baseline:.1f} · '
                f'<span style="color:{sigma_color};font-weight:700">'
                f'σ={peak.z_score:+.2f}'
                f'</span></div>'
            )

        shift_str = _format_shift_types(res)

        sup_block = ""
        if res.suppressed_stats:
            chips_sup = "".join(
                f'<span style="display:inline-block;background:rgba(255,107,107,0.12);'
                f'color:#ff8585;padding:1px 7px;border-radius:9999px;'
                f'font-size:0.68rem;font-weight:600;margin:2px 3px 0 0;'
                f'border:1px solid rgba(255,107,107,0.25)">{s}</span>'
                for s in res.suppressed_stats
            )
            sup_block = (
                f'<div style="margin-top:0.35rem">'
                f'<span style="font-size:0.68rem;color:#6b7280;'
                f'text-transform:uppercase;letter-spacing:0.06em;'
                f'margin-right:4px">NO_CALL</span>{chips_sup}'
                f'</div>'
            )

        rec_block = (
            f'<div style="font-size:0.7rem;color:#6b7280;margin-top:0.3rem">'
            f'recommended last_n={res.recommended_last_n}'
            f'</div>'
        )

        st.markdown(
            f'<div style="border-left:3px solid {accent};'
            f'background:{accent_bg};border-radius:4px;'
            f'padding:0.5rem 0.7rem;margin-bottom:0.45rem">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;gap:0.4rem">'
            f'<div style="font-weight:700;font-size:0.88rem;color:#e5e7eb">'
            f'{icon} {name}</div>'
            f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.06em;'
            f'color:{accent}">{sev_label}</div>'
            f'</div>'
            f'<div style="font-size:0.76rem;color:#9ca3af;margin-top:0.15rem">'
            f'{shift_str}</div>'
            f'{peak_line}'
            f'{sup_block}'
            f'{rec_block}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Feature extraction helper ─────────────────────────────────────────────────


def extract_features_from_games(games_df: "pd.DataFrame") -> dict[str, float]:  # type: ignore[name-defined]
    """Pull role-shift feature values from the latest row of a modeling frame.

    Returns an empty dict when the required columns are absent.
    """
    if games_df is None or games_df.empty:
        return {}

    row = games_df.iloc[-1]

    _keys = [
        "ast_l3", "ast_l30", "ast_std_l10",
        "pts_l3", "pts_l30", "pts_dev_s",
        "fga_l10", "fga_l30",
        "min_l3", "min_l30",
    ]
    out: dict[str, float] = {}
    for k in _keys:
        val = row.get(k) if hasattr(row, "get") else (row[k] if k in row.index else None)
        if val is not None:
            try:
                out[k] = float(val)
            except (TypeError, ValueError):
                pass
    return out
