"""Anthropic Claude integration for Hooplytics.

Mirrors :mod:`hooplytics.openai_agent` for the Anthropic Messages API so the
Streamlit app can transparently use a Claude API key in place of an OpenAI key
for the Hooplytics Scout chat, the Roster Report prose, and the Player
Performance coaching narrative.

Shared assets (system prompts, grounding payload helpers, prose scrubbers,
chart parsing) live in :mod:`hooplytics.openai_agent` and are imported here so
prompt content stays in lockstep across providers. Only the transport layer is
provider-specific.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .openai_agent import (
    SYSTEM_PROMPT,
    STRICT_GROUNDED_SUFFIX,
    _EDGE_EXPLAINER_SYSTEM_PROMPT,
    _PERFORMANCE_REPORT_SYSTEM_PROMPT,
    _REPORT_SYSTEM_PROMPT,
    _SLATE_BRIEF_SYSTEM_PROMPT,
    _redact,
    _scrub_prose_leaks,
    _truncate_user_input,
    format_grounding_block,
)


# ── Model selection ──────────────────────────────────────────────────────────
# Ordered preference for auto-selecting a default Claude chat model. Earlier
# entries win when present in the user's available model list. The patterns
# match both the bare family name (e.g. ``claude-opus-4-7``) and dated
# snapshot ids (e.g. ``claude-opus-4-7-20260115``) the API exposes.
_PREFERRED_MODEL_PATTERNS: tuple[str, ...] = (
    # Claude 4.7 family (April 2026 flagship)
    r"^claude-opus-4-7",
    r"^claude-sonnet-4-7",
    r"^claude-haiku-4-7",
    # Claude 4.6 family
    r"^claude-opus-4-6",
    r"^claude-sonnet-4-6",
    r"^claude-haiku-4-6",
    # Claude 4.5 family
    r"^claude-opus-4-5",
    r"^claude-sonnet-4-5",
    r"^claude-haiku-4-5",
    # Claude 4 family
    r"^claude-opus-4",
    r"^claude-sonnet-4",
    r"^claude-haiku-4",
    # Claude 3.7 / 3.5 (legacy fallback)
    r"^claude-3-7-sonnet",
    r"^claude-3-5-sonnet",
    r"^claude-3-5-haiku",
    r"^claude-3-opus",
    r"^claude-3-",
)


def filter_chat_models(model_ids: Iterable[str]) -> list[str]:
    """Return Claude chat-capable models, ranked by family preference."""
    out: list[str] = []
    for mid in model_ids:
        if not isinstance(mid, str) or not mid:
            continue
        if not mid.startswith("claude-"):
            continue
        out.append(mid)

    def _rank(mid: str) -> tuple[int, str]:
        for i, pat in enumerate(_PREFERRED_MODEL_PATTERNS):
            if re.match(pat, mid):
                # Negative so newer dated snapshots within a family rank first.
                return (i, _negate_for_recency(mid))
        return (len(_PREFERRED_MODEL_PATTERNS), _negate_for_recency(mid))

    return sorted(set(out), key=_rank)


def _negate_for_recency(mid: str) -> str:
    """Sort key that prefers later dated snapshots within the same family."""
    # Lex-sort on the inverse so 20260115 sorts before 20251101.
    return "".join(chr(0x10FFFF - ord(c)) for c in mid)


def auto_select_model(model_ids: Iterable[str]) -> str | None:
    """Return the best available Claude model, preferring flagship families."""
    available = list(model_ids)
    for pattern in _PREFERRED_MODEL_PATTERNS:
        compiled = re.compile(pattern)
        matches = [m for m in available if compiled.match(m)]
        if matches:
            # Prefer the lexicographically latest dated snapshot within the family.
            matches.sort()
            return matches[-1]
    chat_only = filter_chat_models(available)
    return chat_only[0] if chat_only else None


# ── Client + model discovery ─────────────────────────────────────────────────
@dataclass
class AnthropicConnection:
    """Lightweight handle to an authenticated Anthropic client + model list."""

    client: Any
    models: list[str] = field(default_factory=list)
    default_model: str | None = None
    provider: str = "anthropic"


def _import_anthropic():
    try:
        import anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run "
            "`pip install anthropic` to enable the Claude provider."
        ) from exc
    return anthropic


def build_client(api_key: str) -> Any:
    """Construct an Anthropic client. Raises ``ValueError`` for empty keys."""
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Anthropic API key is empty.")
    anthropic = _import_anthropic()
    return anthropic.Anthropic(api_key=key)


def list_available_models(client: Any) -> list[str]:
    """Return all Claude model IDs visible to this key (unfiltered)."""
    try:
        resp = client.models.list()
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None
    ids: list[str] = []
    data = getattr(resp, "data", None) or []
    for item in data:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if mid:
            ids.append(str(mid))
    return ids


def connect(api_key: str) -> AnthropicConnection:
    """Validate a key, list models, and pick a default."""
    client = build_client(api_key)
    raw_models = list_available_models(client)
    chat_models = filter_chat_models(raw_models)
    default = auto_select_model(chat_models)
    return AnthropicConnection(
        client=client, models=chat_models, default_model=default
    )


# ── Message conversion ───────────────────────────────────────────────────────
def _extract_text(message: Any) -> str:
    """Concatenate text blocks from an Anthropic ``Message`` response."""
    content = getattr(message, "content", None)
    if not content:
        return ""
    chunks: list[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
        if btype != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n\n".join(chunks).strip()


def _coerce_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Sanitize chat history into Anthropic's strict alternating shape.

    Anthropic requires messages to start with a ``user`` turn and alternate
    user/assistant. We drop empty turns and collapse consecutive same-role
    turns by joining their content.
    """
    if not history:
        return []
    cleaned: list[dict[str, str]] = []
    for turn in history[-12:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role not in {"user", "assistant"} or not content:
            continue
        text = str(content).strip()
        if not text:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] = cleaned[-1]["content"] + "\n\n" + text
        else:
            cleaned.append({"role": role, "content": text})
    # Anthropic rejects history that starts with assistant; drop leading ones.
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


# ── Chat invocation ──────────────────────────────────────────────────────────
def chat_complete(
    *,
    connection: AnthropicConnection,
    model: str,
    user_message: str,
    grounding_payload: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    strict_grounded: bool = False,
    max_output_tokens: int = 2048,
) -> str:
    """Run a single Claude turn and return the assistant text."""
    if connection is None or connection.client is None:
        raise RuntimeError("No active Anthropic connection.")
    if not model:
        raise ValueError("No Claude model selected.")

    available_chat_models = filter_chat_models(connection.models or [])
    if available_chat_models and model not in available_chat_models:
        fallback = auto_select_model(available_chat_models) or available_chat_models[0]
        model = fallback

    system_text = SYSTEM_PROMPT + (STRICT_GROUNDED_SUFFIX if strict_grounded else "")
    if grounding_payload:
        system_text = system_text + "\n\n" + format_grounding_block(grounding_payload)

    messages = _coerce_history(history)
    user_text = _truncate_user_input(user_message)
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {
            "role": "user",
            "content": messages[-1]["content"] + "\n\n" + user_text,
        }
    else:
        messages.append({"role": "user", "content": user_text})

    client = connection.client
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_text,
            messages=messages,
        )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None

    text = _extract_text(resp)
    if text:
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "max_tokens":
            return (
                text
                + "\n\n_(Response cut off — the model hit the output token "
                "limit. Try a more specific question.)_"
            )
        return text

    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason == "refusal":
        return "(Response blocked by the Claude safety filter. Try rephrasing your question.)"
    return f"(The model returned no text. stop_reason={stop_reason!r})"


# ── Structured JSON generation (reports + performance) ───────────────────────
_JSON_INSTRUCTION_SUFFIX = (
    "\n\nReturn ONLY the JSON object. No prose outside the JSON, no markdown "
    "code fences, no leading or trailing commentary."
)


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating fences and prefix prose."""
    if not raw_text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fenced:
        raw_text = fenced.group(1)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw_text[start : end + 1])
            except Exception:
                return {}
        else:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _claude_json_call(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    grounding_payload: dict[str, Any],
    user_message: str,
    max_output_tokens: int,
) -> str:
    """Invoke Claude with a JSON-mandatory system prompt and return raw text.

    We rely on the system-prompt instruction to return only a JSON object
    rather than assistant prefill — newer Claude models reject prefill, and
    ``_parse_json_response`` already tolerates code fences or stray prose.
    """
    full_system = (
        system_prompt
        + _JSON_INSTRUCTION_SUFFIX
        + "\n\n"
        + format_grounding_block(grounding_payload)
    )
    messages = [{"role": "user", "content": user_message}]
    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=full_system,
        messages=messages,
    )
    return _extract_text(resp) or ""


def generate_report_sections(
    *,
    connection: AnthropicConnection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 4000,
) -> dict[str, Any]:
    """Generate the JSON report prose for the Roster Report PDF using Claude.

    Returns the same dict shape as :func:`hooplytics.openai_agent.generate_report_sections`
    so the PDF builder is provider-agnostic.
    """
    if connection is None or connection.client is None:
        raise RuntimeError("No active Anthropic connection.")
    if not model:
        raise ValueError("No Claude model selected.")

    available_chat_models = filter_chat_models(connection.models or [])
    if available_chat_models and model not in available_chat_models:
        model = auto_select_model(available_chat_models) or available_chat_models[0]

    user_message = (
        "Produce the JSON report sections for the roster in LOCAL CONTEXT. "
        "Return ONLY the JSON object."
    )

    try:
        raw_text = _claude_json_call(
            client=connection.client,
            model=model,
            system_prompt=_REPORT_SYSTEM_PROMPT,
            grounding_payload=grounding_payload,
            user_message=user_message,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None

    parsed = _parse_json_response(raw_text)
    if not parsed:
        return {"executive_summary": "", "slate_outlook": "", "matchups": {}, "players": {}}

    players_raw = parsed.get("players")
    players: dict[str, Any] = {}
    if isinstance(players_raw, dict):
        for k, v in players_raw.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, str) and v.strip():
                players[k] = {
                    "matchup": "",
                    "usage_trend": "",
                    "news": "",
                    "prediction": "",
                    "rationale": _scrub_prose_leaks(v.strip()),
                }
            elif isinstance(v, dict):
                players[k] = {
                    "matchup": _scrub_prose_leaks(str(v.get("matchup", "")).strip()),
                    "usage_trend": _scrub_prose_leaks(str(v.get("usage_trend", "")).strip()),
                    "news": _scrub_prose_leaks(str(v.get("news", "")).strip()),
                    "prediction": _scrub_prose_leaks(str(v.get("prediction", "")).strip()),
                    "rationale": _scrub_prose_leaks(str(v.get("rationale", "")).strip()),
                }

    matchups_raw = parsed.get("matchups")
    matchups: dict[str, Any] = {}
    if isinstance(matchups_raw, dict):
        for k, v in matchups_raw.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict):
                matchups[k] = {
                    "headline": _scrub_prose_leaks(str(v.get("headline", "")).strip()),
                    "narrative": _scrub_prose_leaks(str(v.get("narrative", "")).strip()),
                }
            elif isinstance(v, str) and v.strip():
                matchups[k] = {"headline": "", "narrative": _scrub_prose_leaks(v.strip())}

    return {
        "executive_summary": _scrub_prose_leaks(str(parsed.get("executive_summary", "")).strip()),
        "slate_outlook": _scrub_prose_leaks(str(parsed.get("slate_outlook", "")).strip()),
        "matchups": matchups,
        "players": players,
    }


def generate_performance_sections(
    *,
    connection: AnthropicConnection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 4000,
) -> dict[str, Any]:
    """Generate the JSON coaching prose for the Performance PDF using Claude."""
    if connection is None or connection.client is None:
        raise RuntimeError("No active Anthropic connection.")
    if not model:
        raise ValueError("No Claude model selected.")

    available_chat_models = filter_chat_models(connection.models or [])
    if available_chat_models and model not in available_chat_models:
        model = auto_select_model(available_chat_models) or available_chat_models[0]

    user_message = (
        "Produce the JSON coaching report sections for the roster in LOCAL "
        "CONTEXT. Return ONLY the JSON object."
    )

    try:
        raw_text = _claude_json_call(
            client=connection.client,
            model=model,
            system_prompt=_PERFORMANCE_REPORT_SYSTEM_PROMPT,
            grounding_payload=grounding_payload,
            user_message=user_message,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None

    parsed = _parse_json_response(raw_text)
    if not parsed:
        return {"roster_overview": "", "players": {}}

    players_raw = parsed.get("players")
    players: dict[str, Any] = {}
    if isinstance(players_raw, dict):
        for k, v in players_raw.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict):
                players[k] = {
                    "strengths": _scrub_prose_leaks(str(v.get("strengths", "")).strip()),
                    "growth_areas": _scrub_prose_leaks(str(v.get("growth_areas", "")).strip()),
                    "coaching_focus": _scrub_prose_leaks(str(v.get("coaching_focus", "")).strip()),
                    "matchup_context": _scrub_prose_leaks(str(v.get("matchup_context", "")).strip()),
                }
            elif isinstance(v, str) and v.strip():
                players[k] = {
                    "strengths": "",
                    "growth_areas": "",
                    "coaching_focus": _scrub_prose_leaks(v.strip()),
                    "matchup_context": "",
                }

    return {
        "roster_overview": _scrub_prose_leaks(str(parsed.get("roster_overview", "")).strip()),
        "players": players,
    }


# ════════════════════════════════════════════════════════════════════════════
# Short-form prose helpers (slate brief, edge explainer, news adjuster)
# ════════════════════════════════════════════════════════════════════════════

def _claude_text_call(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    max_output_tokens: int,
) -> str:
    """Run a single Claude text call and return the assistant text."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None
    return _extract_text(resp) or ""


def generate_slate_brief(
    *,
    connection: AnthropicConnection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 600,
) -> str:
    if connection is None or connection.client is None:
        raise RuntimeError("No active Anthropic connection.")
    if not model:
        raise ValueError("No Claude model selected.")
    available_chat_models = filter_chat_models(connection.models or [])
    if available_chat_models and model not in available_chat_models:
        model = auto_select_model(available_chat_models) or available_chat_models[0]

    user_message = (
        "Write tonight's Hooplytics Daily Slate Brief based on the LOCAL "
        "CONTEXT below.\n\n"
        + format_grounding_block(grounding_payload)
    )
    text = _claude_text_call(
        client=connection.client,
        model=model,
        system_prompt=_SLATE_BRIEF_SYSTEM_PROMPT,
        user_message=user_message,
        max_output_tokens=max_output_tokens,
    )
    return _scrub_prose_leaks(text)


def explain_edge(
    *,
    connection: AnthropicConnection,
    model: str,
    edge_row: dict[str, Any],
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 400,
) -> str:
    if connection is None or connection.client is None:
        raise RuntimeError("No active Anthropic connection.")
    if not model:
        raise ValueError("No Claude model selected.")
    available_chat_models = filter_chat_models(connection.models or [])
    if available_chat_models and model not in available_chat_models:
        model = auto_select_model(available_chat_models) or available_chat_models[0]

    edge_block = "EDGE ROW (the one to explain):\n```json\n" + json.dumps(
        edge_row, default=str, indent=2
    ) + "\n```"
    user_message = (
        "Explain the following edge row.\n\n"
        + edge_block
        + "\n\n"
        + format_grounding_block(grounding_payload)
    )
    text = _claude_text_call(
        client=connection.client,
        model=model,
        system_prompt=_EDGE_EXPLAINER_SYSTEM_PROMPT,
        user_message=user_message,
        max_output_tokens=max_output_tokens,
    )
    return _scrub_prose_leaks(text)


__all__ = [
    "AnthropicConnection",
    "auto_select_model",
    "build_client",
    "chat_complete",
    "connect",
    "explain_edge",
    "filter_chat_models",
    "generate_performance_sections",
    "generate_report_sections",
    "generate_slate_brief",
    "list_available_models",
]
