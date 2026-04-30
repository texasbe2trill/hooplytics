"""Provider-agnostic dispatcher for Hooplytics AI features.

Routes calls to either :mod:`hooplytics.openai_agent` or
:mod:`hooplytics.anthropic_agent` based on the active connection's
``provider`` attribute. The Streamlit app imports from this module so the
underlying provider can be swapped at runtime without any call-site churn.

Shared helpers (grounding payload, evidence chips, chart parsing) are
re-exported from :mod:`hooplytics.openai_agent` for convenience — they are
provider-independent.
"""

from __future__ import annotations

from typing import Any, Iterable, Union

from . import anthropic_agent, openai_agent
from .anthropic_agent import AnthropicConnection
from .openai_agent import (
    OpenAIConnection,
    build_grounding_payload,
    evidence_chips,
    format_grounding_block,
    parse_chart_blocks,
)


PROVIDERS: tuple[str, ...] = ("openai", "anthropic")
PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
}

Connection = Union[OpenAIConnection, AnthropicConnection]


def _provider_for(connection: Connection) -> str:
    return getattr(connection, "provider", None) or (
        "anthropic" if isinstance(connection, AnthropicConnection) else "openai"
    )


def connect(api_key: str, provider: str) -> Connection:
    """Validate a key for the given provider and return a typed connection."""
    if provider == "openai":
        return openai_agent.connect(api_key)
    if provider == "anthropic":
        return anthropic_agent.connect(api_key)
    raise ValueError(f"Unknown AI provider: {provider!r}")


def filter_chat_models(model_ids: Iterable[str], provider: str) -> list[str]:
    if provider == "openai":
        return openai_agent.filter_chat_models(model_ids)
    if provider == "anthropic":
        return anthropic_agent.filter_chat_models(model_ids)
    raise ValueError(f"Unknown AI provider: {provider!r}")


def auto_select_model(model_ids: Iterable[str], provider: str) -> str | None:
    if provider == "openai":
        return openai_agent.auto_select_model(model_ids)
    if provider == "anthropic":
        return anthropic_agent.auto_select_model(model_ids)
    raise ValueError(f"Unknown AI provider: {provider!r}")


def chat_complete(
    *,
    connection: Connection,
    model: str,
    user_message: str,
    grounding_payload: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    strict_grounded: bool = False,
    max_output_tokens: int = 2048,
) -> str:
    """Run a single chat turn against whichever provider the connection uses."""
    if _provider_for(connection) == "anthropic":
        return anthropic_agent.chat_complete(
            connection=connection,  # type: ignore[arg-type]
            model=model,
            user_message=user_message,
            grounding_payload=grounding_payload,
            history=history,
            strict_grounded=strict_grounded,
            max_output_tokens=max_output_tokens,
        )
    return openai_agent.chat_complete(
        connection=connection,  # type: ignore[arg-type]
        model=model,
        user_message=user_message,
        grounding_payload=grounding_payload,
        history=history,
        strict_grounded=strict_grounded,
        max_output_tokens=max_output_tokens,
    )


def generate_report_sections(
    *,
    connection: Connection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 4000,
) -> dict[str, Any]:
    if _provider_for(connection) == "anthropic":
        return anthropic_agent.generate_report_sections(
            connection=connection,  # type: ignore[arg-type]
            model=model,
            grounding_payload=grounding_payload,
            max_output_tokens=max_output_tokens,
        )
    return openai_agent.generate_report_sections(
        connection=connection,  # type: ignore[arg-type]
        model=model,
        grounding_payload=grounding_payload,
        max_output_tokens=max_output_tokens,
    )


def generate_slate_brief(
    *,
    connection: Connection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 600,
) -> str:
    if _provider_for(connection) == "anthropic":
        return anthropic_agent.generate_slate_brief(
            connection=connection,  # type: ignore[arg-type]
            model=model,
            grounding_payload=grounding_payload,
            max_output_tokens=max_output_tokens,
        )
    return openai_agent.generate_slate_brief(
        connection=connection,  # type: ignore[arg-type]
        model=model,
        grounding_payload=grounding_payload,
        max_output_tokens=max_output_tokens,
    )


def explain_edge(
    *,
    connection: Connection,
    model: str,
    edge_row: dict[str, Any],
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 400,
) -> str:
    if _provider_for(connection) == "anthropic":
        return anthropic_agent.explain_edge(
            connection=connection,  # type: ignore[arg-type]
            model=model,
            edge_row=edge_row,
            grounding_payload=grounding_payload,
            max_output_tokens=max_output_tokens,
        )
    return openai_agent.explain_edge(
        connection=connection,  # type: ignore[arg-type]
        model=model,
        edge_row=edge_row,
        grounding_payload=grounding_payload,
        max_output_tokens=max_output_tokens,
    )


def generate_performance_sections(
    *,
    connection: Connection,
    model: str,
    grounding_payload: dict[str, Any],
    max_output_tokens: int = 4000,
) -> dict[str, Any]:
    if _provider_for(connection) == "anthropic":
        return anthropic_agent.generate_performance_sections(
            connection=connection,  # type: ignore[arg-type]
            model=model,
            grounding_payload=grounding_payload,
            max_output_tokens=max_output_tokens,
        )
    return openai_agent.generate_performance_sections(
        connection=connection,  # type: ignore[arg-type]
        model=model,
        grounding_payload=grounding_payload,
        max_output_tokens=max_output_tokens,
    )


__all__ = [
    "AnthropicConnection",
    "Connection",
    "OpenAIConnection",
    "PROVIDERS",
    "PROVIDER_LABELS",
    "auto_select_model",
    "build_grounding_payload",
    "chat_complete",
    "connect",
    "evidence_chips",
    "explain_edge",
    "filter_chat_models",
    "format_grounding_block",
    "generate_performance_sections",
    "generate_report_sections",
    "generate_slate_brief",
    "parse_chart_blocks",
]
