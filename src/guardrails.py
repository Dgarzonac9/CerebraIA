# -*- coding: utf-8 -*-
"""
Guardrails para los textos que produce el agente.

Sin dependencias de TensorFlow ni LangChain, para poder probarlos de forma aislada.
  - Advertencia fija en todo reporte.
  - Filtro de afirmaciones diagnósticas definitivas.
  - Llamadas al LLM que fallan o vuelven vacías → error claro (nunca texto inventado).
  - Registro de eventos para que el AgentLogger los trace.
"""

import re

DISCLAIMER = "Imagen sintética / uso educativo / no reemplaza diagnóstico médico"
_HEADER = f"AVISO: {DISCLAIMER}"


class LLMResponseError(RuntimeError):
    """El LLM falló o devolvió una respuesta vacía."""


# ─── Eventos (los consume AgentLogger vía drain_events) ──────────────────────

_EVENTS: list[dict] = []


def record_event(guardrail: str, detail: str):
    _EVENTS.append({"guardrail": guardrail, "detail": detail})


def drain_events() -> list[dict]:
    events = list(_EVENTS)
    _EVENTS.clear()
    return events


# ─── Filtro de diagnósticos definitivos ──────────────────────────────────────

# (patrón, reemplazo). Los reemplazos mantienen la frase gramaticalmente utilizable.
_DEFINITIVE_PATTERNS = [
    (re.compile(r"\b(?:el|la) paciente (?:tiene|padece|sufre de|sufre)\b", re.I),
     "los hallazgos son compatibles con (hallazgo probable)"),
    (re.compile(r"\bdiagn[óo]stico (?:confirmado|definitivo)\b", re.I),
     "hallazgo probable (no confirmado)"),
    (re.compile(r"\bse confirma (?:el|un) diagn[óo]stico(?: de)?\b", re.I),
     "los hallazgos sugieren, como hallazgo probable,"),
    (re.compile(r"\bconfirm(?:a|o|amos) (?:el|un) diagn[óo]stico(?: de)?\b", re.I),
     "los hallazgos sugieren, como hallazgo probable,"),
]


def sanitize_definitive_claims(text: str) -> tuple[str, list[str]]:
    """Reescribe afirmaciones definitivas como hallazgo probable. Devuelve (texto, frases detectadas)."""
    found: list[str] = []

    def make_repl(replacement):
        def repl(m):
            found.append(m.group(0))
            out = replacement
            return out[0].upper() + out[1:] if m.group(0)[0].isupper() else out
        return repl

    for pattern, replacement in _DEFINITIVE_PATTERNS:
        text = pattern.sub(make_repl(replacement), text)
    return text, found


# ─── Advertencia ─────────────────────────────────────────────────────────────

def add_disclaimer(text: str) -> str:
    """Antepone y agrega al final la advertencia fija (idempotente)."""
    if text.startswith(_HEADER):
        return text
    return f"{_HEADER}\n\n{text}\n\n---\n{DISCLAIMER}"


def apply_guardrails(text: str, source: str) -> str:
    """Filtro de frases definitivas + advertencia; registra el evento si el filtro actuó."""
    text, found = sanitize_definitive_claims(text)
    if found:
        record_event("frase_diagnostica_definitiva", f"{source}: reescrito {found}")
    record_event("advertencia_agregada", source)
    return add_disclaimer(text)


# ─── Llamadas seguras al LLM ─────────────────────────────────────────────────

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # algunos modelos devuelven partes de contenido
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def safe_invoke(llm, payload, source: str) -> str:
    """Invoca al LLM; si falla o devuelve vacío, registra el guardrail y lanza LLMResponseError."""
    if llm is None:
        record_event("llm_no_inicializado", source)
        raise LLMResponseError(f"{source}: el LLM no está inicializado (llama a init_llms).")
    try:
        response = llm.invoke(payload)
    except Exception as e:
        record_event("llm_error", f"{source}: {type(e).__name__}: {e}")
        raise LLMResponseError(f"{source}: falló la llamada al LLM ({type(e).__name__}).") from e

    text = _content_to_text(getattr(response, "content", None)).strip()
    if not text:
        record_event("respuesta_vacia", source)
        raise LLMResponseError(f"{source}: el LLM devolvió una respuesta vacía.")
    return text
