# -*- coding: utf-8 -*-
"""
Permite importar src.agent sin TensorFlow, LangChain ni claves de API.
Los módulos pesados solo se reemplazan por stubs si no están instalados.
"""

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _stub(*names: str):
    for n in names:
        sys.modules.setdefault(n, MagicMock(name=n))


if _missing("tensorflow"):
    _stub("tensorflow", "tensorflow.keras", "tensorflow.keras.layers",
          "tensorflow.keras.preprocessing", "tensorflow.keras.preprocessing.image")
if _missing("matplotlib"):
    _stub("matplotlib", "matplotlib.pyplot")
if _missing("tqdm"):
    _stub("tqdm")
if _missing("PIL"):
    _stub("PIL", "PIL.Image")

if _missing("langchain_core") or _missing("langchain"):
    def _tool(fn):
        """Imita @tool: expone .func e .invoke(dict)."""
        return SimpleNamespace(name=fn.__name__, func=fn,
                               invoke=lambda kwargs=None, _f=fn: _f(**(kwargs or {})))

    core_tools = MagicMock(name="langchain_core.tools")
    core_tools.tool = _tool
    sys.modules["langchain_core.tools"] = core_tools
    _stub("langchain_core", "langchain_core.messages", "langchain_core.prompts",
          "langchain", "langchain.agents")
for _pkg in ("langchain_google_genai", "langchain_groq"):
    if _missing(_pkg):
        _stub(_pkg)


@pytest.fixture(autouse=True)
def _limpiar_eventos():
    from src.guardrails import drain_events
    drain_events()
    yield
    drain_events()


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """Módulo src.agent con LLMs simulados (sin llamadas a Gemini ni Groq) y estado limpio."""
    import numpy as np
    import src.agent as a

    monkeypatch.setattr(a, "REPORTS_PATH", str(tmp_path))
    monkeypatch.setattr(a, "STATE_LAST_IMAGE", {
        "array": np.zeros((64, 64, 3)), "path": str(tmp_path / "img.png"),
        "seed": 1, "timestamp": "t", "base64": "AAAA",
    })
    monkeypatch.setattr(a, "STATE_LAST_ANALYSIS", {
        "analisis": "Lesión hiperintensa frontal.", "imagen_path": "x", "timestamp": "t", "contexto": "",
    })
    monkeypatch.setattr(a, "STATE_SESSION_HISTORY", [])
    monkeypatch.setattr(a, "LLM_GEMINI", MagicMock(name="gemini"))
    monkeypatch.setattr(a, "LLM_GROQ", MagicMock(name="groq"))
    return a


def llm_responde(llm, texto):
    llm.invoke.return_value = SimpleNamespace(content=texto)
    llm.invoke.side_effect = None
