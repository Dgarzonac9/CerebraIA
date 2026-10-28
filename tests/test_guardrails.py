# -*- coding: utf-8 -*-
import glob
import os
from types import SimpleNamespace

import pytest

from src.guardrails import (
    DISCLAIMER, LLMResponseError, add_disclaimer, drain_events,
    safe_invoke, sanitize_definitive_claims,
)
from tests.conftest import llm_responde


# ─── Filtro de frases definitivas ────────────────────────────────────────────

@pytest.mark.parametrize("frase", [
    "El paciente tiene un glioma.",
    "la paciente padece meningioma.",
    "Diagnóstico confirmado: glioma.",
    "diagnostico definitivo de adenoma",
    "Se confirma el diagnóstico de meningioma.",
])
def test_filtro_detecta_y_reescribe(frase):
    nuevo, found = sanitize_definitive_claims(frase)
    assert found, frase
    assert "hallazgo probable" in nuevo.lower() or "compatibles" in nuevo.lower() or "sugieren" in nuevo.lower()
    assert nuevo.lower() != frase.lower()


def test_filtro_no_toca_texto_prudente():
    texto = "Los hallazgos sugieren un posible glioma; se recomienda correlación clínica."
    nuevo, found = sanitize_definitive_claims(texto)
    assert nuevo == texto and found == []


def test_filtro_conserva_mayuscula_inicial():
    nuevo, _ = sanitize_definitive_claims("El paciente tiene un glioma.")
    assert nuevo[0].isupper()
    assert "un glioma" in nuevo


# ─── Advertencia ─────────────────────────────────────────────────────────────

def test_advertencia_es_idempotente():
    una_vez = add_disclaimer("texto")
    assert add_disclaimer(una_vez) == una_vez
    assert una_vez.count(DISCLAIMER) == 2  # encabezado y pie


def test_reporte_incluye_advertencia_y_se_guarda(agent):
    llm_responde(agent.LLM_GROQ, "Reporte con hallazgos probables.")
    out = agent.generar_reporte_medico.func()

    files = glob.glob(os.path.join(agent.REPORTS_PATH, "reporte_medico_*.txt"))
    assert len(files) == 1 and "guardado" in out
    contenido = open(files[0], encoding="utf-8").read()
    assert contenido.startswith("AVISO: " + DISCLAIMER)
    assert contenido.rstrip().endswith(DISCLAIMER)
    agent.LLM_GROQ.invoke.assert_called_once()


def test_analisis_y_diagnostico_incluyen_advertencia(agent):
    llm_responde(agent.LLM_GEMINI, "Descripción de la imagen.")
    assert DISCLAIMER in agent.analizar_imagen_llm.func()
    llm_responde(agent.LLM_GROQ, "Impresión probable.")
    assert DISCLAIMER in agent.diagnostico_tumor_llm.func("cefalea")


def test_reporte_reescribe_diagnostico_definitivo(agent):
    llm_responde(agent.LLM_GROQ, "El paciente tiene un glioma. Diagnóstico confirmado.")
    agent.generar_reporte_medico.func()
    contenido = open(glob.glob(os.path.join(agent.REPORTS_PATH, "*.txt"))[0], encoding="utf-8").read()
    assert "el paciente tiene" not in contenido.lower()
    assert "diagnóstico confirmado" not in contenido.lower()
    nombres = [e["guardrail"] for e in drain_events()]
    assert "frase_diagnostica_definitiva" in nombres and "advertencia_agregada" in nombres


# ─── Respuestas vacías y fallos del LLM ──────────────────────────────────────

@pytest.mark.parametrize("vacio", ["", "   \n", None, []])
def test_respuesta_vacia_da_error_y_no_escribe_reporte(agent, vacio):
    llm_responde(agent.LLM_GROQ, vacio)
    out = agent.generar_reporte_medico.func()
    assert out.startswith("ERROR:") and "vacía" in out
    assert glob.glob(os.path.join(agent.REPORTS_PATH, "*.txt")) == []
    assert [e["guardrail"] for e in drain_events()] == ["respuesta_vacia"]


def test_excepcion_del_llm_da_error_claro(agent):
    agent.LLM_GEMINI.invoke.side_effect = TimeoutError("sin red")
    out = agent.analizar_imagen_llm.func()
    assert out.startswith("ERROR:") and "TimeoutError" in out
    assert agent.STATE_LAST_ANALYSIS["analisis"] == "Lesión hiperintensa frontal."  # no se pisó el estado
    assert "llm_error" in [e["guardrail"] for e in drain_events()]


def test_safe_invoke_sin_llm_inicializado():
    with pytest.raises(LLMResponseError):
        safe_invoke(None, "hola", "prueba")
    assert drain_events()[0]["guardrail"] == "llm_no_inicializado"


def test_safe_invoke_acepta_contenido_en_partes():
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=[{"text": "uno "}, {"text": "dos"}]))
    assert safe_invoke(llm, "x", "prueba") == "uno dos"


# ─── Integración con AgentLogger y chat_with_agent ───────────────────────────

def test_logger_registra_guardrail(agent):
    logger = agent.AgentLogger()
    ev = logger.log_guardrail("respuesta_vacia", "generar_reporte_medico")
    assert logger.guardrail_events == [ev]
    assert ev["guardrail"] == "respuesta_vacia" and ev["timestamp"]


def test_chat_reemplaza_respuesta_inventada_por_el_error(agent):
    llm_responde(agent.LLM_GROQ, "")

    class FakeExecutor:
        def invoke(self, _):
            resultado = agent.generar_reporte_medico.func()
            accion = SimpleNamespace(log="log", tool="generar_reporte_medico")
            return {"output": "Reporte listo, todo normal.", "intermediate_steps": [(accion, resultado)]}

    logger = agent.AgentLogger()
    salida = agent.chat_with_agent(FakeExecutor(), logger, "genera un reporte")

    assert salida.startswith("ERROR:") and "listo" not in salida
    nombres = [g["guardrail"] for g in logger.logs[0]["guardrails"]]
    assert "respuesta_vacia" in nombres and "respuesta_reemplazada_por_error" in nombres


def test_chat_filtra_frases_definitivas_en_respuesta_final(agent):
    class FakeExecutor:
        def invoke(self, _):
            return {"output": "El paciente tiene un meningioma.", "intermediate_steps": []}

    logger = agent.AgentLogger()
    salida = agent.chat_with_agent(FakeExecutor(), logger, "dime")
    assert "el paciente tiene" not in salida.lower()
    assert [g["guardrail"] for g in logger.logs[0]["guardrails"]] == ["frase_diagnostica_definitiva"]
