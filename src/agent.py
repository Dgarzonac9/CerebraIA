# -*- coding: utf-8 -*-
"""
Agente de IA para análisis de MRI cerebral.
Herramientas LangChain + Gemini (visión) + Groq (razonamiento médico).
"""

import os
import json
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from io import BytesIO
from datetime import datetime
from PIL import Image

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ─── Estado global de la sesión ───────────────────────────────────────────────

STATE_LAST_IMAGE:    dict | None = None
STATE_LAST_ANALYSIS: dict | None = None
STATE_SESSION_HISTORY: list[dict] = []

# LLMs (se inicializan con init_llms)
LLM_GEMINI = None
LLM_GROQ   = None
GAN_GENERATOR = None
LATENT_DIM    = 100
REPORTS_PATH  = "/content/drive/MyDrive/Practica_3_IA/reportes"


# ─── Inicialización ───────────────────────────────────────────────────────────

def init_llms(google_api_key: str, groq_api_key: str):
    """Inicializa Gemini y Groq."""
    global LLM_GEMINI, LLM_GROQ
    LLM_GEMINI = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.3,
    )
    LLM_GROQ = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        temperature=0.3,
    )
    print("LLMs inicializados: Gemini + Groq")


def init_generator(generator_path: str, latent_dim: int = 100):
    """Carga el generador del GAN entrenado."""
    global GAN_GENERATOR, LATENT_DIM
    GAN_GENERATOR = tf.keras.models.load_model(generator_path)
    LATENT_DIM    = latent_dim
    print(f"Generador cargado desde: {generator_path}")


# ─── Utilidades de imagen ─────────────────────────────────────────────────────

def image_to_base64(image_array: np.ndarray) -> str:
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(image_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_image(image_array: np.ndarray, filename: str) -> str:
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    path = os.path.join(REPORTS_PATH, filename)
    Image.fromarray(image_array).save(path)
    return path


# ─── Tools LangChain ──────────────────────────────────────────────────────────

@tool
def generar_imagen_gan(seed: int = None) -> str:
    """Genera una imagen sintética de MRI cerebral usando la GAN entrenada."""
    global STATE_LAST_IMAGE, STATE_SESSION_HISTORY

    seed = seed or int(np.random.randint(0, 10000))
    tf.random.set_seed(seed)
    noise     = tf.random.normal([1, LATENT_DIM])
    generated = GAN_GENERATOR(noise, training=False)
    img       = np.clip((generated[0].numpy() + 1) / 2, 0, 1)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mri_generated_{ts}_seed{seed}.png"
    path     = save_image(img, filename)

    STATE_LAST_IMAGE = {
        "array": img, "path": path,
        "seed": seed, "timestamp": ts,
        "base64": image_to_base64(img),
    }
    STATE_SESSION_HISTORY.append(
        {"accion": "generar_imagen_gan", "timestamp": ts, "seed": seed, "path": path}
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(img); plt.title(f"MRI Generada (seed={seed})"); plt.axis("off")
    plt.show()

    return f"Imagen generada.\n  Seed: {seed}\n  Guardada en: {path}"


@tool
def analizar_imagen_llm(descripcion_adicional: str = "") -> str:
    """Analiza la última imagen de MRI generada usando Gemini (visión)."""
    global STATE_LAST_IMAGE, STATE_LAST_ANALYSIS, STATE_SESSION_HISTORY

    if STATE_LAST_IMAGE is None:
        return "No hay imagen para analizar. Primero genera una imagen."

    prompt = (
        "Eres un radiólogo experto analizando una imagen de resonancia magnética (MRI) cerebral. "
        "Proporciona descripción general, hallazgos relevantes e impresión inicial. "
        "Imagen generada por IA con fines educativos."
    )
    if descripcion_adicional:
        prompt += f"\nContexto adicional: {descripcion_adicional}"

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{STATE_LAST_IMAGE['base64']}"
        }},
    ])
    analysis = LLM_GEMINI.invoke([message]).content

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_LAST_ANALYSIS = {
        "analisis": analysis,
        "imagen_path": STATE_LAST_IMAGE["path"],
        "timestamp": ts,
        "contexto": descripcion_adicional,
    }
    STATE_SESSION_HISTORY.append({
        "accion": "analizar_imagen_llm", "timestamp": ts,
        "imagen_analizada": STATE_LAST_IMAGE["path"],
        "resumen": analysis[:200] + "...",
    })
    return f"**ANÁLISIS DE IMAGEN MRI**\n\n{analysis}"


@tool
def diagnostico_tumor_llm(sintomas_paciente: str = "") -> str:
    """Evaluación diagnóstica basada en el análisis previo de la imagen."""
    global STATE_LAST_ANALYSIS, STATE_SESSION_HISTORY

    if STATE_LAST_ANALYSIS is None:
        return "No hay análisis previo. Primero analiza una imagen."

    prompt = (
        f"Eres un neurólogo especialista en neuro-oncología.\n\n"
        f"ANÁLISIS RADIOLÓGICO:\n{STATE_LAST_ANALYSIS['analisis']}\n\n"
        f"SÍNTOMAS:\n{sintomas_paciente or 'No especificados'}"
    )
    diagnostico = LLM_GROQ.invoke(prompt).content

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_SESSION_HISTORY.append({
        "accion": "diagnostico_tumor_llm", "timestamp": ts,
        "sintomas": sintomas_paciente, "resumen": diagnostico[:200] + "...",
    })
    return f"**EVALUACIÓN DIAGNÓSTICA**\n\n{diagnostico}"


@tool
def comparar_imagenes(num_imagenes: int = 4) -> str:
    """Genera múltiples imágenes MRI para observar variabilidad de la GAN."""
    global STATE_SESSION_HISTORY

    n = max(2, min(9, num_imagenes))
    images, seeds = [], []

    for _ in range(n):
        seed = int(np.random.randint(0, 10000))
        seeds.append(seed)
        tf.random.set_seed(seed)
        noise     = tf.random.normal([1, LATENT_DIM])
        generated = GAN_GENERATOR(noise, training=False)
        images.append((generated[0].numpy() + 1) / 2)

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for i, ax in enumerate(np.array(axes).flatten()):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f"MRI {i + 1} (seed={seeds[i]})")
        ax.axis("off")

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(REPORTS_PATH, f"comparacion_{ts}.png")
    plt.savefig(save_path, dpi=150); plt.show()

    STATE_SESSION_HISTORY.append({
        "accion": "comparar_imagenes", "timestamp": ts,
        "seeds": seeds, "path": save_path,
    })
    return f"Comparación generada y guardada en {save_path}"


@tool
def generar_reporte_medico(
    nombre_paciente: str = "Paciente Anónimo",
    edad: int = 0,
    motivo_consulta: str = "",
) -> str:
    """Genera un reporte médico estructurado en texto plano."""
    global STATE_LAST_ANALYSIS, STATE_SESSION_HISTORY

    if STATE_LAST_ANALYSIS is None:
        return "No hay análisis disponible. Primero analiza una imagen."

    prompt = (
        f"Genera un reporte médico profesional basado en este análisis:\n"
        f"{STATE_LAST_ANALYSIS['analisis']}"
    )
    reporte  = LLM_GROQ.invoke(prompt).content
    filename = f"reporte_medico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path     = os.path.join(REPORTS_PATH, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(reporte)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_SESSION_HISTORY.append({
        "accion": "generar_reporte_medico",
        "paciente": nombre_paciente,
        "path": path,
    })
    return f"Reporte médico guardado en {path}"


# ─── Registro de tools ────────────────────────────────────────────────────────

TOOLS = [
    generar_imagen_gan,
    analizar_imagen_llm,
    diagnostico_tumor_llm,
    comparar_imagenes,
    generar_reporte_medico,
]


# ─── Agente LangChain ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un asistente médico especializado en análisis de imágenes de resonancia magnética (MRI) cerebral para detección de tumores.

Tu rol es ayudar a profesionales de la salud a:
1. Generar imágenes sintéticas de MRI cerebral para entrenamiento y estudio
2. Analizar imágenes de MRI identificando posibles anomalías
3. Proporcionar evaluaciones diagnósticas preliminares
4. Generar reportes médicos estructurados

HERRAMIENTAS DISPONIBLES:
1. generar_imagen_gan
2. analizar_imagen_llm
3. diagnostico_tumor_llm
4. comparar_imagenes
5. generar_reporte_medico

IMPORTANTE:
- Las imágenes son sintéticas y educativas
- No reemplaza diagnóstico médico real
- Responde siempre en español
"""


def build_agent(llm) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS,
        verbose=True, handle_parsing_errors=True, max_iterations=10,
    )


# ─── Logger de trazabilidad ───────────────────────────────────────────────────

class AgentLogger:
    """Registra y exporta el historial de interacciones del agente."""

    def __init__(self):
        self.logs: list[dict] = []

    def log_interaction(self, user_input, agent_response, tools_used, reasoning, tool_outputs):
        self.logs.append({
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_input":      user_input,
            "agent_response":  agent_response,
            "tools_used":      tools_used,
            "reasoning_steps": reasoning,
            "tool_outputs":    tool_outputs,
        })

    def show_summary(self):
        print("\n" + "=" * 60)
        print("RESUMEN DE TRAZABILIDAD DEL AGENTE")
        print("=" * 60)
        for i, entry in enumerate(self.logs, 1):
            print(f"\nInteracción #{i}  [{entry['timestamp']}]")
            print(f"  Usuario : {entry['user_input']}")
            print(f"  Agente  : {entry['agent_response']}")
            print(f"  Tools   : {', '.join(entry['tools_used']) or '—'}")

    def save_to_json(self, filename: str = "agent_trace.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)
        print(f"Trazabilidad guardada en: {filename}")


# ─── Interfaz de chat ─────────────────────────────────────────────────────────

def chat_with_agent(agent_executor: AgentExecutor, logger: AgentLogger, user_input: str) -> str:
    print(f"\n{'=' * 60}\nUsuario: {user_input}\n{'=' * 60}")
    try:
        response = agent_executor.invoke({"input": user_input, "chat_history": []})
        output   = response["output"]
        steps    = response.get("intermediate_steps", [])

        reasoning, tool_calls, tool_results = [], [], []
        for action, result in steps:
            reasoning.append(action.log)
            tool_calls.append(action.tool)
            tool_results.append(result)

        logger.log_interaction(user_input, output, tool_calls, reasoning, tool_results)
        print(f"\n{'-' * 60}\nAgente:\n{output}\n{'-' * 60}")
        return output

    except Exception as e:
        err = f"Error: {e}"
        print(err)
        return err


def interactive_session(agent_executor: AgentExecutor, logger: AgentLogger):
    print("\n" + "=" * 60)
    print("SISTEMA DE ANÁLISIS DE MRI CEREBRAL")
    print("Escribe 'salir' para terminar, 'resumen' para ver trazabilidad.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nTú: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                logger.show_summary()
                print("\n¡Hasta luego!")
                break
            if user_input.lower() == "resumen":
                logger.show_summary()
                continue
            chat_with_agent(agent_executor, logger, user_input)

        except KeyboardInterrupt:
            logger.show_summary()
            print("\nSesión interrumpida.")
            break
