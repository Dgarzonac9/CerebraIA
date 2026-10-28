# 🧠 CerebraIA

> **IA generativa para el apoyo al diagnóstico temprano de tumores cerebrales mediante síntesis y análisis de imágenes de resonancia magnética.**

En Colombia, el acceso a diagnóstico especializado por imágenes es limitado y desigual. **CerebraIA** combina redes generativas adversariales, modelos de lenguaje multimodal y agentes autónomos para generar imágenes sintéticas de MRI cerebral, analizarlas y producir reportes médicos estructurados — acercando tecnología de punta a quienes más la necesitan.

>Las imágenes generadas son sintéticas y tienen fines educativos. Este sistema no reemplaza el diagnóstico médico profesional.

---

## ✨ Capacidades principales

| Capacidad | Tecnología |
|-----------|-----------|
| Generación de imágenes MRI sintéticas | DCGAN (TensorFlow) |
| Análisis radiológico visual | Gemini 2.5 Flash (multimodal) |
| Evaluación diagnóstica | LLaMA 3.1 vía Groq |
| Orquestación autónoma de tareas | LangChain Agents |
| Trazabilidad de decisiones | AgentLogger (JSON) |

---

## 🖼️ Sistema en acción

### Dataset — 10 560 imágenes de MRI clasificadas en 4 categorías
*glioma · meningioma · pituitary · no_tumor*

![Dataset exploration](docs/images/dataset_exploration.png)

### Preprocesamiento — 330 batches listos para entrenar la GAN

![Preprocessing output](docs/images/preprocessing_output.png)

### Demo del agente — generación, análisis y diagnóstico en lenguaje natural

![Agent demo](docs/images/agent_demo.png)

---

## 🏗️ Arquitectura

```
Usuario (lenguaje natural)
        │
        ▼
 AgentExecutor (LangChain)
        │
        ├── generar_imagen_gan     →  DCGAN Generator (TensorFlow)
        ├── analizar_imagen_llm    →  Gemini 2.5 Flash (visión multimodal)
        ├── diagnostico_tumor_llm  →  LLaMA 3.1 · Groq
        ├── comparar_imagenes      →  DCGAN × N muestras
        └── generar_reporte_medico →  LLaMA 3.1 · Groq
                │
                ▼
        AgentLogger → trazabilidad completa en JSON
```

### DCGAN — Generador (4×4 → 64×64)

```
Ruido (100,) → Dense → 4×4×512
→ Conv2DTranspose(256) → 8×8
→ Conv2DTranspose(128) → 16×16
→ Conv2DTranspose(64)  → 32×32
→ Conv2DTranspose(3, tanh) → 64×64×3
```

---

## 🚀 Inicio rápido

> Recomendado: **Google Colab** con GPU T4 o superior.

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar la GAN

```python
from src.data import setup_kaggle, download_dataset, find_images, preprocess_images, build_tf_dataset
from src.gan  import DCGAN, GANCallback, train_gan

setup_kaggle("TU_USUARIO_KAGGLE", "TU_API_KEY")
download_dataset("/content/dataset")

images  = find_images("/content/dataset")
X_train = preprocess_images(images)
dataset = build_tf_dataset(X_train)

gan      = DCGAN(latent_dim=100)
callback = GANCallback(gan, save_path="/content/outputs", save_interval=10)
history  = train_gan(gan, dataset, epochs=200, callback=callback, models_path="/content/models")

gan.generator.save("/content/models/generator_final.keras")
```

### 3. Lanzar el agente

```python
from src.agent import init_llms, init_generator, build_agent, AgentLogger, chat_with_agent, LLM_GROQ

init_llms(google_api_key="...", groq_api_key="...")
init_generator("/content/models/generator_final.keras", latent_dim=100)

agent  = build_agent(LLM_GROQ)
logger = AgentLogger()

chat_with_agent(agent, logger, "Genera una imagen de MRI cerebral")
chat_with_agent(agent, logger, "Analízala y dame un diagnóstico para un paciente de 52 años con cefaleas intensas")
```

---

## 📁 Estructura del repositorio

```
CerebraIA/
├── src/
│   ├── gan.py           # Arquitectura DCGAN y bucle de entrenamiento
│   ├── data.py          # Carga, exploración y preprocesamiento del dataset
│   ├── agent.py         # Tools LangChain, agente autónomo y logger
│   ├── guardrails.py    # Advertencia, filtro de diagnósticos y manejo de fallos del LLM
│   └── __init__.py
├── eval/
│   ├── quality_metrics.py   # FID y KID
│   ├── classifier_check.py  # Verificación con clasificador
│   └── results/             # Resultados en JSON
├── tests/               # pytest con LLMs simulados
├── notebooks/
│   ├── proyecto_ia.py          # Código completo ejecutable en Google Colab
│   └── evaluacion_colab.ipynb  # Evaluación en Colab (GPU T4)
├── docs/
│   ├── architecture.md  # Descripción detallada de la arquitectura
│   └── images/          # Capturas del sistema en funcionamiento
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.10+
- Google Colab (GPU T4 recomendada)
- Cuenta en [Kaggle](https://www.kaggle.com) para el dataset
- API keys: [Google AI Studio](https://aistudio.google.com) y [Groq](https://console.groq.com)

### Secrets en Google Colab

| Secret | Descripción |
|--------|-------------|
| `GOOGLE_API_KEY` | API key de Gemini (Google AI Studio) |
| `GROQ_API_KEY` | API key de Groq |

---

## 📊 Dataset

[Brain Tumor MRI Dataset — Kaggle (ishans24)](https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset)

10 560 imágenes de resonancia magnética cerebral distribuidas en 4 clases:

| Clase | Imágenes |
|-------|----------|
| Glioma | 3 754 |
| Meningioma | 2 343 |
| Pituitary | 2 706 |
| No tumor | 1 757 |

---

## Evaluación y uso responsable

Una GAN que "se ve bien" no es evidencia de calidad. Esta capa mide la calidad de las imágenes sintéticas y limita lo que el agente puede afirmar.

### Qué se mide y por qué

| Componente | Qué hace | Cómo leerlo |
|------------|----------|-------------|
| **FID** (`eval/quality_metrics.py`) | Distancia de Fréchet entre las características InceptionV3 de imágenes reales y sintéticas. | Menor es mejor. Es sensible a la calidad y a la diversidad. |
| **KID** (`eval/quality_metrics.py`) | MMD² con kernel polinomial sobre las mismas características. | Menor es mejor. Es insesgado y más fiable con pocas muestras. |
| **Verificación con clasificador** (`eval/classifier_check.py`) | Una CNN entrenada solo con imágenes reales clasifica las sintéticas. | Confianza media baja: imágenes poco reconocibles. Distribución concentrada en una clase (entropía baja): posible *mode collapse*. |
| **Guardrails** (`src/guardrails.py`) | Advertencia fija, filtro de diagnósticos definitivos, error claro si el LLM falla o responde vacío, y registro en `AgentLogger`. | Cada activación queda en el campo `guardrails` de la traza JSON. |

Limitaciones:
- El generador es **incondicional** (recibe solo ruido), así que FID y KID se calculan solo de forma **global**, no por clase.
- El clasificador siempre elige una de las 4 clases. Sus predicciones sobre imágenes sintéticas son un indicador, no una etiqueta verdadera.
- Imágenes de 64x64 reescaladas a 299x299 para InceptionV3 no son comparables con FID publicados para otros datasets.

### Guardrails del agente

- Todo análisis, evaluación y reporte lleva la advertencia: *"Imagen sintética / uso educativo / no reemplaza diagnóstico médico"*.
- Frases como "el paciente tiene" o "diagnóstico confirmado" se reescriben como hallazgo probable.
- Si una llamada a Gemini o Groq falla o devuelve texto vacío, el agente responde con el error y no genera contenido.

### Cómo ejecutarlo

```bash
# Pruebas (sin GPU, sin claves, LLMs simulados)
pip install -r requirements-dev.txt
pytest

# Métricas (recomendado en Colab con GPU T4; ver notebooks/evaluacion_colab.ipynb)
python -m eval.quality_metrics  --weights models/generator_final.keras --data-dir /content/dataset --n-samples 5000 --out eval/results/metrics.json
python -m eval.classifier_check --weights models/generator_final.keras --data-dir /content/dataset --n-samples 1000
```

Las claves de API (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) se leen solo de variables de entorno o de los secrets de Colab.

### Resultados

| Métrica | Valor |
|---------|-------|
| FID global | pendiente de ejecutar en Colab |
| KID global (media ± desv.) | pendiente de ejecutar en Colab |
| Exactitud del clasificador en validación (reales) | pendiente de ejecutar en Colab |
| Confianza media sobre sintéticas | pendiente de ejecutar en Colab |
| Distribución de clases predichas (sintéticas) | pendiente de ejecutar en Colab |

---

## 👤 Autor

**Daniel Garzón** — [@Dgarzonac9](https://github.com/Dgarzonac9)
