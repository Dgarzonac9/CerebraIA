# Práctica 3 IA — Generative AI: GANs, LLMs y Agentic AI

Sistema de IA generativa para síntesis y análisis de imágenes de **MRI cerebral**, orientado a la detección de tumores. Combina tres componentes principales:

1. **DCGAN** — genera imágenes sintéticas de resonancia magnética  
2. **LLMs** (Gemini + Groq vía LangChain) — analizan imágenes y razonan sobre diagnósticos  
3. **Agente autónomo** — orquesta las herramientas para completar flujos de trabajo médicos

> ⚠️ Las imágenes generadas son **sintéticas y educativas**. Este sistema no reemplaza el diagnóstico médico profesional.

---

## Estructura del repositorio

```
practica_3_ia/
├── src/
│   ├── __init__.py      # Exports del paquete
│   ├── gan.py           # Arquitectura DCGAN + entrenamiento
│   ├── data.py          # Carga y preprocesamiento del dataset
│   └── agent.py         # Tools LangChain + agente + logger
├── notebooks/
│   └── practica_3_ia.ipynb   # Notebook original para Google Colab
├── docs/
│   └── architecture.md  # Descripción detallada de la arquitectura
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

> El proyecto está pensado para ejecutarse en **Google Colab** con GPU.

---

## Uso rápido

### 1. Entrenamiento de la GAN

```python
from src.data import setup_kaggle, download_dataset, find_images, preprocess_images, build_tf_dataset
from src.gan  import DCGAN, GANCallback, train_gan

# Descargar dataset
setup_kaggle("TU_USUARIO", "TU_API_KEY")
download_dataset("/content/dataset")

# Preprocesar
images    = find_images("/content/dataset")
X_train   = preprocess_images(images)
dataset   = build_tf_dataset(X_train)

# Entrenar
gan      = DCGAN(latent_dim=100)
callback = GANCallback(gan, save_path="/content/outputs", save_interval=10)
history  = train_gan(gan, dataset, epochs=200, callback=callback, models_path="/content/models")

# Guardar modelos
gan.generator.save("/content/models/generator_final.keras")
gan.discriminator.save("/content/models/discriminator_final.keras")
```

### 2. Agente de análisis

```python
from src.agent import init_llms, init_generator, build_agent, AgentLogger, chat_with_agent

init_llms(google_api_key="...", groq_api_key="...")
init_generator("/content/models/generator_final.keras", latent_dim=100)

from src.agent import LLM_GROQ
agent_executor = build_agent(LLM_GROQ)
logger         = AgentLogger()

# Flujo completo
chat_with_agent(agent_executor, logger, "Genera una imagen de MRI")
chat_with_agent(agent_executor, logger, "Analiza la imagen generada")
chat_with_agent(agent_executor, logger,
    "Dame un diagnóstico: hombre 52 años, cefaleas intensas 3 meses")
```

---

## Arquitectura del agente

El agente dispone de **5 herramientas**:

| Tool | Descripción |
|------|-------------|
| `generar_imagen_gan` | Crea una imagen de MRI sintética con la GAN |
| `analizar_imagen_llm` | Análisis radiológico con Gemini (multimodal) |
| `diagnostico_tumor_llm` | Evaluación diagnóstica con Groq/LLaMA |
| `comparar_imagenes` | Genera múltiples imágenes para explorar variabilidad |
| `generar_reporte_medico` | Redacta un reporte clínico estructurado |

---

## Requisitos

- Python 3.10+
- Google Colab (recomendado, GPU T4 o superior)
- Cuenta en [Kaggle](https://www.kaggle.com) (dataset)
- API Keys: Google Gemini y Groq

---

## Variables de entorno / Secrets de Colab

En Colab, configura los siguientes *Secrets*:

| Secret | Descripción |
|--------|-------------|
| `GOOGLE_API_KEY` | API key de Google AI Studio (Gemini) |
| `GROQ_API_KEY` | API key de Groq |

---

## Capturas del sistema

### Exploración del dataset
10 560 imágenes distribuidas en 4 clases: glioma, meningioma, pituitary y no_tumor.

![Dataset exploration](docs/images/dataset_exploration.png)

### Preprocesamiento
10 560 imágenes redimensionadas a 64×64×3 y normalizadas a [-1, 1] en ~55 minutos, formando 330 batches de 32.

![Preprocessing output](docs/images/preprocessing_output.png)

### Demo del agente
El agente genera una imagen MRI sintética con la GAN (seed=123) y la analiza con los LLMs.

![Agent demo](docs/images/agent_demo.png)

---

## Dataset

[Brain Tumor Dataset — Kaggle (ishans24)](https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset)

Imágenes de MRI cerebral clasificadas en carpetas por tipo de tumor.
