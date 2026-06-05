# Arquitectura del Sistema

## 1. DCGAN (Deep Convolutional GAN)

### Generador
Transforma un vector de ruido de dimensión `latent_dim=100` en imágenes de `64×64×3` mediante capas `Conv2DTranspose`:

```
Dense(4×4×512) → Reshape(4,4,512)
→ Conv2DTranspose(256) → 8×8
→ Conv2DTranspose(128) → 16×16
→ Conv2DTranspose(64)  → 32×32
→ Conv2DTranspose(3, activation=tanh) → 64×64×3
```

### Discriminador
Clasifica imágenes reales vs. falsas mediante capas `Conv2D`:

```
Conv2D(64)  → 32×32
Conv2D(128) → 16×16
Conv2D(256) → 8×8
Conv2D(512) → 4×4
Flatten → Dense(1)
```

### Entrenamiento
- Optimizador: Adam (lr=0.0002, β₁=0.5)
- Pérdida: Binary Cross-Entropy
- Épocas: 200 | Batch: 32 | Latent dim: 100

---

## 2. Pipeline del Agente

```
Usuario
   │
   ▼
AgentExecutor (LangChain)
   │
   ├── generar_imagen_gan   ──► GAN Generator (TensorFlow)
   ├── analizar_imagen_llm  ──► Gemini 2.5 Flash (visión multimodal)
   ├── diagnostico_tumor_llm──► Groq / LLaMA-3.1-8b
   ├── comparar_imagenes    ──► GAN Generator × N
   └── generar_reporte_medico──► Groq / LLaMA-3.1-8b
         │
         ▼
   AgentLogger (trazabilidad JSON)
```

### Estado global
El agente mantiene un estado de sesión compartido entre herramientas:

- `STATE_LAST_IMAGE` — última imagen generada (array + base64 + path)
- `STATE_LAST_ANALYSIS` — último análisis radiológico
- `STATE_SESSION_HISTORY` — historial de acciones de la sesión
