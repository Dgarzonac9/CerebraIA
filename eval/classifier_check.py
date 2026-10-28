# -*- coding: utf-8 -*-
"""
Verificación con clasificador: ¿la GAN produce imágenes reconocibles y diversas?

1. Entrena una CNN pequeña SOLO con imágenes reales (split train/val estratificado)
   y reporta la exactitud en validación.
2. Clasifica N imágenes sintéticas y reporta la distribución de clases predichas,
   la confianza media y la entropía normalizada de esa distribución.

Cómo leerlo: el clasificador siempre elige una de las 4 clases, así que una confianza
media baja sugiere imágenes poco reconocibles, y una distribución muy concentrada
en una clase (entropía normalizada baja) sugiere mode collapse. Es un indicador,
no una prueba: el generador es incondicional y las clases predichas no son
"etiquetas verdaderas" de lo generado.

Uso:
    python -m eval.classifier_check --weights models/generator_final.keras \
        --data-dir /content/dataset --n-samples 1000 --epochs 15
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np


# ─── Resumen de predicciones (numpy puro) ────────────────────────────────────

def summarize_predictions(probs: np.ndarray, class_names: list[str], collapse_threshold: float = 0.8) -> dict:
    """probs: (N, K) probabilidades softmax sobre imágenes sintéticas."""
    k = len(class_names)
    pred = probs.argmax(axis=1)
    counts = np.bincount(pred, minlength=k)
    freqs = counts / counts.sum()

    nonzero = freqs[freqs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(k)) if k > 1 else 0.0

    return {
        "distribucion_predicha": {c: float(f) for c, f in zip(class_names, freqs)},
        "conteos": {c: int(n) for c, n in zip(class_names, counts)},
        "confianza_media": float(probs.max(axis=1).mean()),
        "entropia_normalizada": entropy,
        "posible_mode_collapse": bool(freqs.max() >= collapse_threshold),
    }


# ─── Datos ────────────────────────────────────────────────────────────────────

def load_labeled_real(data_dir: str, max_per_class: int | None, seed: int):
    """Etiqueta = nombre de la carpeta que contiene la imagen."""
    from src.data import find_images, preprocess_images
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {data_dir}")

    by_class: dict[str, list[str]] = {}
    for p in paths:
        by_class.setdefault(os.path.basename(os.path.dirname(p)).lower(), []).append(p)

    rng = np.random.default_rng(seed)
    sel_paths, sel_labels = [], []
    class_names = sorted(by_class)
    for ci, c in enumerate(class_names):
        ps = by_class[c]
        if max_per_class and len(ps) > max_per_class:
            ps = [ps[i] for i in rng.choice(len(ps), max_per_class, replace=False)]
        sel_paths += ps
        sel_labels += [ci] * len(ps)

    # preprocess_images descarta las imágenes que fallan: se valida que no se desalineen las etiquetas
    X = preprocess_images(sel_paths)
    y = np.array(sel_labels)
    if len(X) != len(y):
        raise RuntimeError("Algunas imágenes no se pudieron leer; etiquetas desalineadas.")
    return X, y, class_names


# ─── Modelo ───────────────────────────────────────────────────────────────────

def build_cnn(num_classes: int, img_shape=(64, 64, 3)):
    import tensorflow as tf
    from tensorflow.keras import layers
    return tf.keras.Sequential([
        layers.Input(shape=img_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="MRI_Classifier")


def main():
    ap = argparse.ArgumentParser(description="Clasificador entrenado con reales para auditar sintéticas.")
    ap.add_argument("--weights", required=True, help="ruta al generador .keras")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=1000, help="imágenes sintéticas a clasificar")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--max-per-class", type=int, default=None, help="limita imágenes reales por clase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/classifier.json")
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from eval.quality_metrics import load_generator, generate_images

    tf.keras.utils.set_random_seed(args.seed)
    X, y, class_names = load_labeled_real(args.data_dir, args.max_per_class, args.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed
    )

    model = build_cnn(len(class_names), X.shape[1:])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=64, verbose=2)
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)

    generator = load_generator(args.weights)
    fake = generate_images(generator, args.n_samples, args.seed)
    probs = model.predict(fake, verbose=0)

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "clases": class_names,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "val_accuracy": float(val_acc),
        "n_sinteticas": int(len(fake)),
        "sinteticas": summarize_predictions(probs, class_names),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
