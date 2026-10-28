# -*- coding: utf-8 -*-
"""
Métricas de calidad de la DCGAN: FID y KID entre imágenes reales y sintéticas.

Características: InceptionV3 de Keras (ImageNet, pooling promedio, 2048-d),
con las imágenes reescaladas a 299x299.

El generador de src/gan.py es INCONDICIONAL (recibe solo ruido, sin etiqueta de
clase), por lo que solo se calcula la métrica global. Las imágenes sintéticas no
pertenecen a ninguna clase conocida, así que no existe un FID/KID por clase válido.

Uso:
    python -m eval.quality_metrics --weights models/generator_final.keras \
        --data-dir /content/dataset --n-samples 5000 --out eval/results/metrics.json
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
from scipy import linalg


# ─── Métricas (numpy puro) ────────────────────────────────────────────────────

def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """FID entre dos conjuntos de características (N, D)."""
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    covmean = linalg.sqrtm(sigma_a.dot(sigma_b))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    covmean = np.real(covmean)

    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def kernel_inception_distance(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_subsets: int = 50,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """KID (MMD² insesgado, kernel polinomial cúbico). Devuelve (media, desv. estándar)."""
    rng = np.random.default_rng(seed)
    d = feats_a.shape[1]
    m = min(subset_size, len(feats_a), len(feats_b))
    if m < 2:
        raise ValueError("KID necesita al menos 2 muestras por conjunto.")

    def poly(x, y):
        return (x @ y.T / d + 1.0) ** 3

    vals = []
    for _ in range(n_subsets):
        x = feats_a[rng.choice(len(feats_a), m, replace=False)]
        y = feats_b[rng.choice(len(feats_b), m, replace=False)]
        k_xx, k_yy, k_xy = poly(x, x), poly(y, y), poly(x, y)
        mmd = (
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2 * k_xy.mean()
        )
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


# ─── Extracción de características ───────────────────────────────────────────

def build_feature_extractor():
    import tensorflow as tf
    return tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(299, 299, 3)
    )


def extract_features(extractor, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """images: (N, H, W, 3) en [-1, 1], que es el rango que espera InceptionV3."""
    import tensorflow as tf
    feats = []
    for i in range(0, len(images), batch_size):
        batch = tf.image.resize(images[i:i + batch_size], (299, 299), method="bilinear")
        feats.append(extractor(batch, training=False).numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


# ─── Datos ────────────────────────────────────────────────────────────────────

def load_real_images(data_dir: str, n: int, seed: int = 0) -> np.ndarray:
    from src.data import find_images, preprocess_images
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {data_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), min(n, len(paths)), replace=False)
    return preprocess_images([paths[i] for i in idx])


def load_generator(weights: str):
    import tensorflow as tf
    return tf.keras.models.load_model(weights)


def is_class_conditional(generator) -> bool:
    """El generador de src/gan.py tiene una única entrada de ruido (latent_dim,)."""
    shapes = generator.input_shape
    return isinstance(shapes, list) and len(shapes) > 1


def generate_images(generator, n: int, seed: int = 0, batch_size: int = 256) -> np.ndarray:
    import tensorflow as tf
    latent_dim = generator.input_shape[-1]
    tf.random.set_seed(seed)
    out = []
    for i in range(0, n, batch_size):
        noise = tf.random.normal([min(batch_size, n - i), latent_dim])
        out.append(generator(noise, training=False).numpy())
    return np.concatenate(out, axis=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="FID y KID de la DCGAN (global).")
    ap.add_argument("--n-samples", type=int, default=2000, help="imágenes reales y sintéticas a comparar")
    ap.add_argument("--weights", required=True, help="ruta al generador .keras")
    ap.add_argument("--data-dir", required=True, help="carpeta con el dataset real")
    ap.add_argument("--out", default="eval/results/metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generator = load_generator(args.weights)
    conditional = is_class_conditional(generator)

    real = load_real_images(args.data_dir, args.n_samples, args.seed)
    fake = generate_images(generator, len(real), args.seed)

    extractor = build_feature_extractor()
    f_real = extract_features(extractor, real)
    f_fake = extract_features(extractor, fake)

    fid = frechet_distance(f_real, f_fake)
    kid_mean, kid_std = kernel_inception_distance(f_real, f_fake, seed=args.seed)

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "weights": os.path.basename(args.weights),
        "n_real": int(len(real)),
        "n_sinteticas": int(len(fake)),
        "extractor": "InceptionV3 (ImageNet, pooling avg, 299x299)",
        "generador_condicionado_por_clase": conditional,
        "global": {"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std},
        "por_clase": None,
        "nota": (
            "El generador es incondicional: no hay etiqueta de clase en la entrada, "
            "por lo que solo se calcula la métrica global."
            if not conditional else
            "Generador condicionado detectado, pero el cálculo por clase no está implementado."
        ),
    }
    if len(real) < 2048:
        result["advertencia"] = (
            f"n={len(real)} < 2048 (dimensión de las características): la matriz de covarianza "
            "es singular y el FID no es comparable con valores publicados. Úsalo solo como prueba de humo."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
