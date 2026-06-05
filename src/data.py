# -*- coding: utf-8 -*-
"""
Carga y preprocesamiento del dataset de MRI cerebral.
Dataset: https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset
"""

import os
import numpy as np
import tensorflow as tf
from glob import glob
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


IMG_SIZE   = 64
BATCH_SIZE = 32


# ─── Descarga del dataset ─────────────────────────────────────────────────────

def setup_kaggle(username: str, key: str):
    """Configura las credenciales de Kaggle."""
    os.makedirs("/root/.kaggle", exist_ok=True)
    creds = f'{{"username": "{username}", "key": "{key}"}}'
    with open("/root/.kaggle/kaggle.json", "w") as f:
        f.write(creds)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("Credenciales de Kaggle configuradas.")


def download_dataset(dest: str = "/content/dataset"):
    """Descarga y extrae el dataset desde Kaggle."""
    import zipfile, subprocess
    os.makedirs(dest, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "ishans24/brain-tumor-dataset", "-p", dest],
        check=True,
    )
    zip_path = os.path.join(dest, "brain-tumor-dataset.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"Dataset extraído en: {dest}")


# ─── Exploración ──────────────────────────────────────────────────────────────

def find_images(dataset_path: str) -> list[str]:
    """Devuelve la lista de rutas de todas las imágenes del dataset."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))
    return images


def explore_dataset(image_paths: list[str], save_path: str | None = None):
    """Imprime estadísticas y muestra una grilla de ejemplos."""
    print(f"Total de imágenes: {len(image_paths)}")
    folders = [os.path.dirname(p).split("/")[-1] for p in image_paths]
    for folder, count in Counter(folders).items():
        print(f"   {folder}: {count} imágenes")

    sample = np.random.choice(image_paths, min(10, len(image_paths)), replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, path in zip(axes.flatten(), sample):
        img = Image.open(path)
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(os.path.dirname(path).split("/")[-1][:15])
        ax.axis("off")
    plt.suptitle("Muestras del Dataset de MRI Cerebral", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─── Preprocesamiento ─────────────────────────────────────────────────────────

def preprocess_images(
    image_paths: list[str],
    target_size: tuple = (IMG_SIZE, IMG_SIZE),
) -> np.ndarray:
    """
    Carga, redimensiona, convierte a RGB y normaliza imágenes a [-1, 1].

    Returns
    -------
    np.ndarray  shape (N, H, W, 3), dtype float32, rango [-1, 1]
    """
    images, failed = [], 0

    for path in tqdm(image_paths, desc="Procesando imágenes"):
        try:
            img = load_img(path, target_size=target_size)
            arr = img_to_array(img)
            if arr.shape[-1] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            arr = (arr - 127.5) / 127.5          # → [-1, 1]
            images.append(arr)
        except Exception:
            failed += 1

    print(f"Procesadas: {len(images)} | Fallidas: {failed}")
    return np.array(images)


def build_tf_dataset(
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = 1000,
) -> tf.data.Dataset:
    """Construye un tf.data.Dataset listo para entrenar."""
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.shuffle(buffer_size).batch(batch_size)
    print(f"Dataset creado: {len(ds)} batches de {batch_size}")
    return ds
