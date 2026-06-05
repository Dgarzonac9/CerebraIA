# -*- coding: utf-8 -*-
"""
DCGAN para generación de imágenes de MRI cerebral.
Práctica 3 - Generative AI: GANs, LLMs y Agentic AI
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
from datetime import datetime


# ─── Configuración ────────────────────────────────────────────────────────────

IMG_SIZE    = 64
BATCH_SIZE  = 32
LATENT_DIM  = 100
EPOCHS      = 200


# ─── Modelo DCGAN ─────────────────────────────────────────────────────────────

class DCGAN:
    """Deep Convolutional GAN para síntesis de MRI cerebral."""

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (64, 64, 3)):
        self.latent_dim = latent_dim
        self.img_shape  = img_shape

        self.generator     = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.cross_entropy  = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # ── Arquitecturas ──────────────────────────────────────────────────────────

    def build_generator(self) -> tf.keras.Sequential:
        """Transforma ruido aleatorio (latent_dim,) en imágenes (64×64×3)."""
        model = tf.keras.Sequential(name="Generator")

        # Dense → 4×4×512
        model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Reshape((4, 4, 512)))

        # 4×4 → 8×8
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # 8×8 → 16×16
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # 16×16 → 32×32
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # 32×32 → 64×64 (salida tanh → rango [-1, 1])
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same",
                                         use_bias=False, activation="tanh"))
        return model

    def build_discriminator(self) -> tf.keras.Sequential:
        """Clasifica imágenes (64×64×3) como reales o falsas."""
        model = tf.keras.Sequential(name="Discriminator")

        # 64×64 → 32×32
        model.add(layers.Conv2D(64,  (5, 5), strides=(2, 2), padding="same", input_shape=self.img_shape))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        # 32×32 → 16×16
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        # 16×16 → 8×8
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        # 8×8 → 4×4
        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    # ── Pérdidas ───────────────────────────────────────────────────────────────

    def discriminator_loss(self, real_output, fake_output):
        return (
            self.cross_entropy(tf.ones_like(real_output),  real_output) +
            self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        )

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # ── Entrenamiento ──────────────────────────────────────────────────────────

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output      = self.discriminator(real_images,    training=True)
            fake_output      = self.discriminator(generated_images, training=True)
            gen_loss         = self.generator_loss(fake_output)
            disc_loss        = self.discriminator_loss(real_output, fake_output)

        self.gen_optimizer.apply_gradients(
            zip(gen_tape.gradient(gen_loss, self.generator.trainable_variables),
                self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(
            zip(disc_tape.gradient(disc_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    # ── Inferencia ─────────────────────────────────────────────────────────────

    def generate_images(self, num_images: int = 16, seed: int = None) -> np.ndarray:
        if seed is not None:
            tf.random.set_seed(seed)
        noise = tf.random.normal([num_images, self.latent_dim])
        return self.generator(noise, training=False).numpy()


# ─── Callback de entrenamiento ────────────────────────────────────────────────

class GANCallback:
    """Monitoreo visual y guardado de checkpoints durante el entrenamiento."""

    def __init__(self, gan: DCGAN, save_path: str, save_interval: int = 10):
        self.gan           = gan
        self.save_path     = save_path
        self.save_interval = save_interval
        self.history       = {"gen_loss": [], "disc_loss": [], "epochs": []}
        self.fixed_seed    = tf.random.normal([16, gan.latent_dim])

    def on_epoch_end(self, epoch: int, gen_loss: float, disc_loss: float):
        self.history["epochs"].append(epoch)
        self.history["gen_loss"].append(float(gen_loss))
        self.history["disc_loss"].append(float(disc_loss))

        if (epoch + 1) % self.save_interval == 0:
            self._save_images(epoch)
            self._save_metrics_plot(epoch)

    def _save_images(self, epoch: int):
        generated = (self.gan.generator(self.fixed_seed, training=False) + 1) / 2
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax, img in zip(axes.flatten(), generated):
            ax.imshow(img); ax.axis("off")
        plt.suptitle(f"Imágenes Generadas — Época {epoch + 1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f"generated_epoch_{epoch + 1:04d}.png"), dpi=150)
        plt.close()

    def _save_metrics_plot(self, epoch: int):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history["epochs"], self.history["gen_loss"],  "b-", linewidth=2)
        axes[0].set_title("Pérdida del Generador");     axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history["epochs"], self.history["disc_loss"], "r-", linewidth=2)
        axes[1].set_title("Pérdida del Discriminador"); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f"Métricas — Época {epoch + 1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "training_metrics.png"), dpi=150)
        plt.close()

    def save_history(self, path: str | None = None):
        path = path or os.path.join(self.save_path, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Historial guardado en: {path}")


# ─── Bucle de entrenamiento ───────────────────────────────────────────────────

def train_gan(
    gan:      DCGAN,
    dataset:  tf.data.Dataset,
    epochs:   int,
    callback: GANCallback,
    models_path: str,
) -> dict:
    """Entrena la GAN y guarda checkpoints cada 50 épocas."""

    start = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        gen_losses, disc_losses = [], []

        for batch in tqdm(dataset, desc=f"Época {epoch + 1}/{epochs}", leave=False):
            g, d = gan.train_step(batch)
            gen_losses.append(g.numpy())
            disc_losses.append(d.numpy())

        avg_g = np.mean(gen_losses)
        avg_d = np.mean(disc_losses)
        callback.on_epoch_end(epoch, avg_g, avg_d)

        print(f"Época {epoch + 1:3d}/{epochs} | G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | {time.time() - t0:.1f}s")

        if (epoch + 1) % 50 == 0:
            ckpt = os.path.join(models_path, f"checkpoint_epoch_{epoch + 1}")
            gan.generator.save(ckpt + "_generator.keras")
            gan.discriminator.save(ckpt + "_discriminator.keras")
            print(f"  Checkpoint guardado en época {epoch + 1}")

    print(f"\nENTRENAMIENTO COMPLETADO en {(time.time() - start) / 60:.1f} min")
    callback.save_history()
    return callback.history
