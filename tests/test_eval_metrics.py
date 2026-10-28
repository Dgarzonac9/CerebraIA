# -*- coding: utf-8 -*-
import numpy as np

from eval.classifier_check import summarize_predictions
from eval.quality_metrics import frechet_distance, kernel_inception_distance


def _feats(seed, shift=0.0, n=400, d=16):
    return np.random.default_rng(seed).normal(shift, 1.0, size=(n, d))


def test_fid_cercano_a_cero_para_misma_distribucion_y_crece_con_el_sesgo():
    mismo = frechet_distance(_feats(1), _feats(2))
    distinto = frechet_distance(_feats(1), _feats(3, shift=1.0))
    assert mismo < 1.0 < distinto
    assert abs(distinto - 16.0) < 3.0  # desplazamiento de 1 en 16 dimensiones


def test_kid_mayor_con_distribuciones_distintas():
    mismo, _ = kernel_inception_distance(_feats(1), _feats(2), n_subsets=5, subset_size=200)
    distinto, _ = kernel_inception_distance(_feats(1), _feats(3, shift=1.0), n_subsets=5, subset_size=200)
    assert distinto > mismo


def test_resumen_detecta_mode_collapse():
    clases = ["glioma", "meningioma", "pituitary", "no_tumor"]
    colapsado = np.tile([0.9, 0.04, 0.03, 0.03], (20, 1))
    r = summarize_predictions(colapsado, clases)
    assert r["posible_mode_collapse"] and r["distribucion_predicha"]["glioma"] == 1.0


def test_resumen_distribucion_uniforme():
    clases = ["a", "b", "c", "d"]
    probs = np.eye(4)[np.arange(40) % 4] * 0.6 + 0.1
    r = summarize_predictions(probs, clases)
    assert not r["posible_mode_collapse"]
    assert abs(r["entropia_normalizada"] - 1.0) < 1e-9
    assert abs(r["confianza_media"] - 0.7) < 1e-9
