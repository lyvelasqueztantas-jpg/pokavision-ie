"""M2 — Procesamiento de imágenes.

Reducción de ruido, contraste CLAHE, normalización, redimensionamiento y mapa de
anomalía por sustracción de fondo. Excluye del ROI la cerradura y los herrajes según
modelo, fase y vista (Especificación v3, sección 10).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from config import PROC_TAMANO
from datos_sinteticos import zonas_herraje


@dataclass
class ImagenProcesada:
    color: np.ndarray
    gris: np.ndarray
    residuo: np.ndarray
    roi: np.ndarray
    pasos: dict


def mascara_roi(modelo, fase, vista, shape=(480, 640)):
    roi = np.ones(shape, np.uint8)
    for x1, y1, x2, y2 in zonas_herraje(modelo, fase, vista):
        roi[y1:y2, x1:x2] = 0
    return roi


def procesar(img, modelo="M01", fase="carcasa_planchada", vista="frontal") -> ImagenProcesada:
    color = cv2.resize(img, PROC_TAMANO, interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    sin_ruido = cv2.GaussianBlur(gris, (3, 3), 0.8)
    contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sin_ruido)
    norm = cv2.normalize(contraste, None, 0, 255, cv2.NORM_MINMAX)
    roi = mascara_roi(modelo, fase, vista, norm.shape)
    base = norm.astype(np.float32)
    # el fondo se estima rellenando la zona excluida con la mediana para no contaminar el entorno
    relleno = base.copy(); relleno[roi == 0] = float(np.median(base[roi == 1]))
    fondo = cv2.GaussianBlur(relleno, (0, 0), 9)
    residuo = cv2.absdiff(relleno, fondo) * roi
    vis_roi = color.copy(); vis_roi[roi == 0] = (vis_roi[roi == 0] * 0.3 + np.array([0, 0, 120]) * 0.7).astype(np.uint8)
    pasos = {"original": color, "sin_ruido": sin_ruido, "contraste": contraste, "normalizada": norm,
             "roi": vis_roi, "residuo": cv2.normalize(residuo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)}
    return ImagenProcesada(color, norm, residuo, roi, pasos)
