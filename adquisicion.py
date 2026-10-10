"""M1 — Adquisición de imágenes.

Recibe la imagen de la cara inspeccionada (cámara de la estación o archivo) y evalúa la
calidad de captura. Una captura no apta NO se descarta: activa la regla R1 (RETENER) para
que quede trazada y el operario repita la toma.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import ADQ_BRILLO_RANGO, ADQ_NITIDEZ_MIN, ADQ_RES_MIN


class ErrorAdquisicion(Exception):
    """El archivo no puede leerse como imagen (no se puede registrar)."""


@dataclass
class Captura:
    imagen: np.ndarray
    ancho: int
    alto: int
    brillo: float
    nitidez: float
    apta: bool = True
    motivos: list = field(default_factory=list)


def decodificar(datos: bytes) -> np.ndarray:
    if not datos:
        raise ErrorAdquisicion("No se recibió ninguna imagen. Seleccione un archivo o capture nuevamente.")
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ErrorAdquisicion("El archivo no es una imagen válida (use JPG o PNG).")
    return img


def capturar_camara(indice: int = 0) -> np.ndarray:
    cam = cv2.VideoCapture(indice)
    try:
        ok, frame = cam.read()
    finally:
        cam.release()
    if not ok or frame is None:
        raise ErrorAdquisicion(f"No se pudo leer la cámara {indice}. Verifique la conexión.")
    return frame


def evaluar(img: np.ndarray) -> Captura:
    alto, ancho = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo = float(gris.mean())
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    motivos = []
    if ancho < ADQ_RES_MIN[0] or alto < ADQ_RES_MIN[1]:
        motivos.append(f"Resolución insuficiente ({ancho}×{alto}); mínimo {ADQ_RES_MIN[0]}×{ADQ_RES_MIN[1]} px")
    if not ADQ_BRILLO_RANGO[0] <= brillo <= ADQ_BRILLO_RANGO[1]:
        motivos.append(f"Iluminación fuera de rango (brillo {brillo:.0f}; aceptable {ADQ_BRILLO_RANGO[0]}–{ADQ_BRILLO_RANGO[1]})")
    if nitidez < ADQ_NITIDEZ_MIN:
        motivos.append(f"Imagen desenfocada (nitidez {nitidez:.1f} < {ADQ_NITIDEZ_MIN})")
    return Captura(img, ancho, alto, brillo, nitidez, not motivos, motivos)
