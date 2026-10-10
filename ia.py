"""M3 — Inteligencia artificial: detección en dos etapas (TF1, contribución C3).

Etapa 1 · Detección de anomalías entrenada con superficies CONFORMES: el umbral del mapa de
anomalía se aprende de la distribución del residuo en imágenes conformes (percentil alto),
de modo que no se necesitan muchas imágenes defectuosas para localizar regiones anómalas.
Etapa 2 · Clasificación del tipo de defecto: Random Forest sobre descriptores geométricos,
de intensidad y HOG de cada región (rayadura, pintura irregular, soldadura visible o
textura normal).
Salida por región: clase, caja, confianza de la clase y puntaje de defecto s = 1 − P(normal).
Motor v2 planificado: YOLOv8 entrenado en Colab con imágenes reales (se usa si existe
modelos/yolov8_best.pt y ultralytics está instalado).
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from config import CLASES, MODEL_PATH

NORMAL = len(CLASES)  # etiqueta auxiliar "textura normal"


@dataclass
class Deteccion:
    clase: str
    confianza: float
    puntaje: float
    x1: int
    y1: int
    x2: int
    y2: int

    def dict(self):
        return asdict(self)


def candidatos(residuo, umbral, area_min=12):
    mascara = (residuo > umbral).astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, et, st, _ = cv2.connectedComponentsWithStats(mascara, 8)
    return [(x, y, x + w, y + h, et == i) for i, (x, y, w, h, a) in enumerate(st) if i and a >= area_min]


def descriptores(gris, residuo, reg):
    x1, y1, x2, y2, mask = reg
    w, h = x2 - x1, y2 - y1
    m = mask[y1:y2, x1:x2].astype(np.uint8)
    area = float(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(c, True) + 1e-6
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
    ys, xs = np.nonzero(m)
    if len(xs) > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
        elong = float(np.sqrt((ev[1] + 1e-6) / (ev[0] + 1e-6)))
    else:
        elong = 1.0
    r = residuo[y1:y2, x1:x2][m > 0]
    n_comp = cv2.connectedComponents(m)[0] - 1
    X1, Y1 = max(0, x1 - 6), max(0, y1 - 6)
    X2, Y2 = min(gris.shape[1], x2 + 6), min(gris.shape[0], y2 + 6)
    parche = cv2.resize(gris[Y1:Y2, X1:X2], (32, 32), interpolation=cv2.INTER_AREA)
    f_hog = hog(parche, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    geo = [area, w, h, w * h, max(w, h) / (min(w, h) + 1e-6), elong, area / (w * h + 1e-6), area / hull,
           perim * perim / (area + 1e-6), float(r.mean()), float(r.max()), float(r.std()), n_comp]
    return np.concatenate([np.array(geo, np.float32), f_hog.astype(np.float32)])


class DetectorDefectos:
    def __init__(self, ruta=MODEL_PATH):
        self.version, self.yolo = "v1-anomalia+RF", None
        pesos = Path(ruta).parent / "yolov8_best.pt"
        if pesos.exists():
            try:  # pragma: no cover
                from ultralytics import YOLO
                self.yolo, self.version = YOLO(str(pesos)), "v2-YOLOv8"
            except Exception:
                self.yolo = None
        if not Path(ruta).exists() and self.yolo is None:
            raise FileNotFoundError("No existe el modelo entrenado. Ejecute: python entrenar_modelo.py")
        paquete = joblib.load(ruta) if Path(ruta).exists() else {}
        self.rf = paquete.get("rf")
        self.umbral = paquete.get("umbral_anomalia", 12.0)

    def detectar(self, proc):
        if self.yolo is not None:  # pragma: no cover
            r = self.yolo(proc.color, verbose=False)[0]
            return [Deteccion(CLASES[int(b.cls)], float(b.conf), float(b.conf), *map(int, b.xyxy[0].tolist())) for b in r.boxes]
        regs = candidatos(proc.residuo, self.umbral)
        if not regs:
            return []
        X = np.vstack([descriptores(proc.gris, proc.residuo, r) for r in regs])
        self.rf.n_jobs = 1
        P = self.rf.predict_proba(X)
        idx = {c: i for i, c in enumerate(self.rf.classes_)}
        dets = []
        for reg, p in zip(regs, P):
            p_normal = p[idx[NORMAL]] if NORMAL in idx else 0.0
            s = 1.0 - float(p_normal)
            defect_probs = {c: p[idx[c]] for c in idx if c != NORMAL}
            if not defect_probs or s < 0.05:
                continue
            c = max(defect_probs, key=defect_probs.get)
            x1, y1, x2, y2, _ = reg
            dets.append(Deteccion(CLASES[int(c)], round(float(defect_probs[c] / max(s, 1e-6)), 3), round(s, 3),
                                  int(x1), int(y1), int(x2), int(y2)))
        return dets


def puntaje_imagen(dets):
    return max((d.puntaje for d in dets), default=0.0)


def dibujar(img, dets, tau1=0.35):
    out = img.copy()
    col = {"rayadura": (40, 40, 230), "pintura_irregular": (0, 200, 255), "soldadura_visible": (220, 0, 220)}
    for d in dets:
        if d.puntaje < tau1:
            continue
        c = col.get(d.clase, (255, 0, 0))
        cv2.rectangle(out, (d.x1 - 3, d.y1 - 3), (d.x2 + 3, d.y2 + 3), c, 2)
        t = f"{d.clase} s={d.puntaje:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)
        y = max(th + 4, d.y1 - 6)
        cv2.rectangle(out, (d.x1 - 3, y - th - 4), (d.x1 + tw, y + 2), c, -1)
        cv2.putText(out, t, (d.x1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return out
