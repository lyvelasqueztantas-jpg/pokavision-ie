"""Dataset de imágenes SINTÉTICAS para verificar el artefacto (formato YOLO + inventario).

Cobertura de la Especificación v3: 4 modelos × 2 fases × 2 vistas × 4 condiciones
(conforme, rayadura, pintura irregular, soldadura visible) = 64 combinaciones.
Cada caja tiene 4 imágenes (2 fases × 2 vistas); el defecto aparece en la cara afectada.
La división entrenamiento/validación/prueba se hace POR CAJA (sin fuga entre conjuntos).
En producto terminado frontal se dibuja la cerradura/herraje, que se excluye del ROI.

Uso: python datos_sinteticos.py --reps 15    (4×4×15 = 240 cajas, 960 imágenes)
IMPORTANTE: sirve para verificar el funcionamiento; NO reemplaza imágenes reales.
"""
import argparse
import csv
import json
import math
import random

import cv2
import numpy as np

from config import CLASES, DATASET_DIR, MODELOS, FASES, VISTAS

W, H = 640, 480
CONDICIONES = ["conforme"] + CLASES


def zonas_herraje(modelo, fase, vista):
    """Rectángulos (x1,y1,x2,y2) ocupados por cerradura/manija: se excluyen del ROI de inspección."""
    if fase != "producto_terminado" or vista != "frontal":
        return []
    return {"M01": [(440, 180, 560, 300)],
            "M02": [(440, 165, 560, 315)],
            "M03": [(455, 120, 565, 255), (470, 270, 550, 420)],
            "M04": [(425, 175, 535, 305), (530, 200, 615, 280)]}[modelo]


def _dibujar_herraje(img, modelo):
    z = [(x1 + 15, y1 + 15, x2 - 15, y2 - 15) for x1, y1, x2, y2 in zonas_herraje(modelo, "producto_terminado", "frontal")]
    negro, metal = (30, 30, 32), (185, 188, 190)
    if modelo == "M01":
        (x1, y1, x2, y2) = z[0]; c = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(img, c, 45, metal, -1, cv2.LINE_AA); cv2.circle(img, c, 30, negro, -1, cv2.LINE_AA)
        for k in range(12):
            a = k * math.pi / 6
            cv2.circle(img, (int(c[0] + 38 * math.cos(a)), int(c[1] + 38 * math.sin(a))), 2, negro, -1)
    else:
        x1, y1, x2, y2 = z[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), metal, -1)
        cv2.rectangle(img, (x1 + 8, y1 + 8), (x2 - 8, y1 + 30), negro, -1)
        for i in range(3):
            for j in range(4):
                cv2.rectangle(img, (x1 + 12 + i * 26, y1 + 38 + j * 18), (x1 + 30 + i * 26, y1 + 50 + j * 18), negro, -1)
        if len(z) > 1:
            hx1, hy1, hx2, hy2 = z[1]
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), metal, -1, cv2.LINE_AA)
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (90, 90, 92), 2, cv2.LINE_AA)


def superficie(rng, modelo):
    color = np.array(MODELOS[modelo]["color"], np.float32)
    base = np.zeros((H, W, 3), np.float32) + color
    bajo = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6)
    bajo = bajo / (bajo.std() + 1e-6) * rng.uniform(2.0, 4.0)
    grano = rng.normal(0, rng.uniform(1.0, 2.5), (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    grad = ((xx / W - .5) * math.cos(ang) + (yy / H - .5) * math.sin(ang)) * rng.uniform(10, 22)
    vig = -((xx / W - .5) ** 2 + (yy / H - .5) ** 2) * rng.uniform(10, 25)
    return base + (bajo + grano + grad + vig)[..., None], color


def _libre(box, zonas):
    x1, y1, x2, y2 = box
    return all(x2 < a or x1 > c or y2 < b or y1 > d for a, b, c, d in zonas)


def _rayadura(img, color, rng):
    largo, ang = rng.uniform(60, 220), rng.uniform(0, math.pi)
    x0, y0 = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
    curv = rng.uniform(-0.002, 0.002)
    pts = []
    for t in np.linspace(0, 1, 30):
        d = t * largo
        pts.append((x0 + d * math.cos(ang) - curv * d * d * math.sin(ang), y0 + d * math.sin(ang) + curv * d * d * math.cos(ang)))
    pts = np.clip(np.array(pts), [3, 3], [W - 4, H - 4])
    capa = np.zeros((H, W), np.float32)
    cv2.polylines(capa, [pts.astype(np.int32)], False, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
    delta = (1 if color.mean() < 120 else -1) * rng.uniform(35, 70)
    x1, y1 = pts.min(0) - 4; x2, y2 = pts.max(0) + 4
    return capa, delta, (x1, y1, x2, y2)


def _pintura(img, color, rng):
    tipo = rng.choice(["crater", "escurrimiento", "mancha"])
    cx, cy = rng.uniform(40, W - 40), rng.uniform(40, H - 70)
    capa = np.zeros((H, W), np.float32)
    if tipo == "crater":
        r = rng.uniform(6, 16)
        cv2.circle(capa, (int(cx), int(cy)), int(r), 1.0, -1, cv2.LINE_AA)
        cv2.circle(capa, (int(cx), int(cy)), int(r * .5), -0.6, -1, cv2.LINE_AA)
        bw, bh = r, r
    elif tipo == "escurrimiento":
        a, b = rng.uniform(4, 9), rng.uniform(14, 32)
        cv2.ellipse(capa, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.circle(capa, (int(cx), int(cy + b)), int(a * 1.4), 1.0, -1, cv2.LINE_AA)
        bw, bh = a * 1.4, b + a * 1.4; cy = cy + a * .7
    else:
        r = rng.uniform(8, 18)
        for _ in range(5):
            ox, oy = rng.normal(0, r * .35, 2)
            cv2.circle(capa, (int(cx + ox), int(cy + oy)), int(r * rng.uniform(.4, .8)), 1.0, -1, cv2.LINE_AA)
        bw, bh = r * 1.3, r * 1.3
    capa = cv2.GaussianBlur(capa, (5, 5), 1.2)
    delta = (-1 if rng.random() < .65 else 1) * rng.uniform(28, 55)
    return capa, delta, (cx - bw - 4, cy - bh - 4, cx + bw + 4, cy + bh + 4)


def _soldadura(img, color, rng):
    """Cordón visible irregular (gotas solapadas) con salpicaduras: discontinuidad SUPERFICIAL aparente."""
    largo = rng.uniform(50, 130)
    vertical = rng.random() < .5
    x0, y0 = rng.uniform(40, W - 40 - (0 if vertical else largo)), rng.uniform(40, H - 40 - (largo if vertical else 0))
    capa = np.zeros((H, W), np.float32)
    n = int(largo / 3)
    pts = []
    for i in range(n):
        x = x0 + (0 if vertical else i * 3) + rng.normal(0, 1.2)
        y = y0 + (i * 3 if vertical else 0) + rng.normal(0, 1.2)
        r = rng.uniform(3, 6)
        cv2.circle(capa, (int(x), int(y)), int(r), rng.uniform(.7, 1.0), -1, cv2.LINE_AA)
        pts.append((x, y))
    for _ in range(int(rng.integers(4, 10))):  # salpicaduras
        px, py = pts[int(rng.integers(0, len(pts)))]
        cv2.circle(capa, (int(px + rng.normal(0, 12)), int(py + rng.normal(0, 12))), int(rng.integers(1, 3)), 1.0, -1)
    capa = cv2.GaussianBlur(capa, (3, 3), .8)
    delta = (1 if color.mean() < 120 else -1) * rng.uniform(35, 60)
    pts = np.array(pts)
    x1, y1 = pts.min(0) - 14; x2, y2 = pts.max(0) + 14
    return capa, delta, (x1, y1, x2, y2)


GEN = {"rayadura": _rayadura, "pintura_irregular": _pintura, "soldadura_visible": _soldadura}


def generar_imagen(rng, modelo="M01", fase="carcasa_planchada", vista="frontal", defectos=()):
    img, color = superficie(rng, modelo)
    zonas = zonas_herraje(modelo, fase, vista)
    cajas = []
    for clase in defectos:
        for _ in range(30):
            capa, delta, box = GEN[clase](img, color, rng)
            if _libre(box, zonas):
                img += (capa * delta)[..., None]
                x1, y1, x2, y2 = box
                cajas.append((CLASES.index(clase), max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)))
                break
    img = np.clip(img, 0, 255).astype(np.uint8)
    if zonas:
        _dibujar_herraje(img, modelo)
    return img, cajas


def guardar_yolo(path, cajas):
    with open(path, "w") as f:
        for c, x1, y1, x2, y2 in cajas:
            f.write(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")


def main(reps=15, semilla=2026):
    rng = np.random.default_rng(semilla); random.seed(semilla)
    cajas = []
    k = 0
    for m in MODELOS:
        for cond in CONDICIONES:
            for r in range(reps):
                k += 1
                cajas.append({"caja_id": f"CAJ-DS-{k:04d}", "lote_id": f"LDS-{1 + (k - 1) // 16:02d}", "modelo": m, "condicion": cond})
    # división por caja estratificada por modelo × condición (≈70/15/15)
    grupos = {}
    for c in cajas:
        grupos.setdefault((c["modelo"], c["condicion"]), []).append(c)
    for g in grupos.values():
        random.shuffle(g)
        n = len(g); nv = max(1, round(n * .15)); nt = max(1, round(n * .15))
        for i, c in enumerate(g):
            c["split"] = "test" if i < nt else "val" if i < nt + nv else "train"
    filas, cont = [], {}
    for c in cajas:
        cara = random.choice(VISTAS)
        for fase in FASES:
            for vista in VISTAS:
                defs = []
                if c["condicion"] != "conforme" and vista == cara:
                    defs = [c["condicion"]] * (1 if rng.random() < .75 else 2)
                img, boxes = generar_imagen(rng, c["modelo"], fase, vista, defs)
                s = c["split"]
                nombre = f"{c['caja_id']}_{c['modelo']}_{fase}_{vista}"
                for sub in ("images", "labels"):
                    (DATASET_DIR / sub / s).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(DATASET_DIR / "images" / s / f"{nombre}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                guardar_yolo(DATASET_DIR / "labels" / s / f"{nombre}.txt", boxes)
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clase_img = CLASES[boxes[0][0]] if boxes else "conforme"
                filas.append({"image_id": nombre, "image_path": f"images/{s}/{nombre}.jpg", "label_path": f"labels/{s}/{nombre}.txt",
                              "modelo_id": c["modelo"], "caja_id": c["caja_id"], "lote_id": c["lote_id"], "fase": fase,
                              "vista": vista, "categoria": clase_img, "n_defectos": len(boxes), "split": s,
                              "iluminacion": "difusa simulada", "brillo": round(float(g.mean()), 1),
                              "nitidez": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1), "fuente": "SINTÉTICO"})
                key = (s, clase_img); cont[key] = cont.get(key, 0) + 1
    with open(DATASET_DIR / "inventario_imagenes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: rayadura\n  1: pintura_irregular\n  2: soldadura_visible\n")
    fuga = {}
    for r in filas:
        fuga.setdefault(r["caja_id"], set()).add(r["split"])
    combos = {(r["modelo_id"], r["fase"], r["vista"], r["categoria"]) for r in filas}
    res = {"cajas": len(cajas), "imagenes": len(filas), "combinaciones_cubiertas": len(combos),
           "cajas_en_mas_de_un_split": sum(len(v) > 1 for v in fuga.values()),
           "imagenes_por_split_y_clase": {f"{s}|{c}": n for (s, c), n in sorted(cont.items())},
           "fuente": "100 % SINTÉTICO (verificación del artefacto, no validación)"}
    json.dump(res, open(DATASET_DIR / "resumen_dataset.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=15)
    main(ap.parse_args().reps)
