"""Orquestador del DSS: integra M1–M5 en el flujo de inspección de una cara de la caja."""
import json
import time
import uuid

import cv2

from config import IMG_STORE, PARAM_FILE, TAU1_DEFECTO, TAU2_DEFECTO, UMBRALES_PATH
from modulos import adquisicion, pokayoke, procesamiento, registro
from modulos.ia import DetectorDefectos, dibujar

_detector = None
_senal = pokayoke.SalidaSenal()


def detector():
    global _detector
    if _detector is None:
        _detector = DetectorDefectos()
    return _detector


def umbrales():
    """τ1/τ2 vigentes: parámetros del jefe > calibración por costo > valores por defecto."""
    for ruta in (PARAM_FILE, UMBRALES_PATH):
        try:
            d = json.load(open(ruta))
            return float(d["tau1"]), float(d["tau2"])
        except Exception:
            continue
    return TAU1_DEFECTO, TAU2_DEFECTO


def inspeccionar(img, *, caja_id, lote, modelo_id, fase, vista, operario, origen="REAL", fecha_hora=None,
                 verdad_terreno=None, guardar_imagenes=True, tau=None):
    t0 = time.perf_counter()
    tau1, tau2 = tau or umbrales()
    cap = adquisicion.evaluar(img)                                                # M1
    faltan = registro.faltantes_cabecera(caja_id, lote, modelo_id, fase, vista)
    dets, proc = [], None
    if cap.apta and not faltan:
        proc = procesamiento.procesar(cap.imagen, modelo_id, fase, vista)        # M2
        dets = detector().detectar(proc)                                          # M3
    decision = pokayoke.decidir(dets, tau1, tau2, cap.apta, cap.motivos, not faltan, faltan)   # M4 (R1–R6)
    senal = _senal.emitir(decision)
    t_ms = (time.perf_counter() - t0) * 1000
    ruta = ruta_anot = None
    if guardar_imagenes:
        IMG_STORE.mkdir(parents=True, exist_ok=True)
        base = f"{(caja_id or 'SINID').upper()}_{fase[:3] if fase else 'x'}_{vista or 'x'}_{uuid.uuid4().hex[:6]}"
        color = proc.color if proc is not None else cap.imagen
        ruta, ruta_anot = f"{base}.jpg", f"{base}_anotada.jpg"
        cv2.imwrite(str(IMG_STORE / ruta), color, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(IMG_STORE / ruta_anot), dibujar(color, dets, tau1), [cv2.IMWRITE_JPEG_QUALITY, 90])
    reg = registro.guardar_inspeccion(                                            # M5
        caja_id=caja_id, lote=lote, modelo_id=modelo_id, fase=fase, vista=vista, operario=operario,
        decision=decision, dets=dets, tau1=tau1, tau2=tau2, senal=senal, tiempo_ms=t_ms, imagen=ruta,
        imagen_anotada=ruta_anot, version_modelo=detector().version, origen=origen, fecha_hora=fecha_hora,
        verdad_terreno=verdad_terreno)
    return {"registro": reg, "decision": decision, "detecciones": dets, "captura": cap, "proc": proc, "senal": senal}
