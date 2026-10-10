"""M4 — Poka-Yoke de control: núcleo de decisión del DSS (decisión operacional).

Reglas (Especificación v3 / Excel REGLAS_POKAYOKE):
  R1  Imagen borrosa, oscura o de baja resolución            → RETENIDO (repetir captura)
  R2  Metadatos incompletos (lote/modelo/fase/vista)           → RETENIDO
  R3  Puntaje de defecto s ≥ τ2                                → BLOQUEADO (alerta + bloqueo de montaje)
  R4  τ1 ≤ s < τ2                                              → RETENIDO (dictamen del supervisor)
  R5  s < τ1 y verificaciones completas                        → LIBERADO
  R6  Soldadura visible con s ≥ τ1 (no se infiere integridad)  → RETENIDO para verificación por método adecuado
      (si además hay otro defecto con s ≥ τ2 prevalece R3)
Los umbrales τ1 y τ2 se calibran minimizando el costo esperado de la inspección (TF1, C2):
  CE(τ1, τ2) = Ce·P(escape) + Cb·P(falso bloqueo) + Cr·P(retención)    (Elkan, 2001)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DecisionPK:
    estado: str      # LIBERADO | RETENIDO | BLOQUEADO
    regla: str       # R1..R6
    alerta: str      # VERDE | AMBAR | ROJA
    motivo: str
    defecto_principal: str | None
    puntaje: float


def decidir(dets, tau1, tau2, captura_ok=True, motivos_captura=(), metadatos_ok=True, faltantes=()):
    if not captura_ok:
        return DecisionPK("RETENIDO", "R1", "AMBAR", "Captura no apta: " + "; ".join(motivos_captura) +
                          ". Repita la captura.", None, 0.0)
    if not metadatos_ok:
        return DecisionPK("RETENIDO", "R2", "AMBAR", "Metadatos incompletos: " + ", ".join(faltantes) + ".", None, 0.0)
    if not dets:
        return DecisionPK("LIBERADO", "R5", "VERDE", "Sin anomalías: la caja puede pasar a montaje.", None, 0.0)
    top = max(dets, key=lambda d: d.puntaje)
    no_sold = [d for d in dets if d.clase != "soldadura_visible"]
    top_ns = max(no_sold, key=lambda d: d.puntaje) if no_sold else None
    if top_ns and top_ns.puntaje >= tau2:
        return DecisionPK("BLOQUEADO", "R3", "ROJA", f"{top_ns.clase} con puntaje {top_ns.puntaje:.2f} ≥ τ2 = {tau2:.2f}: "
                          "bloqueo de montaje y derivación a reproceso.", top_ns.clase, top_ns.puntaje)
    sold = [d for d in dets if d.clase == "soldadura_visible" and d.puntaje >= tau1]
    if sold:
        d = max(sold, key=lambda x: x.puntaje)
        return DecisionPK("RETENIDO", "R6", "AMBAR", f"Soldadura visible (s = {d.puntaje:.2f}): verificar con el método "
                          "adecuado; la imagen no evalúa integridad interna.", d.clase, d.puntaje)
    if top.puntaje >= tau1:
        return DecisionPK("RETENIDO", "R4", "AMBAR", f"{top.clase} con puntaje {top.puntaje:.2f} entre τ1 y τ2: "
                          "requiere dictamen del supervisor.", top.clase, top.puntaje)
    return DecisionPK("LIBERADO", "R5", "VERDE", "Anomalías por debajo de τ1: la caja puede pasar a montaje.", None, top.puntaje)


def costo_esperado(s, y, tau1, tau2, ce, cb, cr):
    """Costo esperado por unidad. s: puntajes por caja; y: 1 si la caja es defectuosa."""
    s, y = np.asarray(s), np.asarray(y)
    escape = (y == 1) & (s < tau1)
    falso_bloqueo = (y == 0) & (s >= tau2)
    retencion = (s >= tau1) & (s < tau2)
    return float((ce * escape.sum() + cb * falso_bloqueo.sum() + cr * retencion.sum()) / len(s))


def calibrar(s, y, ce, cb, cr, paso=0.05):
    rejilla = np.round(np.arange(paso, 1.0, paso), 2)
    mejor, tabla = None, []
    for t1 in rejilla:
        for t2 in rejilla:
            if t2 <= t1:
                continue
            c = costo_esperado(s, y, t1, t2, ce, cb, cr)
            tabla.append((float(t1), float(t2), c))
            if mejor is None or c < mejor[2] - 1e-12 or (abs(c - mejor[2]) < 1e-12 and t2 - t1 < mejor[1] - mejor[0]):
                mejor = (float(t1), float(t2), c)
    return {"tau1": mejor[0], "tau2": mejor[1], "costo_esperado": round(mejor[2], 2), "tabla": tabla}


class SalidaSenal:
    """Señal lógica del Poka-Yoke: alerta visual en la interfaz y bloqueo del registro en montaje.
    No requiere actuador físico; la misma cadena podría enviarse a un actuador opcional a futuro."""

    def emitir(self, d: DecisionPK) -> str:
        return {"VERDE": "LUZ_VERDE;MONTAJE_HABILITADO", "AMBAR": "LUZ_AMBAR;MONTAJE_BLOQUEADO_PENDIENTE",
                "ROJA": "LUZ_ROJA;SIRENA_2S;MONTAJE_BLOQUEADO"}[d.alerta]
