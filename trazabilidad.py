"""M5/M6 — Trazabilidad digital.

Consolida el historial de cada caja (inspecciones por fase y vista, disposiciones con etapa
de origen e intentos de ingreso a montaje), permite consultas por caja, lote, modelo, fase,
estado, defecto, origen del dato y fechas, y exporta a CSV.
"""
import io
import json

import pandas as pd

from modulos.registro import conectar

CONSULTA = """
SELECT i.*, d.dictamen, d.accion AS accion_supervisor, d.supervisor, d.fecha_hora AS fecha_disposicion,
       d.etapa_origen, d.costo_reproceso, d.observacion
FROM inspecciones i LEFT JOIN disposiciones d ON d.id_registro = i.id_registro
"""
CAMPOS_OBLIGATORIOS = ["caja_id", "lote", "modelo_id", "fase", "vista", "fecha_hora", "operario", "estado",
                       "regla", "hash"]


def consultar(caja_id=None, lote=None, modelo_id=None, fase=None, estado=None, defecto=None, origen=None,
              desde=None, hasta=None):
    cond, par = [], []
    for col, val, op in [("i.caja_id", caja_id and f"%{caja_id.strip().upper()}%", "LIKE"), ("i.lote", lote, "="),
                         ("i.modelo_id", modelo_id, "="), ("i.fase", fase, "="), ("i.estado", estado, "="),
                         ("i.defecto_principal", defecto, "="), ("i.origen", origen, "="),
                         ("i.fecha_hora", desde, ">="), ("i.fecha_hora", hasta and hasta + "T23:59:59", "<=")]:
        if val:
            cond.append(f"{col} {op} ?"); par.append(val)
    sql = CONSULTA + (" WHERE " + " AND ".join(cond) if cond else "") + " ORDER BY i.fecha_hora DESC"
    with conectar() as con:
        return pd.read_sql_query(sql, con, params=par)


def completitud(df):
    """K08: % de registros con todos los campos obligatorios válidos."""
    if df.empty:
        return None
    ok = df[CAMPOS_OBLIGATORIOS].notna().all(axis=1) & (df.caja_id != "SIN-ID") & (df.lote != "SIN-LOTE") \
        & (df.modelo_id != "?")
    return round(100 * ok.mean(), 1)


def historial_caja(caja_id):
    df = consultar(caja_id=caja_id).sort_values("fecha_hora")
    df = df[df.caja_id == caja_id.upper()]
    ev = []
    for _, r in df.iterrows():
        ev.append({"fecha": r.fecha_hora, "tipo": f"Inspección {r.fase.replace('_', '/')} · {r.vista}",
                   "detalle": f"{r.estado} ({r.regla}) · {r.defecto_principal or 'sin defecto'} · s = {r.puntaje:.2f}",
                   "registro": r.id_registro})
        if isinstance(r.dictamen, str):
            ev.append({"fecha": r.fecha_disposicion, "tipo": "Disposición del supervisor",
                       "detalle": f"{r.dictamen} → {r.accion_supervisor} · origen: {r.etapa_origen or '—'} ({r.supervisor})",
                       "registro": r.id_registro})
    with conectar() as con:
        for m in con.execute("SELECT * FROM ingresos_montaje WHERE caja_id=?", (caja_id.upper(),)):
            ev.append({"fecha": m["fecha_hora"], "tipo": "Intento de ingreso a montaje",
                       "detalle": f"{m['resultado']} · {m['motivo']} ({m['usuario']})", "registro": ""})
    return sorted(ev, key=lambda e: e["fecha"])


def exportar_csv(df):
    cols = ["id_registro", "caja_id", "lote", "modelo_id", "fase", "vista", "fecha_hora", "operario", "estado",
            "regla", "defecto_principal", "puntaje", "tau1", "tau2", "tiempo_respuesta_ms", "dictamen",
            "accion_supervisor", "etapa_origen", "costo_reproceso", "supervisor", "fecha_disposicion", "origen", "hash"]
    buf = io.StringIO()
    df[[c for c in cols if c in df.columns]].to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")


def detecciones(row):
    return json.loads(row["detecciones"]) if row["detecciones"] else []
