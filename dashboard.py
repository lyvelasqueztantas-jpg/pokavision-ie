"""M6–M7 — Herramienta analítica del DSS (decisión táctica: ¿sobre qué etapa actuar?).

Dos vistas, nunca mezcladas:
  · Vista OPERATIVA: inspecciones registradas por el sistema (estados, reglas, tiempos,
    sensibilidad/falsas alarmas, Kappa, completitud, acciones cerradas, gráfico p por
    etapa de origen, Pareto, recomendaciones de la matriz defecto–causa–acción).
  · Vista HISTÓRICA: (a) línea base REAL del TF1 2023–2026 y (b) escenario SINTÉTICO del
    Excel maestro (1 250 cajas) usado solo para probar el panel (CP0/CP6).
"""
import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASE_NOMBRE, ETAPAS, LINEA_BASE, MATRIZ_ACCION, META_TD, METRICS_PATH, PARETO_REAL
from modulos import registro
from modulos.importar_excel import historico
from modulos.trazabilidad import completitud, consultar

AZUL, ROJO, AMBAR, VERDE, GRIS = "#1F4E79", "#C0392B", "#E6A100", "#2E8B57", "#7F8C8D"
COSTO_EXCEL = 600


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- utilidades estadísticas
def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return None
    po = (a == b).mean()
    cats = np.union1d(a, b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return round(float((po - pe) / (1 - pe)), 3) if pe < 1 else 1.0


def grafico_p(conteos, tamanos):
    """Límites de control del gráfico p (Montgomery): p̄ ± 3·√(p̄(1−p̄)/n_i)."""
    d, n = np.asarray(conteos, float), np.asarray(tamanos, float)
    p = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
    pbar = d.sum() / n.sum() if n.sum() else 0
    sd = np.sqrt(pbar * (1 - pbar) / np.where(n > 0, n, 1))
    lcs, lci = pbar + 3 * sd, np.clip(pbar - 3 * sd, 0, None)
    alertas = [i for i in range(len(p)) if p[i] > lcs[i]]
    # regla de racha: 7 puntos consecutivos sobre la línea central
    racha = 0
    for i, v in enumerate(p):
        racha = racha + 1 if v > pbar else 0
        if racha >= 7 and i not in alertas:
            alertas.append(i)
    return p, pbar, lcs, lci, sorted(alertas)


def _fig_p(ax, etiquetas, p, pbar, lcs, lci, alertas, titulo):
    x = range(len(p))
    ax.plot(x, p * 100, "o-", color=AZUL, ms=4)
    ax.step(x, lcs * 100, where="mid", color=ROJO, ls="--", lw=1, label="LCS")
    ax.step(x, lci * 100, where="mid", color=ROJO, ls="--", lw=1, label="LCI")
    ax.axhline(pbar * 100, color=VERDE, lw=1, label=f"p̄ = {pbar*100:.1f} %")
    for i in alertas:
        ax.plot(i, p[i] * 100, "o", color=ROJO, ms=9, mfc="none", mew=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(etiquetas, fontsize=7, rotation=45)
    ax.set_title(titulo, fontsize=9); ax.set_ylabel("% no conforme", fontsize=8)


def _pareto(ax, nombres, vals, titulo):
    acum = np.cumsum(vals) / sum(vals) * 100
    ax.bar(range(len(vals)), vals, color=[ROJO if a <= 80.5 or i == 0 else GRIS for i, a in enumerate(acum)])
    ax2 = ax.twinx(); ax2.plot(range(len(vals)), acum, "o-", color=AZUL); ax2.set_ylim(0, 110)
    ax2.axhline(80, color=GRIS, ls=":", lw=1); ax2.set_ylabel("% acumulado", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax2.text(i, acum[i] + 3, f"{acum[i]:.0f}%", ha="center", fontsize=7, color=AZUL)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(nombres, fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, max(vals) * 1.2); ax.set_title(titulo, fontsize=9)


# ---------------------------------------------------------------- vista OPERATIVA
def operativa(filtros=None):
    filtros = {k: v for k, v in (filtros or {}).items() if v}
    df = consultar(**filtros)
    k = {"total": len(df)}
    acc = pd.DataFrame(registro.acciones())
    k["acciones_total"] = len(acc)
    if len(acc):
        cerr = acc[acc.estado == "CERRADA"]
        a_tiempo = (cerr.fecha_cierre.str[:10] <= cerr.fecha_compromiso).sum()
        k["acciones_cerradas"] = round(100 * a_tiempo / len(acc), 1)          # K09
    if df.empty:
        return {"kpi": k, "graficos": {}, "recomendaciones": ["Sin inspecciones para los filtros seleccionados."],
                "modelo": metricas_modelo(), "alertas": []}
    for e in registro.ESTADOS:
        k[e.lower()] = int((df.estado == e).sum())
    k["pct_liberado"] = round(100 * k["liberado"] / len(df), 1)
    k["reglas"] = df.regla.value_counts().sort_index().to_dict()
    k["t_resp"] = round(df.tiempo_respuesta_ms.mean(), 1)
    k["t_resp_p95"] = round(df.tiempo_respuesta_ms.quantile(.95), 1)
    k["completitud"] = completitud(df)                                         # K08
    # K05/K06 contra verdad de terreno (ensayos) — decisión "no liberar" vs caja defectuosa
    gt = df[df.verdad_terreno.notna() & (df.regla.isin(["R3", "R4", "R5", "R6"]))]
    if len(gt):
        pred = gt.estado != "LIBERADO"; real = gt.verdad_terreno == "DEFECTUOSA"
        vp, fn = int((pred & real).sum()), int((~pred & real).sum())
        fp, vn = int((pred & ~real).sum()), int((~pred & ~real).sum())
        k["sensibilidad"] = round(100 * vp / max(vp + fn, 1), 1)
        k["falsas_alarmas"] = round(100 * fp / max(fp + vn, 1), 1)
        k["escapes"] = fn
        k["kappa"] = kappa(pred.astype(int), real.astype(int))
        oblig = gt[real & gt.puntaje.ge(gt.tau2) & (gt.defecto_principal != "soldadura_visible")]
        k["bloqueo_correcto"] = round(100 * (oblig.estado == "BLOQUEADO").mean(), 1) if len(oblig) else None  # K07
    disp = df[df.dictamen.notna()]
    k["pendientes"] = int(((df.estado != "LIBERADO") & df.dictamen.isna() & ~df.regla.isin(["R1", "R2"])).sum())
    if len(disp):
        k["confirmados"] = int((disp.dictamen == "DEFECTO_CONFIRMADO").sum())
        k["costo_reproceso"] = float(disp.costo_reproceso.fillna(0).sum())
    # Pareto por tipo (retenidas/bloqueadas)
    cont = {}
    for det, est in zip(df.detecciones, df.estado):
        if est == "LIBERADO":
            continue
        for d in json.loads(det):
            if d["puntaje"] >= (df.tau1.iloc[0] or .35):
                cont[d["clase"]] = cont.get(d["clase"], 0) + 1
    cont = dict(sorted(cont.items(), key=lambda x: -x[1]))
    k["frecuencia"] = cont
    graf, alertas = {}, []
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    v = [k["liberado"], k["retenido"], k["bloqueado"]]
    axs[0].pie(v, colors=[VERDE, AMBAR, ROJO], startangle=90, wedgeprops=dict(width=.45))
    axs[0].legend([f"{l}: {x}" for l, x in zip(["Liberado", "Retenido", "Bloqueado"], v)], loc="center left",
                  bbox_to_anchor=(.95, .5), fontsize=8, frameon=False)
    axs[0].set_title("Estados Poka-Yoke", fontsize=9)
    if cont:
        _pareto(axs[1], [CLASE_NOMBRE[c] for c in cont], list(cont.values()), "Pareto de defectos (cajas no liberadas)")
    else:
        axs[1].axis("off")
    graf["estados_pareto"] = _png(fig)
    # gráfico p por etapa de origen (semanal, con disposiciones confirmadas)
    d = df.copy(); d["semana"] = pd.to_datetime(d.fecha_hora).dt.to_period("W-SUN").dt.start_time
    cajas = d.groupby("semana").caja_id.nunique()
    conf = d[d.dictamen == "DEFECTO_CONFIRMADO"]
    etapas = [e for e in ETAPAS if e in set(conf.etapa_origen)] or []
    if etapas and len(cajas) >= 2:
        fig, axs = plt.subplots(1, len(etapas), figsize=(3.6 * len(etapas), 2.8), squeeze=False)
        for ax, e in zip(axs[0], etapas):
            c = conf[conf.etapa_origen == e].groupby("semana").caja_id.nunique().reindex(cajas.index, fill_value=0)
            p, pbar, lcs, lci, al = grafico_p(c.values, cajas.values)
            _fig_p(ax, [x.strftime("%d/%m") for x in cajas.index], p, pbar, lcs, lci, al, f"Gráfico p · {e}")
            for i in al:
                alertas.append(f"Etapa «{e}», semana del {cajas.index[i]:%d/%m}: proporción {p[i]*100:.1f} % fuera de control.")
        fig.tight_layout(); graf["p_etapas"] = _png(fig)
    return {"kpi": k, "graficos": graf, "recomendaciones": recomendaciones(cont, conf, k, alertas),
            "modelo": metricas_modelo(), "alertas": alertas}


def recomendaciones(cont, conf, k, alertas):
    rec = []
    if len(conf):
        top = conf.etapa_origen.value_counts()
        e, n = top.index[0], int(top.iloc[0])
        acc = next((m["accion"] for m in MATRIZ_ACCION.values() if m["etapa"] == e), "analizar la causa raíz")
        rec.append(f"Decisión táctica: actuar primero sobre «{e}» ({n} de {len(conf)} defectos confirmados, "
                   f"{100*n/len(conf):.0f} %). Acción sugerida: {acc}.")
    elif cont:
        c = next(iter(cont)); m = MATRIZ_ACCION[c]
        rec.append(f"{CLASE_NOMBRE[c]} es el defecto más frecuente; etapa probable «{m['etapa']}» "
                   f"(causa: {m['causa'].lower()}). Acción: {m['accion']}.")
    rec += [f"ALERTA: {a}" for a in alertas]
    if k.get("falsas_alarmas") is not None and k["falsas_alarmas"] > 10:
        rec.append(f"Falsas alarmas {k['falsas_alarmas']} % > 10 % (K06): recalibrar τ1/τ2 o reentrenar con imágenes revisadas.")
    if k.get("sensibilidad") is not None and k["sensibilidad"] < 90:
        rec.append(f"Sensibilidad {k['sensibilidad']} % < 90 % (K05): bajar τ1 o ampliar el dataset de la clase con más escapes.")
    if k.get("pendientes"):
        rec.append(f"{k['pendientes']} caja(s) retenida(s)/bloqueada(s) sin dictamen del supervisor.")
    if k.get("completitud") is not None and k["completitud"] < 98:
        rec.append(f"Completitud de trazabilidad {k['completitud']} % < 98 % (K08): revisar lectura QR y metadatos (R2).")
    if k.get("acciones_cerradas") is not None and k["acciones_cerradas"] < 90:
        rec.append(f"Solo {k['acciones_cerradas']} % de acciones cerradas a tiempo (K09 ≥ 90 %).")
    return rec or ["Sin desviaciones relevantes en el periodo."]


def metricas_modelo():
    try:
        return json.load(open(METRICS_PATH))
    except Exception:
        return None


# ---------------------------------------------------------------- vista HISTÓRICA
def historica(modelo=None, mes=None):
    out = {"real": linea_base_real()}
    h = historico()
    if h.empty:
        out["sintetico"] = None
        return out
    if modelo:
        h = h[h.modelo_id == modelo]
    if mes:
        h = h[h.mes == str(mes)]
    nc = h[h.estado != "Conforme"]
    s = {"producidas": len(h), "defectuosas": len(nc), "td": round(100 * len(nc) / max(len(h), 1), 2),
         "costo": float(nc.costo_reproceso.sum())}
    s["pareto"] = nc.tipo_defecto.value_counts().to_dict()
    tot = sum(s["pareto"].values()) or 1
    acum, s["pareto_acum"] = 0, {}
    for t, v in s["pareto"].items():
        acum += v; s["pareto_acum"][t] = round(100 * acum / tot, 1)
    s["etapas"] = nc.etapa_origen_prueba.dropna().replace("", np.nan).dropna().value_counts().to_dict()
    # TO-BE: meta estrictamente < 4 %
    obj = int(np.floor(0.04 * len(h) - 1e-9)) if len(h) else 0
    s["tobe_defectos"], s["tobe_costo"] = obj, obj * COSTO_EXCEL
    s["ahorro_bruto"] = s["costo"] - s["tobe_costo"]
    graf = {}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2))
    if s["pareto"]:
        _pareto(axs[0], list(s["pareto"]), list(s["pareto"].values()), "Pareto 80/20 · línea base 2025 (escenario proyectado)")
    mens = h.groupby(h.mes.astype(int)).apply(lambda x: pd.Series({"n": len(x), "d": (x.estado != "Conforme").sum()}),
                                              include_groups=False)
    if len(mens) >= 2:
        p, pbar, lcs, lci, al = grafico_p(mens.d.values, mens.n.values)
        _fig_p(axs[1], [str(m) for m in mens.index], p, pbar, lcs, lci, al, "Gráfico p mensual · escenario proyectado")
        axs[1].axhline(META_TD, color=GRIS, ls=":", lw=1); axs[1].set_xlabel("Mes", fontsize=8)
        s["alertas_p"] = [int(mens.index[i]) for i in al]
    else:
        axs[1].axis("off")
    fig.tight_layout(); graf["pareto_p"] = _png(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.0))
    et = pd.Series(s["etapas"]).sort_values()
    axs[0].barh(et.index, et.values, color=AZUL); axs[0].set_title("No conformes por etapa de origen (asignada solo para pruebas)", fontsize=9)
    for i, v in enumerate(et.values):
        axs[0].text(v, i, f" {v}", va="center", fontsize=8)
    axs[1].bar(["AS-IS", "TO-BE (<4 %)"], [s["costo"], s["tobe_costo"]], color=[ROJO, VERDE])
    for i, v in enumerate([s["costo"], s["tobe_costo"]]):
        axs[1].text(i, v, f"S/ {v:,.0f}", ha="center", va="bottom", fontsize=8)
    axs[1].set_title(f"Costo de reproceso · ahorro bruto teórico S/ {s['ahorro_bruto']:,.0f}", fontsize=9)
    fig.tight_layout(); graf["etapas_costos"] = _png(fig)
    s["graficos"] = graf
    out["sintetico"] = s
    return out


def linea_base_real():
    """Escenario proyectado 2023–2026 (TF1, Tabla 4.2); la función conserva su nombre por compatibilidad."""
    from config import HISTORICO_SINT, SERIE_SINT
    filas = [{"anio": a, "producidas": p, "defectuosas": d, "td": round(100 * d / p, 1), "costo": c}
             for a, (p, d, c) in SERIE_SINT.items()]
    try:
        par = pd.read_excel(HISTORICO_SINT, sheet_name="PARETO_HISTORICO")
        pareto = dict(zip(par.Tipo_defecto, par.Total))
    except Exception:
        pareto = {}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.0))
    axs[0].bar([str(f["anio"]) + ("*" if f["anio"] == 2026 else "") for f in filas], [f["td"] for f in filas], color=AZUL)
    axs[0].axhline(META_TD, color=VERDE, ls="--", label="Meta < 4 %"); axs[0].legend(fontsize=7, loc="lower right")
    axs[0].set_ylim(0, 15)
    for i, f in enumerate(filas):
        axs[0].text(i, f["td"], f"{f['td']} %", ha="center", va="bottom", fontsize=8)
    axs[0].set_title("Tasa de defectos · escenario proyectado 2023–2026 (*ene–may)", fontsize=9)
    if pareto:
        _pareto(axs[1], list(pareto), list(pareto.values()), f"Pareto acumulado 2023–2026 · distribución supuesta ({sum(pareto.values())} cajas)")
    else:
        axs[1].axis("off")
    fig.tight_layout()
    return {"tabla": filas, "grafico": _png(fig)}
