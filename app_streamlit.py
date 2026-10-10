"""PokaVision-IE v2 — Interfaz Streamlit del DSS operacional híbrido.

Ejecutar:  streamlit run app_streamlit.py
Requisitos previos: python generar_registros_proyectados.py, python datos_sinteticos.py y python entrenar_modelo.py
Usuarios de demostración: operario/1111 · supervisor/2222 · jefe/3333 · montaje/4444
Misma lógica que la versión Flask (app.py): ambas usan los módulos M1–M7 de la carpeta modulos/.
"""
import base64
import json
import os
from datetime import date, timedelta

import cv2
import streamlit as st

from config import CLASE_NOMBRE, DATA_DIR, ETAPAS, FASES, IMG_STORE, MATRIZ_ACCION, MODELOS, PARAM_FILE, VISTAS
from modulos import adquisicion, dashboard, importar_excel, pipeline, registro, trazabilidad

st.set_page_config(page_title="PokaVision-IE", page_icon="🔍", layout="wide")
COLOR = {"VERDE": "#2E8B57", "AMBAR": "#E6A100", "ROJA": "#C0392B"}


@st.cache_resource
def preparar():
    registro.inicializar()
    with registro.conectar() as con:
        if not con.execute("SELECT 1 FROM usuarios").fetchone():
            for u, n, r, p in [("operario", "Operario de inspección", "operario", "1111"),
                               ("supervisor", "Supervisor de calidad", "supervisor", "2222"),
                               ("jefe", "Jefe de producción", "jefe", "3333"),
                               ("montaje", "Responsable de montaje", "montaje", "4444")]:
                registro.crear_usuario(u, n, r, p)
    with registro.conectar() as con:
        vacio = not con.execute("SELECT 1 FROM historico").fetchone()
    if vacio:                       # primera ejecución (Colab o Streamlit Cloud): carga la línea base 2025
        try:
            importar_excel.importar()
        except Exception:
            pass
    pipeline.detector()
    return True


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png(b64):
    st.image(base64.b64decode(b64))


def caja_color(color, titulo, texto):
    st.markdown(f"<div style='background:{color};color:white;padding:16px;border-radius:10px;text-align:center'>"
                f"<div style='font-size:24px;font-weight:700'>{titulo}</div><div>{texto}</div></div>",
                unsafe_allow_html=True)


try:
    preparar()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

if "usuario" not in st.session_state:
    st.title("🔍 PokaVision-IE")
    st.caption("DSS operacional híbrido · Poka-Yoke de control · visión artificial · trazabilidad digital")
    with st.form("login"):
        u = st.text_input("Usuario"); p = st.text_input("PIN", type="password")
        if st.form_submit_button("Ingresar"):
            r = registro.autenticar(u.strip(), p)
            if r:
                st.session_state.usuario = r; registro.auditar(r["usuario"], "LOGIN"); st.rerun()
            st.error("Usuario o PIN incorrecto.")
    st.stop()

U = st.session_state.usuario
ROL = U["rol"]
VISTAS_ROL = {"operario": ["Inspección", "Trazabilidad"],
              "supervisor": ["Inspección", "Revisión", "Montaje", "Trazabilidad", "Panel DSS", "Histórico", "Acciones"],
              "jefe": ["Panel DSS", "Histórico", "Acciones", "Inspección", "Revisión", "Montaje", "Trazabilidad",
                       "Datos CP0", "Configuración"],
              "montaje": ["Montaje"]}
with st.sidebar:
    st.markdown(f"**{U['nombre']}** · {ROL}")
    vista = st.radio("Menú", VISTAS_ROL[ROL])
    t1, t2 = pipeline.umbrales()
    st.caption(f"τ1 = {t1} · τ2 = {t2}")
    if st.button("Cerrar sesión"):
        del st.session_state.usuario; st.rerun()

if vista == "Inspección":
    st.header("Estación de inspección")
    c1, c2 = st.columns([1, 2.4])
    DEMO = DATA_DIR / "demo"
    demos = sorted(f for f in os.listdir(DEMO) if f.endswith(".jpg")) if DEMO.exists() else []
    with c1:
        fuente = st.radio("Imagen", ["Subir foto", "Cámara", "Demostración"], horizontal=True)
    with c1, st.form("insp"):
        caja = st.text_input("ID de caja (QR)", placeholder=f"CAJ-{date.today().year}-0154")
        lote = st.text_input("Lote")
        modelo = st.selectbox("Modelo", list(MODELOS), format_func=lambda k: f"{k} · {MODELOS[k]['descripcion']}")
        fase = st.selectbox("Fase", list(FASES), format_func=FASES.get)
        vis = st.selectbox("Vista", VISTAS)
        origen = st.selectbox("Origen del dato", ["REAL", "DEMO"], index=1 if fuente == "Demostración" else 0)
        if fuente == "Cámara":
            foto = st.camera_input("Cámara")
        elif fuente == "Demostración":
            elegida = st.selectbox("Imagen de demostración", demos)
            foto = None
        else:
            foto = st.file_uploader("Imagen", ["jpg", "jpeg", "png"])
        ok = st.form_submit_button("Inspeccionar")
    with c2:
        if ok:
            try:
                datos = (DEMO / elegida).read_bytes() if fuente == "Demostración" and demos else (foto.getvalue() if foto else b"")
                img = adquisicion.decodificar(datos)
                out = pipeline.inspeccionar(img, caja_id=caja, lote=lote, modelo_id=modelo, fase=fase, vista=vis,
                                            operario=U["usuario"], origen=origen)
                d, r = out["decision"], out["registro"]
                a, b, c = st.columns(3)
                with a:
                    caja_color(COLOR[d.alerta], d.estado, f"Regla {d.regla} · {d.motivo}")
                b.metric("Puntaje de defecto s", f"{d.puntaje:.2f}"); c.metric("Tiempo de respuesta", f"{r['tiempo_respuesta_ms']} ms")
                if out["proc"] is not None:
                    i1, i2, i3, i4 = st.columns(4)
                    i1.image(rgb(out["proc"].color), caption="Adquirida"); i2.image(rgb(out["proc"].pasos["roi"]), caption="ROI")
                    i3.image(out["proc"].pasos["residuo"], caption="Mapa de anomalía")
                    i4.image(rgb(cv2.imread(str(IMG_STORE / r["imagen_anotada"]))), caption="Clasificación")
                else:
                    st.warning("No se analizó: " + "; ".join(out["captura"].motivos or ["metadatos incompletos"]))
                st.caption(f"Registro {r['id_registro']} · {r['fecha_hora']} · señal {r['senal']} · hash {r['hash'][:16]}…")
            except adquisicion.ErrorAdquisicion as e:
                st.error(str(e))

elif vista == "Revisión":
    st.header("Zona de revisión — RETENIDAS / BLOQUEADAS")
    df = trazabilidad.consultar()
    pend = df[(df.estado != "LIBERADO") & df.dictamen.isna() & ~df.regla.isin(["R1", "R2"])].head(20)
    if pend.empty:
        st.info("No hay cajas pendientes de dictamen.")
    for _, r in pend.iterrows():
        with st.expander(f"{r.caja_id} · {r.estado} ({r.regla}) · {CLASE_NOMBRE.get(r.defecto_principal, '-')} · s = {r.puntaje:.2f}"):
            c1, c2 = st.columns([1, 2])
            c1.image(str(IMG_STORE / r.imagen_anotada))
            with c2, st.form(f"d{r.id_registro}"):
                dic = st.selectbox("Dictamen", ["DEFECTO_CONFIRMADO", "FALSA_ALARMA"])
                acc = st.selectbox("Acción", ["REPROCESO", "LIBERAR", "DESECHO", "REINSPECCION"])
                sug = MATRIZ_ACCION.get(r.defecto_principal, {}).get("etapa")
                eta = st.selectbox("Etapa de origen", ETAPAS, index=ETAPAS.index(sug) if sug in ETAPAS else 0)
                costo = st.number_input("Costo de reproceso (S/)", value=600.0)
                obs = st.text_input("Observación")
                if st.form_submit_button("Registrar dictamen"):
                    try:
                        registro.registrar_disposicion(r.id_registro, U["usuario"], dic, acc,
                                                       eta if dic == "DEFECTO_CONFIRMADO" else None, costo, obs)
                        st.success("Dictamen registrado."); st.rerun()
                    except registro.ErrorValidacion as e:
                        st.error(str(e))

elif vista == "Montaje":
    st.header("Compuerta de ingreso a montaje (Poka-Yoke de control)")
    caja = st.text_input("ID de caja (lectura QR)")
    if st.button("Registrar ingreso") and caja:
        ok, estado, det = registro.ingresar_montaje(caja, U["usuario"])
        caja_color("#2E8B57" if ok else "#C0392B", "INGRESO ADMITIDO" if ok else "INGRESO BLOQUEADO",
                   f"{caja.upper()} · {estado or 'SIN INSPECCIÓN'} · {det}")

elif vista == "Trazabilidad":
    st.header("Trazabilidad digital por caja")
    c = st.columns(5)
    f = {"caja_id": c[0].text_input("ID caja"), "lote": c[1].text_input("Lote"),
         "modelo_id": c[2].selectbox("Modelo", [""] + list(MODELOS)),
         "estado": c[3].selectbox("Estado", ["", "LIBERADO", "RETENIDO", "BLOQUEADO"]),
         "origen": c[4].selectbox("Origen", ["", "REAL", "SINTÉTICO", "DEMO"])}
    df = trazabilidad.consultar(**{k: v or None for k, v in f.items()})
    st.caption(f"{len(df)} registro(s) · completitud K08 = {trazabilidad.completitud(df)} %")
    if f["caja_id"]:
        st.table(trazabilidad.historial_caja(f["caja_id"]))
    st.dataframe(df[["fecha_hora", "caja_id", "lote", "modelo_id", "fase", "vista", "estado", "regla",
                     "defecto_principal", "puntaje", "dictamen", "etapa_origen", "origen"]])
    st.download_button("Exportar CSV", trazabilidad.exportar_csv(df), "trazabilidad_pokavision.csv")

elif vista == "Panel DSS":
    st.header("Panel DSS operativo")
    c = st.columns(3)
    f = {"modelo_id": c[0].selectbox("Modelo", [""] + list(MODELOS)), "fase": c[1].selectbox("Fase", [""] + list(FASES)),
         "lote": c[2].text_input("Lote")}
    d = dashboard.operativa(f); k = d["kpi"]
    if k["total"]:
        m = st.columns(6)
        m[0].metric("Inspecciones", k["total"]); m[1].metric("K05 Sensibilidad", f"{k.get('sensibilidad', '—')} %")
        m[2].metric("K06 Falsas alarmas", f"{k.get('falsas_alarmas', '—')} %"); m[3].metric("Kappa", k.get("kappa", "—"))
        m[4].metric("K08 Trazabilidad", f"{k['completitud']} %"); m[5].metric("K09 Acciones", f"{k.get('acciones_cerradas', '—')} %")
        for g in d["graficos"].values():
            png(g)
    st.subheader("Recomendaciones para la decisión táctica")
    for r in d["recomendaciones"]:
        st.markdown(f"- {r}")

elif vista == "Histórico":
    st.header("Vista histórica")
    h = dashboard.historica(st.selectbox("Modelo", [""] + list(MODELOS)) or None)
    st.subheader("A. Línea base 2025 — registros generados a partir del escenario proyectado (no son mediciones)")
    s = h["sintetico"]
    if s:
        m = st.columns(4)
        m[0].metric("Producidas", s["producidas"]); m[1].metric("TD", f"{s['td']} %")
        m[2].metric("Costo AS-IS", f"S/ {s['costo']:,.0f}"); m[3].metric("Ahorro bruto teórico", f"S/ {s['ahorro_bruto']:,.0f}")
        for g in s["graficos"].values():
            png(g)
    else:
        st.info("Importe los registros en «Datos CP0».")
    st.subheader("B. Escenario proyectado 2023–2026 (TF1, Tabla 4.2)")
    st.table(h["real"]["tabla"]); png(h["real"]["grafico"])

elif vista == "Acciones":
    st.header("Acciones DSS (decisión táctica)")
    with st.form("acc"):
        c = st.columns(4)
        eta = c[0].selectbox("Etapa", ETAPAS); dfe = c[1].selectbox("Defecto", [""] + list(CLASE_NOMBRE))
        resp = c[2].text_input("Responsable"); fc = c[3].date_input("Compromiso", date.today() + timedelta(days=7))
        desc = st.text_input("Descripción", value=MATRIZ_ACCION.get(dfe, {}).get("accion", ""))
        if st.form_submit_button("Registrar"):
            try:
                registro.crear_accion(eta, dfe or None, desc, resp, fc.isoformat()); st.rerun()
            except registro.ErrorValidacion as e:
                st.error(str(e))
    for a in registro.acciones():
        c = st.columns([4, 1])
        c[0].write(f"**#{a['id']} · {a['estado']}** · {a['etapa']} · {a['descripcion']} · {a['responsable']} · {a['fecha_compromiso']}")
        if a["estado"] == "ABIERTA" and c[1].button("Cerrar", key=f"c{a['id']}"):
            registro.cerrar_accion(a["id"], "Cerrada desde el panel"); st.rerun()

elif vista == "Datos CP0":
    st.header("CP0 — Importación y conciliación de los registros del escenario proyectado (año 2025)")
    if st.button("Importar REGISTRO_CAJAS"):
        rep = importar_excel.importar()
        (st.success if rep["ok"] else st.error)("Conciliación " + ("PASA" if rep["ok"] else "NO PASA"))
        st.table({k: {"esperado": v["esperado"], "obtenido": v["obtenido"]} for k, v in rep["controles"].items()})

elif vista == "Configuración":
    st.header("Regla de dos umbrales")
    t1, t2 = pipeline.umbrales()
    a = st.number_input("τ1 (retener)", 0.01, 0.98, float(t1), 0.05); b = st.number_input("τ2 (bloquear)", 0.02, 0.99, float(t2), 0.05)
    if st.button("Guardar"):
        if 0 < a < b < 1:
            json.dump({"tau1": a, "tau2": b, "fuente": "ajuste manual"}, open(PARAM_FILE, "w"))
            registro.auditar(U["usuario"], "CAMBIO_UMBRALES", f"{a},{b}"); st.success("Guardado.")
        else:
            st.error("Debe cumplirse 0 < τ1 < τ2 < 1.")
