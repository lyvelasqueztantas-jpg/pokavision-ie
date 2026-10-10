"""M5 — Subsistema de datos: registro automático de inspección y trazabilidad (SQLite).

Cada inspección se guarda con identificador único, fecha/hora del sistema (dinámica),
modelo, fase, vista, lote, caja (QR), estado LIBERADO/RETENIDO/BLOQUEADO, regla aplicada,
puntaje, umbrales vigentes, origen del dato (REAL / SINTÉTICO / DEMO) y hash SHA-256.
Incluye: disposiciones del supervisor (con etapa de origen), compuerta de ingreso a montaje
(Poka-Yoke de control), acciones DSS con responsable y cierre, histórico importado del Excel
(CP0), usuarios con PIN cifrado (PBKDF2) y auditoría.
"""
import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime

from config import DB_PATH, ETAPAS, FASES, MODELOS, ROLES, VISTAS

PATRON_CAJA = re.compile(r"^CAJ-(\d{4}|DS)-\d{3,5}$")   # CAJ-2025-0001 (Excel) · CAJ-DS-0001 (dataset)
ESTADOS = ("LIBERADO", "RETENIDO", "BLOQUEADO")
ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuarios(
  usuario TEXT PRIMARY KEY, nombre TEXT NOT NULL, rol TEXT NOT NULL, pin_hash TEXT NOT NULL, sal TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inspecciones(
  id_registro TEXT PRIMARY KEY, caja_id TEXT NOT NULL, lote TEXT NOT NULL, modelo_id TEXT NOT NULL,
  fase TEXT NOT NULL, vista TEXT NOT NULL, fecha_hora TEXT NOT NULL, operario TEXT NOT NULL,
  estado TEXT NOT NULL CHECK(estado IN ('LIBERADO','RETENIDO','BLOQUEADO')),
  regla TEXT NOT NULL, motivo TEXT, defecto_principal TEXT, puntaje REAL, tau1 REAL, tau2 REAL,
  n_defectos INTEGER NOT NULL, detecciones TEXT NOT NULL, senal TEXT, tiempo_respuesta_ms REAL,
  imagen TEXT, imagen_anotada TEXT, version_modelo TEXT,
  origen TEXT NOT NULL CHECK(origen IN ('REAL','SINTÉTICO','DEMO')), verdad_terreno TEXT, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS disposiciones(
  id INTEGER PRIMARY KEY AUTOINCREMENT, id_registro TEXT NOT NULL REFERENCES inspecciones(id_registro),
  fecha_hora TEXT NOT NULL, supervisor TEXT NOT NULL,
  dictamen TEXT NOT NULL CHECK(dictamen IN ('DEFECTO_CONFIRMADO','FALSA_ALARMA')),
  accion TEXT NOT NULL CHECK(accion IN ('REPROCESO','LIBERAR','DESECHO','REINSPECCION')),
  etapa_origen TEXT, costo_reproceso REAL, observacion TEXT);
CREATE TABLE IF NOT EXISTS ingresos_montaje(
  id INTEGER PRIMARY KEY AUTOINCREMENT, caja_id TEXT NOT NULL, fecha_hora TEXT NOT NULL, usuario TEXT NOT NULL,
  resultado TEXT NOT NULL CHECK(resultado IN ('ADMITIDO','RECHAZADO')), motivo TEXT);
CREATE TABLE IF NOT EXISTS acciones(
  id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_creacion TEXT NOT NULL, etapa TEXT NOT NULL, defecto TEXT,
  descripcion TEXT NOT NULL, responsable TEXT NOT NULL, fecha_compromiso TEXT NOT NULL,
  estado TEXT NOT NULL CHECK(estado IN ('ABIERTA','CERRADA')), fecha_cierre TEXT, resultado TEXT, origen TEXT);
CREATE TABLE IF NOT EXISTS historico(
  caja_id TEXT PRIMARY KEY, fecha TEXT, mes TEXT, modelo_id TEXT, lote TEXT, estado TEXT, tipo_defecto TEXT,
  punto_deteccion TEXT, etapa_origen_prueba TEXT, costo_produccion REAL, precio_venta REAL, costo_reproceso REAL,
  origen TEXT);
CREATE TABLE IF NOT EXISTS auditoria(
  id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_hora TEXT, usuario TEXT, evento TEXT, detalle TEXT);
CREATE INDEX IF NOT EXISTS ix_caja ON inspecciones(caja_id);
CREATE INDEX IF NOT EXISTS ix_fecha ON inspecciones(fecha_hora);
"""


class ErrorValidacion(Exception):
    pass


def ahora():
    return datetime.now().isoformat(timespec="seconds")


def conectar():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def inicializar():
    with conectar() as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(historico)")]
        if cols and "punto_deteccion" not in cols:   # migración: el histórico es derivado y se reimporta
            con.execute("DROP TABLE historico")
        con.executescript(ESQUEMA)


def _hash_pin(pin, sal):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), sal.encode(), 100_000).hex()


def crear_usuario(usuario, nombre, rol, pin):
    if rol not in ROLES:
        raise ErrorValidacion(f"Rol inválido: {rol}")
    if not re.fullmatch(r"\d{4,6}", pin):
        raise ErrorValidacion("El PIN debe tener entre 4 y 6 dígitos.")
    sal = uuid.uuid4().hex
    with conectar() as con:
        con.execute("INSERT OR REPLACE INTO usuarios VALUES(?,?,?,?,?)", (usuario, nombre, rol, _hash_pin(pin, sal), sal))


def autenticar(usuario, pin):
    with conectar() as con:
        u = con.execute("SELECT * FROM usuarios WHERE usuario=?", (usuario,)).fetchone()
    if u and _hash_pin(pin, u["sal"]) == u["pin_hash"]:
        return dict(usuario=u["usuario"], nombre=u["nombre"], rol=u["rol"])
    return None


def auditar(usuario, evento, detalle=""):
    with conectar() as con:
        con.execute("INSERT INTO auditoria(fecha_hora,usuario,evento,detalle) VALUES(?,?,?,?)",
                    (ahora(), usuario, evento, detalle))


def faltantes_cabecera(caja_id, lote, modelo_id, fase, vista):
    """Devuelve la lista de metadatos faltantes o inválidos (activa la regla R2)."""
    f = []
    if not PATRON_CAJA.match((caja_id or "").strip().upper()):
        f.append("ID de caja (formato CAJ-AAAA-NNNN)")
    if not (lote or "").strip():
        f.append("lote")
    if modelo_id not in MODELOS:
        f.append("modelo")
    if fase not in FASES:
        f.append("fase")
    if vista not in VISTAS:
        f.append("vista")
    return f


def _hash_registro(r):
    campos = {k: r[k] for k in sorted(r) if k != "hash"}
    return hashlib.sha256(json.dumps(campos, sort_keys=True, default=str).encode()).hexdigest()


def guardar_inspeccion(*, caja_id, lote, modelo_id, fase, vista, operario, decision, dets, tau1, tau2, senal,
                       tiempo_ms, imagen, imagen_anotada, version_modelo, origen="REAL", fecha_hora=None,
                       verdad_terreno=None):
    if not (caja_id or "").strip():
        caja_id = "SIN-ID"
    r = dict(id_registro=str(uuid.uuid4()), caja_id=caja_id.strip().upper(), lote=(lote or "").strip() or "SIN-LOTE",
             modelo_id=modelo_id or "?", fase=fase or "?", vista=vista or "?", fecha_hora=fecha_hora or ahora(),
             operario=operario, estado=decision.estado, regla=decision.regla, motivo=decision.motivo,
             defecto_principal=decision.defecto_principal, puntaje=round(decision.puntaje, 3), tau1=tau1, tau2=tau2,
             n_defectos=len(dets), detecciones=json.dumps([d.dict() for d in dets]), senal=senal,
             tiempo_respuesta_ms=round(tiempo_ms, 1), imagen=imagen, imagen_anotada=imagen_anotada,
             version_modelo=version_modelo, origen=origen, verdad_terreno=verdad_terreno)
    r["hash"] = _hash_registro(r)
    with conectar() as con:
        con.execute(f"INSERT INTO inspecciones({','.join(r)}) VALUES({','.join('?' * len(r))})", list(r.values()))
    return r


def verificar_integridad(id_registro):
    with conectar() as con:
        row = con.execute("SELECT * FROM inspecciones WHERE id_registro=?", (id_registro,)).fetchone()
    return bool(row) and _hash_registro(dict(row)) == row["hash"]


def registrar_disposicion(id_registro, supervisor, dictamen, accion, etapa_origen=None, costo_reproceso=None,
                          observacion=""):
    if dictamen == "DEFECTO_CONFIRMADO" and etapa_origen not in ETAPAS:
        raise ErrorValidacion("Indique la etapa de origen del defecto confirmado (cierra el lazo detección–origen–acción).")
    with conectar() as con:
        row = con.execute("SELECT estado FROM inspecciones WHERE id_registro=?", (id_registro,)).fetchone()
        if not row:
            raise ErrorValidacion("El registro no existe.")
        if row["estado"] == "LIBERADO":
            raise ErrorValidacion("Una caja liberada no requiere disposición.")
        if row["estado"] == "BLOQUEADO" and accion == "LIBERAR":
            raise ErrorValidacion("Una caja BLOQUEADA no se libera por dictamen: el bloqueo es automático; se reprocesa "
                                  "(o se reinspecciona si fue falsa alarma) y solo una nueva inspección LIBERADA la habilita.")
        if con.execute("SELECT 1 FROM disposiciones WHERE id_registro=?", (id_registro,)).fetchone():
            raise ErrorValidacion("La caja ya tiene una disposición registrada.")
        con.execute("INSERT INTO disposiciones(id_registro,fecha_hora,supervisor,dictamen,accion,etapa_origen,"
                    "costo_reproceso,observacion) VALUES(?,?,?,?,?,?,?,?)",
                    (id_registro, ahora(), supervisor, dictamen, accion, etapa_origen, costo_reproceso, observacion))


def estado_vigente(caja_id):
    """Estado actual de la caja = última inspección (para cada fase/vista), corregido por la disposición."""
    with conectar() as con:
        rows = con.execute("""SELECT i.id_registro, i.fase, i.vista, i.estado, i.fecha_hora, d.accion
            FROM inspecciones i LEFT JOIN disposiciones d ON d.id_registro=i.id_registro
            WHERE i.caja_id=? AND i.fase='carcasa_planchada' ORDER BY i.fecha_hora""", (caja_id.upper(),)).fetchall()
    if not rows:
        return None, "La caja no tiene inspección de carcasa/planchada registrada."
    ultimo = {}
    for r in rows:
        ultimo[r["vista"]] = r
    estados = []
    for v, r in ultimo.items():
        e = r["estado"]
        if e == "RETENIDO" and r["accion"] == "LIBERAR":
            e = "LIBERADO"        # dictamen final del supervisor sobre una caja retenida
        elif e == "RETENIDO" and r["accion"] in ("REPROCESO", "DESECHO", "REINSPECCION"):
            e = "BLOQUEADO"       # el supervisor confirma el defecto: queda bloqueada hasta nueva inspección
        # BLOQUEADO (R3) es automático: solo una nueva inspección LIBERADA de esa vista la habilita
        estados.append((v, e))
    if all(e == "LIBERADO" for _, e in estados):
        return "LIBERADO", "Todas las vistas inspeccionadas están liberadas."
    peor = "BLOQUEADO" if any(e == "BLOQUEADO" for _, e in estados) else "RETENIDO"
    return peor, "; ".join(f"{v}: {e}" for v, e in estados)


def ingresar_montaje(caja_id, usuario):
    """Poka-Yoke de control: solo una caja LIBERADO puede registrarse en montaje."""
    estado, detalle = estado_vigente(caja_id)
    ok = estado == "LIBERADO"
    with conectar() as con:
        con.execute("INSERT INTO ingresos_montaje(caja_id,fecha_hora,usuario,resultado,motivo) VALUES(?,?,?,?,?)",
                    (caja_id.upper(), ahora(), usuario, "ADMITIDO" if ok else "RECHAZADO", detalle))
    return ok, estado, detalle


def crear_accion(etapa, defecto, descripcion, responsable, fecha_compromiso, origen="DSS"):
    if not responsable.strip() or not fecha_compromiso:
        raise ErrorValidacion("Toda acción requiere responsable y fecha compromiso.")
    with conectar() as con:
        con.execute("INSERT INTO acciones(fecha_creacion,etapa,defecto,descripcion,responsable,fecha_compromiso,"
                    "estado,origen) VALUES(?,?,?,?,?,?,'ABIERTA',?)",
                    (ahora(), etapa, defecto, descripcion, responsable, fecha_compromiso, origen))


def cerrar_accion(id_accion, resultado):
    with conectar() as con:
        con.execute("UPDATE acciones SET estado='CERRADA', fecha_cierre=?, resultado=? WHERE id=?",
                    (ahora(), resultado, id_accion))


def acciones():
    with conectar() as con:
        return [dict(r) for r in con.execute("SELECT * FROM acciones ORDER BY estado, fecha_compromiso")]
