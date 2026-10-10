"""CP0 — Ingesta y conciliación de los registros por caja del escenario proyectado.

Fuente por defecto: Registros_Escenario_Proyectado_2023-2026.xlsx (hoja REGISTRO_CAJAS), filtrada al
año de la línea base (2025). Se concilian los controles del TF1 (Tabla 4.2):
  · 1 250 ID de caja únicos · 150 cajas no conformes · costo de reproceso S/ 90 000.
Los registros se cargan en la tabla `historico` con origen = 'PROYECTADO': son registros generados a partir
del escenario proyectado, no mediciones de la empresa.

También acepta el formato anterior de la Data Maestra v3 (columna Etapa_deteccion). En ese caso:
  · el punto de detección se fija en «Inspección final» (situación actual) y la etapa de origen queda vacía,
    porque esa columna mezclaba el punto de detección con la etapa de origen;
  · el prefijo de los ID se normaliza a CAJ-.
"""
import pandas as pd

from config import HISTORICO_PROY
from modulos.registro import conectar

COLUMNAS = ["ID_caja", "Fecha", "Mes", "Modelo", "Lote", "Estado", "Tipo_defecto", "Punto_deteccion",
            "Etapa_origen_prueba", "Costo_produccion_S", "Precio_venta_S", "Costo_reproceso_S", "Origen_dato"]
CONTROLES = {"ids_unicos": 1250, "no_conformes": 150, "costo_reproceso": 90000}


def leer(ruta=HISTORICO_PROY, anio=2025):
    df = pd.read_excel(ruta, sheet_name="REGISTRO_CAJAS")
    if "Punto_deteccion" not in df.columns and "Etapa_deteccion" in df.columns:   # formato Data Maestra v3
        df = df.drop(columns="Etapa_deteccion")
        df["Punto_deteccion"] = "Inspección final"
        df["Etapa_origen_prueba"] = ""
        df["Origen_dato"] = "PROYECTADO"
    df["ID_caja"] = df.ID_caja.astype(str).str.replace(r"^MIN-", "CAJ-", regex=True)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en REGISTRO_CAJAS: {faltan}")
    df["Etapa_origen_prueba"] = df.Etapa_origen_prueba.fillna("")
    if anio is not None:
        df = df[pd.to_datetime(df.Fecha).dt.year == anio].reset_index(drop=True)
    return df


def conciliar(df):
    r = {"filas": len(df), "ids_unicos": int(df.ID_caja.nunique()),
         "no_conformes": int((df.Estado != "Conforme").sum()),
         "costo_reproceso": float(df.Costo_reproceso_S.sum()),
         "modelos": sorted(df.Modelo.unique().tolist()), "lotes": int(df.Lote.nunique()),
         "origen": sorted(df.Origen_dato.unique().tolist()),
         "nulos_obligatorios": int(df[["ID_caja", "Fecha", "Modelo", "Lote", "Estado"]].isna().sum().sum())}
    r["controles"] = {k: {"esperado": v, "obtenido": r[k], "ok": abs(r[k] - v) < 1e-6} for k, v in CONTROLES.items()}
    r["ok"] = all(c["ok"] for c in r["controles"].values()) and r["nulos_obligatorios"] == 0 and r["ids_unicos"] == r["filas"]
    return r


def importar(ruta=HISTORICO_PROY, anio=2025):
    df = leer(ruta, anio)
    rep = conciliar(df)
    filas = [(str(x.ID_caja), str(x.Fecha)[:10], str(x.Mes), x.Modelo, x.Lote, x.Estado, x.Tipo_defecto,
              x.Punto_deteccion, x.Etapa_origen_prueba or None, float(x.Costo_produccion_S), float(x.Precio_venta_S),
              float(x.Costo_reproceso_S), "PROYECTADO") for x in df.itertuples()]
    with conectar() as con:
        con.execute("DELETE FROM historico")
        con.executemany("INSERT INTO historico VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", filas)
    rep["cargadas"] = len(filas)
    return rep


def historico():
    with conectar() as con:
        return pd.read_sql_query("SELECT * FROM historico", con)
