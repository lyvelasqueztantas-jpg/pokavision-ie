"""Configuración central de PokaVision-IE v2.

DSS operacional híbrido con mecanismo Poka-Yoke basado en visión artificial y
trazabilidad digital para la fabricación de cajas de seguridad.
Fechas dinámicas: el sistema registra la fecha/hora actual de cada inspección
(no depende de un año fijo). Los datos 2025 del Excel son un escenario SINTÉTICO.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
IMG_STORE = DATA_DIR / "imagenes_inspeccion"
EXCEL_DEMO = DATA_DIR / "excel" / "Data_Maestra_4_Modelos.xlsx"
DB_PATH = DATA_DIR / "pokavision.db"
MODEL_DIR = BASE_DIR / "modelos"
MODEL_PATH = MODEL_DIR / "detector.joblib"
METRICS_PATH = MODEL_DIR / "metricas_validacion.json"
UMBRALES_PATH = MODEL_DIR / "umbrales_calibrados.json"
PARAM_FILE = DATA_DIR / "parametros.json"

# --- Alcance visual del piloto (Especificación v3)
CLASES = ["rayadura", "pintura_irregular", "soldadura_visible"]
CLASE_NOMBRE = {"rayadura": "Rayadura", "pintura_irregular": "Pintura irregular",
                "soldadura_visible": "Soldadura visible"}
MODELOS = {
    "M01": {"descripcion": "Horizontal gris oscuro", "cerradura": "electrónica circular", "color": (58, 56, 54)},
    "M02": {"descripcion": "Horizontal marfil", "cerradura": "electrónica rectangular", "color": (200, 222, 232)},
    "M03": {"descripcion": "Vertical gris", "cerradura": "electrónica + manija", "color": (112, 112, 110)},
    "M04": {"descripcion": "Horizontal gris claro", "cerradura": "electrónica + manija", "color": (170, 170, 168)},
}
FASES = {"carcasa_planchada": "Carcasa/planchada (pintada, antes del montaje)",
         "producto_terminado": "Producto terminado"}
VISTAS = ["frontal", "lateral"]

# Etapas de origen (matriz defecto–causa–acción, TF1 Tabla 3.3)
ETAPAS = ["Corte y punzonado", "Plegado y conformado", "Soldadura y ensamblaje", "Preparación superficial",
          "Pintura y curado", "Manipulación y almacenamiento", "Montaje"]
MATRIZ_ACCION = {
    "rayadura": {"etapa": "Manipulación y almacenamiento",
                 "causa": "Manipulación sin protección, contacto metal–metal, almacenamiento intermedio inadecuado",
                 "accion": "Estándar de manipulación, protectores en puntos de contacto y racks de almacenamiento"},
    "pintura_irregular": {"etapa": "Pintura y curado",
                          "causa": "Pretratamiento insuficiente, espesor no uniforme, temperatura/tiempo de curado fuera de rango",
                          "accion": "Control del pretratamiento, medición de espesor y verificación del perfil del horno"},
    "soldadura_visible": {"etapa": "Soldadura y ensamblaje",
                          "causa": "Parámetros de soldadura no estandarizados, preparación deficiente de juntas",
                          "accion": "Especificación de parámetros, plantilla de fijación y verificación de juntas"},
}

# --- M1: condiciones mínimas de captura (regla R1)
ADQ_RES_MIN = (640, 480)
ADQ_BRILLO_RANGO = (25, 230)
ADQ_NITIDEZ_MIN = 8.0   # varianza del Laplaciano; calibrar con fotos reales de la estación
PROC_TAMANO = (640, 480)

# --- M4: regla de dos umbrales (s < τ1 libera; τ1 ≤ s < τ2 retiene; s ≥ τ2 bloquea)
TAU1_DEFECTO, TAU2_DEFECTO = 0.35, 0.70
# Costos para calibrar τ1, τ2 por costo esperado (Elkan, 2001). SUPUESTOS editables por el jefe de producción.
COSTO_ESCAPE = 1200.0        # Ce: defecto que llega al montaje (reproceso + desmontaje), supuesto 2 × S/ 600
COSTO_FALSO_BLOQUEO = 600.0  # Cb: reproceso innecesario de una caja conforme (costo de reproceso del escenario base)
COSTO_REVISION = 15.0        # Cr: revisión del supervisor (≈15 min × S/ 60/h), supuesto

# --- Línea base del PROYECTO: escenario SINTÉTICO del Excel maestro v3 (se usa en cálculos y metas)
BASE_SINT = {"producidas": 1250, "defectuosas": 150, "td": 12.0, "costo_unit": 600, "costo_total": 90000,
             "tobe_defectos": 49, "tobe_costo": 29400, "ahorro_bruto": 60600}
PARETO_SINT = {"Rayadura": 48, "Pintura irregular": 39, "Soldadura visible": 30, "Deformación de láminas": 20,
               "Desalineación de componentes": 13}
COSTO_REPROCESO = 600

# --- Escenario proyectado 2023–2026 (TF1, Tabla 4.2); registros por caja en generar_registros_proyectados.py
#     producción × 1,250/380 y defectuosas × 150/28 sobre el registro del TF1; costo S/ 600 por caja
SERIE_SINT = {2023: (1053, 112, 67200), 2024: (1151, 129, 77400), 2025: (1250, 150, 90000), 2026: (526, 64, 38400)}
HISTORICO_PROY = DATA_DIR / "excel" / "Registros_Escenario_Proyectado_2023-2026.xlsx"   # generar_registros_proyectados.py
HISTORICO_SINT = HISTORICO_PROY   # alias por compatibilidad

# --- REFERENCIA del registro del TF1 (origen de las proporciones; NO se usa en cálculos)
LINEA_BASE = {2023: (320, 21, 11550), 2024: (350, 24, 13200), 2025: (380, 28, 15400), 2026: (160, 12, 6600)}
PARETO_REAL = {"Rayaduras superficiales": 28, "Pintura irregular": 24, "Soldadura defectuosa": 15,
               "Deformación metálica": 11, "Perforación fuera de medida": 5, "Desalineación de componentes": 2}
META_TD = 4.0

ROLES = ["operario", "supervisor", "jefe", "montaje"]
SECRET_KEY = "pokavision-ie-v2-cambiar-en-produccion"
