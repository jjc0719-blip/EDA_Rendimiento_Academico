import io
import os
import contextlib
import nbformat
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import pandas as pd
import numpy as np  
from pathlib import Path
from streamlit_option_menu import option_menu
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
import glob
import hashlib
from pathlib import Path
import requests 


st.set_page_config(page_title="Análisis Exploratorio del Rendimiento Académico de Estudiantes Universitarios",
                   layout="wide")

# Sidebar configuration
with st.sidebar:
    # Sidebar header (bold + larger font)
    st.markdown(
        """
        <h1 style='text-align: left; font-weight: 700; font-family: Tahoma, "Tahoma", Geneva, sans-serif; font-size: 30px;'>Análisis Exploratorio del Rendimiento Académico de Estudiantes Universitarios</h1>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("""
 
                """)
    
    # Navigation menu
    choice = option_menu(
        "Capítulos",
        ["Introducción", "Objetivos", "Exploración de Datos (EDA)","Modelo", "Conclusiones", "Referencias"],
        icons=["book", "bullseye", "bar-chart","collection-play", "pencil", "bookmarks"],
        menu_icon="cast",
        default_index=0,

    )

with st.sidebar:
    
    st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: gray;">
                María José Berrio Chasoy
                <br>
                José Castro Cervantes
                <br>
                César Anachury Pacheco
    </div>
    """, unsafe_allow_html=True)


# Define page functions

def page_intro():
    
    # Título Introducción 
    st.markdown("""
                    <div style='position:fixed; top:40px; left:400px; right:24px; background:#ffffff; padding:10px 16px; z-index:9999; border-bottom:1px solid rgba(0,0,0,0.06);'>
                    <h1 style='color:#111111; font-weight:700; font-size:50px; margin:0;'>INTRODUCCIÓN</h1>
                </div>
                    <div style='height:64px;'></div>
                """, unsafe_allow_html=True)
    
    #Contenido Introducción
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                    <br>
                    La deserción estudiantil y el rendimiento académico se han convertido en preocupaciones centrales para las instituciones de educación superior, dado que las tasas de deserción son ampliamente reconocidas como indicadores de la calidad educativa, con implicaciones directas en la reputación, la financiación y la sostenibilidad institucional (Gallego et al., 2021).  Sin embargo, las definiciones y enfoques de medición varían: mientras que las perspectivas macro consideran únicamente a los estudiantes que abandonan el sistema sin obtener un título, las perspectivas micro incluyen también los cambios de programa o de institución como eventos de deserción, lo que genera tasas reportadas más altas (Realinho et al., 2022). 
                    <br>
                    <br>
                    Frente a esta problemática, se han enfatizado las intervenciones intensivas y continuas como estrategias clave para reducir la deserción, y la creciente disponibilidad de datos educativos ha impulsado el desarrollo de la <strong>Minería de Datos Educativos (MDE)</strong>, un campo que utiliza modelos predictivos para comprender fenómenos como el rendimiento, la retención, la satisfacción, el logro y la deserción (Alyahyan & Düştegör, 2020). En Colombia, este enfoque se complementa con el <strong>Sistema de Prevención y Análisis de la Deserción en Instituciones de Educación Superior (SPADIES)</strong>, diseñado por el Centro de Estudios Económicos de la Universidad de los Andes, que permite monitorear el fenómeno, calcular el riesgo individual de cada estudiante y apoyar el diseño de estrategias de intervención diferenciadas (Pérez et al., 2018).
                    <br>
                    <br>
                    La deserción estudiantil en Colombia es significativamente más alta que el promedio de los países miembros de la OCDE, lo que evidencia una problemática estructural en el sistema educativo superior del país (Ministerio de Educación Nacional – SPADIES, Informe OCDE sobre educación en Colombia, 2025). Este fenómeno tiene múltiples causas: dificultades económicas, bajo rendimiento académico, falta de orientación vocacional, problemas de salud mental, y condiciones sociales adversas. Además, el sistema educativo colombiano enfrenta retos adicionales como la baja cobertura en zonas rurales, la escasa articulación entre la educación media y superior, y la limitada capacidad de respuesta institucional ante estudiantes en riesgo (LEE, 2023; Valencia-Arias et al., 2023).
                    <br>
                    <br>
                    Por estas razones, desarrollar modelos predictivos que anticipen el riesgo de deserción es clave para implementar estrategias de intervención temprana, mejorar la retención y fortalecer el sistema educativo colombiano. El presente proyecto utiliza un conjunto de datos públicos de una universidad pública de Colombia, correspondientes al período 2014–2023, que incluyen información de notas parciales, definitivas y de habilitación por asignatura, junto con variables relacionadas al programa académico en diferentes titulaciones de grado, como licenciatura en educación, ingenierías, medicina, lenguas, administración  y ciencias básicas. A partir de estos datos, se plantea la construcción de modelos estadísticos y de machine learning que permitan estimar la probabilidad de bajo rendimiento como un proxy de riesgo de deserción estudiantil, evaluando el desempeño de distintos algoritmos de clasificación y analizando la relevancia de las variables académicas en la predicción.</p>
                </div>
                """, unsafe_allow_html=True)


def page_objectives():
    # Título 
    st.markdown("""
                    <div style='position:fixed; top:40px; left:400px; right:24px; background:#ffffff; padding:10px 16px; z-index:9999; border-bottom:1px solid rgba(0,0,0,0.06);'>
                    <h1 style='color:#111111; font-weight:700; font-size:50px; margin:0;'>OBJETIVOS</h1>
                </div>
                    <div style='height:64px;'></div>
                """, unsafe_allow_html=True)
    
        #Contenido 
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#333333; font-size:24px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'> GENERAL </h1>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                     Desarrollar un análisis estadístico (descriptivo e inferencial) en función del conjunto de datos denominados “Student Academic Record”, con el propósito de implementar y validar un modelo de Machine Learning (XGBoost) que permita estimar la probabilidad de deserción estudiantil en una Institución de Educación Superior de Colombia.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#333333; font-size:24px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'> ESPECIFICOS </h1>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                     - Identificar la estructura y características del conjunto de datos, resaltando sus dimensiones, tipos de varibles (𝑋𝑖: cuantitativas y cualitativas) y variable de interés (𝑌: dependiente).
                    <br>
                    </p>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                     - Realizar un análisis exploratorio en función de la variable dependiente (𝑌), caracterizando su distribución y evaluando patrones, tendencias y posibles sesgos en los datos.
                    <br>
                    </p>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                     - Llevar a cabo un análisis estadístico de las variables numéricas y categóricas, con el fin de: verificar la normalidad de las distribuciones (Ho : Xi ∼ N(μ,σ2)), estimar correlaciones significativas entre variables cuantitativas (rSpearman) y analizar asociaciones entre variables categóricas mediante pruebas de independencia (𝜒2).
                    <br>
                    </p>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                     - Aplicar un modelo de Machine Learning (XGBoost) que permita estimar la probabilidad de deserción estudiantil, formulada como un problema de clasificación binaria (Y ∈ {0,1}), evaluando métricas de desempeño (Accuracy, Precision, Recall, AUC).
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    

def page_eda():
    st.markdown("""
                <div style='position:fixed; top:40px; left:400px; right:24px; background:#ffffff; padding:10px 16px; z-index:9999; border-bottom:1px solid rgba(0,0,0,0.06);'>
                    <h1 style='color:#111111; font-weight:700; font-size:32px; margin:0;'>EXPLORACIÓN DE LOS DATOS (EDA)</h1>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
               
                """, unsafe_allow_html=True)
    st.markdown("""
                <div
                    <h1 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>1. TRANSFORMACIÓN DE DATOS</h1>
                    <br>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>Antes de realizar el Análisis Exploratorio de los Datos, es necesario preparar y transformar la información en diferentes formatos que faciliten su comprensión y procesamiento. Para ello, se emplean diversos paquetes y librerías que proporcionan funciones diseñadas para organizar, limpiar y estructurar los datos de manera eficiente.
                     A continuación, se presentan las principales características del DataFrame y se ejecutarán las transformaciones necesarias para dar inicio al análisis detallado de la información.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ---------- Utils ----------

    @st.cache_data(ttl=60*60, show_spinner=False)
    def load_parquet(path_str: str) -> pd.DataFrame:
        """Lee un archivo parquet y lo cachea."""
        return pd.read_parquet(path_str)

    @st.cache_data(ttl=60*60, show_spinner="Descargando datos...")
    def download_once(url: str, dest_path_str: str) -> str:
        """
        Descarga el archivo una sola vez y lo guarda en disco.
        Si ya existe, simplemente devuelve la ruta.
        """
        dest_path = Path(dest_path_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not dest_path.exists():
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        return str(dest_path)

    def human_fmt(n):
        try:
            return f"{n:,}".replace(",", ".")
        except Exception:
            return str(n)

    def get_secret(key: str, default=None):
        """
        Intenta leer de st.secrets; si no existe secrets.toml,
        usa variables de entorno; si no, devuelve default.
        """
        try:
            _ = st.secrets  # fuerza parseo; puede lanzar StreamlitSecretNotFoundError
            return st.secrets.get(key, default)
        except Exception:
            return os.getenv(key, default)

    # ---------- Config de datos ----------
    # Directorio del archivo actual (app.py)
    APP_DIR = Path(__file__).resolve().parent

    # Nombre del archivo parquet (puedes cambiarlo)
    FILENAME = get_secret("DATA_FILENAME", "dataset.parquet")

    # Si defines un DATA_URL (en secrets o como variable de entorno), lo descargamos una sola vez
    DATA_URL = get_secret("DATA_URL", None)

    # Ruta local por defecto: MISMO DIRECTORIO que app.py (evita problemas de backslashes)
    LOCAL_PATH = APP_DIR / FILENAME

    st.title("Demo: Carga optimizada con Parquet")

    # ---------- Resolución de la ruta ----------
    if DATA_URL:
        # Descarga una sola vez y usa ese archivo
        try:
            local_file = download_once(DATA_URL, str(LOCAL_PATH))
            st.caption(f"Fuente: descargado desde DATA_URL → {FILENAME}")
        except Exception as e:
            st.error(f"No se pudo descargar el archivo desde DATA_URL.\nDetalle: {e}")
            st.stop()
    else:
        # Sin URL: usamos archivo local junto a app.py
        local_file = str(LOCAL_PATH)
        if not Path(local_file).exists():
            st.error(
                "No se encontró el archivo local de datos.\n\n"
                f"Busqué en: `{local_file}`\n\n"
                "Soluciones:\n"
                f"1) Copia tu `.parquet` al mismo directorio que este `app.py` con el nombre `{FILENAME}`, o\n"
                "2) Define `DATA_URL` (en `.streamlit/secrets.toml` o variable de entorno) para descargarlo automáticamente.\n"
            )
            st.stop()

    # ---------- Carga de datos ----------
    with st.spinner("Cargando datos (Parquet)..."):
        try:
            df = load_parquet(local_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo Parquet `{local_file}`.\nDetalle: {e}")
            st.stop()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 # iNICIA EDA
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    st.markdown("### 📋 Información de la Estructura del DataFrame")

    # Mostrar métricas rápidas: filas, columnas y elementos
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{df.shape[0]:,}")
        c2.metric("Variables", f"{df.shape[1]:,}")
        c3.metric("Observaciones", f"{df.size:,}")
        
    # Construir una tabla similar a df.info() pero en formato DataFrame
    if df is not None:
        non_null = df.notnull().sum()
        nulls = df.isnull().sum()
        dtypes = df.dtypes.astype(str)
        df_info = pd.DataFrame({
            'Variable': df.columns,
            'Tipo': dtypes.values,
            'Cantidad Registros': non_null.values,
            'Registros Nulos': nulls.values,
            '% No Nulos': ((non_null / len(df)) * 100).round(0).values
        })

        # Mostrar tabla estilizada
        styled = df_info.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f8f9fa"), ("color", "#111111"), ("font-weight", "600")]},
            {"selector": "tbody td", "props": [("font-size", "13px"), ("text-align", "center")]}
        ]).format({"Porc No Nulos": "{:.2f}%"})

        st.dataframe(styled, use_container_width=True)
        
        st.markdown("""
                <div
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>
                    El conjunto de datos está conformado por 590.415 registros y 16 variables, todas relacionadas con el rendimiento académico de los estudiantes. Las variables incluyen información institucional (Facultad, Programa), académica (Asignatura, Grupo) y de desempeño (Notas 1 a 4).
                    <br>
                    Cabe resaltar que no se registran valores nulos, lo que garantiza la integridad y consistencia de la información disponible.
                    A continuación, se procederá a verificar la existencia de datos duplicados y posibles valores faltantes, con el fin de garantizar la calidad e integridad del conjunto de datos antes de avanzar en el Análisis Exploratorio de Datos (EDA).
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
                    
                    """)
                     
        # -------- Resumen de valores faltantes --------
        st.markdown("### 🔎 Resumen de valores duplicados y faltantes")

        # -------- Eliminar duplicados y mostrar métricas --------

        initial_count = int(df.shape[0])
        dup_count = int(df.duplicated().sum())
        
        # eliminar duplicados si existen
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        after_count = int(df.shape[0])

        d1, d2, d3 = st.columns(3)
        d1.metric("Registros iniciales", f"{initial_count:,}")
        d2.metric("Duplicados encontrados", f"{dup_count:,}")
        d3.metric("Registros Sin Duplicados", f"{after_count:,}")

        # Por columna
        missing_by_col = df.isnull().sum().sort_values(ascending=False)

        # Totales y por fila
        total_missing = int(missing_by_col.sum())
        rows_with_missing = int(df.isnull().any(axis=1).sum())
        rows_without_missing = int(df.shape[0] - rows_with_missing)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total faltantes", f"{total_missing:,}")
        m2.metric("Filas con faltante", f"{rows_with_missing:,}")
        m3.metric("Filas sin faltantes", f"{rows_without_missing:,}")

        # Mostrar faltantes por Variable (tabla)
        st.markdown("**Faltantes por Variable**")
        st.dataframe(missing_by_col.to_frame(name='Registros Nulos'), use_container_width=True)

        st.markdown("""
                    
                    """)
        
        st.markdown("""
                <div
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>
                    Tras la eliminación de las filas duplicadas y la verificación de la ausencia de datos faltantes en el conjunto de datos, el proceso de análisis se ve considerablemente simplificado, ya que no es necesario aplicar técnicas de imputación. A continuación, se presenta la descripción detallada de las variables que conforman el dataset, distinguiendo entre atributos categóricos —tanto nominales como ordinales— y variables numéricas, las cuales pueden ser continuas o discretas según su naturaleza.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
               
                """, unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h1 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>2. ESTRUCTURA Y CARACTERISTICAS DE LA BASE DE DATOS</h1>
                    <br>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                     La siguiente tabla, detalla la Estructura y Características de la base de datos "Student Academic Record". Su configuración incluye cinco columnas: "Clase de Atributo", que agrupa las variables según su naturaleza; "Atributo", que indica el nombre de cada variable registrada en la base; "Tipo", que define la naturaleza de los datos como "categóricos" o "numéricos", y dentro de ellos, "nominales", "ordinales", "continuos" o "discretos"; "Count", que presenta el número total de registros (observaciones), en este caso 590.412 para todas las variables; y Missing (value), que señala la ausencia de valores faltantes, siendo 0 en todos los casos (tanto para las filas como para las columnas).
                     </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
               
                """, unsafe_allow_html=True)
    
    var_types = [
                ("Información Académica", "Facultad", "Categórica / Nominal", "590412", "0"),
                ("Información Académica", "Programa Académico", "Categórica / Nominal", "590412", "0"),
                ("Información Académica", "Código Asignatura", "Categórica / Nominal", "590412", "0"),
                ("Información Académica", "Asignatura / Materia", "Categórica / Nominal", "590412", "0"),
                ("Información Académica", "Grupo", "Categórica / Nominal", "590412", "0"),
                ("Información Académica", "Código Estudiantil", "Numérica / Continua",  "590412", "0"),
                ("Record de Notas", "Nota 1", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota 2", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota 3", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota 4", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota Definitiva", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota Habilitación", "Numérica / Continua", "590412", "0"),
                ("Record de Notas", "Nota Final", "Numérica / Continua", "590412", "0"),
                ("Desempeño Estudiantil", "Rendimiento", "Categórica / Ordinal", "590412", "0"),
                ("Datos Temporales", "Año", "Numérica / Discreta", "590412", "0"),
                ("Datos Temporales", "Periodo", "Numérica / Discreta", "590412", "0")
            ]

            # Crear DataFrame
    tabla_vars = pd.DataFrame(
                var_types,
                columns=["Clase de Atributo", "Atributo", "Tipo", "Count", "Missing"]
            )

            # Mostrar con estilo
    tabla_vars = (
        tabla_vars.style
            .set_table_attributes('style="width:100%; margin-left:auto; margin-right:auto;"')
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            .hide(axis="index")
        )
    # Mostrar la tabla en Streamlit (st.write/st.dataframe requiere llamada explícita)
    st.dataframe(tabla_vars, use_container_width=True)
    st.markdown("""
                
                    """, unsafe_allow_html=True)

    st.markdown("""
                    <div>
                        <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                        Dentro de la <strong>clase Información Académica</strong> se agrupan las variables que describen el contexto académico de cada estudiante. Estas incluyen: Facultad, Programa Académico, Código de Asignatura, Asignatura/Materia, Grupo y Código Estudiantil. Las cinco primeras corresponden a <strong>variables categóricas nominales</strong>, mientras que el Código Estudiantil se clasifica como <strong>variable numérica continua</strong>.
                        <br>
                        La <strong>clase Registro de Notas</strong> reúne las variables asociadas al desempeño académico, entre ellas las <strong>notas parciales</strong> (Nota 1, Nota 2, Nota 3 y Nota 4), la Nota Definitiva, la Nota de Habilitación y la Nota Final. Todas estas variables son de tipo <strong>numérico continuo</strong>.
                        <br>
                        En la <strong>clase Desempeño Estudiantil</strong> se encuentra el atributo <strong>Rendimiento</strong>, definido como una variable categórica ordinal con cinco niveles: Deficiente, Bajo, Medio, Alto y Superior.
                        <br>
                        Por último, la <strong>clase Datos Temporales</strong> contiene los atributos Año y Periodo, ambos definidos como <strong>variables numéricas discretas</strong>. En síntesis, la tabla organiza de manera estructurada la información del conjunto de datos, evidenciando que todas las variables cuentan con <strong>590.412 registros completos y sin valores faltantes</strong>, además de clasificar cada atributo según su tipo de dato y naturaleza.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("""
                
                    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(20), use_container_width=True)
        
    st.markdown("""
                    
                    """)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    st.markdown("""
                <div
                    <h1 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3. ESTADISTICAS DESCRIPTIVAS</h1>
                    <br>
                </div>
                """, unsafe_allow_html=True)    

    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.1. RECORD DE NOTAS</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)  

    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    La siguiente tabla presenta un análisis descriptivo de las variables relacionadas con las calificaciones dentro de la base "Student Academic Record". Cada fila representa una de las notas evaluadas, mientras que las columnas ofrecen estadísticas descriptivas clave: número de observaciones, media, desviación estándar, valores mínimos y máximos, percentiles (25%, 50% y 75%), mediana, asimetría (skewness) y curtosis (kurtosis). Todas las variables cuentan con 590.412 registros, lo que evidencia que no hay valores perdidos en estas mediciones.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)    
    
    # Tabla de Record de Notas (mostrar en Streamlit)
    # Variables que se excluyen
    excluir = ["Código Estudiantil", "Año", "Periodo", "Código Asignatura"]

    # Se crea un nuevo DataFrame sin esas variables
    df1 = df.drop(columns=excluir, errors="ignore")

    # Resumen para variables numéricas
    resumen_numerico = df1.describe().T  # Transpuesto para mayor legibilidad
    resumen_numerico["median"] = df1.median(numeric_only=True)
    resumen_numerico["skewness"] = df1.skew(numeric_only=True)
    resumen_numerico["kurtosis"] = df1.kurtosis(numeric_only=True)

    # Preparar DataFrame para visualización
    resumen_numerico = resumen_numerico.reset_index().rename(columns={"index": "Variable"})

    # Formatear números: counts como enteros, el resto con 2 decimales
    num_cols = resumen_numerico.select_dtypes(include=['number']).columns.tolist()
    fmt = {c: "{:,.2f}" for c in num_cols}
    # Detectar columna 'count' (case-insensitive) y usar 0 decimales
    for c in resumen_numerico.columns:
        if c.lower() == 'count':
            fmt[c] = "{:,.0f}"

    # Mostrar con estilo para mejor legibilidad
    st.dataframe(resumen_numerico.style.format(fmt), use_container_width=True)
    
    st.markdown("""
                    
                    """)
    
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    En el caso de Nota 1 y Nota 2, ambas presentan promedios cercanos a 2.38 y 2.42 respectivamente, con desviaciones estándar superiores a 1.89, lo cual refleja una gran dispersión en las calificaciones. Sus medianas están alrededor de 3.0 y 3.1, con valores mínimos de 0 y máximos de 5. La asimetría es ligeramente negativa, lo que indica una ligera concentración de valores hacia la parte alta de la escala, mientras que la curtosis negativa refleja distribuciones más aplanadas en comparación con una normal.
                    <br><br>
                    En Nota 3, el promedio es más alto, de 3.45, con una mediana de 4 y un rango intercuartílico entre 3.1 y 4.5. La distribución tiene una asimetría negativa y una curtosis positiva, lo que sugiere un ligero sesgo hacia notas más altas y una mayor concentración alrededor de la media en comparación con las notas 1 y 2.
                    <br><br>
                    Por su parte, Nota 4 muestra un promedio muy bajo de 0.03, con una mediana en 0 y un rango intercuartílico también en 0, lo que refleja que en la mayoría de los registros esta nota no se presenta o su valor es nulo. Sin embargo, aparecen casos con calificaciones hasta de 5, lo cual se refleja en la desviación estándar de 0.36 y en la elevada asimetría y curtosis, que indican una distribución fuertemente concentrada en 0 pero con valores atípicos en el extremo superior.
                    <br><br>
                    La Nota Definitiva alcanza un promedio de 3.76, con una mediana de 4, valores mínimos en 0 y máximos en 5. Su distribución está sesgada negativamente, lo que sugiere una mayor acumulación de estudiantes con notas más altas, y con una curtosis de 4.5 que indica una concentración mayor de valores cerca de la media con colas más pesadas que una distribución normal.
                    <br><br>
                    En la Nota de Habilitación, el promedio es de apenas 0.06, con una mediana de 0 y valores que alcanzan como máximo 5. Esto refleja que la mayoría de los estudiantes no presentan habilitación, aunque existen registros de quienes sí la tienen. La distribución muestra alta asimetría positiva y una curtosis elevada, lo que indica que se trata de un evento poco frecuente pero con presencia de valores extremos.
                    <br><br>
                    Finalmente, la Nota Final presenta un promedio de 3.78 y una mediana de 4, con valores mínimos en 0 y máximos en 5. Su comportamiento es muy similar al de la Nota Definitiva, aunque con ligeras diferencias por la influencia de la habilitación. La asimetría negativa y la curtosis positiva reafirman que los resultados tienden hacia calificaciones más altas, con una mayor concentración alrededor de la media y presencia de valores extremos.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("""
                    
                    """)   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.2. DISTRIBUCIÓN DE LOS DATOS RELATIVOS A VARIABLE NOTA FINAL</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)     
    
    # Diagrama de Caja para Nota Final (Plotly) con media marcada
    if 'Nota Final' in df.columns:
        # convertir a numérico y calcular media de forma robusta
        nota_numeric = pd.to_numeric(df['Nota Final'], errors='coerce')
        media = nota_numeric.mean()

        # crear figura a partir de un DataFrame limpio
        df_plot = nota_numeric.to_frame(name='Nota Final')
        fig2 = px.box(df_plot, x='Nota Final', points='outliers',
                      title=f'Diagrama de Caja - Nota Final (media {media:.2f})',
                      labels={'Nota Final': 'Nota Final'})

        # añadir línea vertical indicando la media y una anotación
        try:
            fig2.add_vline(x=media, line=dict(color='red', dash='dash'))
            fig2.add_annotation(x=media, y=1.05, xref='x', yref='paper',
                                text=f'Media: {media:.2f}', showarrow=False,
                                font=dict(color='red'))
        except Exception:
            # si la API falla por versiones antiguas, omitir la línea
            pass

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("La columna 'Nota Final' no está presente en el DataFrame.")

    st.markdown("""
                    
                    """) 
    
    st.markdown("""
                <div
                    <P style='color:#111111; font-weight:600; font-size:20px; margin:18px 0 6px 0;'><strong>Medidas de Tendencia Central</strong></P>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>La mediana se ubica en el valor 4, representada por la línea dentro de la caja, La media está representada con una linea punteada roja y tiene un valor de 3.78.</p>
                    <p style='color:#111111; font-weight:600; font-size:20px; margin:18px 0 6px 0;'><strong>Medidas de Disperción</strong></p>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>El rango intercuartílico (IQR) abarca valores entre 3.4 y 4.4, lo que representa el 50% central de la distribución de datos. El bigote superior alcanza el valor máximo registrado de 5. El bigote inferior llega hasta un valor cercano a 1.8.</p>
                    <p style='color:#111111; font-weight:600; font-size:20px; margin:18px 0 6px 0;'><strong>Valores Atípicos</strong></h3>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>Se observan numerosos valores atípicos en el rango de 0 a 1.8, representados por puntos individuales. Estos valores se encuentran por debajo del límite inferior del bigote, indicando calificaciones más bajas en comparación con la mayoría de los registros.</p>
                </div>
                """, unsafe_allow_html=True)     
    
    st.markdown("""
                    
                    """)   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.3. INFORMACIÓN ACADÉMICA Y DESEMPEÑO ESTUDIANTIL</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)     
 
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    La siguiente tabla muestra un resumen descriptivo de las variables categóricas contenidas en la base de datos: Student Academic Record. Cada fila corresponde a una variable, indicando la cantidad de registros (observaciones), la cantidad de categorías únicas, la categoría más frecuente y su representación en frecuencias absolutas, relativas y porcentuales.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)  

    # Variable categóricas (mostrar en Streamlit)
    # Variables que se excluyen
    excluir = ["Código Estudiantil", "Código Asignatura"]

    # Se crea un nuevo DataFrame sin esas variables
    df2 = df.drop(columns=excluir, errors="ignore")

    # Resumen para variables categóricas
    resumen_categorico = df2.describe(include=["object", "category"]).T.reset_index()
    resumen_categorico.rename(columns={"index": "Variable", "freq": "A.freq"}, inplace=True)

    # Convertir columnas a tipo numérico (por seguridad)
    resumen_categorico["A.freq"] = pd.to_numeric(resumen_categorico["A.freq"], errors="coerce")
    resumen_categorico["count"] = pd.to_numeric(resumen_categorico["count"], errors="coerce")

    # Calcular frecuencia relativa del valor más frecuente
    resumen_categorico["R.freq"] = (
        resumen_categorico["A.freq"] / resumen_categorico["count"]
    ).round(3)

    # Crear columna con el porcentaje en formato string con dos decimales y símbolo %
    resumen_categorico["Pct"] = resumen_categorico["R.freq"].map(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
    )

    # Mostrar resultados en Streamlit con estilo (tabla estática)
    st.markdown("### 🏷️ Resumen de Variables Categóricas")

    # Formato para columnas numéricas
    fmt = {
        "A.freq": "{:,.0f}",
        "count": "{:,.0f}",
        "R.freq": "{:.3f}"
    }

    styled_cat = (
        resumen_categorico.style
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f8f9fa"), ("color", "#111111"), ("font-weight", "600")]},
            {"selector": "tbody td", "props": [("font-size", "13px")]}
        ])
        .format(fmt)
        .set_properties(**{"text-align": "left"}, subset=["Variable"]) 
        .set_properties(**{"text-align": "center"}, subset=["A.freq", "count", "R.freq", "Pct"]) 
        .hide(axis="index")
    )

    st.dataframe(styled_cat, use_container_width=True)

    st.markdown("""
                    
                    """)  
     
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    En la variable Facultad, con 590.412 registros y 10 categorías únicas, la más frecuente es Ciencias Económicas, que aparece 162.135 veces. Esto representa una frecuencia relativa de 0.2750 y un 27.50% del total.
                    En cuanto a Programa (Académico), que contiene 23 categorías, el más común es Derecho, con 66.000 registros. Esto equivale a una frecuencia relativa de 0.112 y un 11.20% del total de observaciones.
                    La variable Asignatura cuenta con 3.999 categorías diferentes, siendo INGLÉS I la más frecuente, con 6.072 apariciones. Su representación porcentual es de 1.00%, lo cual evidencia su baja proporción frente al total de registros.
                    Respecto al Grupo, con 86 categorías, la más repetida es A1, con 400.042 registros. Esta categoría concentra una frecuencia relativa de 0.6776, es decir, un 67.76% del total de casos.
                    Por último, la variable Rendimiento, con 5 categorías posibles, tiene como valor más frecuente Alto, con 166.349 registros. Esto equivale a una frecuencia relativa de 0.2817 y un 28.18% del total de observaciones.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.4. COMPORTAMIENTO DEL RENDIMIENTO ACADÉMICO ESTUDIANTILA</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)     
 
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    La siguiente figura, corresponde a una gráfica de barras que muestra la distribución de los estudiantes según los niveles de rendimiento académico (alcanzado). En el eje de las ordenadas (es decir, en el eje "y") se encuentra el número de estudiantes, mientras que en el eje de las abscisas (esto es, en el eje "x") se ubican las sesis (06) categorías de rendimiento definidas por intervalos de calificación.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)

    # Gráfico de barras para Rendimiento (Plotly / Matplotlib opcional)
    if 'Rendimiento' not in df.columns:
        st.warning("La columna 'Rendimiento' no está presente en el DataFrame. Imposible generar el gráfico de rendimiento.")
    else:
        # Conteo y orden jerárquico
        orden = ["Insuficiente", "Deficiente", "Bajo", "Medio", "Alto", "Superior"]
        rendimiento_counts = df['Rendimiento'].value_counts().reindex(orden).fillna(0).astype(int)

        # Preparar DataFrame (asegurar columna 'count')
        df_rend = rendimiento_counts.reset_index(name='count').rename(columns={"index": "Rendimiento"})

        # Forzar tipo numérico y calcular totales de forma segura
        df_rend['count'] = pd.to_numeric(df_rend['count'], errors='coerce').fillna(0).astype(int)
        total = int(df_rend['count'].sum())
        df_rend['perc'] = (df_rend['count'] / total * 100).round(2) if total > 0 else 0

        # Paleta y etiquetas
        palette = {
            "Insuficiente": "#E3F2FD",
            "Deficiente": "#BBDEFB",
            "Bajo": "#90CAF9",
            "Medio": "#64B5F6",
            "Alto": "#1976D2",
            "Superior": "#0D47A1"
        }

        labels_dict = {
            "Insuficiente": "Insuficiente < 2.0",
            "Deficiente": "Deficiente ≥ 2.0",
            "Bajo": "Bajo ≥ 3.0",
            "Medio": "Medio ≥ 3.5",
            "Alto": "Alto ≥ 4.0",
            "Superior": "Superior > 4.5"
        }

        # Controles interactivos
        #st.markdown("**Filtrar categorías de Rendimiento**")
        available_cats = df_rend.loc[df_rend['count'] > 0, 'Rendimiento'].tolist()
        default = [c for c in orden if c in available_cats]
        selected = st.multiselect("Selecciona categorías", options=available_cats, default=default)

        if not selected:
            st.warning("Selecciona al menos una categoría para mostrar el gráfico.")
        else:
            plot_df = df_rend[df_rend['Rendimiento'].isin(selected)].copy()
            plot_df['label'] = plot_df['Rendimiento'].map(labels_dict)
            plot_df['color'] = plot_df['Rendimiento'].map(palette)
            plot_df['text'] = plot_df.apply(lambda r: f"{r['count']:,} ({r['perc']}%)", axis=1)

            # Option: static Matplotlib or interactive Plotly
            use_static = st.checkbox("Ver gráfico estático estilo presentación (Matplotlib)", value=False)

            if use_static:
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(
                    plot_df['label'],
                    plot_df['count'],
                    color=[palette[r] for r in plot_df['Rendimiento']],
                    edgecolor='black'
                )

                max_count = pd.to_numeric(plot_df['count'], errors='coerce').max()
                max_count = int(max_count) if pd.notnull(max_count) else 0
                ymax = int(max_count * 1.08) if max_count > 0 else 1
                ax.set_ylim(0, max(180000, ymax))

                # Anotaciones
                for bar, txt in zip(bars, plot_df['text']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + (ymax * 0.005), txt,
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Leyenda con patches
                handles = [mpatches.Patch(color=palette[r], label=labels_dict[r]) for r in plot_df['Rendimiento']]
                ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

                ax.set_title('Distribución del Rendimiento Académico')
                ax.set_xlabel('Rendimiento')
                ax.set_ylabel('Número de Estudiantes')
                ax.set_xticks(range(len(plot_df['label'])))
                ax.set_xticklabels(plot_df['label'], rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                fig = px.bar(
                    plot_df,
                    x='label',
                    y='count',
                    text='text',
                    color='Rendimiento',
                    color_discrete_map=palette,
                    category_orders={'Rendimiento': orden}
                )

                fig.update_traces(textposition='outside')
                fig.update_layout(
                    title='Distribución del Rendimiento Académico',
                    xaxis_title='',
                    yaxis_title='Número de Estudiantes',
                    legend_title_text='Rendimiento',
                    margin=dict(r=180),
                    uniformtext_minsize=8,
                    uniformtext_mode='hide'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Tabla resumen debajo del gráfico
            summary = plot_df[['label', 'count', 'perc']].copy()
            summary = summary.rename(columns={'label': 'Categoría', 'count': 'Frecuencia', 'perc': 'Porcentaje (%)'})

            # Formatear columnas para presentación
            summary['Frecuencia'] = summary['Frecuencia'].map(lambda x: f"{int(x):,}")
            summary['Porcentaje (%)'] = summary['Porcentaje (%)'].map(lambda x: f"{float(x):.2f}")

            # Mostrar encabezado y la tabla estilizada
            #st.markdown("### Distribución del Rendimiento Académico")
            # usar st.dataframe para mantener el ancho y permitir copiar/paginado
            styled_summary = (
                summary.style
                .set_table_styles([
                    {"selector": "thead th", "props": [("background-color", "#111111"), ("color", "#ffffff"), ("font-weight", "600")]},
                    {"selector": "tbody td", "props": [("font-size", "13px"), ("text-align", "center")]}
                ])
                .hide(axis="index")
            )

            st.dataframe(styled_summary, use_container_width=True)
   
    st.markdown("""
                    
                    """)

    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    El nivel Insuficiente (< 2.00), se corresponde con 25.606 estudiantes, que equivale al 4.34%. El nivel Deficiente (≥ 2.00) cuenta con 31.947 estudiantes, lo que representa el 5.41% del total. El nivel Bajo (≥ 3.00) agrupa a 98.758 estudiantes, equivalentes al 16.73%. Posteriormente, el nivel Medio (≥ 3.50) alcanza un total de 133.608 estudiantes, que corresponden al 22.63% de la muestra.
                    La categoría Alto (≥ 4.00) concentra la mayor proporción de estudiantes, con 166.349 casos, es decir, el 28.18% del total. Por su parte, el nivel Superior (> 4.50) reúne 134.144 estudiantes, equivalente al 22.72%.
                    En conjunto, la gráfica permite visualizar la distribución de las frecuencias absolutas y relativas de los estudiantes en cada nivel de rendimiento, destacando la presencia de un mayor número de estudiantes en las categorías altas en comparación con las más bajas.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.5. NOTA FINAL PROMEDIO POR PROGRAMA ACADEMICO</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)     
 
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    La siguiente figura corresponde a una gráfica de barras horizontales que muestra el valor medio de la nota final por cada programa académico. En el eje de las ordenadas (es decir, en el eje de las "y") se encuentran listados los programas, mientras que en el eje de las abscisas (esto es, en el eje de las "x") se representan las notas promedio, con un rango que va de 0 a 5.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """) 
    
    # Gráfico de barras horizontales para Nota Final por Programa (Plotly / Matplotlib opcional)
    if 'Programa' not in df.columns or 'Nota Final' not in df.columns: 
        st.warning("Las columnas 'Programa' o 'Nota Final' no están presentes en el DataFrame. Imposible generar el gráfico de nota final por programa.")     
    else:
        # Calcular promedio de Nota Final por Programa
        nota_numeric = pd.to_numeric(df['Nota Final'], errors='coerce')
        df_prog = df[['Programa']].copy()
        df_prog['Nota Final'] = nota_numeric
        avg_nota_prog = (
            df_prog.groupby('Programa', as_index=False)
            .agg({'Nota Final': 'mean'})
            .rename(columns={'Nota Final': 'Avg Nota Final'})
        )
        
        # Ordenar de mayor a menor promedio
        avg_nota_prog = avg_nota_prog.sort_values(by='Avg Nota Final', ascending=False)

        # Controles interactivos
        st.markdown("**Filtrar Programas Académicos**")
        available_programs = avg_nota_prog['Programa'].tolist()
        default_programs = available_programs[:23]  # seleccionar los 23 por defecto
        selected_programs = st.multiselect("Selecciona programas", options=available_programs, default=default_programs)

        if not selected_programs:
            st.warning("Selecciona al menos un programa para mostrar el gráfico.")
        else:
            plot_df = avg_nota_prog[avg_nota_prog['Programa'].isin(selected_programs)].copy()
            plot_df['text'] = plot_df['Avg Nota Final'].map(lambda x: f"{x:.2f}")

            # Option: static Matplotlib or interactive Plotly
            use_static1 = st.checkbox("Ver gráfico estático estilo presentación (Matplotlib)", value=False, key="eda_static")

            if use_static1:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(
                    plot_df['Programa'],
                    plot_df['Avg Nota Final'],
                    color='#1976D2',
                    edgecolor='black'
                )

                max_avg = pd.to_numeric(plot_df['Avg Nota Final'], errors='coerce').max()
                max_avg = float(max_avg) if pd.notnull(max_avg) else 0
                xmax = max_avg * 1.08 if max_avg > 0 else 1
                ax.set_xlim(0, max(5.0, xmax))

                # Anotaciones
                for bar, txt in zip(bars, plot_df['text']):
                    width = bar.get_width()
                    ax.text(width + (xmax * 0.005), bar.get_y() + bar.get_height() / 2, txt,
                            ha='left', va='center', fontsize=9, fontweight='bold')  
                ax.set_title('Nota Final Promedio por Programa Académico')
                ax.set_xlabel('Nota Final Promedio')
                ax.set_ylabel('Programa')
                plt.tight_layout()
                st.pyplot(fig)
                
            else:   
                fig = px.bar(
                    plot_df,
                    x='Avg Nota Final',
                    y='Programa',
                    text='text',
                    orientation='h',
                    color_discrete_sequence=["#0C549C"]
                )

                fig.update_traces(textposition='outside',
                                  textfont=dict(size=11, color='black', family='Arial',),
                                  hovertemplate='<b>%{y}</b><br>Nota Promedio: %{x:.2f}<extra></extra>'
                                  )
                
                fig.update_layout(
                    height=600,
                    title='Nota Final Promedio por Programa Académico',
                    xaxis_title='Nota Final Promedio',
                    yaxis_title='Programa',
                    margin=dict(r=200),
                    uniformtext_minsize=8,
                    uniformtext_mode='hide'
                )

                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                    
                    """) 
    
    st.markdown("""
                    <div>
                        <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                        El programa con el promedio más alto corresponde a <b>Licenciatura en Educación Infantil</b>, con una nota de <b>4.42</b>, seguido de <b>Licenciatura en Educación</b> con <b>4.22</b>. En tercer lugar se ubica <b>Comunicación Social</b> (4.02), mientras que <b>Ingeniería Química</b> e <b>Historia</b> registran un promedio de <b>3.99</b>. Estos programas conforman el grupo de mejor desempeño académico dentro de la gráfica.
                        <br><br>
                        En un nivel intermedio destacan <b>Medicina</b> (3.92), <b>Lenguas Extranjeras</b> (3.91), <b>Administración de Empresas</b> (3.85), <b>Derecho</b> (3.83), <b>Lingüística y Literatura</b> (3.83) y <b>Administración Industrial</b> (3.80). Todos ellos se concentran en torno al valor de 3.8, representando una franja de rendimiento medio-alto.
                        <br><br>
                        Con promedios ligeramente inferiores se encuentran <b>Contaduría Pública</b> (3.76), <b>Odontología</b> (3.74), <b>Ingeniería Civil</b> (3.74) y <b>Química Farmacéutica</b> (3.72), seguidos de <b>Enfermería</b> (3.66), <b>Química</b> (3.64) y <b>Economía</b> (3.61).
                        <br><br>
                        Finalmente, los resultados más bajos corresponden a <b>Ingeniería de Alimentos</b> (3.53), <b>Matemáticas</b> (3.49), <b>Biología</b> (3.49), <b>Ingeniería de Sistemas</b> (3.48) y <b>Filosofía</b> (3.38), este último con el promedio más bajo registrado.
                        <br><br>
                        En conjunto, la visualización permite contrastar el rendimiento promedio de los distintos programas académicos, evidenciando una variación que va desde <b>4.42</b> hasta <b>3.38</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """) 


    # --- Tabla Rendimiento  ---
    # 0) Orden de Rendimiento
    orden_rend = ["Insuficiente", "Deficiente", "Bajo", "Medio", "Alto", "Superior"]
    df["Rendimiento"] = pd.Categorical(df["Rendimiento"], categories=orden_rend, ordered=True)

    # 1) Promedios por rendimiento (pivot)
    promedios = df.pivot_table(
        values="Nota Final",
        index="Programa",
        columns="Rendimiento",
        aggfunc="mean",
        observed=False
    ).round(2)

    # 2) Conteos y porcentajes
    conteos = df.pivot_table(
        values="Nota Final",
        index="Programa",
        columns="Rendimiento",
        aggfunc="count",
        observed=False
    )
    totales = conteos.sum(axis=1)
    porcentajes = (conteos.div(totales, axis=0) * 100).round(1)

    # 3) Combinar promedios y porcentajes
    tabla_final = promedios.combine(
        porcentajes,
        lambda prom, perc: prom.round(2).astype(str) + " (" + perc.round(1).astype(str) + "%)"
    ).reset_index()

    # ---------- PROMEDIO GENERAL ----------
    promedios_generales = df.groupby("Programa")["Nota Final"].mean().round(2)
    max_programa = promedios_generales.max()
    porcentajes_generales = ((promedios_generales / max_programa) * 100).round(1)
    formatted_general = promedios_generales.astype(str) + " (" + porcentajes_generales.astype(str) + "%)"

    tabla_final["Promedio General"] = tabla_final["Programa"].map(lambda x: f"{promedios_generales[x]:.2f}")

    orden_promedio = promedios_generales.sort_values(ascending=False).index
    tabla_final = tabla_final.set_index("Programa").loc[orden_promedio].reset_index()

    # MultiIndex columnas
    tabla_final.columns = pd.MultiIndex.from_tuples(
        [("","Programa")] +
        [("Rendimiento Académico", col) for col in orden_rend] +
        [("","Media")]
    )

    # Estilo (Pandas Styler)
    styled = (
        tabla_final.style
        .set_table_styles([
            {"selector": "caption",
            "props": [("font-size", "16px"), ("font-weight", "bold"), ("color", "white"), ("background-color", "#1976D2"), ("padding", "6px 10px"), ("border-radius", "6px")]},
            {"selector": "th",
            "props": [("text-align", "center")]},
            {"selector": "td",
            "props": [("text-align", "right")]},
            {"selector": "td:first-child",
            "props": [("text-align", "left")]}
        ])
        .hide(axis="index")
    )

    # --- Visualización en Streamlit ---
    st.subheader("Distribución de la Nota Final Promedio por Rendimiento y Programa Académico")

    # Prefijo para claves (evita IDs duplicados si lo usas en varias páginas)
    PREFIX = "eda_tabla_rend_prog"

    modo = st.radio(
        "Modo de visualización",
        ["Interactiva (st.dataframe)", "HTML estilizado (to_html)"],
        index=0,
        key=f"{PREFIX}_modo"
    )

    if modo.startswith("Interactiva"):
        # st.dataframe no muestra bien los MultiIndex de columnas; los aplanamos para legibilidad
        df_flat = tabla_final.copy()
        df_flat.columns = [
        ("{} - {}".format(a.strip(), b.strip()) if a and b else (a or b)).strip()
        for a, b in df_flat.columns.to_list()
    ]
        st.dataframe(df_flat, use_container_width=True, key=f"{PREFIX}_df")
    else:
        # Render HTML del Styler (mejor respeta el CSS del Styler)
        st.markdown(styled.to_html(), unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>3.6. NOTA FINAL PROMEDIO POR AÑO Y PERÍODO ACADÉMICO</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)     
 
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    La siguiente gráfica de columnas muestra la evolución del valor medio de la nota final entre los años 2014 y 2023, diferenciado por períodos académicos (a saber: 1 semestre y 2 semestre). En general, los valores se mantienen relativamente estables en la mayoría de los años, con un comportamiento particular en el 2020 y 2021.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """) 

# Gráfico de columnas para Nota Final por Año y Periodo
    # --- 1) Preparar datos ---
    # Ajustar el orden del periodo si es necesario (ejemplo "I", "II")
    if pd.api.types.is_numeric_dtype(df["Periodo"]):
        period_order = sorted(df["Periodo"].dropna().unique().tolist())
    else:
        period_order = sorted(df["Periodo"].dropna().unique().tolist())

    df = df.copy()
    df["Periodo"] = pd.Categorical(df["Periodo"], categories=period_order, ordered=True)

    # Calcular promedio por año y periodo
    df_prom = (
        df.groupby(["Año", "Periodo"], as_index=False)["Nota Final"]
        .mean()
        .sort_values(["Año", "Periodo"])
    )
    df_prom["label"] = df_prom["Nota Final"].map(lambda x: f"{x:.2f}")

    # --- 2) Crear gráfico Plotly ---
    fig = px.bar(
        df_prom,
        x="Año",
        y="Nota Final",
        color="Periodo",
        text="label",
        barmode="group",
        color_discrete_sequence=["#0D47A1", "#90CAF9"][:len(period_order)]
    )

    # --- 3) Configuración visual ---
    ymax = df_prom["Nota Final"].max() if len(df_prom) else 0
    upper = 0.5 * np.ceil((ymax + 0.05) / 0.5) if ymax > 0 else 0.5

    fig.update_yaxes(
        title_text="Nota Final Promedio",
        tickformat=".2f",
        dtick=0.5,
        range=[0, upper]
    )
    fig.update_xaxes(title_text="Año")

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>Año %{x}</b><br>Nota: %{y:.2f}<extra></extra>",
        cliponaxis=False
    )

    fig.update_layout(
        height=500,
        title="Nota Final Promedio por Año y Periodo Académico",
        legend_title_text="Periodo Académico",
        uniformtext_minsize=8,
        uniformtext_mode="show",
        margin=dict(t=60, r=40, b=40, l=60)
    )

    # --- 4) Mostrar en Streamlit ---
    st.plotly_chart(fig, use_container_width=True, key="eda_cols_anio_periodo_plotly")

    st.markdown("""
                    
                    """)    
    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    Entre 2014 y 2019, los promedios rondan entre 3.64 y 3.78, evidenciando estabilidad y sin variaciones significativas entre los dos períodos de cada año. Los valores más bajos de este lapso se observan en 2015 (3.64 en el período 2) y los más altos en 2018 (3.78 en el período 2).
                    <br><br>
                    En el año 2020 se presenta un incremento destacado, con promedios de 4.22 en el período 1 y 4.19 en el período 2, lo que representa un aumento considerable respecto a los años anteriores. Esta tendencia ascendente se acentúa en 2021, donde se alcanzan los valores más altos de toda la serie: 4.18 en el período 1 y un máximo de 4.50 en el período 2.
                    <br><br>
                    A partir de 2022, los promedios retornan a niveles similares a los de años previos, ubicándose entre 3.75 y 3.71 en 2022 y en 3.76 y 3.73 en el 2023, lo que sugiere un regreso a la estabilidad después del repunte observado entre 2020 y el 2021.
                    <br><br>
                    En conclusión, la gráfica refleja un comportamiento estable en los promedios entre 2014 y 2019, un repunte significativo en 2020 y 2021 y un retorno a valores regulares en los años posteriores.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)    


   
def page_model():
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#111111; font-weight:700; font-size:32px; margin:0 0 6px 0;'>MODELO ESTADÍSTICO</h1>
                    <p style='color:#333333; font-size:16px; margin:0;'>Descripción del enfoque de modelado y resultados principales.</p>
                </div>
                """, unsafe_allow_html=True)

def page_conclusions():
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#111111; font-weight:700; font-size:32px; margin:0 0 6px 0;'>CONCLUSIONES</h1>
                    <p style='color:#333333; font-size:16px; margin:0;'>Resumen de hallazgos clave, limitaciones y recomendaciones.</p>
                </div>
                """, unsafe_allow_html=True)

def page_refs():
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#111111; font-weight:700; font-size:32px; margin:0 0 6px 0;'>REFERENCIAS</h1>
                    <p style='color:#333333; font-size:16px; margin:0;'>Enlaces, datasets y bibliografía utilizada en el análisis.</p>
                </div>
                """, unsafe_allow_html=True)

# Map routes to functions
ROUTES = {
    "Introducción": page_intro,
    "Objetivos": page_objectives,
    "Exploración de Datos (EDA)": page_eda,
    "Modelo": page_model,
    "Conclusiones": page_conclusions,
    "Referencias": page_refs,
}

ROUTES[choice]()



