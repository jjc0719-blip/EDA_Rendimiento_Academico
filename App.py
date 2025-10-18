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
from scipy.stats import kstest, pearsonr, chi2_contingency, norm   
import pickle
import traceback 
import plotly.graph_objects as go 
from PIL import Image  


st.set_page_config(page_title="Modelado del Rendimiento Académicos de Estudiantes Universitarios de Programas de Pregrado Presencial con el algoritmo XGBoost",
                   layout="wide")

# Sidebar configuration
with st.sidebar:
    # Sidebar header (bold + larger font)
    st.markdown(
        """
        <h1 style='text-align: left; font-weight: 600; font-family: Tahoma, "Tahoma", Geneva, sans-serif; font-size: 26px;'>🎓 Modelado del Rendimiento Académicos de Estudiantes Universitarios de Programas de Pregrado Presencial con el algoritmo XGBoost</h1>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("""
 
                """)
    
    # Navigation menu
    choice = option_menu(
        "Capítulos",
        ["Exploración de Datos (EDA)","Modelo Predicción"],
        icons=["book", "bar-chart","collection-play"],
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

def page_eda():
    st.markdown("""
                <div style='position:fixed; top:40px; left:400px; right:24px; background:#ffffff; padding:10px 16px; z-index:9999; border-bottom:1px solid rgba(0,0,0,0.06);'>
                    <h1 style='color:#111111; font-weight:700; font-size:50px; margin:0;'>🔍 EXPLORACIÓN DE LOS DATOS (EDA)</h1>
                </div>
                """, unsafe_allow_html=True)
    
    #Contenido Contexto
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <h1 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>🧭 CONTEXTO</h1>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                    El presente proyecto tiene como objetivo analizar, modelar y predecir el rendimiento académico de los estudiantes —considerado un indicador indirecto de deserción— a partir de variables académicas y de desempeño, mediante técnicas estadísticas descriptivas, inferenciales y de aprendizaje automático. Con datos públicos de una universidad colombiana (2014–2023), que incluyen calificaciones y variables de contexto académico, se busca identificar los factores más influyentes sobre la nota final y estimar la probabilidad de bajo rendimiento. Los resultados aspiran a fortalecer la detección temprana de estudiantes en riesgo y apoyar la toma de decisiones institucionales orientadas a mejorar la calidad y la permanencia en la educación superior.
                    </p>
                </div>
                """, unsafe_allow_html=True)
       
    # ---------- Cargue cache y Parquet ----------

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
 # INICIA EDA
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    st.markdown("""
               
                """, unsafe_allow_html=True)

    st.markdown("### 📋 INFORMACIÓN DATAFRAME")

    # Mostrar métricas rápidas: filas, columnas y elementos
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros Iniciales", f"{df.shape[0]:,}")
        c2.metric("Variables", f"{df.shape[1]:,}")
        c3.metric("Observaciones", f"{df.size:,}")
    
        # -------- Eliminar duplicados y mostrar métricas --------

        #initial_count = int(df.shape[0])
        dup_count = int(df.duplicated().sum())
        
        # eliminar duplicados si existen
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        after_count = int(df.shape[0])
        
        # Por columna
        missing_by_col = df.isnull().sum().sort_values(ascending=False)
        total_missing = int(missing_by_col.sum())

        d1, d2, d3 = st.columns(3)
        d1.metric("Total faltantes", f"{total_missing:,}")
        d2.metric("Total Duplicados", f"{dup_count:,}")
        d3.metric("Registros Finales", f"{after_count:,}")

    st.markdown("""
                
                """, unsafe_allow_html=True)
        
    # Construir una tabla similar a df.info() pero en formato DataFrame
    if df is not None:
        # ----------------------------
        # 1️⃣ Tu tabla de variables base
        # ----------------------------
        var_types = [ 
            ("Información Académica", "Facultad", "Categórica / Nominal", "590412", "0"),
            ("Información Académica", "Programa", "Categórica / Nominal", "590412", "0"),
            ("Información Académica", "Código Asignatura", "Categórica / Nominal", "590412", "0"),
            ("Información Académica", "Asignatura", "Categórica / Nominal", "590412", "0"),
            ("Información Académica", "Grupo", "Categórica / Nominal", "590412", "0"),
            ("Información Académica", "Código Estudiantil", "Numérica / Discreta",  "590412", "0"),
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

        # Convertimos a DataFrame para hacer merge
        var_df = pd.DataFrame(var_types, columns=[
            "Clase de Atributo", "Variable", "Tipo de Variable", "Total Registros", "Registros Nulos (Ref)"
        ])
        
        # ----------------------------
        # 2️⃣ Construcción de df_info base
        # ----------------------------
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
        
        # Merge con tus clasificaciones
        df_info = df_info.merge(var_df, on="Variable", how="left")

        # justo antes de reordenar columnas
        if 'Tipo' in df_info.columns and 'Tipo (Pandas)' not in df_info.columns:
            df_info = df_info.rename(columns={'Tipo': 'Tipo (Pandas)'})

        orden = [
            "Variable", "Clase de Atributo", "Tipo de Variable", 
            "Tipo (Pandas)", "Cantidad Registros", "Registros Nulos", "% No Nulos",
            "Total Registros", "Registros Nulos (Ref)"
        ]
        # reordenar solo con columnas que existan (evita KeyError)
        df_info = df_info.reindex(columns=[c for c in orden if c in df_info.columns])

        # Reordenar columnas
        df_info = df_info[[
            "Variable", "Clase de Atributo", "Tipo de Variable", 
            "Tipo (Pandas)", "Cantidad Registros", "Registros Nulos"
        ]]        

        # Mostrar tabla estilizada
        styled = df_info.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f8f9fa"), ("color", "#111111"), ("font-weight", "600")]},
            {"selector": "tbody td", "props": [("font-size", "13px"), ("text-align", "center")]}
        ]).format({"Porc No Nulos": "{:.2f}%"})

        st.dataframe(styled, use_container_width=True)
                        
        st.markdown("""
                    
                    """)
        
        st.markdown("""
                <div
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'>
                    Tras la eliminación de las filas duplicadas y la verificación de la ausencia de datos faltantes en el conjunto de datos, el proceso de análisis se ve considerablemente simplificado, ya que no es necesario aplicar técnicas de imputación.
                """, unsafe_allow_html=True)

    st.markdown("""
               
                """, unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    st.markdown("""
                <div
                    <h2 style='color:#111111; font-weight:600; font-size:30px; margin:18px 0 6px 0;'>📝 RECORD DE NOTAS</h2>
                    <br>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
               
                """, unsafe_allow_html=True)  

    
    # 1️⃣ Detectar columnas tipo "Nota"
    record_cols = [c for c in df.columns if c.strip().lower().startswith("nota")]

    if not record_cols:
        st.warning("No se encontraron columnas de record de notas (p. ej., 'Nota 1', 'Nota Final').")
        st.stop()

    # 2️⃣ Selector de variables
    sel_cols = st.multiselect(
        "Selecciona las variables de notas a comparar:",
        options=record_cols,
        default=[c for c in record_cols if any(x in c for x in ["Final", "Definitiva"])] or record_cols[:2],
        help="Puedes escoger varias variables de notas para comparar su distribución."
    )

    if not sel_cols:
        st.info("Selecciona al menos una variable para continuar.")
        st.stop()

    # 3️⃣ Convertir a formato largo
    df_long = df[sel_cols].copy()
    for c in sel_cols:
        df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    df_long = df_long.melt(var_name="Variable", value_name="Valor").dropna(subset=["Valor"])
    n = len(df_long)
    if n == 0:
        st.warning("No hay datos numéricos válidos para las columnas seleccionadas.")
        st.stop()

    # 4️⃣ Crear boxplot
    fig = px.box(
        df_long,
        x="Variable",
        y="Valor",
        color="Variable",
        points="outliers",
        title=f"Distribución de Notas – Comparación de Variables (n={n})",
        labels={"Valor": "Nota"},
        template="plotly_white"
    )

    # 5️⃣ Añadir puntos de media
    medias = df_long.groupby("Variable")["Valor"].mean()
    fig.add_trace(
        go.Scatter(
            x=medias.index,
            y=medias.values,
            mode="markers+text",
            marker_symbol="diamond",
            marker_size=10,
            marker_color="red",
            text=[f"{m:.2f}" for m in medias.values],
            textposition="top center",
            name="Media"
        )
    )

    # 6️⃣ Ajustes visuales
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18)),
        yaxis_title="Nota",
        xaxis_title="Variable",
        boxmode="group",
        showlegend=True,
        margin=dict(t=70, b=50, l=50, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    # 7️⃣ Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                    
                    """)   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
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
    st.markdown("### 🏷️ INFORMACIÓN ACADEMICA")
    
    st.markdown("""
               
                """, unsafe_allow_html=True)  

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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("### 📈 COMPORTAMIENTO DEL RENDIMIENTO ACADÉMICO ESTUDIANTIL")
    
    st.markdown("""
                    
                    """)
    
    
        # Validaciones de columnas requeridas
    col_programa = "Programa"  # ajusta si tu columna se llama distinto
    col_rend = "Rendimiento"

    if col_rend not in df.columns:
        st.warning(f"La columna '{col_rend}' no está presente en el DataFrame.")
    else:
        if col_programa not in df.columns:
            st.warning(f"La columna '{col_programa}' no está presente en el DataFrame. Se mostrará sin filtro por programa.")
            df_filtrado = df.copy()
            seleccion_modo = "Todos"
            seleccion_programas = []
        else:
            # ---- Selector de programa(s) con opción 'Todos' ----
            #st.subheader("🎓 Filtro por Programa Académico")
            programas_unicos = (
                df[col_programa]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            seleccion_modo = st.radio(
                "Ver:",
                ["Todos los programas", "Filtrar por programa(s)"],
                horizontal=True
            )

            if seleccion_modo == "Todos los programas":
                df_filtrado = df.copy()
                seleccion_programas = []
            else:
                seleccion_programas = st.multiselect(
                    "Selecciona programa(s):",
                    options=programas_unicos,
                    default=[],
                    help="Si no seleccionas ninguno, no se mostrará el gráfico."
                )
                if len(seleccion_programas) == 0:
                    st.info("Selecciona al menos un programa o cambia a 'Todos'.")
                    st.stop()
                df_filtrado = df[df[col_programa].astype(str).isin(seleccion_programas)].copy()
                if df_filtrado.empty:
                    st.warning("No hay registros para los programas seleccionados.")
                    st.stop()

        # ---- Agregación de rendimiento (con orden estándar y extras) ----
        orden_std = ["Insuficiente", "Deficiente", "Bajo", "Medio", "Alto", "Superior"]
        categorias_presentes = (
            df_filtrado[col_rend]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        extras = [c for c in categorias_presentes if c not in orden_std]
        orden = orden_std + sorted(extras)

        counts = (
            df_filtrado[col_rend]
            .astype(str)
            .value_counts()
            .reindex(orden)
            .fillna(0)
            .astype(int)
        )

        df_rend = counts.reset_index(name="count").rename(columns={"index": col_rend})
        total = int(df_rend["count"].sum())
        df_rend["perc"] = (df_rend["count"] / total * 100).round(2) if total > 0 else 0.0

        # ---- Paleta y etiquetas ----
        base_palette = {
            "Insuficiente": "#E3F2FD",
            "Deficiente":  "#BBDEFB",
            "Bajo":         "#90CAF9",
            "Medio":        "#64B5F6",
            "Alto":         "#1976D2",
            "Superior":     "#0D47A1",
        }
        palette = {c: base_palette.get(c, "#9E9E9E") for c in orden}

        labels_dict = {
            "Insuficiente": "Insuficiente < 2.0",
            "Deficiente":   "Deficiente ≥ 2.0",
            "Bajo":         "Bajo ≥ 3.0",
            "Medio":        "Medio ≥ 3.5",
            "Alto":         "Alto ≥ 4.0",
            "Superior":     "Superior > 4.5",
        }
        for c in extras:
            labels_dict.setdefault(c, c)

        # ---- UI de categorías visibles y modo de eje ----
        # st.subheader("📊 Distribución del Rendimiento Académico")
        available_cats = df_rend.loc[df_rend["count"] > 0, col_rend].tolist() or orden

        mostrar_porcentaje = st.checkbox("Ver eje Y en porcentaje (%)", value=False)
        #use_static = st.checkbox("Ver gráfico estático (Matplotlib)", value=False)


        # ---- Construcción del DataFrame a graficar ----
        plot_df = df_rend[df_rend[col_rend].isin(available_cats)].copy()
        plot_df["label"] = plot_df[col_rend].map(labels_dict)
        plot_df["color"] = plot_df[col_rend].map(palette)
        plot_df["text_count"] = plot_df.apply(lambda r: f"{r['count']:,} ({r['perc']}%)", axis=1)

        # ---- Título dinámico según el filtro ----
        if seleccion_modo == "Todos":
            subtitulo = "Todos los programas"
        else:
            # limitar en título si hay muchos
            if len(seleccion_programas) <= 3:
                subtitulo = ", ".join(seleccion_programas)
            else:
                subtitulo = f"{len(seleccion_programas)} programas seleccionados"

        titulo = f"Distribución del Rendimiento Académico — {subtitulo}"

        # ---- Render: Plotly ----
        y_col = "perc" if mostrar_porcentaje else "count"
        y_title = "Porcentaje de Estudiantes (%)" if mostrar_porcentaje else "Número de Estudiantes"

        fig = px.bar(
                plot_df,
                x="label",
                y=y_col,
                text="text_count",
                color=col_rend,
                color_discrete_map=palette,
                category_orders={col_rend: orden},
                title=titulo
            )

        fig.update_traces(textposition="outside", cliponaxis=False)
        if mostrar_porcentaje:
                fig.update_yaxes(range=[0, 100])

        fig.update_layout(
                xaxis_title="",
                yaxis_title=y_title,
                legend_title_text="Rendimiento",
                margin=dict(r=180),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title_x=0.5
            )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
                    
                    """)

    st.markdown("""
                    <div>
                    <p style='color:#444444; text-align:justify; font-size:20px; margin:0 0 12px 0;'> 
                    El análisis comparativo del rendimiento académico por programa revela diferencias claras entre áreas. Los promedios más altos corresponden a Licenciatura en Educación Infantil (4.42) y Licenciatura en Educación (4.22), donde predomina el nivel Superior (cerca del 50% de los estudiantes). También destacan Comunicación Social (4.02) e Ingeniería Química/Historia (3.99), con fuerte presencia en los niveles Alto y Superior. En un rango intermedio se ubican programas como Medicina, Lenguas Extranjeras, Administración de Empresas y Derecho, con medias alrededor de 3.8 - 3.9. En contraste, los promedios más bajos se observan en Matemáticas (3.49), Biología (3.49), Ingeniería de Sistemas (3.48) y especialmente Filosofía (3.38), donde las categorías Deficiente y Bajo concentran una proporción importante.
                    </P>
                    </div>
                    """, unsafe_allow_html=True)   
    
    st.markdown("""
                    
                    """)    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
    st.markdown("### 📈 COMPORTAMIENTO NOTA FINAL") 
    
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
        #title="Nota Final Promedio por Año y Periodo Académico",
        legend_title_text="Periodo Académico",
        uniformtext_minsize=8,
        uniformtext_mode="show",
        margin=dict(t=60, r=40, b=40, l=60)
    )

    # --- 4) Mostrar en Streamlit ---
    st.plotly_chart(fig, use_container_width=True, key="eda_cols_anio_periodo_plotly")

    st.markdown("""
                    
                    """)    

    # =========================
    # Construir tabla formateada y styler
    # =========================
    def construir_tabla_formateada(df: pd.DataFrame):
        orden_rend = ["Deficiente", "Bajo", "Medio", "Alto", "Superior"]
        rend_cat = pd.CategoricalDtype(categories=orden_rend, ordered=True)

        d = df.copy()
        d["Rendimiento"] = d["Rendimiento"].astype(rend_cat)

        # Promedios por rendimiento (Año, Periodo)
        pivot_mean = (
            d.pivot_table(
                values="Nota Final",
                index=["Año", "Periodo"],
                columns="Rendimiento",
                aggfunc="mean"
            )
            .reindex(columns=orden_rend)
        )

        # Conteos por rendimiento (para %)
        pivot_cnt = (
            d.pivot_table(
                values="Nota Final",
                index=["Año", "Periodo"],
                columns="Rendimiento",
                aggfunc="count"
            )
            .reindex(columns=orden_rend)
            .fillna(0)
            .astype(int)
        )

        # % por fila
        row_totals = pivot_cnt.sum(axis=1)
        pct = pivot_cnt.div(row_totals.replace(0, np.nan), axis=0) * 100
        pct = pct.fillna(0)

        # Armar "promedio (porcentaje%)"
        tabla_fmt = pd.DataFrame(index=pivot_mean.index)
        for col in orden_rend:
            mean_col = pivot_mean[col].round(2)
            pct_col = pct[col].round(1)
            tabla_fmt[col] = (
                mean_col.fillna(0).map("{:.2f}".format)
                + " ("
                + pct_col.fillna(0).map("{:.1f}%".format)
                + ")"
            )

        tabla_fmt = tabla_fmt.reset_index()

        # MultiIndex de encabezado
        tabla_fmt.columns = pd.MultiIndex.from_tuples(
            [("", "Año"), ("", "Periodo")] + [("Rendimiento Académico", c) for c in orden_rend]
        )

        # Styler (blanco y negro sobrio)
        styled = (
            tabla_fmt.style
            .set_table_styles([
                {"selector": "caption", "props": [("text-align", "center"),
                                                ("font-weight", "bold"),
                                                ("font-size", "14px")]},
                {"selector": "th", "props": [("text-align", "center"),
                                            ("font-weight", "bold"),
                                            ("border", "1px solid black")]},
                {"selector": "td", "props": [("text-align", "center"),
                                            ("border", "1px solid black")]}
            ], overwrite=False)
            .hide(axis="index")
        )

        return tabla_fmt, styled, pivot_mean.round(2), pct.round(1)

    # Construir 
    try:
        tabla_fmt, styled, pivot_mean, pct = construir_tabla_formateada(df)
    except Exception as e:
        st.error(f"Ocurrió un error al construir la tabla. Verifica columnas y tipos. Detalle: {e}")
        st.stop()

    # selección para vista
    #ver_interactiva = st.radio(
    #    "Modo de visualización",
    #    ["Interactiva (st.dataframe)", "HTML estilizado (to_html)"],
    #    index=0
    #)
    
    # Mostrar
    #if ver_interactiva.startswith("Interactiva"):
    st.dataframe(tabla_fmt, use_container_width=True, height=500)
    #else:
    #    st.markdown(styled.to_html(), unsafe_allow_html=True)
 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                    
                    """)
    # -------------------------------------------------
    # Matriz de Correlación de Person
    # -------------------------------------------------
    
    st.subheader("🔥 MATRIZ DE CORRELACIÓN DE PEARSON")

    # ---------------------------
    # Controles
    # ---------------------------
    # Sugerencia inicial (si existen esas columnas); si no, toma numéricas
    sugeridas = [c for c in ["Nota 1", "Nota 2", "Nota 3", "Nota 4",
                            "Nota Definitiva", "Nota Habilitación", "Nota Final"]
                if c in df.columns]

    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    opciones_cols = sugeridas if len(sugeridas) >= 2 else numericas

    cols = st.multiselect(
        "Selecciona las variables a correlacionar",
        options=opciones_cols if opciones_cols else df.columns.tolist(),
        default=opciones_cols[:min(6, len(opciones_cols))] if opciones_cols else []
    )

      
    solo_triangulo_superior = st.checkbox("Mostrar solo triángulo superior", value=False)
    #usar_absoluto_para_orden = st.checkbox("Ordenar variables por |correlación| con la primera seleccionada", value=False)
    #aplicar_clustering = st.checkbox("Aplicar ordenamiento por clustering jerárquico", value=False)

    if len(cols) < 2:
        st.info("Selecciona al menos dos columnas para calcular la matriz de correlaciones.")
        st.stop()

    # ---------------------------
    # Cálculo de la matriz
    # ---------------------------
    X = df[cols].copy()

    # Eliminar columnas constantes (corr = NaN)
    constantes = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constantes:
        st.warning(f"Columnas con varianza cero (excluidas): {', '.join(constantes)}")
        X = X.drop(columns=constantes)

    if X.shape[1] < 2:
        st.error("No hay suficientes columnas con variación para calcular correlaciones.")
        st.stop()

    corr = X.corr(method="pearson")

    # ---------------------------
    # Ordenamiento opcional
    # ---------------------------
    #orden = list(corr.columns)

    # a) Orden por |corr| respecto a la primera seleccionada
    #if usar_absoluto_para_orden and len(orden) > 1:
    #    ref = orden[0]
    #    orden = [ref] + [c for c in sorted(orden[1:], key=lambda c: -abs(corr.loc[ref, c]))]
    #    corr = corr.loc[orden, orden]


    # ---------------------------
    # Máscara de triángulo superior
    # ---------------------------
    plot_corr = corr.copy().round(2)
    if solo_triangulo_superior:
        mask = np.tril(np.ones_like(plot_corr, dtype=bool), k=-1)
        plot_corr = plot_corr.mask(mask)

    # ---------------------------
    # Heatmap interactivo Plotly
    # ---------------------------
    fig = px.imshow(
        plot_corr,
        text_auto=True,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )

    fig.update_layout(
        title=f"Matriz de correlaciones de Pearson ({len(plot_corr)} variables)",
        margin=dict(l=40, r=20, t=60, b=40),
        coloraxis_colorbar=dict(title="r")
    )
    fig.update_traces(hovertemplate="Fila: %{y}<br>Columna: %{x}<br>r = %{z}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("""
                    
                    """)  
    
    st.subheader("📉 PRUEBA CHI²")
    
    # ---------------------------------------------------
    # Función para prueba Chi²
    # ---------------------------------------------------
    def prueba_chi2(df, var_filas, var_columnas, nombre):
        tabla = pd.crosstab(df[var_filas], df[var_columnas])
        chi2, p, dof, ex = chi2_contingency(tabla)
        interpretacion = "✅ Sí existe asociación" if p < 0.05 else "❌ No existe asociación"
        return {
            "Comparación": f"{nombre} ~ {var_columnas}",
            "Estadístico Chi²": chi2,
            "Grados de libertad": dof,
            "Valor p": p,
            "Interpretación": interpretacion
        }

    # ---------------------------------------------------
    # Ejecutar pruebas solo para las combinaciones deseadas
    # ---------------------------------------------------
    resultados = []

    if {"Programa", "Rendimiento"}.issubset(df.columns):
        resultados.append(prueba_chi2(df, "Programa", "Rendimiento", "Programa"))
    else:
        st.warning("⚠️ No se encontraron las columnas 'Programa' y 'Rendimiento' en el DataFrame.")

    if {"Asignatura", "Rendimiento"}.issubset(df.columns):
        resultados.append(prueba_chi2(df, "Asignatura", "Rendimiento", "Asignatura"))
    else:
        st.warning("⚠️ No se encontraron las columnas 'Asignatura' y 'Rendimiento' en el DataFrame.")

    # ---------------------------------------------------
    # Mostrar resultados si existen
    # ---------------------------------------------------
    if resultados:
        resultados_df = pd.DataFrame(resultados)
        resultados_df["Estadístico Chi²"] = resultados_df["Estadístico Chi²"].map("{:,.2f}".format)
        resultados_df["Valor p"] = resultados_df["Valor p"].map("{:.4e}".format)

        st.dataframe(
            resultados_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No se pudieron generar resultados de Chi² con las columnas disponibles.")

    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

def page_model():

    # Título 
    st.markdown("""
                    <div style='position:fixed; top:40px; left:400px; right:24px; background:#ffffff; padding:10px 16px; z-index:9999; border-bottom:1px solid rgba(0,0,0,0.06);'>
                    <h1 style='color:#111111; font-weight:700; font-size:45px; margin:0;'>MODELO ESTADÍSTICO PREDICCIÓN DEL RENDIMIENTO 🎓 </h1>
                </div>
                    <div style='height:64px;'></div>
                """, unsafe_allow_html=True)
    
    #Contenido 
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                    Este módulo implementa un modelo predictivo desarrollado mediante el algoritmo <strong>XGBoost (Extreme Gradient Boosting)</strong>, reconocido por su alta eficiencia en tareas de clasificación supervisada y su capacidad para manejar grandes volúmenes de datos.  
                    El modelo fue entrenado con una base de datos que reúne más de <strong>470.000 observaciones</strong> de estudiantes de programas universitarios presenciales, integrando tanto variables <strong>numéricas</strong> (notas parciales) como <strong>categóricas</strong> (programa académico y asignatura).  
                    </p>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
---
                """, unsafe_allow_html=True)    
    
    
   
    # ============================================
    # 0) Parche de compatibilidad sklearn (monkey-patch)
    #    Para modelos serializados con otra versión
    #    que usan sklearn.compose._column_transformer._RemainderColsList
    # ============================================
    def _try_monkey_patch_sklearn_for_remainder_cols_list():
        try:
            import sklearn.compose._column_transformer as _ct
            if not hasattr(_ct, "_RemainderColsList"):
                class _RemainderColsList(list):
                    """Compat shim for objects pickled in other sklearn versions."""
                    pass
                _ct._RemainderColsList = _RemainderColsList
        except Exception:
            # Si falla, no interrumpimos la app: el loader manejará el error
            pass

    # ============================================
    # 1) Utilidades de carga automática
    # ============================================
    REQUIRED_KEYS = {"model", "classes", "vars_num", "vars_cat"}

    def _load_backend():
        """Prefiere cloudpickle (soporta lambdas); si no, usa joblib."""
        try:
            import cloudpickle as cp
            return "cloudpickle", cp
        except Exception:
            import joblib
            return "joblib", joblib

    def _candidate_paths():
        """Orden de búsqueda: MODEL_PKL, modelo_xgb.pkl, primer .pkl de la carpeta."""
        env = os.getenv("MODEL_PKL")
        if env and os.path.isfile(env):
            yield env
        default = "modelo_xgb.pkl"
        if os.path.isfile(default):
            yield default
        for p in glob.glob("*.pkl"):
            yield p

    @st.cache_resource
    def load_artifacts_auto():
        """Carga el artefacto dict con claves requeridas, con reintento tras monkey-patch."""
        loader_name, loader_pkg = _load_backend()
        last_err = None

        for path in _candidate_paths():
            # 1er intento de carga
            try:
                if loader_name == "cloudpickle":
                    with open(path, "rb") as f:
                        data = loader_pkg.load(f)
                else:
                    data = loader_pkg.load(path)
            except AttributeError as e:
                # Parche y reintento si es el caso de _RemainderColsList
                if "_RemainderColsList" in str(e):
                    _try_monkey_patch_sklearn_for_remainder_cols_list()
                    try:
                        if loader_name == "cloudpickle":
                            with open(path, "rb") as f:
                                data = loader_pkg.load(f)
                        else:
                            data = loader_pkg.load(path)
                    except Exception as e2:
                        last_err = f"{type(e2).__name__}: {e2}"
                        continue
                else:
                    last_err = f"{type(e).__name__}: {e}"
                    continue
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

            # Validación del contenido
            if not isinstance(data, dict) or not REQUIRED_KEYS.issubset(set(data.keys())):
                last_err = f"El archivo {path} no contiene las claves requeridas: {REQUIRED_KEYS}"
                continue

            model      = data["model"]
            classes    = data["classes"]
            vars_num   = data["vars_num"]
            vars_cat   = data["vars_cat"]
            target     = data.get("target", "Rendimiento")
            expected   = vars_cat + vars_num

            meta = {
                "path": path,
                "loader": loader_name,
                "target": target,
                "expected_cols": expected,
                # Si guardaste versiones del entorno al entrenar:
                "env": data.get("env", None)
            }
            return model, classes, vars_num, vars_cat, meta

        # Si no encontró nada válido:
        raise FileNotFoundError(
            last_err or "No se encontró un archivo .pkl válido. Coloca 'modelo_xgb.pkl' en esta carpeta o define MODEL_PKL."
        )

    # ============================================
    # 2) UI Streamlit
    # ============================================
    #st.title("📘 Predicción de Rendimiento (XGBoost)")

    # Carga automática al iniciar
    try:
        model, classes, vars_num, vars_cat, meta = load_artifacts_auto()
        #st.success(f"Modelo cargado automáticamente desde **{meta['path']}** usando **{meta['loader']}**.")
        #cols_txt = ", ".join(meta["expected_cols"])
        #t.caption(f"Target: **{meta['target']}** • Columnas esperadas: {cols_txt}")
        if meta.get("env"):
            st.caption("Entorno de entrenamiento: " + ", ".join([f"{k}={v}" for k, v in meta["env"].items()]))
    except Exception as e:
        st.error(f"No se pudo cargar el modelo automáticamente: {e}")
        st.stop()

    tabs = st.tabs(["🔹 Predicción Rendimiento", "📤 Predicción Masiva por CSV"])

    # ---------- Predicción manual ----------
    with tabs[0]:
        #st.subheader("Permite seleccionar el programa académico y la asignatura, e ingresar las cuatro calificaciones principales del estudiante.") 
        st.subheader("Ingresar datos del estudiante para predicción individual")
        col1, col2 = st.columns(2)
        
        # ---------- Carga y preparación (una sola vez) ----------
        @st.cache_data
        def cargar_mapa_programa_asignatura(ruta_txt: str):
            # El archivo viene con columnas: Programa, Asignatura (separadas por tabulador)
            df = pd.read_csv(ruta_txt, sep="\t", dtype=str, engine="python")

            # Limpiezas básicas
            df["Programa"] = df["Programa"].str.strip()
            df["Asignatura"] = df["Asignatura"].str.strip()
            df = df.dropna(subset=["Programa", "Asignatura"])
            df = df[df["Asignatura"].str.len() > 0]

            # Construir dict: {programa: [asignaturas únicas ordenadas]}
            mapa = (
                df.groupby("Programa")["Asignatura"]
                .apply(lambda s: sorted(set(s)))
                .to_dict()
            )
            programas = sorted(mapa.keys())
            return programas, mapa

        programas, mapa_prog_asig = cargar_mapa_programa_asignatura("Lista Programa Asignatura.txt")
        
        with col1:
            programa = st.selectbox("Programa", options=programas)
            nota1 = st.number_input("Nota 1", value=3.5, step=0.1, format="%.2f")
            nota3 = st.number_input("Nota 3", value=3.8, step=0.1, format="%.2f")

        with col2:
            asignaturas_disp = mapa_prog_asig.get(programa, [])
            asignatura = st.selectbox("Asignatura", options=asignaturas_disp)
            nota2 = st.number_input("Nota 2", value=4.0, step=0.1, format="%.2f")
            #nota4 = st.number_input("Nota 4", value=4.2, step=0.1, format="%.2f")

        if st.button("Predecir Rendimiento"):
            try:
                df_input = pd.DataFrame([{
                    "Programa": programa,
                    "Asignatura": asignatura,
                    "Nota 1": nota1,
                    "Nota 2": nota2,
                    "Nota 3": nota3
                }])

                # Alinear columnas y tipos
                df_input = df_input.reindex(columns=meta["expected_cols"])
                for c in vars_num:
                    df_input[c] = pd.to_numeric(df_input[c], errors="coerce")

                y_idx = model.predict(df_input)
                try:
                    y_prob = model.predict_proba(df_input)
                except Exception:
                    y_prob = None

                y_label = [classes[i] for i in y_idx][0]
                st.markdown(f"**Predicción:** {y_label}")

                if y_prob is not None:
                    prob_df = pd.DataFrame(y_prob, columns=classes)
                    st.markdown("**Probabilidades:**")
                    st.dataframe(prob_df.style.format("{:.4f}"))
            except Exception as e:
                st.error(f"Error en predicción: {e}")

    # ---------- Predicción por CSV ----------
    with tabs[1]:
        st.subheader("Cargar CSV para predicciones masivas")
        cols_txt = ", ".join(meta["expected_cols"])
        st.caption(f"Target: **{meta['target']}** • Columnas esperadas: {cols_txt}")
        st.caption(f"Se esperan exactamente estas columnas: {', '.join(meta['expected_cols'])}")
        file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if file is not None:
            try:
                df_csv = pd.read_csv(file)

                # Crear faltantes y ordenar columnas
                for c in meta["expected_cols"]:
                    if c not in df_csv.columns:
                        df_csv[c] = np.nan
                df_csv = df_csv.reindex(columns=meta["expected_cols"])
                for c in vars_num:
                    df_csv[c] = pd.to_numeric(df_csv[c], errors="coerce")

                y_idx = model.predict(df_csv)
                y_labels = [classes[i] for i in y_idx]

                try:
                    y_prob = model.predict_proba(df_csv)
                    prob_cols = [f"prob_{c}" for c in classes]
                    out = pd.concat(
                        [df_csv, pd.Series(y_labels, name="pred_clase"),
                        pd.DataFrame(y_prob, columns=prob_cols)],
                        axis=1
                    )
                except Exception:
                    out = pd.concat([df_csv, pd.Series(y_labels, name="pred_clase")], axis=1)

                st.success(f"Predicciones generadas: {len(out)} filas.")
                st.dataframe(out.head(50))

                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Descargar resultados CSV",
                    data=buf.getvalue(),
                    file_name="predicciones.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error procesando el CSV: {e}")

    st.markdown("""
---
                """, unsafe_allow_html=True)  
    
    st.markdown("""
                <div style='padding:8px 0; margin-bottom:8px;'>
                    <p style='color:#333333; font-size:18px; font-family: Tahoma, "Tahoma", Geneva, sans-serif; text-align:justify; text-justify:inter-word; line-height:1.6; margin:0;'>
                    Para optimizar su desempeño se aplicó un proceso de <strong>búsqueda en malla (GridSearchCV)</strong> y <strong>validación cruzada de dos particiones (2-Fold CV)</strong>, evaluando 16 combinaciones de hiperparámetros.  
                    Como resultado, el modelo alcanzó una <strong>exactitud global del 93%</strong> y un <strong>índice Kappa ponderado de 0.90</strong>, evidenciando una alta concordancia entre las categorías reales y las predicciones estimadas.  
                    </p>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""

                """, unsafe_allow_html=True)

    # Datos de la tabla
    data = {
        "Clase": [
            "Insuficiente",
            "Deficiente",
            "Bajo",
            "Medio",
            "Alto",
            "Superior",
            "accuracy",
            "macro avg",
            "weighted avg"
        ],
        "precision": [0.87, 0.77, 0.91, 0.94, 0.97, 0.95, None, 0.90, 0.93],
        "recall":    [0.79, 0.82, 0.91, 0.93, 0.96, 0.97, None, 0.90, 0.93],
        "f1-score":  [0.82, 0.80, 0.91, 0.94, 0.97, 0.96, None, 0.90, 0.93],
        "support":   [5121, 6389, 19752, 26722, 33270, 26829, 118083, 118083, 118083]
    }

    df_metrics = pd.DataFrame(data)

    # Crear dos columnas: una ancha (70%) y una más angosta (30%)
    col1, col2 = st.columns([2, 2])

    # --- Columna 1: tabla ---
    with col1:
        st.markdown("#### 📊 Métricas de Evaluación del Modelo")
        st.dataframe(
            df_metrics.style.format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:,.0f}"
            })
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}
            ]),
            use_container_width=False
        )

        # Mostrar el valor de Kappa
        st.markdown("Kappa ponderado (test): **0.9037**")

    # --- Columna 2: imagen ---
    with col2:
        try:
            image = Image.open("output.png")
            st.image(image, use_container_width=False)
        except FileNotFoundError:
            st.warning("⚠️ No se encontró la imagen 'output.png'. Verifica la ruta o el nombre del archivo.")



# Map routes to functions
ROUTES = {
    "Exploración de Datos (EDA)": page_eda,
    "Modelo Predicción": page_model,
}

ROUTES[choice]()



