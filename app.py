import base64
from pathlib import Path

import fitz  # PyMuPDF: convierte páginas de PDF a imágenes sin depender de software externo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Del artículo a la evidencia | Intento suicida",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores
RED = "#D7263D"
NAVY = "#0F172A"
BLUE = "#2563EB"
GREEN = "#059669"
ORANGE = "#D97706"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT = "#F8FAFC"
BORDER = "#E2E8F0"

# ============================================================
# FUNCIONES DE CARGA Y RENDERIZADO DE PDF
# ============================================================
@st.cache_resource
def get_pdf_doc():
    """Busca y abre el documento PDF."""
    base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    
    # Lista de nombres posibles
    possible_names = [
        "articulo.pdf",
        "intento suicidad.pdf",
        "intento suicida.pdf",
        "intento_suicidad.pdf",
        "intento_suicida.pdf",
    ]
    
    for name in possible_names:
        file_p = base_dir / name
        if file_p.exists():
            return fitz.open(str(file_p)), file_p
            
    # Búsqueda de cualquier .pdf en la carpeta si no coinciden los nombres
    pdf_files = list(base_dir.glob("*.pdf"))
    if pdf_files:
        return fitz.open(str(pdf_files[0])), pdf_files[0]
        
    return None, None

doc_pdf, path_pdf = get_pdf_doc()

def render_pdf_page_as_image(doc, page_number):
    """Convierte una página específica del PDF a imagen PNG alta resolución."""
    if doc is None:
        return None
    
    # Ajuste de índice (PyMuPDF usa base 0)
    page_idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc.load_page(page_idx)
    
    # Renderizar a 2x de zoom para nitidez (300 DPI aprox)
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    return pix.tobytes("png")

# ============================================================
# DATOS TRANSCRITOS DE PREVALENCIA
# ============================================================
age_groups = [
    "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29",
    "30 a 34", "35 a 39", "40 a 44", "45 a 49", "50 a 54",
    "55 a 59", "60 a 64", "65 a 69", "70 a 74", "75 a 79", "80 y más"
]

prevalence = {
    2012: [0.0, 5.8, 16.9, 16.9, 8.8, 6.0, 4.1, 3.4, 1.1, 0.5, 1.1, 0.6, 0.0, 0.0, 0.0, 0.0],
    2013: [0.0, 9.6, 23.5, 21.4, 16.3, 6.8, 5.9, 5.1, 3.1, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2014: [0.0, 0.4, 16.9, 15.9, 11.7, 10.8, 5.0, 0.9, 0.0, 0.7, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    2015: [0.0, 4.2, 16.2, 17.0, 8.9, 5.9, 3.0, 0.9, 1.5, 0.7, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    2016: [0.0, 9.4, 26.8, 15.2, 5.0, 2.0, 4.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    2017: [0.0, 4.3, 21.0, 19.1, 9.0, 8.1, 5.0, 3.8, 3.1, 0.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.7],
}

prev_rows = []
for year, values in prevalence.items():
    for age, value in zip(age_groups, values):
        prev_rows.append({"Año": year, "Grupo de edad": age, "Prevalencia": value})
prev_df = pd.DataFrame(prev_rows)

# ============================================================
# BARRA LATERAL (NAVEGACIÓN)
# ============================================================
with st.sidebar:
    st.markdown("## 📊 MODELOS LINEALES GENERALIZADOS")
    st.caption("UNIVERSIDAD DEL TOLIMA")
    st.markdown("---")
    
    section = st.radio(
        "Etapa de la exposición",
        [
            "01 · Introducción",
            "02 · Contexto y pregunta",
            "03 · Datos y diseño",
            "04 · Prevalencia",
            "05 · Tabla 1 · Descriptivos",
            "06 · Modelo logístico",
        ],
        label_visibility="collapsed",
    )

page_map = {
    "01 · Introducción": 1,
    "02 · Contexto y pregunta": 3,
    "03 · Datos y diseño": 5,
    "04 · Prevalencia": 6,
    "05 · Tabla 1 · Descriptivos": 7,
    "06 · Modelo logístico": 8,
}

current_page = page_map.get(section, 1)

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
st.markdown('<div style="font-size: 2rem; font-weight: 800; margin-bottom: 1rem;">Intento suicida: análisis municipal</div>', unsafe_allow_html=True)

left, right = st.columns([1.02, 1.18], gap="large")

# ------------------------------------------------------------
# COLUMNA IZQUIERDA: VISOR DE PDF
# ------------------------------------------------------------
with left:
    st.markdown("### 📄 Artículo original")
    st.caption(f"Visualizando Página {current_page} del PDF")
    
    if doc_pdf is not None:
        png_bytes = render_pdf_page_as_image(doc_pdf, current_page)
        if png_bytes:
            st.image(png_bytes, use_container_width=True)
        else:
            st.error("No se pudo procesar la página del PDF.")
            
        with open(path_pdf, "rb") as f:
            st.download_button(
                label="📥 Descargar PDF completo",
                data=f.read(),
                file_name=path_pdf.name,
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.error("⚠️ No se encontró el archivo PDF. Coloca `articulo.pdf` en la misma carpeta que `app.py`.")

# ------------------------------------------------------------
# COLUMNA DERECHA: EXPLICACIÓN Y GRÁFICOS
# ------------------------------------------------------------
with right:
    if section == "01 · Introducción":
        st.subheader("1. Entrar al estudio")
        st.write("Visión general sobre los factores asociados al intento de suicidio en Sogamoso, Boyacá (2012–2017).")
        
        st.info("💡 Utiliza el menú lateral para navegar por los hallazgos y revisar la página correspondiente del artículo a la izquierda.")
        
    elif section == "02 · Contexto y pregunta":
        st.subheader("2. Contexto y pregunta de investigación")
        st.write("Identificación de variables demográficas y socioeconómicas clave relacionadas con eventos de intento suicida.")

    elif section == "03 · Datos y diseño":
        st.subheader("3. Datos y diseño metodológico")
        st.write("Estudio observacional analítico basado en los registros del SIVIGILA.")

    elif section == "04 · Prevalencia":
        st.subheader("4. Prevalencia según edad y año")
        
        selected_year = st.selectbox("Seleccionar Año:", options=list(prevalence.keys()))
        filtered_df = prev_df[prev_df["Año"] == selected_year]
        
        fig = px.bar(
            filtered_df,
            x="Grupo de edad",
            y="Prevalencia",
            title=f"Prevalencia de Intento Suicida ({selected_year})",
            labels={"Prevalencia": "Prevalencia (%)", "Grupo de edad": "Grupo de Edad"},
            color_discrete_sequence=[BLUE]
        )
        st.plotly_chart(fig, use_container_width=True)

    elif section == "05 · Tabla 1 · Descriptivos":
        st.subheader("5. Características descriptivas (Tabla 1)")
        st.write("Resumen de las características socio-demográficas de los casos estudiados.")

    elif section == "06 · Modelo logístico":
        st.subheader("6. Modelo de Regresión Logística Binaria")
        st.write("Estimación de Razones de Momios (Odds Ratios - OR) para factores de riesgo identificados.")
        
        st.latex(r"\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k")
