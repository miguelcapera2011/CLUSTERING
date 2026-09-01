import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Del artículo a la evidencia | Intento suicida",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta sobria para exposición académica
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
# FUNCIONES DE CARGA Y VISUALIZACIÓN DE PDF
# ============================================================
def get_pdf_path():
    """Busca el archivo PDF en el directorio actual ignorando diferencias de mayúsculas/espacios."""
    base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    
    # Lista de nombres posibles
    possible_names = [
        "intento suicidad.pdf",
        "intento suicida.pdf",
        "intento_suicidad.pdf",
        "intento_suicida.pdf",
        "articulo.pdf",
        "paper.pdf"
    ]
    
    for name in possible_names:
        file_p = base_dir / name
        if file_p.exists():
            return file_p
            
    # Búsqueda flexible de cualquier .pdf en la carpeta si los nombres exactos fallan
    pdf_files = list(base_dir.glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]
        
    return None

def pdf_viewer_embed(bytes_data, page=1, height=820):
    """Genera el visor HTML embed asegurando compatibilidad con navegadores."""
    encoded = base64.b64encode(bytes_data).decode("utf-8")
    
    # Estructura del iframe compatible con objeto embebido de respaldo
    html_code = f"""
    <div style="width: 100%; height: {height}px; border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; background: white;">
        <object data="data:application/pdf;base64,{encoded}#page={page}&zoom=100" type="application/pdf" width="100%" height="100%">
            <embed src="data:application/pdf;base64,{encoded}#page={page}&zoom=100" type="application/pdf" width="100%" height="100%" />
            <p style="padding: 20px; text-align: center;">
                Tu navegador no soporta la visualización directa de PDFs en Base64.<br>
                <a href="data:application/pdf;base64,{encoded}" download="articulo.pdf" style="color: {BLUE}; font-weight: bold;">
                    Haz clic aquí para descargar el PDF.
                </a>
            </p>
        </object>
    </div>
    """
    components.html(html_code, height=height + 10, scrolling=False)

# Carga del PDF
pdf_path = get_pdf_path()
pdf_bytes = pdf_path.read_bytes() if pdf_path else None

# ============================================================
# DATOS TRANSCRITOS
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
# LAYOUT PRINCIPAL
# ============================================================
st.markdown('<div style="font-size: 2rem; font-weight: 800;">Intento suicida: análisis municipal</div>', unsafe_allow_html=True)

left, right = st.columns([1.02, 1.18], gap="large")

with left:
    st.markdown("### 📄 Artículo original")
    st.caption(f"Página mostrada: {current_page}")
    
    if pdf_bytes is not None:
        pdf_viewer_embed(pdf_bytes, page=current_page, height=800)
        st.download_button(
            label="📥 Descargar PDF completo",
            data=pdf_bytes,
            file_name=pdf_path.name if pdf_path else "documento.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.error(f"⚠️ No se encontró ningún archivo PDF en la carpeta del proyecto. Asegúrate de colocar el archivo `.pdf` junto a tu script `app.py`.")

with right:
    if section == "01 · Introducción":
        st.subheader("1. Entrar al estudio")
        st.write("Visión general sobre los factores asociados al intento de suicidio en Sogamoso, Boyacá (2012–2017).")
    elif section == "06 · Modelo logístico":
        st.subheader("6. Modelo Logístico")
        st.write("Especificación matemática de la regresión logística binaria.")
