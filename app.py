import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM - VERSIÓN INSTITUCIONAL"
# ==============================================================================
st.set_page_config(page_title="Exposición Avanzada - Orden Público", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS para Colores Institucionales y Modernos
st.markdown("""
    <style>
    /* Fondo principal claro y limpio */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Contenedor de la diapositiva en Blanco Puro */
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }

    /* Títulos estilo Consultoría (Verde Policía) */
    .slide-title {
        color: #1B3921; /* Verde Policía Nacional */
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .slide-subtitle {
        color: #64748B;
        font-size: 19px;
        margin-bottom: 25px;
        font-weight: 400;
    }

    /* BOTONES: Corrección de color oscuro a claro */
    div.stButton > button {
        background-color: #F1F5F9 !important; /* Gris muy claro para no seleccionados */
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        border-color: #8F9779 !important; /* Verde Ejército */
        color: #1B3921 !important;
        background-color: #F8FAF0 !important;
    }

    /* Botón de la Diapositiva Actual (Seleccionado) */
    .st-emotion-cache-7ym5gk {
        background-color: #1B3921 !important; /* Verde Policía */
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(27, 57, 33, 0.3);
    }

    /* Tarjetas de Insights con colores institucionales */
    .insight-card {
        background-color: #F8FAF0; /* Fondo crema muy suave */
        border-left: 6px solid #8F9779; /* Verde Ejército */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .insight-critical {
        background-color: #FFF1F2;
        border-left: 6px solid #E11D48; /* Rojo alerta */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .insight-success {
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A; /* Verde éxito */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización del paginador (diapositivas)
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# ==============================================================================
# LÓGICA DE CARGA DE DATOS (SIN TOCAR ESTRUCTURA)
# ==============================================================================
def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
            
    if archivo_encontrado is None:
        return None, "No se encontró el registro de datos."
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error: {str(e)}"

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# ==============================================================================
# NAVEGACIÓN SUPERIOR CON COLORES CORREGIDOS
# ==============================================================================
cols_nav = st.columns(6)
nombres_diapo = ["🏠 Portada", "🎯 Introducción", "📖 Teoría", "⚙️ Método", "📊 Resultados", "🏁 Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 1: PORTADA OFICIAL
# ==============================================================================
if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; border-top: 10px solid #1B3921;'>
        <img src='https://www.ut.edu.co/images/logos/logo_ut.png' width='160' style='margin-bottom: 10px;'>
        <h1 style='color: #1B3921; font-size: 45px; margin-bottom:0;'>Análisis de Novedades del Orden Público</h1>
        <h3 style='color: #64748B; font-weight: 400; margin-top:0;'>Segmentación de Vulnerabilidad mediante Machine Learning</h3>
        <hr style='margin: 30px auto; width: 60%; opacity: 0.2;'>
        <p style='font-size: 20px;'><b>Expositor:</b> Miguel Angel Garatejo</p>
        <p style='font-size: 18px; color: #8F9779;'><b>Evaluador:</b> Yuri Saavedra</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Iniciar Presentación ➡️", type="primary"):
        ir_a_diapositiva(2)

# ==============================================================================
# DIAPOSITIVA 2: INTRODUCCIÓN
# ==============================================================================
elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>🎯 El Reto de la Información</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-subtitle'>De registros planos a inteligencia territorial</div>", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""<div class='insight-critical'>
            <h3 style='margin-top:0; color:#991B1B;'>🛑 Limitación Técnica</h3>
            <p>El histórico oficial presenta una restricción: es <b>90% texto</b>. Los modelos de inteligencia artificial necesitan números y distancias para agrupar municipios, no palabras.</p>
        </div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""<div class='insight-card'>
            <h3 style='margin-top:0; color:#1B3921;'>💡 Solución Propuesta</h3>
            <p>Transformar el registro en una matriz de frecuencias para medir qué tan similares son los ataques entre municipios y priorizar la atención.</p>
        </div>""", unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 3: MARCO TEÓRICO
# ==============================================================================
elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>📖 Sustento Matemático</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='slide-container' style='height:250px;'><h4>1. Pivotado</h4><p>Convertimos categorías en columnas numéricas.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='slide-container' style='height:250px;'><h4>2. K-Means</h4><p>Encontramos los grupos por cercanía geográfica y táctica.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='slide-container' style='height:250px;'><h4>3. PCA</h4><p>Reducimos dimensiones para ver el mapa del riesgo en 3D.</p></div>", unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (Aquí se aplicó el cambio de fondo de gráficas)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>📊 Resultados del Agrupamiento</div>", unsafe_allow_html=True)
    
    # --- PROCESAMIENTO MATEMÁTICO (NO SE TOCA LÓGICA) ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos = total_municipio.join([pivot_accion]).reset_index().dropna()
    
    scaler = StandardScaler()
    numericas = [col for col in datos.columns if col not in index_cols]
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=index_cols)
    
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)

    # --- GRÁFICAS CON FONDO CL
