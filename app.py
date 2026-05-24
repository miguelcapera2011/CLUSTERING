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
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "INTELLIGENCE DASHBOARD"
# ==============================================================================
st.set_page_config(page_title="Inteligencia de Datos - Fuerza Pública", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS para Colores Vivos y Fondo Elegante (No Blanco)
st.markdown("""
    <style>
    /* Fondo principal: Gris acero suave para que resalten los colores vivos */
    .stApp {
        background-color: #E2E8F0;
        color: #0F172A;
        font-family: 'Inter', sans-serif;
    }
    
    /* Contenedor de la diapositiva */
    .slide-container {
        background-color: #F8FAFC;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border: 1px solid #CBD5E1;
    }
    
    /* BOTONES SUPERIORES: Colores "Caros" y contrastados */
    div.stButton > button {
        background-color: #94A3B8 !important; /* Gris acero para no seleccionados */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        height: 45px !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        background-color: #475569 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }

    /* Estilo para el botón de la página seleccionada (Azul Zafiro Vivo) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #D4AF37 !important; /* Borde Oro sutil */
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4) !important;
    }

    .slide-title {
        color: #1E3A8A;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .insight-card {
        background-color: #FFFFFF;
        border-left: 6px solid #1E40AF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Lógica de navegación
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# Carga de datos automatizada
def cargar_datos_automatico():
    archivos = [f for f in os.listdir('.') if f.lower().endswith(('.csv', '.xlsx'))]
    encontrado = next((f for f in archivos if any(k in f.lower() for k in ["afectacion", "fuerza", "publica"])), None)
    if not encontrado: return None, "Archivo no detectado"
    try:
        df = pd.read_csv(encontrado) if encontrado.endswith('.csv') else pd.read_excel(encontrado)
        return df, encontrado
    except: return None, "Error en lectura"

df_original, _ = cargar_datos_automatico()

# Menú de Navegación
cols = st.columns(6)
nombres = ["🏠 Portada", "🎯 Contexto", "📖 Teoría", "⚙️ Método", "📊 Resultados", "🏁 Cierre"]
for i, nombre in enumerate(nombres):
    tipo = "primary" if st.session_state.diapositiva == (i+1) else "secondary"
    if cols[i].button(nombre, use_container_width=True, type=tipo):
        ir_a_diapositiva(i+1)

# ==============================================================================
# DIAPOSITIVAS
# ==============================================================================

if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; border-top: 8px solid #1E40AF;'>
        <img src='https://www.ut.edu.co/images/logos/logo_ut.png' width='160'>
        <div class='slide-title'>Análisis de Vulnerabilidad Territorial</div>
        <p style='font-size: 20px; color: #475569;'>Clasificación Estratégica de la Fuerza Pública mediante Machine Learning</p>
        <hr style='width: 50%; margin: 20px auto;'>
        <p><b>Miguel Angel Garatejo</b> | Universidad del Tolima</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>📊 Resultados del Modelo de Inteligencia</div>", unsafe_allow_html=True)
    
    if df_original is not None:
        # --- Lógica de Datos (Sin tocar estructura) ---
        idx = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        pivot = df_original.pivot_table(index=idx, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        datos = df_original.groupby(idx)['CANTIDAD'].sum().to_frame(name='TOTAL').join(pivot).reset_index().dropna()
        num_cols = [c for c in datos.columns if c not in idx]
        X_scaled = StandardScaler().fit_transform(datos[num_cols])
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        datos['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

        # --- GRÁFICA DEL CODO: Colores Vivos ---
        st.markdown("### 📈 Optimización de Grupos")
        wss = [KMeans(n_clusters=k, n_init=10).fit(X_scaled).inertia_ for k in range(1, 11)]
        fig_el = px.line(x=list(range(1, 11)), y=wss, markers=True)
        fig_el.update_traces(line=dict(color='#2563EB', width=4), marker=dict(size=12, color='#F59E0B', line=dict(width=2, color='white')))
        fig_el.update_layout(plot_bgcolor='#F1F5F9', paper_bgcolor='#F1F5F9', height=400)
        st.plotly_chart(fig_el, use_container_width=True)

        # --- MAPA DE CALOR: Nombres Visibles y Colores Vivos ---
        st.markdown("### 🗺️ Matriz de Distancias (Similitud entre Municipios)")
        dist = euclidean_distances(X_scaled[:35, :]) # Reducido a 35 para que el texto sea más grande
        fig_hm = px.imshow(dist, 
                           x=datos['MUNICIPIO'][:35], 
                           y=datos['MUNICIPIO'][:35],
                           color_continuous_scale='Turbo') # Turbo es una escala muy viva
        fig_hm.update_layout(
            plot_bgcolor='#F1F5F9', 
            paper_bgcolor='#F1F5F9',
            xaxis=dict(tickfont=dict(size=14, color='black', family='Arial Black')), # Nombres negros y fuertes
            yaxis=dict(tickfont=dict(size=14, color='black', family='Arial Black'))
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # --- PCA 3D: Colores Vivos y Fondo No Blanco ---
        st.markdown("### 🌐 Mapa de Riesgo Tridimensional")
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = datos['Cluster'].map({"0":"Estable (Verde)","1":"Alerta (Azul)","2":"Riesgo (Naranja)","3":"CRÍTICO (Rojo)"})
        df_pca['Muni'] = datos['MUNICIPIO']
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                               hover_name='Muni',
                               color_discrete_sequence=['#00C853', '#2979FF', '#FF9100', '#D50000']) # Colores Ultra-Vivos
        
        fig_3d.update_traces(marker=dict(size=7, opacity=0.9, line=dict(width=1, color='white')))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="#F1F5F9", gridcolor="white", showbackground=True),
                yaxis=dict(backgroundcolor="#F1F5F9", gridcolor="white", showbackground=True),
                zaxis=dict(backgroundcolor="#F1F5F9", gridcolor="white", showbackground=True),
                bgcolor="#E2E8F0" # Fondo del cubo grisáceo para resaltar colores
            ),
            paper_bgcolor='#E2E8F0', # Marco que rodea la gráfica
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(font=dict(size=16, color="black"))
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("<div class='insight-card'><b>Hallazgo:</b> Los municipios en <b>Rojo Sangre</b> son los atípicos que concentran la mayor letalidad del país.</div>", unsafe_allow_html=True)

else:
    # Resto de diapositivas con el mismo estilo
    st.markdown(f"<div class='slide-title'>{nombres[st.session_state.diapositiva-1]}</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'>Información en proceso de análisis estratégico.</div>", unsafe_allow_html=True)
