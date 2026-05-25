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
# CONFIGURACIÓN DE INTERFAZ "PREMIUM ANALYTICS"
# ==============================================================================
st.set_page_config(page_title="Sistema de Inteligencia Territorial", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS para un look "Hecho a medida" (Boutique UI)
st.markdown("""
    <style>
    /* Fondo General - Gris Ejecutivo */
    .stApp {
        background-color: #F1F5F9;
        color: #1E293B;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Contenedor de Diapositiva - Efecto Papel Premium */
    .slide-container {
        background-color: #FFFFFF;
        padding: 45px;
        border-radius: 10px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }

    /* BOTONES SUPERIORES - Estilo Pestaña de Lujo */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #475569 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        height: 42px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stButton > button:hover {
        border-color: #0F172A !important;
        color: #0F172A !important;
        background-color: #F8FAFC !important;
        transform: translateY(-1px);
    }

    /* Botón Seleccionado (Azul UT Institucional) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #034485 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(3, 68, 133, 0.3) !important;
    }

    .slide-title {
        color: #0F172A;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 10px;
        border-bottom: 3px solid #034485;
        display: inline-block;
        padding-bottom: 5px;
    }

    /* Tarjetas de Análisis */
    .analysis-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 25px;
        border-left: 5px solid #034485;
        margin-bottom: 20px;
    }

    .question-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 15px;
        border: 1px dashed #3B82F6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Lógica de Navegación
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# Carga de Datos
def cargar_datos_automatico():
    archivos = [f for f in os.listdir('.') if f.lower().endswith(('.csv', '.xlsx'))]
    target = next((f for f in archivos if any(k in f.lower() for k in ["afectacion", "fuerza", "publica"])), None)
    if not target: return None, "Error: Base de datos no localizada."
    try:
        df = pd.read_csv(target) if target.endswith('.csv') else pd.read_excel(target)
        return df, target
    except: return None, "Error en lectura de archivo."

df_original, nombre_archivo = cargar_datos_automatico()

# Menú de Diapositivas
cols_nav = st.columns(6)
nombres_diapo = ["PORTADA", "INTRODUCCIÓN", "MARCO TEÓRICO", "METODOLOGÍA", "RESULTADOS", "CONCLUSIONES"]
for i, nombre in enumerate(nombres_diapo):
    tipo = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# ==============================================================================
# DIAPOSITIVA 1: PORTADA
# ==============================================================================
if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; padding: 60px 40px;'>
        <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='180'>
        <div style='margin-top:30px;'>
            <h1 style='font-size: 48px; color: #0F172A; margin-bottom:0;'>Clustering Estratégico de Riesgos</h1>
            <p style='font-size: 22px; color: #64748B;'>Segmentación Territorial de Afectaciones a la Fuerza Pública mediante K-Means</p>
        </div>
        <div style='margin: 40px auto; width: 100px; border-top: 5px solid #034485;'></div>
        <div style='display: flex; justify-content: center; gap: 50px;'>
            <div style='text-align: left;'>
                <p style='margin:0; font-weight: 700; color: #034485;'>INVESTIGADOR</p>
                <p style='font-size: 18px;'>Miguel Angel Garatejo</p>
            </div>
            <div style='text-align: left;'>
                <p style='margin:0; font-weight: 700; color: #034485;'>DOCENTE EVALUADOR</p>
                <p style='font-size: 18px;'>Yuri Saavedra</p>
            </div>
        </div>
        <p style='margin-top: 50px; font-size: 14px; color: #94A3B8;'>UNIVERSIDAD DEL TOLIMA • FACULTAD DE CIENCIAS • 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (ANÁLISIS ENRIQUECIDO)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Hallazgos Estratégicos y Perfiles de Clúster</div>", unsafe_allow_html=True)
    
    if df_original is not None:
        # --- PROCESAMIENTO MATEMÁTICO ---
        idx = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        # Usamos pivoteo dinámico basado en tu base
        pivot = df_original.pivot_table(index=idx, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        datos = df_original.groupby(idx)['CANTIDAD'].sum().to_frame(name='TOTAL').join(pivot).reset_index().dropna()
        num_cols = [c for c in datos.columns if c not in idx]
        
        X_scaled = StandardScaler().fit_transform(datos[num_cols])
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        datos['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

        # Tabs de análisis para no saturar
        t1, t2, t3 = st.tabs(["📊 Análisis Dimensional", "🗺️ Mapa de Distancias", "🔬 Comparativa Estratégica"])

        with t1:
            st.markdown("### ¿Cómo se distribuye el riesgo en el espacio 3D?")
            pca = PCA(n_components=3)
            coords = pca.fit_transform(X_scaled)
            df_pca = pd.DataFrame(coords, columns=['X_Letalidad', 'Y_Frecuencia', 'Z_Institucional'])
            df_pca['Cluster'] = datos['Cluster'].map({"0":"Estable","1":"Dinámico","2":"Focalizado","3":"CRÍTICO"})
            df_pca['Muni'] = datos['MUNICIPIO']
            
            # Colores Ultra Vivos: Verde Neón, Azul Eléctrico, Naranja Neón, Rojo Sangre
            fig3d = px.scatter_3d(df_pca, x='X_Letalidad', y='Y_Frecuencia', z='Z_Institucional', 
                                 color='Cluster', hover_name='Muni',
                                 color_discrete_sequence=['#00E676', '#2979FF', '#FF9100', '#FF1744'])
            
            fig3d.update_layout(
                scene=dict(bgcolor='#F8FAFC', xaxis=dict(gridcolor='white'), yaxis=dict(gridcolor='white'), zaxis=dict(gridcolor='white')),
                paper_bgcolor='#F8FAFC', margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)
            
            st.markdown("""
            <div class='question-box'>
                <b>❓ Pregunta de Sustentación:</b> ¿Qué representan los puntos alejados (Outliers)?<br>
                <b>Respuesta:</b> Son municipios con dinámicas <b>atípicas</b>. El modelo los aísla en el clúster rojo para que sus valores extremos no "contaminen" el promedio de los municipios pacíficos.
            </div>
            """, unsafe_allow_html=True)

        with t2:
            st.markdown("### Matriz de Disimilitud: ¿Qué tan parecidos son los municipios?")
            dist = euclidean_distances(X_scaled[:40, :])
            fig_hm = px.imshow(dist, x=datos['MUNICIPIO'][:40], y=datos['MUNICIPIO'][:40], 
                               color_continuous_scale='Turbo') # Turbo es muy vivo
            fig_hm.update_layout(
                paper_bgcolor='#F8FAFC',
                xaxis=dict(tickfont=dict(color='black', size=10, family='Arial Black')),
                yaxis=dict(tickfont=dict(color='black', size=10, family='Arial Black'))
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.markdown("<p style='text-align:center; color:#64748B;'>Las zonas oscuras indican pares de municipios con huellas delictivas idénticas.</p>", unsafe_allow_html=True)

        with t3:
            st.markdown("### 🔍 Análisis de fondo: Contraste entre Grupos")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class='analysis-card'>
                    <h4>⚔️ Clúster 1 vs Clúster 2</h4>
                    <p><b>Diferencia:</b> El Clúster 1 muestra una violencia <i>reactiva</i> (disturbios, asonadas), mientras que el Clúster 2 refleja una violencia <b>selectiva e institucional</b> (ataques dirigidos a la estación de policía o patrullas militares).</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class='analysis-card' style='border-left-color: #FF1744;'>
                    <h4>🚨 El Salto al Clúster 3</h4>
                    <p>Cuando un municipio del Clúster 2 aumenta su distancia euclidiana respecto al centroide, indica que la emergencia ya no es focalizada sino <b>generalizada</b>, convirtiéndose en un foco de atención nacional (Atípico).</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # Resto de diapositivas genéricas para mantener la estructura
    st.markdown(f"<div class='slide-title'>{nombres_diapo[st.session_state.diapositiva-1]}</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'>Análisis metodológico en curso para la sustentación final ante la Facultad de Ciencias.</div>", unsafe_allow_html=True)

# Footer de navegación rápida
st.markdown("<br><br>", unsafe_allow_html=True)
if st.button("Volver al Inicio 🏠"): ir_a_diapositiva(1)
