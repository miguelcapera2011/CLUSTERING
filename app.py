# =============================================================================
# APP PRINCIPAL - app.py
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title="Minería de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS GLOBAL PREMIUM
# =============================================================================

st.markdown("""
<style>

/* ======================================================
FONDO GENERAL
====================================================== */
.stApp{
    background: linear-gradient(135deg,#06141f,#0b1120,#071c2f);
    color:white;
}

/* ======================================================
OCULTAR MENU STREAMLIT
====================================================== */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ======================================================
HERO
====================================================== */
.hero{
    padding:80px 40px;
    border-radius:30px;
    background: linear-gradient(135deg,#071b2d,#102b46);
    border:1px solid rgba(255,255,255,0.08);
    text-align:center;
    box-shadow:0px 10px 40px rgba(0,0,0,0.4);
}

.hero-title{
    font-size:68px;
    font-weight:900;
    color:#00ffd5;
    line-height:1;
}

.hero-sub{
    font-size:24px;
    color:#9ed8ff;
    margin-top:20px;
}

/* ======================================================
CARDS
====================================================== */
.card{
    background: rgba(255,255,255,0.04);
    border-radius:25px;
    padding:30px;
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    box-shadow:0px 4px 20px rgba(0,0,0,0.4);
    transition:0.4s;
}

.card:hover{
    transform:translateY(-8px);
    box-shadow:0px 10px 30px rgba(0,255,213,0.2);
}

/* ======================================================
TITULOS
====================================================== */
h1,h2,h3{
    color:#7ef9ff;
}

/* ======================================================
BOTONES
====================================================== */
.stButton>button{
    background: linear-gradient(90deg,#00ffd5,#00a8ff);
    color:black;
    border:none;
    border-radius:15px;
    padding:0.8rem 1.5rem;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    color:white;
}

/* ======================================================
METRICAS
====================================================== */
.metric-card{
    background: linear-gradient(135deg,#081726,#0e2f4d);
    padding:25px;
    border-radius:20px;
    text-align:center;
    border:1px solid rgba(0,255,213,0.1);
}

.metric-title{
    font-size:16px;
    color:#89a9c0;
}

.metric-value{
    font-size:40px;
    font-weight:bold;
    color:#00ffd5;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        MINERÍA<br>DE DATOS
    </div>

    <div class="hero-sub">
        Clusterización de Municipios Colombianos mediante K-Means
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# =============================================================================
# INTRO
# =============================================================================

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Registros Originales</div>
        <div class="metric-value">17.553</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Variables Categóricas</div>
        <div class="metric-value">8</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Municipios Finales</div>
        <div class="metric-value">884</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# =============================================================================
# EXPLICACION
# =============================================================================

st.markdown("""
<div class="card">

# 🎯 Objetivo del Proyecto

El objetivo principal fue identificar patrones similares entre municipios colombianos afectados por eventos relacionados con la fuerza pública utilizando técnicas de minería de datos y aprendizaje no supervisado.

---

# 🚨 Problema Inicial

La base de datos original NO era apta para aplicar directamente K-Means.

¿Por qué?

- Solo existía 1 variable numérica.
- Existían 8 variables categóricas.
- K-Means funciona mediante distancias matemáticas.
- Las variables categóricas no pueden usarse directamente.

---

# 💡 Solución

Transformar las variables categóricas en información numérica útil mediante consolidación y conteos estadísticos.

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# FLUJO VISUAL
# =============================================================================

st.markdown("""
<div class="card">

# 🔄 Transformación de Datos

### Flujo metodológico

BASE ORIGINAL  
⬇  
17.553 registros  
⬇  
Variables categóricas  
⬇  
pivot_table()  
⬇  
Conversión a conteos numéricos  
⬇  
Consolidación por municipio  
⬇  
884 municipios finales  
⬇  
Estandarización  
⬇  
Aplicación de K-Means  
⬇  
Visualización PCA

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# BOTONES
# =============================================================================

c1,c2 = st.columns(2)

with c1:
    if st.button("📘 Ver Infografía Completa"):
        st.switch_page("pages/1_📘_Infografia.py")

with c2:
    if st.button("📊 Ir al Análisis Interactivo"):
        st.switch_page("pages/2_📊_Analisis.py")
