import html
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# CONFIGURACIÓN
st.set_page_config(
    page_title="Modelos lineales generalizados | Universidad del Tolima",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

RED = "#D7263D"
NAVY = "#0F172A"
BLUE = "#2563EB"
GREEN = "#059669"
ORANGE = "#D97706"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT = "#F8FAFC"
BORDER = "#E2E8F0"

# ESTILOS
st.markdown(
    """
    <style>
        .stApp {
            background: #F8FAFC;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #172554 100%);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span:not([data-baseweb]),
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] div[role="listbox"] * {
            color: #F8FAFC !important;
        }
        /* Solo los controles del contenido principal van en negro */
        [data-testid="stAppViewContainer"] .stSelectbox label,
        [data-testid="stAppViewContainer"] .stRadio label,
        [data-testid="stAppViewContainer"] [data-baseweb="select"] *,
        [data-testid="stAppViewContainer"] div[role="listbox"] *,
        [data-testid="stAppViewContainer"] .control-label {
            color: #000000 !important;
        }
        /* Texto de controles del área principal: negro y visible */
        [data-testid="stAppViewContainer"] .stSelectbox label,
        [data-testid="stAppViewContainer"] .stSelectbox div,
        [data-testid="stAppViewContainer"] .stRadio label,
        [data-testid="stAppViewContainer"] .stRadio div,
        [data-testid="stAppViewContainer"] [data-baseweb="select"],
        [data-testid="stAppViewContainer"] [data-baseweb="select"] span {
            color: #000000 !important;
        }
        .control-label {
            font-weight: 700;
            font-size: .95rem;
            margin: .35rem 0;
            color: #000000 !important;
        }
        .math-visible {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: .7rem 0;
            color: #000000 !important;
            font-size: 1.08rem;
            line-height: 1.8;
            overflow-x: auto;
        }
        .math-visible * { color: #000000 !important; }
        .formula-title { font-weight: 800; color: #000000 !important; margin-bottom: .35rem; }
        .ut-header { text-align: center; padding: .35rem .2rem 1rem .2rem; }
        .ut-logo-fallback {
            width: 76px; height: 76px; margin: 0 auto .6rem auto;
            border: 3px solid #FFFFFF; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.55rem; font-weight: 900; color: #FFFFFF;
            background: #7A0C14; box-shadow: 0 3px 12px rgba(0,0,0,.25);
        }
        .ut-name { color: #FFFFFF !important; font-weight: 800; font-size: 1.05rem; line-height: 1.2; text-align: center; }
        .ut-program { color: #E2E8F0 !important; font-size: .78rem; text-align: center; margin-top: .25rem; }
        /* Fuerza la barra lateral a conservar el color claro */
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stRadio label span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #F8FAFC !important;
        }
        .main-university-header { text-align: center; margin: .2rem 0 1.1rem 0; }
        .main-university-header .title { color: #0F172A; font-size: 1.85rem; font-weight: 900; margin: 0; }
        .main-university-header .university { color: #7A0C14; font-size: 1rem; font-weight: 800; letter-spacing: .04em; margin-top: .25rem; }
        .main-title {
            font-size: 2.0rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.1rem;
        }
        .subtitle {
            color: #64748B;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 0.2rem;
        }
        .section-subtitle {
            color: #64748B;
            margin-bottom: 0.8rem;
        }
        .card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1rem 1.15rem;
            box-shadow: 0 2px 10px rgba(15,23,42,.05);
            height: 100%;
            color: #000000 !important;
        }
        .card h4 {
            margin: 0 0 .45rem 0;
            color: #0F172A !important;
        }
        .card p, .card div, .card li {
            color: #000000 !important;
        }
        .metric {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0F172A;
        }
        .metric-label {
            font-size: .85rem;
            color: #64748B;
        }
        .interpretation {
            background: #FFF7ED;
            border-left: 5px solid #D97706;
            padding: .9rem 1rem;
            border-radius: 8px;
            margin: .7rem 0;
            color: #000000 !important;
        }
        .math-box {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 12px;
            padding: 1rem;
            color: #000000 !important;
        }
        .source-box {
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: .75rem 1rem;
            font-size: .82rem;
            color: #000000 !important;
        }
        .step {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: .9rem;
            text-align: center;
            min-height: 120px;
            color: #000000 !important;
        }
        .step p {
            color: #000000 !important;
        }
        .step-number {
            font-size: .8rem;
            font-weight: 700;
            color: #D7263D;
        }
        .step-title {
            font-weight: 800;
            color: #0F172A;
        }
        .small-note {
            color: #000000;
            font-size: .82rem;
        }
        .big-question {
            font-size: 1.25rem;
            font-weight: 750;
            color: #0F172A;
            background: white;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            padding: 1rem 1.2rem;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #E2E8F0;
            padding: .7rem;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] * {
            color: #000000 !important;
        }
        .footer {
            color: #64748B;
            text-align: center;
            font-size: .78rem;
            margin-top: 1.5rem;
        }
        .pdf-frame {
            width: 100%;
            height: 820px;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            background: white;
        }
        /* Estilos de alto contraste para destacar cajas y títulos específicos */
        .highlight-card {
            background: #0F172A !important;
            border: 2px solid #F59E0B !important;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
            color: #FFD700 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .highlight-card * {
            color: #FFD700 !important;
        }
        .highlight-header {
            background: #0F172A !important;
            color: #FFD700 !important;
            border: 2px solid #F59E0B !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 800;
            font-size: 1.2rem;
            display: inline-block;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# DATOS TRANSCRITOS DEL ARTÍCULO
age_groups = [
    "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29",
    "30 a 34", "35 a 39", "40 a 44", "45 a 49", "50 a 54",
    "55 a 59", "60 a 64", "65 a 69", "70 a 74", "75 a 79", "80 y más",
]
prevalence = {
    2012: [0.0, 5.8, 16.9, 16.9, 8.8, 6.0, 4.1, 3.4, 1.1, 0.5, 1.1, 0.6, 0.0, 0.0, 0.0, 0.0],
    2013: [0.0, 9.6, 23.5, 21.4, 16.3, 6.8, 5.9, 5.1, 3.1, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2014: [0.0, 0.4, 16.9, 15.9, 11.7, 10.8, 5.0, 0.9, 0.0, 0.7, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    2015: [0.0, 4.2, 16.2, 17.0, 8.9, 5.9, 3.0, 0.9, 1.5, 0.7, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    2016: [0.0, 9.4, 26.8, 15.2, 5.0, 2.0, 4.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    2017: [0.0, 4.3, 21.0, 19.1, 9.0, 8.1, 5.0, 3.8, 3.1, 0.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.7],
}
prev_rows = [
    {"Año": year, "Grupo de edad": age, "Prevalencia": value}
    for year, values in prevalence.items()
    for age, value in zip(age_groups, values)
]
prev_df = pd.DataFrame(prev_rows)

table1 = [
    ["Área de ocurrencia", "Urbano", 94.0, 315, 89.8, 169, 92.0, 484, 0.02],
    ["Área de ocurrencia", "Rural disperso", 6.2, 21, 10.1, 19, 8.0, 40, 0.02],
    ["Edad agrupada", "3 a 11 años (niñez)", 0.9, 3, 1.6, 3, 1.1, 6, 0.000],
    ["Edad agrupada", "11 a 20 años (adolescencia)", 50.3, 169, 35.1, 66, 44.8, 235, 0.000],
    ["Edad agrupada", "20 a 40 (adultez temprana)", 42.0, 141, 45.7, 86, 43.3, 227, 0.000],
    ["Edad agrupada", "40 a 65 (adultez mediana y tardía)", 6.8, 23, 17.6, 33, 10.7, 56, 0.000],
    ["Estado civil", "Soltero", 64.8, 210, 54.4, 99, 61.7, 309, 0.115],
    ["Estado civil", "Casado", 31.2, 101, 3.9, 71, 34.0, 172, 0.115],
    ["Estado civil", "Separado", 3.1, 10, 5.5, 10, 4.0, 20, 0.115],
    ["Estado civil", "Viudo", 0.9, 3, 1.1, 2, 1.0, 5, 0.115],
    ["Ocupación", "Ama de casa", 30.2, 101, 1.1, 2, 20.2, 103, 0.000],
    ["Ocupación", "Estudiante", 47.6, 159, 35.4, 62, 43.4, 221, 0.000],
    ["Ocupación", "Empleado auxiliar", 3.89, 13, 2.6, 47, 11.9, 60, 0.000],
    ["Ocupación", "Empleado profesional", 2.9, 10, 5.7, 10, 3.9, 20, 0.000],
    ["Ocupación", "Independiente", 8.9, 30, 14.2, 25, 10.8, 55, 0.000],
    ["Ocupación", "Población carcelaria", 0.2, 1, 0.0, 0, 0.2, 1, 0.000],
    ["Ocupación", "Desempleado", 3.3, 11, 12.6, 22, 6.5, 33, 0.000],
    ["Ocupación", "Pensionado", 0.2, 1, 1.14, 2, 0.6, 3, 0.000],
    ["Forma de realización", "Impulsiva", 84.8, 279, 82.8, 149, 84.1, 428, 0.551],
    ["Forma de realización", "Planeada", 15.2, 50, 17.2, 31, 15.9, 81, 0.551],
    ["Antecedentes de intento", "No", 68.5, 24, 67.4, 120, 68.1, 44, 0.80],
    ["Antecedentes de intento", "Sí", 31.5, 103, 32.6, 58, 31.9, 161, 0.80],
    ["Método del intento", "Medicamentos", 54.4, 182, 40.0, 70, 49.2, 252, 0.000],
    ["Método del intento", "Plaguicidas", 21.8, 73, 29.1, 51, 24.2, 124, 0.000],
    ["Método del intento", "Sustancias psicoactivas (SPA)", 0.2, 1, 2.28, 4, 1.0, 5, 0.000],
    ["Método del intento", "Heridas", 18.5, 61, 17.9, 33, 18.3, 94, 0.000],
    ["Método del intento", "Otros métodos", 3.59, 12, 13.7, 24, 7.1, 36, 0.000],
    ["Método del intento", "Arma de fuego", 0.0, 0, 1.1, 2, 0.4, 2, 0.000],
    ["Posible desencadenante", "Conflicto con la pareja", 36.9, 118, 29.8, 54, 34.3, 172, 0.005],
    ["Posible desencadenante", "Conflicto familiar", 27.2, 87, 18.2, 33, 24.0, 120, 0.005],
    ["Posible desencadenante", "Indeterminado", 20.9, 67, 30.4, 55, 24.4, 122, 0.005],
    ["Posible desencadenante", "Consumo de alcohol", 9.1, 29, 12.7, 23, 10.4, 52, 0.005],
    ["Posible desencadenante", "Conflicto laboral o escolar", 3.4, 11, 2.2, 4, 3.0, 15, 0.005],
    ["Posible desencadenante", "Problemas económicos", 2.5, 8, 12.7, 12, 4.0, 20, 0.005],
    ["Enfermedad mental", "No", 65.8, 210, 67.6, 115, 66.5, 325, 0.68],
    ["Enfermedad mental", "Sí", 34.2, 109, 32.4, 55, 33.5, 164, 0.68],
    ["Violencia", "No", 46.0, 137, 62.5, 95, 51.6, 232, 0.001],
    ["Violencia", "Sí", 54.0, 161, 37.5, 57, 48.4, 218, 0.001],
    ["Consumo de alcohol", "No", 59.7, 181, 38.5, 65, 52.1, 246, 0.000],
    ["Consumo de alcohol", "Sí", 40.3, 122, 59.7, 104, 47.9, 226, 0.000],
    ["Relaciones familiares", "Disfuncionales", 79.7, 248, 78.2, 129, 79.2, 377, 0.69],
    ["Relaciones familiares", "Funcionales", 20.3, 63, 21.8, 36, 20.8, 99, 0.69],
    ["Redes de apoyo", "No", 9.2, 28, 12.7, 22, 10.4, 49, 0.23],
    ["Redes de apoyo", "Sí", 90.8, 278, 87.3, 145, 89.6, 423, 0.23],
]
table1_df = pd.DataFrame(
    table1,
    columns=[
        "Variable", "Categoría",
        "Mujer %", "Mujer N",
        "Hombre %", "Hombre N",
        "Total %", "Total N", "p",
    ],
)

table2 = [
    ["Edad", "Niñez", 0.32, 0.05, 0.83, 1.38, 0.08, 24.34],
    ["Edad", "Adolescencia", -1.04, 7.52, 0.01, 0.35, 0.17, 0.74],
    ["Edad", "Adultez temprana", -0.81, 4.81, 0.03, 0.45, 0.22, 0.92],
    ["Posible desencadenante", "Conflicto con la pareja", -0.84, 7.67, 0.01, 0.43, 0.24, 0.78],
    ["Posible desencadenante", "Conflicto familiar", -0.66, 3.81, 0.05, 0.52, 0.26, 1.00],
    ["Posible desencadenante", "Conflicto laboral o escolar", -0.70, 1.09, 0.30, 0.50, 0.13, 1.85],
    ["Posible desencadenante", "Problemas económicos", 0.74, 1.43, 0.23, 2.09, 0.62, 7.02],
    ["Posible desencadenante", "Consumo de alcohol", -0.81, 3.89, 0.05, 0.44, 0.20, 1.00],
    ["Posible desencadenante", "Violencia", -0.89, 13.80, 0.00, 0.41, 0.26, 0.66],
    ["Consumo de alcohol", "Sí", 1.28, 25.09, 0.00, 3.58, 2.17, 5.90],
    ["Constante", "Intercepto", 0.44, 1.33, 0.25, 1.55, None, None],
]
table2_df = pd.DataFrame(
    table2,
    columns=["Variable", "Categoría", "B", "Wald", "p", "OR", "IC95% inf.", "IC95% sup."],
)

# FUNCIONES AUXILIARES
def source_box(text):
    st.markdown(
        f'<div class="source-box">📌 {html.escape(text)}</div>',
        unsafe_allow_html=True,
    )

def card(title, body, icon=""):
    st.markdown(
        f"""
        <div class="card">
            <h4>{icon} {html.escape(title)}</h4>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_header(number, title, subtitle):
    st.markdown(
        f'<div class="section-title">{number}. {html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )

def fmt_p(x):
    if pd.isna(x):
        return "—"
    if x < 0.001:
        return "<0,001"
    return f"{x:.3f}".replace(".", ",")

def fmt_num(x):
    if pd.isna(x):
        return "—"
    return f"{x:.2f}".replace(".", ",")

# BARRA LATERAL
sections = [
    "01 · Introducción",
    "02 · Contexto y pregunta",
    "03 · Datos y diseño",
    "04 · Prevalencia",
    "05 · Tabla 1 · Descriptivos",
    "06 · Modelo logístico",
    "07 · Tabla 2 · Modelo final",
    "08 · Interpretación del OR",
    "09 · Evaluación del modelo",
    "10 · Discusión",
    "11 · Conclusiones",
]
with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=105)
    else:
        st.markdown('<div class="ut-logo-fallback">UT</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="ut-header">
            <div class="ut-name">UNIVERSIDAD DEL TOLIMA</div>
            <div class="ut-program">Modelos lineales generalizados</div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Contenido")
    section = st.radio(
        "Etapa de la exposición",
        sections,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 📚 Ficha del estudio")
    st.write("**Lugar:** Sogamoso, Boyacá")
    st.write("**Periodo:** 2012–2017")
    st.write("**Casos:** 524")
    st.write("**Mujeres:** 336 (64,2%)")
    st.write("**Hombres:** 188 (35,8%)")
    st.write("**Fuente:** SIVIGILA")
    st.write("**Diseño:** analítico transversal")
    st.markdown("---")
    st.caption("Fuente de los resultados: Vásquez-Escobar & Benítez-Camargo (2021).")

page_map = {
    "01 · Introducción": 1,
    "02 · Contexto y pregunta": 3,
    "03 · Datos y diseño": 5,
    "04 · Prevalencia": 6,
    "05 · Tabla 1 · Descriptivos": 7,
    "06 · Modelo logístico": 8,
    "07 · Tabla 2 · Modelo final": 9,
    "08 · Interpretación del OR": 9,
    "09 · Evaluación del modelo": 9,
    "10 · Discusión": 11,
    "11 · Conclusiones": 13,
}
current_page = page_map[section]

# ENCABEZADO PRINCIPAL
logo_path = Path(__file__).resolve().parent / "logo.png"
if logo_path.exists():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(str(logo_path), width=105)
st.markdown(
    """
    <div class="main-university-header">
        <div class="title">INTENTO SUICIDA:  un análisis municipal 
de factores asociados 2012-2017</div>
        <div class="university"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 01 INTRODUCCIÓN
if section == "01 · Introducción":
    section_header(
        "1",
        "Adentremonos en el estudio",
        "Antes de hablar de modelos, conozcamos qué estudiaron los autores y dónde ocurrió.",
    )
    a, b = st.columns(2)
    with a:
        card(
            "¿Qué investigaron?",
            "El comportamiento epidemiológico del intento de suicidio y las diferencias entre género y variables sociodemográficas, psicosociales y específicas durante 2012–2017.",
            "🎯",
        )
    with b:
        card(
            "¿Dónde?",
            "El estudio se realizó en <b>Sogamoso, Boyacá, Colombia</b>. Se utilizaron casos reportados al SIVIGILA.",
            "📍",
        )
    st.markdown("<br>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        card("Periodo", "<b>2012–2017</b>", "📅")
    with b:
        card("Casos analizados", "<b>524</b>", "👥")
    with c:
        card("Modelo", "<b>Regresión logística binaria</b>", "📐")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="interpretation"><b>Pregunta General:</b><br>¿Qué diferencias existen entre hombres y mujeres entre los casos de intento de suicidio registrados en Sogamoso y qué variables aparecen asociadas estadísticamente con el género?</div>',
        unsafe_allow_html=True,
    )
    source_box("Resumen y objetivo del artículo.")

# ========================================================
# 02 CONTEXTO
# ========================================================
elif section == "02 · Contexto y pregunta":
    section_header(
        "2",
        "Del problema de salud pública a la pregunta estadística",
        "La investigación nace de un contexto territorial concreto.",
    )
    st.markdown(
        """
        <div class="card">
            <h4>🌎 ¿Por qué Sogamoso?</h4>
            <p>
            El artículo señala que Sogamoso reportaba, desde 2010, el mayor número
            de casos dentro del departamento y plantea la necesidad de caracterizar
            y comprender el comportamiento epidemiológico del intento de suicidio
            para orientar acciones de salud pública.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="big-question">🔎 ¿Qué queremos explicar estadísticamente?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="highlight-card">
            <div style="font-size:25px; font-weight:900; margin-bottom:14px;">
                <i>Y</i> = género
            </div>
            <div style="margin-bottom:12px;">
                <strong>Variable dependiente / de interés:</strong> género.
            </div>
            <div>
                <strong>Variables explicativas:</strong> edad o ciclo vital, área de ocurrencia, ocupación, estado civil, método del intento, posible desencadenante, violencia, enfermedad mental, consumo de alcohol, relaciones familiares, redes de apoyo, entre otras consideradas por los autores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_box(
        "El artículo define género como la variable de interés y describe las variables sociodemográficas, específicas y psicosociales."
    )

# 03 DATOS Y DISEÑO
elif section == "03 · Datos y diseño":
    section_header(
        "3",
        "¿Cómo se construyó la información?",
        "Aquí seguimos el recorrido de los datos antes de analizarlos.",
    )
    cols = st.columns(5)
    steps = [
        ("01", "Caso", "Persona con intento de suicidio"),
        ("02", "UPGD", "Captación y notificación"),
        ("03", "SIVIGILA", "Sistema de vigilancia"),
        ("04", "Seguimiento", "Ficha e historia clínica"),
        ("05", "Análisis", "524 casos incluidos"),
    ]
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step">
                    <div class="step-number">{num}</div>
                    <div class="step-title">{title}</div>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        card("Casos potenciales", "<b>579</b>", "📥")
    with b:
        card("Excluidos", "<b>55</b>", "↘️")
    with c:
        card("Analizados", "<b>524</b>", "✅")
    st.markdown(
        '<div class="interpretation"><b>Importante:</b> los 55 casos excluidos correspondieron a personas que no residían en Sogamoso o casos sin seguimiento.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Diseño")
    st.markdown(
        """
        <div class="highlight-card">
            <p style="margin-bottom: 0.8rem;">El artículo describe un <strong>estudio analítico transversal</strong>, con datos recolectados entre 2012 y 2017.</p>
            <p style="margin-bottom: 0;">Para el análisis univariado se calcularon <strong>prevalencias ajustadas por edad</strong> mediante el método directo y una población estándar propuesta por la OMS. Para las variables asociadas al género se utilizó regresión logística bivariada.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_box("Materiales y métodos del artículo.")

# ========================================================
# 04 PREVALENCIA
# ========================================================
elif section == "04 · Prevalencia":
    section_header(
        "4",
        "Prevalencia ajustada por edad: ¿dónde se concentra el fenómeno?",
        "Esta es una de las figuras centrales del artículo. No es una tasa anual simple: cruza año y grupo de edad.",
    )
    st.markdown('<div class="control-label">Forma de observar la Figura 1</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Forma de observar la Figura 1",
        ["Curvas por año", "Mapa de calor", "Comparar un año"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Curvas por año":
        fig = px.line(
            prev_df,
            x="Grupo de edad",
            y="Prevalencia",
            color="Año",
            markers=True,
            category_orders={"Grupo de edad": age_groups},
            labels={
                "Grupo de edad": "Grupo de edad (años)",
                "Prevalencia": "Prevalencia ajustada",
                "Año": "Año",
            },
        )
        fig.update_layout(
            height=510,
            margin=dict(l=10, r=10, t=30, b=10),
            legend_title="Año",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            <div class="interpretation">
            <b>Lectura estadística:</b>
            el patrón más marcado aparece en edades tempranas. El grupo de
            <b>15–19 años</b> presenta valores elevados a lo largo del periodo;
            el artículo señala un pico de 22,3 por 100.000 habitantes al inicio
            y 21 por 100.000 al final del periodo.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Mapa de calor":
        heat = prev_df.pivot(
            index="Grupo de edad",
            columns="Año",
            values="Prevalencia",
        ).reindex(age_groups)
        fig = px.imshow(
            heat,
            text_auto=".1f",
            aspect="auto",
            labels={"x": "Año", "y": "Grupo de edad", "color": "Prevalencia"},
        )
        fig.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "El mapa de calor permite identificar rápidamente dónde se concentran los valores altos: principalmente en adolescencia y adultez temprana."
        )
    else:
        st.markdown('<div class="control-label">Seleccione un año</div>', unsafe_allow_html=True)
        selected_year = st.selectbox(
            "Seleccione un año",
            sorted(prevalence.keys()),
            label_visibility="collapsed",
        )
        d = prev_df[prev_df["Año"] == selected_year]
        fig = px.bar(
            d,
            x="Grupo de edad",
            y="Prevalencia",
            text="Prevalencia",
            labels={
                "Grupo de edad": "Grupo de edad",
                "Prevalencia": "Prevalencia ajustada",
            },
        )
        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### ¿Cómo interpretarla sin equivocarnos?")
    a, b = st.columns(2)
    with a:
        card(
            "No mirar solamente el máximo",
            "La figura debe leerse como una estructura de edades a través del tiempo. El interés está en la concentración y en su comportamiento, no solo en encontrar el año con el valor más alto.",
            "👁️",
        )
    with b:
        card(
            "Conclusión de los autores",
            "Las prevalencias ajustadas por edad no muestran una reducción significativa desde el inicio hasta el final del periodo y existe concentración en edades tempranas.",
            "📌",
        )
    source_box(
        "Figura 1 y texto de resultados: prevalencia ajustada por grupo de edad y año de ocurrencia."
    )

# ========================================================
# 05 TABLA 1
# ========================================================
elif section == "05 · Tabla 1 · Descriptivos":
    section_header(
        "5",
        "Tabla 1: primero conozcamos quiénes están en los datos",
        "La estadística descriptiva permite descubrir patrones antes de llegar al modelo.",
    )
    variable = st.selectbox(
        "Seleccione una variable de la Tabla 1",
        table1_df["Variable"].drop_duplicates().tolist(),
    )
    d = table1_df[table1_df["Variable"] == variable].copy()
    display = d[
        [
            "Categoría", "Mujer %", "Mujer N",
            "Hombre %", "Hombre N", "Total %", "Total N", "p",
        ]
    ].copy()
    display["Mujer %"] = display["Mujer %"].map(lambda x: f"{x:.2f}%")
    display["Hombre %"] = display["Hombre %"].map(lambda x: f"{x:.2f}%")
    display["Total %"] = display["Total %"].map(lambda x: f"{x:.2f}%")
    display["p"] = display["p"].map(fmt_p)
    st.dataframe(display, use_container_width=True, hide_index=True)
    chart_df = d.melt(
        id_vars=["Categoría"],
        value_vars=["Mujer %", "Hombre %"],
        var_name="Sexo",
        value_name="Porcentaje",
    )
    chart_df["Sexo"] = chart_df["Sexo"].str.replace(" %", "", regex=False)
    fig = px.bar(
        chart_df,
        x="Categoría",
        y="Porcentaje",
        color="Sexo",
        barmode="group",
        text="Porcentaje",
        labels={"Porcentaje": "% dentro del sexo"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🧠 ¿Qué debe decir el expositor?")
    interpretation_map = {
        "Área de ocurrencia": "La mayoría de los casos corresponde al área urbana. La diferencia por sexo aparece estadísticamente significativa según el valor p reportado (0,02).",
        "Edad agrupada": "La adolescencia concentra 50,3% de los casos de mujeres, mientras que en hombres la adultez temprana representa 45,7%. El valor p reportado es <0,001.",
        "Estado civil": "Soltero es la categoría más frecuente en ambos grupos, pero el valor p reportado para la comparación global es 0,115.",
        "Ocupación": "Estudiante es la categoría de mayor representación en ambos sexos; también aparecen diferencias marcadas en ama de casa, desempleo y otras categorías. El valor p reportado es <0,001.",
        "Forma de realización": "La forma impulsiva es predominante en ambos grupos. El valor p es 0,551, por lo que no se observa evidencia de diferencia estadísticamente significativa bajo el umbral de 0,05.",
        "Antecedentes de intento": "La distribución entre presencia y ausencia de antecedentes es parecida entre sexos; el valor p reportado es 0,80.",
        "Método del intento": "La intoxicación con medicamentos es la categoría más frecuente en mujeres y también representa una proporción importante en hombres. El valor p reportado es <0,001.",
        "Posible desencadenante": "El conflicto con la pareja es el desencadenante con mayor porcentaje en mujeres y uno de los principales en hombres. El valor p reportado es 0,005.",
        "Enfermedad mental": "No se observa una diferencia marcada entre mujeres y hombres en la presencia de diagnóstico; el valor p es 0,68.",
        "Violencia": "La proporción de violencia reportada es mayor en mujeres (54%) que en hombres (37,5%). El valor p reportado es 0,001.",
        "Consumo de alcohol": "El consumo de alcohol aparece en 40,3% de mujeres y 59,7% de hombres. El valor p reportado es <0,001.",
        "Relaciones familiares": "Las relaciones familiares disfuncionales son muy frecuentes en ambos grupos, con porcentajes cercanos al 80%. El valor p es 0,69.",
        "Redes de apoyo": "La mayoría reporta redes de apoyo en ambos grupos. El valor p reportado es 0,23.",
    }
    st.markdown(
        f'<div class="interpretation">{interpretation_map.get(variable, "Revise las diferencias descriptivas y el valor p reportado.")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="small-note">⚠️ El valor p de la Tabla 1 describe la comparación reportada por los autores; no es todavía el resultado del modelo logístico.</div>',
        unsafe_allow_html=True,
    )
    source_box(
        "Tabla 1 del artículo. Los porcentajes y frecuencias fueron transcritos de la tabla publicada."
    )

# ========================================================
# 06 MODELO LOGÍSTICO
# ========================================================
elif section == "06 · Modelo logístico":
    section_header(
        "6",
        "¿Por qué aparece la regresión logística?",
        "parte matemática.",
  
      
    )
    st.markdown(
        '<div class="big-question">La variable de interés es binaria: queremos modelar la pertenencia al grupo de género.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="math-visible">
            <div class="formula-title">Variable respuesta</div>
            <div style="font-size:1.12rem; line-height:2;">
                <b><i>Y<sub>i</sub></i> =</b>
                <span style="display:inline-block; vertical-align:middle; margin-left:.5rem;">
                    { 1, &nbsp; <b>Si la persona que realizó el intento de suicidio pertenece al género femenino</b><br>
                      &nbsp;&nbsp;0, &nbsp; <b>Si la persona pertenece al género masculino</b> }
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # MODIFICADO: Títulos con destaque alto contraste y fórmulas en color negro
    st.markdown('<div class="highlight-header">1. Probabilidad condicionada</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="math-visible"><b>π<sub>i</sub> = P(Y<sub>i</sub> = 1 | X)</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="highlight-header">2. Transformación logit</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="math-visible">
            <b>logit(π<sub>i</sub>) = ln( π<sub>i</sub> / (1 − π<sub>i</sub>) )</b><br>
            <b>= β<sub>0</sub> + β<sub>1</sub>X<sub>1i</sub> + ⋯ + β<sub>k</sub>X<sub>ki</sub></b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="math-visible"><b>β = 1.28</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="math-visible"><b>OR = e<sup>1.28</sup> ≈ 3.58</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="math-box">
        <b>¿Por qué no usar regresión lineal simple?</b><br>
        La regresión lineal clásica podría predecir probabilidades menores a 0 o mayores a 1.
        <br><br>
        La transformación logit mapea el rango de probabilidad (0, 1) a toda la recta real (-∞, +∞).
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_box(
        "Marco teórico del modelo lineal generalizado para respuesta binaria."
    )

# ========================================================
# 07 TABLA 2
# ========================================================
elif section == "07 · Tabla 2 · Modelo final":
    section_header(
        "7",
        "Modelo multivariado final (Tabla 2)",
        "Resultados estimados en el artículo para el modelo multivariado.",
    )
    display_t2 = table2_df.copy()
    display_t2["p"] = display_t2["p"].map(fmt_p)
    display_t2["B"] = display_t2["B"].map(fmt_num)
    display_t2["Wald"] = display_t2["Wald"].map(fmt_num)
    display_t2["OR"] = display_t2["OR"].map(fmt_num)
    display_t2["IC95% inf."] = display_t2["IC95% inf."].map(fmt_num)
    display_t2["IC95% sup."] = display_t2["IC95% sup."].map(fmt_num)
    st.dataframe(display_t2, use_container_width=True, hide_index=True)
    st.markdown("### Forest Plot de los Odds Ratio (OR)")
    plot_data = table2_df[table2_df["Variable"] != "Constante"].copy()
    fig = go.Figure()
    for _, row in plot_data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["IC95% inf."], row["OR"], row["IC95% sup."]],
                y=[f"{row['Variable']}: {row['Categoría']}"] * 3,
                mode="lines+markers",
                marker=dict(size=[6, 10, 6], color=BLUE),
                line=dict(color=GRAY, width=2),
                showlegend=False,
            )
        )
    fig.add_vline(x=1.0, line_dash="dash", line_color=RED)
    fig.update_layout(
        title="Odds Ratio (OR) e Intervalos de Confianza del 95%",
        xaxis_title="OR (Escala logarítmica sugerida para interpretación)",
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    source_box("Tabla 2 del artículo: estimadores del modelo multivariado.")

# ========================================================
# 08 INTERPRETACIÓN DEL OR
# ========================================================
elif section == "08 · Interpretación del OR":
    section_header(
        "8",
        "¿Cómo interpretar el Odds Ratio (OR)?",
        "Relación entre el coeficiente estimado B y la razón de momios e interpretación práctica.",
    )
    st.markdown(
        '<div class="math-visible"><b><i>OR</i> = e<sup>β</sup></b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="math-visible">
            <ul style="margin:0; padding-left:1.3rem;">
                <li><b>Si OR &gt; 1:</b> la presencia de la característica incrementa los momios (odds) del evento frente a la categoría de referencia.</li>
                <li><b>Si OR &lt; 1:</b> la presencia de la característica disminuye los momios (odds) del evento frente a la categoría de referencia.</li>
                <li><b>Si OR = 1:</b> no hay diferencia en los momios entre categorías.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="formula-title" style="font-size:1.15rem; margin-top:1rem;">Ejemplo 1 · Consumo de alcohol</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        card(
            "Consumo de Alcohol (Sí)",
            '<span style="color:#000000 !important;"><b>B = 1.28</b><br><b><i>OR</i> = e<sup>1.28</sup> ≈ 3.58</b><br>'
            "<i>(IC 95%: 2.17 – 5.90, p < 0.001)</i><br><br>"
            "Los momios asociados a la categoría de interés son 3.58 veces mayores cuando se reporta consumo de alcohol en comparación con quienes no consumieron.</span>",
            "🍺",
        )
    with col_b:
        card(
            "Edad: Adolescencia",
            "<b>B = -1.04</b><br><b>OR = e^(-1.04) ≈ 0.35</b><br>"
            "<i>(IC 95%: 0.17 – 0.74, p = 0.01)</i><br><br>"
            "El OR menor que 1 indica una reducción significativa en los momios respecto a la categoría de referencia.",
            "📉",
        )
    source_box("Interpretación metodológica del Odds Ratio.")

# ========================================================
# 09 EVALUACIÓN
# ========================================================
elif section == "09 · Evaluación del modelo":
    section_header(
        "9",
        "Evaluación de la bondad de ajuste del modelo",
        "Pruebas estadísticas para verificar la validez del modelo multivariado.",
    )
    c_a, c_b = st.columns(2)
    with c_a:
        card(
            "Prueba de Hosmer-Lemeshow",
            "Evalúa la coincidencia entre las frecuencias observadas y esperadas por el modelo.<br><br>"
            "Un valor p > 0.05 indica que no hay diferencia significativa entre lo observado y lo predicho, sugiriendo un <b>buen ajuste del modelo</b>.",
            "🧪",
        )
    with c_b:
        card(
            "Estadístico de Wald",
            "Evalúa la significancia individual de cada coeficiente beta en el modelo.<br><br>"
            "Valores de p < 0.05 sugieren que la variable aporta significativamente a la predicción del modelo.",
            "📊",
        )
    source_box("Diagnóstico y validación del modelo estadístico.")

# ========================================================
# 10 DISCUSIÓN
# ========================================================
elif section == "10 · Discusión":
    section_header(
        "10",
        "Discusión e integración con la literatura",
        "Comparación de los hallazgos con otros estudios epidemiológicos.",
    )
    st.markdown(
        """
        <div class="card">
            <h4>💡 Puntos clave de discusión:</h4>
            <ul>
                <li><b>Concentración en adolescentes y jóvenes:</b>
                coincide con reportes nacionales e internacionales donde las conductas
                autolesivas se manifiestan marcadamente en el ciclo vital joven.</li>
                <li><b>Consumo de alcohol como factor crítico:</b>
                actúa como desinhibidor e incrementa la vulnerabilidad en situaciones
                de crisis emocional.</li>
                <li><b>Diferencias según disparadores psicosociales:</b>
                los conflictos de pareja y familiares son detonantes recurrentes
                expuestos en la caracterización epidemiológica.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_box("Sección de Discusión del artículo.")

# ========================================================
# 11 CONCLUSIONES
# ========================================================
elif section == "11 · Conclusiones":
    section_header(
        "11",
        "Conclusiones y recomendaciones de salud pública",
        "Implicaciones de la investigación para la toma de decisiones.",
    )
    a, b = st.columns(2)
    with a:
        card(
            "Conclusiones Estadísticas",
            "El modelo multivariado permitió identificar que factores como la edad, el consumo de alcohol y ciertos desencadenantes psicosociales presentan asociaciones estadísticamente significativas.",
            "📌",
        )
    with b:
        card(
            "Recomendaciones en Salud Pública",
            "Fortalecer estrategias preventivas dirigidas a adolescentes y adultos jóvenes, con énfasis en intervención temprana frente al consumo de alcohol y la resolución de conflictos familiares.",
            "🏛️",
        )
    source_box("Sección de Conclusiones y recomendaciones del artículo.")

# ============================================================
# PIE
# ============================================================
st.markdown(
    '<div class="footer">Universidad del Tolima · Exposición de estadística · Del artículo a la evidencia</div>',
    unsafe_allow_html=True,
)
