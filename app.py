import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Intento suicida | Análisis estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta académica
NAVY = "#0F172A"
BLUE = "#2563EB"
RED = "#D7263D"
GREEN = "#059669"
ORANGE = "#D97706"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT = "#F8FAFC"
WHITE = "#FFFFFF"
BORDER = "#E2E8F0"


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {LIGHT};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0F172A 0%, #172554 100%);
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{
        color: #F8FAFC !important;
    }}

    [data-baseweb="select"] *,
    div[role="listbox"] * {{
        color: #111827 !important;
    }}

    .hero {{
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        border-radius: 20px;
        padding: 1.5rem 1.8rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(15,23,42,.15);
    }}

    .hero h1 {{
        color: white !important;
        font-size: 2.15rem;
        margin: 0;
        font-weight: 800;
    }}

    .hero p {{
        color: #DBEAFE !important;
        margin: .35rem 0 0 0;
        font-size: 1rem;
    }}

    .section-title {{
        color: {NAVY};
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: .25rem;
        margin-bottom: .15rem;
    }}

    .section-subtitle {{
        color: {GRAY};
        font-size: .96rem;
        margin-bottom: 1rem;
    }}

    .card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 3px 12px rgba(15,23,42,.055);
        height: 100%;
        color: #111827 !important;
    }}

    .card h3, .card h4 {{
        color: {NAVY} !important;
        margin-top: 0;
    }}

    .card p, .card li, .card div {{
        color: #111827 !important;
    }}

    .highlight {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 5px solid {BLUE};
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .8rem 0;
        color: #111827 !important;
    }}

    .warning {{
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-left: 5px solid {ORANGE};
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .8rem 0;
        color: #111827 !important;
    }}

    .success {{
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid {GREEN};
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .8rem 0;
        color: #111827 !important;
    }}

    .source {{
        background: #F1F5F9;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: .7rem .9rem;
        color: {GRAY} !important;
        font-size: .82rem;
        margin-top: .8rem;
    }}

    .flow {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: .9rem .5rem;
        text-align: center;
        min-height: 105px;
        box-shadow: 0 2px 8px rgba(15,23,42,.04);
    }}

    .flow-number {{
        color: {RED};
        font-weight: 800;
        font-size: .78rem;
    }}

    .flow-title {{
        color: {NAVY};
        font-weight: 800;
        margin: .2rem 0;
    }}

    .flow-text {{
        color: {GRAY};
        font-size: .8rem;
    }}

    .formula {{
        background: #F8FAFC;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: .8rem;
        text-align: center;
        margin: .7rem 0;
    }}

    .footer {{
        text-align: center;
        color: {GRAY};
        font-size: .78rem;
        padding: 1.5rem 0 .5rem;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: .7rem;
    }}

    div[data-testid="stMetric"] * {{
        color: #111827 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATOS DEL ARTÍCULO
# ============================================================

age_groups = [
    "5 a 9", "10 a 14", "15 a 19", "20 a 24",
    "25 a 29", "30 a 34", "35 a 39", "40 a 44",
    "45 a 49", "50 a 54", "55 a 59", "60 a 64",
    "65 a 69", "70 a 74", "75 a 79", "80 y más",
]

prevalence = {
    2012: [0.0, 5.8, 16.9, 16.9, 8.8, 6.0, 4.1, 3.4, 1.1, 0.5, 1.1, 0.6, 0.0, 0.0, 0.0, 0.0],
    2013: [0.0, 9.6, 23.5, 21.4, 16.3, 6.8, 5.9, 5.1, 3.1, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2014: [0.0, 0.4, 16.9, 15.9, 11.7, 10.8, 5.0, 0.9, 0.0, 0.7, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    2015: [0.0, 4.2, 16.2, 17.0, 8.9, 5.9, 3.0, 0.9, 1.5, 0.7, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    2016: [0.0, 9.4, 26.8, 15.2, 5.0, 2.0, 4.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    2017: [0.0, 4.3, 21.0, 19.1, 9.0, 8.1, 5.0, 3.8, 3.1, 0.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.7],
}

prev_df = pd.DataFrame(
    [
        {"Año": year, "Grupo de edad": age, "Prevalencia": value}
        for year, values in prevalence.items()
        for age, value in zip(age_groups, values)
    ]
)

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
        "Variable", "Categoría", "Mujer %", "Mujer N",
        "Hombre %", "Hombre N", "Total %", "Total N", "p"
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


# ============================================================
# FUNCIONES
# ============================================================

def section_header(number, title, subtitle):
    st.markdown(
        f'<div class="section-title">{number}. {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def card(title, body, icon=""):
    st.markdown(
        f"""
        <div class="card">
            <h4>{icon} {title}</h4>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_box(text):
    st.markdown(
        f'<div class="source">📌 {text}</div>',
        unsafe_allow_html=True,
    )


def fmt_p(value):
    if pd.isna(value):
        return "—"
    if value < 0.001:
        return "<0,001"
    return f"{value:.3f}".replace(".", ",")


def fmt_num(value):
    if pd.isna(value):
        return "—"
    return f"{value:.2f}".replace(".", ",")


# ============================================================
# SIDEBAR
# ============================================================

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
    st.markdown("## 📊 EXPOSICIÓN")
    st.markdown("### Modelos lineales generalizados")
    st.caption("UNIVERSIDAD DEL TOLIMA")
    st.markdown("---")

    section = st.radio(
        "Navegación",
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
    st.caption("Vásquez-Escobar & Benítez-Camargo (2021)")


# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Intento suicida: un análisis municipal de factores asociados</h1>
        <p>Sogamoso, Boyacá · 2012–2017 · Una lectura estadística del artículo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Casos analizados", "524")

with m2:
    st.metric("Mujeres", "336", "64,2%")

with m3:
    st.metric("Hombres", "188", "35,8%")

with m4:
    st.metric("Periodo", "2012–2017")

st.markdown("---")


# ============================================================
# 01 · INTRODUCCIÓN
# ============================================================

if section == "01 · Introducción":

    section_header(
        "1",
        "Entrar al estudio",
        "Primero entendemos qué estudiaron los autores y cuál es la pregunta estadística.",
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
            "¿Dónde se realizó?",
            "El estudio se desarrolló en <b>Sogamoso, Boyacá, Colombia</b>, utilizando casos reportados al SIVIGILA.",
            "📍",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        card("Periodo", "<b>2012–2017</b>", "📅")

    with b:
        card("Casos", "<b>524</b>", "👥")

    with c:
        card("Método principal", "<b>Regresión logística binaria</b>", "📐")

    st.markdown(
        """
        <div class="highlight">
            <b>Pregunta para abrir la exposición</b><br><br>
            ¿Qué diferencias existen entre hombres y mujeres entre los casos de
            intento de suicidio registrados en Sogamoso y qué variables aparecen
            asociadas estadísticamente con el género?
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Resumen y objetivo del artículo.")


# ============================================================
# 02 · CONTEXTO
# ============================================================

elif section == "02 · Contexto y pregunta":

    section_header(
        "2",
        "Del problema de salud pública a la pregunta estadística",
        "La investigación parte de un fenómeno observado en un territorio concreto.",
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
        '<div class="highlight"><b>🔎 ¿Qué queremos explicar estadísticamente?</b></div>',
        unsafe_allow_html=True,
    )

    st.latex(r"Y = \text{género}")

    a, b = st.columns(2)

    with a:
        card(
            "Variable de interés",
            "La variable dependiente o de interés es <b>género</b>.",
            "Y",
        )

    with b:
        card(
            "Variables explicativas",
            "Edad/ciclo vital, área de ocurrencia, ocupación, estado civil, método del intento, posible desencadenante, violencia, enfermedad mental, consumo de alcohol, relaciones familiares y redes de apoyo.",
            "X",
        )

    source_box("Variables sociodemográficas, específicas y psicosociales descritas en el artículo.")


# ============================================================
# 03 · DATOS Y DISEÑO
# ============================================================

elif section == "03 · Datos y diseño":

    section_header(
        "3",
        "¿Cómo se construyó la información?",
        "Seguimos el recorrido de los datos antes de aplicar las técnicas estadísticas.",
    )

    cols = st.columns(5)

    steps = [
        ("01", "Caso", "Persona con intento de suicidio"),
        ("02", "UPGD", "Captación y notificación"),
        ("03", "SIVIGILA", "Sistema de vigilancia"),
        ("04", "Seguimiento", "Ficha e historia clínica"),
        ("05", "Análisis", "524 casos incluidos"),
    ]

    for col, (num, title, text) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="flow">
                    <div class="flow-number">{num}</div>
                    <div class="flow-title">{title}</div>
                    <div class="flow-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.metric("Casos potenciales", "579")

    with b:
        st.metric("Excluidos", "55")

    with c:
        st.metric("Analizados", "524")

    st.markdown(
        """
        <div class="warning">
            <b>Importante:</b> los 55 casos excluidos correspondieron a personas
            que no residían en Sogamoso o a casos sin seguimiento.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Diseño del estudio")

    a, b = st.columns(2)

    with a:
        card(
            "Diseño",
            "El artículo describe un <b>estudio analítico transversal</b> con información recolectada entre 2012 y 2017.",
            "🧩",
        )

    with b:
        card(
            "Análisis",
            "Se calcularon prevalencias ajustadas por edad mediante el método directo y posteriormente se estudiaron asociaciones mediante regresión logística.",
            "📊",
        )

    source_box("Materiales y métodos del artículo.")


# ============================================================
# 04 · PREVALENCIA
# ============================================================

elif section == "04 · Prevalencia":

    section_header(
        "4",
        "Prevalencia ajustada por edad",
        "¿En qué grupos de edad se concentran los valores más altos?",
    )

    mode = st.radio(
        "Explorar la Figura 1",
        ["Curvas por año", "Mapa de calor", "Comparar un año"],
        horizontal=True,
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
                "Grupo de edad": "Grupo de edad",
                "Prevalencia": "Prevalencia ajustada",
                "Año": "Año",
            },
        )

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_tickangle=-45,
            legend_title="Año",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif mode == "Mapa de calor":

        heat = (
            prev_df
            .pivot(index="Grupo de edad", columns="Año", values="Prevalencia")
            .reindex(age_groups)
        )

        fig = px.imshow(
            heat,
            text_auto=".1f",
            aspect="auto",
            labels={
                "x": "Año",
                "y": "Grupo de edad",
                "color": "Prevalencia",
            },
        )

        fig.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        selected_year = st.selectbox(
            "Seleccione un año",
            sorted(prevalence.keys()),
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

    a, b = st.columns(2)

    with a:
        card(
            "Patrón principal",
            "El grupo de <b>15–19 años</b> presenta valores elevados a lo largo del periodo. El artículo señala valores de 22,3 por 100.000 habitantes al inicio y 21 por 100.000 al final del periodo.",
            "📈",
        )

    with b:
        card(
            "Lectura correcta",
            "No debemos mirar solamente el máximo. La figura permite observar cómo se distribuye la prevalencia por edad y cómo cambia ese patrón a través de los años.",
            "🔎",
        )

    st.markdown(
        """
        <div class="success">
            <b>Conclusión de los autores:</b> las prevalencias ajustadas por edad
            no muestran una reducción significativa desde el inicio hasta el final
            del periodo y existe concentración en edades tempranas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Figura 1 y resultados sobre prevalencia ajustada por edad y año de ocurrencia.")


# ============================================================
# 05 · TABLA 1
# ============================================================

elif section == "05 · Tabla 1 · Descriptivos":

    section_header(
        "5",
        "Tabla 1 · Primero conozcamos los datos",
        "La estadística descriptiva permite identificar patrones antes de construir el modelo.",
    )

    variable = st.selectbox(
        "Seleccione una variable",
        table1_df["Variable"].drop_duplicates().tolist(),
    )

    d = table1_df[table1_df["Variable"] == variable].copy()

    display = d[
        [
            "Categoría", "Mujer %", "Mujer N",
            "Hombre %", "Hombre N",
            "Total %", "Total N", "p",
        ]
    ].copy()

    display["Mujer %"] = display["Mujer %"].map(lambda x: f"{x:.2f}%")
    display["Hombre %"] = display["Hombre %"].map(lambda x: f"{x:.2f}%")
    display["Total %"] = display["Total %"].map(lambda x: f"{x:.2f}%")
    display["p"] = display["p"].map(fmt_p)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

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

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=-35,
    )

    st.plotly_chart(fig, use_container_width=True)

    interpretation_map = {
        "Área de ocurrencia": "La mayoría de los casos corresponde al área urbana. La diferencia por sexo aparece estadísticamente significativa según el valor p reportado (0,02).",
        "Edad agrupada": "La adolescencia concentra 50,3% de los casos de mujeres, mientras que en hombres la adultez temprana representa 45,7%. El valor p reportado es <0,001.",
        "Estado civil": "Soltero es la categoría más frecuente en ambos grupos, pero el valor p reportado para la comparación global es 0,115.",
        "Ocupación": "Estudiante es la categoría de mayor representación en ambos sexos; también aparecen diferencias en ama de casa, desempleo y otras categorías. El valor p reportado es <0,001.",
        "Forma de realización": "La forma impulsiva es predominante en ambos grupos. El valor p es 0,551, por lo que no se observa evidencia de diferencia estadísticamente significativa bajo el umbral de 0,05.",
        "Antecedentes de intento": "La distribución entre presencia y ausencia de antecedentes es parecida entre sexos; el valor p reportado es 0,80.",
        "Método del intento": "Los medicamentos representan la categoría más frecuente en mujeres y una proporción importante en hombres. El valor p reportado es <0,001.",
        "Posible desencadenante": "El conflicto con la pareja es uno de los principales desencadenantes. El valor p reportado es 0,005.",
        "Enfermedad mental": "No se observa una diferencia marcada entre mujeres y hombres en la presencia de diagnóstico; el valor p es 0,68.",
        "Violencia": "La proporción de violencia reportada es mayor en mujeres (54%) que en hombres (37,5%). El valor p reportado es 0,001.",
        "Consumo de alcohol": "El consumo de alcohol aparece en 40,3% de mujeres y 59,7% de hombres. El valor p reportado es <0,001.",
        "Relaciones familiares": "Las relaciones familiares disfuncionales son muy frecuentes en ambos grupos, con porcentajes cercanos al 80%. El valor p es 0,69.",
        "Redes de apoyo": "La mayoría reporta redes de apoyo en ambos grupos. El valor p reportado es 0,23.",
    }

    st.markdown(
        f'<div class="highlight"><b>🧠 Lectura:</b><br>{interpretation_map.get(variable, "Revise las diferencias descriptivas y el valor p reportado.")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning">
            <b>Recuerda:</b> el valor p de la Tabla 1 corresponde a la comparación
            descriptiva reportada por los autores; todavía no es el resultado del
            modelo logístico.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Tabla 1 del artículo. Porcentajes y frecuencias transcritos de la tabla publicada.")


# ============================================================
# 06 · MODELO LOGÍSTICO
# ============================================================

elif section == "06 · Modelo logístico":

    section_header(
        "6",
        "¿Por qué utilizar regresión logística?",
        "Aquí empieza la parte matemática del análisis.",
    )

    st.markdown(
        """
        <div class="highlight">
            <b>Idea central:</b> la regresión logística se utiliza cuando la
            respuesta que queremos modelar es binaria.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        Y_i =
        \begin{cases}
        1 & \text{categoría de interés}\\
        0 & \text{categoría de referencia}
        \end{cases}
        """
    )

    st.markdown("### 1. Probabilidad condicionada")

    st.latex(r"\pi_i = P(Y_i = 1 \mid X)")

    st.markdown("### 2. Transformación logit")

    st.latex(
        r"""
        \operatorname{logit}(\pi_i)
        =
        \ln\left(\frac{\pi_i}{1-\pi_i}\right)
        =
        \beta_0+\beta_1X_{1i}+\cdots+\beta_kX_{ki}
        """
    )

    a, b = st.columns(2)

    with a:
        card(
            "¿Por qué no regresión lineal?",
            "Porque una regresión lineal podría producir valores predichos menores que 0 o mayores que 1 cuando estamos intentando representar una probabilidad.",
            "❌",
        )

    with b:
        card(
            "¿Qué hace el logit?",
            "Transforma las probabilidades del intervalo (0,1) a toda la recta real, permitiendo modelarlas mediante una combinación lineal de variables explicativas.",
            "✅",
        )

    st.markdown("### 3. Del coeficiente al Odds Ratio")

    st.latex(r"\boxed{OR=e^\beta}")

    st.markdown(
        """
        <div class="success">
            <b>Puente conceptual:</b><br>
            coeficiente β → exponenciación → OR → intervalo de confianza → interpretación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Marco metodológico de la regresión logística para respuesta binaria.")


# ============================================================
# 07 · TABLA 2
# ============================================================

elif section == "07 · Tabla 2 · Modelo final":

    section_header(
        "7",
        "Modelo multivariado final",
        "Los resultados del modelo se presentan mediante β, Wald, p, OR e intervalo de confianza.",
    )

    display_t2 = table2_df.copy()

    display_t2["B"] = display_t2["B"].map(fmt_num)
    display_t2["Wald"] = display_t2["Wald"].map(fmt_num)
    display_t2["p"] = display_t2["p"].map(fmt_p)
    display_t2["OR"] = display_t2["OR"].map(fmt_num)
    display_t2["IC95% inf."] = display_t2["IC95% inf."].map(fmt_num)
    display_t2["IC95% sup."] = display_t2["IC95% sup."].map(fmt_num)

    st.dataframe(
        display_t2,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Forest plot")

    plot_data = table2_df[table2_df["Variable"] != "Constante"].copy()

    fig = go.Figure()

    for _, row in plot_data.iterrows():
        label = f"{row['Variable']}: {row['Categoría']}"

        fig.add_trace(
            go.Scatter(
                x=[
                    row["IC95% inf."],
                    row["OR"],
                    row["IC95% sup."],
                ],
                y=[label, label, label],
                mode="lines+markers",
                marker=dict(
                    size=[6, 11, 6],
                    color=BLUE,
                ),
                line=dict(
                    color=GRAY,
                    width=2,
                ),
                showlegend=False,
            )
        )

    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color=RED,
    )

    fig.update_layout(
        title="Odds Ratio e intervalos de confianza del 95%",
        xaxis_title="Odds Ratio",
        yaxis_title="Variable",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="highlight">
            <b>Cómo leer el gráfico:</b><br>
            La línea vertical en OR = 1 representa ausencia de diferencia en los
            momios. Si el intervalo de confianza no cruza 1, existe evidencia
            estadística de asociación al nivel correspondiente.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Tabla 2 del artículo: estimadores del modelo multivariado.")


# ============================================================
# 08 · INTERPRETACIÓN DEL OR
# ============================================================

elif section == "08 · Interpretación del OR":

    section_header(
        "8",
        "¿Cómo interpretar el Odds Ratio?",
        "Pasamos de los coeficientes del modelo a una interpretación comprensible.",
    )

    st.latex(r"\boxed{OR=e^\beta}")

    a, b, c = st.columns(3)

    with a:
        card(
            "OR > 1",
            "Los momios son mayores en la categoría de interés respecto a la categoría de referencia.",
            "⬆️",
        )

    with b:
        card(
            "OR < 1",
            "Los momios son menores en la categoría de interés respecto a la referencia.",
            "⬇️",
        )

    with c:
        card(
            "OR = 1",
            "No hay diferencia en los momios entre las categorías comparadas.",
            "⚖️",
        )

    st.markdown("### Ejemplo 1 · Consumo de alcohol")

    st.latex(r"\beta=1.28")

    st.latex(r"OR=e^{1.28}\approx3.58")

    st.markdown(
        """
        <div class="success">
            <b>Resultado reportado:</b> OR = 3,58; IC95% = 2,17–5,90;
            p < 0,001.<br><br>
            En el modelo del artículo, la categoría de consumo de alcohol presenta
            unos momios 3,58 veces mayores respecto a la categoría de referencia,
            manteniendo constantes las demás variables incluidas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Ejemplo 2 · Adolescencia")

    st.latex(r"\beta=-1.04")

    st.latex(r"OR=e^{-1.04}\approx0.35")

    st.markdown(
        """
        <div class="warning">
            <b>Interpretación:</b> un OR de 0,35 indica menores momios respecto a
            la categoría de referencia. Si queremos expresar la comparación en el
            sentido inverso, podemos considerar el recíproco:
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"\frac{1}{0.35}\approx2.86")

    source_box("Interpretación del OR a partir de los resultados reportados en la Tabla 2.")


# ============================================================
# 09 · EVALUACIÓN DEL MODELO
# ============================================================

elif section == "09 · Evaluación del modelo":

    section_header(
        "9",
        "Evaluación del modelo",
        "No basta con estimar coeficientes: también debemos evaluar el comportamiento del modelo.",
    )

    a, b = st.columns(2)

    with a:
        card(
            "Hosmer–Lemeshow",
            "Permite evaluar la correspondencia entre frecuencias observadas y esperadas. Un valor p mayor que 0,05 es compatible con un buen ajuste bajo este criterio.",
            "🧪",
        )

    with b:
        card(
            "Wald",
            "Evalúa la significancia individual de los coeficientes. Valores de p menores que 0,05 indican evidencia de que el coeficiente difiere de cero.",
            "📊",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="highlight">
            <b>Dos preguntas diferentes:</b><br><br>
            <b>Wald:</b> ¿esta variable aporta evidencia estadística dentro del modelo?<br>
            <b>Hosmer–Lemeshow:</b> ¿el modelo presenta un ajuste razonable según esta prueba?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Otros criterios reportados")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Durbin–Watson", "1,94")

    with c2:
        st.metric("VIF", "1")

    st.markdown(
        """
        <div class="success">
            El artículo reporta que el modelo cumple el supuesto de independencia
            de errores y no presenta problemas de multicolinealidad según los
            indicadores reportados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Resultados de evaluación del modelo reportados por los autores.")


# ============================================================
# 10 · DISCUSIÓN
# ============================================================

elif section == "10 · Discusión":

    section_header(
        "10",
        "Discusión",
        "Ahora conectamos los resultados estadísticos con la interpretación epidemiológica.",
    )

    a, b = st.columns(2)

    with a:
        card(
            "Adolescentes y jóvenes",
            "El artículo destaca la concentración del fenómeno en edades tempranas y relaciona este patrón con la literatura epidemiológica revisada.",
            "👥",
        )

    with b:
        card(
            "Consumo de alcohol",
            "El modelo identifica una asociación importante con el consumo de alcohol, que los autores interpretan dentro del contexto psicosocial del intento de suicidio.",
            "📈",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        card(
            "Conflictos interpersonales",
            "Los conflictos de pareja y familiares aparecen entre los posibles desencadenantes analizados.",
            "🤝",
        )

    with b:
        card(
            "Violencia",
            "La violencia también presenta una asociación estadísticamente significativa en el modelo reportado.",
            "⚠️",
        )

    st.markdown(
        """
        <div class="highlight">
            <b>Idea para la exposición:</b><br>
            La estadística no termina en el valor p. El objetivo es transformar
            los resultados del modelo en una interpretación coherente con el
            contexto y con las preguntas de investigación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Sección de Discusión del artículo.")


# ============================================================
# 11 · CONCLUSIONES
# ============================================================

elif section == "11 · Conclusiones":

    section_header(
        "11",
        "Conclusiones",
        "Cerramos la exposición regresando a la pregunta inicial.",
    )

    st.markdown(
        """
        <div class="card">
            <h4>🎯 ¿Qué aprendimos del análisis?</h4>
            <ol>
                <li>El fenómeno se concentra especialmente en edades tempranas.</li>
                <li>La estadística descriptiva permite identificar diferencias entre los grupos.</li>
                <li>La regresión logística permite modelar una respuesta binaria mediante variables explicativas.</li>
                <li>El Odds Ratio transforma los coeficientes del modelo en una medida interpretable de asociación.</li>
                <li>Los intervalos de confianza y valores p permiten valorar la evidencia estadística.</li>
                <li>La evaluación del modelo es necesaria para valorar su comportamiento.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        card(
            "Conclusión estadística",
            "El modelo multivariado permitió identificar asociaciones entre género y diferentes factores sociodemográficos y psicosociales reportados por los autores.",
            "📊",
        )

    with b:
        card(
            "Implicación",
            "Los resultados aportan elementos para comprender el fenómeno y orientar estrategias de prevención y atención en el contexto estudiado.",
            "🏛️",
        )

    st.markdown(
        """
        <div class="success">
            <b>Mensaje final:</b><br><br>
            Del dato pasamos a la descripción; de la descripción a la asociación;
            y de la asociación a una interpretación estadística que puede apoyar
            la comprensión del problema.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_box("Conclusiones y recomendaciones del artículo.")


# ============================================================
# PIE
# ============================================================

st.markdown(
    """
    <div class="footer">
        Universidad del Tolima · Exposición de Estadística ·
        Del artículo a la evidencia
    </div>
    """,
    unsafe_allow_html=True,
)
