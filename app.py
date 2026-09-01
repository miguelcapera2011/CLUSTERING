import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN DE PÁGINA (ANCHO COMPLETO)
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
# ESTILOS CSS PERSONALIZADOS
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
        [data-testid="stSidebar"] span:not([data-baseweb]) {{
            color: #F8FAFC;
        }}
        .card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(15,23,42,.04);
            margin-bottom: 1rem;
        }}
        .card h4 {{
            margin-top: 0;
            color: {NAVY};
        }}
        .big-question {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {NAVY};
            background: white;
            border-left: 5px solid {BLUE};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        .interpretation {{
            background: #FFF7ED;
            border-left: 5px solid {ORANGE};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #000000;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    
    st.markdown("---")
    st.markdown("### 📚 Datos clave")
    st.write("**Población:** Sogamoso, Boyacá")
    st.write("**Muestra:** 524 casos analizados")
    st.write("**Periodo:** 2012 – 2017")
    st.write("**Fuente:** SIVIGILA")

# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================
st.markdown('<div style="font-size: 2.3rem; font-weight: 800; color: #0F172A; margin-bottom: 0.2rem;">Intento suicida: un análisis municipal de factores asociados</div>', unsafe_allow_html=True)
st.markdown('<div style="color: #64748B; font-size: 1.1rem; margin-bottom: 1.5rem;">Sogamoso, Boyacá (2012–2017) · Reconstrucción metodológica y estadística</div>', unsafe_allow_html=True)

# Métricas rápidas superiores
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Casos Totales", "524")
with m2:
    st.metric("Mujeres", "336 (64.2%)")
with m3:
    st.metric("Hombres", "188 (35.8%)")
with m4:
    st.metric("Periodo Evaluado", "6 Años")

st.markdown("---")

# ============================================================
# CONTENIDO DE LA PRESENTACIÓN (ANCHO COMPLETO)
# ============================================================

if section == "01 · Introducción":
    st.markdown("## 1. Entrar al estudio")
    st.markdown("Visión general sobre el comportamiento epidemiológico del intento de suicidio y sus factores asociados.")
    
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(
            """
            <div class="card">
                <h4>🎯 Objetivo de la investigación</h4>
                <p>Caracterizar el comportamiento epidemiológico del intento de suicidio e identificar las diferencias de género según variables sociodemográficas, psicosociales y específicas en Sogamoso durante el periodo 2012–2017.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            """
            <div class="card">
                <h4>📍 Contexto Territorial</h4>
                <p>Estudio realizado en el municipio de <b>Sogamoso (Boyacá, Colombia)</b>. Es una zona de relevancia en salud pública debido al incremento continuo en la tasa de notificaciones en el sistema de vigilancia nacional.</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown(
        """
        <div class="interpretation">
            <b>💡 Nota para la exposición:</b> Resalta que el objetivo central es usar la estadística multivariada para entender qué variables se asocian de forma independiente al género en los casos reportados.
        </div>
        """, unsafe_allow_html=True
    )

elif section == "02 · Contexto y pregunta":
    st.markdown("## 2. Contexto y pregunta de investigación")
    
    st.markdown(
        """
        <div class="big-question">
            ❓ Pregunta central: ¿Qué factores sociodemográficos y psicosociales diferencian significativamente el intento de suicidio entre hombres y mujeres?
        </div>
        """, unsafe_allow_html=True
    )
    
    col_x, col_y = st.columns(2, gap="large")
    with col_x:
        st.markdown("### Variable Dependiente ($Y$)")
        st.write("Se define el **género** como la variable de respuesta binaria:")
        st.latex(r"Y = \begin{cases} 1 & \text{si es Hombre} \\ 0 & \text{si es Mujer} \end{cases}")
        
    with col_y:
        st.markdown("### Variables Explicativas ($X$)")
        st.write("Grupo de factores evaluados en la población:")
        st.markdown("""
        * **Sociodemográficas:** Edad, área de ocurrencia, ocupación, estado civil.
        * **Específicas:** Método utilizado, desencadenante, antecedentes previos.
        * **Psicosociales:** Consumo de alcohol, presencia de violencia, apoyo familiar.
        """)

elif section == "03 · Datos y diseño":
    st.markdown("## 3. Datos y diseño metodológico")
    st.write("Ruta de recolección, selección de la muestra y metodología estadística empleada.")
    
    # Diagrama de flujo del dato
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='card'><b>1. Evento</b><br>Notificación de intento de suicidio.</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>2. Captura</b><br>Registro UPGD en la ficha SIVIGILA.</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><b>3. Depuración</b><br>Filtro de 579 casos iniciales.</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><b>4. Muestra Final</b><br><b>524 casos</b> incluidos.</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Tipo de Estudio")
    st.write("Estudio **observacional, analítico y transversal**. Se calcularon prevalencias ajustadas por edad mediante el método directo (población estándar OMS) y se aplicaron modelos de regresión logística.")

elif section == "04 · Prevalencia":
    st.markdown("## 4. Prevalencia según edad y año")
    st.write("Distribución de la prevalencia ajustada por 100.000 habitantes.")
    
    tab1, tab2 = st.columns([1, 2], gap="large")
    
    with tab1:
        selected_year = st.selectbox("Seleccionar año a inspeccionar:", options=list(prevalence.keys()))
        filtered_df = prev_df[prev_df["Año"] == selected_year]
        
        st.markdown(f"**Año seleccionado:** {selected_year}")
        st.write("Observa cómo los picos de prevalencia se concentran principalmente en los grupos de **15 a 19** y **20 a 24 años**.")
        
    with tab2:
        fig = px.bar(
            filtered_df,
            x="Grupo de edad",
            y="Prevalencia",
            title=f"Prevalencia Ajustada por Edad ({selected_year})",
            labels={"Prevalencia": "Prevalencia (por 100k hab)", "Grupo de edad": "Grupo de Edad"},
            color_discrete_sequence=[BLUE]
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

elif section == "05 · Tabla 1 · Descriptivos":
    st.markdown("## 5. Características descriptivas (Tabla 1)")
    st.write("Comparación bivariada de características sociodemográficas según el sexo.")
    
    st.markdown(
        """
        <div class="card">
            <h4>📌 Hallazgos descriptivos clave</h4>
            <ul>
                <li><b>Edad:</b> En las mujeres predomina la adolescencia (50.3%), mientras que en hombres destaca la adultez temprana (45.7%).</li>
                <li><b>Método:</b> Los medicamentos son el método más usado por mujeres (54.4%), mientras que los hombres registran mayor uso de plaguicidas y otros medios.</li>
                <li><b>Consumo de Alcohol:</b> Mayor proporción en hombres (59.7%) que en mujeres (40.3%) durante el evento.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

elif section == "06 · Modelo logístico":
    st.markdown("## 6. Modelo de Regresión Logística Binaria")
    st.write("Estructura matemática del Modelo Lineal Generalizado empleado.")
    
    st.markdown("### Función de Enlace (Logit)")
    st.write("Para modelar la probabilidad $\pi = P(Y=1)$ de pertenecer a un grupo determinado, se utiliza la transformación Logit:")
    
    st.latex(r"\text{logit}(\pi) = \ln\left(\frac{\pi}{1-\pi}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Razón de Momios (Odds Ratio - OR)")
    st.write("La interpretación del impacto de cada variable se obtiene exponenciando los coeficientes del modelo:")
    
    st.latex(r"\text{OR} = e^{\beta_i}")
    
    st.markdown(
        """
        <div class="interpretation">
            <b>Interpretación:</b> Un $\text{OR} > 1$ indica que la variable aumenta la oportunidad del evento frente a la categoría de referencia; un $\text{OR} < 1$ indica que actúa como factor con menor oportunidad relativa.
        </div>
        """, unsafe_allow_html=True
    )
