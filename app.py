import streamlit as st
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Presentación Proyecto Final",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. ESTILO CSS PERSONALIZADO (Consulting Aesthetic)
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Lato:wght@400;700&display=swap');

        /* Fondo y contenedores */
        .stApp {
            background-color: #f4f7f9;
        }
        
        /* Títulos principales */
        .main-title {
            font-family: 'Poppins', sans-serif;
            color: #001f3f;
            font-size: 52px;
            font-weight: 700;
            margin-bottom: 0px;
            line-height: 1.2;
        }
        
        /* Subtítulos */
        .sub-title {
            font-family: 'Poppins', sans-serif;
            color: #008080;
            font-size: 24px;
            font-weight: 600;
            margin-top: -10px;
        }

        /* Títulos de Diapositiva */
        .slide-header {
            font-family: 'Poppins', sans-serif;
            color: #001f3f;
            font-size: 38px;
            border-left: 10px solid #008080;
            padding-left: 20px;
            margin-bottom: 30px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Tarjetas de contenido */
        .content-card {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border-top: 5px solid #008080;
            margin-bottom: 20px;
        }

        /* Texto general */
        p, li {
            font-family: 'Lato', sans-serif;
            font-size: 19px;
            color: #475569;
            line-height: 1.6;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #001f3f;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Métricas */
        [data-testid="stMetricValue"] {
            color: #008080 !important;
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# 3. BARRA LATERAL (Navegación y Logo)
with st.sidebar:
    st.image("http://googleusercontent.com/image_collection/image_retrieval/14714984316335715874", width=150)
    st.markdown("---")
    st.title("Navegación")
    choice = st.radio(
        "Seleccione Diapositiva:",
        [
            "1. Portada",
            "2. Introducción",
            "3. Objetivos y Justificación",
            "4. Marco Teórico",
            "5. Metodología",
            "6. Software y Herramientas",
            "7. Resultados",
            "8. Conclusiones",
            "9. Referencias"
        ]
    )
    st.markdown("---")
    st.info("💡 Use las flechas o el mouse para navegar por el contenido.")

# 4. LÓGICA DE LAS DIAPOSITIVAS

if choice == "1. Portada":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown('<h1 class="main-title">Optimización de Procesos con Inteligencia Artificial</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Proyecto Final de Grado</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Expositores:**
            - Juan Pérez
            - María García
            
            **Curso:**
            Ingeniería de Sistemas 2026
            """)
        with c2:
            st.markdown("""
            **Docente:**
            Dr. Roberto Martínez
            
            **Fecha:**
            23 de Mayo de 2026
            """)

elif choice == "2. Introducción":
    st.markdown('<h2 class="slide-header">Introducción</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>Contextualización</h3>
            <p>En el panorama actual, la eficiencia operativa es el factor determinante para la competitividad empresarial. 
            Este proyecto surge ante la necesidad de reducir cuellos de botella en la cadena de suministro.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card">
            <h3>Planteamiento del Problema</h3>
            <p>Actualmente, el 40% de los retrasos se deben a una mala asignación de recursos basada en predicciones manuales.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.warning("🎯 **Motivación:** Transformar datos brutos en decisiones automatizadas de alto impacto.")

elif choice == "3. Objetivos y Justificación":
    st.markdown('<h2 class="slide-header">Objetivos y Justificación</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Objetivos", "💡 Justificación"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="content-card">
                <h4>General</h4>
                <p>Diseñar e implementar un sistema de recomendación basado en algoritmos de Random Forest.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="content-card">
                <h4>Específicos</h4>
                <ul>
                    <li>Normalizar bases de datos heterogéneas.</li>
                    <li>Entrenar un modelo con precisión > 90%.</li>
                    <li>Desplegar interfaz en Streamlit.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.info("**Importancia:** El proyecto permite una reducción del 15% en costos operativos anuales, validando el uso de herramientas Open Source en entornos corporativos.")

elif choice == "4. Marco Teórico":
    st.markdown('<h2 class="slide-header">Marco Teórico / Conceptual</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown("""
        ### Algoritmos Utilizados
        Se seleccionó **Random Forest** debido a su capacidad para manejar variables categóricas y su resistencia al sobreajuste (overfitting).
        
        **Conceptos Clave:**
        - Árboles de Decisión.
        - Ensamblado (Ensemble Learning).
        - Feature Importance.
        """)
    with col2:
        st.image("http://googleusercontent.com/image_collection/image_retrieval/6219206533574995524", caption="Estructura de Red y Árboles")

elif choice == "5. Metodología":
    st.markdown('<h2 class="slide-header">Metodología de Desarrollo</h2>', unsafe_allow_html=True)
    
    steps = {
        "Fase 1: Recolección": "Limpieza de datos (Data Wrangling) usando Python.",
        "Fase 2: Modelado": "Entrenamiento de algoritmos con Scikit-Learn.",
        "Fase 3: Desarrollo": "Construcción de la UI con Streamlit API.",
        "Fase 4: Validación": "Pruebas A/B y métricas de error (MAE/RMSE)."
    }
    
    for title, desc in steps.items():
        with st.expander(title):
            st.write(desc)

elif choice == "6. Software y Herramientas":
    st.markdown('<h2 class="slide-header">Software Utilizado</h2>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Lenguaje", "Python 3.11")
    c2.metric("Framework", "Streamlit")
    c3.metric("DB", "MySQL")
    
    st.markdown("""
    <div class="content-card">
        <h4>Stack Tecnológico</h4>
        <ul>
            <li><strong>Pandas:</strong> Manipulación de datos.</li>
            <li><strong>Plotly:</strong> Gráficos interactivos.</li>
            <li><strong>Git/GitHub:</strong> Control de versiones.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif choice == "7. Resultados":
    st.markdown('<h2 class="slide-header">Resultados y Hallazgos</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Métricas Finales")
        st.metric("Precisión", "94.2%", "+2.1%")
        st.metric("Velocidad", "0.4s", "-150ms")
    with col2:
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Modelo A', 'Modelo B', 'Propuesto']
        )
        st.area_chart(chart_data)
        st.caption("Comparativa de rendimiento entre modelos probados.")

elif choice == "8. Conclusiones":
    st.markdown('<h2 class="slide-header">Conclusiones y Recomendaciones</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Aprendizajes:** La integración ágil permitió un desarrollo 20% más rápido de lo esperado.")
        st.error("⚠️ **Dificultades:** Calidad de datos inicial insuficiente; requirió limpieza profunda.")
    with col2:
        st.info("🚀 **Sugerencias Futuras:** Implementar aprendizaje por refuerzo y despliegue en microservicios (Docker).")

elif choice == "9. Referencias":
    st.markdown('<h2 class="slide-header">Referencias Bibliográficas</h2>', unsafe_allow_html=True)
    
    data = {
        "Fuente": ["Streamlit Docs", "Scikit-learn", "Python for Data Analysis", "IEEE Research"],
        "Año": [2024, 2023, 2022, 2023],
        "Uso": ["UI/UX", "Algoritmos", "Limpieza de Datos", "Marco Teórico"]
    }
    df = pd.DataFrame(data)
    st.table(df)
    
    st.markdown("<br><br><center><h3>¡Muchas gracias por su atención!</h3></center>", unsafe_allow_html=True)
