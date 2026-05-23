import streamlit as st
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Presentación Moderna y Clara",
    page_icon="sc",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. INYECCIÓN DE CSS CON COLORES CLAROS, MODERNOS Y VIVOS
def aplicar_diseno_claro():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

        /* Fondo general de la app (Blanco/Marfil limpio y luminoso) */
        .stApp {
            background-color: #fafbfc;
        }
        
        /* DISEÑO DE LA BARRA LATERAL (Clara con un toque sutil de color) */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0f4f8 0%, #e1e8f0 100%);
            border-right: 1px solid #dcdfe4;
        }
        [data-testid="stSidebar"] * {
            color: #1e293b !important; /* Texto oscuro y legible en el menú */
        }
        
        /* TÍTULOS PRINCIPALES CON DEGRADADO VIVO (Menta -> Azul -> Lavanda) */
        .titulo-gradiente {
            font-family: 'Outfit', sans-serif;
            font-size: 54px;
            font-weight: 800;
            background: linear-gradient(135deg, #059669 0%, #3b82f6 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
            line-height: 1.2;
        }
        
        /* TÍTULOS DE LAS DIAPOSITIVAS */
        .slide-header {
            font-family: 'Outfit', sans-serif;
            font-size: 40px;
            font-weight: 800;
            color: #0f172a; /* Azul casi negro, muy elegante */
            border-left: 8px solid #3b82f6; /* Línea de acento Azul Vivo */
            padding-left: 20px;
            margin-bottom: 35px;
        }

        /* SUBTÍTULOS */
        .sub-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #6366f1; /* Índigo vibrante */
            font-size: 24px;
            font-weight: 600;
            margin-top: -5px;
            margin-bottom: 30px;
        }

        /* TARJETAS DE CONTENIDO (Fondo blanco puro con sombras suaves y coloridas) */
        .tarjeta-clara {
            background: #ffffff;
            padding: 30px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.04); /* Sombra azulada muy sutil */
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }
        .tarjeta-clara:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(139, 92, 246, 0.08); /* Brillo lavanda suave al pasar el mouse */
            border-color: #cbd5e1;
        }
        
        .tarjeta-clara h3, .tarjeta-clara h4 {
            color: #3b82f6; /* Azul vivo para resaltar títulos de secciones */
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            margin-bottom: 15px;
        }

        /* TEXTO GENERAL */
        p, li, span {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 18px;
            color: #334155; /* Gris pizarra oscuro para lectura perfecta sin cansar la vista */
            line-height: 1.6;
        }
        
        strong {
            color: #0f172a !important; /* Negritas bien marcadas */
        }

        /* Pestañas (Tabs) personalizadas */
        .stTabs [data-baseweb="tab"] {
            color: #64748b !important;
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #3b82f6 !important;
            border-bottom-color: #3b82f6 !important;
        }
    </style>
    """, unsafe_allow_html=True)

aplicar_diseno_claro()

# 3. NAVEGADOR EN LA BARRA LATERAL
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center; color: #3b82f6; font-family:Outfit;'>MENÚ</h2>", unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio(
        "Seleccione diapositiva:",
        [
            "✨ Portada Principal",
            "🚀 Introducción",
            "🎯 Objetivos",
            "🧠 Marco Conceptual",
            "🛠️ Metodología",
            "💻 Stack Tecnológico",
            "📊 Resultados y Datos",
            "💡 Conclusiones"
        ]
    )
    st.markdown("---")
    st.write("✨ *Estilo Limpio & Premium*")

# 4. CONTENIDO DE LAS DIAPOSITIVAS

if choice == "✨ Portada Principal":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown('<h1 class="titulo-gradiente">TRANSFORMACIÓN DIGITAL & MODELOS INTELIGENTES</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Optimización de Procesos con Inteligencia Artificial</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tarjeta-clara" style="border-top: 6px solid #8b5cf6;">
            <table style="width:100%; border:none;">
                <tr style="background:none;">
                    <td style="padding:15px; border:none;"><strong>Presentado por:</strong> Nombre del Estudiante</td>
                    <td style="padding:15px; border:none;"><strong>Especialidad:</strong> Ingeniería de Sistemas</td>
                </tr>
                <tr style="background:none;">
                    <td style="padding:15px; border:none;"><strong>Profesor Guía:</strong> Dr. Carlos Mendoza</td>
                    <td style="padding:15px; border:none;"><strong>Fecha:</strong> Mayo 2026</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🚀 Introducción":
    st.markdown('<h2 class="slide-header">Introducción al Problema</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="tarjeta-clara">
            <h3>El Contexto Actual</h3>
            <p>En el ecosistema corporativo actual, el procesamiento manual de datos genera retrasos críticos en la toma de decisiones estratégicas. Las empresas pierden competitividad al no adoptar flujos automatizados.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tarjeta-clara">
            <h3>La Solución Planteada</h3>
            <p>La integración de un modelo predictivo permite anticiparse a las anomalías de los procesos antes de que afecten al cliente final.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-clara" style="text-align: center; border-bottom: 4px solid #059669;">
            <h2 style="color: #059669; font-size: 64px; font-weight:800; margin:0;">-35%</h2>
            <p style="font-size:16px; font-weight:600; color:#475569;">Reducción en tiempos muertos operativos documentada en la fase piloto.</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🎯 Objetivos":
    st.markdown('<h2 class="slide-header">Metas del Proyecto</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Dirección Estratégica", "⚡ Impacto Esperado"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="tarjeta-clara">
                <h4>Objetivo General</h4>
                <p>Implementar una aplicación interactiva que centralice los algoritmos predictivos y permita a los gerentes simular escenarios de producción de forma autónoma.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="tarjeta-clara">
                <h4>Objetivos Específicos</h4>
                <ul>
                    <li>Diseñar una base de datos optimizada.</li>
                    <li>Garantizar una precisión del modelo superior al 92%.</li>
                    <li>Desplegar una interfaz intuitiva con Streamlit.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.info("💡 **Dato de valor:** Al usar software libre (Python y Streamlit), el costo de licenciamiento del proyecto se reduce a cero, haciendo la solución altamente escalable.")

elif choice == "🧠 Marco Conceptual":
    st.markdown('<h2 class="slide-header">Fundamentos Científicos</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tarjeta-clara">
            <h3>Machine Learning Aplicado</h3>
            <p>Para este proyecto se optó por un enfoque basado en <strong>Algoritmos de Ensamble</strong>. Estos combinan las predicciones de varios modelos base para mejorar la robustez general.</p>
            <p><strong>Criterios de Selección:</strong></p>
            <ul>
                <li>Resistencia al sobreajuste (Overfitting).</li>
                <li>Excelente manejo de variables de texto y números simultáneamente.</li>
                <li>Bajo consumo de memoria en servidores corporativos.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-clara" style="border-right: 4px solid #8b5cf6;">
            <h4>Modelos Comparados en el Estudio</h4>
            <p>1. <strong>XGBoost Regressor:</strong> Elegido final por su óptimo rendimiento.</p>
            <p>2. <strong>Random Forest:</strong> Descartado por archivos de peso excesivo.</p>
            <p>3. <strong>Regresión Lineal Múltiple:</strong> Descartada por baja precisión inicial (71%).</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🛠️ Metodología":
    st.markdown('<h2 class="slide-header">Metodología de Trabajo</h2>', unsafe_allow_html=True)
    
    with st.expander("🟢 FASE 1: Preparación de Datos"):
        st.write("Limpieza profunda de registros duplicados, tratamiento de datos vacíos mediante interpolación y normalización estadística de variables.")
        
    with st.expander("🔵 FASE 2: Modelado Analítico"):
        st.write("Entrenamiento del algoritmo XGBoost y ajuste fino de parámetros empleando técnicas de validación cruzada.")
        
    with st.expander("🟣 FASE 3: Despliegue de la Aplicación"):
        st.write("Empaquetamiento del código en Streamlit y publicación en la nube para el consumo en tiempo real de los usuarios finales.")

elif choice == "💻 Stack Tecnológico":
    st.markdown('<h2 class="slide-header">Herramientas Utilizadas</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="tarjeta-clara" style="text-align:center; border-top: 4px solid #059669;"><h4>Base de Código</h4><h2 style="color:#059669;">Python 3.11</h2><p>Procesamiento Robusto</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tarjeta-clara" style="text-align:center; border-top: 4px solid #3b82f6;"><h4>Interfaz Web</h4><h2 style="color:#3b82f6;">Streamlit</h2><p>Framework Reactivo</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="tarjeta-clara" style="text-align:center; border-top: 4px solid #8b5cf6;"><h4>Módulo Predictivo</h4><h2 style="color:#8b5cf6;">XGBoost</h2><p>Alta Precisión</p></div>', unsafe_allow_html=True)

elif choice == "📊 Resultados y Datos":
    st.markdown('<h2 class="slide-header">Métricas de Rendimiento</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Resultados Clave")
        st.metric(label="Precisión del Modelo (R²)", value="94.8%", delta="+5.3% de mejora")
        st.metric(label="Tiempo de Ejecución", value="0.25 seg", delta="-0.40 seg menos")
    with col2:
        # Generamos un gráfico limpio sobre el fondo blanco
        chart_data = pd.DataFrame(
            np.random.randn(25, 2) + [3, 3],
            columns=['Método Anterior', 'Sistema Propuesto']
        )
        st.line_chart(chart_data)
        st.caption("Gráfico analítico interactivo: Estabilización del error a lo largo del tiempo.")

elif choice == "💡 Conclusiones":
    st.markdown('<h2 class="slide-header">Conclusión e Impacto Final</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tarjeta-clara" style="border-left: 5px solid #3b82f6;">
            <h3>Conclusiones Principales</h3>
            <p>• La solución diseñada resolvió eficazmente el problema de predicción de recursos.</p>
            <p>• Las interfaces construidas en Streamlit reducen la curva de aprendizaje de los usuarios no técnicos de semanas a solo minutos.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-clara" style="border-top: 5px solid #8b5cf6; text-align:center; background:#f8fafc;">
            <h3>¿Tiene alguna pregunta?</h3>
            <br>
            <h4 style="color: #1e293b;">¡Muchas gracias por su valioso tiempo!</h4>
            <p style="font-size: 15px; margin-top:25px; color:#64748b;">Correo electrónico: contacto@universidad.edu</p>
        </div>
        """, unsafe_allow_html=True)
