import streamlit as st
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA (Estilo Amplio)
st.set_page_config(
    page_title="Presentación Impactante",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. INYECCIÓN DE CSS CON COLORES MODERNOS Y LLAMATIVOS
def aplicar_diseno_moderno():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@800&family=Inter:wght@400;600;700&display=swap');

        /* Fondo general de la app (Gris ultra claro/azulado muy limpio) */
        .stApp {
            background-color: #0b0f19; /* Fondo oscuro futurista */
        }
        
        /* DISEÑO DE LA BARRA LATERAL */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            border-right: 1px solid #374151;
        }
        [data-testid="stSidebar"] * {
            color: #f3f4f6 !important;
        }
        
        /* TÍTULOS PRINCIPALES CON DEGRADADO LLAMATIVO */
        .titulo-impactante {
            font-family: 'Cabinet Grotesk', 'Inter', sans-serif;
            font-size: 56px;
            font-weight: 800;
            background: linear-gradient(90deg, #ff007f 0%, #7928ca 50%, #00dfd8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            line-height: 1.1;
        }
        
        /* TÍTULOS DE LAS DIAPOSITIVAS */
        .slide-header {
            font-family: 'Cabinet Grotesk', 'Inter', sans-serif;
            font-size: 42px;
            font-weight: 800;
            color: #ffffff;
            border-bottom: 4px solid #ff007f; /* Línea de acento rosa neón */
            padding-bottom: 10px;
            margin-bottom: 35px;
            letter-spacing: -0.5px;
        }

        /* SUBTÍTULOS */
        .sub-title {
            font-family: 'Inter', sans-serif;
            color: #00dfd8; /* Turquesa neón */
            font-size: 24px;
            font-weight: 600;
            margin-top: -5px;
            margin-bottom: 30px;
        }

        /* TARJETAS DE CONTENIDO (Estilo Neumórfico Oscuro / Glassmorphism) */
        .tarjeta-contenido {
            background: rgba(31, 41, 55, 0.7);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }
        .tarjeta-contenido:hover {
            border-color: #7928ca; /* Brillo violeta al pasar el mouse */
        }
        
        .tarjeta-contenido h3, .tarjeta-contenido h4 {
            color: #00dfd8;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            margin-bottom: 15px;
        }

        /* TEXTO GENERAL */
        p, li, span {
            font-family: 'Inter', sans-serif;
            font-size: 18px;
            color: #d1d5db; /* Gris claro para excelente lectura sobre oscuro */
            line-height: 1.6;
        }
        
        strong {
            color: #ffffff !important;
        }

        /* Pestañas (Tabs) personalizadas */
        .stTabs [data-baseweb="tab"] {
            color: #9ca3af !important;
            font-size: 18px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #ff007f !important;
            border-bottom-color: #ff007f !important;
        }
    </style>
    """, unsafe_allow_html=True)

aplicar_diseno_moderno()

# 3. NAVEGADOR EN LA BARRA LATERAL
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center; color: #00dfd8;'>MENÚ</h2>", unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio(
        "Ir a la sección:",
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
    st.write("✨ *Diseño de Alta Fidelidad*")

# 4. CONTENIDO DE LAS DIAPOSITIVAS

if choice == "✨ Portada Principal":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        # Título gigante con degradado
        st.markdown('<h1 class="titulo-impactante">TRANSFORMACIÓN DIGITAL & MODELOS INTELIGENTES</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Reingeniería de Procesos mediante Inteligencia Artificial</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tarjeta-contenido" style="border-left: 6px solid #ff007f;">
            <table style="width:100%; border:none;">
                <tr style="background:none;">
                    <td style="padding:10px; border:none;"><strong>Presentado por:</strong> Equipo de Innovación</td>
                    <td style="padding:10px; border:none;"><strong>Especialidad:</strong> Ciencia de Datos</td>
                </tr>
                <tr style="background:none;">
                    <td style="padding:10px; border:none;"><strong>Evaluador:</strong> Comité Académico</td>
                    <td style="padding:10px; border:none;"><strong>Año:</strong> 2026</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🚀 Introducción":
    st.markdown('<h2 class="slide-header">El Desafío Actual</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="tarjeta-contenido">
            <h3>El Contexto Global</h3>
            <p>En un entorno hiperconectado, las organizaciones saturan sus capacidades operativas debido al procesamiento de datos manual y aislado. La falta de automatización reduce la competitividad global.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tarjeta-contenido">
            <h3>La Oportunidad</h3>
            <p>Implementar capas de analítica avanzada permite transformar la incertidumbre operativa en <strong>predicciones exactas en tiempo real</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-contenido" style="text-align: center; border-top: 4px solid #7928ca;">
            <h2 style="color: #ff007f; font-size: 60px; font-weight:800; margin:0;">40%</h2>
            <p style="font-size:16px;">De pérdida de eficiencia en empresas que no adoptan flujos automatizados de datos.</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🎯 Objetivos":
    st.markdown('<h2 class="slide-header">Metas del Proyecto</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Dirección Estratégica", "⚡ Impacto Esperado"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="tarjeta-contenido">
                <h4>Objetivo Principal</h4>
                <p>Desplegar una arquitectura de software inteligente que centralice, procese y prediga la demanda de infraestructura con un margen de error mínimo.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="tarjeta-contenido">
                <h4>Hitos Técnicos</h4>
                <ul>
                    <li>Construcción de pipeline ETL automatizado.</li>
                    <li>Modelado predictivo con redes neuronales recurrentes.</li>
                    <li>Despliegue de interfaz táctica e interactiva.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.info("💡 **Retorno de Inversión:** Se estima una reducción drástica de tiempos de respuesta y una optimización de recursos de hardware cercana al 25%.")

elif choice == "🧠 Marco Conceptual":
    st.markdown('<h2 class="slide-header">Fundamentos Tecnológicos</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tarjeta-contenido">
            <h3>¿Por qué Machine Learning Estructurado?</h3>
            <p>Utilizamos algoritmos de ensamble (Ensemble Learning) combinados con redes densas para asegurar estabilidad frente a datos ruidosos o atípicos.</p>
            <p><strong>Ventajas de este enfoque:</strong></p>
            <ul>
                <li>Alta interpretabilidad de variables.</li>
                <li>Entrenamiento veloz sin requerir supercómputo masivo.</li>
                <li>Fácil mantenimiento en producción.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-contenido" style="border-left: 4px solid #00dfd8;">
            <h4>Modelos Evaluados</h4>
            <p>1. XGBoost (Ganador por precisión)</p>
            <p>2. Random Forest Regressor</p>
            <p>3. LSTM (Redes de memoria a largo/corto plazo)</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🛠️ Metodología":
    st.markdown('<h2 class="slide-header">Fases del Desarrollo</h2>', unsafe_allow_html=True)
    
    # Acordeones elegantes
    with st.expander("🟣 FASE 1: Ingesta y Limpieza"):
        st.write("Filtrado de datos nulos, eliminación de outliers y normalización de escalas estadísticas mediante pipelines reproducibles.")
        
    with st.expander("🔵 FASE 2: Entrenamiento Predictivo"):
        st.write("Búsqueda de hiperparámetros óptimos empleando Grid Search y validación cruzada de 5 pliegues para evitar el sobreajuste.")
        
    with st.expander("🟢 FASE 3: Interfaz e Integración"):
        st.write("Conexión del modelo serializado con el frontend interactivo en Streamlit, agilizando el consumo de analíticas.")

elif choice == "💻 Stack Tecnológico":
    st.markdown('<h2 class="slide-header">Herramientas y Ecosistema</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="tarjeta-contenido" style="text-align:center;"><h4>Core</h4><h2 style="color:#ff007f;">Python</h2><p>Versión 3.11</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tarjeta-contenido" style="text-align:center;"><h4>Frontend</h4><h2 style="color:#7928ca;">Streamlit</h2><p>Reactive UI</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="tarjeta-contenido" style="text-align:center;"><h4>Modelado</h4><h2 style="color:#00dfd8;">Scikit-Learn</h2><p>& XGBoost</p></div>', unsafe_allow_html=True)

elif choice == "📊 Resultados y Datos":
    st.markdown('<h2 class="slide-header">Resultados Clínicos del Modelo</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Métricas Clave")
        st.metric(label="Precisión General (R²)", value="96.5%", delta="+4.2% vs Base")
        st.metric(label="Latencia de Inferencia", value="12 ms", delta="-8ms óptimo")
    with col2:
        # Gráfico dinámico adaptado a la estética oscura
        chart_data = pd.DataFrame(
            np.random.randn(30, 2) + [2, 2],
            columns=['Rendimiento Tradicional', 'Nuestro Algoritmo']
        )
        st.line_chart(chart_data)
        st.caption("Gráfico interactivo: Comparación temporal del error remanente.")

elif choice == "💡 Conclusiones":
    st.markdown('<h2 class="slide-header">Cierre e Impacto</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tarjeta-contenido" style="border-top: 4px solid #00dfd8;">
            <h3>Conclusiones</h3>
            <p>• La solución desarrollada cumple con los estándares exigidos de velocidad y precisión.</p>
            <p>• Streamlit demostró ser una alternativa sumamente viable y rápida para construir prototipos de datos sin perder calidad visual.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tarjeta-contenido" style="border-top: 4px solid #ff007f; text-align:center;">
            <h3>¿Preguntas o Comentarios?</h3>
            <br>
            <h4 style="color: #ffffff;">¡Muchas gracias por su atención!</h4>
            <p style="font-size: 15px; margin-top:20px;">Contacto: desarrollo@innovacion.com</p>
        </div>
        """, unsafe_allow_html=True)
