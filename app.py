import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# IMPORTS PARA LA CONTINUACIÓN SUPERVISADA (MODELO HÍBRIDO)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ESTILO PREMIUM PARA GRAFICAS
def aplicar_estilo_premium(fig):
    fig.update_layout(
        paper_bgcolor="#EAF4FF",
        plot_bgcolor="#F4F9FF",
        font=dict(color="#0F172A", size=14),
        title=dict(font=dict(size=20, color="#0F172A")),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# FUNCIÓN ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    m = int(0.1 * n) if int(0.1 * n) > 1 else 1
    
    np.random.seed(42)
    vecinos = NearestNeighbors(n_neighbors=2).fit(X)

    puntos_aleatorios = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (m, d))
    dist_aleatoria, _ = vecinos.kneighbors(puntos_aleatorios, n_neighbors=1)

    indices = np.random.choice(n, m, replace=False)
    dist_real, _ = vecinos.kneighbors(X[indices], n_neighbors=2)

    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])
    return U / (U + W) if (U + W) > 0 else 0.5

# CONFIGURACIÓN DE LA APP
st.set_page_config(page_title="Modelo Híbrido - Orden Público", layout="wide", initial_sidebar_state="collapsed")

# CSS Avanzado
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Helvetica Neue', Arial, sans-serif; }
    .slide-container { background-color: #FFFFFF; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #E2E8F0; }
    .slide-title { color: #0F172A; font-size: 36px; font-weight: 700; margin-bottom: 5px; }
    .slide-subtitle { color: #64748B; font-size: 18px; margin-bottom: 25px; }
    div.stButton > button { background-color: #E0F2FE !important; color: #0369A1 !important; border: 1px solid #BAE6FD !important; border-radius: 8px !important; font-weight: 700 !important; }
    div.stButton > button[kind="primary"] { background-color: #0284C7 !important; color: #FFFFFF !important; border: 1px solid #0284C7 !important; }
    .insight-card { background-color: #F1F5F9; border-left: 5px solid #38BDF8; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
    .insight-critical { background-color: #FEF2F2; border-left: 5px solid #DC2626; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# Navegación
cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]
for i, nombre in enumerate(nombres_diapo):
    tipo = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# Carga automática de datos
def cargar_datos():
    for archivo in os.listdir('.'):
        if ("afectacion" in archivo.lower() or "fuerza" in archivo.lower() or "publica" in archivo.lower()) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            return pd.read_csv(archivo) if archivo.endswith('.csv') else pd.read_excel(archivo), archivo
    return None, None

df_original, nombre_archivo = cargar_datos()

# DIAPOSITIVAS 1 A 4 (Mantenidas simplificadas para control del flujo)
if st.session_state.diapositiva == 1:
    st.markdown("<div class='slide-container' style='text-align: center;'><div class='slide-title'>Análisis de Clústeres (K-Means) & Redes Neuronales (MLP)</div><div class='slide-subtitle'>Modelo Híbrido para la Evaluación del Riesgo en Municipios</div><p><b>Autor:</b> Miguel Angel Garatejo</p></div>", unsafe_allow_html=True)
    if st.button("Empezar", type="primary", use_container_width=True): ir_a_diapositiva(5)

elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Resultados del Modelo Híbrido</div><div class='slide-subtitle'>La unión del Aprendizaje No Supervisado y Supervisado</div>", unsafe_allow_html=True)
    
    if df_original is None:
        st.error("Por favor, coloca el archivo de datos en la carpeta del script.")
        st.stop()

    # --- PIPELINE DE DATOS ORIGINAL ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if 'NOMBRE_FUERZA' in df_original.columns else pd.DataFrame(index=pivot_accion.index)
    total_muni = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    
    datos = total_muni.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
    numericas = [c for c in datos.columns if c not in index_cols]

    # ESCALAMIENTO PARA K-MEANS
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(datos[numericas])

    # --- SECCIONES EN PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 FASE 1: Clustering (No Supervisado)", "🧠 FASE 2: Red Neuronal (Supervisado)", "🔮 Simulador Predictivo"])

    with tab1:
        st.subheader("Análisis de Agrupamiento Territorial")
        
        # Métricas de validación del clustering
        hopkins_v = calcular_hopkins(X_scaled_km)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Estadístico de Hopkins (Tendencia de Clúster)", f"{hopkins_v:.3f}")
        
        # Ejecución de K-Means (Fijamos K=4 para estabilidad del modelo híbrido)
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        datos['Cluster'] = kmeans.fit_predict(X_scaled_km)
        
        sil_v = silhouette_score(X_scaled_km, datos['Cluster'])
        col_m2.metric("Coeficiente de Silueta Promedio", f"{sil_v:.3f}")

        # Gráfica 1: Curva del Codo e Inercia
        inercias = []
        K_rango = range(1, 8)
        for k in K_rango:
            km_test = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled_km)
            inercias.append(km_test.inertia_)
        
        fig_codo = px.line(x=list(K_rango), y=inercias, markers=True, title="Método del Codo (Optimización de K)", labels={'x': 'Número de Clústeres (K)', 'y': 'Inercia (WSS)'})
        st.plotly_chart(aplicar_estilo_premium(fig_codo), use_container_width=True)

        # Gráfica 2: Proyección PCA 3D (Tus datos reales estructurados geométricamente)
        pca = PCA(n_components=3)
        componentes = pca.fit_transform(X_scaled_km)
        df_pca = pd.DataFrame(componentes, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = datos['Cluster'].astype(str)
        df_pca['Municipio'] = datos['MUNICIPIO']
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Municipio', title='Segmentación Territorial en el Espacio Reducido (PCA 3D)', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(aplicar_estilo_premium(fig_3d), use_container_width=True)

    with tab2:
        st.subheader("Entrenamiento y Generalización de la Red Neuronal (MLP)")
        st.write("Tomamos las etiquetas del K-Means como variables objetivo ($y$) y entrenamos al Perceptrón Multicapa.")

        # --- FLUJO SUPERVISADO SEGURO (Previene Data Leakage) ---
        X = datos[numericas]
        y = datos['Cluster']
        
        # 1. División Train / Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 2. Escalamiento Aislado
        scaler_mlp = StandardScaler()
        X_train_scaled = scaler_mlp.fit_transform(X_train)
        X_test_scaled = scaler_mlp.transform(X_test)
        
        # 3. Entrenamiento MLP
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=42)
        with st.spinner("Entrenando Red Neuronal..."):
            mlp.fit(X_train_scaled, y_train)
        
        y_pred = mlp.predict(X_test_scaled)
        acc = mlp.score(X_test_scaled, y_test)
        
        st.metric("Precisión (Accuracy) de la IA en Datos de Prueba", f"{acc*100:.2f}%")

        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("**Matriz de Confusión (Evaluación Cruzada):**")
            cm = confusion_matrix(y_test, y_pred)
            nombres_clusters = [f"Clúster {i}" for i in range(4)]
            fig_cm = px.imshow(cm, text_auto=True, x=nombres_clusters, y=nombres_clusters, labels=dict(x="Predicho por Red Neuronal", y="Asignado por K-Means"), color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            st.markdown("**Métricas Académicas de Clasificación:**")
            rep = classification_report(y_test, y_pred, output_dict=True)
            df_rep = pd.DataFrame(rep).transpose().round(3)
            st.dataframe(df_rep, use_container_width=True)

    with tab3:
        st.subheader("Simulación Operativa en Tiempo Real")
        st.write("Inserta los datos de un municipio simulado y la Red Neuronal calculará su riesgo inmediatamente sin recalcular el clustering.")
        
        c_ins = st.columns(3)
        user_features = {}
        for idx, col in enumerate(numericas[:9]): # Limitamos a los primeros 9 para no saturar la vista
            with c_ins[idx % 3]:
                user_features[col] = st.number_input(f"Cantidad de: {col}", min_value=0, value=2)
        
        # Rellenar el resto si existen más de 9 variables
        for col in numericas[9:]:
            user_features[col] = 0
            
        if st.button("Clasificar con Inteligencia Artificial", type="primary"):
            df_muni_nuevo = pd.DataFrame([user_features])[numericas]
            scaled_nuevo = scaler_mlp.transform(df_muni_nuevo)
            clase_ia = mlp.predict(scaled_nuevo)[0]
            
            colores_alertas = {0: "insight-card", 1: "insight-card", 2: "insight-critical", 3: "insight-critical"}
            estilo = colores_alertas.get(clase_ia, "insight-card")
            
            st.markdown(f"""
            <div class='{estilo}'>
                <h4>🚨 Predicción del Sistema Híbrido:</h4>
                <p>El municipio simulado pertenece al <b>Clúster {clase_ia}</b>.</p>
                <p><i>La Red Neuronal clasificó este perfil criminal de manera autónoma con base en los patrones geométricos aprendidos.</i></p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("<div class='slide-container'><h3>Fin del Proceso</h3><p>Usa la barra superior para navegar.</p></div>", unsafe_allow_html=True)
