# PESTAÑA 4: DESPLIEGUE OPERATIVO Y FORMULARIO PREDICTIVO (BLINDADO)
with tab4:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fa-solid fa-terminal title-icon'></i> Módulo de Inferencia Táctica Individual (Entorno Operativo)</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1.2, 0.8])
    
    with col_f1:
        st.markdown("""
        #### Simulación de Escenarios Operativos a Futuro
        Este entorno calcula automáticamente el impacto acumulado para garantizar la consistencia geométrica ante los modelos de Machine Learning.
        """)
        
        with st.form("formulario_principal_prospecto"):
            nombre_muni_futuro = st.text_input("Nombre del Territorio Evaluado", "Municipio Prospecto S-1")
            valores_ingresados = {}
            
            st.markdown("##### Ajuste de Variables del Escenario Territorial:")
            
            # Filtrar las columnas para no pedir el TOTAL_AFECTADOS manualmente (se calculará automático)
            variables_para_input = [v for v in numericas if v != 'TOTAL_AFECTADOS']
            
            sub_col1, sub_col2 = st.columns(2)
            for i, var in enumerate(variables_para_input):
                with sub_col1 if i % 2 == 0 else sub_col2:
                    valores_ingresados[var] = st.number_input(f"Cantidad de: {var}", min_value=0, value=0, step=1)
                    
            boton_predecir_tab = st.form_submit_button("Ejecutar Clasificación Estratégica con IA")

        if boton_predecir_tab:
            # 1. CÁLCULO AUTOMÁTICO DEL TOTAL para evitar contradicciones en el modelo
            if 'TOTAL_AFECTADOS' in numericas:
                valores_ingresados['TOTAL_AFECTADOS'] = sum(valores_ingresados.values())
            
            # 2. ORDENAMIENTO ESTRICTO DE COLUMNAS (Crucial para el StandardScaler)
            # Creamos un DataFrame asegurando que las columnas tengan el MISMO ORDEN EXACTO que el set de entrenamiento
            df_registro_futuro = pd.DataFrame([valores_ingresados])
            df_registro_futuro = df_registro_futuro.reindex(columns=numericas, fill_value=0)
            
            # 3. TRANSFORMACIÓN Y PREDICCIÓN
            registro_escalado = st.session_state.escalador_entrenado.transform(df_registro_futuro)
            prediccion_ia = st.session_state.red_entrenada.predict(registro_escalado)[0]
            
            # Colores dinámicos para los clústeres
            colores_cluster = {0: "#22C55E", 1: "#0EA5E9", 2: "#F59E0B", 3: "#EF4444"}
            color_resaltado = colores_cluster.get(prediccion_ia, "#3b82f6")
            
            with col_f2:
                st.markdown("<h4><i class='fa-solid fa-receipt title-icon'></i> Dictamen Generado por la Red</h4>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='insight-success' style='padding: 25px; font-size:16px; border-radius:10px; border-left: 5px solid {color_resaltado} !important;'>
                        <h3 style='color: #1E293B !important; margin-top:0;'><i class='fa-solid fa-circle-check' style='margin-right:8px; color: {color_resaltado};'></i>Análisis Exitoso</h3>
                        El territorio simulado <b>{nombre_muni_futuro}</b> ha sido asignado al: <br><br>
                        <span style='font-size: 34px; font-weight:800; color: {color_resaltado};'>CLÚSTER {prediccion_ia}</span><br><br>
                        <p style='font-size:13px; color:#475569;'><b>Firma del Vector:</b> Columnas ordenadas e indexadas correctamente con base en el histórico nacional.</p>
                    </div>
                """, unsafe_allow_html=True)
                
    with col_f2:
        if not boton_predecir_tab:
            st.markdown("<h4><i class='fa-solid fa-gear title-icon'></i> Conclusiones de Arquitectura de Producción</h4>", unsafe_allow_html=True)
            st.markdown("""
            <div class='insight-card' style='background-color: #F8FAFC; border-left-color: #6366F1;'>
                <h5 style='margin-top:0; color:#4f46e5;'><i class='fa-solid fa-bolt' style='margin-right:8px;'></i>Alineación de Matriz Activa</h5>
                <p>Esta versión fuerza la alineación estructural indexada mediante reindexación explícita sobre <code>df_registro_futuro</code>. Ingrese valores altos en cualquiera de las casillas para forzar el salto hacia clústeres de criticidad superior.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)
