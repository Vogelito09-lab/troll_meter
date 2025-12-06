import streamlit as st
import pandas as pd
from src.bayes_logic import TrollBrain

# --- Configuracion de pagina ---
st.set_page_config(page_title="Troll-O-Meter", page_icon=" ")

#---Instanciar el cerebro---
# Usamos session:state para que el objeto persista mientras usemos la app
if 'cerebro' not in st.session_state:
    st.session_state['cerebro'] = TrollBrain()

brain = st.session_state['cerebro']

# --- Interfaz gráfica ---
st.title("Troll-O-Meter 3000")
st.markdown("Detector de toxicidad usando **Naive Bayes Simplificado**.")

# --- Barra Lateral: Entrenamiento ---
with st.sidebar:
    st.header(" Zona de Aprendizaje")
    st.info("La IA nace sin saber. ¡Enséñale!")

    nuevo_txt = st.text_input("Frase de ejemplo:")
    tipo = st.radio("Etiqueta:", ["toxico", "pro"])

    if st.button("Entrenar IA"):
        if nuevo_txt:
            brain.aprender(nuevo_txt, tipo)
            st.success(f"Aprendido: '{nuevo_txt}' es {tipo}")
        else:
            st.warning("Escribe algo primero.")

    st.divider()
    st.caption("Estado de la Memoria:")
    st.text(f"Palabras Tóxicas: {len(brain.vocab_toxico)}")
    st.text(f"Palabras Amigables: {len(brain.vocab_pro)}")

# -- Área principal: Prediccion ---
st.subheader("Analizar Chat")
mensaje = st.text_input("Escribe un mensaje para moderar:", placeholder="Ej: gg wp equipo")

if st.button("Analizar Mensaje"):
    if not mensaje:
        st.warning("Escribe un mensaje.")
    else:
        s_tox, s_pro, explicacion = brain.predecir(mensaje)
        total = s_tox + s_pro

        if total == 0:
            st.info(" No conozco estas palabras. Entrename primero.")
        else:
            col1, col2 = st.columns(2)

            prob_tox = s_tox / total
            prob_pro = s_pro / total

            with col1:
                st.metric("Nivel Tóxico ", f"{prob_tox:.1%}")
            with col2:
                st.metric("Nivel Amigable ", f"{prob_pro:.1%}")

# Gráfico de Barras
            df = pd.DataFrame({
            'Categoría': ['Tóxico', 'Amigable'],
            'Puntos': [s_tox, s_pro]
            })
            st.bar_chart(df, x='Categoría', y='Puntos', color='Categoría')

# Explicabilidad (White Box AI)
            with st.expander("¿Por qué dice esto la IA?"):
                st.write("Palabras clave detectadas:")
                st.write(explicacion)

                st.divider()
                if prob_tox > 0.5:
                    st.error("VEREDICTO FINAL: EL MENSAJE ES TÓXICO")
                else:
                    st.success("VEREDICTO FINAL: EL MENSAJE ES AMIGABLE")