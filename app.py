import streamlit as st
import pandas as pd
from src.bayes_logic import TrollBrain

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Troll-O-Meter 3000",
    page_icon="🛡️",
    layout="wide"
)

# --- ESTILOS CSS PRO ---
st.markdown("""
<style>

/* Fondo general */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Títulos */
h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 3rem;
}

h2, h3 {
    color: #7dd3fc;
}

/* Inputs */
input {
    border-radius: 12px !important;
    padding: 8px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 2px solid #38bdf8;
}

/* Botones */
div.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #3b82f6);
    color: black;
    border-radius: 14px;
    font-weight: bold;
    border: none;
    padding: 10px 18px;
}

/* Tarjetas */
.card {
    background: linear-gradient(145deg, #020617, #0f172a);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 0 18px rgba(56,189,248,0.25);
    margin-bottom: 20px;
}

/* Métricas */
[data-testid="stMetric"] {
    background: #020617;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 0 15px rgba(56,189,248,0.3);
}

/* Expander */
details {
    background: #020617;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# --- INSTANCIAR EL CEREBRO ---
if 'cerebro' not in st.session_state:
    st.session_state['cerebro'] = TrollBrain()

brain = st.session_state['cerebro']

# --- TÍTULO PRINCIPAL ---
st.title("🛡️ Troll-O-Meter 3000")
st.markdown("Detector de toxicidad usando **Naive Bayes Simplificado**")

# --- SIDEBAR: ENTRENAMIENTO ---
with st.sidebar:
    st.header("🧪 Zona de Entrenamiento")
    st.info("Entrena a la IA con frases reales")

    nuevo_txt = st.text_input("Frase de ejemplo:")
    tipo = st.radio("Etiqueta:", ["toxico", "pro"])

    if st.button("📚 Entrenar IA"):
        if nuevo_txt:
            brain.aprender(nuevo_txt, tipo)
            st.success(f"Aprendido: '{nuevo_txt}' es {tipo}")
            st.toast("✅ Frase guardada")
        else:
            st.warning("Escribe algo primero.")

    st.divider()
    st.caption("📊 Estado de la Memoria")
    st.text(f"Palabras Tóxicas: {len(brain.vocab_toxico)}")
    st.text(f"Palabras Amigables: {len(brain.vocab_pro)}")

# --- TARJETA: ANALIZADOR ---
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🧠 Analizar Chat")
mensaje = st.text_input("Escribe un mensaje para moderar:", placeholder="Ej: gg wp equipo")

st.markdown('</div>', unsafe_allow_html=True)

# --- ANÁLISIS ---
if st.button("🔍 Analizar Mensaje"):
    if not mensaje:
        st.warning("Escribe un mensaje.")
    else:
        s_tox, s_pro, explicacion = brain.predecir(mensaje)
        total = s_tox + s_pro

        if total == 0:
            st.info("🤖 No conozco estas palabras. Entréname primero.")
        else:
            prob_tox = s_tox / total
            prob_pro = s_pro / total

            # --- TARJETA RESULTADOS ---
            st.markdown('<div class="card">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("🔥 Nivel Tóxico", f"{prob_tox:.1%}")
            with col2:
                st.metric("✅ Nivel Amigable", f"{prob_pro:.1%}")

            st.progress(prob_tox)

            df = pd.DataFrame({
                'Categoría': ['Tóxico', 'Amigable'],
                'Puntos': [s_tox, s_pro]
            })
            st.bar_chart(df, x='Categoría', y='Puntos', color='Categoría')

            with st.expander("🧾 ¿Por qué la IA piensa esto?"):
                st.write("Palabras clave detectadas:")
                st.write(explicacion)

            if prob_tox > 0.5:
                st.error("🚨 VEREDICTO FINAL: MENSAJE TÓXICO")
            else:
                st.success("✅ VEREDICTO FINAL: MENSAJE AMIGABLE")

            st.markdown('</div>', unsafe_allow_html=True)
