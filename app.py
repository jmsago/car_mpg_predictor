import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ======================
# 🌟 CONFIGURAÇÕES
# ======================
st.set_page_config(page_title="🚗 Car Performance Predictor", page_icon="🚘", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .title {
            text-align: center;
            color: #90e0ef;
            font-size: 2.2em;
            font-weight: bold;
        }
        .metric {
            text-align: center;
            font-size: 1.2em;
        }
        .stButton>button {
            background: linear-gradient(to right, #00b4d8, #0077b6);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #0077b6, #023e8a);
        }
        .footer {
            text-align: center;
            color: #adb5bd;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# 📦 CARREGAR MODELO
# ======================
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

# ======================
# 🎛️ SIDEBAR
# ======================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/743/743007.png", width=100)
st.sidebar.title("⚙️ Parâmetros do Carro")

horsepower = st.sidebar.slider("Potência (HP)", 40, 250, 100)
weight = st.sidebar.slider("Peso (lbs)", 1500, 5000, 3000)
acceleration = st.sidebar.slider("Aceleração (0-60 mph)", 8, 25, 15)
model_year = st.sidebar.slider("Ano do Modelo", 70, 82, 76)

st.sidebar.markdown("---")
theme_choice = st.sidebar.radio("🎨 Tema visual", ["Escuro", "Claro"])
if theme_choice == "Claro":
    st.markdown("""
    <style>.stApp { background: #f8f9fa; color: black; }</style>
    """, unsafe_allow_html=True)

# ======================
# 🧮 PREDIÇÃO
# ======================
st.markdown("<div class='title'>🚗 Previsor de Consumo de Combustível</div>", unsafe_allow_html=True)
st.markdown("Use os controles na lateral para ajustar os valores do carro e veja os resultados abaixo 👇")

input_data = np.array([[horsepower, weight, acceleration, model_year]])
prediction = model.predict(input_data)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("MPG Estimado (milhas por galão)", f"{prediction:.2f}")
    if prediction > 30:
        st.success("💚 Excelente eficiência de combustível!")
    elif prediction > 20:
        st.info("💛 Eficiência média.")
    else:
        st.error("❤️‍🔥 Alto consumo de combustível!")

    st.markdown(f"""
    ### Explicação:
    Este carro, com **{horsepower} HP**, peso de **{weight} lbs**, aceleração **{acceleration}s** e modelo de **{model_year}**,  
    foi avaliado pelo modelo e deve consumir cerca de **{prediction:.1f} MPG**.
    """)

with col2:
    st.markdown("### ⚖️ Comparativo com média")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(x=["Seu carro", "Média (Global)"], y=[prediction, 23.5], palette=["#00b4d8", "#90e0ef"])
    ax.set_ylabel("Milhas por Galão (MPG)")
    st.pyplot(fig)

# ======================
# 📊 DASHBOARD
# ======================
st.markdown("### 📈 Importância das Variáveis no Modelo")
st.image("feature_importance.png", width=600)
