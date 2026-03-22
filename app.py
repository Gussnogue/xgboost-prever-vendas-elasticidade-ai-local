import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ai_explainer import generate_insight

st.set_page_config(page_title="Simulador de Demanda", layout="wide")
st.title("📊 Simulador de Demanda com Elasticidade de Preço")
st.markdown("Ajuste os atributos do produto e da loja para prever as vendas (XGBoost)")

# Carregar modelo e encoders
@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_model.pkl")
    le_dict = joblib.load("models/label_encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, le_dict, feature_names

model, le_dict, feature_names = load_model()

# --- Interface ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Atributos do Produto")
    item_weight = st.slider("Peso do item (kg)", 0.0, 20.0, 10.0, 0.1)
    item_fat = st.selectbox("Teor de gordura", ["Low Fat", "Regular"])
    item_visibility = st.slider("Visibilidade (% do espaço)", 0.0, 0.5, 0.1, 0.01)
    item_type = st.selectbox("Tipo de produto", 
        ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods',
         'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks',
         'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'])
    item_mrp = st.slider("Preço máximo de varejo (R$)", 0.0, 300.0, 100.0, 5.0)

with col2:
    st.subheader("Atributos da Loja")
    outlet_id = st.selectbox("Identificador da loja", 
        ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
    outlet_year = st.slider("Ano de estabelecimento", 1985, 2010, 2005)
    outlet_size = st.selectbox("Tamanho da loja", ["Small", "Medium", "High"])
    outlet_location = st.selectbox("Localização", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Tipo de loja", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

# Codificar variáveis categóricas
try:
    fat_encoded = le_dict["Item_Fat_Content"].transform([item_fat])[0]
except ValueError:
    # fallback: mapear manualmente (caso o encoder não tenha o valor)
    fat_encoded = 0 if item_fat == "Low Fat" else 1

try:
    type_encoded = le_dict["Item_Type"].transform([item_type])[0]
except:
    type_encoded = 0  # fallback

try:
    outlet_enc = le_dict["Outlet_Identifier"].transform([outlet_id])[0]
except:
    outlet_enc = 0

try:
    size_enc = le_dict["Outlet_Size"].transform([outlet_size])[0]
except:
    size_enc = 0

try:
    loc_enc = le_dict["Outlet_Location_Type"].transform([outlet_location])[0]
except:
    loc_enc = 0

try:
    type_out_enc = le_dict["Outlet_Type"].transform([outlet_type])[0]
except:
    type_out_enc = 0

# Montar array de entrada na ordem correta das features
input_dict = {
    "Item_Weight": item_weight,
    "Item_Fat_Content": fat_encoded,
    "Item_Visibility": item_visibility,
    "Item_Type": type_encoded,
    "Item_MRP": item_mrp,
    "Outlet_Identifier": outlet_enc,
    "Outlet_Establishment_Year": outlet_year,
    "Outlet_Size": size_enc,
    "Outlet_Location_Type": loc_enc,
    "Outlet_Type": type_out_enc,
}
input_df = pd.DataFrame([input_dict])[feature_names]
pred = model.predict(input_df)[0]

# Exibir previsão
st.markdown("---")
st.subheader("📈 Previsão de Vendas")
st.metric("Vendas previstas", f"R$ {pred:.2f}")

# Botão para gerar insight com IA
if st.button("🤖 Analisar com IA (Hermes 3)"):
    with st.spinner("Consultando IA local..."):
        insight = generate_insight(input_df.values[0], pred, feature_names)
    st.markdown("### 💡 Análise da IA")
    st.markdown(insight)

# Gráfico de elasticidade: ajuste de preço (Item_MRP)
st.markdown("---")
st.subheader("📉 Elasticidade – Preço vs. Vendas")
price_range = np.linspace(0, max(300, item_mrp*1.5), 50)
preds_price = []
for p in price_range:
    temp = input_dict.copy()
    temp["Item_MRP"] = p
    temp_df = pd.DataFrame([temp])[feature_names]
    preds_price.append(model.predict(temp_df)[0])
fig = px.line(x=price_range, y=preds_price, title="Vendas em função do Preço")
fig.update_layout(xaxis_title="Preço (R$)", yaxis_title="Vendas (R$)")
st.plotly_chart(fig, use_container_width=True)

