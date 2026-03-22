# 📊 XGBoost Simulador de Demanda com Elasticidade de Preço e IA Local

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LM Studio](https://img.shields.io/badge/LM_Studio-0A0A0A?style=flat-square&logo=ai&logoColor=white)](https://lmstudio.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)

> Simulador interativo que utiliza **XGBoost** para prever vendas de produtos a partir de dados reais de varejo, permitindo simular o impacto do preço (elasticidade) e obter explicações automáticas via **IA local** (Hermes 3).

---

## 📌 Sobre o Projeto

Este projeto demonstra um pipeline completo de **modelagem preditiva** e **simulação de cenários** aplicado ao varejo.  Com base no dataset **BigMart Sales**, um modelo **XGBoost** é treinado para prever as vendas unitárias de produtos em diferentes lojas.  
A interface **Streamlit** permite que o usuário ajuste atributos do produto e da loja, visualize a previsão instantânea e explore a **curva de elasticidade** variando o preço. Além disso, um agente de IA local (Hermes 3) analisa o cenário atual e gera insights em português, explicando como as variáveis influenciam o resultado.

---

## 🛠️ Stack Principal

| **Linguagem** | **Bibliotecas de ML** | **IA Local** | **Visualização** |
|---------------|-----------------------|--------------|------------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![LM Studio](https://img.shields.io/badge/LM_Studio-0A0A0A?style=flat-square) ![Hermes 3](https://img.shields.io/badge/Hermes_3-3B-FFD700?style=flat-square) | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |

---

## 📊 Dataset

**Fonte:** [BigMart Sales Data – Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)  
**Descrição:**  
- 8.523 produtos em 10 lojas diferentes.  
- Atributos: peso, visibilidade, tipo de produto, preço (MRP), identificador da loja, ano de fundação, tamanho, localização e tipo de loja.  
- **Objetivo:** prever a quantidade vendida (`Item_Outlet_Sales`).

---

## 🧠 Como funciona

1. **Treinamento (`train.py`)**  
   - Baixa automaticamente o dataset via `kagglehub`.  
   - Pré‑processa variáveis categóricas com `LabelEncoder`.  
   - Treina um modelo **XGBoost Regressor** (200 árvores, learning rate 0.05).  
   - Salva o modelo e os encoders em arquivos `.pkl` (não versionados).

2. **Simulador Streamlit (`app.py`)**  
   - Carrega o modelo e os encoders.  
   - Interface com controles para todos os atributos (sliders, selects).  
   - A cada alteração, recalcula a previsão e exibe um gráfico de **elasticidade** (vendas × preço).  
   - Botão **“Analisar com IA”** envia os dados atuais para o **Hermes 3** (via LM Studio) e retorna uma análise em português.

3. **IA Explicativa (`ai_explainer.py`)**  
   - Conecta‑se ao servidor local do LM Studio (API compatível com OpenAI).  
   - Cria um prompt com os valores das variáveis e a previsão.  
   - Retorna um texto explicativo sobre os principais fatores e recomendações.

---

## 🚀 Como Executar

### Pré‑requisitos
- Python 3.9+  
- (Opcional) **LM Studio** com modelo **Hermes 3** carregado e servidor ativo (porta 1234)  
- Conta Kaggle (para download do dataset – gratuito)

### Passo a passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Gussnogue/xgboost-prever-vendas-elasticidade-ai-local.git
   cd xgboost-prever-vendas-elasticidade-ai-local
   ```
2. **Crie e ative um ambiente virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate   - Linux/Mac
   venv\Scripts\activate      - Windows
   ```
3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```
4. **Treine o modelo**
   ```bash
   python train.py
   ```
O dataset será baixado e o modelo será salvo em models/xgboost_model.pkl

5. **Execute a interface**
   ```bash
   streamlit run app.py
   ```
## 📄 Licença

MIT License – sinta‑se à vontade para usar, modificar e distribuir.

🔗 Dataset original: BigMart Sales Data – Kaggle
