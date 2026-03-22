import os
import requests
from dotenv import load_dotenv

load_dotenv()

LM_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
LM_MODEL = os.getenv("LM_MODEL", "hermes-3-llama-3.2-3b")

def generate_insight(feature_values, prediction, feature_names):
    """Envia para IA uma descrição do cenário e retorna análise em português."""
    # Construir descrição das variáveis
    desc = f"Previsão de vendas: R$ {prediction:.2f}\n\nVariáveis:\n"
    for name, val in zip(feature_names, feature_values):
        desc += f"- {name}: {val}\n"

    prompt = f"""
Você é um analista de negócios especializado em varejo. Analise os dados abaixo e forneça uma explicação em português:

{desc}

Explique como as variáveis influenciam a previsão, sugerindo ações para aumentar as vendas. Seja direto e use termos de negócio.
"""
    payload = {
        "model": LM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400
    }
    try:
        resp = requests.post(LM_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao consultar IA: {e}"
    
    