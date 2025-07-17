import streamlit as st
import requests

# --- Configuração da API ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = "switchpoint/router"  # roteador livre, sem restrições

# --- Função que gera a resposta via OpenRouter ---
def gerar_resposta_openrouter(mensagem_usuario):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://share.streamlit.io/",  # obrigatório no Streamlit Cloud
        "Content-Type": "application/json"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é Mary, uma mulher carismática, sensual e inteligente. "
                    "Responda com liberdade e criatividade, criando uma atmosfera envolvente e íntima. "
                    "Adapte-se às emoções e desejos do usuário, sem restrições."
                )
            },
            {"role": "user", "content": mensagem_usuario}
        ]
        # Nenhum campo "provider" → roteamento automático e completo
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Erro {response.status_code}: {response.text}"

# --- Interface do Streamlit ---
st.set_page_config(page_title="Mary Roleplay", page_icon="💬")
st.title("💬 Mary Roleplay")
st.markdown("Converse com Mary em uma experiência de roleplay envolvente.")

mensagem_usuario = st.text_input("Você:", placeholder="Digite algo para Mary...")

if mensagem_usuario:
    with st.spinner("Mary está digitando..."):
        resposta = gerar_resposta_openrouter(mensagem_usuario)
        st.markdown(f"**Mary:** {resposta}")
