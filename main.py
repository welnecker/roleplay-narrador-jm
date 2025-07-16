import streamlit as st
import requests
import os

# Configurações iniciais
st.set_page_config(page_title="Roleplay Mary", page_icon="🧠")
st.title("🎭 Roleplay com Mary via OpenRouter")

# Entrada do usuário
user_input = st.text_area("Digite sua mensagem:", height=150)

# Botão para enviar
if st.button("Enviar"):
    if not user_input.strip():
        st.warning("Digite algo para iniciar a conversa.")
    else:
        with st.spinner("Gerando resposta..."):
            resposta = gerar_resposta_openrouter(user_input)
            st.markdown("**Mary:** " + resposta)

# Função para gerar resposta via OpenRouter
def gerar_resposta_openrouter(prompt, modelo="gryphe/mythomax-l2-13b"):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "❌ Chave de API não encontrada."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://welnecker-roleplay-mary-streamlit.streamlit.app",  # ou use share.streamlit.io se for local
        "X-Title": "Roleplay Mary",
    }

    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Você é Mary, uma mulher madura, sensual e intensa. Responda com emoções e profundidade, sempre em tom roleplay."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ Erro OpenRouter: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"❌ Erro na requisição: {e}"
