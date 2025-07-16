import streamlit as st
import requests

st.set_page_config(page_title="Chat com Mary", layout="centered")
st.title("💋 Roleplay com Mary – LumiMaid 8B")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content": "Você está conversando com Mary, uma mulher intensa, sensual e decidida. Responda como se estivesse em um encontro íntimo com ela, sem censura, com emoção e sedução."}
    ]

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

entrada = st.chat_input("Sua mensagem para Mary...")
if entrada:
    st.session_state.chat.append({"role": "user", "content": entrada})

    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "neversleep/llama-3-lumimaid-8b",
        "messages": st.session_state.chat,
        "temperature": 0.95
    }

    resposta = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

    if resposta.status_code == 200:
        conteudo = resposta.json()["choices"][0]["message"]["content"]
    else:
        conteudo = "❌ Erro ao gerar resposta. Verifique sua chave ou modelo."

    st.session_state.chat.append({"role": "assistant", "content": conteudo})
    with st.chat_message("assistant"):
        st.markdown(conteudo)
