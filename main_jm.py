import streamlit as st
import gspread
import json
import re
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import time

st.set_page_config(page_title="Narrador JM", page_icon="🎬")

# --------------------------- #
# Conectar à planilha
# --------------------------- #
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-Mn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# --------------------------- #
# Carregar memórias
# --------------------------- #
def carregar_memorias():
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[mary]"]
        mem_janio = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[jânio]"]
        mem_all = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[all]"]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

# --------------------------- #
# Carregar resumo salvo
# --------------------------- #
def carregar_resumo_salvo():
    try:
        aba = planilha.worksheet("perfil_jm")
        valores = aba.col_values(7)
        for val in reversed(valores[1:]):
            if val.strip():
                return val.strip()
        return ""
    except Exception as e:
        st.warning(f"Erro ao carregar resumo salvo: {e}")
        return ""

# --------------------------- #
# Salvar resumo na planilha
# --------------------------- #
def salvar_resumo(resumo):
    try:
        aba = planilha.worksheet("perfil_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        linha = ["", "", "", "", "", timestamp, resumo]
        aba.append_row(linha, value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

# --------------------------- #
# Construir prompt narrativo
# --------------------------- #
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-15:] if len(registros) > 15 else registros
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
    except:
        texto_ultimas = ""

    prompt = f"""
Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.

⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista.

🎭 Emoção oculta da cena: {emocao}

📖 Capítulo anterior:
{resumo if resumo else 'Nenhum resumo salvo.'}

### 🧠 Memórias:
Mary:
- {'\n- '.join(mem_mary) if mem_mary else 'Nenhuma.'}

Jânio:
- {'\n- '.join(mem_janio) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {'\n- '.join(mem_all) if mem_all else 'Nenhuma.'}

### 📖 Últimas interações:
{texto_ultimas}
"""
    return prompt.strip()

# --------------------------- #
# Salvar interação
# --------------------------- #
def salvar_interacao(role, content):
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

# --------------------------- #
# Carregar resumo ao iniciar
# --------------------------- #
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()

# --------------------------- #
# Título e resumo na tela principal
# --------------------------- #
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("""
---
#### 📖 Último resumo salvo:
""")
st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

# --------------------------- #
# Sidebar
# --------------------------- #
with st.sidebar:
    st.title("🎛️ Controle do Roteirista")

    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0)

    if provedor == "OpenRouter":
        modelos = {
            "💬 DeepSeek V3 (OpenRouter)": "deepseek/deepseek-chat-v3-0324",
            "🧠 GPT-4.1 (OpenRouter)": "openai/gpt-4.1"
        }
        chave_api = st.secrets["OPENROUTER_API_KEY"]
        url_api = "https://openrouter.ai/api/v1/chat/completions"
    else:
        modelos = {
            "🧠 Mixtral (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "🧠 Qwen (Together)": "qwen3-coder-480b-a35b-instruct"
        }
        chave_api = st.secrets["TOGETHER_API_KEY"]
        url_api = "https://api.together.xyz/v1/chat/completions"

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos.keys()), index=0)
    st.session_state.modelo_escolhido = modelos[modelo_nome]
    st.session_state.url_api = url_api
    st.session_state.chave_api = chave_api

    emocao = st.selectbox("🎭 Emoção oculta da cena", ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"], index=0)
    st.session_state.emocao_oculta = emocao

    if st.button("📝 Gerar resumo do capítulo"):
        try:
            aba = planilha.worksheet("interacoes_jm")
            registros = aba.get_all_records()
            ultimas = registros[-6:] if len(registros) > 6 else registros
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
            prompt_resumo = f"Resuma o seguinte trecho como um capítulo de novela:\n\n{texto}\n\nResumo:"

            resposta = requests.post(
                url_api,
                headers={
                    "Authorization": f"Bearer {chave_api}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": modelos[modelo_nome],
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85
                }
            )
            if resposta.status_code == 200:
                resumo = resposta.json()["choices"][0]["message"]["content"]
                st.session_state.resumo_capitulo = resumo
                salvar_resumo(resumo)
                st.success("Resumo gerado e salvo com sucesso!")
        except Exception as e:
            st.error(f"Erro ao resumir: {e}")
