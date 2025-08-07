import streamlit as st
import gspread
import json
import re
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

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
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# --------------------------- #
# Carregar perfis e memórias
# --------------------------- #
def carregar_perfis():
    try:
        aba = planilha.worksheet("perfil_jm")
        dados = aba.get_all_values()
        resumo_mary, resumo_janio = "", ""
        for linha in dados[1:]:
            if linha[0].strip().lower() == "mary" and len(linha) >= 7:
                resumo_mary = linha[6].strip()
            if linha[0].strip().lower() == "jânio" and len(linha) >= 7:
                resumo_janio = linha[6].strip()
        return resumo_mary, resumo_janio
    except Exception as e:
        st.error(f"Erro ao carregar perfis: {e}")
        return "", ""

def carregar_memorias():
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r["conteudo"] for r in registros if r["tipo"].strip().lower() == "[mary]"]
        mem_janio = [r["conteudo"] for r in registros if r["tipo"].strip().lower() == "[jânio]"]
        mem_all = [r["conteudo"] for r in registros if r["tipo"].strip().lower() == "[all]"]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

# --------------------------- #
# Construir prompt narrativo
# --------------------------- #
def construir_prompt_com_narrador():
    resumo_mary, resumo_janio = carregar_perfis()
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")

    prompt = f"""
Você é o narrador de uma história em construção. Os protagonistas são:

🔴 Mary — {resumo_mary}
🔵 Jânio — {resumo_janio}

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.

⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista.

🎭 Emoção oculta da cena: {emocao}

### 🧠 Memórias:
Mary:
- {'\n- '.join(mem_mary) if mem_mary else 'Nenhuma.'}

Jânio:
- {'\n- '.join(mem_janio) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {'\n- '.join(mem_all) if mem_all else 'Nenhuma.'}
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
# Sidebar (modelos, emoção)
# --------------------------- #
with st.sidebar:
    st.title("🎛️ Controle do Roteirista")

    modelos = {
        "💬 DeepSeek V3 (OpenRouter)": "deepseek/deepseek-chat-v3-0324",
        "🧠 GPT-4.1 (OpenRouter)": "openai/gpt-4.1"
    }
    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos.keys()), index=0)
    st.session_state.modelo_escolhido = modelos[modelo_nome]

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
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85
                }
            )
            if resposta.status_code == 200:
                resumo = resposta.json()["choices"][0]["message"]["content"]
                st.success("Resumo gerado com sucesso!")
                st.text_area("📖 Capítulo anterior", resumo, height=200)
            else:
                st.error("Erro ao gerar resumo.")
        except Exception as e:
            st.error(f"Erro ao resumir: {e}")

# --------------------------- #
# Interface principal
# --------------------------- #
st.title("🎬 Narrador JM")
st.markdown("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")

entrada = st.chat_input("Ex: Mary acorda atrasada. Jânio está malhando na academia...")
if "historico" not in st.session_state:
    st.session_state.historico = []

if entrada:
    st.chat_message("user").markdown(entrada)
    salvar_interacao("user", entrada)
    st.session_state.historico.append({"role": "user", "content": entrada})

    prompt = construir_prompt_com_narrador()
    mensagens = [{"role": "system", "content": prompt}] + st.session_state.historico

    st.chat_message("assistant").markdown("*(Narrando...)*")
    try:
        resposta = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": st.session_state.modelo_escolhido,
                "messages": mensagens,
                "max_tokens": 800,
                "temperature": 0.85
            },
            timeout=120
        )
        if resposta.status_code == 200:
            conteudo = resposta.json()["choices"][0]["message"]["content"]
            salvar_interacao("assistant", conteudo)
            st.session_state.historico.append({"role": "assistant", "content": conteudo})
            st.chat_message("assistant").markdown(conteudo)
        else:
            st.error("Erro ao gerar resposta da IA.")
    except Exception as e:
        st.error(f"Erro de conexão: {e}")

# Exibir histórico
for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
