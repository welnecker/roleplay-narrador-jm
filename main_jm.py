import streamlit as st
import gspread
import json
import re
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import time
import numpy as np
from openai import OpenAI
import asyncio

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
# Utilidades de Planilha
# --------------------------- #
def salvar_interacao(role, content):
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.warning(f"Erro ao salvar interação: {e}")

def salvar_resumo(resumo):
    try:
        aba = planilha.worksheet("perfil_jm")
        linhas = aba.get_all_values()
        ultima_linha = len(linhas) + 1
        aba.update_cell(ultima_linha, 7, resumo)
    except Exception as e:
        st.warning(f"Erro ao salvar resumo: {e}")

def carregar_resumo():
    try:
        aba = planilha.worksheet("perfil_jm")
        col = aba.col_values(7)
        for item in reversed(col):
            if item.strip():
                return item
        return ""
    except:
        return ""

# --------------------------- #
# Carregar memórias
# --------------------------- #
def carregar_memorias():
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() in ("[mary]", "mary")]
        mem_janio = [r["conteudo"] for r in registros if r.get("tipo", "").strip() in ("[Jânio]", "Jânio")]
        mem_all = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() in ("[all]", "all")]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

# --------------------------- #
# Construir prompt narrativo
# --------------------------- #
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")
    resumo = carregar_resumo()
    st.session_state["resumo_capitulo"] = resumo

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
# Modelos disponíveis
# --------------------------- #
modelos_disponiveis = {
    "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
    "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
    "🧠 DeepSeek R1T2 Chimera ★★★★ (free)": "tngtech/deepseek-r1t2-chimera:free",
    "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
    "👑 WizardLM 8x22B ★★★★☆ ($$$)": "microsoft/wizardlm-2-8x22b",
    "👑 Qwen 235B 2507 ★★★★★ (PAID)": "qwen/qwen3-235b-a22b-07-25",
    "👑 EVA Qwen2.5 72B ★★★★★ (RP Pro)": "eva-unit-01/eva-qwen-2.5-72b",
    "👑 EVA Llama 3.33 70B ★★★★★ (RP Pro)": "eva-unit-01/eva-llama-3.33-70b",
    "🎭 Nous Hermes 2 Yi 34B ★★★★☆": "nousresearch/nous-hermes-2-yi-34b",
    "🔥 MythoMax 13B ★★★☆ ($)": "gryphe/mythomax-l2-13b",
    "💋 LLaMA3 Lumimaid 8B ★★☆ ($)": "neversleep/llama-3-lumimaid-8b",
    "🌹 Midnight Rose 70B ★★★☆": "sophosympatheia/midnight-rose-70b",
    "🌶️ Noromaid 20B ★★☆": "neversleep/noromaid-20b",
    "💀 Mythalion 13B ★★☆": "pygmalionai/mythalion-13b",
    "🐉 Anubis 70B ★★☆": "thedrummer/anubis-70b-v1.1",
    "🧚 Rocinante 12B ★★☆": "thedrummer/rocinante-12b",
    "🍷 Magnum v2 72B ★★☆": "anthracite-org/magnum-v2-72b",
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

# --------------------------- #
# Função de resposta (OpenRouter + Together)
# --------------------------- #
def responder_com_modelo_escolhido():
    modelo = st.session_state.get("modelo_escolhido_id", "deepseek/deepseek-chat-v3-0324")
    if modelo.startswith("togethercomputer/") or modelo.startswith("mistralai/"):
        st.session_state["provedor_ia"] = "together"
        return gerar_resposta_together_stream(modelo)
    else:
        st.session_state["provedor_ia"] = "openrouter"
        return gerar_resposta_openrouter_stream(modelo)





# (O restante do script permanece igual e já estava completo na versão anterior.)


# --------------------------- #
# Sidebar
# --------------------------- #
with st.sidebar:
    st.title("🎛️ Opções do Roteirista")
    st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia")
    st.text_input("🎭 Emoção oculta da cena", key="emocao_oculta")
    if st.button("💾 Salvar resumo atual"):
        salvar_resumo(st.session_state.get("resumo_capitulo", ""))
        st.success("Resumo salvo com sucesso!")

    if st.button("📝 Criar novo resumo"):
        try:
            aba = planilha.worksheet("interacoes_jm")
            registros = aba.get_all_records()
            ultimas = registros[-15:] if len(registros) > 15 else registros
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)

            # Chamada à IA para resumir
            st.info("Gerando resumo automático com base nas últimas interações...")
            resposta = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",  # ou outro modelo que preferir
                    "messages": [
                        {"role": "system", "content": "Resuma os principais eventos dessa história para ser usado como introdução do próximo capítulo. Seja claro, emocional e sem revelar tudo."},
                        {"role": "user", "content": texto}
                    ]
                }
            )

            dados = resposta.json()
            resumo_gerado = dados["choices"][0]["message"]["content"]
            st.session_state["resumo_capitulo"] = resumo_gerado
            st.success("Resumo gerado com sucesso! Revise antes de salvar.")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")


# --------------------------- #
# Tela principal
# --------------------------- #
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

st.markdown("#### 📖 Último resumo salvo:")
st.session_state.resumo_capitulo = carregar_resumo()
st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

# Mostrar histórico recente de interações
with st.container():
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-20:] if len(registros) > 20 else registros

        for r in ultimas:
            role = r["role"]
            content = r["content"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")

entrada_usuario = st.chat_input("Digite sua direção de cena...")
if entrada_usuario:
    salvar_interacao("user", entrada_usuario)
    st.session_state.entrada_atual = entrada_usuario

    with st.spinner("Narrando..."):
        prompt = construir_prompt_com_narrador()

        modelo_id = modelos_disponiveis[st.session_state.modelo_ia]
        headers = {
            "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY'] if 'openrouter' in modelo_id or 'deepseek' in modelo_id or 'openai' in modelo_id else st.secrets['TOGETHER_API_KEY']}",
            "Content-Type": "application/json"
        }
        endpoint = "https://openrouter.ai/api/v1/chat/completions" if "openai" in modelo_id or "deepseek" in modelo_id or "openrouter" in modelo_id else "https://api.together.xyz/v1/chat/completions"

        payload = {
            "model": modelo_id,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": entrada_usuario}
            ],
            "stream": True
        }

        resposta = requests.post(endpoint, headers=headers, json=payload, stream=True)

        mensagem_final = ""
        placeholder = st.empty()
        for linha in resposta.iter_lines():
            if linha:
                try:
                    dados = json.loads(linha.decode("utf-8").replace("data: ", ""))
                    delta = dados["choices"][0]["delta"].get("content")
                    if delta:
                        mensagem_final += delta
                        placeholder.markdown(mensagem_final + "▌")
                except:
                    continue

        placeholder.markdown(mensagem_final)
        salvar_interacao("assistant", mensagem_final)














