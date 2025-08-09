import streamlit as st
import gspread
import json
import requests
import time
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Narrador JM", page_icon="🎬")

# =========================== #
# Conectar à planilha
# =========================== #
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        # Convertendo "\n" literais da chave privada
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

# =========================== #
# Utilidades de Planilha
# =========================== #
def carregar_memorias():
    """Lê a aba memorias_jm e separa por [mary], [jânio], [all]."""
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r.get("conteudo", "").strip() for r in registros if r.get("tipo", "").strip().lower() == "[mary]"]
        mem_janio = [r.get("conteudo", "").strip() for r in registros if r.get("tipo", "").strip().lower() == "[jânio]"]
        mem_all = [r.get("conteudo", "").strip() for r in registros if r.get("tipo", "").strip().lower() == "[all]"]
        mem_mary = [m for m in mem_mary if m]
        mem_janio = [m for m in mem_janio if m]
        mem_all = [m for m in mem_all if m]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

def carregar_resumo_salvo():
    """Pega o último resumo não vazio da aba perfil_jm, coluna 7."""
    try:
        aba = planilha.worksheet("perfil_jm")
        valores = aba.col_values(7)
        for val in reversed(valores[1:]):
            if val and val.strip():
                return val.strip()
        return ""
    except Exception as e:
        st.warning(f"Erro ao carregar resumo salvo: {e}")
        return ""

def salvar_resumo(resumo: str):
    """Anexa um resumo na aba perfil_jm (coluna 7), com timestamp na coluna 6."""
    try:
        aba = planilha.worksheet("perfil_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        linha = ["", "", "", "", "", timestamp, resumo]
        aba.append_row(linha, value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def salvar_interacao(role: str, content: str):
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

# =========================== #
# Construir prompt narrativo
# =========================== #
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-15:] if len(registros) > 15 else registros
        texto_ultimas = "\n".join(f"{r.get('role','')}: {r.get('content','')}" for r in ultimas)
    except Exception:
        texto_ultimas = ""

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.

Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista.

Emoção oculta da cena: {emocao}

Capítulo anterior:
{resumo if resumo else 'Nenhum resumo salvo.'}

Memórias:
Mary:
- {('\n- '.join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {('\n- '.join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {('\n- '.join(mem_all)) if mem_all else 'Nenhuma.'}

Últimas interações:
{texto_ultimas}"""
    return prompt.strip()

# =========================== #
# Provedores e Modelos (IDs exatos)
# =========================== #
MODELOS_OPENROUTER = {
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
}
MODELOS_TOGETHER = {
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/qwen3-coder-480b-a35b-instruct",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/mixtral-8x7b-instruct-v0.1",
}

# =========================== #
# Roteador por provedor
# =========================== #
def responder_com_modelo_escolhido(modelo_escolhido_id: str):
    if modelo_escolhido_id.startswith(("togethercomputer/", "mistralai/")):
        st.session_state["provedor_ia"] = "together"
        return gerar_resposta_together_stream(modelo_escolhido_id)
    else:
        st.session_state["provedor_ia"] = "openrouter"
        return gerar_resposta_openrouter_stream(modelo_escolhido_id)

# =========================== #
# OpenRouter - Streaming (SSE)
# =========================== #
def gerar_resposta_openrouter_stream(modelo_escolhido_id: str):
    prompt = construir_prompt_com_narrador().strip()

    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico

    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 800,
        "temperature": 0.85,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_text = ""

    try:
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=300,
        ) as r:
            if r.status_code != 200:
                st.error(f"Erro OpenRouter: {r.status_code} - {r.text}")
                return "[ERRO STREAM]"
            for raw_line in r.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text += delta
                        placeholder.markdown(full_text + "▌")
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Erro no streaming com OpenRouter: {e}")
        return "[ERRO STREAM]"

    placeholder.markdown(full_text)
    return full_text.strip()

# =========================== #
# Together - Streaming (SSE)
# =========================== #
def gerar_resposta_together_stream(modelo_escolhido_id: str):
    prompt = construir_prompt_com_narrador().strip()

    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico

    payload = {
        "model": modelo_escolhido_id,  # ex.: "togethercomputer/qwen3-coder-480b-a35b-instruct"
        "messages": mensagens,
        "max_tokens": 800,
        "temperature": 0.85,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}",
        "Content-Type": "application/json",
    }

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_text = ""

    try:
        with requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=300,
        ) as r:
            if r.status_code != 200:
                st.error(f"Erro Together: {r.status_code} - {r.text}")
                return "[ERRO STREAM]"
            for raw_line in r.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text += delta
                        placeholder.markdown(full_text + "▌")
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Erro no streaming com Together: {e}")
        return "[ERRO STREAM]"

    placeholder.markdown(full_text)
    return full_text.strip()

# =========================== #
# Sidebar – provedor, modelo e resumo
# =========================== #
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio("Provedor de IA", ["OpenRouter", "Together"], index=0, key="provedor_ia")

    modelos_disponiveis = MODELOS_TOGETHER if provedor == "Together" else MODELOS_OPENROUTER
    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), index=0, key="modelo_ia_nome")
    st.session_state.modelo_escolhido_id = modelos_disponiveis[modelo_nome]

    st.markdown("---")
    st.caption("Gerar resumo usando o modelo selecionado acima")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            aba_i = planilha.worksheet("interacoes_jm")
            registros = aba_i.get_all_records()
            ultimas = registros[-6:] if len(registros) > 6 else registros
            texto = "\n".join(f"{r.get('role','')}: {r.get('content','')}" for r in ultimas)
            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n"
                + texto
                + "\n\nResumo:"
            )

            # endpoint e key coerentes com o provedor escolhido
            if st.session_state.modelo_escolhido_id.startswith(("togethercomputer/", "mistralai/")):
                endpoint = "https://api.together.xyz/v1/chat/completions"
                api_key = st.secrets["TOGETHER_API_KEY"]
            else:
                endpoint = "https://openrouter.ai/api/v1/chat/completions"
                api_key = st.secrets["OPENROUTER_API_KEY"]

            r = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": st.session_state.modelo_escolhido_id,
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85,
                },
                timeout=120,
            )
            if r.status_code == 200:
                novo_resumo = r.json()["choices"][0]["message"]["content"].strip()
                salvar_resumo(novo_resumo)
                st.session_state.resumo_capitulo = novo_resumo
                st.success("Resumo gerado e salvo com sucesso!")
            else:
                st.error(f"Erro ao resumir: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")

    st.markdown("---")
    st.caption("Opções de narrativa")
    st.session_state.bloqueio_intimo = st.checkbox("Bloquear avanços íntimos sem ordem", value=False)
    st.session_state.emocao_oculta = st.selectbox(
        "🎭 Emoção oculta", ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"], index=0
    )

# =========================== #
# Tela principal – título, resumo e histórico
# =========================== #
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Carregar resumo ao iniciar (apenas 1x)
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()

st.markdown("#### 📖 Último resumo salvo:")
st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

# Mostrar histórico recente de interações
with st.container():
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-20:] if len(registros) > 20 else registros
        for r in ultimas:
            role = r.get("role", "user")
            content = r.get("content", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")

# =========================== #
# Entrada do usuário + chamada do roteador
# =========================== #
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []

entrada_usuario = st.chat_input("Digite sua direção de cena...")
if entrada_usuario:
    salvar_interacao("user", entrada_usuario)
    st.session_state.session_msgs.append({"role": "user", "content": entrada_usuario})

    resposta_txt = responder_com_modelo_escolhido(st.session_state.modelo_escolhido_id)

    if resposta_txt and resposta_txt != "[ERRO STREAM]":
        salvar_interacao("assistant", resposta_txt)
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt})
