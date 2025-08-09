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
        # Corrige quebras de linha da private_key
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        # Substitua pela sua KEY
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
        def norm(x): 
            return (x or "").strip().lower()
        mem_mary = [r.get("conteudo","").strip() for r in registros if norm(r.get("tipo")) == "[mary]"]
        mem_janio = [r.get("conteudo","").strip() for r in registros if norm(r.get("tipo")) == "[jânio]"]
        mem_all  = [r.get("conteudo","").strip() for r in registros if norm(r.get("tipo")) == "[all]"]
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
        for val in reversed(valores[1:]):  # ignora header
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
        # col6 = timestamp, col7 = resumo
        linha = ["", "", "", "", "", timestamp, resumo]
        aba.append_row(linha, value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def salvar_interacao(role: str, content: str):
    """Salva a interação na aba interacoes_jm."""
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_interacoes_recente(n=20):
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        return registros[-n:] if len(registros) > n else registros
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")
        return []

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

    regra_intimo = "\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista." if st.session_state.get("bloqueio_intimo", False) else ""

    prompt = f"""
Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.
{regra_intimo}

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

# =========================== #
# Provedores e Modelos
# =========================== #
MODELOS_OPENROUTER = {
    # === OPENROUTER === (IDs exatos informados por você)
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
    # === TOGETHER AI === (IDs exatos informados por você)
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

def api_config_for_provider(provider: str):
    if provider == "OpenRouter":
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            st.secrets["OPENROUTER_API_KEY"],
            MODELOS_OPENROUTER,
        )
    else:
        return (
            "https://api.together.xyz/v1/chat/completions",
            st.secrets["TOGETHER_API_KEY"],
            MODELOS_TOGETHER,
        )

# =========================== #
# Roteamento do provedor
# =========================== #
def responder_com_modelo_escolhido(modelo_escolhido_id: str):
    # Prefixos que identificam modelos do Together
    if modelo_escolhido_id.startswith(("togethercomputer/", "mistralai/")):
        st.session_state["provedor_ia"] = "Together"
        return gerar_resposta_together_stream(modelo_escolhido_id)
    else:
        st.session_state["provedor_ia"] = "OpenRouter"
        return gerar_resposta_openrouter_stream(modelo_escolhido_id)

# =========================== #
# OpenRouter - Streaming (SSE)
# =========================== #
def gerar_resposta_openrouter_stream(modelo_escolhido_id: str):
    prompt = construir_prompt_com_narrador().strip()
    # histórico local da sessão (somente mensagens relevantes do chat)
    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico

    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 900,
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
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 900,
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
# UI – Cabeçalho
# =========================== #
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Inicializa estado
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []  # [{role, content}, ...]
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()
if "modelo_escolhido_id" not in st.session_state:
    # default: DeepSeek V3 no OpenRouter
    st.session_state.modelo_escolhido_id = list(MODELOS_OPENROUTER.values())[0]

# =========================== #
# Sidebar – Provedor, Modelo, Emoção e Resumo
# =========================== #
with st.sidebar:
    st.title("🌐 Configurações de IA")
    provedor = st.radio("Provedor de IA", ["OpenRouter", "Together"], index=0)
    api_url, api_key, modelos_disponiveis = api_config_for_provider(provedor)

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), index=0, key="sb_modelo")
    st.session_state.modelo_escolhido_id = modelos_disponiveis[modelo_nome]

    st.markdown("---")
    st.subheader("🎭 Emoção / Regras")
    st.session_state.emocao_oculta = st.selectbox(
        "Emoção oculta da cena", ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"], index=0
    )
    st.session_state.bloqueio_intimo = st.checkbox(
        "Bloquear avanços íntimos sem ordem explícita", value=False
    )

    st.markdown("---")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            # Pega últimas interações (6)
            regs = carregar_interacoes_recente(n=6)
            texto = "\n".join(f"{r.get('role','')}: {r.get('content','')}" for r in regs)

            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n"
                + texto + "\n\nResumo:"
            )

            # Decide endpoint pelo modelo escolhido atualmente no sidebar
            modelo_id = st.session_state.modelo_escolhido_id
            if modelo_id.startswith(("togethercomputer/", "mistralai/")):
                endpoint = "https://api.together.xyz/v1/chat/completions"
                key = st.secrets["TOGETHER_API_KEY"]
            else:
                endpoint = "https://openrouter.ai/api/v1/chat/completions"
                key = st.secrets["OPENROUTER_API_KEY"]

            r = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": modelo_id,
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

    st.caption("As interações mais recentes aparecem abaixo. Role para cima para rever respostas anteriores.")

# =========================== #
# Último resumo salvo (na tela principal)
# =========================== #
st.markdown("#### 📖 Último resumo salvo:")
st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

# =========================== #
# Mostrar histórico recente (role para cima para ver mais)
# =========================== #
with st.container():
    recentes = carregar_interacoes_recente(n=20)
    for r in recentes:
        role = r.get("role", "")
        content = r.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

# =========================== #
# Entrada do usuário + Streaming
# =========================== #
entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    # Salva e injeta no histórico da sessão
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    # Gera resposta via modelo selecionado
    with st.spinner("Narrando..."):
        resposta_txt = responder_com_modelo_escolhido(st.session_state.modelo_escolhido_id)

    # Exibe e salva a resposta final
    if resposta_txt and resposta_txt.strip():
        salvar_interacao("assistant", resposta_txt.strip())
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt.strip()})
