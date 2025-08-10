import streamlit as st
import gspread
import json
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Narrador JM", page_icon="🎬")

# =========================== #
# Conectar à planilha
# =========================== #
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        # cuidado com a quebra de linha da chave privada
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        # ID fixo que você usa
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# =========================== #
# Utilidades: memórias, resumo, interações
# =========================== #
def carregar_memorias():
    """Lê a aba memorias_jm e separa por [mary], [jânio], [all]."""
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
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
    except Exception:
        texto_ultimas = ""

    regra_intimo = (
        "\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista."
        if st.session_state.get("bloqueio_intimo", False) else ""
    )

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.{regra_intimo}

🎭 Emoção oculta da cena: {emocao}

📖 Capítulo anterior:
{resumo if resumo else 'Nenhum resumo salvo.'}

### 🧠 Memórias:
Mary:
- {('\n- '.join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {('\n- '.join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {('\n- '.join(mem_all)) if mem_all else 'Nenhuma.'}

### 📖 Últimas interações:
{texto_ultimas}"""
    return prompt.strip()

# =========================== #
# Modelos / Roteamento de Provedor
# =========================== #
MODELOS_OPENROUTER = {
    # === OPENROUTER === (IDs EXATOS que você passou)
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

# Together — IDs oficiais (como você mostrou nos exemplos funcionais)
MODELOS_TOGETHER = {
    "🧠 Qwen3 Coder 480B (Together)": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Detectar Together por prefixo (escala fácil p/ futuros modelos)
TOGETHER_PREFIXES = ("Qwen/", "mistralai/")
def is_together_model(model_id: str) -> bool:
    return model_id.startswith(TOGETHER_PREFIXES)

# =========================== #
# Streaming – OpenRouter / Together
# =========================== #
def gerar_resposta_openrouter_stream(modelo_id: str):
    prompt = construir_prompt_com_narrador()
    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico

    payload = {
        "model": modelo_id,
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

def gerar_resposta_together_stream(modelo_id: str):
    prompt = construir_prompt_com_narrador()
    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico

    payload = {
        "model": modelo_id,
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

def responder_com_modelo_escolhido(modelo_id: str):
    if is_together_model(modelo_id):
        return gerar_resposta_together_stream(modelo_id)
    else:
        return gerar_resposta_openrouter_stream(modelo_id)

# =========================== #
# UI – Cabeçalho, estado inicial
# =========================== #
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# inicia estado
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()
if "bloqueio_intimo" not in st.session_state:
    st.session_state.bloqueio_intimo = False
if "emocao_oculta" not in st.session_state:
    st.session_state.emocao_oculta = "nenhuma"

# =========================== #
# Sidebar – Modelos e opções
# =========================== #
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    st.session_state.bloqueio_intimo = st.checkbox("Bloquear avanços íntimos sem ordem", value=st.session_state.bloqueio_intimo)
    st.session_state.emocao_oculta = st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.emocao_oculta)
    )

    st.markdown("---")
    st.markdown("**Seleção de modelo**")

    # monta catálogos juntos para o selectbox
    modelos_disponiveis = {}
    modelos_disponiveis.update(MODELOS_OPENROUTER)
    modelos_disponiveis.update(MODELOS_TOGETHER)

    chave_opcao = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), index=0)
    st.session_state.modelo_escolhido_id = modelos_disponiveis[chave_opcao]

    # Botão de gerar resumo (no sidebar)
    st.markdown("---")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            # Puxa últimas interações
            try:
                aba_i = planilha.worksheet("interacoes_jm")
                registros = aba_i.get_all_records()
                ultimas = registros[-6:] if len(registros) > 6 else registros
                texto = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
            except Exception:
                texto = ""

            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n"
                + texto + "\n\nResumo:"
            )

            modelo_id_resumo = st.session_state.modelo_escolhido_id
            if is_together_model(modelo_id_resumo):
                endpoint = "https://api.together.xyz/v1/chat/completions"
                api_key = st.secrets["TOGETHER_API_KEY"]
            else:
                endpoint = "https://openrouter.ai/api/v1/chat/completions"
                api_key = st.secrets["OPENROUTER_API_KEY"]

            r = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": modelo_id_resumo,
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85,
                },
                timeout=120,
            )
            if r.status_code == 200:
                resumo = r.json()["choices"][0]["message"]["content"].strip()
                st.session_state.resumo_capitulo = resumo
                salvar_resumo(resumo)
                st.success("Resumo gerado e salvo com sucesso!")
            else:
                st.error(f"Erro ao resumir: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")

    st.caption("As interações mais recentes aparecem na tela principal. Role para cima para rever as anteriores.")

# =========================== #
# Tela principal – Resumo + Histórico
# =========================== #
st.markdown("#### 📖 Último resumo salvo:")
st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

# mostra histórico recente (somente da planilha, para referência rápida)
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

# =========================== #
# Entrada do usuário + Resposta (streaming)
# =========================== #
entrada_usuario = st.chat_input("Digite sua direção de cena...")
if entrada_usuario:
    # mostra a fala do usuário
    with st.chat_message("user"):
        st.markdown(entrada_usuario)

    # registra na planilha e no estado
    salvar_interacao("user", entrada_usuario)
    st.session_state.session_msgs.append({"role": "user", "content": entrada_usuario})

    # responde (stream) com o modelo escolhido
    with st.spinner("Narrando..."):
        resposta_txt = responder_com_modelo_escolhido(st.session_state.modelo_escolhido_id)

    # salva e mantém no estado
    salvar_interacao("assistant", resposta_txt)
    st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt})
