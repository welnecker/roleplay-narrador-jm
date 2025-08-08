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
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
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

# Modelos EXATOS por provedor (IDs aceitos pela API)
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
# UI Principal
# =========================== #

st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Carregar resumo ao iniciar (apenas 1x)
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()

# Controles principais
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### 📖 Último resumo salvo:")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")
with col2:
    st.markdown("#### ⚙️ Opções")
    st.session_state.bloqueio_intimo = st.checkbox("Bloquear avanços íntimos sem ordem", value=False)
    st.session_state.emocao_oculta = st.selectbox(
        "🎭 Emoção oculta", ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"], index=0
    )

# Botão de gerar resumo (na tela principal)
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

        # Seleção de provedor/modelo para o resumo (usa sessão atual, se existir; senão defaults)
        provider = st.session_state.get("provedor_ia_for_summary", st.session_state.get("provedor_ia", "OpenRouter"))
        api_url, api_key, modelos = api_config_for_provider(provider)
        modelo_id = st.session_state.get("modelo_resumo_id", list(MODELOS_OPENROUTER.values())[0])

        prompt_resumo = (
            "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n" + texto + "\n\nResumo:"
        )

        r = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": modelo_id,
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

# --------------------------- #
# Sidebar - Provedor/Modelo para NARRATIVA
# --------------------------- #
with st.sidebar:
    st.title("🌐 Configurações de IA")
    provedor = st.radio("Provedor de IA", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_disponiveis = api_config_for_provider(provedor)
    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), index=0)
    modelo_id = modelos_disponiveis[modelo_nome]

    # Persistir seleção para uso na chamada
    st.session_state.api_url = api_url
    st.session_state.api_key = api_key
    st.session_state.modelo_escolhido = modelo_id

    # (Opcional) escolher modelo do resumo separadamente
    st.markdown("---")
    st.caption("Modelo p/ Gerar Resumo (opcional)")
    prov_resumo = st.selectbox("Provedor do Resumo", ["OpenRouter", "Together"], index=0)
    st.session_state.provedor_ia_for_summary = prov_resumo
    _, _, modelos_res = api_config_for_provider(prov_resumo)
    nome_res = st.selectbox("Modelo do Resumo", list(modelos_res.keys()), index=0)
    st.session_state.modelo_resumo_id = modelos_res[nome_res]

# --------------------------- #
# Entrada do usuário e geração da narrativa
# --------------------------- #

entrada_usuario = st.chat_input("Digite sua direção de cena...")


def exibir_resposta_digitando(texto: str, delay: float = 0.02):
    box = st.empty()
    acumulado = ""
    for ch in texto:
        acumulado += ch
        box.markdown(acumulado)
        time.sleep(delay)


if entrada_usuario:
    # Salva entrada do usuário
    salvar_interacao("user", entrada_usuario)

    # Constrói prompt narrativo
    prompt = construir_prompt_com_narrador() + f"\n\n🎬 Direção do roteirista: {entrada_usuario}"

    # Chama a IA
    try:
        resp = requests.post(
            st.session_state.api_url,
            headers={"Authorization": f"Bearer {st.session_state.api_key}", "Content-Type": "application/json"},
            json={
                "model": st.session_state.modelo_escolhido,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.85,
            },
            timeout=180,
        )
        if resp.status_code == 200:
            conteudo = resp.json()["choices"][0]["message"]["content"].strip()
            salvar_interacao("assistant", conteudo)
            exibir_resposta_digitando(conteudo)
        else:
            st.error(f"Erro {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
