# main_jm.py
import streamlit as st
import requests
import gspread
import json
import re
import time
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIG BÁSICA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Narrador JM", page_icon="🎬")

# Secrets esperados:
# - st.secrets["GOOGLE_CREDS_JSON"]
# - st.secrets["OPENROUTER_API_KEY"]
# - st.secrets["TOGETHER_API_KEY"]
# - st.secrets["OPENAI_API_KEY"]   (para embeddings semânticos)

# Cliente OpenAI p/ embeddings (SEM usar OpenRouter/Together)
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------------------------------------------------------
# CONEXÃO COM PLANILHA
# -----------------------------------------------------------------------------
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        # Troque a KEY abaixo pela sua key da planilha, se necessário
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()


# -----------------------------------------------------------------------------
# UTILIDADES DE PLANILHA (memórias/interações/resumo)
# -----------------------------------------------------------------------------
def carregar_memorias():
    """Lê a aba 'memorias_jm' e retorna (mem_mary, mem_janio, mem_all)."""
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[mary]"]
        mem_janio = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() in ("[jânio]", "[janio]")]
        mem_all = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[all]"]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

def carregar_resumo_salvo():
    """Busca o último resumo (coluna 7) da aba 'perfil_jm'."""
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
    """Salva um novo resumo na aba 'perfil_jm' (timestamp na coluna 6, resumo na 7)."""
    try:
        aba = planilha.worksheet("perfil_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        linha = ["", "", "", "", "", timestamp, resumo]
        aba.append_row(linha, value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def salvar_interacao(role: str, content: str):
    """Anexa uma interação na aba 'interacoes_jm'."""
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_interacoes(n=20):
    """Carrega as últimas n interações (role, content) da aba interacoes_jm."""
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        return registros[-n:] if len(registros) > n else registros
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")
        return []


# -----------------------------------------------------------------------------
# VALIDAÇÕES (sintática + semântica via OpenAI)
# -----------------------------------------------------------------------------
def resposta_valida(texto: str) -> bool:
    padroes_invalidos = [
        r"check if.*string", r"#\s?1(\.\d+)+", r"\d{10,}", r"the cmd package",
        r"(111\s?)+", r"#+\s*\d+", r"\bimport\s", r"\bdef\s", r"```", r"class\s"
    ]
    for padrao in padroes_invalidos:
        if re.search(padrao, texto.lower()):
            return False
    return True

def gerar_embedding_openai(texto: str):
    try:
        resp = client_openai.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return np.array(resp.data[0].embedding)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def verificar_quebra_semantica_openai(texto1: str, texto2: str, limite=0.6) -> str:
    e1 = gerar_embedding_openai(texto1)
    e2 = gerar_embedding_openai(texto2)
    if e1 is None or e2 is None:
        return ""
    sim = cosine_similarity(e1, e2)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""


# -----------------------------------------------------------------------------
# MEMÓRIA LONGA (Sheets + Embeddings OpenAI)
# -----------------------------------------------------------------------------
def _sheet_ensure_memoria_longa():
    """Retorna a aba memoria_longa_jm se existir (não cria)."""
    try:
        return planilha.worksheet("memoria_longa_jm")
    except Exception:
        return None  # silencioso

def _serialize_vec(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))

def _deserialize_vec(s: str) -> np.ndarray:
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        return np.zeros(1)

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
    """Salva uma memória com embedding e score inicial."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        st.warning("Aba 'memoria_longa_jm' não encontrada — crie com cabeçalhos: texto | embedding | tags | timestamp | score")
        return False
    emb = gerar_embedding_openai(texto)
    if emb is None:
        return False
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [texto.strip(), _serialize_vec(emb), (tags or "").strip(), ts, 1.0]
    try:
        aba.append_row(linha, value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar memória longa: {e}")
        return False

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78):
    """Retorna top-K memórias (texto, score, sim, rr) com base no embedding do query_text."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        return []
    q = gerar_embedding_openai(query_text)
    if q is None:
        return []
    try:
        dados = aba.get_all_records()
    except Exception as e:
        st.warning(f"Erro ao carregar memoria_longa_jm: {e}")
        return []
    candidatos = []
    for row in dados:
        texto = row.get("texto", "").strip()
        emb_s = row.get("embedding", "")
        try:
            score = float(row.get("score", 1.0) or 1.0)
        except Exception:
            score = 1.0
        if not texto or not emb_s:
            continue
        vec = _deserialize_vec(emb_s)
        if vec.ndim != 1 or vec.size < 10:
            continue
        sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
        if sim >= limiar:
            rr = 0.7 * sim + 0.3 * score
            candidatos.append((texto, score, sim, rr))
    candidatos.sort(key=lambda x: x[3], reverse=True)
    return candidatos[:k]

def memoria_longa_reforcar(textos_usados: list):
    """Aumenta o score das memórias usadas (pequeno reforço)."""
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = aba.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = dados[0]
        idx_texto = headers.index("texto")
        idx_score = headers.index("score")
        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) <= max(idx_texto, idx_score):
                continue
            t = linha[idx_texto].strip()
            if t in textos_usados:
                try:
                    sc = float(linha[idx_score] or 1.0)
                except Exception:
                    sc = 1.0
                sc = min(sc + 0.2, 2.0)
                aba.update_cell(i, idx_score + 1, sc)
    except Exception:
        pass

def memoria_longa_decadencia(fator: float = 0.97):
    """Decadência leve aplicada a todos os scores (pode ser chamada esporadicamente)."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        return
    try:
        dados = aba.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = dados[0]
        idx_score = headers.index("score")
        for i in range(2, len(dados) + 1):
            try:
                sc = float(aba.cell(i, idx_score + 1).value or 1.0)
            except Exception:
                sc = 1.0
            sc = max(sc * fator, 0.1)
            aba.update_cell(i, idx_score + 1, sc)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# REGRAS DE CENA (castidade de Mary / liberdade do Jânio)
# -----------------------------------------------------------------------------
def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    regras = """
⚖️ Regras imutáveis de coerência:
- Mary é virgem e decidiu permanecer casta até seu encontro definitivo com Jânio. Ela pode beijar, sentir desejo, mas NÃO cede a atos sexuais (sem penetração, sexo oral ou manual explícito). Se pressionada, impõe limites com firmeza e elegância.
- Jânio é livre para ter encontros íntimos com outras personagens quando fizer sentido na história (consensual, coerente e sem pornografia explícita).
- Qualquer violação dessas regras deve ser evitada na narrativa.
"""
    return prompt_base + "\n" + regras.strip()


# -----------------------------------------------------------------------------
# PROMPT
# -----------------------------------------------------------------------------
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("app_emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    # últimas 15 interações (texto plano do SHEET para garantir persistência)
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-15:] if len(registros) > 15 else registros
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
    except Exception:
        texto_ultimas = ""

    regra_intimo = ("\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista."
                    if st.session_state.get("app_bloqueio_intimo", False) else "")

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.{regra_intimo}

🎭 Emoção oculta da cena: {emocao}

📖 Capítulo anterior:
{(resumo or 'Nenhum resumo salvo.')}

### 🧠 Memórias canônicas:
Mary:
- {("\n- ".join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {("\n- ".join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {("\n- ".join(mem_all)) if mem_all else 'Nenhuma.'}

### 📖 Últimas interações (recorte):
{texto_ultimas}"""

    # Memória longa relevante
    if st.session_state.get("use_memoria_longa", True):
        try:
            ultima_entrada = ""
            if st.session_state.get("session_msgs"):
                for m in reversed(st.session_state.session_msgs):
                    if m.get("role") == "user":
                        ultima_entrada = m.get("content", "")
                        break
            query = (resumo or "") + "\n" + (ultima_entrada or "")
            k = int(st.session_state.get("k_memoria_longa", 3))
            limiar = float(st.session_state.get("limiar_memoria_longa", 0.78))
            topk = memoria_longa_buscar_topk(query_text=query, k=k, limiar=limiar)
            if topk:
                linhas = [f"- {t}" for (t, _sc, _sim, _rr) in topk]
                prompt += "\n\n### 🗃️ Memórias de longo prazo relevantes\n" + "\n".join(linhas)
        except Exception:
            pass

    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt.strip()


# -----------------------------------------------------------------------------
# PROVEDORES E MODELOS
# -----------------------------------------------------------------------------
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

MODELOS_TOGETHER_UI = {
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
}

def model_id_for_together(api_ui_model_id: str) -> str:
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in api_ui_model_id:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if api_ui_model_id.lower().startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-V0.1".replace("V0.1","v0.1")  # normaliza
    return api_ui_model_id

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
            MODELOS_TOGETHER_UI,
        )


# -----------------------------------------------------------------------------
# UI – CABEÇALHO E CONTROLES
# -----------------------------------------------------------------------------
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Estado inicial
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []
if "use_memoria_longa" not in st.session_state:
    st.session_state.use_memoria_longa = True
if "k_memoria_longa" not in st.session_state:
    st.session_state.k_memoria_longa = 3
if "limiar_memoria_longa" not in st.session_state:
    st.session_state.limiar_memoria_longa = 0.78
if "app_bloqueio_intimo" not in st.session_state:
    st.session_state.app_bloqueio_intimo = False
if "app_emocao_oculta" not in st.session_state:
    st.session_state.app_emocao_oculta = "nenhuma"

# Linha de opções rápidas
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### 📖 Último resumo salvo:")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")
with col2:
    st.markdown("#### ⚙️ Opções")
    st.checkbox(
        "Bloquear avanços íntimos sem ordem",
        value=st.session_state.app_bloqueio_intimo,
        key="ui_bloqueio_intimo",
    )
    st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.app_emocao_oculta),
        key="ui_app_emocao_oculta",
    )
    st.session_state.app_bloqueio_intimo = st.session_state.get("ui_bloqueio_intimo", False)
    st.session_state.app_emocao_oculta = st.session_state.get("ui_app_emocao_oculta", "nenhuma")


# -----------------------------------------------------------------------------
# Sidebar – Provedor, modelos, resumo e memória longa
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    st.session_state.modelo_escolhido_id = modelo_escolhido_id_ui

    st.markdown("---")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in inter) if inter else ""
            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n"
                + texto + "\n\nResumo:"
            )
            model_id_call = model_id_for_together(modelo_escolhido_id_ui) if provedor == "Together" else modelo_escolhido_id_ui
            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_id_call,
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

    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")

    st.checkbox(
        "Usar memória longa no prompt",
        value=st.session_state.get("use_memoria_longa", True),
        key="use_memoria_longa",
    )
    st.slider(
        "Top-K memórias",
        1, 5,
        st.session_state.get("k_memoria_longa", 3),
        1,
        key="k_memoria_longa",
    )
    st.slider(
        "Limiar de similaridade",
        0.50, 0.95,
        float(st.session_state.get("limiar_memoria_longa", 0.78)),
        0.01,
        key="limiar_memoria_longa",
    )

    if st.button("💾 Salvar última resposta como memória"):
        ultimo_assist = ""
        for m in reversed(st.session_state.get("session_msgs", [])):
            if m.get("role") == "assistant":
                ultimo_assist = m.get("content", "").strip()
                break
        if ultimo_assist:
            ok = memoria_longa_salvar(ultimo_assist, tags="auto")
            st.success("Memória de longo prazo salva!" if ok else "Falha ao salvar memória.")
        else:
            st.info("Ainda não há resposta do assistente nesta sessão.")

    st.caption("Role a tela principal para ver interações anteriores.")


# -----------------------------------------------------------------------------
# EXIBIR HISTÓRICO RECENTE (primeiro interações, depois resumo)
# -----------------------------------------------------------------------------
with st.container():
    interacoes = carregar_interacoes(n=20)
    for r in interacoes:
        role = r.get("role", "user")
        content = r.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

# Resumo no fim da tela (como você pediu)
if st.session_state.get("resumo_capitulo"):
    with st.expander("🧠 Resumo do capítulo (mais recente)"):
        st.markdown(st.session_state.resumo_capitulo)


# -----------------------------------------------------------------------------
# ENVIO DO USUÁRIO + STREAMING (OpenRouter/Together) + FALLBACKS
# -----------------------------------------------------------------------------
entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    prompt = construir_prompt_com_narrador()

    # Histórico curto (somente sessão atual; o prompt já inclui últimas do sheet)
    historico = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in st.session_state.session_msgs]

    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets["TOGETHER_API_KEY"]
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets["OPENROUTER_API_KEY"]
        model_to_call = st.session_state.modelo_escolhido_id

    # System para PT-BR e sem “rascunho” — mas NÃO bloqueia <think> (só não exibe)
    system_pt = {
        "role": "system",
        "content": (
            "Responda em português do Brasil. Evite conteúdo meta. "
            "Mostre apenas a narrativa final ao leitor."
        ),
    }
    messages = [system_pt, {"role": "system", "content": prompt}] + historico

    payload = {
        "model": model_to_call,
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.9,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()
        in_think = False
        pending = ""

        def render_tail(t):
            # corta <think>...</think> da tela, mas NÃO altera o texto salvo
            return t

        # 1) STREAM
        stream_ok = False
        try:
            with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=300) as r:
                if r.status_code == 200:
                    stream_ok = True
                    for raw in r.iter_lines(decode_unicode=False):
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                            delta = j["choices"][0]["delta"].get("content", "")
                            if not delta:
                                continue
                            # Não removo <think> do conteúdo final (pra não “comer” texto),
                            # só evito mostrar na tela (se quiser esconder, filtra aqui).
                            resposta_txt += delta
                            if time.time() - last_update > 0.10:
                                placeholder.markdown(render_tail(resposta_txt) + "▌")
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro no streaming: {e}")

        # 2) FALLBACKS se veio vazio
        if not resposta_txt.strip():
            # 2a) retry sem stream
            try:
                r2 = requests.post(
                    endpoint, headers=headers,
                    json={**payload, "stream": False},
                    timeout=180
                )
                if r2.status_code == 200:
                    resposta_txt = r2.json()["choices"][0]["message"]["content"].strip()
                else:
                    st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
            except Exception as e:
                st.error(f"Fallback (sem stream) erro: {e}")

        if not resposta_txt.strip():
            # 2b) retry sem o system extra (alguns modelos travam com system duplo)
            try:
                r3 = requests.post(
                    endpoint, headers=headers,
                    json={
                        "model": model_to_call,
                        "messages": [{"role": "system", "content": prompt}] + historico,
                        "max_tokens": 900,
                        "temperature": 0.9,
                        "stream": False,
                    },
                    timeout=180
                )
                if r3.status_code == 200:
                    resposta_txt = r3.json()["choices"][0]["message"]["content"].strip()
                else:
                    st.error(f"Fallback (prompts limpos) falhou: {r3.status_code} - {r3.text}")
            except Exception as e:
                st.error(f"Fallback (prompts limpos) erro: {e}")

        # Flush final na tela
        placeholder.markdown(render_tail(resposta_txt) if resposta_txt.strip() else "[Sem conteúdo]")

        # Validação sintática + regeneração simples
        if not resposta_valida(resposta_txt):
            st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar...")
            try:
                regen = requests.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model_to_call,
                        "messages": [{"role": "system", "content": prompt}] + historico,
                        "max_tokens": 900,
                        "temperature": 0.9,
                        "stream": False,
                    },
                    timeout=180,
                )
                if regen.status_code == 200:
                    resposta_txt = regen.json()["choices"][0]["message"]["content"].strip()
                    placeholder.markdown(render_tail(resposta_txt))
                else:
                    st.error(f"Erro ao regenerar: {regen.status_code} - {regen.text}")
            except Exception as e:
                st.error(f"Erro ao regenerar: {e}")

        # Validação semântica (entrada do user vs resposta)
        if len(st.session_state.session_msgs) >= 1 and resposta_txt and resposta_txt != "[Sem conteúdo]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]  # última entrada do user
            alerta = verificar_quebra_semantica_openai(texto_anterior, resposta_txt)
            if alerta:
                st.info(alerta)

        # Salvar resposta SEMPRE (mesmo que curta)
        salvar_interacao("assistant", resposta_txt or "[Sem conteúdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt or "[Sem conteúdo]"})

        # Reforço de memórias usadas
        try:
            usados = []
            topk_usadas = memoria_longa_buscar_topk(
                query_text=resposta_txt,
                k=int(st.session_state.k_memoria_longa),
                limiar=float(st.session_state.limiar_memoria_longa),
            )
            for t, _sc, _sim, _rr in topk_usadas:
                usados.append(t)
            memoria_longa_reforcar(usados)
        except Exception:
            pass

