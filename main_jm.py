# main_jm.py
import streamlit as st
import requests
import gspread
import json
import re
import time
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import hashlib

import numpy as np
from openai import OpenAI
import gspread.utils as gsu

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
        # Corrige quebra da private_key
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
# UTILS: Embeddings com CACHE + normalização (OTIMIZAÇÕES #1 e #3)
# -----------------------------------------------------------------------------
def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n else vec

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_embed_raw(text_hash: str, text: str):
    """Gera embedding e devolve como lista de floats (cache por hash)."""
    resp = client_openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    emb = resp.data[0].embedding
    return emb  # list

def gerar_embedding_openai(texto: str):
    try:
        # hash para chave estável de cache
        h = hashlib.sha256(texto.encode("utf-8")).hexdigest()
        emb_list = _cached_embed_raw(h, texto)
        return _unit(np.array(emb_list, dtype=float))
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None


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
    """Carrega as últimas n interações (role, content)."""
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

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))  # já normalizados

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
#   - Otimização #2: deduplicação
#   - Otimização #3: vetores normalizados
#   - Otimização #4: batch update (update_cells)
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
        v = np.array(json.loads(s), dtype=float)
        return _unit(v)  # normaliza por garantia
    except Exception:
        return np.zeros(1)

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
        sim = float(np.dot(q, vec))  # vetores unitários
        if sim >= limiar:
            # re-ranking: 0.7*sim + 0.3*score
            rr = 0.7 * sim + 0.3 * score
            candidatos.append((texto, score, sim, rr))

    candidatos.sort(key=lambda x: x[3], reverse=True)
    return candidatos[:k]

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
    """Salva uma memória com embedding e score inicial (com anti-duplicata e mínimo de tamanho)."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        st.warning("Aba 'memoria_longa_jm' não encontrada — crie com cabeçalhos: texto | embedding | tags | timestamp | score")
        return False

    # bloqueia memórias muito curtas (ruído)
    if len(texto.strip()) < 80:
        st.info("Memória não salva (muito curta).")
        return False

    # checa duplicata forte
    dup = memoria_longa_buscar_topk(query_text=texto, k=1, limiar=0.95)
    if dup:
        st.info("Memória semelhante já existe (≥0.95).")
        return False

    emb = gerar_embedding_openai(texto)
    if emb is None:
        return False

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [
        texto.strip(),
        _serialize_vec(emb),
        (tags or "").strip(),
        ts,
        1.0,  # score inicial
    ]
    try:
        aba.append_row(linha, value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar memória longa: {e}")
        return False

def _score_column_letter(aba, headers):
    idx_score = headers.index("score") + 1
    a1 = gsu.rowcol_to_a1(1, idx_score)  # ex: 'E1'
    m = re.match(r"([A-Z]+)", a1)
    return m.group(1), idx_score

def memoria_longa_reforcar(textos_usados: list):
    """Aumenta o score das memórias usadas (pequeno reforço) com update_cells (batch)."""
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = aba.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = dados[0]
        idx_texto = headers.index("texto")
        col_letter, idx_score_col = _score_column_letter(aba, headers)

        # pega toda a coluna de score (linhas 2..N)
        start_row = 2
        end_row = len(dados)
        if end_row < 2:
            return
        cells = aba.range(f"{col_letter}{start_row}:{col_letter}{end_row}")

        # mapeia texto -> row
        texto_para_row = {}
        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) <= max(idx_texto, idx_score_col-1):
                continue
            t = linha[idx_texto].strip()
            texto_para_row[t] = i

        # aplica reforço
        for t in textos_usados:
            row = texto_para_row.get(t)
            if not row:
                continue
            cell_index = row - start_row  # zero-based no vetor cells
            try:
                sc = float(cells[cell_index].value or 1.0)
            except Exception:
                sc = 1.0
            sc = min(sc + 0.2, 2.0)
            cells[cell_index].value = sc

        aba.update_cells(cells, value_input_option="RAW")
    except Exception:
        pass

def memoria_longa_decadencia(fator: float = 0.97):
    """Decadência leve aplicada a todos os scores (batch update)."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        return
    try:
        dados = aba.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = dados[0]
        col_letter, idx_score_col = _score_column_letter(aba, headers)

        start_row = 2
        end_row = len(dados)
        cells = aba.range(f"{col_letter}{start_row}:{col_letter}{end_row}")

        for i, c in enumerate(cells):
            try:
                sc = float(c.value or 1.0)
            except Exception:
                sc = 1.0
            sc = max(sc * fator, 0.1)
            cells[i].value = sc

        aba.update_cells(cells, value_input_option="RAW")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# PROMPT (com higiene de contexto – OTIMIZAÇÃO #5)
# -----------------------------------------------------------------------------
def _cap_str(s: str, max_chars: int = 800) -> str:
    s = s.strip()
    return (s[:max_chars] + "…") if len(s) > max_chars else s

def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("app_emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    # últimas 10–15 interações (cap em 10 para foco/custo)
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ult = registros[-10:] if len(registros) > 10 else registros
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ult)
    except Exception:
        texto_ultimas = ""

    # flags de castidade/ liberdade sexual
    mary_chaste = st.session_state.get("mary_chaste", True)  # por padrão, Mary mantém virgindade
    janio_free = st.session_state.get("janio_free", True)    # por padrão, Jânio pode transar

    # regras narrativas
    regra_mary = (
        "\n💎 Mary é casta por vontade própria e NÃO terá relações sexuais, apenas beijos/carícias leves, até o encontro definitivo que selará sua escolha."
        if mary_chaste else
        "\n💎 Mary decide seus limites; cenas íntimas só com consentimento expresso e coerência emocional."
    )
    regra_janio = (
        "\n🔥 Jânio é livre para ter relações sexuais consensuais com outras personagens quando fizer sentido na história; descreva com elegância e sem grafismo explícito."
        if janio_free else
        "\n🔥 Jânio evita relações sexuais neste momento; mantenha tensão e insinuação."
    )

    # bloqueio íntimo global (se o roteirista quiser travar tudo)
    regra_intimo = (
        "\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista."
        if st.session_state.get("app_bloqueio_intimo", False) else ""
    )

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.
Mantenha o texto em português do Brasil.{regra_intimo}{regra_mary}{regra_janio}

🎭 Emoção oculta da cena: {emocao}

📖 Capítulo anterior (resumo):
{(_cap_str(resumo) or 'Nenhum resumo salvo.')}

### 🧠 Memórias principais
Mary:
- {("\n- ".join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {("\n- ".join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {("\n- ".join(mem_all)) if mem_all else 'Nenhuma.'}

### 📖 Últimas interações (recente)
{_cap_str(texto_ultimas, 1200)}"""

    # === Memórias longas relevantes (Top-K) — opcional, se habilitado ===
    if st.session_state.get("use_memoria_longa", True):
        try:
            ultima_entrada = ""
            if st.session_state.get("session_msgs"):
                for m in reversed(st.session_state.session_msgs):
                    if m.get("role") == "user":
                        ultima_entrada = m.get("content", "")
                        break
            query = f"{resumo}\n{ultima_entrada}".strip()

            k = int(st.session_state.get("k_memoria_longa", 3))
            limiar = float(st.session_state.get("limiar_memoria_longa", 0.78))
            topk = memoria_longa_buscar_topk(query_text=query, k=k, limiar=limiar)
            if topk:
                linhas = [f"- {_cap_str(t, 800)}" for (t, _sc, _sim, _rr) in topk]
                prompt += "\n\n### 🗃️ Memórias de longo prazo relevantes\n" + "\n".join(linhas)
        except Exception:
            pass

    # estilo curto: mostre-não-conte, falas concisas
    prompt += """
### 🎛️ Estilo desejado
- Mostre, não conte; 2–3 frases ativas por parágrafo.
- Uma pista sensorial por frase (som OU cheiro OU tato).
- Falas curtas com subtexto (≤18 palavras), sem discurso longo.
- Evite tags técnicas (<think>, SFX) e linguagem de câmera.
"""
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

# IDs exibidos p/ usuário (mantidos)
MODELOS_TOGETHER_UI = {
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 perplexity-ai_r1-1776 (Together)": "perplexity-ai/r1-1776",
}

# Conserta ID do Together p/ endpoint oficial
def model_id_for_together(api_ui_model_id: str) -> str:
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in api_ui_model_id:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if api_ui_model_id.lower().startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # outros (ex.: perplexity-ai/r1-1776) ficam como estão
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
# flags de castidade/liberdade
if "mary_chaste" not in st.session_state:
    st.session_state.mary_chaste = True
if "janio_free" not in st.session_state:
    st.session_state.janio_free = True

# Linha de opções rápidas
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### 📖 Interações recentes")
    interacoes_preview = carregar_interacoes(n=10)
    if interacoes_preview:
        for r in interacoes_preview:
            role = r.get("role", "user")
            content = r.get("content", "")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)
    else:
        st.info("Nenhuma interação salva ainda.")

    # Resumo aparece DEPOIS das interações (pedido do usuário)
    st.markdown("#### 🧠 Último resumo salvo")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")

with col2:
    st.markdown("#### ⚙️ Opções")

    # defaults seguros antes de renderizar widgets
    if "app_bloqueio_intimo" not in st.session_state:
        st.session_state.app_bloqueio_intimo = False
    if "app_emocao_oculta" not in st.session_state:
        st.session_state.app_emocao_oculta = "nenhuma"

    # opções de castidade/liberdade
    st.checkbox(
        "Manter Mary casta (virgem)",
        value=st.session_state.mary_chaste,
        key="ui_mary_chaste",
    )
    st.checkbox(
        "Permitir Jânio ter relações (consensuais)",
        value=st.session_state.janio_free,
        key="ui_janio_free",
    )

    # widgets com keys distintas (prefixo ui_)
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

    # espelha valores UI → estado lógico
    st.session_state.app_bloqueio_intimo = st.session_state.get("ui_bloqueio_intimo", False)
    st.session_state.app_emocao_oculta   = st.session_state.get("ui_app_emocao_oculta", "nenhuma")
    st.session_state.mary_chaste         = st.session_state.get("ui_mary_chaste", True)
    st.session_state.janio_free          = st.session_state.get("ui_janio_free", True)


# -----------------------------------------------------------------------------
# Sidebar – Provedor, modelos, resumo e memória longa
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    # guarda o ID escolhido para uso na chamada de inferência
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

            # Ajusta o ID somente se for Together
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

    # Widgets controlam o session_state pelas keys (sem atribuição direta)
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
            if ok:
                st.success("Memória de longo prazo salva!")
            else:
                st.error("Falha ao salvar memória de longo prazo.")
        else:
            st.info("Ainda não há resposta do assistente nesta sessão.")

    st.caption("As interações completas aparecem na área principal.")


# -----------------------------------------------------------------------------
# EXIBIR HISTÓRICO (completo)
# -----------------------------------------------------------------------------
st.markdown("### 🗂️ Histórico completo")
with st.container():
    interacoes = carregar_interacoes(n=50)
    for r in interacoes:
        role = r.get("role", "user")
        content = r.get("content", "")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)


# -----------------------------------------------------------------------------
# ENVIO DO USUÁRIO + STREAMING (OpenRouter/Together, anti-<think>, PT-BR)
# -----------------------------------------------------------------------------
entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    # salva e mostra a fala do usuário
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    # prompt + histórico
    prompt = construir_prompt_com_narrador()
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

    # System extra para PT-BR e sem <think>
    suppress_think_ptbr = {
        "role": "system",
        "content": (
            "Responda exclusivamente em português do Brasil. "
            "Nunca inclua rascunhos de raciocínio nem use as tags <think>...</think>. "
            "Forneça apenas a resposta final ao leitor, no tom narrativo solicitado."
        ),
    }
    messages = [suppress_think_ptbr, {"role": "system", "content": prompt}] + historico

    payload = {
        "model": model_to_call,
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.85,
        "stream": True,
    }
    # pequena ajuda: se Together, corta se vier fechamento de tag
    if prov == "Together":
        payload["stop"] = ["</think>"]

    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    # Streaming estilo “digitação” com filtro leve
    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        try:
            with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
                    resposta_txt = "[ERRO STREAM]"
                else:
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
                            if delta:
                                resposta_txt += delta
                                # “digitação” suave
                                if time.time() - last_update > 0.10:
                                    placeholder.markdown(resposta_txt + "▌")
                                    last_update = time.time()
                        except Exception:
                            continue
        except Exception as e:
            st.error(f"Erro no streaming: {e}")
            resposta_txt = "[Erro ao gerar resposta]"

        # flush final
        placeholder.markdown(resposta_txt or "[Sem conteúdo]")

        # Validação sintática
        if not resposta_valida(resposta_txt):
            st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar...")
            try:
                regen = requests.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model_to_call,
                        "messages": messages,  # reaproveita system + prompt + histórico
                        "max_tokens": 900,
                        "temperature": 0.85,
                        "stream": False,
                    },
                    timeout=180,
                )
                if regen.status_code == 200:
                    resposta_txt = regen.json()["choices"][0]["message"]["content"].strip()
                    placeholder.markdown(resposta_txt)
                else:
                    st.error(f"Erro ao regenerar: {regen.status_code} - {regen.text}")
            except Exception as e:
                st.error(f"Erro ao regenerar: {e}")

        # Validação semântica (com OpenAI embeddings), compara última entrada do user vs resposta
        if len(st.session_state.session_msgs) >= 1 and resposta_txt and resposta_txt != "[ERRO STREAM]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]  # última entrada do user
            alerta = verificar_quebra_semantica_openai(texto_anterior, resposta_txt)
            if alerta:
                st.info(alerta)

        # Salva resposta + reforça memórias relevantes
        salvar_interacao("assistant", resposta_txt)
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt})
        try:
            # reforço das memórias que voltaram a ser relevantes (baseado na própria resposta)
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
