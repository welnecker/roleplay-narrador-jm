# -*- coding: utf-8 -*-
# main_jm.py — build estendido (~700 linhas)
# Requisitos: streamlit, requests, gspread, oauth2client, numpy, openai
# Segredos esperados em st.secrets:
# - OPENAI_API_KEY (para embeddings)
# - OPENROUTER_API_KEY (OpenRouter)
# - TOGETHER_API_KEY (Together)
# - GOOGLE_CREDS_JSON (service account JSON do Google)
# - SHEET_KEY (opcional; se ausente, usa chave padrão abaixo)
# Google Sheet esperado por chave em SHEET_KEY:
#   abas: interacoes_jm, memoria_longa_jm, perfil_jm (p/ resumos)
#   schema da aba memoria_longa_jm (ordem das colunas):
#       timestamp | texto | tags | embedding_json | score
#   (esta ordem está 100% alinhada a este script.)

import streamlit as st
import requests
import gspread
import json
import time
import re
import numpy as np
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from typing import List, Tuple, Optional, Dict, Any

# ======================================================================
# CONFIG BÁSICA E CONSTANTES
# ======================================================================
st.set_page_config(
    page_title="Mary / Jânio — Novela Interativa (JM)",
    page_icon="🌹",
    layout="wide",
)

SHEET_KEY = st.secrets.get("SHEET_KEY", "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ======================================================================
# CONEXÃO COM PLANILHA (c/ cache)
# ======================================================================
@st.cache_resource(show_spinner=False)
def conectar_planilha():
    """Autentica na planilha do Google via service account e retorna o objeto Spreadsheet.
    Fica em cache para evitar reautenticações a cada ação de UI.
    """
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"]) if "GOOGLE_CREDS_JSON" in st.secrets else None
        if not creds_dict:
            st.error("Faltou configurar GOOGLE_CREDS_JSON em st.secrets.")
            st.stop()
        # normaliza \n da chave privada
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(SHEET_KEY)
        return sh
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

sh = conectar_planilha()

@st.cache_data(ttl=20, show_spinner=False)
def get_ws(nome: str):
    """Retorna worksheet pelo nome, ou None se não existir."""
    if not sh:
        return None
    try:
        return sh.worksheet(nome)
    except Exception:
        return None

# ======================================================================
# CLIENTE OPENAI PARA EMBEDDINGS
# ======================================================================
try:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _client = None

EMBED_MODEL = "text-embedding-3-small"  # estável/rápido/barato


def _embedding(texto: str) -> Optional[np.ndarray]:
    """Gera embedding para um texto. Retorna np.ndarray float32 ou None."""
    if not _client:
        return None
    try:
        r = _client.embeddings.create(model=EMBED_MODEL, input=texto)
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception as e:
        st.warning(f"Falha no embedding: {e}")
        return None

# ======================================================================
# MEMÓRIA LONGA — Sheets + embeddings
#   Schema fixo (ordem): timestamp | texto | tags | embedding_json | score
# ======================================================================

def _serialize_vec(vec: Optional[np.ndarray]) -> str:
    if vec is None:
        return "[]"
    try:
        return json.dumps(vec.tolist(), separators=(",", ":"))
    except Exception:
        return "[]"


def _deserialize_vec(s: str) -> np.ndarray:
    try:
        a = np.array(json.loads(s), dtype=np.float32)
        if a.ndim == 1 and a.size >= 10:
            return a
    except Exception:
        pass
    return np.zeros(1, dtype=np.float32)


def memoria_longa_schema_ok() -> bool:
    """Valida se a aba memoria_longa_jm possui as colunas nas posições esperadas."""
    ws = get_ws("memoria_longa_jm")
    if not ws:
        return False
    try:
        headers = ws.row_values(1)
    except Exception:
        return False
    alvo = ["timestamp", "texto", "tags", "embedding_json", "score"]
    return [h.strip().lower() for h in headers[:5]] == alvo


def salvar_memoria_longa(texto: str, tags: str = "auto", score: float = 1.0) -> bool:
    """Salva uma memória longa na aba memoria_longa_jm.
    Colunas: timestamp, texto, tags, embedding_json, score
    """
    ws = get_ws("memoria_longa_jm")
    if not ws:
        st.warning("Aba 'memoria_longa_jm' não encontrada.")
        return False
    emb = _embedding(texto) if texto else None
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [ts, texto.strip(), (tags or "").strip(), _serialize_vec(emb), float(score)]
    try:
        ws.append_row(linha, value_input_option="RAW")
        return True
    except Exception as e:
        st.warning(f"Falha ao salvar memória longa: {e}")
        return False


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78) -> List[Tuple[str, float, float, float]]:
    """Busca as top-K memórias: retorna lista de tuplas (texto, score, sim, rr)
    rr = 0.7*similaridade + 0.3*score (rank de relevância).
    """
    ws = get_ws("memoria_longa_jm")
    if not ws:
        return []
    try:
        registros = ws.get_all_records()
    except Exception as e:
        st.warning(f"Erro ao ler memoria_longa_jm: {e}")
        return []
    if not registros:
        return []
    q = _embedding(query_text or "")
    if q is None:
        return []
    out: List[Tuple[str, float, float, float]] = []
    for row in registros:
        t = (row.get("texto") or "").strip()
        emb_s = row.get("embedding_json") or "[]"
        try:
            sc = float(row.get("score", 1.0) or 1.0)
        except Exception:
            sc = 1.0
        if not t:
            continue
        v = _deserialize_vec(emb_s)
        if v.ndim != 1 or v.size < 10:
            continue
        sim = _cos(q, v)
        if sim >= limiar:
            rr = 0.7 * sim + 0.3 * sc
            out.append((t, sc, sim, rr))
    out.sort(key=lambda x: x[3], reverse=True)
    return out[:k]


def memoria_longa_reforcar(textos_usados: List[str], delta: float = 0.2, max_sc: float = 2.0) -> None:
    """Aumenta levemente o score das memórias presentes em textos_usados."""
    ws = get_ws("memoria_longa_jm")
    if not ws or not textos_usados:
        return
    try:
        dados = ws.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = [h.strip().lower() for h in dados[0]]
        idx_texto = headers.index("texto")
        idx_score = headers.index("score")
        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) <= max(idx_texto, idx_score):
                continue
            t = (linha[idx_texto] or "").strip()
            if t in textos_usados:
                try:
                    sc = float(linha[idx_score] or 1.0)
                except Exception:
                    sc = 1.0
                sc = min(sc + delta, max_sc)
                ws.update_cell(i, idx_score + 1, sc)
    except Exception:
        pass


def memoria_longa_decadencia(fator: float = 0.97, min_sc: float = 0.1) -> None:
    """Aplica decadência a todos os scores (chame ocasionalmente)."""
    ws = get_ws("memoria_longa_jm")
    if not ws:
        return
    try:
        dados = ws.get_all_values()
        if not dados or len(dados) < 2:
            return
        headers = [h.strip().lower() for h in dados[0]]
        idx_score = headers.index("score")
        for i in range(2, len(dados) + 1):
            try:
                cell = ws.cell(i, idx_score + 1)
                sc = float(cell.value or 1.0)
            except Exception:
                sc = 1.0
            sc = max(sc * fator, min_sc)
            ws.update_cell(i, idx_score + 1, sc)
    except Exception:
        pass

# ======================================================================
# INTERAÇÕES — grava/recupera (aba interacoes_jm)
# ======================================================================

def salvar_interacao(role: str, content: str) -> None:
    ws = get_ws("interacoes_jm")
    if not ws:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.warning(f"Falha ao salvar interação: {e}")


@st.cache_data(ttl=15, show_spinner=False)
def carregar_ultimas_interacoes(n: int = 20) -> List[Dict[str, Any]]:
    ws = get_ws("interacoes_jm")
    if not ws:
        return []
    try:
        data = ws.get_all_records()
        return data[-n:]
    except Exception:
        return []

# ======================================================================
# RESUMO (aba perfil_jm) — aparece DEPOIS do histórico
# ======================================================================

def salvar_resumo(resumo: str) -> None:
    ws = get_ws("perfil_jm")
    if not ws:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # layout simples: [timestamp, resumo]
        ws.append_row([ts, resumo], value_input_option="RAW")
    except Exception as e:
        st.warning(f"Falha ao salvar resumo: {e}")


@st.cache_data(ttl=30, show_spinner=False)
def carregar_ultimo_resumo() -> Optional[str]:
    ws = get_ws("perfil_jm")
    if not ws:
        return None
    try:
        vals = ws.get_all_values()
        if len(vals) >= 2 and len(vals[-1]) >= 2:
            return vals[-1][1]
    except Exception:
        pass
    return None

# ======================================================================
# VALIDAÇÕES — sintática + semântica (quebra de continuidade)
# ======================================================================

PADROES_INVALIDOS = [
    r"check if.*string",
    r"#\s?1(\.\d+)+",
    r"\d{10,}",
    r"the cmd package",
    r"(111\s?)+",
    r"#+\s*\d+",
    r"\bimport\s",
    r"\bdef\s",
    r"```",
    r"class\s",
]


def resposta_valida(texto: str) -> bool:
    t = texto.lower()
    for p in PADROES_INVALIDOS:
        if re.search(p, t):
            return False
    return True


def similaridade_semantica(a: str, b: str) -> float:
    ea = _embedding(a or "")
    eb = _embedding(b or "")
    if ea is None or eb is None:
        return 1.0  # se não der pra calcular, não acusa
    return _cos(ea, eb)


def alerta_quebra_semantica(pergunta: str, resposta: str, limite: float = 0.60) -> str:
    sim = similaridade_semantica(pergunta, resposta)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# ======================================================================
# GUARD-RAILS / REGRAS DE CENA
# ======================================================================
VIRGINIDADE_MARY = True  # Mary permanece virgem por vontade própria
JANIO_LIVRE = True       # Jânio pode ter relações com qualquer personagem

REGRAS_MARY = (
    "Mary é virgem e escolhe manter-se casta até o encontro definitivo com Jânio. "
    "Ela pode namorar, beijar, flertar e viver tensão romântica, mas recusa sexo. "
    "Nunca descreva Mary transando, nem algo que implique penetração ou perda da virgindade. "
    "Se a cena ameaçar ultrapassar limites, Mary impõe limites com firmeza e elegância."
)

REGRAS_JANIO = (
    "Jânio não tem essa limitação: ele pode ter relações consensuais com outros personagens, "
    "desde que a cena mantenha o bom gosto e evite descrições gráficas explícitas."
)

BASE_SYSTEM = f"""
Você é o narrador de uma novela em português do Brasil. Escreva em 3ª pessoa a narração e use 1ª pessoa para falas e pensamentos dos personagens.
Mantenha continuidade com o histórico e memórias fornecidas. Seja sensorial sem apelar para pornografia explícita.

REGRAS FIXAS:
- {REGRAS_MARY if VIRGINIDADE_MARY else ''}
- {REGRAS_JANIO if JANIO_LIVRE else ''}
- Não invente falas do usuário. O usuário só fala quando ele enviar mensagem.
- Evite comandos técnicos ([SFX], (cut), etc.).
""".strip()

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



def _stream_openrouter(model_id: str, messages, temperature=0.9, max_tokens=900):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    resposta_txt = ""
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j["choices"][0]["delta"].get("content", "")
                if delta:
                    resposta_txt += delta
                    yield resposta_txt
            except Exception:
                continue


def _stream_together(model_id: str, messages, temperature=0.9, max_tokens=900):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    resposta_txt = ""
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        if r.status_code == 400:
            raise RuntimeError(f"Together 400: {r.text}")
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.strip() == "data: [DONE]":
                break
            if raw.startswith("data:"):
                data = raw[5:].strip()
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"].get("content", "")
                    # Perplexity pode enviar <think>…</think>; exibimos integralmente.
                    if delta:
                        resposta_txt += delta
                        yield resposta_txt
                except Exception:
                    continue


def gerar_resposta_stream(modelo_cfg: Dict[str, str], historico_msgs: List[Dict[str, str]]):
    """Injeta memórias longas relevantes e abre stream do provedor."""
    prov = modelo_cfg["prov"]
    model_id = modelo_cfg["id"]

    # memoria longa baseada na última fala do usuário
    ult_user = next((m["content"] for m in reversed(historico_msgs) if m["role"] == "user"), "")
    topk = memoria_longa_buscar_topk(ult_user, k=st.session_state.get("k_memoria_longa", 3),
                                     limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)))
    bloco_mem = ("\n\nMemórias relevantes:\n" + "\n".join(f"- {t}" for (t, _sc, _sim, _rr) in topk)) if topk else ""

    system = {"role": "system", "content": BASE_SYSTEM + bloco_mem}
    msgs = [system] + historico_msgs

    if prov == "openrouter":
        return _stream_openrouter(model_id, msgs)
    elif prov == "together":
        return _stream_together(model_id, msgs)
    else:
        raise ValueError("Provedor desconhecido")

# ======================================================================
# ESTADO INICIAL DA UI
# ======================================================================
if "hist" not in st.session_state:
    st.session_state.hist = []  # histórico da sessão atual (mostrado na tela)
if "modelo_nome" not in st.session_state:
    st.session_state.modelo_nome = list(MODELOS.keys())[0]
if "use_memoria_longa" not in st.session_state:
    st.session_state.use_memoria_longa = True
if "k_memoria_longa" not in st.session_state:
    st.session_state.k_memoria_longa = 3
if "limiar_memoria_longa" not in st.session_state:
    st.session_state.limiar_memoria_longa = 0.78
if "app_emocao_oculta" not in st.session_state:
    st.session_state.app_emocao_oculta = "nenhuma"
if "app_bloqueio_intimo" not in st.session_state:
    st.session_state.app_bloqueio_intimo = False

# ======================================================================
# TOPO DA PÁGINA / INFO RÁPIDA
# ======================================================================
col_head = st.container()
with col_head:
    st.title("🌹 Mary / 🎸 Jânio — Novela Interativa (JM)")
    st.caption("Se algo travar, troque de modelo no painel lateral. As memórias longas são salvas com embeddings.")

# ======================================================================
# LATERAL — Config, modelos, memória longa, resumo
# ======================================================================
with st.sidebar:
    st.subheader("Configuração")
    st.session_state.modelo_nome = st.selectbox(
        "Modelo",
        list(MODELOS.keys()),
        index=list(MODELOS.keys()).index(st.session_state.modelo_nome),
    )

    st.checkbox(
        "Usar memória longa no prompt",
        value=st.session_state.use_memoria_longa,
        key="ui_use_memoria_longa",
    )
    st.session_state.use_memoria_longa = st.session_state.get("ui_use_memoria_longa", True)

    st.slider("Top-K memórias", 1, 5, st.session_state.k_memoria_longa, 1, key="ui_k_memoria")
    st.session_state.k_memoria_longa = st.session_state.get("ui_k_memoria", 3)

    st.slider(
        "Limiar de similaridade",
        0.50, 0.95,
        float(st.session_state.limiar_memoria_longa),
        0.01,
        key="ui_limiar_memoria",
    )
    st.session_state.limiar_memoria_longa = st.session_state.get("ui_limiar_memoria", 0.78)

    st.markdown("---")
    st.subheader("Diretrizes de Cena")
    st.checkbox(
        "Bloquear avanços íntimos sem ordem",
        value=st.session_state.app_bloqueio_intimo,
        key="ui_bloq_intimo",
    )
    st.session_state.app_bloqueio_intimo = st.session_state.get("ui_bloq_intimo", False)

    st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.app_emocao_oculta),
        key="ui_emocao",
    )
    st.session_state.app_emocao_oculta = st.session_state.get("ui_emocao", "nenhuma")

    st.markdown("---")
    st.subheader("Memória Longa — Ações")
    if st.button("💾 Salvar última resposta como memória", use_container_width=True):
        ult = next((m["content"] for m in reversed(st.session_state.hist) if m["role"] == "assistant"), "")
        if ult:
            ok = salvar_memoria_longa(ult[:1200], tags="auto")
            st.success("Memória salva." if ok else "Falha ao salvar memória.")
        else:
            st.info("Ainda não há resposta desta sessão.")

    if st.button("⬇️ Aplicar decadência suave (scores)", use_container_width=True):
        memoria_longa_decadencia(0.97)
        st.success("Decadência aplicada.")

    st.markdown("---")
    st.subheader("Resumo do Capítulo")
    if st.button("📝 Gerar e salvar resumo", use_container_width=True):
        trecho = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.hist[-10:]]) or "(vazio)"
        user_prompt = (
            "Resuma como capítulo de novela brasileiro, coeso, sem pornografia, mantendo as emoções.\n\n" + trecho
        )
        modelo_cfg = MODELOS[st.session_state.modelo_nome]
        try:
            placeholder = st.empty()
            out = ""
            with st.spinner("Resumindo..."):
                for parcial in gerar_resposta_stream(modelo_cfg, [
                    {"role": "user", "content": user_prompt}
                ]):
                    out = parcial
                    placeholder.markdown(out)
            if out.strip():
                salvar_resumo(out.strip())
                st.success("Resumo salvo na aba perfil_jm.")
        except Exception as e:
            st.error(f"Erro ao resumir: {e}")

# ======================================================================
# HISTÓRICO — primeiro o da planilha, depois o da sessão
# ======================================================================
recentes = carregar_ultimas_interacoes(15)
for r in recentes:
    with st.chat_message(r.get("role", "user")):
        st.markdown(r.get("content", ""))

for m in st.session_state.hist:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

# Resumo ao final
ultimo_resumo = carregar_ultimo_resumo()
if ultimo_resumo:
    with st.chat_message("assistant"):
        st.markdown("### 🧠 Resumo do capítulo anterior\n\n" + ultimo_resumo)

# ======================================================================
# ENTRADA DO USUÁRIO + STREAM + FALLBACKS + VALIDAÇÕES + REFORÇOS
# ======================================================================
entrada = st.chat_input("Digite sua direção de cena…")
if entrada:
    # salva user
    st.chat_message("user").markdown(entrada)
    st.session_state.hist.append({"role": "user", "content": entrada})
    salvar_interacao("user", entrada)

    # constrói histórico compacto (últimas 6 da planilha + sessão)
    contexto_base: List[Dict[str, str]] = []
    for r in recentes[-6:]:
        contexto_base.append({"role": r.get("role", "user"), "content": r.get("content", "")})

    # sinalização de emoção / bloqueio — injeta como fala do sistema logo antes do histórico
    regra_intimo = ("\n⛔ Jamais antecipe encontros ou cenas íntimas sem ordem explícita do roteirista."
                    if st.session_state.app_bloqueio_intimo else "")
    emocao = st.session_state.app_emocao_oculta
    sinal = {
        "role": "system",
        "content": (
            BASE_SYSTEM
            + f"\n\n🎭 Emoção oculta: {emocao}."
            + regra_intimo
        ),
    }

    contexto = contexto_base + st.session_state.hist[-8:]
    contexto = [sinal] + contexto  # system sinalizador no topo

    modelo_cfg = MODELOS[st.session_state.modelo_nome]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""

        # 1) streaming principal
        try:
            for parcial in gerar_resposta_stream(modelo_cfg, contexto):
                resposta_txt = parcial
                placeholder.markdown(resposta_txt if resposta_txt.strip() else "[Gerando…]")
        except Exception as e:
            placeholder.markdown(f"[Erro no streaming: {e}]")

        # 2) fallback sem stream
        if not resposta_txt.strip():
            try:
                prov = modelo_cfg["prov"]; model_id = modelo_cfg["id"]
                if prov == "openrouter":
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                else:
                    url = "https://api.together.xyz/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [{"role": "system", "content": BASE_SYSTEM}] + contexto[1:],
                    "temperature": 0.9,
                    "max_tokens": 900,
                    "stream": False,
                }
                r = requests.post(url, headers=headers, json=payload, timeout=180)
                if r.status_code == 200:
                    resposta_txt = r.json()["choices"][0]["message"]["content"].strip()
                    placeholder.markdown(resposta_txt or "[Sem conteúdo]")
                else:
                    placeholder.markdown(f"Fallback falhou: {r.status_code} - {r.text}")
            except Exception as e:
                placeholder.markdown(f"[Erro no fallback: {e}]")

        # 3) validação sintática + regeneração simples se necessário
        if resposta_txt and not resposta_valida(resposta_txt):
            st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar…")
            try:
                prov = modelo_cfg["prov"]; model_id = modelo_cfg["id"]
                if prov == "openrouter":
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                else:
                    url = "https://api.together.xyz/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
                regen = requests.post(
                    url, headers=headers,
                    json={
                        "model": model_id,
                        "messages": [{"role": "system", "content": BASE_SYSTEM}] + contexto[1:],
                        "temperature": 0.9,
                        "max_tokens": 900,
                        "stream": False,
                    }, timeout=180
                )
                if regen.status_code == 200:
                    resposta_txt = regen.json()["choices"][0]["message"]["content"].strip()
                    placeholder.markdown(resposta_txt)
                else:
                    st.error(f"Erro ao regenerar: {regen.status_code} - {regen.text}")
            except Exception as e:
                st.error(f"Erro ao regenerar: {e}")

    # salva resposta SEMPRE
    salvar_interacao("assistant", resposta_txt or "[Sem conteúdo]")
    st.session_state.hist.append({"role": "assistant", "content": resposta_txt or "[Sem conteúdo]"})

    # alerta de continuidade semântica (entrada do user vs resposta)
    try:
        alert = alerta_quebra_semantica(entrada, resposta_txt)
        if alert:
            st.info(alert)
    except Exception:
        pass

    # reforça memórias usadas com base na própria resposta
    if st.session_state.use_memoria_longa and resposta_txt:
        try:
            usados = [t for (t, _sc, _sim, _rr) in memoria_longa_buscar_topk(
                query_text=resposta_txt,
                k=st.session_state.k_memoria_longa,
                limiar=float(st.session_state.limiar_memoria_longa),
            )]
            memoria_longa_reforcar(usados)
        except Exception:
            pass

    # auto-salva um pedaço como memória longa (limita tamanho)
    if st.session_state.use_memoria_longa and resposta_txt and len(resposta_txt) > 300:
        try:
            salvar_memoria_longa(resposta_txt[:1200], tags="auto")
        except Exception:
            pass

# ======================================================================
# FERRAMENTAS DE ADMIN (opcionais) — exibidas no final
# ======================================================================
with st.expander("🛠️ Utilidades de manutenção"):
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Verificar schema 'memoria_longa_jm'", use_container_width=True):
            st.write("Schema OK" if memoria_longa_schema_ok() else "Schema divergente (esperado: timestamp|texto|tags|embedding_json|score)")
    with colB:
        if st.button("Aplicar decadência (rápido)", use_container_width=True):
            memoria_longa_decadencia(0.97)
            st.success("Decadência aplicada.")
    with colC:
        if st.button("Reforçar top memórias do último output", use_container_width=True):
            ult = next((m["content"] for m in reversed(st.session_state.hist) if m["role"] == "assistant"), "")
            if ult:
                usados = [t for (t, _sc, _sim, _rr) in memoria_longa_buscar_topk(ult, k=3, limiar=0.78)]
                memoria_longa_reforcar(usados)
                st.success("Reforço aplicado em memórias relevantes.")
            else:
                st.info("Sem resposta ainda nesta sessão.")

# ======================================================================
# FIM
# ======================================================================
