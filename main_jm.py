# main.py

# ============================================================
# Narrador JM — Roleplay adulto (sem pornografia explícita)
# Compatível com o método antigo: GOOGLE_CREDS_JSON + oauth2client
# ============================================================

import os
import re
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict, Any

import streamlit as st
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from gspread.exceptions import APIError

# ---- Backoff p/ 429 do Sheets ----
def _retry_429(callable_fn, *args, _retries=5, _base=0.6, **kwargs):
    for i in range(_retries):
        try:
            return callable_fn(*args, **kwargs)
        except APIError as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                time.sleep((_base * (2 ** i)) + random.uniform(0, 0.25))
                continue
            raise
    # última tentativa
    return callable_fn(*args, **kwargs)

# ---- Cache de leitura (TTL curto) ----
@st.cache_data(ttl=45, show_spinner=False)
def _sheet_all_records_cached(sheet_name: str):
    ws = _ws(sheet_name, create_if_missing=False)
    if not ws:
        return []
    return _retry_429(ws.get_all_records)

@st.cache_data(ttl=45, show_spinner=False)
def _sheet_all_values_cached(sheet_name: str):
    ws = _ws(sheet_name, create_if_missing=False)
    if not ws:
        return []
    return _retry_429(ws.get_all_values)

def _invalidate_sheet_caches():
    try:
        _sheet_all_records_cached.clear()
        _sheet_all_values_cached.clear()
    except Exception:
        pass

# (Opcional) Embeddings OpenAI para verificação semântica/memória longa
try:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    OPENAI_OK = bool(st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OPENAI_CLIENT = None
    OPENAI_OK = False

# =========================
# CONFIG BÁSICA DO APP
# =========================

st.set_page_config(page_title="Narrador JM", page_icon="🎬", layout="wide")

# Gate 18+
if "age_ok" not in st.session_state:
    st.session_state.age_ok = False
if not st.session_state.age_ok:
    st.title("🔞 Conteúdo adulto")
    st.caption("Narrativa adulta, sensual, **sem pornografia explícita**. Confirme para prosseguir.")
    if st.checkbox("Confirmo que tenho 18 anos ou mais e desejo prosseguir."):
        st.session_state.age_ok = True
    st.stop()

# =========================
# GOOGLE SHEETS — MODO ANTIGO
# =========================

PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"

def conectar_planilha():
    """
    Conecta via GOOGLE_CREDS_JSON (modo antigo/estável).
    Espera:
    - st.secrets["GOOGLE_CREDS_JSON"]: string JSON do service account
    - (opcional) st.secrets["SPREADSHEET_ID"]: ID da planilha
    """
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        spreadsheet_id = st.secrets.get("SPREADSHEET_ID", "").strip() or PLANILHA_ID_PADRAO
        return client.open_by_key(spreadsheet_id)
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# Abas esperadas
TAB_INTERACOES = "interacoes_jm"  # timestamp | role | content
TAB_PERFIL     = "perfil_jm"      # timestamp | resumo
TAB_MEMORIAS   = "memorias_jm"    # tipo | conteudo
TAB_ML         = "memoria_longa_jm"  # texto | embedding | tags | timestamp | score

def _ws(name: str, create_if_missing: bool = True):
    if not planilha:
        return None
    try:
        return planilha.worksheet(name)
    except Exception:
        if not create_if_missing:
            return None
        try:
            ws = planilha.add_worksheet(title=name, rows=5000, cols=10)
            # cria cabeçalhos padrão
            if name == TAB_INTERACOES:
                _retry_429(ws.append_row, ["timestamp", "role", "content"])
            elif name == TAB_PERFIL:
                _retry_429(ws.append_row, ["timestamp", "resumo"])
            elif name == TAB_MEMORIAS:
                _retry_429(ws.append_row, ["tipo", "conteudo"])
            elif name == TAB_ML:
                _retry_429(ws.append_row, ["texto", "embedding", "tags", "timestamp", "score"])
            return ws
        except Exception:
            return None

# =========================
# UTILIDADES: MEMÓRIAS / HISTÓRICO
# =========================

def carregar_memorias_brutas() -> Dict[str, List[str]]:
    """Lê 'memorias_jm' e devolve um dict {tag_lower: [linhas]} com cache TTL."""
    try:
        regs = _sheet_all_records_cached(TAB_MEMORIAS)
        buckets: Dict[str, List[str]] = {}
        for r in regs:
            tag = (r.get("tipo","") or "").strip().lower()
            txt = (r.get("conteudo","") or "").strip()
            if tag and txt:
                buckets.setdefault(tag, []).append(txt)
        return buckets
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return {}

def persona_block(nome: str, buckets: dict, max_linhas: int = 8) -> str:
    """Monta bloco compacto da persona (ordena por prefixos úteis)."""
    tag = f"[{nome}]"
    linhas = buckets.get(tag, [])
    ordem = ["OBJ:", "TAT:", "LV:", "VOZ:", "BIO:", "ROTINA:", "LACOS:", "APS:", "CONFLITOS:"]
    def peso(l):
        up = l.upper()
        for i, p in enumerate(ordem):
            if up.startswith(p):
                return i
        return len(ordem)
    linhas_ordenadas = sorted(linhas, key=peso)[:max_linhas]
    titulo = "Jânio" if nome in ("janio","jânio") else "Mary" if nome=="mary" else nome.capitalize()
    return (f"{titulo}:\n- " + "\n- ".join(linhas_ordenadas)) if linhas_ordenadas else ""

def carregar_resumo_salvo() -> str:
    """Busca o último resumo da aba 'perfil_jm' (cabeçalho: timestamp | resumo) com cache TTL."""
    try:
        registros = _sheet_all_records_cached(TAB_PERFIL)
        for r in reversed(registros):
            txt = (r.get("resumo") or "").strip()
            if txt:
                return txt
        return ""
    except Exception as e:
        st.warning(f"Erro ao carregar resumo salvo: {e}")
        return ""

def salvar_resumo(resumo: str):
    """Salva uma nova linha em 'perfil_jm' (timestamp | resumo) e invalida caches."""
    try:
        aba = _ws(TAB_PERFIL)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _retry_429(aba.append_row, [timestamp, resumo], value_input_option="RAW")
        _invalidate_sheet_caches()
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def carregar_interacoes(n: int = 20):
    """
    Carrega últimas n interações (role, content) usando cache de sessão
    para evitar leituras repetidas.
    """
    cache = st.session_state.get("_cache_interacoes", None)
    if cache is None:
        regs = _sheet_all_records_cached(TAB_INTERACOES)
        st.session_state["_cache_interacoes"] = regs
        cache = regs
    return cache[-n:] if len(cache) > n else cache

def salvar_interacao(role: str, content: str):
    """Append no Sheets + atualiza cache local (sem reler) com backoff 429."""
    if not planilha:
        return
    try:
        aba = _ws(TAB_INTERACOES)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, f"{role or ''}".strip(), f"{content or ''}".strip()]

        _retry_429(aba.append_row, row, value_input_option="RAW")

        # atualiza cache local
        lst = st.session_state.get("_cache_interacoes")
        if isinstance(lst, list):
            lst.append({"timestamp": row[0], "role": row[1], "content": row[2]})
        else:
            st.session_state["_cache_interacoes"] = [{"timestamp": row[0], "role": row[1], "content": row[2]}]

        _invalidate_sheet_caches()

    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

# =========================
# EMBEDDINGS / SIMILARIDADE
# =========================

def gerar_embedding_openai(texto: str):
    if not OPENAI_OK:
        return None
    try:
        resp = OPENAI_CLIENT.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def verificar_quebra_semantica_openai(texto1: str, texto2: str, limite=0.6) -> str:
    if not OPENAI_OK:
        return ""
    e1 = gerar_embedding_openai(texto1)
    e2 = gerar_embedding_openai(texto2)
    if e1 is None or e2 is None:
        return ""
    sim = cosine_similarity(e1, e2)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# =========================
# MEMÓRIA LONGA (Sheets + Embeddings/OpenAI opcional)
# =========================

def _sheet_ensure_memoria_longa():
    """Retorna a aba memoria_longa_jm se existir (não cria automaticamente)."""
    return _ws(TAB_ML, create_if_missing=False)

def _serialize_vec(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))

def _deserialize_vec(s: str) -> np.ndarray:
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        return np.zeros(1, dtype=float)

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
    """Salva uma memória com embedding (se possível) e score inicial. Invalida cache."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        st.warning("Aba 'memoria_longa_jm' não encontrada — crie com cabeçalhos: texto | embedding | tags | timestamp | score")
        return False
    emb = gerar_embedding_openai(texto)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [texto.strip(), _serialize_vec(emb) if emb is not None else "", (tags or "").strip(), ts, 1.0]
    try:
        _retry_429(aba.append_row, linha, value_input_option="RAW")
        _invalidate_sheet_caches()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar memória longa: {e}")
        return False

def memoria_longa_listar_registros():
    """Retorna todos os registros da aba memoria_longa_jm (cache TTL)."""
    try:
        return _sheet_all_records_cached(TAB_ML)
    except Exception:
        return []

def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-zà-ú0-9]+", (s or "").lower()))

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78):
    """Top-K memórias. Usa embeddings se existir; senão, Jaccard simples."""
    try:
        dados = _sheet_all_records_cached(TAB_ML)
    except Exception as e:
        st.warning(f"Erro ao carregar memoria_longa_jm: {e}")
        return []

    q = gerar_embedding_openai(query_text) if OPENAI_OK else None
    candidatos = []
    for row in dados:
        texto = (row.get("texto") or "").strip()
        emb_s = (row.get("embedding") or "").strip()
        try:
            score = float(row.get("score", 1.0) or 1.0)
        except Exception:
            score = 1.0
        if not texto:
            continue

        if q is not None and emb_s:
            vec = _deserialize_vec(emb_s)
            if vec.ndim == 1 and vec.size >= 10:
                sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
            else:
                sim = 0.0
        else:
            # fallback lexical
            s1 = _tokenize(texto)
            s2 = _tokenize(query_text)
            sim = len(s1 & s2) / max(1, len(s1 | s2))

        if sim >= limiar:
            rr = 0.7 * sim + 0.3 * score
            candidatos.append((texto, score, sim, rr))
    candidatos.sort(key=lambda x: x[3], reverse=True)
    return candidatos[:k]

def memoria_longa_reforcar(textos_usados: list):
    """Aumenta o score das memórias usadas (pequeno reforço) com backoff + correção de índices."""
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = _sheet_all_values_cached(TAB_ML)
        if not dados or len(dados) < 2:
            return
        headers = dados[0]  # <- CORREÇÃO: cabeçalho é a primeira linha
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
                sc = min(sc + 0.2, 2.0)
                _retry_429(aba.update_cell, i, idx_score + 1, sc)
        _invalidate_sheet_caches()
    except Exception:
        pass


# =========================
# ROMANCE (FASES) + MOMENTO
# =========================

FASES_ROMANCE: Dict[int, Dict[str, str]] = {
    0: {"nome": "Estranhos",
        "permitidos": "olhares; near-miss (mesmo café/rua/ônibus); detalhe do ambiente",
        "proibidos": "troca de nomes; toques; conversa pessoal"},
    1: {"nome": "Percepção",
        "permitidos": "cumprimento neutro; pergunta impessoal curta",
        "proibidos": "contato físico; confidências"},
    2: {"nome": "Conhecidos",
        "permitidos": "troca de nomes; pequena ajuda; 1 pergunta pessoal leve",
        "proibidos": "toque prolongado; encontro a sós planejado"},
    3: {"nome": "Amizade",
        "permitidos": "conversa 10–20 min; caminhar juntos; troca de contatos; 1 gesto de afeto leve (com consentimento)",
        "proibidos": "beijos; carícias intimistas"},
    4: {"nome": "Confiança / Quase",
        "permitidos": "confidências; abraço com consentimento expresso; marcar encontro futuro claro",
        "proibidos": "sexo; sexo oral/manual; pressa ou “provas de amor” físicas"},
    5: {"nome": "Compromisso / Encontro definitivo",
        "permitidos": "beijo prolongado; dormir juntos; consumação **implícita** (fade-to-black); manhã seguinte sugerida",
        # Proibições originais retiradas para permitir explícito:
        "proibidos": ""},
}

FLAG_FASE_TXT_PREFIX = "FLAG: mj_fase="

def _fase_label(n: int) -> str:
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE[0])
    return f"{int(n)} — {d['nome']}"

def mj_set_fase(n: int, persist: bool=True):
    n = max(0, min(max(FASES_ROMANCE.keys()), int(n)))
    st.session_state.mj_fase = n
    if persist:
        try:
            memoria_longa_salvar(f"{FLAG_FASE_TXT_PREFIX}{n}", tags="[flag]")
        except Exception:
            pass

def mj_carregar_fase_inicial() -> int:
    if "mj_fase" in st.session_state:
        return int(st.session_state.mj_fase)
    try:
        recs = memoria_longa_listar_registros()
        for r in reversed(recs):
            t = (r.get("texto") or "").strip()
            if t.startswith(FLAG_FASE_TXT_PREFIX):
                n = int(t.split("=")[1])
                st.session_state.mj_fase = n
                return n
    except Exception:
        pass
    st.session_state.mj_fase = 0
    return 0

# --------- Motor de Momento ----------
MOMENTOS = {
    0: {"nome": "Aproximação logística",
        "objetivo": "um acompanha o outro (ex.: até o píer), clima cordial",
        "permitidos": "gentilezas; proximidade leve; diálogo casual",
        "proibidos": "declaração; revelações íntimas; toques prolongados",
        "gatilhos": [r"\b(p[ií]er|acompanhar|vamos embora|te levo)\b"],
        "proximo": 1},
    1: {"nome": "Declaração",
        "objetivo": "um deles declara amor/ importância",
        "permitidos": "confissão afetiva; silêncio tenso; abraço curto",
        "proibidos": "negociação sexual; tirar roupas; exploração do corpo",
        "gatilhos": [r"\b(amo voc[eê]|te amo|n[aã]o paro de pensar)\b"],
        "proximo": 2},
    2: {"nome": "Revelação sensível",
        "objetivo": "Mary revela que é virgem / vulnerabilidade equivalente",
        "permitidos": "dizer 'sou virgem'; estipular limites; conforto mútuo",
        "proibidos": "carícias íntimas; tirar roupas",
        "gatilhos": [r"\b(sou virgem|nunca fiz|meu limite)\b"],
        "proximo": 3},
    3: {"nome": "Consentimento explícito",
        "objetivo": "alinhamento de limites e um 'sim' claro",
        "permitidos": "nomear fronteiras; pedir/receber consentimento; decidir 'agora sim'",
        # Proibição removida para permitir descrição explícita:
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eê])\b", r"\b(at[eé] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade (elíptica)",
        "objetivo": "intimidade sugerida (fade-to-black) / pós-ato implícito",
        "permitidos": "beijos longos; proximidade forte; fade-to-black; manhã seguinte implícita",
        # Proibição removida para permitir descrição gráfica:
        "proibidos": "",
        "gatilhos": [r"\b(quarto|cama|luz baixa|porta fechada|manh[aã] seguinte)\b"],
        "proximo": 4},
}

def _momento_label(n: int) -> str:
    m = MOMENTOS.get(int(n), MOMENTOS[0])
    return f"{int(n)} — {m['nome']}"

def detectar_momento_sugerido(texto: str, fallback: int = 0) -> int:
    t = (texto or "").lower()
    for i in range(4, -1, -1):
        for gx in MOMENTOS[i]["gatilhos"]:
            if re.search(gx, t, flags=re.IGNORECASE):
                return i
    # Removido o freio por termos explícitos: não força parada.
    return fallback

def clamp_momento(atual: int, proposto: int, max_steps: int) -> int:
    if proposto > atual + max_steps:
        return atual + max_steps
    if proposto < atual:
        return max(proposto, atual - 1)
    return proposto

def momento_set(n: int, persist: bool = True):
    n = max(0, min(max(MOMENTOS.keys()), int(n)))
    st.session_state.momento = n
    if persist:
        try:
            memoria_longa_salvar(f"FLAG: mj_momento={n}", tags="[flag]")
        except Exception:
            pass

def momento_carregar() -> int:
    if "momento" in st.session_state:
        return int(st.session_state.momento)
    try:
        recs = memoria_longa_listar_registros()
        for r in reversed(recs):
            t = (r.get("texto") or "").strip()
            if t.startswith("FLAG: mj_momento="):
                n = int(t.split("=")[1])
                st.session_state.momento = n
                return n
    except Exception:
        pass
    st.session_state.momento = 0
    return 0

def viola_momento(texto: str, momento: int) -> str:
    # Não bloquear/censurar conteúdo explícito por momento.
    return ""

# =========================
# PROVEDORES E MODELOS
# =========================

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
    key = api_ui_model_id.strip()
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in key:
        return "Qwen/Qwen3-Coder-480B-Instruct-FP8"
    if key.lower().startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return key

def api_config_for_provider(provider: str):
    if provider == "OpenRouter":
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            st.secrets.get("OPENROUTER_API_KEY", ""),
            MODELOS_OPENROUTER,
        )
    else:
        return (
            "https://api.together.xyz/v1/chat/completions",
            st.secrets.get("TOGETHER_API_KEY", ""),
            MODELOS_TOGETHER_UI,
        )

# =========================
# PROMPT BUILDER
# =========================

def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    # padrão elevado para 3 (explícito)
    calor = int(st.session_state.get("steam_level", st.session_state.get("nsfw_max_level", 3)))
    regras = f"""
⚖️ Regras de coerência:
- Narre em terceira pessoa; não se dirija ao leitor como "você".
- Consentimento claro antes de qualquer gesto significativo.
- Jânio não pressiona; respeita o ritmo de Mary.
- Linguagem sensual proporcional ao nível de calor ({calor}).
""".strip()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    if fase >= 5:
        # liberar explícito em fase 5
        regras += """
- Intimidade pode ser descrita com detalhes explícitos (sem necessidade de corte)."""
    else:
        regras += """
- Sem consumação em cena; foque em progressão coerente."""
    return prompt_base + "\n" + regras

def construir_prompt_com_narrador() -> str:
    memos = carregar_memorias_brutas()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS)
    proximo_nome = MOMENTOS[mdata["proximo"]]["nome"]
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")

    # Histórico do Sheets
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r['role']}: {r['content']}" for r in hist) if hist else "(sem histórico)"

    # Memória longa Top-K (texto apenas; se não houver, "(nenhuma)")
    ml_topk_txt = "(nenhuma)"
    if st.session_state.get("use_memoria_longa", True) and hist:
        try:
            topk = memoria_longa_buscar_topk(
                query_text=hist[-1]["content"],
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
            )
            if topk:
                ml_topk_txt = "\n".join([f"- {t}" for (t, _sc, _sim, _rr) in topk])
                st.session_state["_ml_topk_texts"] = [t for (t, *_rest) in topk]
            else:
                st.session_state["_ml_topk_texts"] = []
        except Exception:
            st.session_state["_ml_topk_texts"] = []
    else:
        st.session_state["_ml_topk_texts"] = []

    recorrentes = [c for (t, lst) in memos.items() if t == "[all]" for c in lst]
    st.session_state["_ml_recorrentes"] = recorrentes

    dossie = []
    mary = persona_block("mary", memos, 8)
    janio = persona_block("janio", memos, 8)
    if mary: dossie.append(mary)
    if janio: dossie.append(janio)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    prompt = f"""
Você é o Narrador de um roleplay dramático brasileiro, foque em Mary e Jânio. Não repita instruções nem títulos.

### Dossiê (personas)
{dossie_txt}

### Diretrizes gerais (ALL)
{chr(10).join(['- '+c for c in recorrentes]) if recorrentes else '(vazio)'}

### Perfil (resumo mais recente)
{perfil or "(vazio)"}

### Histórico recente (planilha)
{hist_txt}

### Estilo
- Use o estilo **{estilo}**:
{("- Frases curtas, cortes rápidos, foco em gesto/ritmo.") if estilo=="AÇÃO" else
("- Atmosfera sombria, subtexto, silêncio que pesa.") if estilo=="NOIR" else
("- Ritmo lento, tensão emocional, detalhes sensoriais (sem grafismo).")}

### Memória longa — Top-K relevantes
{ml_topk_txt}

### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}

### 🎯 Momento dramático (agora)
- Momento: {_momento_label(momento_atual)}
- Objetivo da cena: {mdata['objetivo']}
- Nesta cena, **permita**: {mdata['permitidos']}
- Evite/adiar: {mdata['proibidos']}
- **Micropassos:** avance no máximo **{int(st.session_state.get("max_avancos_por_cena",1))}** subpasso(s) rumo a: {proximo_nome}.
- Se o roteirista pedir salto maior, **negocie**: nomeie limites, peça consentimento, e **prepare** a transição (não pule etapas).

### Regra de saída
- Narre em **terceira pessoa**; não fale com "você".
- Não exiba rótulos/meta (ex.: "Microconquista:", "Gancho:").
- Entregue uma cena coesa e finalizada; feche com um gancho implícito.
""".strip()

    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# FILTROS DE SAÍDA
# =========================

def render_tail(t: str) -> str:
    if not t: return ""
    # remove rótulos meta e
    t = re.sub(r'^\s*\**\s*(microconquista|gancho)\s*:\s*.*$', '', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'&lt;\s*think\s*&gt;.*?&lt;\s*/\s*think\s*&gt;', '', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    return t

EXPL_PAT = re.compile(
    r"\b(seio[s]?|mamilos?|bunda|fio[- ]?dental|genit[aá]lia|ere[cç][aã]o|penetra[cç][aã]o|"
    r"boquete|gozada|gozo|sexo oral|chupar|enfiar)\b",
    flags=re.IGNORECASE
)

def classify_nsfw_level(t: str) -> int:
    if EXPL_PAT.search(t or ""):
        return 3 # explícito
    if re.search(r"\b(cintura|pesco[cç]o|costas|beijo prolongado|respira[cç][aã]o curta)\b", (t or ""), re.IGNORECASE):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aã]os dadas|beijo)\b", (t or ""), re.IGNORECASE):
        return 1
    return 0

def sanitize_explicit(t: str, max_level: int, action: str) -> str:
    # Liberação: se o conteúdo for de nível <= max_level, retorna tal como está.
    lvl = classify_nsfw_level(t)
    if lvl <= max_level:
        return t
    # Se extrapolar o máximo definido, não cortar por padrão (liberar NSFW). Apenas retorna sem censura.
    return t

def redact_for_logs(t: str) -> str:
    if not t: return ""
    # Mantém ofuscação no log se desejar (opcional). Pode deixar como estava:
    t = re.sub(EXPL_PAT, "[…]", t, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

# =========================
# UI — CABEÇALHO E CONTROLES
# =========================

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
if "mj_fase" not in st.session_state:
    st.session_state.mj_fase = mj_carregar_fase_inicial()
if "momento" not in st.session_state:
    st.session_state.momento = momento_carregar()
if "max_avancos_por_cena" not in st.session_state:
    st.session_state.max_avancos_por_cena = 1
if "nsfw_max_level" not in st.session_state:
    # padrão elevado para 3 (explícito)
    st.session_state.nsfw_max_level = 3
if "estilo_escrita" not in st.session_state:
    st.session_state.estilo_escrita = "AÇÃO"

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

# =========================
# SIDEBAR — Provedor, modelo, resumo, memória, romance manual
# =========================

with st.sidebar:
    st.title("🧭 Painel do Roteirista")
    # Provedor/modelos
    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)
    if not api_key:
        st.warning("⚠️ API key ausente para o provedor selecionado. Defina em st.secrets.")
    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    st.session_state.modelo_escolhido_id = modelo_escolhido_id_ui

    st.markdown("---")
    st.markdown("### ✍️ Estilo & NSFW")
    st.selectbox(
        "Estilo de escrita",
        ["AÇÃO", "ROMANCE LENTO", "NOIR"],
        index=["AÇÃO","ROMANCE LENTO","NOIR"].index(st.session_state.get("estilo_escrita","AÇÃO")),
        key="estilo_escrita",
    )
    # Elevar slider para aceitar 3
    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=int(st.session_state.get("nsfw_max_level",3)), key="nsfw_max_level")

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    # Resumo rápido
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
                json={"model": model_id_call, "messages": [{"role": "user", "content": prompt_resumo}], "max_tokens": 800, "temperature": 0.85},
                timeout=int(st.session_state.get("timeout_s", 300)),
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

    # Memória longa
    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")
    st.checkbox("Usar memória longa no prompt", value=st.session_state.get("use_memoria_longa", True), key="use_memoria_longa")
    st.slider("Top-K memórias", 1, 5, int(st.session_state.get("k_memoria_longa", 3)), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95, float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")
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

    # Histórico no prompt
    st.markdown("---")
    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")

    # ROMANCE MANUAL
    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio (manual)")

    fase_default = mj_carregar_fase_inicial()
    options_fase = sorted(FASES_ROMANCE.keys())
    max_phase = max(options_fase)
    fase_ui_val = int(st.session_state.get("mj_fase", fase_default))
    fase_ui_val = max(min(fase_ui_val, max_phase), min(options_fase))
    fase_escolhida = st.select_slider("Fase do romance", options=options_fase, value=fase_ui_val, format_func=_fase_label, key="ui_mj_fase")
    if fase_escolhida != st.session_state.get("mj_fase", fase_default):
        mj_set_fase(fase_escolhida, persist=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Avançar 1 passo"):
            mj_set_fase(min(st.session_state.get("mj_fase", 0) + 1, max_phase), persist=True)
    with col_b:
        if st.button("↺ Reiniciar (0)"):
            mj_set_fase(0, persist=True)

    st.slider("Micropassos por cena", 1, 3, value=int(st.session_state.get("max_avancos_por_cena", 1)), key="max_avancos_por_cena")

    st.markdown("### 🎚️ Momento (manual)")
    options_momento = sorted(MOMENTOS.keys())
    mom_default = momento_carregar()
    mom_ui_val = int(st.session_state.get("momento", mom_default))
    mom_ui_val = max(min(mom_ui_val, max(options_momento)), min(options_momento))
    mom_ui = st.select_slider("Momento atual", options=options_momento, value=mom_ui_val, format_func=_momento_label, key="ui_momento")
    if mom_ui != st.session_state.get("momento", mom_default):
        momento_set(mom_ui, persist=True)

    st.caption("Regra: 1 microavanço por cena. A fase só muda quando você decidir.")
    st.caption("Role a tela principal para ver interações anteriores.")

# =========================
# EXIBIR HISTÓRICO (depois resumo)
# =========================

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
    if st.session_state.get("resumo_capitulo"):
        with st.expander("🧠 Resumo do capítulo (mais recente)"):
            st.markdown(st.session_state.resumo_capitulo)

# =========================
# ENVIO DO USUÁRIO + STREAMING (OpenRouter/Together) + FALLBACKS
# =========================

entrada = st.chat_input("Digite sua direção de cena...")

if entrada:
    # Atualiza Momento sugerido (opcional e seguro)
    try:
        mom_atual = int(st.session_state.get("momento", momento_carregar()))
        mom_sug = detectar_momento_sugerido(entrada, fallback=mom_atual)
        mom_novo = clamp_momento(mom_atual, mom_sug, int(st.session_state.get("max_avancos_por_cena", 1)))
        if st.session_state.get("app_bloqueio_intimo", False):
            # não retrocede automaticamente; apenas sobe até no máx. 1 passo
            mom_novo = clamp_momento(mom_atual, mom_sug, 1)
        momento_set(mom_novo, persist=True)
    except Exception:
        pass

    # salva a entrada e mantém histórico de sessão
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    # constrói prompt principal
    prompt = construir_prompt_com_narrador()

    # histórico curto (somente sessão atual; o prompt já inclui últimas do sheet)
    historico = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.session_msgs
    ]

    # provedor + modelo
    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets.get("TOGETHER_API_KEY", "")
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets.get("OPENROUTER_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id

    if not auth:
        st.error("A chave de API do provedor selecionado não foi definida em st.secrets.")
        st.stop()

    # mensagens
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
        "max_tokens": int(st.session_state.get("max_tokens_rsp", 1200)),
        "temperature": 0.9,
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    # Layout centralizado (3 colunas: 1-3-1)
    left_sp, main_col, right_sp = st.columns([1, 3, 1])
    with main_col:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            resposta_txt = ""  # texto bruto vindo do stream
            last_update = time.time()

            # Reforço antecipado: memórias que ENTRARAM no prompt (topk + recorrentes)
            try:
                usados_prompt = []
                usados_prompt.extend(st.session_state.get("_ml_topk_texts", []))
                usados_prompt.extend(st.session_state.get("_ml_recorrentes", []))
                usados_prompt = [t for t in usados_prompt if t]
                if usados_prompt:
                    memoria_longa_reforcar(usados_prompt)
            except Exception:
                pass

            # 1) STREAM
            try:
                with requests.post(
                    endpoint, headers=headers, json=payload, stream=True,
                    timeout=int(st.session_state.get("timeout_s", 300))
                ) as r:
                    if r.status_code == 200:
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
                                # CORRETO: choices[0]['delta']['content']
                                delta = j["choices"][0]["delta"].get("content", "")
                                if not delta:
                                    continue
                                resposta_txt += delta
                                if time.time() - last_update > 0.10:
                                    out = render_tail(resposta_txt)
                                    # Mantém corte elíptico p/ não explodir o nível NSFW
                                    out = sanitize_explicit(
                                        out,
                                        max_level=int(st.session_state.get("nsfw_max_level", 2)),
                                        action="corte-eliptico"
                                    )
                                    placeholder.markdown(out + "▌")
                                    last_update = time.time()
                            except Exception:
                                continue
                    else:
                        st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
            except Exception as e:
                st.error(f"Erro no streaming: {e}")

            # 2) FALLBACKS se veio vazio
            visible_txt = sanitize_explicit(
                render_tail(resposta_txt).strip(),
                max_level=int(st.session_state.get("nsfw_max_level", 2)),
                action="corte-eliptico",
            )

            if not visible_txt:
                # 2a) retry sem stream
                try:
                    r2 = requests.post(
                        endpoint, headers=headers,
                        json={**payload, "stream": False},
                        timeout=int(st.session_state.get("timeout_s", 300))
                    )
                    if r2.status_code == 200:
                        try:
                            # CORRETO: choices[0]['message']['content']
                            resposta_txt = r2.json()["choices"][0]["message"]["content"].strip()
                        except Exception:
                            resposta_txt = ""
                        visible_txt = sanitize_explicit(
                            render_tail(resposta_txt).strip(),
                            max_level=int(st.session_state.get("nsfw_max_level", 2)),
                            action="corte-eliptico",
                        )
                    else:
                        st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
                except Exception as e:
                    st.error(f"Fallback (sem stream) erro: {e}")

            if not visible_txt:
                # 2b) retry sem o system extra (alguns modelos travam com system duplo)
                try:
                    r3 = requests.post(
                        endpoint, headers=headers,
                        json={
                            "model": model_to_call,
                            "messages": [{"role": "system", "content": prompt}] + historico,
                            "max_tokens": int(st.session_state.get("max_tokens_rsp", 1200)),
                            "temperature": 0.9,
                            "stream": False,
                        },
                        timeout=int(st.session_state.get("timeout_s", 300))
                    )
                    if r3.status_code == 200:
                        try:
                            # CORRETO: choices[0]['message']['content']
                            resposta_txt = r3.json()["choices"][0]["message"]["content"].strip()
                        except Exception:
                            resposta_txt = ""
                        visible_txt = sanitize_explicit(
                            render_tail(resposta_txt).strip(),
                            max_level=int(st.session_state.get("nsfw_max_level", 2)),
                            action="corte-eliptico",
                        )
                    else:
                        st.error(f"Fallback (prompts limpos) falhou: {r3.status_code} - {r3.text}")
                except Exception as e:
                    st.error(f"Fallback (prompts limpos) erro: {e}")

            # 3) Exibição final (texto filtrado de acordo com o nível)
            placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

            # 4) Validação de momento (aviso leve, não bloqueia)
            try:
                viol = viola_momento(visible_txt, int(st.session_state.get("momento", 0)))
                if viol and st.session_state.get("app_bloqueio_intimo", False):
                    st.info(f"⚠️ {viol}")
            except Exception:
                pass

            # 5) Validação semântica (entrada do user vs resposta) usando texto visível
            if len(st.session_state.session_msgs) >= 1 and visible_txt and visible_txt != "[Sem conteúdo]":
                texto_anterior = st.session_state.session_msgs[-1]["content"]
                alerta = verificar_quebra_semantica_openai(texto_anterior, visible_txt)
                if alerta:
                    st.info(alerta)

            # 6) Salvar resposta SEMPRE (usa o texto visível)
            salvar_interacao("assistant", visible_txt or "[Sem conteúdo]")
            st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt or "[Sem conteúdo]"})

            # 7) Reforço de memórias usadas (pós-resposta) com base no texto visível
            try:
                usados = []
                topk_usadas = memoria_longa_buscar_topk(
                    query_text=visible_txt,
                    k=int(st.session_state.k_memoria_longa),
                    limiar=float(st.session_state.limiar_memoria_longa),
                )
                for t, _sc, _sim, _rr in topk_usadas:
                    usados.append(t)
                memoria_longa_reforcar(usados)
            except Exception:
                pass

