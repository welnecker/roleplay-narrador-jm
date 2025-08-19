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
    return callable_fn(*args, **kwargs)

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

# === FALAS SOBRE VIRGINDADE DA MARY (brandas; você pode trocar depois) ===
FALAS_VIRGINDADE_MARY = [
    "Eu preciso te dizer… eu ainda sou virgem.",
    "Quero ir devagar, no meu tempo — fica comigo.",
    "Hoje eu quero carinho e beijo, sem passar do meu limite.",
    "Se for com você, quero que seja especial; me escuta, tá?",
]


# =========================
# CONFIG BÁSICA DO APP
# =========================

PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
TAB_INTERACOES = "interacoes_jm"
TAB_PERFIL     = "perfil_jm"
TAB_MEMORIAS   = "memoria_jm"
TAB_ML         = "memoria_longa_jm"
TAB_TEMPLATES  = "templates_jm"
TAB_FALAS_MARY = "falas_mary_jm"   # <<< NOVA ABA: fala | timestamp

def conectar_planilha():
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
            if name == TAB_INTERACOES:
                _retry_429(ws.append_row, ["timestamp", "role", "content"])
            elif name == TAB_PERFIL:
                _retry_429(ws.append_row, ["timestamp", "resumo"])
            elif name == TAB_MEMORIAS:
                _retry_429(ws.append_row, ["tipo", "conteudo", "timestamp"])
            elif name == TAB_ML:
                _retry_429(ws.append_row, ["texto", "embedding", "tags", "timestamp", "score"])
            elif name == TAB_TEMPLATES:
                _retry_429(ws.append_row, ["template", "etapa", "texto"])
            elif name == TAB_FALAS_MARY:  # <<< cria cabeçalho
                _retry_429(ws.append_row, ["fala", "timestamp"])
            return ws
        except Exception:
            return None

# -------- templates (NÍVEL DE MÓDULO; NÃO DENTRO DE _ws !) --------
def carregar_templates_planilha():
    """Carrega todos os templates sequenciais da aba templates_jm em {template:[etapa1, etapa2,...]}"""
    try:
        ws = _ws(TAB_TEMPLATES, create_if_missing=False)
        if not ws:
            return {}
        rows = _sheet_all_records_cached(TAB_TEMPLATES)
        templates = {}
        for row in rows:
            r = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            nome = r.get("template", "").strip()
            etapa_str = r.get("etapa", "1")
            try:
                etapa = int(etapa_str)
            except Exception:
                etapa = 1
            texto = r.get("texto", "").strip()
            if nome and texto:
                templates.setdefault(nome, []).append((etapa, texto))
        for nome in templates:
            templates[nome].sort(key=lambda x: x[0])
            templates[nome] = [t[1] for t in templates[nome]]
        return templates
    except Exception as e:
        st.warning(f"Erro ao carregar templates do Sheets: {e}")
        return {}

def _init_templates_once():
    if "templates_jm" not in st.session_state:
        st.session_state.templates_jm = carregar_templates_planilha()
    if "template_ativo" not in st.session_state:
        st.session_state.template_ativo = None
    if "etapa_template" not in st.session_state:
        st.session_state.etapa_template = 0

_init_templates_once()

# =====
# UTILIDADES: MEMÓRIAS / HISTÓRICO
# =====

def _normalize_tag(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t:
        return ""
    return t if t.startswith("[") else f"[{t}]"

def _parse_ts(s: str) -> str:
    s = (s or "").strip()
    try:
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return s
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def carregar_memorias_brutas() -> Dict[str, List[dict]]:
    try:
        regs = _sheet_all_records_cached(TAB_MEMORIAS)
        buckets: Dict[str, List[dict]] = {}
        for r in regs:
            tag = _normalize_tag(r.get("tipo"))
            txt = (r.get("conteudo") or "").strip()
            ts  = (r.get("timestamp") or "").strip()
            if tag and txt:
                buckets.setdefault(tag, []).append({"conteudo": txt, "timestamp": ts})
        return buckets
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return {}

def persona_block(nome: str, buckets: dict, max_linhas: int = 8) -> str:
    tag = f"[{nome}]"
    linhas = buckets.get(tag, [])
    ordem = ["OBJ:", "TAT:", "LV:", "VOZ:", "BIO:", "ROTINA:", "LACOS:", "APS:", "CONFLITOS:"]
    def peso(linha_dict):
        up = (linha_dict.get("conteudo","")).upper()
        for i, p in enumerate(ordem):
            if up.startswith(p):
                return i
        return len(ordem)
    linhas_ordenadas = sorted(linhas, key=peso)[:max_linhas]
    titulo = "Jânio" if nome in ("janio", "jânio") else "Mary" if nome == "mary" else nome.capitalize()
    return f"{titulo}:\n- " + "\n- ".join(l['conteudo'] for l in linhas_ordenadas) if linhas_ordenadas else ""

def persona_block_temporal(nome: str, buckets: dict, ate_ts: str, max_linhas: int = 8) -> str:
    tag = f"[{nome}]"
    linhas = []
    for d in buckets.get(tag, []) or []:
        if not isinstance(d, dict):
            continue
        c = (d.get("conteudo") or "").strip()
        ts = (d.get("timestamp") or "").strip()
        if not c:
            continue
        if ts and ate_ts and ts > ate_ts:
            continue
        linhas.append((ts, c))
    linhas.sort(key=lambda x: x[0])
    ult = [c for _, c in linhas][-max_linhas:]
    if not ult:
        return ""
    titulo = "Jânio" if nome in ("janio", "jânio") else "Mary" if nome == "mary" else nome.capitalize()
    return f"{titulo}:\n- " + "\n- ".join(ult)

def detectar_virgindade_mary(memos: dict, ate_ts: str = "") -> bool:
    """
    True se há, em [mary], alguma memória com 'virgem' ou 'virgindade'
    (respeitando corte temporal, se ate_ts vier).
    """
    for d in memos.get("[mary]", []) or []:
        c = (d.get("conteudo") or "").lower()
        ts = (d.get("timestamp") or "")
        if ate_ts and ts and ts > ate_ts:
            continue
        if ("virgem" in c) or ("virgindade" in c):
            return True
    return False


def montar_bloco_virgindade(ativar: bool) -> str:
    """
    Se ativar=True, devolve bloco de regras + 2–3 falas brandas (você pode editar depois).
    """
    if not ativar:
        return ""
    exemplos = "\n".join(f"- {s}" for s in FALAS_VIRGINDADE_MARY[:3])
    return (
        "### Virgindade & Limites (prioritário)\n"
        "- Mary é virgem e valoriza isso. Se a cena se aproximar de sexo explícito, "
        "**ela verbaliza seus limites antes** de qualquer avanço.\n"
        "- Mostre desejo com respeito ao ritmo: consentimento claro, trocas gentis, pausas.\n"
        "- Inclua **uma fala curta** de Mary sobre isso quando apropriado (exemplos abaixo):\n"
        f"{exemplos}\n"
    )


def carregar_resumo_salvo() -> str:
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
    cache = st.session_state.get("_cache_interacoes", None)
    if cache is None:
        regs = _sheet_all_records_cached(TAB_INTERACOES)
        st.session_state["_cache_interacoes"] = regs
        cache = regs
    return cache[-n:] if len(cache) > n else cache

def salvar_interacao(role: str, content: str):
    if not planilha:
        return
    try:
        aba = _ws(TAB_INTERACOES)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_role = f"{role or ''}".strip()
        row_content = f"{content or ''}".strip()
        _retry_429(aba.append_row, [timestamp, row_role, row_content], value_input_option="RAW")
        lst = st.session_state.get("_cache_interacoes")
        if isinstance(lst, list):
            lst.append({"timestamp": timestamp, "role": row_role, "content": row_content})
        else:
            st.session_state["_cache_interacoes"] = [{
                "timestamp": timestamp, "role": row_role, "content": row_content
            }]
        _invalidate_sheet_caches()
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

# ----- Falas da Mary (planilha) -----
def carregar_falas_mary() -> List[str]:
    try:
        rows = _sheet_all_records_cached(TAB_FALAS_MARY)
        falas = []
        for r in rows:
            f = (r.get("fala") or "").strip()
            if f:
                falas.append(f)
        return falas
    except Exception as e:
        st.warning(f"Erro ao carregar {TAB_FALAS_MARY}: {e}")
        return []

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
# MEMÓRIA LONGA (opcional)
# =========================

def _sheet_ensure_memoria_longa():
    return _ws(TAB_ML, create_if_missing=False)

def _serialize_vec(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))

def _deserialize_vec(s: str) -> np.ndarray:
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        return np.zeros(1, dtype=float)

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
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
    try:
        return _sheet_all_records_cached(TAB_ML)
    except Exception:
        return []

def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-zà-ú0-9]+", (s or "").lower()))

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78, ate_ts=None):
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
        row_ts = (row.get("timestamp") or "").strip()
        try:
            score = float(row.get("score", 1.0) or 1.0)
        except Exception:
            score = 1.0
        if not texto:
            continue
        if ate_ts and row_ts and row_ts > ate_ts:
            continue
        if q is not None and emb_s:
            vec = _deserialize_vec(emb_s)
            if vec.ndim == 1 and vec.size >= 10 and np.linalg.norm(vec) > 0 and np.linalg.norm(q) > 0:
                sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
            else:
                sim = 0.0
        else:
            s1 = _tokenize(texto); s2 = _tokenize(query_text)
            sim = len(s1 & s2) / max(1, len(s1 | s2))
        if sim >= limiar:
            rr = 0.7 * sim + 0.3 * score
            candidatos.append((texto, score, sim, rr))
    candidatos.sort(key=lambda x: x[3], reverse=True)
    return candidatos[:k]

def memoria_longa_reforcar(textos_usados: list):
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = _sheet_all_values_cached(TAB_ML)
        if not dados or len(dados) < 2:
            return
        headers = dados[0]
        idx_texto = headers.index("texto")
        idx_score = headers.index("score")
        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) <= max(idx_texto, idx_score): continue
            t = (linha[idx_texto] or "").strip()
            if t in textos_usados:
                try: sc = float(linha[idx_score] or 1.0)
                except Exception: sc = 1.0
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
        "proibidos": "pressa ou “provas de amor” físicas"},
    5: {"nome": "Compromisso / Encontro definitivo",
        "permitidos": "beijo prolongado; dormir juntos; consumação implícita (fade-to-black); manhã seguinte sugerida",
        "proibidos": ""},
}

FLAG_FASE_TXT_PREFIX = "FLAG: mj_fase="

def _fase_label(n: int) -> str:
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE[0])
    return f"{int(n)} — {d['nome']}"

def mj_set_fase(n: int, persist: bool=False):
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

MOMENTOS = {
    0: {"nome": "Aproximação logística",
        "objetivo": "um acompanha o outro",
        "permitidos": "gentilezas; proximidade leve; diálogo casual",
        "proibidos": "declaração; revelações íntimas; toques prolongados",
        "gatilhos": [r"\b(p[ií]er|acompanhar|vamos embora|te levo)\b"],
        "proximo": 1},
    1: {"nome": "Declaração",
        "objetivo": "um deles declara importância",
        "permitidos": "confissão afetiva; silêncio tenso; abraço curto",
        "proibidos": "negociação sexual; tirar roupas",
        "gatilhos": [r"\b(amo voc[eê]|te amo|n[aã]o paro de pensar)\b"],
        "proximo": 2},
    2: {"nome": "Revelação sensível",
        "objetivo": "Mary revela vulnerabilidade / limites",
        "permitidos": "nomear limites; conforto mútuo",
        "proibidos": "carícias íntimas; tirar roupas",
        "gatilhos": [r"\b(meu limite|prefiro ir devagar)\b"],
        "proximo": 3},
    3: {"nome": "Consentimento explícito",
        "objetivo": "alinhamento de limites e um 'sim' claro",
        "permitidos": "pedir/receber consentimento; decidir 'agora sim'",
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eê])\b", r"\b(at[eé] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade (elíptica)",
        "objetivo": "intimidade sugerida (fade-to-black) / pós-ato implícito",
        "permitidos": "beijos longos; proximidade forte; fade-to-black",
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
    return ""

# =========================
# PROVEDORES E MODELOS (enxuto)
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
    "🧠 Qwen3 Coder 480B (Together)": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
}

def model_id_for_together(api_ui_model_id: str) -> str:
    return api_ui_model_id or "mistralai/Mixtral-8x7B-Instruct-v0.1"

def api_config_for_provider(provider: str):
    if provider == "OpenRouter":
        return ("https://openrouter.ai/api/v1/chat/completions",
                st.secrets.get("OPENROUTER_API_KEY", ""),
                MODELOS_OPENROUTER)
    else:
        return ("https://api.together.xyz/v1/chat/completions",
                st.secrets.get("TOGETHER_API_KEY", ""),
                MODELOS_TOGETHER_UI)

# =========================
# PROMPT BUILDER
# =========================

def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    calor = int(st.session_state.get("nsfw_max_level", 3))
    regras = f"""
⚖️ Regras de coerência:
- Narre em terceira pessoa; não se dirija ao leitor como "você".
- Consentimento claro antes de qualquer gesto significativo.
- Jânio respeita o ritmo de Mary.
- Linguagem sensual proporcional ao nível de calor ({calor}).
""".strip()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    if fase >= 5:
        regras += "\n- A intimidade pode ser sugerida sem elipses forçadas."
    else:
        regras += "\n- Evite consumação explícita; foque em progressão coerente."
    return prompt_base + "\n" + regras

def gerar_mary_sensorial(level: int = 2, n: int = 2, hair_on: bool = True, sintonia: bool = False) -> str:
    if level <= 0 or n <= 0:
        return ""
    base_leve = [
        "Mary caminha com ritmo seguro; há algo hipnótico no balanço dos quadris.",
        "O olhar de Mary prende fácil: direto, firme, cativante.",
        "O perfume de Mary chega antes dela, discreto e morno.",
        "O sorriso aparece quando quer; breve, afiado, certeiro.",
    ]
    base_marcado = [
        "Os quadris de Mary balançam num compasso que chama atenção sem pedir licença.",
        "Enquanto caminha, os seios balançam de leve sob o tecido.",
        "O tecido roça nas pernas e denuncia o passo: firme, íntimo, decidido.",
        "O olhar de Mary é um convite silencioso — confiante e difícil de sustentar por muito tempo.",
    ]
    base_ousado = [
        "O balanço dos quadris de Mary fica na cabeça de quem vê.",
        "Os seios acompanham a passada num movimento suave.",
        "O olhar de Mary encosta na pele de quem cruza com ela: quente e demorado.",
        "O perfume fica na memória como um toque atrás da nuca.",
    ]
    hair_leve = [
        "Os cabelos de Mary — negros, volumosos, levemente ondulados — descansam nos ombros e acompanham o passo.",
        "Os cabelos negros, volumosos e levemente ondulados moldam o rosto quando ela vira de leve.",
    ]
    hair_marcado = [
        "Cabelos negros, volumosos, levemente ondulados, fazem um arco quando ela vira o rosto.",
        "Os cabelos, negros e volumosos, ondulam de leve a cada passada e enquadram o olhar.",
    ]
    hair_ousado = [
        "Os cabelos negros, volumosos e levemente ondulados deslizam pela clavícula.",
        "O balanço dos cabelos negros — volumosos, levemente ondulados — marca o compasso do corpo.",
    ]
    if level == 1:
        pool = list(base_leve); hair_pool = list(hair_leve)
    elif level == 2:
        pool = list(base_leve) + list(base_marcado); hair_pool = list(hair_leve) + list(hair_marcado)
    else:
        pool = base_leve + base_marcado + base_ousado; hair_pool = hair_leve + hair_marcado + hair_ousado

    if sintonia:
        filtros = [r"\binsinuante\b", r"\bacende o ambiente\b"]
        def _ok(fr): return not any(re.search(p, fr, re.IGNORECASE) for p in filtros)
        pool = [f for f in pool if _ok(f)]
        pool += [
            "A respiração de Mary encontra o compasso do parceiro, sem pressa.",
            "Ela desacelera um passo, deixando o momento guiar o ritmo.",
            "Há pausas gentis entre olhares; tudo acontece no tempo certo.",
            "O gesto nasce do encontro, não da urgência; Mary prefere sentir antes de conduzir.",
        ]
    n_eff = max(1, min(n, len(pool)))
    frases = random.sample(pool, k=n_eff)
    if hair_on and hair_pool:
        hair_line = random.choice(hair_pool)
        if hair_line not in frases:
            frases.insert(0, hair_line)
            if len(frases) > n_eff:
                frases = frases[:n_eff]
    return " ".join(frases)

def encontrar_memorias_relevantes(pergunta, buckets):
    keywords = ["nome","integrante","banda","integrantes","profissão","rotina","cargo","ocupação",
                "onde","quem","quando","idade","universidade","curso","história","grupo"]
    relevantes = []
    pergunta_lc = (pergunta or "").lower()
    for tag, items in buckets.items():
        tag_limp = tag.strip("[]")
        if any(k in pergunta_lc for k in keywords) or tag_limp in pergunta_lc:
            relevantes.extend(items)
    return relevantes

def _last_user_text(hist):
    if not hist: return ""
    for r in reversed(hist):
        if str(r.get("role","")).lower() == "user":
            return r.get("content","")
    return ""

def _deduzir_ancora(texto: str) -> dict:
    t = (texto or "").lower()
    if "motel" in t or "suíte" in t or "suite" in t:
        return {"local": "Motel — Suíte Master", "hora": "noite"}
    if "quarto" in t:
        return {"local": "Quarto", "hora": "noite"}
    if "praia" in t:
        return {"local": "Praia", "hora": "fim de tarde"}
    if "bar" in t or "pub" in t:
        return {"local": "Bar", "hora": "noite"}
    if "biblioteca" in t:
        return {"local": "Biblioteca", "hora": "tarde"}
    if "ufes" in t:
        return {"local": "UFES — campus", "hora": "manhã"}
    gatilhos_motel = ["cama redonda", "espelho no teto", "piscina aquecida"]
    if any(g in t for g in gatilhos_motel):
        return {"local": "Motel — Suíte Master", "hora": "noite"}
    return {}

def construir_prompt_com_narrador() -> str:
    memos = carregar_memorias_brutas()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS[0])
    proximo_nome = MOMENTOS.get(mdata.get("proximo", 0), MOMENTOS[0])["nome"]
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")

    # Camada sensorial (Mary)
    _sens_on = bool(st.session_state.get("mary_sensorial_on", True))
    _sens_level = int(st.session_state.get("mary_sensorial_level", 2))
    _sens_n = int(st.session_state.get("mary_sensorial_n", 2))
    mary_sens_txt = gerar_mary_sensorial(
        _sens_level, n=_sens_n, sintonia=bool(st.session_state.get("modo_sintonia", True))
    ) if _sens_on else ""

    # Sintonia & Ritmo
    modo_sintonia = bool(st.session_state.get("modo_sintonia", True))
    ritmo_cena = int(st.session_state.get("ritmo_cena", 1))
    ritmo_label = ["muito lento", "lento", "médio", "rápido"][max(0, min(3, ritmo_cena))]

    # Histórico
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r.get('role','user')}: {r.get('content','')}" for r in hist) if hist else "(sem histórico)"
    ultima_fala_user = _last_user_text(hist)
    ancora = _deduzir_ancora(ultima_fala_user)
    ancora_bloco = ""
    if ancora:
        ancora_bloco = (
            "### Âncora de cenário (OBRIGATÓRIA)\n"
            f"- Local: **{ancora['local']}**\n"
            f"- Hora: **{ancora['hora']}**\n"
            "- Regra: mantenha a cena **neste local**; **não** troque para UFES, bar, biblioteca etc.\n"
            "- Primeira frase deve ancorar **lugar e hora** neste formato: `Local — Hora — ...`.\n"
        )
    # Corte temporal
    if hist:
        ate_ts = _parse_ts(hist[-1].get("timestamp", ""))
    else:
        ate_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
    # >>> Virgindade (auto, a partir de memoria_jm e corte temporal)
    virg_bloco = montar_bloco_virgindade(
        ativar=detectar_virgindade_mary(memos, ate_ts)
    )

    
    # Memória longa Top-K (opcional)
    ml_topk_txt = "(nenhuma)"
    st.session_state["_ml_topk_texts"] = []
    if st.session_state.get("use_memoria_longa", True) and hist:
        try:
            topk = memoria_longa_buscar_topk(
                query_text=hist[-1]["content"],
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
                ate_ts=ate_ts,
            )
            if topk:
                ml_topk_txt = "\n".join([f"- {t}" for (t, _sc, _sim, _rr) in topk])
                st.session_state["_ml_topk_texts"] = [t for (t, *_rest) in topk]
        except Exception:
            st.session_state["_ml_topk_texts"] = []
    else:
        st.session_state["_ml_topk_texts"] = []

    # Diretrizes [all] respeitando o tempo
    recorrentes = [
        (d.get("conteudo") or "").strip()
        for d in memos.get("[all]", [])
        if isinstance(d, dict) and d.get("conteudo") and (not d.get("timestamp") or d.get("timestamp") <= ate_ts)
    ]
    st.session_state["_ml_recorrentes"] = recorrentes

    # Dossiê temporal
    dossie = []
    mary = persona_block_temporal("mary", memos, ate_ts, 8)
    janio = persona_block_temporal("janio", memos, ate_ts, 8)
    if mary: dossie.append(mary)
    if janio: dossie.append(janio)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    flag_parallel = bool(st.session_state.get("no_coincidencias", True))

    # Falas da Mary (planilha)
    falas_mary_bloco = ""
    if st.session_state.get("usar_falas_mary", False):
        falas_mary = carregar_falas_mary()
        if falas_mary:
            falas_mary_bloco = "### Falas de Mary (usar literalmente)\n" + "\n".join(f"- {s}" for s in falas_mary)

    # Sintonia & Ritmo
    sintonia_bloco = ""
    if modo_sintonia:
        sintonia_bloco = (
            "### Sintonia & Ritmo (prioritário)\n"
            f"- Ritmo da cena: **{ritmo_label}**.\n"
            "- Condução harmônica: Mary sintoniza com o parceiro; evite ordens ríspidas. Prefira convites, sussurros, pedidos gentis.\n"
            "- Respeite pausas, respiração, olhar; o desejo é mostrado pela troca, não por imposição.\n"
        )

   prompt = f"""
Você é o Narrador de um roleplay dramático brasileiro, foque em Mary e Jânio. Não repita instruções nem títulos.

{ancora_bloco}{sintonia_bloco}{virg_bloco}### Dossiê (personas)
{dossie_txt}

### Diretrizes gerais (ALL)
{chr(10).join(f"- {c}" for c in recorrentes) if recorrentes else "(vazio)"}

### Perfil (resumo mais recente)
{perfil or "(vazio)"}

### Histórico recente (planilha)
{hist_txt}

### Estilo
- Use o estilo **{estilo}**:
{("- Frases curtas, cortes rápidos, foco em gesto/ritmo.") if estilo=="AÇÃO" else
("- Atmosfera sombria, subtexto, silêncio que pesa.") if estilo=="NOIR" else
("- Ritmo lento, tensão emocional, detalhes sensoriais (sem grafismo).")}
- As falas de Mary devem soar naturais, diretas e sensoriais; evite grosserias. Ajuste o tom ao **modo de sintonia** e ao **ritmo**.

### Camada sensorial — Mary (OBRIGATÓRIA no 1º parágrafo)
{mary_sens_txt or "- Comece com 1–2 frases sobre caminhar/olhar/perfume/cabelos (negros, volumosos, levemente ondulados)."}
- Aplique essa camada ANTES do primeiro diálogo.
- Frases curtas, físicas; evite metáforas rebuscadas.

{falas_mary_bloco}

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
- Se o roteirista pedir salto maior, **negocie** consentimento e **prepare** a transição.

### Geografia & Montagem
- **Não force coincidências**: sem ponte explícita, mantenha locais distintos e use **montagem paralela** (A/B) = {flag_parallel}.
- **Comece cada bloco** com uma frase que **ancore lugar e hora** (ex.: “Local — Hora — …”). Escreva isso na primeira frase do parágrafo.
- Havendo ponte diegética plausível, convergir ao final é permitido (sem teletransporte).

### Formato OBRIGATÓRIO da cena
- **Inclua DIÁLOGOS diretos** com travessão (—), intercalados com ação/reação física/visual (mínimo 4 falas).
- Garanta **pelo menos 2 falas de Mary e 2 de Jânio** (quando ambos estiverem na cena).
- **Não inclua** pensamentos em itálico, reflexões internas ou monólogos subjetivos.
- Sem blocos finais de créditos, microconquistas, resumos ou ganchos.

### Regra de saída
- Narre em **terceira pessoa**; nunca fale com "você".
- Produza uma cena fechada e natural.
""".strip()

    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# FILTROS DE SAÍDA
# =========================

def render_tail(t: str) -> str:
    if not t:
        return ""
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
        return 3
    if re.search(r"\b(cintura|pesco[cç]o|costas|beijo prolongado|respira[cç][aã]o curta)\b", (t or ""), re.IGNORECASE):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aã]os dadas|beijo)\b", (t or ""), re.IGNORECASE):
        return 1
    return 0

def sanitize_explicit(t: str, max_level: int, action: str) -> str:
    lvl = classify_nsfw_level(t)
    if lvl <= max_level:
        return t
    return t

def redact_for_logs(t: str) -> str:
    if not t:
        return ""
    t = re.sub(EXPL_PAT, "[…]", t, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

def precisa_reforcar_dialogo(texto: str) -> bool:
    if not texto:
        return True
    n_dialog = len(re.findall(r'(^|\n)\s*(—|")', texto))
    n_thoughts = len(re.findall(r'\*[^*\n]{4,}\*', texto))
    return (n_dialog < 4) or (n_thoughts < 2)

# =========================
# UI — CABEÇALHO E CONTROLES
# =========================

st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Estados básicos
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
if "estilo_escrita" not in st.session_state:
    st.session_state.estilo_escrita = "AÇÃO"
if "templates_jm" not in st.session_state:
    st.session_state.templates_jm = carregar_templates_planilha()
if "template_ativo" not in st.session_state:
    st.session_state.template_ativo = None
if "etapa_template" not in st.session_state:
    st.session_state.etapa_template = 0

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### 📖 Último resumo salvo:")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")
with col2:
    st.markdown("#### ⚙️ Opções")
    st.write(
        f'- Bloqueio íntimo: {"Sim" if st.session_state.get("app_bloqueio_intimo", False) else "Não"}\n'
        f'- Emoção oculta: {st.session_state.get("app_emocao_oculta", "").capitalize()}'
    )

# =========================
# SIDEBAR — Reorganizado
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
    st.markdown("### ✍️ Estilo & Progresso Dramático")
    st.selectbox(
        "Estilo de escrita",
        ["AÇÃO", "ROMANCE LENTO", "NOIR"],
        index=["AÇÃO", "ROMANCE LENTO", "NOIR"].index(st.session_state.get("estilo_escrita", "AÇÃO")),
        key="estilo_escrita",
    )
    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=int(st.session_state.get("nsfw_max_level", 3)), key="nsfw_max_level")

    # Sintonia & Ritmo (DENTRO do sidebar)
    st.checkbox(
        "Sintonia com o parceiro (modo harmônico)",
        key="modo_sintonia",
        value=st.session_state.get("modo_sintonia", True),
    )
    st.select_slider(
        "Ritmo da cena",
        options=[0, 1, 2, 3],
        value=int(st.session_state.get("ritmo_cena", 1)),
        format_func=lambda n: ["muito lento", "lento", "médio", "rápido"][n],
        key="ritmo_cena",
    )

    # Falas de Mary — planilha
    st.checkbox(
        "Usar falas da Mary da planilha (usar literalmente)",
        value=st.session_state.get("usar_falas_mary", False),
        key="usar_falas_mary",
    )

    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio")
    fase_default = mj_carregar_fase_inicial()
    options_fase = sorted(FASES_ROMANCE.keys())
    fase_ui_val = int(st.session_state.get("mj_fase", fase_default))
    fase_ui_val = max(min(fase_ui_val, max(options_fase)), min(options_fase))
    fase_escolhida = st.select_slider("Fase do romance", options=options_fase, value=fase_ui_val, format_func=_fase_label, key="ui_mj_fase")
    if fase_escolhida != st.session_state.get("mj_fase", fase_default):
        mj_set_fase(fase_escolhida, persist=True)
    options_momento = sorted(MOMENTOS.keys())
    mom_default = momento_carregar()
    mom_ui_val = int(st.session_state.get("momento", mom_default))
    mom_ui_val = max(min(mom_ui_val, max(options_momento)), min(options_momento))
    mom_ui = st.select_slider("Momento atual", options=options_momento, value=mom_ui_val, format_func=_momento_label, key="ui_momento")
    if mom_ui != st.session_state.get("momento", mom_default):
        momento_set(mom_ui, persist=False)
    st.slider("Micropassos por cena", 1, 3, value=int(st.session_state.get("max_avancos_por_cena", 1)), key="max_avancos_por_cena")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Avançar 1 passo"):
            mj_set_fase(min(st.session_state.get("mj_fase", 0) + 1, max(options_fase)), persist=True)
    with col_b:
        if st.button("↺ Reiniciar (0)"):
            mj_set_fase(0, persist=True)

    st.markdown("---")
    st.markdown("### 🎬 Roteiros Sequenciais (Templates)")
    nomes_templates = list(st.session_state.templates_jm.keys())
    if st.button("🔄 Recarregar templates"):
        st.session_state.templates_jm = carregar_templates_planilha()
        st.success("Templates atualizados da planilha!")
    if nomes_templates:
        roteiro_escolhido = st.selectbox("Escolha o roteiro:", nomes_templates, key="sb_rota_sel")
        etapas = st.session_state.templates_jm.get(roteiro_escolhido, [])

        if st.button("Iniciar roteiro", key="btn_iniciar_roteiro"):
            st.session_state.template_ativo = roteiro_escolhido
            st.session_state.etapa_template = 0
            if etapas:
                comando = etapas[0]
                salvar_interacao("user", comando)
                st.session_state.session_msgs.append({"role": "user", "content": comando})
                st.session_state["_trigger_input"] = comando
                st.session_state.etapa_template = 1

        if st.session_state.get("template_ativo"):
            etapas_ativas = st.session_state.templates_jm.get(st.session_state.template_ativo, [])
            etap = int(st.session_state.get("etapa_template", 0))
            if etap < len(etapas_ativas):
                st.markdown(f"Etapa atual: {etap + 1} de {len(etapas_ativas)}")
                if st.button("Próxima etapa (*)", key="btn_proxima_etapa"):
                    comando = etapas_ativas[etap]
                    salvar_interacao("user", comando)
                    st.session_state.session_msgs.append({"role": "user", "content": comando})
                    st.session_state.etapa_template = etap + 1
                    st.session_state["_trigger_input"] = comando
            else:
                st.success("Roteiro concluído!")
                st.session_state.template_ativo = None
                st.session_state.etapa_template = 0
    else:
        st.info("Nenhum template encontrado na aba templates_jm.")

    st.markdown("---")
    st.checkbox(
        "Evitar coincidências forçadas (montagem paralela A/B)",
        value=st.session_state.get("no_coincidencias", True),
        key="no_coincidencias",
    )
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

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")
    st.checkbox("Usar memória longa no prompt", value=st.session_state.get("use_memoria_longa", True), key="use_memoria_longa")
    st.slider("Top-K memórias", 1, 5, int(st.session_state.get("k_memoria_longa", 3)), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95, float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")

    st.markdown("---")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r.get('role','user')}: {r.get('content','')}" for r in inter) if inter else ""
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

    if st.button("🔁 Reforçar memórias biográficas"):
        memos = carregar_memorias_brutas()
        count = 0
        for k in ["[mary]", "[janio]", "[all]"]:
            for entrada in memos.get(k, []):
                texto = entrada.get("conteudo", "").strip()
                if texto:
                    ok = memoria_longa_salvar(texto, tags=k)
                    if ok: count += 1
        st.success(f"{count} memórias biográficas reforçadas na memória longa!")

    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")

# =========================
# EXIBIR HISTÓRICO
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
# ENVIO DO USUÁRIO + STREAMING
# =========================

entrada = st.chat_input("Digite sua direção de cena...")

if entrada:
    try:
        mom_atual = int(st.session_state.get("momento", momento_carregar()))
        mom_sug   = detectar_momento_sugerido(entrada, fallback=mom_atual)
        mom_novo  = clamp_momento(mom_atual, mom_sug, int(st.session_state.get("max_avancos_por_cena", 1)))
        if st.session_state.get("app_bloqueio_intimo", False):
            mom_novo = clamp_momento(mom_atual, mom_sug, 1)
        momento_set(mom_novo, persist=False)
    except Exception:
        pass

    salvar_interacao("user", str(entrada))
    st.session_state.session_msgs.append({"role": "user", "content": str(entrada)})

    prompt = construir_prompt_com_narrador()

    historico = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in st.session_state.session_msgs]

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

    system_pt = {"role": "system", "content": "Responda em português do Brasil. Mostre apenas a narrativa final."}
    messages = [system_pt, {"role": "system", "content": prompt}] + historico

    payload = {
        "model": model_to_call,
        "messages": messages,
        "max_tokens": int(st.session_state.get("max_tokens_rsp", 1200)),
        "temperature": 0.9,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    def _render_visible(t: str) -> str:
        out = render_tail(t)
        out = sanitize_explicit(out, max_level=int(st.session_state.get("nsfw_max_level", 3)), action="livre")
        return out

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        try:
            usados_prompt = []
            usados_prompt.extend(st.session_state.get("_ml_topk_texts", []))
            usados_prompt.extend(st.session_state.get("_ml_recorrentes", []))
            usados_prompt = [t for t in usados_prompt if t]
            if usados_prompt:
                memoria_longa_reforcar(usados_prompt)
        except Exception:
            pass

        try:
            with requests.post(endpoint, headers=headers, json=payload, stream=True,
                               timeout=int(st.session_state.get("timeout_s", 300))) as r:
                if r.status_code == 200:
                    for raw in r.iter_lines(decode_unicode=False):
                        if not raw: continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        try:
                            j = json.loads(data)
                            delta = j["choices"][0]["delta"].get("content", "")
                            if not delta: continue
                            resposta_txt += delta
                            if time.time() - last_update > 0.10:
                                placeholder.markdown(_render_visible(resposta_txt) + "▌")
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro no streaming: {e}")

        visible_txt = _render_visible(resposta_txt).strip()
        if not visible_txt:
            try:
                r2 = requests.post(endpoint, headers=headers,
                                   json={**payload, "stream": False},
                                   timeout=int(st.session_state.get("timeout_s", 300)))
                if r2.status_code == 200:
                    try:
                        resposta_txt = r2.json()["choices"][0]["message"]["content"].strip()
                    except Exception:
                        resposta_txt = ""
                    visible_txt = _render_visible(resposta_txt).strip()
                else:
                    st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
            except Exception as e:
                st.error(f"Fallback (sem stream) erro: {e}")

        if not visible_txt:
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
                        resposta_txt = r3.json()["choices"][0]["message"]["content"].strip()
                    except Exception:
                        resposta_txt = ""
                    visible_txt = _render_visible(resposta_txt).strip()
                else:
                    st.error(f"Fallback (prompts limpos) falhou: {r3.status_code} - {r3.text}")
            except Exception as e:
                st.error(f"Fallback (prompts limpos) erro: {e}")

        placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

        try:
            viol = viola_momento(visible_txt, int(st.session_state.get("momento", 0)))
            if viol and st.session_state.get("app_bloqueio_intimo", False):
                st.info(f"⚠️ {viol}")
        except Exception:
            pass

        if len(st.session_state.session_msgs) >= 1 and visible_txt and visible_txt != "[Sem conteúdo]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]
            alerta = verificar_quebra_semantica_openai(texto_anterior, visible_txt)
            if alerta:
                st.info(alerta)

        salvar_interacao("assistant", visible_txt if visible_txt else "[Sem conteúdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt if visible_txt else "[Sem conteúdo]"})

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


