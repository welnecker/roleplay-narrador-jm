# ============================================================
# Narrador JM — Variante “Somente FASE do romance” (corrigido)
# ============================================================

import os
import re
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import requests
import gspread
import numpy as np
from gspread.exceptions import APIError, GSpreadException
from oauth2client.service_account import ServiceAccountCredentials
from huggingface_hub import InferenceClient


# =========================
# CONFIG BÁSICA DO APP
# =========================

# ATENÇÃO: este modo ignora “Momento atual”. Só a FASE manda.
ONLY_FASE_MODE = True

# --- LM Studio (via Cloudflare Tunnel) ---
LMS_BASE_URL = "https://against-project-secondary-giants.trycloudflare.com/v1"  # pode mudar no sidebar


PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
TAB_INTERACOES = "interacoes_jm"
TAB_PERFIL = "perfil_jm"
TAB_MEMORIAS = "memoria_jm"
TAB_ML = "memoria_longa_jm"
TAB_TEMPLATES = "templates_jm"
TAB_FALAS_MARY = "falas_mary_jm"   # opcional (coluna: fala)

# Modelos (pode expandir depois)
MODELOS_OPENROUTER = {
    "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
    "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
    "🧠 DeepSeek R1T2 Chimera ★★★★ (free)": "tngtech/deepseek-r1t2-chimera:free",
    "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
    "⚡ Google Gemini 2.5 Pro": "google/gemini-2.5-pro",
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
    "🧠 Qwen2.5-VL (72B) Instruct (Together)": "Qwen/Qwen2.5-VL-72B-Instruct",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
    "👑 DeepSeek R1-0528 (Together)": "deepseek-ai/DeepSeek-R1",
}

MODELOS_HF = {
    "Llama 3.1 8B Instruct (HF)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5 7B Instruct (HF)":   "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai-org: GLM-4.5-Air (HF)":  "zai-org/GLM-4.5-Air",
    "Mixtral 8x7B Instruct (HF)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "DeepSeek R1 (HF)":           "deepseek-ai/DeepSeek-R1",
}


@st.cache_data(ttl=15, show_spinner=False)
def _lms_models_dict() -> Dict[str, str]:
    """
    Lê /v1/models do LM Studio e devolve um dict {"<rótulo (LM Studio)>": "<id>"}
    """
    try:
        base = (st.session_state.get("lms_base_url") or LMS_BASE_URL or "").rstrip("/")
        if not base:
            return {"<digite manualmente> (LM Studio)": "llama-3.1-8b-instruct"}
        url = base + "/models"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {f"{m['id']} (LM Studio)": m["id"] for m in data.get("data", []) if m.get("id")}
    except Exception:
        # Fallback: ainda mostra uma opção manual
        return {"<digite manualmente> (LM Studio)": "llama-3.1-8b-instruct"}

# === Roleplay: força parágrafos e falas em linhas separadas ===

MAX_SENT_PER_PARA = 2
MAX_CHARS_PER_PARA = 240

_DASHES = re.compile(r"(?:\s*--\s*|\s*–\s*|\s*—\s*)")    # normaliza -- / – / —
_SENT_END = re.compile(r"[.!?…](?=\s|$)")                # fim de frase
_UPPER_OR_DASH = re.compile(r"([.!?…])\s+(?=(?:—|[A-ZÁÉÍÓÚÂÊÔÃÕÀÇ0-9]))")

def roleplay_paragraphizer(t: str) -> str:
    if not t:
        return ""
    # 1) Normaliza travessão e força quebra antes de qualquer fala
    t = _DASHES.sub("\n— ", t)

    # 2) Quebra após ponto/exclamação/interrogação quando vier outra frase/fala
    t = _UPPER_OR_DASH.sub(r"\1\n", t)

    # 3) Limpa espaços
    t = re.sub(r"[ \t]+", " ", t)
    linhas = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # 4) Agrupa narrativa (máx 2 frases ou 240 chars); falas isoladas
    out, buf = [], []
    sent = chars = 0

    for ln in linhas:
        if ln.startswith("—"):
            if buf:
                out.append(" ".join(buf).strip())
                out.append("")  # linha em branco entre parágrafos
                buf, sent, chars = [], 0, 0
            out.append(ln)      # fala fica sozinha
        else:
            buf.append(ln)
            sent += len(_SENT_END.findall(ln))
            chars += len(ln)
            if sent >= MAX_SENT_PER_PARA or chars >= MAX_CHARS_PER_PARA:
                out.append(" ".join(buf).strip())
                out.append("")
                buf, sent, chars = [], 0, 0

    if buf:
        out.append(" ".join(buf).strip())

    # 5) Remove brancos duplicados e finais
    final = []
    for ln in out:
        if ln == "" and (not final or final[-1] == ""):
            continue
        final.append(ln)
    if final and final[-1] == "":
        final.pop()
    return "\n".join(final).strip()


def model_id_for_together(api_ui_model_id: str) -> str:
    key = (api_ui_model_id or "").strip()
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in key:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    low = key.lower()
    if low.startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return key or "mistralai/Mixtral-8x7B-Instruct-v0.1"

def api_config_for_provider(provider: str):
    if provider == "OpenRouter":
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            st.secrets.get("OPENROUTER_API_KEY", ""),
            MODELOS_OPENROUTER,
        )
    elif provider == "Hugging Face":
        return (
            "HF_CLIENT",
            st.secrets.get("HUGGINGFACE_API_KEY", ""),
            MODELOS_HF,
        )
    elif provider == "LM Studio":
        return (
            (st.session_state.get("lms_base_url") or LMS_BASE_URL).rstrip("/") + "/chat/completions",
            "",
            _lms_models_dict(),
        )
    else:  # Together
        return (
            "https://api.together.xyz/v1/chat/completions",
            st.secrets.get("TOGETHER_API_KEY", ""),
            MODELOS_TOGETHER_UI,
        )


# =========================
# BACKOFF + CACHES
# =========================

def _retry_429(callable_fn, *args, _retries=5, _base=0.6, **kwargs):
    for i in range(_retries):
        try:
            return callable_fn(*args, **kwargs)
        except (APIError, GSpreadException) as e:
            msg = (str(e) or "").lower()
            transient = any(k in msg for k in ["429", "quota", "rate", "timed out", "timeout", "backenderror", "deadline"])
            if transient:
                time.sleep((_base * (2 ** i)) + random.uniform(0, 0.25))
                continue
            raise
    return callable_fn(*args, **kwargs)


# ========= helpers do Google Sheets =========

def _rows_to_records(vals):
    if not vals:
        return []
    header = vals[0] if vals else []
    body = vals[1:] if len(vals) > 1 else []
    if not header or all(not (c or "").strip() for c in header):
        maxlen = max((len(r) for r in body), default=0)
        header = [f"col_{i+1}" for i in range(maxlen)]
    else:
        seen, new = set(), []
        for i, h in enumerate(header):
            name = (h or f"col_{i+1}").strip() or f"col_{i+1}"
            base, j = name, 2
            while name in seen:
                name = f"{base}_{j}"
                j += 1
            seen.add(name)
            new.append(name)
        header = new
    out = []
    for r in body:
        row = list(r) + [""] * max(0, len(header) - len(r))
        row = row[:len(header)]
        out.append({header[i]: row[i] for i in range(len(header))})
    return out


@st.cache_data(ttl=45, show_spinner=False)
def _sheet_all_records_cached(sheet_name: str):
    ws = _ws(sheet_name, create_if_missing=False)
    if not ws:
        return []
    try:
        return _retry_429(ws.get_all_records)
    except GSpreadException:
        vals = _retry_429(ws.get_all_values)
        return _rows_to_records(vals)
    except Exception as e:
        st.warning(f"Falha ao ler '{sheet_name}': {e}")
        return []


@st.cache_data(ttl=45, show_spinner=False)
def _sheet_all_values_cached(sheet_name: str):
    ws = _ws(sheet_name, create_if_missing=False)
    if not ws:
        return []
    try:
        return _retry_429(ws.get_all_values)
    except Exception as e:
        st.warning(f"Falha ao ler valores de '{sheet_name}': {e}")
        return []


def _invalidate_sheet_caches():
    try:
        _sheet_all_records_cached.clear()
        _sheet_all_values_cached.clear()
    except Exception:
        pass


# =========================
# GOOGLE SHEETS
# =========================

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
            elif name == TAB_FALAS_MARY:
                _retry_429(ws.append_row, ["fala"])
            return ws
        except Exception:
            return None


# =========================
# UTILIDADES: MEMÓRIAS / HISTÓRICO
# =========================

TAB_MEMORIAS   = "memoria_jm"      # cabeçalho: tipo | conteudo | timestamp
TAB_INTERACOES = "interacoes_jm"
TAB_PERFIL     = "perfil_jm"       # aba que guarda 'resumo'


CTX_INICIAL = {
    "tempo": None,        # ex: "Domingo de manhã"
    "lugar": None,        # ex: "em casa"
    "figurino": None,     # ex: "short jeans e regata branca"
    "topico": None,       # assunto resumido do turno
    "diretiva": None,     # comando bruto do usuário
}

def extrair_diretriz_contexto(texto_usuario: str, ctx: dict | None = None) -> dict:
    ctx = dict(ctx) if ctx else dict(CTX_INICIAL)
    t = (texto_usuario or "").strip()
    ctx["diretiva"] = t

    # Tempo
    m_tempo = re.search(
        r"(?i)\b(hoje|amanh[ãa]|ontem|segunda|ter[cç]a|quarta|quinta|sexta|s[áa]bado|domingo)(?:\s+de\s+(manh[ãa]|tarde|noite))?",
        t)
    if m_tempo:
        p1 = m_tempo.group(1).capitalize()
        p2 = f" de {m_tempo.group(2)}" if m_tempo.group(2) else ""
        ctx["tempo"] = f"{p1}{p2}"

    # Lugar
    m_lugar = re.search(
        r"(?i)\b(em casa|no apartamento|na lanchonete|no shopping|na escola|no est[úu]dio|no quarto|no bar|na praia)\b", t)
    if m_lugar:
        ctx["lugar"] = m_lugar.group(1)

    # Figurino
    m_fig = re.search(r"(?i)\b(veste|usando|de)\s+([a-z0-9\s\-ãáéíóúç]+)", t)
    if m_fig:
        ctx["figurino"] = (m_fig.group(2) or "").strip()

    # Tópico do turno
    m_top = re.search(r":\s*(.+)$", t)
    ctx["topico"] = (m_top.group(1).strip() if m_top else t)[:140]

    # Herda últimos valores se vazio
    for campo in ["tempo", "lugar", "figurino"]:
        if not ctx.get(campo):
            ultimo_ctx = st.session_state.get("ctx_cena", {})
            if ultimo_ctx and ultimo_ctx.get(campo):
                ctx[campo] = ultimo_ctx.get(campo)

    return ctx

def gerar_linha_abertura(ctx: dict) -> str:
    if not ctx:
        return ""
    tempo = (ctx.get("tempo") or "").strip().rstrip(".")
    figurino = (ctx.get("figurino") or "").strip().rstrip(".")
    lugar = (ctx.get("lugar") or "").strip().rstrip(".")

    pedacos = []
    if tempo:
        pedacos.append(tempo.capitalize())
    mary_part = "Mary"
    if figurino:
        mary_part += f", {figurino}"
    pedacos.append(mary_part)
    if lugar:
        pedacos.append(lugar)

    if not any(pedacos):
        return ""
    return ". ".join(pedacos) + "."


# ------------- VIRGINDADE -------------

PADROES_NAO_VIRGEM = [
    r"\b(n[aã]o\s+sou\s+mais\s+virgem)\b",
    r"\b(deix(ei|ou)\s+de\s+ser\s+virgem)\b",
    r"\b(perd[ei]u?\s+a\s+virgindade)\b",
    r"\b(minha|sua|nossa)\s+primeira\s+vez\s+(foi|aconteceu|rolou)\b",
    r"\b(tivemos|tive)\s+(a\s+)?primeira\s+vez\b",
]
PADROES_VIRGEM = [
    r"\b(sou|era|continuo?|permane[cç]o)\s+virgem\b",
    r"\b(nunca\s+(tran(s|ç)ei|fiz\s+sexo|tive\s+rela[cç][oõ]es))\b",
    r"\b(virgindade)\b(?!\s*(perd[ií]|\bn[aã]o\b))",
]

def _to_dt(ts: str) -> Optional[datetime]:
    ts = (ts or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None

def _ultimo_evento_virgindade_memoria(ate: Optional[datetime] = None) -> Optional[Tuple[bool, datetime, str]]:
    try:
        buckets = carregar_memorias_brutas()
    except Exception:
        buckets = {}
    candidatos: List[Tuple[bool, datetime]] = []
    ate = ate or datetime.now()

    for tag in ("[mary]", "[all]"):
        for item in buckets.get(tag, []) or []:
            txt = (item.get("conteudo") or "").strip()
            ts = _to_dt(item.get("timestamp"))
            if ts and ts > ate:
                continue
            low = txt.lower()
            if any(re.search(p, low, re.IGNORECASE) for p in PADROES_NAO_VIRGEM):
                candidatos.append((False, ts or datetime.min))
            elif any(re.search(p, low, re.IGNORECASE) for p in PADROES_VIRGEM):
                candidatos.append((True, ts or datetime.min))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: (x[1], 0 if x[0] is False else 1))
    estado, ts = candidatos[-1]
    return (estado, ts, "memoria_jm")

def _ultimo_evento_virgindade_interacoes(ate: Optional[datetime] = None) -> Optional[Tuple[bool, datetime, str]]:
    ate = ate or datetime.now()
    inter = carregar_interacoes(n=40)
    if not inter:
        return None

    candidatos: List[Tuple[bool, datetime]] = []
    for r in inter:
        ts = _to_dt(r.get("timestamp"))
        if ts and ts > ate:
            continue
        low = (r.get("content") or "").strip().lower()
        if any(re.search(p, low, re.IGNORECASE) for p in PADROES_NAO_VIRGEM):
            candidatos.append((False, ts or datetime.min))
        elif any(re.search(p, low, re.IGNORECASE) for p in PADROES_VIRGEM):
            candidatos.append((True, ts or datetime.min))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: (x[1], 0 if x[0] is False else 1))
    estado, ts = candidatos[-1]
    return (estado, ts, "interacoes_jm")

def estado_virgindade_ate(ate: Optional[datetime] = None) -> Optional[bool]:
    ate = ate or datetime.now()
    res = _ultimo_evento_virgindade_memoria(ate)
    if res is None:
        res = _ultimo_evento_virgindade_interacoes(ate)
    return None if res is None else res[0]


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

def carregar_interacoes(n: int = 25):
    cache = st.session_state.get("_cache_interacoes", None)
    if cache is None:
        regs = _sheet_all_records_cached(TAB_INTERACOES)
        norm = []
        for r in regs:
            norm.append({
                "timestamp": (r.get("timestamp") or "").strip(),
                "role": (r.get("role") or "user").strip(),
                "content": (r.get("content") or "").strip(),
            })
        st.session_state["_cache_interacoes"] = norm
        cache = norm
    return cache[-n:] if len(cache) > n else cache

def salvar_interacao(role: str, content: str):
    if not planilha:
        return
    try:
        aba = _ws(TAB_INTERACOES)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, f"{role or ''}".strip(), f"{content or ''}".strip()]
        _retry_429(aba.append_row, row, value_input_option="RAW")
        lst = st.session_state.get("_cache_interacoes")
        if isinstance(lst, list):
            lst.append({"timestamp": timestamp, "role": row[1], "content": row[2]})
        else:
            st.session_state["_cache_interacoes"] = [{"timestamp": timestamp, "role": row[1], "content": row[2]}]
        _invalidate_sheet_caches()
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")


# =========================
# MEMÓRIA LONGA (opcional simples)
# =========================

def memoria_longa_listar_registros():
    try:
        return _sheet_all_records_cached(TAB_ML)
    except Exception:
        return []

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
    aba = _ws(TAB_ML, create_if_missing=False)
    if not aba:
        return False
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [texto.strip(), "", (tags or "").strip(), ts, 1.0]
    try:
        _retry_429(aba.append_row, linha, value_input_option="RAW")
        _invalidate_sheet_caches()
        return True
    except Exception:
        return False

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78, ate_ts=None):
    try:
        dados = _sheet_all_records_cached(TAB_ML)
    except Exception:
        return []
    def _tok(s): return set(re.findall(r"[a-zà-ú0-9]+", (s or "").lower()))
    s2 = _tok(query_text)
    cands = []
    for row in dados:
        texto = (row.get("texto") or "").strip()
        if not texto:
            continue
        row_ts = (row.get("timestamp") or "").strip()
        if ate_ts and row_ts and row_ts > ate_ts:
            continue
        s1 = _tok(texto)
        sim = len(s1 & s2) / max(1, len(s1 | s2))
        try:
            score = float(row.get("score", 1.0) or 1.0)
        except Exception:
            score = 1.0
        if sim >= limiar:
            rr = 0.7*sim + 0.3*score
            cands.append((texto, score, sim, rr))
    cands.sort(key=lambda x: x[3], reverse=True)
    return cands[:k]

def memoria_longa_reforcar(textos_usados: list):
    aba = _ws(TAB_ML, create_if_missing=False)
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
# REGRAS DE FASE (APENAS FASE)
# =========================

FASES_ROMANCE: Dict[int, Dict[str, str]] = {
    0: {"nome": "Estranhos",
        "permitidos": "olhares; near-miss (mesmo café/rua/ônibus); detalhe do ambiente",
        "proibidos": "troca de nomes; toques; conversa pessoal; beijo"},
    1: {"nome": "Percepção",
        "permitidos": "cumprimento neutro; pergunta impessoal curta; beijo no rosto",
        "proibidos": "beijo na boca; confidências"},
    2: {"nome": "Conhecidos",
        "permitidos": "troca de nomes; pequena ajuda; 1 pergunta pessoal leve; beijo suave na boca",
        "proibidos": "toque prolongado; encontro a sós planejado"},
    3: {"nome": "Romance",
        "permitidos": "conversa 10–20 min; caminhar juntos; trocar contatos; beijos intensos (sem carícias íntimas)",
        "proibidos": "carícias íntimas; tirar roupas"},
    4: {"nome": "Namoro",
        "permitidos": "beijos intensos; carícias íntimas; **sem clímax** até usuário liberar",
        "proibidos": "sexo explícito sem consentimento claro"},
    5: {"nome": "Compromisso / Encontro definitivo",
        "permitidos": "beijos intensos; carícias íntimas; sexo com consentimento; **clímax somente se usuário liberar**",
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


# =========================
# FALAS DA MARY (BRANDAS) + PLANILHA
# =========================

FALAS_EXPLICITAS_MARY = [
    "Vem mais perto...sem pressa.",
    "Assim está bom… continua...assim... desse jeito.",
    "Eu quero sentir você...devagar.",
    "Fica comigo....só mais um pouco.",
]

def carregar_falas_mary() -> List[str]:
    try:
        ws = _ws(TAB_FALAS_MARY, create_if_missing=False)
        if not ws:
            return []
        rows = _sheet_all_records_cached(TAB_FALAS_MARY)
        falas = []
        for r in rows:
            fala = (r.get("fala") or "").strip()
            if fala:
                falas.append(fala)
        return falas or []
    except Exception:
        return []


# =========================
# AJUSTES DE TOM / SINTONIA
# =========================

def gerar_mary_sensorial(level: int = 2, n: int = 2, hair_on: bool = True, sintonia: bool = True) -> str:
    if level <= 0 or n <= 0:
        return ""

    base_leve = [
        "Os cabelos de Mary — negros, volumosos, levemente ondulados — acompanham o passo.",
        "O olhar de Mary é firme e sereno; a respiração, tranquila.",
        "Há calor discreto no perfume que ela deixa no ar.",
        "O sorriso surge pequeno, sincero e atento.",
    ]
    base_marcado = [
        "O tecido roça levemente nas pernas; o passo é seguro, cadenciado.",
        "Os ombros relaxam quando ela encontra o olhar de Jânio.",
        "A pele arrepia sutil ao menor toque.",
        "O olhar de Mary segura o dele por um instante a mais.",
    ]
    base_ousado = [
        "O ritmo do corpo de Mary é deliberado; chama sem exigir.",
        "O perfume dela exala convidando a aproximação.",
        "Os lábios entreabertos, esperando o momento certo.",
        "O olhar pousa e permanece, sugerindo contato.",
    ]

    if level == 1:
        pool = list(base_leve)
    elif level == 2:
        pool = list(base_leve) + list(base_marcado)
    else:
        pool = list(base_leve) + list(base_marcado) + list(base_ousado)

    termos_banidos = re.compile(
        r"\b(c[ée]u|mar|onda?s?|vento|brisa|chuva|nublado|luar|horizonte|pier|paisage?m|cen[áa]rio|amanhecer|entardecer|p[ôo]r do sol)\b",
        re.IGNORECASE,
    )
    pool = [f for f in pool if not termos_banidos.search(f)]

    if sintonia:
        filtros = [r"\bexigir\b"]
        def _ok(fr): return not any(re.search(p, fr, re.I) for p in filtros)
        pool = [f for f in pool if _ok(f)]
        pool.extend([
            "A respiração de Mary busca o mesmo compasso de Jânio.",
            "Ela desacelera e deixa o momento guiar.",
            "O toque começa suave, sem precipitação.",
        ])

    n_eff = max(1, min(n, len(pool)))
    frases = random.sample(pool, k=n_eff) if pool else []

    if hair_on:
        hair_line = "Os cabelos negros, volumosos e levemente ondulados moldam o rosto quando ela vira de leve."
        if hair_line not in frases:
            frases.insert(0, hair_line)
            if len(frases) > n_eff:
                frases = frases[:n_eff]
    return " ".join(frases)

def _last_user_text(hist):
    if not hist:
        return ""
    for r in reversed(hist):
        if str(r.get("role","")).lower() == "user":
            return r.get("content","")
    return ""

def prompt_da_cena(ctx: dict | None = None, modo_finalizacao: str = "ponte") -> str:
    ctx = ctx or {}
    tempo    = (ctx.get("tempo") or "").strip().rstrip(".")
    lugar    = (ctx.get("lugar") or "").strip().rstrip(".")
    figurino = (ctx.get("figurino") or "").strip().rstrip(".")

    pedacos = []
    if tempo: pedacos.append(tempo.capitalize())
    pedacos.append(f"Mary{(', ' + figurino) if figurino else ''}".strip())
    if lugar: pedacos.append(lugar)
    primeira_linha = ". ".join([p for p in pedacos if p]) + "."

    modo = (modo_finalizacao or "ponte").lower()
    if modo == "eu":
        regra_fim = "- Feche com 1 frase curta em 1ª pessoa (Mary), conectando o próximo gesto."
    elif modo == "seco":
        regra_fim = "- Termine sem gancho, frase final objetiva."
    else:  # ponte
        regra_fim = "- Feche com micro-ação que deixe gancho natural para continuação, sem concluir a cena."

    return (
        "### Diretrizes de abertura e fechamento\n"
        "- Comece com UMA linha: Tempo. Mary[, figurino]. [Lugar]. (somente se houver dados; caso contrário, pule)\n"
        "- Em seguida, entre direto em ação e diálogo, focada na DIRETIVA do usuário.\n"
        f"{regra_fim}\n"
        f"ABERTURA_SUGERIDA: {primeira_linha if primeira_linha != '.' else ''}\n"
    )

def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    calor = int(st.session_state.get("nsfw_max_level", 0))
    regras = f"""
⚖️ Regras de coerência:
- Narre em terceira pessoa; não se dirija ao leitor como "você".
- Consentimento claro antes de qualquer gesto significativo.
- Mary prefere ritmo calmo, sintonizado com o parceiro (modo harmônico ativo).
- Linguagem sensual proporcional ao nível de calor ({calor}).
- Proibido natureza/ambiente/clima/metáforas (céu, mar, vento, ondas, luar, paisagem etc.).
- Sem “fade to black”: a progressão é mostrada, mas sem pornografia explícita.
""".strip()
    return prompt_base + "\n" + regras

def construir_prompt_com_narrador() -> str:
    BLOCO_RESTRICAO_SENSORY = """
NUNCA escreva frases sobre ambiente, clima, natureza, luz, pier, mar, vento, céu, luar, som das ondas, paisagem, cenário ou metáforas.
NÃO inicie textos com lugar ou "Pier de Camburi — Noite —", nem descreva onde estão ou o horário.
PROIBIDO absolutamente qualquer menção a natureza, cenário, paisagem, efeitos de clima ou metáforas.
Apenas sensação física, diálogo direto, calor, suor, desejo, roçar, toque, excitação, palavras, gemidos, ações do corpo, reação, ritmo físico.
""".strip()

    ctx = st.session_state.get("ctx_cena", {})
    cena_bloco = prompt_da_cena(ctx, st.session_state.get("finalizacao_modo", "ponte"))

    def _hist_sanitizado(hist):
        _split = re.compile(r'(?<=[\.\!\?])\s+')
        _amb = re.compile(
            r'\b(c[ée]u|nuvens?|horizonte|luar|mar|onda?s?|pier|praia|vento|brisa|chuva|garoa|sereno|amanhecer|entardecer|p[ôo]r do sol|paisage?m|cen[áa]rio|temperatura|verão|quiosques?)\b',
            re.I
        )
        L = []
        for r in hist or []:
            role = r.get("role", "user")
            txt = (r.get("content") or "").strip()
            if not txt:
                continue
            s = [t for t in _split.split(txt) if t.strip() and not _amb.search(t)]
            if s:
                L.append(f"{role}: {' '.join(s)[:900]}")
        return "\n".join(L) if L else "(sem histórico)"

    memos = carregar_memorias_brutas()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    modo_mary = bool(st.session_state.get("interpretar_apenas_mary", False))

    _sens_on = bool(st.session_state.get("mary_sensorial_on", True))
    _sens_level = int(st.session_state.get("mary_sensorial_level", 2))
    _sens_n = int(st.session_state.get("mary_sensorial_n", 2))
    mary_sens_txt = (
        gerar_mary_sensorial(_sens_level, n=_sens_n, sintonia=bool(st.session_state.get("modo_sintonia", True)))
        if _sens_on else ""
    )

    ritmo_cena = int(st.session_state.get("ritmo_cena", 0))
    ritmo_label = ["muito lento", "lento", "médio", "rápido"][max(0, min(3, ritmo_cena))]
    modo_sintonia = bool(st.session_state.get("modo_sintonia", True))

    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = _hist_sanitizado(hist)
    ate_ts = _parse_ts(hist[-1].get("timestamp", "")) if hist else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                ml_topk_txt = "\n".join([f"- {t}" for (t, *_rest) in topk])
                st.session_state["_ml_topk_texts"] = [t for (t, *_r) in topk]
        except Exception:
            st.session_state["_ml_topk_texts"] = []

    memos_all = [
        (d.get("conteudo") or "").strip()
        for d in memos.get("[all]", [])
        if isinstance(d, dict) and d.get("conteudo") and (not d.get("timestamp") or d.get("timestamp") <= ate_ts)
    ]
    st.session_state["_ml_recorrentes"] = memos_all

    dossie = []
    m = persona_block_temporal("mary", memos, ate_ts, 8)
    j = persona_block_temporal("janio", memos, ate_ts, 8)
    if m:
        dossie.append(m)
    if j:
        dossie.append(j)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    falas_mary_bloco = ""
    st.session_state["_falas_mary_list"] = []
    if st.session_state.get("usar_falas_mary", False):
        falas = carregar_falas_mary() or FALAS_EXPLICITAS_MARY
        if falas:
            st.session_state["_falas_mary_list"] = falas[:]
            falas_mary_bloco = (
                "### Falas de Mary — use literalmente 1–2 destas (no máximo 1 por parágrafo)\n"
                "NÃO reescreva as frases abaixo; quando usar, mantenha exatamente como está.\n"
                + "\n".join(f"- {s}" for s in falas)
            )

    sintonia_bloco = ""
    if modo_sintonia:
        sintonia_bloco = (
            "### Sintonia & Ritmo (prioritário)\n"
            f"- Ritmo da cena: **{ritmo_label}**.\n"
            "- Condução harmônica: Mary sintoniza com o parceiro; evite ordens ríspidas/imperativas. Prefira convites e pedidos gentis.\n"
            "- Pausas e respiração contam; mostre desejo pela troca, não por imposição.\n"
        )

    try:
        _ref_dt = _to_dt(ate_ts)
        est = estado_virgindade_ate(_ref_dt)
        if est is True:
            virg_bloco = (
                "### Estado canônico — Virgindade\n"
                "- Mary é **virgem** neste momento da história.\n"
                "- Mantenha coerência: sem histórico de penetração; se houver exploração íntima, trate como **primeira descoberta**, cuidadosa e sem contradições.\n"
            )
        elif est is False:
            virg_bloco = (
                "### Estado canônico — Virgindade\n"
                "- Mary **não é mais virgem** (evento passado registrado).\n"
                "- Não reescreva o passado; mantenha consistência com a cena que marcou a primeira vez.\n"
            )
        else:
            virg_bloco = (
                "### Estado canônico — Virgindade\n"
                "- **Sem evidência temporal** suficiente: **não** afirme que é a primeira vez e **não** invente perda de virgindade.\n"
            )
    except Exception:
        virg_bloco = (
            "### Estado canônico — Virgindade\n"
            "- Falha ao ler o estado; **evite** afirmar status e **não** contradiga cenas anteriores.\n"
        )

    climax_bloco = ""
    fase = int(st.session_state.get("mj_fase", 0))
    if bool(st.session_state.get("app_bloqueio_intimo", True)) and fase < 5:
        climax_bloco = (
            "### Proteção de avanço íntimo (ATIVA)\n"
            "- **Sem clímax por padrão**: não descreva orgasmo/finalização **a menos que o usuário tenha liberado explicitamente na mensagem anterior**.\n"
            "- Encerre em **pausa sensorial** (respiração, silêncio, carinho), **sem** 'fade-to-black'.\n"
        )

    modo_mary = bool(st.session_state.get("interpretar_apenas_mary", False))
    if modo_mary:
        papel_header = "Você é **Mary**, responda **em primeira pessoa**, sem narrador externo. Use apenas o que Mary vê/sente/ouve. Não descreva pensamentos de Jânio. Não use títulos nem repita instruções."
        regra_saida = "- Narre **em primeira pessoa (eu)** como Mary; nunca use narrador onisciente.\n- Produza uma cena fechada e natural, sem comentários externos."
        formato_cena = "- DIÁLOGOS diretos com travessão (—), intercalados com ação/reação **em 1ª pessoa (Mary)**."
    else:
        papel_header = "Você é o **Narrador** de um roleplay dramático brasileiro; foque em Mary e Jânio. Não repita instruções nem títulos."
        regra_saida = "- Narre **em terceira pessoa**; nunca fale com 'você'.\n- Produza uma cena fechada e natural, sem comentários externos."
        formato_cena = "- DIÁLOGOS diretos com travessão (—), intercalados com ação/reação física/visual."
        climax_bloco += (
            "### Regra permanente de clímax\n"
            "- **Não** descreva orgasmo/ejaculação/clímax **sem liberação explícita na ÚLTIMA fala do usuário**.\n"
            "- Se não houver liberação, pare no **limiar**: respiração, tremor, pausa; **sem finalizar**.\n"
            "- **Sem fade-to-black** em qualquer circunstância.\n"
        )

    prompt = f"""
{BLOCO_RESTRICAO_SENSORY}
{cena_bloco}
{papel_header}
{sintonia_bloco}{virg_bloco}{climax_bloco}{falas_mary_bloco}
### Dossiê (personas)
{dossie_txt}
### Diretrizes gerais (ALL)
{chr(10).join(f"- {c}" for c in memos_all) if memos_all else "(vazio)"}
### Perfil (resumo mais recente)
{perfil or "(vazio)"}
### Histórico recente (sanitizado)
{hist_txt}
### BLOCO DE ESTILO E TOM — ROLEPLAY SENSORIAL
- Parágrafos curtos (até 2 frases), intercale fala e ação.
- Sem cenário/clima/metáforas. Foque corpo, gesto, diálogo, desejo, contato.
### Camada sensorial — Mary (OBRIGATÓRIA no 1º parágrafo)
{mary_sens_txt or "- Apenas sensações físicas, nunca ambiente."}
### Memória longa — Top-K relevantes
{ml_topk_txt}
### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}
### Formato OBRIGATÓRIO da cena
{formato_cena}
### Regra de saída
{regra_saida}
""".strip()

    return inserir_regras_mary_e_janio(prompt)


# --- Remoção de "paisagem/clima" (sem mexer em sentido da cena) ---
SCENERY_TERMS = [
    r"c[ée]u", r"nuvens?", r"horizonte", r"luar",
    r"mar", r"onda?s?", r"pier",
    r"vento", r"brisa", r"neblina|brumas?",
    r"chuva|garoa|sereno",
    r"amanhecer|entardecer|crep[uú]sculo|p[ôo]r do sol",
    r"paisage?m|cen[áa]rio",
    r"luz\s+do\s+luar", r"som\s+das?\s+ondas?"
]
SCENERY_WORD = re.compile(r"\b(" + "|".join(SCENERY_TERMS) + r")\b", re.IGNORECASE)

def sanitize_scenery(text: str) -> str:
    if not text:
        return ""
    return SCENERY_WORD.sub("", text)

def sanitize_scenery_preserve_opening(t: str) -> str:
    if not t:
        return ""
    linhas = t.strip().split('\n')
    if not linhas:
        return ""
    primeira_linha = linhas[0].strip()
    resto = '\n'.join(linhas[1:]).strip()
    if resto:
        resto_filtrado = sanitize_scenery(resto)
        return primeira_linha + ('\n' + resto_filtrado if resto_filtrado else '')
    else:
        return primeira_linha

EXPL_PAT = re.compile(
    r"\b(mamilos?|genit[aá]lia|ere[cç][aã]o|penetra[cç][aã]o|boquete|gozada|gozo|sexo oral|chupar|enfiar)\b",
    flags=re.IGNORECASE
)

def classify_nsfw_level(t: str) -> int:
    if EXPL_PAT.search(t or ""):
        return 3
    if re.search(r"\b(cintura|pesco[cç]o|costas|beijo prolongado|respira[cç][aã]o curta)\b", (t or ""), re.I):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aã]os dadas|beijo)\b", (t or ""), re.I):
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
    t = re.sub(EXPL_PAT, "[…]", t, flags=re.I)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

def is_mary_mode_active() -> bool:
    return bool(
        st.session_state.get("interpretar_apenas_mary", False)
        or st.session_state.get("modo_resposta") == "Mary (1ª pessoa)"
    )

def _render_visible(t: str) -> str:
    t = sanitize_scenery_preserve_opening(t)
    t = roleplay_paragraphizer(t)
    if st.session_state.get("app_bloqueio_intimo", True):
        t = sanitize_explicit(t, int(st.session_state.get("nsfw_max_level", 0)), action="soften")
    return t

def force_linebreak_on_falas(txt):
    return re.sub(r"([^\n])\s*(—)", r"\1\n\n\2", txt)


# ===============
# CONTEXTO LIMIT
# ===============

def _msg_len_tokens(m):
    # heurística simples: ~4 chars por token
    txt = (m.get("content") or "")
    return max(1, len(txt) // 4 + 1)

def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(_msg_len_tokens(m) for m in messages)

def compact_messages(messages: List[Dict[str, str]], max_ctx: int, reserve_out: int = 512) -> List[Dict[str, str]]:
    """
    Mantém system prompts e as últimas falas, cortando o histórico se exceder o limite aproximado.
    """
    if not messages:
        return messages
    budget = max(1024, max_ctx - reserve_out)
    # Preserve todos os systems
    systems = [m for m in messages if m.get("role") == "system"]
    non_systems = [m for m in messages if m.get("role") != "system"]
    # Comece do fim
    kept = []
    total = estimate_tokens(systems)
    for m in reversed(non_systems):
        ln = _msg_len_tokens(m)
        if total + ln > budget and kept:
            break
        kept.append(m)
        total += ln
    kept.reverse()
    out = systems + kept
    # se ainda exceder, aparar conteúdo do primeiro user
    while estimate_tokens(out) > max_ctx - reserve_out and any(m.get("role") != "system" for m in out):
        for i, m in enumerate(out):
            if m.get("role") != "system":
                c = (m.get("content") or "")
                if len(c) > 1200:
                    out[i] = {**m, "content": c[-1200:]}
                else:
                    # remove o mais antigo não-system
                    del out[i]
                break
    return out


# =========================
# UI — CABEÇALHO
# =========================

st.title("🎬 Narrador JM — Somente Fase")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# =========================
# ESTADOS INICIAIS
# =========================

for k, v in {
    "resumo_capitulo": carregar_resumo_salvo(),
    "session_msgs": [],
    "use_memoria_longa": True,
    "k_memoria_longa": 3,
    "limiar_memoria_longa": 0.78,
    "app_bloqueio_intimo": True,
    "app_emocao_oculta": "nenhuma",
    "mj_fase": mj_carregar_fase_inicial(),
    "max_avancos_por_cena": 1,
    "estilo_escrita": "AÇÃO",
    "templates_jm": {},
    "template_ativo": None,
    "etapa_template": 0,
    "ctx_cena": dict(CTX_INICIAL),
    "finalizacao_modo": "ponto de gancho",
    "ctx_max_tokens": 4096,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# SIDEBAR — Reorganizado
# =========================

with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio(
        "🌐 Provedor",
        ["OpenRouter", "Together", "Hugging Face", "LM Studio"],
        index=0,
        key="provedor_ia"
    )

    if provedor == "LM Studio":
        LMS_BASE_URL = st.text_input(
            "Base URL (LM Studio)",
            value=st.session_state.get("lms_base_url", LMS_BASE_URL),
            key="lms_base_url",
            help="Ex.: https://<tunnel>.trycloudflare.com/v1"
        )

    api_url, api_key, modelos_map = api_config_for_provider(provedor)

    if provedor in ("OpenRouter", "Together", "Hugging Face") and not api_key:
        st.warning("⚠️ API key ausente para o provedor selecionado. Defina em st.secrets.")

    if provedor == "LM Studio":
        modelos_lms = list(modelos_map.keys())
        if not modelos_lms:
            st.info("Não consegui listar modelos do LM Studio. Digite o ID manualmente.")
            modelo_nome = st.text_input("🤖 Modelo (LM Studio)", value="<digite manualmente> (LM Studio)", key="modelo_nome_ui")
            st.session_state.modelo_escolhido_id = st.text_input(
                "ID do modelo (LM Studio)",
                value="llama-3.1-8b-instruct",
                key="modelo_id_lmstudio"
            )
        else:
            modelo_nome = st.selectbox("🤖 Modelo de IA", modelos_lms, index=0, key="modelo_nome_ui")
            st.session_state.modelo_escolhido_id = modelos_map[modelo_nome]
    else:
        modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
        st.session_state.modelo_escolhido_id = modelos_map[modelo_nome]

    st.markdown("---")
    st.markdown("### ✍️ Estilo & Progresso Dramático")

    modo_op = st.selectbox(
        "Modo de resposta",
        ["Narrador padrão", "Mary (1ª pessoa)"],
        index=0,
        key="modo_resposta",
    )
    st.session_state.interpretar_apenas_mary = (modo_op == "Mary (1ª pessoa)")

    st.selectbox(
        "Estilo de escrita",
        ["AÇÃO", "ROMANCE LENTO", "NOIR"],
        index=["AÇÃO", "ROMANCE LENTO", "NOIR"].index(st.session_state.get("estilo_escrita", "AÇÃO")),
        key="estilo_escrita",
    )

    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=0, key="nsfw_max_level")

    st.checkbox(
        "Sintonia com o parceiro (modo harmônico)",
        key="modo_sintonia",
        value=st.session_state.get("modo_sintonia", True),
    )

    st.select_slider(
        "Ritmo da cena",
        options=[0, 1, 2, 3],
        value=0,
        format_func=lambda n: ["muito lento", "lento", "médio", "rápido"][n],
        key="ritmo_cena",
    )

    st.selectbox(
        "Finalização",
        ["ponto de gancho", "fecho suave", "deixar no suspense"],
        index=["ponto de gancho","fecho suave","deixar no suspense"].index(
            st.session_state.get("finalizacao_modo", "ponto de gancho")
        ),
        key="finalizacao_modo",
    )

    st.checkbox(
        "Usar falas da Mary da planilha (usar literalmente)",
        value=st.session_state.get("usar_falas_mary", False),
        key="usar_falas_mary",
    )

    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio (apenas Fase)")
    fase_default = mj_carregar_fase_inicial()
    options_fase = sorted(FASES_ROMANCE.keys())
    fase_ui_val = int(st.session_state.get("mj_fase", fase_default))
    fase_ui_val = max(min(fase_ui_val, max(options_fase)), min(options_fase))
    fase_escolhida = st.select_slider(
        "Fase do romance",
        options=options_fase,
        value=fase_ui_val,
        format_func=_fase_label,
        key="ui_mj_fase",
    )
    if fase_escolhida != st.session_state.get("mj_fase", fase_default):
        mj_set_fase(fase_escolhida, persist=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Avançar 1 fase"):
            mj_set_fase(min(st.session_state.get("mj_fase", 0) + 1, max(options_fase)), persist=True)
    with col_b:
        if st.button("↺ Reiniciar (0)"):
            mj_set_fase(0, persist=True)

    st.markdown("---")
    st.checkbox(
        "Evitar coincidências forçadas (montagem paralela A/B)",
        value=st.session_state.get("no_coincidencias", True),
        key="no_coincidencias",
    )
    st.checkbox(
        "Bloquear avanços íntimos sem ordem",
        value=st.session_state.get("app_bloqueio_intimo", True),
        key="app_bloqueio_intimo",
    )
    st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.get("app_emocao_oculta", "nenhuma")),
        key="ui_app_emocao_oculta",
    )
    st.session_state.app_emocao_oculta = st.session_state.get("ui_app_emocao_oculta", "nenhuma")

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 900)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")
    st.number_input("Contexto máx (tokens, LM Studio)", min_value=2048, max_value=32768, value=int(st.session_state.get("ctx_max_tokens", 4096)), step=512, key="ctx_max_tokens")

    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")
    st.checkbox("Usar memória longa no prompt", value=st.session_state.get("use_memoria_longa", True), key="use_memoria_longa")
    st.slider("Top-K memórias", 1, 5, int(st.session_state.get("k_memoria_longa", 3)), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95, float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")

    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 12)), step=1, key="n_sheet_prompt")

    st.markdown("---")
    st.markdown("### 📝 Utilitários")

    # Gerar resumo do capítulo (pega as últimas interações do Sheets)
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in inter) if inter else ""
            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileira, mantendo tom e emoções.\n\n"
                + texto + "\n\nResumo:"
            )
    
            # Usa o provedor/modelo selecionados no topo do sidebar
            provedor = st.session_state.get("provedor_ia", "OpenRouter")
            api_url_local, api_key_local, _catalogo = api_config_for_provider(provedor)
    
            model_id_call = (
                model_id_for_together(st.session_state.modelo_escolhido_id)
                if provedor == "Together"
                else st.session_state.modelo_escolhido_id
            )
    
            if provedor == "Hugging Face":
                try:
                    hf_client = InferenceClient(
                        token=api_key_local,
                        timeout=int(st.session_state.get("timeout_s", 300))
                    )
                    out = hf_client.chat.completions.create(
                        model=model_id_call,
                        messages=[{"role": "user", "content": prompt_resumo}],
                        max_tokens=800,
                        temperature=0.85,
                        stream=False,
                    )
                    resumo = out.choices[0].message.content.strip()
                    st.session_state.resumo_capitulo = resumo
                    salvar_resumo(resumo)
                    st.success("Resumo gerado e salvo com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao resumir (HF): {e}")
            else:
                headers = {"Content-Type": "application/json"}
                if api_key_local:
                    headers["Authorization"] = f"Bearer {api_key_local}"
    
                payload = {
                    "model": model_id_call,
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85,
                }
    
                r = requests.post(
                    api_url_local,
                    headers=headers,
                    json=payload,
                    timeout=int(st.session_state.get("timeout_s", 300)),
                )
                r.raise_for_status()
                data = r.json()
    
                resumo = None
                if isinstance(data, dict):
                    ch = data.get("choices") or []
                    if ch and isinstance(ch[0], dict):
                        resumo = (
                            ch[0].get("message", {}).get("content")
                            or ch[0].get("text")
                            or ch[0].get("delta", {}).get("content")
                        )
                if not resumo:
                    resumo = json.dumps(data)
    
                resumo = (resumo or "").strip()
                st.session_state.resumo_capitulo = resumo
                salvar_resumo(resumo)
                st.success("Resumo gerado e salvo com sucesso!")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")


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
    salvar_interacao("user", str(entrada))
    st.session_state.session_msgs.append({"role": "user", "content": str(entrada)})

    st.session_state["ctx_cena"] = extrair_diretriz_contexto(
        entrada,
        st.session_state.get("ctx_cena", CTX_INICIAL)
    )
    ctx = st.session_state["ctx_cena"]

    linha_abertura = gerar_linha_abertura(ctx)

    mary_mode_active = bool(
        st.session_state.get("interpretar_apenas_mary")
        or st.session_state.get("modo_resposta") == "Mary (1ª pessoa)"
    )

    # Historico com linha de abertura anexada na última fala do usuário
    historico: List[Dict[str, str]] = []
    last_idx = len(st.session_state.session_msgs) - 1
    for ix, m in enumerate(st.session_state.session_msgs):
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if ix == last_idx and role.lower() == "user":
            content = (linha_abertura.strip() + ("\n" + content if content else "")).strip()
        if mary_mode_active and role.lower() == "user":
            content = f"JÂNIO: {content}"
        historico.append({"role": role, "content": content})

    prompt = construir_prompt_com_narrador()

    # Provedor / modelo
    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets.get("TOGETHER_API_KEY", "")
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
        need_auth = True
    elif prov == "Hugging Face":
        endpoint = "HF_CLIENT"
        auth = st.secrets.get("HUGGINGFACE_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id
        need_auth = True
    elif prov == "LM Studio":
        endpoint = (st.session_state.get("lms_base_url") or LMS_BASE_URL).rstrip("/") + "/chat/completions"
        auth = ""
        model_to_call = st.session_state.modelo_escolhido_id
        need_auth = False
    else:  # OpenRouter
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets.get("OPENROUTER_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id
        need_auth = True

    headers = {"Content-Type": "application/json"}
    if need_auth:
        if not auth:
            st.error("A chave de API do provedor selecionado não foi definida em st.secrets.")
            st.stop()
        headers["Authorization"] = f"Bearer {auth}"

    system_pt = {"role": "system", "content": "Responda em português do Brasil. Mostre apenas a narrativa final."}
    system_mary = {
        "role": "system",
        "content": (
            "MODO MARY (ATIVO):\n"
            "- Trate a fala do usuário como ações/falas de Jânio.\n"
            "- Responda SOMENTE como Mary, em primeira pessoa.\n"
            "- Não invente falas de Jânio; descreva apenas o que Mary diz/sente/faz.\n"
            "- Se usar diálogo, use travessão (—) apenas para a fala de Mary."
        )
    }

    messages = [system_pt]
    if mary_mode_active:
        messages.append(system_mary)
    messages.append({"role": "system", "content": prompt})
    messages += historico

    # Ajuste para o contexto do modelo
    ctx_max = int(st.session_state.get("ctx_max_tokens", 4096))
    messages_fit = compact_messages(messages, max_ctx=ctx_max, reserve_out= max(256, int(st.session_state.get("max_tokens_rsp", 900)) + 128))

    payload_base = {
        "model": model_to_call,
        "messages": messages_fit,
        "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
        "temperature": 0.9,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        # Reforço memórias usadas no prompt
        try:
            usados_prompt = []
            usados_prompt.extend(st.session_state.get("_ml_topk_texts", []))
            usados_prompt.extend(st.session_state.get("_ml_recorrentes", []))
            usados_prompt = [t for t in usados_prompt if t]
            if usados_prompt:
                memoria_longa_reforcar(usados_prompt)
        except Exception:
            pass

        # STREAM
        try:
            if prov == "Hugging Face":
                hf_client = InferenceClient(
                    token=auth,
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
                for chunk in hf_client.chat.completions.create(
                    model=model_to_call,
                    messages=messages_fit,
                    temperature=0.9,
                    max_tokens=int(st.session_state.get("max_tokens_rsp", 900)),
                    stream=True,
                ):
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if not delta:
                        continue
                    resposta_txt += delta
                    if time.time() - last_update > 0.10:
                        parcial = _render_visible(resposta_txt) + "▌"
                        placeholder.markdown(parcial)
                        last_update = time.time()
            else:
                payload = dict(payload_base, **{"stream": True})
                with requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=int(st.session_state.get("timeout_s", 300)),
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
                                delta = j["choices"][0]["delta"].get("content", "")
                                if not delta:
                                    continue
                                resposta_txt += delta
                                if time.time() - last_update > 0.10:
                                    parcial = _render_visible(resposta_txt) + "▌"
                                    placeholder.markdown(parcial)
                                    last_update = time.time()
                            except Exception:
                                continue
                    else:
                        prov_nome = "LM Studio" if prov == "LM Studio" else ("Together" if prov == "Together" else "OpenRouter")
                        st.error(f"Erro {prov_nome}: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro no streaming: {e}")

        visible_txt = _render_visible(resposta_txt).strip()

        # Fallback sem stream
        if not visible_txt:
            try:
                if prov == "Hugging Face":
                    out = InferenceClient(token=auth).chat.completions.create(
                        model=model_to_call,
                        messages=messages_fit,
                        temperature=0.9,
                        max_tokens=int(st.session_state.get("max_tokens_rsp", 900)),
                        stream=False,
                    )
                    resposta_txt = (out.choices[0].message.content or "").strip()
                    visible_txt = _render_visible(resposta_txt).strip()
                else:
                    r2 = requests.post(
                        endpoint,
                        headers=headers,
                        json=dict(payload_base, **{"stream": False}),
                        timeout=int(st.session_state.get("timeout_s", 300)),
                    )
                    if r2.status_code == 400 and "context length" in (r2.text or "").lower():
                        # Encolhe mais e tenta de novo
                        messages_fit2 = compact_messages(messages_fit, max_ctx=ctx_max, reserve_out= int(st.session_state.get("max_tokens_rsp", 900)) + 256)
                        r2 = requests.post(
                            endpoint, headers=headers,
                            json={"model": model_to_call, "messages": messages_fit2, "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)), "temperature": 0.9, "stream": False},
                            timeout=int(st.session_state.get("timeout_s", 300)),
                        )
                    if r2.status_code == 200:
                        try:
                            j = r2.json()
                            if isinstance(j, dict):
                                ch = j.get("choices") or []
                                if ch and isinstance(ch[0], dict):
                                    resposta_txt = (
                                        ch[0].get("message", {}).get("content")
                                        or ch[0].get("text")
                                        or ch[0].get("delta", {}).get("content")
                                        or ""
                                    )
                            else:
                                resposta_txt = ""
                        except Exception:
                            resposta_txt = ""
                        visible_txt = _render_visible(resposta_txt).strip()
                    else:
                        st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
            except Exception as e:
                st.error(f"Fallback (sem stream) erro: {e}")

        # Fallback com prompts limpos
        if not visible_txt:
            try:
                clean_messages = [{"role": "system", "content": prompt}] + historico
                clean_fit = compact_messages(clean_messages, max_ctx=ctx_max, reserve_out=int(st.session_state.get("max_tokens_rsp", 900)) + 256)
                if prov == "Hugging Face":
                    out2 = InferenceClient(token=auth).chat.completions.create(
                        model=model_to_call,
                        messages=clean_fit,
                        temperature=0.9,
                        max_tokens=int(st.session_state.get("max_tokens_rsp", 900)),
                        stream=False,
                    )
                    resposta_txt = (out2.choices[0].message.content or "").strip()
                    visible_txt = _render_visible(resposta_txt).strip()
                else:
                    r3 = requests.post(
                        endpoint,
                        headers=headers,
                        json={"model": model_to_call, "messages": clean_fit, "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)), "temperature": 0.9, "stream": False},
                        timeout=int(st.session_state.get("timeout_s", 300)),
                    )
                    if r3.status_code == 200:
                        try:
                            j = r3.json()
                            if isinstance(j, dict):
                                ch = j.get("choices") or []
                                if ch and isinstance(ch[0], dict):
                                    resposta_txt = (
                                        ch[0].get("message", {}).get("content")
                                        or ch[0].get("text")
                                        or ch[0].get("delta", {}).get("content")
                                        or ""
                                    )
                            else:
                                resposta_txt = ""
                        except Exception:
                            resposta_txt = ""
                        visible_txt = _render_visible(resposta_txt).strip()
                    else:
                        st.error(f"Fallback (prompts limpos) falhou: {r3.status_code} - {r3.text}")
            except Exception as e:
                st.error(f"Fallback (prompts limpos) erro: {e}")

        # BLOQUEIO DE CLÍMAX FINAL
        def _user_allows_climax(msgs: list) -> bool:
            CLIMAX_USER_TRIGGER = re.compile(
                r"(?:\\b("
                r"finaliza(?:r)?|"
                r"pode\\s+(?:gozar|finalizar)|"
                r"liber(?:a|o)\\s+(?:o\\s+)?(?:cl[ií]max|orgasmo)|"
                r"cheg(?:a|ou)\\s+ao?\\s+(?:cl[ií]max|orgasmo)|"
                r"goza(?:r)?\\s+(?:agora|já)|"
                r"agora\\s+goza|"
                r"permite\\s+orgasmo|"
                r"explod(?:e|iu)\\s+em\\s+orgasmo"
                r")\\b)",
                flags=re.IGNORECASE
            )
            last_user = ""
            for r in reversed(msgs or []):
                if str(r.get("role","")).lower() == "user":
                    last_user = r.get("content","") or ""
                    break
            return bool(CLIMAX_USER_TRIGGER.search(last_user))

        def _strip_or_soften_climax(texto: str) -> str:
            ORGASM_TERMS = r"(?:cl[ií]max|orgasmo|orgásm(?:ic)o|gozou|gozando|gozaram|ejacul(?:a|ou|ar)|cheg(?:a|ou)\\s+lá|explod(?:e|iu))"
            ORGASM_SENT = re.compile(rf"([^.!\n]*\\b{ORGASM_TERMS}\\b[^.!?\n]*[.!?])", flags=re.IGNORECASE)
            if not texto:
                return texto
            texto = ORGASM_SENT.sub("", texto)
            texto = re.sub(r"\\n{3,}", "\\n\\n", texto).strip()
            if not texto.endswith((".", "…", "!", "?")):
                texto += "…"
            finais = [
                " A tensão fica no ar, sem conclusão, apenas a respiração quente entre eles.",
                " Eles param no limiar, ainda ofegantes, guardando o resto para o próximo passo.",
                " Um silêncio elétrico preenche o quarto; nenhum desfecho, só a pele e o pulso acelerado.",
            ]
            if all(f not in texto for f in finais):
                texto += random.choice(finais)
            return texto

        if st.session_state.get("app_bloqueio_intimo", True):
            if not _user_allows_climax(st.session_state.session_msgs):
                visible_txt = _strip_or_soften_climax(visible_txt)

        # ENFORCER: garantir ao menos 1 fala de Mary se opção ativa
        if st.session_state.get("usar_falas_mary", False):
            falas = st.session_state.get("_falas_mary_list", []) or []
            if falas and visible_txt:
                tem_fala = any(re.search(re.escape(f), visible_txt, flags=re.IGNORECASE) for f in falas)
                if not tem_fala:
                    escolha = random.choice(falas)
                    if st.session_state.get("interpretar_apenas_mary", False):
                        inj = f"— {escolha}\\n\\n"
                    else:
                        inj = f"— {escolha} — diz Mary.\\n\\n"
                    visible_txt = inj + visible_txt

        # Render final
        placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

        # Persistência
        if visible_txt and visible_txt != "[Sem conteúdo]":
            salvar_interacao("assistant", visible_txt)
            st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt})
        else:
            salvar_interacao("assistant", "[Sem conteúdo]")
            st.session_state.session_msgs.append({"role": "assistant", "content": "[Sem conteúdo]"})

        # Reforço pós-resposta
        try:
            usados = []
            topk_usadas = memoria_longa_buscar_topk(
                query_text=visible_txt,
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
            )
            for t, _sc, _sim, _rr in topk_usadas:
                usados.append(t)
            memoria_longa_reforcar(usados)
        except Exception:
            pass

