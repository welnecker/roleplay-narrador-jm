# ============================================================
# Narrador JM — Variante “Somente FASE do romance”
# ============================================================

import os
import re
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st
import requests
import gspread
from gspread.exceptions import APIError
from oauth2client.service_account import ServiceAccountCredentials
from huggingface_hub import InferenceClient

# =========================
# CONFIG BÁSICA DO APP
# =========================

ONLY_FASE_MODE = True

PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
TAB_INTERACOES = "interacoes_jm"

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
    "zai-org: LM-4.5-Air (HF)":   "zai-org/GLM-4.5-Air",
    "Mixtral 8x7B Instruct (HF)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "DeepSeek R1 (HF)":           "deepseek-ai/DeepSeek-R1",
}

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
    else:
        return (
            "https://api.together.xyz/v1/chat/completions",
            st.secrets.get("TOGETHER_API_KEY", ""),
            MODELOS_TOGETHER_UI,
        )

def model_id_for_together(api_ui_model_id: str) -> str:
    key = (api_ui_model_id or "").strip()
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in key:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    low = key.lower()
    if low.startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-V0.1"
    return key or "mistralai/Mixtral-8x7B-Instruct-v0.1"

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
            return ws
        except Exception:
            return None

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

# =========================
# HISTÓRICO DE TURNOS
# =========================

def carregar_interacoes(n: int = 20):
    """Carrega as interações recentes de TAB_INTERACOES."""
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
    """Salva uma interação (turno)."""
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

# ---------------------------------------------
# VIRGINDADE — leitura memoria_jm + fallback (interacoes_jm) e inferência temporal
# ---------------------------------------------
from typing import Optional, Tuple, List, Dict
import re
from datetime import datetime

# Padrões (priorize "não virgem" sobre "virgem")
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
    """Converte timestamp em datetime."""
    ts = (ts or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None

def _ultimo_evento_virgindade_memoria(ate: Optional[datetime] = None) -> Optional[Tuple[bool, datetime, str]]:
    """
    Busca evidência em memoria_jm (tags [mary] e [all]), priorizando perda de virgindade.
    Retorna (estado_bool, timestamp, 'memoria_jm')
    """
    try:
        buckets = carregar_memorias_brutas()  # {'[tag]': [{'conteudo','timestamp'}, ...]}
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
    # Mais recente, False (não virgem) vence empate
    candidatos.sort(key=lambda x: (x[1], 0 if x[0] is False else 1))
    estado, ts = candidatos[-1]
    return (estado, ts, "memoria_jm")

def _ultimo_evento_virgindade_interacoes(ate: Optional[datetime] = None) -> Optional[Tuple[bool, datetime, str]]:
    """
    Fallback: busca padrão nas últimas interações (interacoes_jm).
    Retorna (estado_bool, timestamp, 'interacoes_jm')
    """
    ate = ate or datetime.now()
    inter = carregar_interacoes(n=20)
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
    """
    Determina status canônico de virgindade para uso no prompt.
    True  -> virgem
    False -> não virgem
    None  -> desconhecido (sem evidências)
    """
    ate = ate or datetime.now()
    res = _ultimo_evento_virgindade_memoria(ate)
    if res is None:
        res = _ultimo_evento_virgindade_interacoes(ate)
    return None if res is None else res[0]

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
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE)
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
    "Vem mais perto, sem pressa.",
    "Assim está bom… continua desse jeito.",
    "Eu quero sentir você devagar.",
    "Fica comigo, só mais um pouco.",
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
import re, random

import re, random

def gerar_mary_sensorial(level: int = 2, n: int = 2, hair_on: bool = True, sintonia: bool = True) -> str:
    if level <= 0 or n <= 0:
        return ""

    # Frases SEM cenário/clima/metáforas
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
        "O perfume na clavícula convida a aproximação.",
        "Os lábios entreabertos, esperando o momento certo.",
        "O olhar pousa e permanece, pedindo gentileza.",
    ]

    if level == 1:
        pool = list(base_leve)
    elif level == 2:
        pool = list(base_leve) + list(base_marcado)
    else:
        pool = list(base_leve) + list(base_marcado) + list(base_ousado)

    # Filtro extra contra clima/paisagem
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

def _deduzir_ancora(texto: str) -> dict:
    t = (texto or "").lower()
    if "motel" in t or "suíte" in t or "suite" in t:
        return {"local": "Motel — Suíte", "hora": "noite"}
    if "quarto" in t:
        return {"local": "Quarto", "hora": "noite"}
    if "praia" in t:
        return {"local": "Praia", "hora": "fim de tarde"}
    if "bar" in t or "pub" in t:
        return {"local": "Bar", "hora": "noite"}
    if "biblioteca" in t:
        return {"local": "Biblioteca", "hora": "tarde"}
    return {}


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


# =========================
# VIRGINDADE (opcional)
# =========================

def detectar_virgindade_mary(memos: dict, ate_ts: str) -> bool:
    for d in memos.get("[mary]", []):
        c = (d.get("conteudo") or "").lower()
        ts = (d.get("timestamp") or "")
        if ts and ate_ts and ts > ate_ts:
            continue
        if "sou virgem" in c or "virgindade" in c or "nunca fiz" in c:
            return True
    return False

def montar_bloco_virgindade(ativar: bool) -> str:
    if not ativar:
        return ""
    return (
        "### Nota de virgindade (prioridade)\n"
        "- Mary valoriza sua primeira vez. O ritmo é cuidadoso, com comunicação clara.\n"
        "- Evite pressa; foco em respeito, confiança e conforto.\n"
    )

def prompt_da_cena(ctx: dict | None = None, modo_finalizacao: str = "ponte") -> str:
    ctx = ctx or {}
    tempo    = (ctx.get("tempo") or "").strip().rstrip(".")
    lugar    = (ctx.get("lugar") or "").strip().rstrip(".")
    figurino = (ctx.get("figurino") or "").strip().rstrip(".")

    # Monta a 1ª linha (permitida pela exceção)
    pedacos = []
    if tempo: pedacos.append(tempo.capitalize())
    pedacos.append(f"Mary{(', ' + figurino) if figurino else ''}".strip())
    if lugar: pedacos.append(lugar)
    primeira_linha = ". ".join([p for p in pedacos if p]) + "."

    # Regras de fechamento
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

# =========================
# PROMPT BUILDER (APENAS FASE) — compatível com Modo Mary
# =========================

def construir_prompt_com_narrador() -> str:
    BLOCO_ROLEPLAY = """
OBRIGATÓRIO — FORMATO ESTRUTURADO DE ROLEPLAY

- Cada fala ("—") deve começar linha nova isolada, SEMPRE seguida ou precedida de bloco de ação/descrição corporal.
- Nunca una mais de 2 frases no mesmo parágrafo narrativo; em narração, troque de linha a cada ação/reação física importante.
- O texto final sempre terá parágrafos curtos: bloco de ação (máx. 2 frases), bloco de fala, bloco de reação, bloco de fala. Nunca prosa longa.
- NÃO formate como prosa de romance/livro; sempre como roteiro estruturado de roleplay moderno e comercial.
""".strip()

    BLOCO_RESTRICAO_SENSORY = """
Conduza como romance contemporâneo: descreva ações, sensações, diálogos de forma rica, emotiva e natural, incluindo cenários, introspecções leves e clima emocional.
Diálogos podem aparecer intercalados com narração, integrados em parágrafos maiores quando pertinente.
Jamais use início seco tipo 'Parque da cidade.' nem fechos como 'um silêncio elétrico preenche o quarto'.
"""


...
""".strip()

    ctx = st.session_state.get("ctx_cena", {})
    try:
        voz_bloco = instrucao_llm(st.session_state.get("finalizacao_modo", "ponto de gancho"), ctx)
    except Exception:
        voz_bloco = prompt_da_cena(ctx, st.session_state.get("finalizacao_modo", "ponte"))
    cena_bloco = prompt_da_cena(ctx, st.session_state.get("finalizacao_modo", "ponte"))

    # Sanitizador leve de histórico (sem praia/clima)
    import re
    _split = re.compile(r'(?<=[\.\!\?])\s+')
    _amb = re.compile(
        r'\b(c[ée]u|nuvens?|horizonte|luar|mar|onda?s?|pier|praia|vento|brisa|chuva|garoa|sereno|amanhecer|entardecer|p[ôo]r do sol|paisage?m|cen[áa]rio|temperatura|verão|quiosques?)\b',
        re.I
    )

    def _hist_sanitizado(hist):
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
        if _sens_on
        else ""
    )

    ritmo_cena = int(st.session_state.get("ritmo_cena", 0))
    ritmo_label = ["muito lento", "lento", "médio", "rápido"][max(0, min(3, ritmo_cena))]
    modo_sintonia = bool(st.session_state.get("modo_sintonia", True))

    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = _hist_sanitizado(hist)
    ultima_fala_user = _last_user_text(hist)

    ancora_bloco = ""
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
        _ref_dt = _to_dt(ate_ts) if ("_to_dt" in globals() or "_to_dt" in dir()) else None
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
    if bool(st.session_state.get("app_bloqueio_intimo", True)) and fase < 5:
        climax_bloco = (
            "### Proteção de avanço íntimo (ATIVA)\n"
            "- **Sem clímax por padrão**: não descreva orgasmo/finalização **a menos que o usuário tenha liberado explicitamente na mensagem anterior**.\n"
            "- Encerre em **pausa sensorial** (respiração, silêncio, carinho), **sem** 'fade-to-black'.\n"
        )

    if modo_mary:
        papel_header = "Você é **Mary**, responda **em primeira pessoa**, sem narrador externo. Use apenas o que Mary vê/sente/ouve. Não descreva pensamentos de Jânio. Não use títulos nem repita instruções."
        regra_saida = "- Narre **em primeira pessoa (eu)** como Mary; nunca use narrador onisciente.\n- Produza uma cena fechada e natural, sem comentários externos."
        formato_cena = (
            "- DIÁLOGOS diretos com travessão (—), intercalados com ação/reação **em 1ª pessoa (Mary)**."
        )
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
{voz_bloco}
{cena_bloco}
{papel_header}
{ancora_bloco}{sintonia_bloco}{virg_bloco}{climax_bloco}{falas_mary_bloco}
### Dossiê (personas)
{dossie_txt}
### Diretrizes gerais (ALL)
{chr(10).join(f"- {c}" for c in memos_all) if memos_all else "(vazio)"}
### Perfil (resumo mais recente)
{perfil or "(vazio)"}
### Histórico recente (sanitizado)
{hist_txt}
### BLOCO DE ESTILO E TOM — ROLEPLAY SENSORIAL

- Não descreva cenário, natureza, ambiente, temperatura, clima, praia, quiosques, verão, paisagem ou localização em nenhum momento. PROIBIDO iniciar cena com tempo, lugar ou natureza.

- Cada turno da resposta deve obrigatoriamente incluir sensação corporal e visual de Mary: curvas naturais, seios, quadris, boca, coxas, gestos sedutores, calor físico, tensão, suor, desejo, pele, reações físicas e troca de olhares. Escreva sempre o efeito do olhar e dos gestos do par sobre o corpo de Mary.

- Descreva as reações físicas intensas de Mary: peito arfando, seios balançando, quadris se movendo, músculos tensionando, pele arrepiando, corpo se inclinando, respiração acelerada, boca entreaberta, lábios úmidos, mãos deslizando pela roupa ou corpo, gemidos, suspiros, movimentos provocantes. Jamais apenas "pensamento" ou "emoção", sempre efeito físico visível.

- Use diálogos alternados e ações entre os personagens. Comece e termine com ganchos físicos/visuais: gesto, aproximação, olhar, toque, convite, microação. Jamais feche uma cena com frases clichê como "A tensão fica no ar…", "Um silêncio elétrico…", "Eles param no limiar…" ou qualquer variação parecida.

- Parágrafos devem ser curtos (até 3 frases cada), com linhas em branco entre cada bloco de frase/fala. Nunca agrupe textos longos nem explique o contexto; sempre priorize o momento presente, corpo-a-corpo, sem retrospectiva, sem ambientação.

- Nunca descreva tempo ("Domingo de manhã"), nunca local ("Praia de Jacaraípe"), nunca ambiente ("areias quentes", "vento", "céu azul", "quiosques"). Proibido metáfora, poesia ambiental, ou qualquer menção implícita de natureza/cenário.

- Quando Mary se move, sempre explicitamente cite curvas, seios, quadris, boca, cabelo, toque, calor e reação física. Se o usuário der uma ordem ou direção, Mary sempre reage com gesto físico e fala conectando o desejo explícito do momento.

- Nunca faça pausas narrativas, não explique ou analise. Apenas descreva ação física, reação corporal e diálogo rápido.

- Se usar falas fixas da Mary, sempre intercale com ação corporal, nunca use só fala.

- Se não puder avançar por limitação de fase, encerre com microação física de tensão, mas nunca diga "a tensão fica no ar".

- Proibido monólogo introspectivo ou pensamento maior que uma frase. Priorize sempre calor, desejo, corpo, graça e sensualidade natural do encontro.

- Escreva apenas o momento imediato; nunca use frases de abertura como "Naquela manhã…", "No fim de tarde…", "No bar…", "Ao entardecer…", "Na praia…", "Enquanto o vento soprava…", "Ela caminhava pelas areias…".

- Este é um roleplay comercial e engajante, não uma novela nem conto literário. O texto deve ser sempre vivo, sensual, direto, curto e visual.

### FIM DO BLOCO DE ESTILO SENSORIAL
### Camada sensorial — Mary (OBRIGATÓRIA no 1º parágrafo)
{mary_sens_txt or "- Apenas sensações físicas, nunca ambiente."}
### Memória longa — Top-K relevantes
{ml_topk_txt}
### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Siga **somente** as regras da fase (permitidos/proibidos) abaixo:
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}
### Geografia & Montagem
- Não force coincidências. Sem teletransporte.
### Formato OBRIGATÓRIO da cena
{formato_cena}
### Regra de saída
{regra_saida}
""".strip()

prompt = inserir_regras_mary_e_janio(prompt)
return prompt


import re

# --- Remoção de termos de paisagem/clima ---
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
    """Remove termos de natureza/clima definidos em SCENERY_WORD do texto."""
    if not text:
        return ""
    return SCENERY_WORD.sub("", text)

def sanitize_scenery_preserve_opening(t: str) -> str:
    """Remove natureza/clima, mas preserva a primeira linha intacta."""
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
    return primeira_linha

def _render_visible(t: str) -> str:
    t = sanitize_scenery_preserve_opening(t)  # Preserva linha de abertura limpa
    t = roleplay_paragraphizer(t)             # Organiza parágrafos e falas
    if st.session_state.get("app_bloqueio_intimo", True):
        t = sanitize_explicit(t, int(st.session_state.get("nsfw_max_level", 0)), action="soften")
    return t

def force_linebreak_on_falas(txt):
    """Força quebra em toda fala iniciada por travessão em texto corrido."""
    return re.sub(r"([^\n])\s*(—)", r"\1\n\n\2", txt)

# --- Filtro de termos explícitos de sexo ---
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
    """Corta ou suaviza respostas que excedem o nível permitido de NSFW."""
    lvl = classify_nsfw_level(t)
    if lvl <= max_level:
        return t
    # Adapte este bloco se quiser cortar ou substituir além do recomendado
    return t  # Não aplica corte automático por padrão

def redact_for_logs(t: str) -> str:
    if not t:
        return ""
    t = re.sub(EXPL_PAT, "[…]", t, flags=re.I)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    """Verifica se há conteúdo suficiente para ser considerado uma resposta válida."""
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

# Exemplo de uso após gerar resposta:
# visible_txt = force_linebreak_on_falas(_render_visible(resposta_txt).strip())

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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
# =========================
# SIDEBAR — Reorganizado (apenas FASE)
# =========================

with st.sidebar:
    st.title("🧭 Painel do Roteirista")
    
    # Provedor / modelos
    provedor = st.radio(
        "🌐 Provedor",
        ["OpenRouter", "Together", "Hugging Face"],
        index=["OpenRouter", "Together", "Hugging Face"].index(st.session_state.get("provedor_ia", "OpenRouter")),
        key="provedor_ia"
    )
    api_url, api_key, modelos_map = api_config_for_provider(provedor)
    if not api_key:
        st.warning("⚠️ API key ausente para o provedor selecionado. Defina em st.secrets.")

    modelo_nome = st.selectbox(
        "🤖 Modelo de IA",
        list(modelos_map.keys()),
        index=list(modelos_map.keys()).index(next((k for k, v in modelos_map.items() if v == st.session_state.get("modelo_escolhido_id")), list(modelos_map.keys())[0])),
        key="modelo_nome_ui"
    )
    st.session_state.modelo_escolhido_id = modelos_map[modelo_nome]

    st.markdown("---")
    st.markdown("### ✍️ Estilo & Progresso Dramático")

    # Modo de resposta (NARRADOR ou MARY 1ª pessoa)
    modo_op = st.selectbox(
        "Modo de resposta",
        ["Narrador padrão", "Mary (1ª pessoa)"],
        index=["Narrador padrão", "Mary (1ª pessoa)"].index(st.session_state.get("modo_resposta", "Narrador padrão")),
        key="modo_resposta"
    )
    st.session_state.interpretar_apenas_mary = (modo_op == "Mary (1ª pessoa)")

    st.selectbox(
        "Estilo de escrita",
        ["AÇÃO", "ROMANCE LENTO", "NOIR"],
        index=["AÇÃO", "ROMANCE LENTO", "NOIR"].index(st.session_state.get("estilo_escrita", "AÇÃO")),
        key="estilo_escrita"
    )

    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=int(st.session_state.get("nsfw_max_level", 0)), key="nsfw_max_level")

    st.checkbox(
        "Sintonia com o parceiro (modo harmônico)",
        key="modo_sintonia",
        value=st.session_state.get("modo_sintonia", True)
    )

    st.select_slider(
        "Ritmo da cena",
        options=[0, 1, 2, 3],
        value=int(st.session_state.get("ritmo_cena", 0)),
        format_func=lambda n: ["muito lento", "lento", "médio", "rápido"][n],
        key="ritmo_cena"
    )

    st.selectbox(
        "Finalização",
        ["ponto de gancho", "fecho suave", "deixar no suspense"],
        index=["ponto de gancho", "fecho suave", "deixar no suspense"].index(st.session_state.get("finalizacao_modo", "ponto de gancho")),
        key="finalizacao_modo"
    )

    st.checkbox(
        "Usar falas da Mary da planilha (usar literalmente)",
        value=st.session_state.get("usar_falas_mary", False),
        key="usar_falas_mary"
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
        key="ui_mj_fase"
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
        key="no_coincidencias"
    )
    st.checkbox(
        "Bloquear avanços íntimos sem ordem",
        value=st.session_state.get("app_bloqueio_intimo", True),
        key="app_bloqueio_intimo"
    )
    st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.get("app_emocao_oculta", "nenhuma")),
        key="ui_app_emocao_oculta"
    )
    st.session_state.app_emocao_oculta = st.session_state.get("ui_app_emocao_oculta", "nenhuma")

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp",1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")
    st.checkbox("Usar memória longa no prompt", value=st.session_state.get("use_memoria_longa", True), key="use_memoria_longa")
    st.slider("Top-K memórias", 1, 5, int(st.session_state.get("k_memoria_longa", 3)), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95, float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")

    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")

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
            provedor_local = st.session_state.get("provedor_ia", provedor)
            api_url_local = api_url
            api_key_local = api_key
            model_id_call = (
                model_id_for_together(st.session_state.modelo_escolhido_id)
                if provedor_local == "Together"
                else st.session_state.modelo_escolhido_id
            )
            if not api_key_local:
                st.error("⚠️ API key ausente para o provedor selecionado (defina em st.secrets).")
            else:
                if provedor_local == "Hugging Face":
                    # --- HF sem requests: usa InferenceClient ---
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
                    # OpenRouter / Together (requests)
                    r = requests.post(
                        api_url_local,
                        headers={"Authorization": f"Bearer {api_key_local}", "Content-Type": "application/json"},
                        json={
                            "model": model_id_call,
                            "messages": [{"role": "user", "content": prompt_resumo}],
                            "max_tokens": 800,
                            "temperature": 0.85
                        },
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

# =========================
# ENVIO DO USUÁRIO + STREAMING
# =========================

entrada = st.chat_input("Digite sua direção de cena...")

if entrada:
    salvar_interacao("user", str(entrada))
    st.session_state.session_msgs.append({"role": "user", "content": str(entrada)})
    st.session_state["ctx_cena"] = extrair_diretriz_contexto(entrada, st.session_state.get("ctx_cena", CTX_INICIAL))
    ctx = st.session_state["ctx_cena"]

    linha_abertura = gerar_linha_abertura(ctx)

    mary_mode_active = bool(
        st.session_state.get("interpretar_apenas_mary")
        or st.session_state.get("modo_resposta") == "Mary (1ª pessoa)"
    )

    historico = []
    for ix, m in enumerate(st.session_state.session_msgs):
        role = m.get("role", "user")
        content = m.get("content", "")
        # Só a ÚLTIMA mensagem do usuário recebe o contexto de abertura (mas prefixa só depois do JÂNIO:)
        if ix == len(st.session_state.session_msgs) - 1 and role.lower() == "user":
            content = linha_abertura
        if mary_mode_active and role.lower() == "user":
            content = f"JÂNIO: {content}"
        historico.append({"role": role, "content": content})

    prompt = construir_prompt_com_narrador()

    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets.get("TOGETHER_API_KEY", "")
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    elif prov == "Hugging Face":
        endpoint = "HF_CLIENT"
        auth = st.secrets.get("HUGGINGFACE_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets.get("OPENROUTER_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id

    if not auth:
        st.error("A chave de API do provedor selecionado não foi definida em st.secrets.")
        st.stop()

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

    payload = {
        "model": model_to_call,
        "messages": messages,
        "max_tokens": int(st.session_state.get("max_tokens_rsp", 1200)),
        "temperature": 0.9,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    # ============ BLOQUEIO DE CLÍMAX ============
    CLIMAX_USER_TRIGGER = re.compile(
        r"(?:\b("
        r"finaliza(?:r)?|"
        r"pode\s+(?:gozar|finalizar)|"
        r"liber(?:a|o)\s+(?:o\s+)?(?:cl[ií]max|orgasmo)|"
        r"cheg(?:a|ou)\s+ao?\s+(?:cl[ií]max|orgasmo)|"
        r"goza(?:r)?\s+(?:agora|já)|"
        r"agora\s+goza|"
        r"permite\s+orgasmo|"
        r"explod(?:e|iu)\s+em\s+orgasmo"
        r")\b)",
        flags=re.IGNORECASE
    )

    ORGASM_TERMS = r"(?:cl[ií]max|orgasmo|orgásm(?:ic)o|gozou|gozando|gozaram|ejacul(?:a|ou|ar)|cheg(?:a|ou)\s+lá|explod(?:e|iu))"
    ORGASM_SENT = re.compile(rf"([^.!\n]*\b{ORGASM_TERMS}\b[^.!?\n]*[.!?])", flags=re.IGNORECASE)
    def _user_allows_climax(msgs: list) -> bool:
        last_user = ""
        for r in reversed(msgs or []):
            if str(r.get("role", "")).lower() == "user":
                last_user = r.get("content", "") or ""
                break
        return bool(CLIMAX_USER_TRIGGER.search(last_user))
    def _strip_or_soften_climax(texto: str) -> str:
        if not texto:
            return texto
        texto = ORGASM_SENT.sub("", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
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

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        # STREAMING: HF ou requests
        if prov == "Hugging Face":
            from huggingface_hub import InferenceClient
            hf_client = InferenceClient(token=auth, timeout=int(st.session_state.get("timeout_s", 300)))
            for chunk in hf_client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.9,
                max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                stream=True,
            ):
                delta = getattr(chunk.choices.delta, "content", None)
                if not delta:
                    continue
                resposta_txt += delta
                if time.time() - last_update > 0.10:
                    parcial = _render_visible(resposta_txt) + "▌"
                    if st.session_state.get("app_bloqueio_intimo", True):
                        if not _user_allows_climax(st.session_state.session_msgs):
                            parcial = _strip_or_soften_climax(parcial)
                    placeholder.markdown(parcial)
                    last_update = time.time()
        else:
            with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=int(st.session_state.get("timeout_s", 300))) as r:
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
                            delta = j["choices"]["delta"].get("content", "")
                            if not delta:
                                continue
                            resposta_txt += delta
                            if time.time() - last_update > 0.10:
                                parcial = _render_visible(resposta_txt) + "▌"
                                if st.session_state.get("app_bloqueio_intimo", True):
                                    if not _user_allows_climax(st.session_state.session_msgs):
                                        parcial = _strip_or_soften_climax(parcial)
                                placeholder.markdown(parcial)
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")

        # FINALIZA TEXTO VISÍVEL completo
        visible_txt = _render_visible(resposta_txt).strip()
        if st.session_state.get("app_bloqueio_intimo", True):
            if not _user_allows_climax(st.session_state.session_msgs):
                visible_txt = _strip_or_soften_climax(visible_txt)

        # Renderização final
        placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

        # Persistência sempre no histórico
        if visible_txt and visible_txt != "[Sem conteúdo]":
            salvar_interacao("assistant", visible_txt)
            st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt})
        else:
            salvar_interacao("assistant", "[Sem conteúdo]")
            st.session_state.session_msgs.append({"role": "assistant", "content": "[Sem conteúdo]"})

        # Reforço de memória longa
        try:
            usados = []
            topk_usadas = memoria_longa_buscar_topk(
                query_text=visible_txt,
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
            )
            for t, *_ in topk_usadas:
                usados.append(t)
            memoria_longa_reforcar(usados)
        except Exception:
            pass

       # ======================
    # STREAM (PATCH C) — com suporte a HF
    # ======================
    try:
        if prov == "Hugging Face":
            # --- STREAM via InferenceClient (sem SSE) ---
            hf_client = InferenceClient(
                token=auth,
                timeout=int(st.session_state.get("timeout_s", 300))
            )
            for chunk in hf_client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.9,
                max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                stream=True,
            ):
                delta = getattr(chunk.choices[0].delta, "content", None)
                if not delta:
                    continue
                resposta_txt += delta
    
                # Atualização parcial
                if time.time() - last_update > 0.10:
                    parcial = _render_visible(resposta_txt) + "▌"
    
                    # BLOQUEIO ON-THE-FLY (sempre ativo se a opção estiver ligada)
                    if st.session_state.get("app_bloqueio_intimo", True):
                        if not _user_allows_climax(st.session_state.session_msgs):
                            parcial = _strip_or_soften_climax(parcial)
    
                    placeholder.markdown(parcial)
                    last_update = time.time()
    
        else:
            # --- STREAM via SSE (OpenRouter/Together) — mantém seu fluxo atual via requests ---
            headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}
            payload = {
                "model": model_to_call,
                "messages": messages,
                "max_tokens": int(st.session_state.get("max_tokens_rsp", 1200)),
                "temperature": 0.9,
                "stream": True,
            }
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
                            delta = j["choices"][0]["delta"].get("content", "")
                            if not delta:
                                continue
                            resposta_txt += delta
    
                            # Atualização parcial
                            if time.time() - last_update > 0.10:
                                parcial = _render_visible(resposta_txt) + "▌"
    
                                # BLOQUEIO ON-THE-FLY (sempre ativo se a opção estiver ligada)
                                if st.session_state.get("app_bloqueio_intimo", True):
                                    if not _user_allows_climax(st.session_state.session_msgs):
                                        parcial = _strip_or_soften_climax(parcial)
    
                                placeholder.markdown(parcial)
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Erro no streaming: {e}")

    # FINALIZA TEXTO VISÍVEL
    visible_txt = _render_visible(resposta_txt).strip()

    # Fallback sem stream
    if not visible_txt:
        try:
            if prov == "Hugging Face":
                out = InferenceClient(token=auth).chat.completions.create(
                    model=model_to_call,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                    stream=False,
                )
                resposta_txt = (out.choices[0].message.content or "").strip()
                visible_txt = _render_visible(resposta_txt).strip()
            else:
                r2 = requests.post(
                    endpoint, headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"},
                    json={**payload, "stream": False},
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
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

        # Fallback com prompts limpos
    if not visible_txt:
        try:
            if prov == "Hugging Face":
                out2 = InferenceClient(token=auth).chat.completions.create(
                    model=model_to_call,
                    messages=[{"role": "system", "content": prompt}] + historico,
                    temperature=0.9,
                    max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                    stream=False,
                )
                resposta_txt = (out2.choices[0].message.content or "").strip()
                visible_txt = _render_visible(resposta_txt).strip()
            else:
                r3 = requests.post(
                    endpoint, headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"},
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

    # BLOQUEIO DE CLÍMAX FINAL (sempre que a opção estiver ativa, só libera com comando do usuário)
    if st.session_state.get("app_bloqueio_intimo", True):
        if not _user_allows_climax(st.session_state.session_msgs):
            visible_txt = _strip_or_soften_climax(visible_txt)

        # --- ENFORCER: garantir ao menos 1 fala de Mary, se a opção estiver ativa ---
    if st.session_state.get("usar_falas_mary", False):
        falas = st.session_state.get("_falas_mary_list", []) or []
        if falas and visible_txt:
            tem_fala = any(re.search(re.escape(f), visible_txt, flags=re.IGNORECASE) for f in falas)
            if not tem_fala:
                escolha = random.choice(falas)
                if st.session_state.get("interpretar_apenas_mary", False):
                    inj = f"— {escolha}\n\n"
                else:
                    inj = f"— {escolha} — diz Mary.\n\n"
                visible_txt = inj + visible_txt
    
    # ===== Render final (sempre) =====
    placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")
    
    # ===== Persistência (sempre) =====
    if visible_txt and visible_txt != "[Sem conteúdo]":
        salvar_interacao("assistant", visible_txt)
        st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt})
    else:
        salvar_interacao("assistant", "[Sem conteúdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": "[Sem conteúdo]"})
    
    # ===== Reforço pós-resposta (sempre) =====
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









