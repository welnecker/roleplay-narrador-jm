# ============================================================
# Narrador JM — Variante “Somente FASE do romance”
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
import numpy as np
from gspread.exceptions import APIError
from oauth2client.service_account import ServiceAccountCredentials

# =========================
# CONFIG BÁSICA DO APP
# =========================

# ATENÇÃO: este modo ignora “Momento atual”. Só a FASE manda.
ONLY_FASE_MODE = True

PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
TAB_INTERACOES = "interacoes_jm"
TAB_PERFIL      = "perfil_jm"
TAB_MEMORIAS    = "memoria_jm"
TAB_ML          = "memoria_longa_jm"
TAB_TEMPLATES   = "templates_jm"
TAB_FALAS_MARY  = "falas_mary_jm"   # opcional (coluna: fala)

# Modelos (pode expandir depois)
MODELOS_OPENROUTER = {
    "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
    "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
    "🧠 DeepSeek R1T2 Chimera ★★★★ (free)": "tngtech/deepseek-r1t2-chimera:free",
    "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
    "⚡ Google Gemini 2.5 Flash Lite": "google/gemini-2.5-flash",
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

def model_id_for_together(api_ui_model_id: str) -> str:
    key = (api_ui_model_id or "").strip()
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in key:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    low = key.lower()
    if low.startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-V0.1"
    return key or "mistralai/Mixtral-8x7B-Instruct-v0.1"

# =========================
# BACKOFF + CACHES
# =========================

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
    titulo = "Jânio" if nome in ("janio","jânio") else "Mary" if nome=="mary" else nome.capitalize()
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

def carregar_interacoes(n: int = 20):
    cache = st.session_state.get("_cache_interacoes", None)
    if cache is None:
        regs = _sheet_all_records_cached(TAB_INTERACOES)
        # normaliza
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
    # Brandas por padrão — troque depois se quiser
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
        "A pele arrepia sutil ao menor toque de vento.",
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
    frases = random.sample(pool, k=n_eff)

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

# =========================
# PROMPT BUILDER (APENAS FASE)
# =========================

def construir_prompt_com_narrador() -> str:
    memos = carregar_memorias_brutas()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])

    # Momento desligado
    momento_atual = 0
    mdata = {"nome": "(desligado)", "objetivo": "", "permitidos": "", "proibidos": "", "proximo": 0}
    proximo_nome = ""
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")

    # Sensorial Mary
    _sens_on = bool(st.session_state.get("mary_sensorial_on", True))
    _sens_level = int(st.session_state.get("mary_sensorial_level", 2))
    _sens_n = int(st.session_state.get("mary_sensorial_n", 2))
    mary_sens_txt = gerar_mary_sensorial(
        _sens_level, n=_sens_n, sintonia=bool(st.session_state.get("modo_sintonia", True))
    ) if _sens_on else ""

    # Ritmo & sintonia
    ritmo_cena = int(st.session_state.get("ritmo_cena", 0))
    ritmo_label = ["muito lento", "lento", "médio", "rápido"][max(0, min(3, ritmo_cena))]
    modo_sintonia = bool(st.session_state.get("modo_sintonia", True))

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
            "- Primeira frase deve ancorar lugar e hora neste formato: `Local — Hora — ...`.\n"
        )

    # Corte temporal
    ate_ts = _parse_ts(hist[-1].get("timestamp","")) if hist else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Memória longa Top-K
    ml_topk_txt = "(nenhuma)"
    st.session_state["_ml_topk_texts"] = []
    if st.session_state.get("use_memoria_longa", True) and hist:
        try:
            topk = memoria_longa_buscar_topk(
                query_text=hist[-1]["content"], k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)), ate_ts=ate_ts,
            )
            if topk:
                ml_topk_txt = "\n".join([f"- {t}" for (t, _sc, _sim, _rr) in topk])
                st.session_state["_ml_topk_texts"] = [t for (t, *_rest) in topk]
        except Exception:
            st.session_state["_ml_topk_texts"] = []

    # Diretrizes [all] até o corte
    memos_all = [
        (d.get("conteudo") or "").strip()
        for d in memos.get("[all]", [])
        if isinstance(d, dict) and d.get("conteudo")
        and (not d.get("timestamp") or d.get("timestamp") <= ate_ts)
    ]
    st.session_state["_ml_recorrentes"] = memos_all

    # Dossiê temporal
    dossie = []
    mary = persona_block_temporal("mary", memos, ate_ts, 8)
    janio = persona_block_temporal("janio", memos, ate_ts, 8)
    if mary: dossie.append(mary)
    if janio: dossie.append(janio)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    # Falas Mary
    falas_mary_bloco = ""
    if st.session_state.get("usar_falas_mary", False):
        falas = carregar_falas_mary()
        if not falas:
            falas = FALAS_EXPLICITAS_MARY
        if falas:
            falas_mary_bloco = "### Falas de Mary (usar literalmente)\n" + "\n".join(f"- {s}" for s in falas)

    # Sintonia
    sintonia_bloco = ""
    if modo_sintonia:
        sintonia_bloco = (
            "### Sintonia & Ritmo (prioritário)\n"
            f"- Ritmo da cena: **{ritmo_label}**.\n"
            "- Condução harmônica: Mary sintoniza com o parceiro; evite ordens ríspidas/imperativas. Prefira convites, pedidos gentis.\n"
            "- Pausas e respiração contam; o desejo é mostrado pela troca, não por imposição.\n"
        )

    # Virgindade
    virg_bloco = montar_bloco_virgindade(ativar=detectar_virgindade_mary(memos, ate_ts))

    flag_parallel = bool(st.session_state.get("no_coincidencias", True))

    # MONTAGEM DO PROMPT
    prompt = f"""
Você é o Narrador de um roleplay dramático brasileiro, foque em Mary e Jânio. Não repita instruções nem títulos.

{ancora_bloco}{sintonia_bloco}{virg_bloco}{falas_mary_bloco}
### Dossiê (personas)
{dossie_txt}

### Diretrizes gerais (ALL)
{chr(10).join(f"- {c}" for c in memos_all) if memos_all else "(vazio)"}

### Perfil (resumo mais recente)
{perfil or "(vazio)"}

### Histórico recente (planilha)
{hist_txt}

### Estilo
- Use o estilo **{estilo}**:
{("- Frases curtas, cortes rápidos, foco em gesto/ritmo.") if estilo=="AÇÃO" else
("- Atmosfera sombria, subtexto, silêncio que pesa.") if estilo=="NOIR" else
("- Ritmo lento, tensão emocional, detalhes sensoriais (sem grafismo).")}
- Todas as cenas são sensoriais e físicas (toques, temperatura, respiração), sem vulgaridade.

### Camada sensorial — Mary (OBRIGATÓRIA no 1º parágrafo)
{mary_sens_txt or "- Comece com 1–2 frases sobre caminhar/olhar/perfume/cabelos (negros, volumosos, levemente ondulados)."}
- Aplique essa camada ANTES do primeiro diálogo.
- Frases curtas, diretas, físicas; evite metáforas rebuscadas.

### Memória longa — Top-K relevantes
{ml_topk_txt}

### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Siga **somente** as regras da fase (permitidos/proibidos) abaixo:
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}

### Geografia & Montagem
- Não force coincidências: sem ponte explícita, mantenha locais distintos e use montagem paralela (A/B) conforme {flag_parallel}.
- Comece cada bloco com **lugar e hora** (“Local — Hora — …”) na primeira frase.
- Se houver ponte diegética, convergir para co-presença no final é permitido (sem teletransporte).

### Formato OBRIGATÓRIO da cena
- Inclua DIÁLOGOS diretos com travessão (—), intercalados com ação/reação (mínimo 4 falas quando ambos estiverem em cena).
- Evite pensamentos internos longos; priorize gestos, respiração, olhares.

### Regra de saída
- Narre em terceira pessoa; nunca fale com "você".
- Produza uma cena fechada e natural, sem comentários externos.
""".strip()

    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# FILTROS DE SAÍDA
# =========================

def render_tail(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r'^\s*\**\s*(microconquista|gancho)\s*:\s*.*$', '', t, flags=re.I|re.M)
    t = re.sub(r'&lt;\s*think\s*&gt;.*?&lt;\s*/\s*think\s*&gt;', '', t, flags=re.I|re.S)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    return t

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
    return t  # não corta por padrão

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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# SIDEBAR — Reorganizado (apenas FASE)
# =========================

with st.sidebar:
    st.title("🧭 Painel do Roteirista")
    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)
    if not api_key:
        st.warning("⚠️ API key ausente para o provedor selecionado. Defina em st.secrets.")
    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    st.session_state.modelo_escolhido_id = modelos_map[modelo_nome]

    st.markdown("---")
    st.markdown("### ✍️ Estilo & Progresso Dramático")
    st.selectbox(
        "Estilo de escrita",
        ["AÇÃO", "ROMANCE LENTO", "NOIR"],
        index=["AÇÃO", "ROMANCE LENTO", "NOIR"].index(st.session_state.get("estilo_escrita", "AÇÃO")),
        key="estilo_escrita",
    )
    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=0, key="nsfw_max_level")
    st.checkbox("Sintonia com o parceiro (modo harmônico)", key="modo_sintonia", value=st.session_state.get("modo_sintonia", True))
    st.select_slider(
        "Ritmo da cena",
        options=[0, 1, 2, 3],
        value=0,
        format_func=lambda n: ["muito lento", "lento", "médio", "rápido"][n],
        key="ritmo_cena",
    )
    st.checkbox("Usar falas da Mary da planilha (usar literalmente)", value=st.session_state.get("usar_falas_mary", False), key="usar_falas_mary")

    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio (apenas Fase)")
    fase_default = mj_carregar_fase_inicial()
    options_fase = sorted(FASES_ROMANCE.keys())
    fase_ui_val = int(st.session_state.get("mj_fase", fase_default))
    fase_ui_val = max(min(fase_ui_val, max(options_fase)), min(options_fase))
    fase_escolhida = st.select_slider("Fase do romance", options=options_fase, value=fase_ui_val, format_func=_fase_label, key="ui_mj_fase")
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
    st.checkbox("Evitar coincidências forçadas (montagem paralela A/B)", value=st.session_state.get("no_coincidencias", True), key="no_coincidencias")
    st.checkbox("Bloquear avanços íntimos sem ordem", value=st.session_state.get("app_bloqueio_intimo", True), key="app_bloqueio_intimo")
    st.selectbox(
        "🎭 Emoção oculta",
        ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensão", "raiva"].index(st.session_state.get("app_emocao_oculta", "nenhuma")),
        key="ui_app_emocao_oculta",
    )
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
    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")

# =========================
# EXIBIR HISTÓRICO
# =========================

with st.container():
    interacoes = carregar_interacoes(n=20)
    for r in interacoes:
        role = r.get("role","user")
        content = r.get("content","")
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
    # SOMENTE FASE: não alteramos “momento”
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

    # Helpers de clímax
    CLIMAX_USER_TRIGGER = re.compile(
        r"\b(finalmente\b.*orgasmo|explode\b.*orgasmo|cheg(a|ou)\b.*cl[ií]max|pode\b.*finalizar|libero\b.*cl[ií]max|goza(r)?\b.*agora)\b",
        flags=re.IGNORECASE
    )
    ORGASM_OUT_PAT = re.compile(
        r"([^.!\n]*\b(cl[ií]max|orgasmo|gozou|gozaram|ejacul[ao]u)\b[^.!?\n]*[.!?])",
        flags=re.IGNORECASE
    )
    def _user_allows_climax(msgs: list) -> bool:
        last_user = ""
        for r in reversed(msgs or []):
            if str(r.get("role","")).lower() == "user":
                last_user = r.get("content","")
                break
        return bool(CLIMAX_USER_TRIGGER.search(last_user or ""))

    def _strip_or_soften_climax(texto: str) -> str:
        if not texto:
            return texto
        texto = ORGASM_OUT_PAT.sub("", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
        if not texto.endswith((".", "…", "!", "?")):
            texto += "…"
        finais = [
            " Eles param um instante, respirando juntos, sem apressar o desfecho.",
            " A tensão fica no ar, guardada para o próximo passo.",
            " Eles se encostam em silêncio, deixando o resto para depois."
        ]
        if all(f not in texto for f in finais):
            texto += random.choice(finais)
        return texto

    def _render_visible(t: str) -> str:
        out = render_tail(t)
        out = sanitize_explicit(out, max_level=int(st.session_state.get("nsfw_max_level", 0)), action="livre")
        return out

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        # reforço memórias usadas no prompt
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
                                # atualização parcial
                                parcial = _render_visible(resposta_txt) + "▌"
                                # bloqueio de clímax on-the-fly (fase <5 ou sem liberação)
                                if st.session_state.get("app_bloqueio_intimo", True):
                                    fase_atual = int(st.session_state.get("mj_fase", 0))
                                    if (fase_atual < 5) and (not _user_allows_climax(st.session_state.session_msgs)):
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

        # Fallback prompts limpos
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

        # BLOQUEIO DE CLÍMAX FINAL (apenas fase + gatilho do usuário)
        if st.session_state.get("app_bloqueio_intimo", True):
            fase_atual = int(st.session_state.get("mj_fase", 0))
            if (fase_atual < 5) and (not _user_allows_climax(st.session_state.session_msgs)):
                visible_txt = _strip_or_soften_climax(visible_txt)

        placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

        # validação simples
        if len(st.session_state.session_msgs) >= 1 and visible_txt and visible_txt != "[Sem conteúdo]":
            pass  # mantido simples

        salvar_interacao("assistant", visible_txt if visible_txt else "[Sem conteúdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt if visible_txt else "[Sem conteúdo]"})

        # Reforço pós-resposta
        try:
            usados = []
            topk_usadas = memoria_longa_buscar_topk(
                query_text=visible_txt,
                k=int(st.session_state.get("k_memoria_longa",3)),
                limiar=float(st.session_state.get("limiar_memoria_longa",0.78)),
            )
            for t, _sc, _sim, _rr in topk_usadas:
                usados.append(t)
            memoria_longa_reforcar(usados)
        except Exception:
            pass
