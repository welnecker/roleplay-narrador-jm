# ============================================================
# Narrador JM â€” Roleplay adulto (sem pornografia explÃ­cita)
# CompatÃ­vel com o mÃ©todo antigo: GOOGLE_CREDS_JSON + oauth2client
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

# (Opcional) Embeddings OpenAI para verificaÃ§Ã£o semÃ¢ntica/memÃ³ria longa
try:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    OPENAI_OK = bool(st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OPENAI_CLIENT = None
    OPENAI_OK = False

# =========================
# CONFIG BÃSICA DO APP
# =========================

PLANILHA_ID_PADRAO = st.secrets.get("SPREADSHEET_ID", "").strip() or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
TAB_INTERACOES = "interacoes_jm"
TAB_PERFIL = "perfil_jm"
TAB_MEMORIAS = "memoria_jm"
TAB_ML = "memoria_longa_jm"
TAB_TEMPLATES = "templates_jm"

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
        st.error(f"Erro ao conectar Ã  planilha: {e}")
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
            return ws
        except Exception:
            return None

def carregar_templates_planilha():
    """Carrega todos os templates sequenciais da aba templates_jm em {template:[etapa1, etapa2,...]}"""
    try:
        ws = _ws(TAB_TEMPLATES, create_if_missing=False)
        if not ws:
            return {}
        rows = _sheet_all_records_cached(TAB_TEMPLATES)
        templates = {}
        for row in rows:
            nome = (row.get("template") or "").strip()
            etapa_str = row.get("etapa") or "1"
            try:
                etapa = int(etapa_str)
            except Exception:
                etapa = 1
            texto = (row.get("texto") or "").strip()
            if nome and texto:
                templates.setdefault(nome, []).append((etapa, texto))
        for nome in templates:
            templates[nome].sort(key=lambda x: x[0])
            templates[nome] = [t[1] for t in templates[nome]]
        return templates
    except Exception as e:
        st.warning(f"Erro ao carregar templates do Sheets: {e}")
        return {}

# --- INICIALIZE AS VARIÃVEIS DE TEMPLATE ---
if "templates_jm" not in st.session_state:
    st.session_state.templates_jm = carregar_templates_planilha()
if "template_ativo" not in st.session_state:
    st.session_state.template_ativo = None
if "etapa_template" not in st.session_state:
    st.session_state.etapa_template = 0
# =====
# UTILIDADES: MEMÃ“RIAS / HISTÃ“RICO
# =====

# ObservaÃ§Ã£o 1: o cÃ³digo espera TAB_MEMORIAS = "memoria_jm"
# Ex.: TAB_MEMORIAS = "memoria_jm"

def _normalize_tag(raw: str) -> str:
    """Normaliza 'tipo' para o formato com colchetes: [all], [mary], [janio], etc."""
    t = (raw or "").strip().lower()
    if not t:
        return ""
    return t if t.startswith("[") else f"[{t}]"

def _parse_ts(s: str) -> str:
    """
    Normaliza timestamp para 'YYYY-MM-DD HH:MM:SS'; se vier vazio/ruim, usa 'agora'.
    Importante: strings nesse formato sÃ£o comparÃ¡veis lexicograficamente.
    """
    s = (s or "").strip()
    try:
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return s
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def carregar_memorias_brutas() -> Dict[str, List[dict]]:
    """
    LÃª 'memoria_jm' (cabeÃ§alho: tipo | conteudo | timestamp) e devolve
    {tag: [ {"conteudo":..., "timestamp":...} ]} com tags normalizadas ([all],[mary],[janio]) e
    timestamp padronizado.
    """
    try:
        regs = _sheet_all_records_cached(TAB_MEMORIAS)  # espera "memoria_jm"
        buckets: Dict[str, List[dict]] = {}
        for r in regs:
            tag = _normalize_tag(r.get("tipo"))
            txt = (r.get("conteudo") or "").strip()
            ts = (r.get("timestamp") or "").strip()
            if tag and txt:
                buckets.setdefault(tag, []).append({"conteudo": txt, "timestamp": ts})
        return buckets
    except Exception as e:
        st.warning(f"Erro ao carregar memÃ³rias: {e}")
        return {}

def persona_block(nome: str, buckets: dict, max_linhas: int = 8) -> str:
    """
    Monta bloco compacto da persona (ordena por prefixos Ãºteis).
    ObservaÃ§Ã£o 2: usa tags no formato [mary], [janio].
    """
    tag = f"[{nome}]"
    linhas = buckets.get(tag, [])
    ordem = ["OBJ:", "TAT:", "LV:", "VOZ:", "BIO:", "ROTINA:", "LACOS:", "APS:", "CONFLITOS:"]

    def peso(linha_dict):
        l = linha_dict["conteudo"]
        up = l.upper()
        for i, p in enumerate(ordem):
            if up.startswith(p):
                return i
        return len(ordem)

    linhas_ordenadas = sorted(linhas, key=peso)[:max_linhas]
    titulo = "JÃ¢nio" if nome in ("janio", "jÃ¢nio") else "Mary" if nome == "mary" else nome.capitalize()
    return (
        f"{titulo}:\n- " + "\n- ".join(linha['conteudo'] for linha in linhas_ordenadas)
    ) if linhas_ordenadas else ""


def persona_block_temporal(nome: str, buckets: dict, ate_ts: str, max_linhas: int = 8) -> str:
    """
    VersÃ£o temporal do bloco de persona.
    Usa apenas memÃ³rias com timestamp <= ate_ts (se houver timestamp).
    MantÃ©m compatibilidade com registros sem timestamp (sÃ£o incluÃ­dos).
    """
    tag = f"[{nome}]"
    linhas = []
    for d in buckets.get(tag, []) or []:
        if not isinstance(d, dict):
            continue
        c = (d.get("conteudo") or "").strip()
        ts = (d.get("timestamp") or "").strip()  # pode estar vazio
        if not c:
            continue
        # Se houver timestamp e corte temporal, exclui memÃ³rias "do futuro"
        if ts and ate_ts and ts > ate_ts:
            continue
        linhas.append((ts, c))

    # Ordena por timestamp crescente (strings ISO ordenam lexicograficamente)
    # Registros sem timestamp ("") ficam no inÃ­cio.
    linhas.sort(key=lambda x: x[0])

    # Pega as Ãºltimas N (mais recentes atÃ© o corte)
    ult = [c for _, c in linhas][-max_linhas:]
    if not ult:
        return ""

    titulo = "JÃ¢nio" if nome in ("janio", "jÃ¢nio") else "Mary" if nome == "mary" else nome.capitalize()
    return f"{titulo}:\n- " + "\n- ".join(ult)


def carregar_resumo_salvo() -> str:
    """
    Busca o Ãºltimo resumo da aba 'perfil_jm' (cabeÃ§alho: timestamp | resumo) com cache TTL.
    """
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
    """
    Salva uma nova linha em 'perfil_jm' (timestamp | resumo) e invalida caches.
    """
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
    Carrega Ãºltimas n interaÃ§Ãµes (role, content) usando cache de sessÃ£o
    para evitar leituras repetidas.
    """
    cache = st.session_state.get("_cache_interacoes", None)
    if cache is None:
        regs = _sheet_all_records_cached(TAB_INTERACOES)
        st.session_state["_cache_interacoes"] = regs
        cache = regs
    return cache[-n:] if len(cache) > n else cache


def salvar_interacao(role: str, content: str):
    """
    Append no Sheets + atualiza cache local (sem reler) com backoff 429.
    (ObservaÃ§Ã£o 3: garante timestamp no padrÃ£o e corrige o else do cache.)
    """
    if not planilha:
        return
    try:
        aba = _ws(TAB_INTERACOES)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # ObservaÃ§Ã£o 3 (formato)
        row_role = f"{role or ''}".strip()
        row_content = f"{content or ''}".strip()
        row = [timestamp, row_role, row_content]
        _retry_429(aba.append_row, row, value_input_option="RAW")

        # atualiza cache local
        lst = st.session_state.get("_cache_interacoes")
        if isinstance(lst, list):
            lst.append({"timestamp": timestamp, "role": row_role, "content": row_content})
        else:
            # (fix) cria corretamente a primeira entrada do cache
            st.session_state["_cache_interacoes"] = [{
                "timestamp": timestamp,
                "role": row_role,
                "content": row_content,
            }]

        _invalidate_sheet_caches()
    except Exception as e:
        st.error(f"Erro ao salvar interaÃ§Ã£o: {e}")

# = BUSCA TEMPORAL DE MEMÃ“RIAS =
def buscar_status_persona_ate(persona_tag: str, momento_ts: str, buckets: dict) -> List[str]:
    """
    Busca os traÃ§os mais recentes da persona atÃ© o timestamp informado.
    persona_tag: ex: '[mary]' (serÃ¡ normalizado)
    momento_ts: timestamp limite (ex: '2025-08-21 16:04:00'; serÃ¡ normalizado)
    buckets: dict retornado por carregar_memorias_brutas()
    """
    tag = _normalize_tag(persona_tag)
    limite = _parse_ts(momento_ts)
    linhas = buckets.get(tag, [])

    # filtra atÃ© o limite e ordena por timestamp (strings jÃ¡ normalizadas)
    linhas_filtradas = [l for l in linhas if (l.get("timestamp") or "") <= limite]
    linhas_ord = sorted(linhas_filtradas, key=lambda x: x.get("timestamp", ""))

    return [l.get("conteudo", "") for l in linhas_ord if l.get("conteudo")]



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
        return f"âš ï¸ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# =========================
# MEMÃ“RIA LONGA (Sheets + Embeddings/OpenAI opcional) â€” TIME-AWARE
# =========================

def _sheet_ensure_memoria_longa():
    """Retorna a aba memoria_longa_jm se existir (nÃ£o cria automaticamente)."""
    return _ws(TAB_ML, create_if_missing=False)

def _serialize_vec(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))

def _deserialize_vec(s: str) -> np.ndarray:
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        return np.zeros(1, dtype=float)

def memoria_longa_salvar(texto: str, tags: str = "") -> bool:
    """Salva uma memÃ³ria com embedding (se possÃ­vel) e score inicial. Invalida cache."""
    aba = _sheet_ensure_memoria_longa()
    if not aba:
        st.warning("Aba 'memoria_longa_jm' nÃ£o encontrada â€” crie com cabeÃ§alhos: texto | embedding | tags | timestamp | score")
        return False
    emb = gerar_embedding_openai(texto)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = [texto.strip(), _serialize_vec(emb) if emb is not None else "", (tags or "").strip(), ts, 1.0]
    try:
        _retry_429(aba.append_row, linha, value_input_option="RAW")
        _invalidate_sheet_caches()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar memÃ³ria longa: {e}")
        return False

def memoria_longa_listar_registros():
    """Retorna todos os registros da aba memoria_longa_jm (cache TTL)."""
    try:
        return _sheet_all_records_cached(TAB_ML)
    except Exception:
        return []

def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-zÃ -Ãº0-9]+", (s or "").lower()))

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78, ate_ts=None):
    """
    Top-K memÃ³rias (time-aware). Usa embeddings se existir; senÃ£o, Jaccard simples.
    Se 'ate_ts' for informado (formato 'YYYY-MM-DD HH:MM:SS'), ignora registros com
    timestamp > ate_ts (ou seja, memÃ³rias 'do futuro' em relaÃ§Ã£o ao histÃ³rico atual).
    Retorna lista de tuplas (texto, score, sim, rr).
    """
    try:
        dados = _sheet_all_records_cached(TAB_ML)  # [{'texto','embedding','tags','timestamp','score'}, ...]
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

        # --- Corte temporal: ignora memÃ³rias mais novas que o corte (se fornecido)
        if ate_ts and row_ts and row_ts > ate_ts:
            continue  # strings no formato YYYY-MM-DD HH:MM:SS sÃ£o comparÃ¡veis lexicograficamente

        # Similaridade por embedding (quando disponÃ­vel), senÃ£o fallback lexical
        if q is not None and emb_s:
            vec = _deserialize_vec(emb_s)
            if vec.ndim == 1 and vec.size >= 10 and np.linalg.norm(vec) > 0 and np.linalg.norm(q) > 0:
                sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
            else:
                sim = 0.0
        else:
            s1 = _tokenize(texto)
            s2 = _tokenize(query_text)
            sim = len(s1 & s2) / max(1, len(s1 | s2))

        if sim >= limiar:
            rr = 0.7 * sim + 0.3 * score
            candidatos.append((texto, score, sim, rr))

    candidatos.sort(key=lambda x: x[3], reverse=True)
    return candidatos[:k]

def memoria_longa_reforcar(textos_usados: list):
    """Aumenta o score das memÃ³rias usadas (pequeno reforÃ§o) com backoff + correÃ§Ã£o de Ã­ndices."""
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = _sheet_all_values_cached(TAB_ML)
        if not dados or len(dados) < 2:
            return
        headers = dados[0]  # cabeÃ§alho Ã© a primeira linha
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
        "permitidos": "olhares; near-miss (mesmo cafÃ©/rua/Ã´nibus); detalhe do ambiente",
        "proibidos": "troca de nomes; toques; conversa pessoal"},
    1: {"nome": "PercepÃ§Ã£o",
        "permitidos": "cumprimento neutro; pergunta impessoal curta",
        "proibidos": "contato fÃ­sico; confidÃªncias"},
    2: {"nome": "Conhecidos",
        "permitidos": "troca de nomes; pequena ajuda; 1 pergunta pessoal leve",
        "proibidos": "toque prolongado; encontro a sÃ³s planejado"},
    3: {"nome": "Amizade",
        "permitidos": "conversa 10â€“20 min; caminhar juntos; troca de contatos; 1 gesto de afeto leve (com consentimento)",
        "proibidos": "beijos; carÃ­cias intimistas"},
    4: {"nome": "ConfianÃ§a / Quase",
        "permitidos": "confidÃªncias; abraÃ§o com consentimento expresso; marcar encontro futuro claro",
        "proibidos": "sexo; sexo oral/manual; pressa ou â€œprovas de amorâ€ fÃ­sicas"},
    5: {"nome": "Compromisso / Encontro definitivo",
        "permitidos": "beijo prolongado; dormir juntos; consumaÃ§Ã£o implÃ­cita (fade-to-black); manhÃ£ seguinte sugerida",
        "proibidos": ""},
}

FLAG_FASE_TXT_PREFIX = "FLAG: mj_fase="

def _fase_label(n: int) -> str:
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE[0])
    return f"{int(n)} â€” {d['nome']}"

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

# --------- Motor de Momento ----------
MOMENTOS = {
    0: {"nome": "AproximaÃ§Ã£o logÃ­stica",
        "objetivo": "um acompanha o outro (ex.: atÃ© o pÃ­er), clima cordial",
        "permitidos": "gentilezas; proximidade leve; diÃ¡logo casual",
        "proibidos": "declaraÃ§Ã£o; revelaÃ§Ãµes Ã­ntimas; toques prolongados",
        "gatilhos": [r"\b(p[iÃ­]er|acompanhar|vamos embora|te levo)\b"],
        "proximo": 1},
    1: {"nome": "DeclaraÃ§Ã£o",
        "objetivo": "um deles declara amor/ importÃ¢ncia",
        "permitidos": "confissÃ£o afetiva; silÃªncio tenso; abraÃ§o curto",
        "proibidos": "negociaÃ§Ã£o sexual; tirar roupas; exploraÃ§Ã£o do corpo",
        "gatilhos": [r"\b(amo voc[eÃª]|te amo|n[aÃ£]o paro de pensar)\b"],
        "proximo": 2},
    2: {"nome": "RevelaÃ§Ã£o sensÃ­vel",
        "objetivo": "Mary revela que Ã© virgem / vulnerabilidade equivalente",
        "permitidos": "dizer 'sou virgem'; estipular limites; conforto mÃºtuo",
        "proibidos": "carÃ­cias Ã­ntimas; tirar roupas",
        "gatilhos": [r"\b(sou virgem|nunca fiz|meu limite)\b"],
        "proximo": 3},
    3: {"nome": "Consentimento explÃ­cito",
        "objetivo": "alinhamento de limites e um 'sim' claro",
        "permitidos": "nomear fronteiras; pedir/receber consentimento; decidir 'agora sim'",
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eÃª])\b", r"\b(at[eÃ©] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade (elÃ­ptica)",
        "objetivo": "intimidade sugerida (fade-to-black) / pÃ³s-ato implÃ­cito",
        "permitidos": "beijos longos; proximidade forte; fade-to-black; manhÃ£ seguinte implÃ­cita",
        "proibidos": "",
        "gatilhos": [r"\b(quarto|cama|luz baixa|porta fechada|manh[aÃ£] seguinte)\b"],
        "proximo": 4},
}

def _momento_label(n: int) -> str:
    m = MOMENTOS.get(int(n), MOMENTOS[0])
    return f"{int(n)} â€” {m['nome']}"

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
    # NÃ£o bloquear/censurar conteÃºdo explÃ­cito por momento.
    return ""

# =========================
# PROVEDORES E MODELOS
# =========================

MODELOS_OPENROUTER = {
    "ðŸ’¬ DeepSeek V3 â˜…â˜…â˜…â˜… ($)": "deepseek/deepseek-chat-v3-0324",
    "ðŸ§  DeepSeek R1 0528 â˜…â˜…â˜…â˜…â˜† ($$)": "deepseek/deepseek-r1-0528",
    "ðŸ§  DeepSeek R1T2 Chimera â˜…â˜…â˜…â˜… (free)": "tngtech/deepseek-r1t2-chimera:free",
    "ðŸ§  GPT-4.1 â˜…â˜…â˜…â˜…â˜… (1M ctx)": "openai/gpt-4.1",
    "ðŸ‘‘ WizardLM 8x22B â˜…â˜…â˜…â˜…â˜† ($$$)": "microsoft/wizardlm-2-8x22b",
    "ðŸ‘‘ Qwen 235B 2507 â˜…â˜…â˜…â˜…â˜… (PAID)": "qwen/qwen3-235b-a22b-07-25",
    "ðŸ‘‘ EVA Qwen2.5 72B â˜…â˜…â˜…â˜…â˜… (RP Pro)": "eva-unit-01/eva-qwen-2.5-72b",
    "ðŸ‘‘ EVA Llama 3.33 70B â˜…â˜…â˜…â˜…â˜… (RP Pro)": "eva-unit-01/eva-llama-3.33-70b",
    "ðŸŽ­ Nous Hermes 2 Yi 34B â˜…â˜…â˜…â˜…â˜†": "nousresearch/nous-hermes-2-yi-34b",
    "ðŸ”¥ MythoMax 13B â˜…â˜…â˜…â˜† ($)": "gryphe/mythomax-l2-13b",
    "ðŸ’‹ LLaMA3 Lumimaid 8B â˜…â˜…â˜† ($)": "neversleep/llama-3-lumimaid-8b",
    "ðŸŒ¹ Midnight Rose 70B â˜…â˜…â˜…â˜†": "sophosympatheia/midnight-rose-70b",
    "ðŸŒ¶ï¸ Noromaid 20B â˜…â˜…â˜†": "neversleep/noromaid-20b",
    "ðŸ’€ Mythalion 13B â˜…â˜…â˜†": "pygmalionai/mythalion-13b",
    "ðŸ‰ Anubis 70B â˜…â˜…â˜†": "thedrummer/anubis-70b-v1.1",
    "ðŸ§š Rocinante 12B â˜…â˜…â˜†": "thedrummer/rocinante-12b",
    "ðŸ· Magnum v2 72B â˜…â˜…â˜†": "anthracite-org/magnum-v2-72b",
}

MODELOS_TOGETHER_UI = {
    "ðŸ§  Qwen3 Coder 480B (Together)": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "ðŸ‘‘ Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "ðŸ‘‘ Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
}

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
    # padrÃ£o elevado para 3 (explÃ­cito)
    calor = int(st.session_state.get("steam_level", st.session_state.get("nsfw_max_level", 3)))
    regras = f"""
âš–ï¸ Regras de coerÃªncia:
- Narre em terceira pessoa; nÃ£o se dirija ao leitor como "vocÃª".
- Consentimento claro antes de qualquer gesto significativo.
- JÃ¢nio nÃ£o pressiona; respeita o ritmo de Mary.
- Linguagem sensual proporcional ao nÃ­vel de calor ({calor}).
""".strip()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    if fase >= 5:
        regras += """
- Intimidade pode ser descrita com detalhes explÃ­citos (sem necessidade de corte)."""
    else:
        regras += """
- Sem consumaÃ§Ã£o em cena; foque em progressÃ£o coerente."""
    return prompt_base + "\n" + regras


def gerar_mary_sensorial(level: int = 2, n: int = 2, hair_on: bool = True) -> str:
    """
    Gera 1â€“3 frases sensoriais sobre Mary.
      level: 0=off, 1=leve, 2=marcado, 3=ousado
      n: quantidade de frases
      hair_on: garante ao menos 1 frase sobre os cabelos (negros, volumosos, levemente ondulados)
    """
    if level <= 0 or n <= 0:
        return ""

    # Base
    base_leve = [
        "Mary caminha com ritmo seguro; hÃ¡ algo hipnÃ³tico no balanÃ§o dos quadris.",
        "O olhar de Mary prende fÃ¡cil: direto, firme, cativante.",
        "O perfume de Mary chega antes dela, discreto e morno.",
        "O sorriso aparece quando quer; breve, afiado, certeiro.",
    ]
    base_marcado = [
        "Os quadris de Mary balanÃ§am num compasso que chama atenÃ§Ã£o sem pedir licenÃ§a.",
        "Enquanto caminha, os seios balanÃ§am de leve sob o tecido.",
        "O tecido roÃ§a nas pernas e denuncia o passo: firme, Ã­ntimo, decidido.",
        "O olhar de Mary Ã© um convite silencioso â€” confiante e difÃ­cil de sustentar por muito tempo.",
    ]
    base_ousado = [
        "O balanÃ§o dos quadris de Mary Ã© quase cruel: entra na cabeÃ§a e nÃ£o sai.",
        "Os seios acompanham a passada num movimento suave que acende o ambiente.",
        "O olhar de Mary encosta na pele de quem cruza com ela: quente, demorado, insinuante.",
        "O perfume fica na memÃ³ria como um toque atrÃ¡s da nuca.",
    ]

    # Frases especÃ­ficas de cabelo (negros, volumosos, levemente ondulados)
    hair_leve = [
        "Os cabelos de Mary â€” negros, volumosos, levemente ondulados â€” descansam nos ombros e acompanham o passo.",
        "Os cabelos negros, volumosos e levemente ondulados moldam o rosto quando ela vira de leve.",
    ]
    hair_marcado = [
        "Cabelos negros, volumosos, levemente ondulados, fazem um arco quando ela vira o rosto, reforÃ§ando o balanÃ§o do corpo.",
        "Os cabelos, negros e volumosos, ondulam de leve a cada passada e criam uma moldura hipnÃ³tica.",
    ]
    hair_ousado = [
        "Os cabelos negros, volumosos e levemente ondulados deslizam pela clavÃ­cula como um toque que fica.",
        "O balanÃ§o dos cabelos negros â€” volumosos, levemente ondulados â€” marca o compasso do corpo de Mary.",
    ]

    pool = []
    hair_pool = []
    if level == 1:
        pool = base_leve
        hair_pool = hair_leve
    elif level == 2:
        pool = base_leve + base_marcado
        hair_pool = hair_leve + hair_marcado
    else:  # level >= 3
        pool = base_leve + base_marcado + base_ousado
        hair_pool = hair_leve + hair_marcado + hair_ousado

    n_eff = max(1, min(n, len(pool)))
    frases = random.sample(pool, k=n_eff)

    if hair_on and hair_pool:
        hair_line = random.choice(hair_pool)
        if hair_line not in frases:
            frases.insert(0, hair_line)
            if len(frases) > n_eff:
                frases = frases[:n_eff]

    return " ".join(frases)

def construir_prompt_com_narrador() -> str:
    memos = carregar_memorias_brutas()
    # recorrentes = [c["conteudo"] for (t, lst) in memos.items() if t == "[all]" for c in lst]
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS[0])
    proximo_nome = MOMENTOS.get(mdata.get("proximo", 0), MOMENTOS[0])["nome"]
    estilo = st.session_state.get("estilo_escrita", "AÃ‡ÃƒO")

    
    # Camada sensorial de Mary (para o 1Âº parÃ¡grafo da cena)
    _sens_on = bool(st.session_state.get("mary_sensorial_on", True))
    _sens_level = int(st.session_state.get("mary_sensorial_level", 2))
    _sens_n = int(st.session_state.get("mary_sensorial_n", 2))
    mary_sens_txt = gerar_mary_sensorial(_sens_level, n=_sens_n) if _sens_on else ""
# HistÃ³rico do Sheets
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r['role']}: {r['content']}" for r in hist) if hist else "(sem histÃ³rico)"

    # >>> CORTE TEMPORAL (atÃ© o timestamp da Ãºltima interaÃ§Ã£o)
    if hist:
        # Se vocÃª jÃ¡ tem _parse_ts(), pode usar: ate_ts = _parse_ts(hist[-1].get("timestamp", ""))
        ate_ts = _parse_ts(hist[-1].get("timestamp", "")) if hist else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        ate_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MemÃ³ria longa Top-K (texto apenas; se nÃ£o houver, "(nenhuma)") â€” respeitando o tempo
    ml_topk_txt = "(nenhuma)"
    st.session_state["_ml_topk_texts"] = []
    if st.session_state.get("use_memoria_longa", True) and hist:
        try:
            topk = memoria_longa_buscar_topk(
                query_text=hist[-1]["content"],
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
                ate_ts=ate_ts,  # << corte temporal aplicado aqui
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

    # Diretrizes [all] respeitando o tempo (<= ate_ts)
    recorrentes = [
        (d.get("conteudo") or "").strip()
        for d in memos.get("[all]", [])
        if isinstance(d, dict) and d.get("conteudo") and (not d.get("timestamp") or d.get("timestamp") <= ate_ts)
    ]
    st.session_state["_ml_recorrentes"] = recorrentes

    # DossiÃª temporal (somente memÃ³rias atÃ© ate_ts)
    dossie = []
    # Requer a funÃ§Ã£o persona_block_temporal(nome, memos, ate_ts, max_linhas)
    mary = persona_block_temporal("mary", memos, ate_ts, 8)
    janio = persona_block_temporal("janio", memos, ate_ts, 8)
    if mary:
        dossie.append(mary)
    if janio:
        dossie.append(janio)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    # DefiniÃ§Ã£o da flag para montagem paralela
    flag_parallel = bool(st.session_state.get("no_coincidencias", True))

    prompt = f"""
VocÃª Ã© o Narrador de um roleplay dramÃ¡tico brasileiro, foque em Mary e JÃ¢nio. NÃ£o repita instruÃ§Ãµes nem tÃ­tulos.

### DossiÃª (personas)
{dossie_txt}

### Diretrizes gerais (ALL)
{chr(10).join(f"- {c}" for c in recorrentes) if recorrentes else "(vazio)"}

### Perfil (resumo mais recente)
{perfil or "(vazio)"}

### HistÃ³rico recente (planilha)
{hist_txt}

### Estilo
- Use o estilo **{estilo}**:
{("- Frases curtas, cortes rÃ¡pidos, foco em gesto/ritmo.") if estilo=="AÃ‡ÃƒO" else
("- Atmosfera sombria, subtexto, silÃªncio que pesa.") if estilo=="NOIR" else
("- Ritmo lento, tensÃ£o emocional, detalhes sensoriais (sem grafismo).")}
- Todas as cenas devem ser altamente sensoriais e fÃ­sicas. Detalhe toques, apertos, tremores, suor, respiraÃ§Ã£o, gradaÃ§Ã£o do prazer atÃ© o limite do orgasmo e depois.
- As falas de Mary devem ser sensuais, realistas e explÃ­citas, sem linguagem poÃ©tica ou frases floreadas.
- Mary verbaliza desejo sexual, pede e orienta sexo e prazer sem pudor: frases diretas e sensoriais como "Me fode agora!", "Quero gozar!", "Tira minha roupa!", "Mete fundo!", "Estou molhada!", "Faz eu gritar de prazer!", "Estou gozando!", "Que delÃ­cia!", "Vai, nÃ£o para!", "Me pega desse jeito!", "PÃµe a mÃ£o aqui!", â€œGoza dentro!â€, â€œPorra, tÃ´ gozando!â€, â€œMais forte!â€ etc.
- NÃ£o use metÃ¡foras, floreios lÃ­ricos ou reflexÃµes profundas; priorize comandos, reaÃ§Ãµes corporais e escancaradas do tesÃ£o e pÃ³s-tesÃ£o.
- No clÃ­max, **obrigatoriamente inclua frases explosivas e altas de orgasmo**: Mary e JÃ¢nio devem gemer, gritar ou falar explicitamente seu prazer ("Estou gozando!", "Ahhh, que delÃ­cia!", "Goza dentro de mim!", "Mais forte!", etc.) antes de qualquer relaxamento ou carinho.
- O desfecho da cena sexual sempre inclui reaÃ§Ã£o explosiva: gemidos altos, frases de gozo ditas em voz alta, corpo pulsando, tremores intensos; sÃ³ depois, podem relaxar exaustos, suados, abraÃ§ados, comentando o prazer ("Nunca gozei desse jeito!", "Estou toda mole", "NÃ£o sinto as pernas", etc).
- Jamais encerre sem que pelo menos um dos personagens verbalize de forma explÃ­cita o orgasmo e a intensidade do prazer.
- Mary reage fisicamente com agarrÃµes, mordidas, pedidos ousados e sinceros, sem vergonha. Nunca escreva desfecho seco ou cÃ´mico; sempre finalize mostrando o impacto corporal real, prazeroso e o orgasmo verbalizado.

### Camada sensorial â€” Mary (OBRIGATÃ“RIA no 1Âº parÃ¡grafo)
{mary_sens_txt or "- Comece com 1â€“2 frases curtas sobre o caminhar, olhar, perfume e cabelos (negros, volumosos, levemente ondulados) de Mary; pode mencionar o balanÃ§o suave dos seios sob o tecido (sem vulgaridade)."}
- Aplique essa camada ANTES do primeiro diÃ¡logo.
- Frases curtas, diretas, fÃ­sicas; evite metÃ¡foras rebuscadas.

### MemÃ³ria longa â€” Top-K relevantes
{ml_topk_txt}
### â±ï¸ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}
### ðŸŽ¯ Momento dramÃ¡tico (agora)
- Momento: {_momento_label(momento_atual)}
- Objetivo da cena: {mdata['objetivo']}
- Nesta cena, **permita**: {mdata['permitidos']}
- Evite/adiar: {mdata['proibidos']}
- **Micropassos:** avance no mÃ¡ximo **{int(st.session_state.get("max_avancos_por_cena",1))}** subpasso(s) rumo a: {proximo_nome}.
- Se o roteirista pedir salto maior, **negocie**: nomeie limites, peÃ§a consentimento, e **prepare** a transiÃ§Ã£o (nÃ£o pule etapas).
### Geografia & Montagem
- **NÃ£o force coincidÃªncias**: se nÃ£o houver ponte clara (mensagem, convite, â€œensaio 18hâ€¦â€, pedido do usuÃ¡rio), mantenha **Mary e JÃ¢nio em locais distintos** e use **montagem paralela** (A â†” B).
- **Comece cada bloco** com uma frase que **ancore lugar e hora** (ex.: â€œUFES â€” corredor de Pedagogia, 9h15 â€” â€¦â€ ou â€œTerminal Laranjeiras, 9h18 â€” â€¦â€). NÃ£o use tÃ­tulos; escreva isso na **primeira frase** do parÃ¡grafo.
- **Se `montagem paralela`** (valor sugerido: {flag_parallel}):
  - Estruture em **2 blocos alternados**: primeiro Mary, depois JÃ¢nio (ou vice-versa), cada um no **seu lugar**.
  - Os blocos podem se â€œresponderâ€ por subtexto (mensagens, lembranÃ§as, sons Ã  distÃ¢ncia), mas **sem co-presenÃ§a fÃ­sica**.
- **Se houver ponte plausÃ­vel explÃ­cita**, pode convergir para co-presenÃ§a ao final da cena (de forma plausÃ­vel), **sem teletransporte**.
- **Sem ponte diegÃ©tica explÃ­cita, um personagem nÃ£o pode saber, afirmar ou reagir a fatos que sÃ³ ocorreram no bloco do outro; se houver pressentimento/ciÃºme, redigir sem afirmar o fato. Exemplos de ponte: mensagem, foto/story, ligaÃ§Ã£o, testemunha, encontro marcado â€” se existir, mostre isso na cena (ex.: celular vibra e aparece um story)**.
- **Objetos diegÃ©ticos: se a cÃ¢mera nÃ£o couber na situaÃ§Ã£o (encontro, banho, mar, revista), mostre a aÃ§Ã£o de guarda antes e ignore o objeto atÃ© a retomada; nÃ£o descreva interaÃ§Ã£o fÃ­sica com a cÃ¢mera nesses contextos**.
### Formato OBRIGATÃ“RIO da cena
- **Inclua DIÃLOGOS diretos** com travessÃ£o (â€”), intercalados com aÃ§Ã£o e reaÃ§Ã£o fÃ­sica/visual. MÃ­nimo: **4 falas** no total.
- Garanta **pelo menos 2 falas de Mary e 2 de JÃ¢nio** (quando ambos estiverem na cena).
- **NÃ£o inclua pensamentos internos em itÃ¡lico, reflexÃµes internas ou monÃ³logos subjetivos dos personagens.**
- NÃ£o escreva blocos finais de crÃ©ditos, microconquistas, resumos ou ganchos. Apenas narraÃ§Ã£o e interaÃ§Ã£o direta.
- Mostre somente aÃ§Ãµes, gestos, expressÃµes do ambiente, clima corporal e diÃ¡logos.
- Sem tÃ­tulos de seÃ§Ã£o, microconquista ou gancho, nem qualquer nota meta ao final.
### Regra de saÃ­da
- Narre em **terceira pessoa**; nunca fale com "vocÃª".
- Produza uma cena fechada e natural, sem inserir comentÃ¡rios externos ou instruÃ§Ãµes.
""".strip()
    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# FILTROS DE SAÃDA
# =========================

def render_tail(t: str) -> str:
    if not t:
        return ""
    # remove rÃ³tulos meta e blocos <think>
    t = re.sub(r'^\s*\**\s*(microconquista|gancho)\s*:\s*.*$', '', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'&lt;\s*think\s*&gt;.*?&lt;\s*/\s*think\s*&gt;', '', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    return t

EXPL_PAT = re.compile(
    r"\b(seio[s]?|mamilos?|bunda|fio[- ]?dental|genit[aÃ¡]lia|ere[cÃ§][aÃ£]o|penetra[cÃ§][aÃ£]o|"
    r"boquete|gozada|gozo|sexo oral|chupar|enfiar)\b",
    flags=re.IGNORECASE
)

def classify_nsfw_level(t: str) -> int:
    if EXPL_PAT.search(t or ""):
        return 3  # explÃ­cito
    if re.search(r"\b(cintura|pesco[cÃ§]o|costas|beijo prolongado|respira[cÃ§][aÃ£]o curta)\b", (t or ""), re.IGNORECASE):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aÃ£]os dadas|beijo)\b", (t or ""), re.IGNORECASE):
        return 1
    return 0

def sanitize_explicit(t: str, max_level: int, action: str) -> str:
    # LiberaÃ§Ã£o: se o conteÃºdo for de nÃ­vel <= max_level, retorna tal como estÃ¡.
    lvl = classify_nsfw_level(t)
    if lvl <= max_level:
        return t
    # Se extrapolar o mÃ¡ximo definido, nÃ£o cortar por padrÃ£o (liberar NSFW).
    return t

def redact_for_logs(t: str) -> str:
    if not t:
        return ""
    t = re.sub(EXPL_PAT, "[â€¦]", t, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteÃºdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

def precisa_reforcar_dialogo(texto: str) -> bool:
    if not texto:
        return True
    n_dialog = len(re.findall(r'(^|\n)\s*(â€”|")', texto))
    n_thoughts = len(re.findall(r'\*[^*\n]{4,}\*', texto))
    return (n_dialog < 4) or (n_thoughts < 2)

# =========================
# UI â€” CABEÃ‡ALHO E CONTROLES
# =========================

st.title("ðŸŽ¬ Narrador JM")
st.subheader("VocÃª Ã© o roteirista. Digite uma direÃ§Ã£o de cena. A IA narrarÃ¡ Mary e JÃ¢nio.")
st.markdown("---")

# InicializaÃ§Ã£o dos estados de sessÃ£o (inclusive dos templates)
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
#if "nsfw_max_level" not in st.session_state:
 #   st.session_state.nsfw_max_level = 3
if "estilo_escrita" not in st.session_state:
    st.session_state.estilo_escrita = "AÃ‡ÃƒO"
if "templates_jm" not in st.session_state:
    st.session_state.templates_jm = carregar_templates_planilha()
if "template_ativo" not in st.session_state:
    st.session_state.template_ativo = None
if "etapa_template" not in st.session_state:
    st.session_state.etapa_template = 0

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### ðŸ“– Ãšltimo resumo salvo:")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponÃ­vel.")
with col2:
    st.markdown("#### âš™ï¸ OpÃ§Ãµes")
    st.write(
        f'- Bloqueio Ã­ntimo: {"Sim" if st.session_state.get("app_bloqueio_intimo", False) else "NÃ£o"}\n'
        f'- EmoÃ§Ã£o oculta: {st.session_state.get("app_emocao_oculta", "").capitalize()}'
    )


# =========================
# SIDEBAR â€” Reorganizado
# =========================

with st.sidebar:
    st.title("ðŸ§­ Painel do Roteirista")
    # Provedor/modelos
    provedor = st.radio("ðŸŒ Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)
    if not api_key:
        st.warning("âš ï¸ API key ausente para o provedor selecionado. Defina em st.secrets.")
    modelo_nome = st.selectbox("ðŸ¤– Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    st.session_state.modelo_escolhido_id = modelo_escolhido_id_ui

    st.markdown("---")
    st.markdown("### âœï¸ Estilo & Progresso DramÃ¡tico")
    st.selectbox(
        "Estilo de escrita",
        ["AÃ‡ÃƒO", "ROMANCE LENTO", "NOIR"],
        index=["AÃ‡ÃƒO", "ROMANCE LENTO", "NOIR"].index(st.session_state.get("estilo_escrita", "AÃ‡ÃƒO")),
        key="estilo_escrita",
    )
    st.slider("NÃ­vel de calor (0=leve, 3=explÃ­cito)", 0, 3, value=3, key="nsfw_max_level")

    st.markdown("---")
    st.markdown("### ðŸ’ž Romance Mary & JÃ¢nio")
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
        if st.button("âž• AvanÃ§ar 1 passo"):
            mj_set_fase(min(st.session_state.get("mj_fase", 0) + 1, max(options_fase)), persist=True)
    with col_b:
        if st.button("â†º Reiniciar (0)"):
            mj_set_fase(0, persist=True)

    st.markdown("---")
    st.markdown("### ðŸŽ¬ Roteiros Sequenciais (Templates)")
    nomes_templates = list(st.session_state.templates_jm.keys())
    if st.button("ðŸ”„ Recarregar templates"):
        st.session_state.templates_jm = carregar_templates_planilha()
        st.success("Templates atualizados da planilha!")
    if nomes_templates:
        roteiro_escolhido = st.selectbox("Escolha o roteiro:", nomes_templates)
        if st.button("Iniciar roteiro") or (st.session_state.template_ativo != roteiro_escolhido):
            st.session_state.template_ativo = roteiro_escolhido
            st.session_state.etapa_template = 0
        if st.session_state.template_ativo:
            etapas = st.session_state.templates_jm.get(st.session_state.template_ativo, [])
            etap = st.session_state.etapa_template
            if etap < len(etapas):
                st.markdown(f"Etapa atual: {etap + 1} de {len(etapas)}")
                if st.button("PrÃ³xima etapa (*)"):
                    comando = etapas[etap]
                    salvar_interacao("user", comando)
                    st.session_state.session_msgs.append({"role": "user", "content": comando})
                    st.session_state.etapa_template += 1
            else:
                st.success("Roteiro concluÃ­do!")
                st.session_state.template_ativo = None
                st.session_state.etapa_template = 0
    else:
        st.info("Nenhum template encontrado na aba templates_jm.")

    st.markdown("---")
    st.checkbox(
        "Evitar coincidÃªncias forÃ§adas (montagem paralela A/B)",
        value=st.session_state.get("no_coincidencias", True),
        key="no_coincidencias",
    )
    st.checkbox(
        "Bloquear avanÃ§os Ã­ntimos sem ordem",
        value=st.session_state.app_bloqueio_intimo,
        key="ui_bloqueio_intimo",
    )
    st.selectbox(
        "ðŸŽ­ EmoÃ§Ã£o oculta",
        ["nenhuma", "tristeza", "felicidade", "tensÃ£o", "raiva"],
        index=["nenhuma", "tristeza", "felicidade", "tensÃ£o", "raiva"].index(st.session_state.app_emocao_oculta),
        key="ui_app_emocao_oculta",
    )
    st.session_state.app_bloqueio_intimo = st.session_state.get("ui_bloqueio_intimo", False)
    st.session_state.app_emocao_oculta = st.session_state.get("ui_app_emocao_oculta", "nenhuma")

    st.markdown("---")
    st.markdown("### â±ï¸ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    st.markdown("---")
    st.markdown("### ðŸ—ƒï¸ MemÃ³ria Longa")
    st.checkbox("Usar memÃ³ria longa no prompt", value=st.session_state.get("use_memoria_longa", True), key="use_memoria_longa")
    st.slider("Top-K memÃ³rias", 1, 5, int(st.session_state.get("k_memoria_longa", 3)), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95, float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")

    st.markdown("---")
    if st.button("ðŸ“ Gerar resumo do capÃ­tulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in inter) if inter else ""
            prompt_resumo = (
                "Resuma o seguinte trecho como um capÃ­tulo de novela brasileiro, mantendo tom e emoÃ§Ãµes.\n\n"
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

    if st.button("ðŸ’¾ Salvar Ãºltima resposta como memÃ³ria"):
        ultimo_assist = ""
        for m in reversed(st.session_state.get("session_msgs", [])):
            if m.get("role") == "assistant":
                ultimo_assist = m.get("content", "").strip()
                break
        if ultimo_assist:
            ok = memoria_longa_salvar(ultimo_assist, tags="auto")
            st.success("MemÃ³ria de longo prazo salva!" if ok else "Falha ao salvar memÃ³ria.")
        else:
            st.info("Ainda nÃ£o hÃ¡ resposta do assistente nesta sessÃ£o.")

    st.markdown("### ðŸ§© HistÃ³rico no prompt")
    st.slider("InteraÃ§Ãµes do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")



   
# =========================
# EXIBIR HISTÃ“RICO (depois resumo)
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
        with st.expander("ðŸ§  Resumo do capÃ­tulo (mais recente)"):
            st.markdown(st.session_state.resumo_capitulo)

# =========================
# ENVIO DO USUÃRIO + STREAMING (OpenRouter/Together) + FALLBACKS
# =========================

entrada = st.chat_input("Digite sua direÃ§Ã£o de cena...")

if entrada:
    # 0) Atualiza Momento sugerido (opcional e seguro)
    try:
        mom_atual = int(st.session_state.get("momento", momento_carregar()))
        mom_sug = detectar_momento_sugerido(entrada, fallback=mom_atual)
        mom_novo = clamp_momento(mom_atual, mom_sug, int(st.session_state.get("max_avancos_por_cena", 1)))
        if st.session_state.get("app_bloqueio_intimo", False):
            mom_novo = clamp_momento(mom_atual, mom_sug, 1)
        momento_set(mom_novo, persist=False)
    except Exception:
        pass

    # 1) Salva a entrada e mantÃ©m histÃ³rico de sessÃ£o
    salvar_interacao("user", str(entrada))
    st.session_state.session_msgs.append({"role": "user", "content": str(entrada)})

    # 2) ConstrÃ³i prompt principal
    prompt = construir_prompt_com_narrador()

    # 3) HistÃ³rico curto (somente sessÃ£o atual; o prompt jÃ¡ inclui Ãºltimas do sheet)
    historico = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in st.session_state.session_msgs]

    # 4) Provedor + modelo
    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets.get("TOGETHER_API_KEY", "")
        # Garanta que o ID tenha o A35B:
        # ex.: "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets.get("OPENROUTER_API_KEY", "")
        model_to_call = st.session_state.modelo_escolhido_id

    if not auth:
        st.error("A chave de API do provedor selecionado nÃ£o foi definida em st.secrets.")
        st.stop()

    # 5) Mensagens
    system_pt = {
        "role": "system",
        "content": (
            "Responda em portuguÃªs do Brasil. Evite conteÃºdo meta. "
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

    # 6) Render / Filtro de saÃ­da
    def _render_visible(t: str) -> str:
        out = render_tail(t)
        out = sanitize_explicit(
            out,
            max_level=int(st.session_state.get("nsfw_max_level", 3)),
            action="livre"
        )
        return out

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""   # texto bruto vindo do stream
        last_update = time.time()

        # 7) ReforÃ§o antecipado: memÃ³rias que ENTRARAM no prompt (topk + recorrentes)
        try:
            usados_prompt = []
            usados_prompt.extend(st.session_state.get("_ml_topk_texts", []))
            usados_prompt.extend(st.session_state.get("_ml_recorrentes", []))
            usados_prompt = [t for t in usados_prompt if t]
            if usados_prompt:
                memoria_longa_reforcar(usados_prompt)
        except Exception:
            pass

        # 8) STREAM
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
                            # CORRETO: choices[0]["delta"]["content"]
                            delta = j["choices"][0]["delta"].get("content", "")
                            if not delta:
                                continue
                            resposta_txt += delta
                            if time.time() - last_update > 0.10:
                                placeholder.markdown(_render_visible(resposta_txt) + "â–Œ")
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro no streaming: {e}")

        # 9) FALLBACKS se veio vazio
        visible_txt = _render_visible(resposta_txt).strip()

        if not visible_txt:
            # 9a) retry sem stream
            try:
                r2 = requests.post(
                    endpoint, headers=headers,
                    json={**payload, "stream": False},
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
                if r2.status_code == 200:
                    try:
                        # CORRETO: choices[0]["message"]["content"]
                        resposta_txt = r2.json()["choices"][0]["message"]["content"].strip()
                    except Exception:
                        resposta_txt = ""
                    visible_txt = _render_visible(resposta_txt).strip()
                else:
                    st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
            except Exception as e:
                st.error(f"Fallback (sem stream) erro: {e}")

        if not visible_txt:
            # 9b) retry sem o system extra (alguns modelos travam com system duplo)
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

        # 10) ExibiÃ§Ã£o final
        placeholder.markdown(visible_txt if visible_txt else "[Sem conteÃºdo]")

        # 11) Aviso de momento (nÃ£o bloqueia)
        try:
            viol = viola_momento(visible_txt, int(st.session_state.get("momento", 0)))
            if viol and st.session_state.get("app_bloqueio_intimo", False):
                st.info(f"âš ï¸ {viol}")
        except Exception:
            pass

        # 12) ValidaÃ§Ã£o semÃ¢ntica (entrada do user vs resposta) usando texto visÃ­vel
        if len(st.session_state.session_msgs) >= 1 and visible_txt and visible_txt != "[Sem conteÃºdo]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]
            alerta = verificar_quebra_semantica_openai(texto_anterior, visible_txt)
            if alerta:
                st.info(alerta)

        # 13) Salvar resposta SEMPRE (usa o texto visÃ­vel)
        salvar_interacao("assistant", visible_txt if visible_txt else "[Sem conteÃºdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt if visible_txt else "[Sem conteÃºdo]"})

        # 14) ReforÃ§o de memÃ³rias usadas (pÃ³s-resposta)
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

#























