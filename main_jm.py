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

# =========================
# CONFIG BÁSICA DO APP
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
                r = { (k or "").strip().lower(): (v or "") for k, v in row.items() }
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

# --- INICIALIZE AS VARIÁVEIS DE TEMPLATE ---
if "templates_jm" not in st.session_state:
    st.session_state.templates_jm = carregar_templates_planilha()
if "template_ativo" not in st.session_state:
    st.session_state.template_ativo = None
if "etapa_template" not in st.session_state:
    st.session_state.etapa_template = 0
# =====
# UTILIDADES: MEMÓRIAS / HISTÓRICO
# =====

# Observação 1: o código espera TAB_MEMORIAS = "memoria_jm"
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
    Importante: strings nesse formato são comparáveis lexicograficamente.
    """
    s = (s or "").strip()
    try:
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return s
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def carregar_memorias_brutas() -> Dict[str, List[dict]]:
    """
    Lê 'memoria_jm' (cabeçalho: tipo | conteudo | timestamp) e devolve
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
        st.warning(f"Erro ao carregar memórias: {e}")
        return {}

def persona_block(nome: str, buckets: dict, max_linhas: int = 8) -> str:
    """
    Monta bloco compacto da persona (ordena por prefixos úteis).
    Observação 2: usa tags no formato [mary], [janio].
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
    titulo = "Jânio" if nome in ("janio", "jânio") else "Mary" if nome == "mary" else nome.capitalize()
    return (
        f"{titulo}:\n- " + "\n- ".join(linha['conteudo'] for linha in linhas_ordenadas)
    ) if linhas_ordenadas else ""


def persona_block_temporal(nome: str, buckets: dict, ate_ts: str, max_linhas: int = 8) -> str:
    """
    Versão temporal do bloco de persona.
    Usa apenas memórias com timestamp <= ate_ts (se houver timestamp).
    Mantém compatibilidade com registros sem timestamp (são incluídos).
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
        # Se houver timestamp e corte temporal, exclui memórias "do futuro"
        if ts and ate_ts and ts > ate_ts:
            continue
        linhas.append((ts, c))

    # Ordena por timestamp crescente (strings ISO ordenam lexicograficamente)
    # Registros sem timestamp ("") ficam no início.
    linhas.sort(key=lambda x: x[0])

    # Pega as últimas N (mais recentes até o corte)
    ult = [c for _, c in linhas][-max_linhas:]
    if not ult:
        return ""

    titulo = "Jânio" if nome in ("janio", "jânio") else "Mary" if nome == "mary" else nome.capitalize()
    return f"{titulo}:\n- " + "\n- ".join(ult)


def carregar_resumo_salvo() -> str:
    """
    Busca o último resumo da aba 'perfil_jm' (cabeçalho: timestamp | resumo) com cache TTL.
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
    """
    Append no Sheets + atualiza cache local (sem reler) com backoff 429.
    (Observação 3: garante timestamp no padrão e corrige o else do cache.)
    """
    if not planilha:
        return
    try:
        aba = _ws(TAB_INTERACOES)
        if not aba:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Observação 3 (formato)
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
        st.error(f"Erro ao salvar interação: {e}")

# = BUSCA TEMPORAL DE MEMÓRIAS =
def buscar_status_persona_ate(persona_tag: str, momento_ts: str, buckets: dict) -> List[str]:
    """
    Busca os traços mais recentes da persona até o timestamp informado.
    persona_tag: ex: '[mary]' (será normalizado)
    momento_ts: timestamp limite (ex: '2025-08-21 16:04:00'; será normalizado)
    buckets: dict retornado por carregar_memorias_brutas()
    """
    tag = _normalize_tag(persona_tag)
    limite = _parse_ts(momento_ts)
    linhas = buckets.get(tag, [])

    # filtra até o limite e ordena por timestamp (strings já normalizadas)
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
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# =========================
# MEMÓRIA LONGA (Sheets + Embeddings/OpenAI opcional) — TIME-AWARE
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

def memoria_longa_buscar_topk(query_text: str, k: int = 3, limiar: float = 0.78, ate_ts=None):
    """
    Top-K memórias (time-aware). Usa embeddings se existir; senão, Jaccard simples.
    Se 'ate_ts' for informado (formato 'YYYY-MM-DD HH:MM:SS'), ignora registros com
    timestamp > ate_ts (ou seja, memórias 'do futuro' em relação ao histórico atual).
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

        # --- Corte temporal: ignora memórias mais novas que o corte (se fornecido)
        if ate_ts and row_ts and row_ts > ate_ts:
            continue  # strings no formato YYYY-MM-DD HH:MM:SS são comparáveis lexicograficamente

        # Similaridade por embedding (quando disponível), senão fallback lexical
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
    """Aumenta o score das memórias usadas (pequeno reforço) com backoff + correção de índices."""
    aba = _sheet_ensure_memoria_longa()
    if not aba or not textos_usados:
        return
    try:
        dados = _sheet_all_values_cached(TAB_ML)
        if not dados or len(dados) < 2:
            return
        headers = dados[0]  # cabeçalho é a primeira linha
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
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eê])\b", r"\b(at[eé] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade (elíptica)",
        "objetivo": "intimidade sugerida (fade-to-black) / pós-ato implícito",
        "permitidos": "beijos longos; proximidade forte; fade-to-black; manhã seguinte implícita",
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
    "🧠 Qwen3 Coder 480B (Together)": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
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
        regras += """
- Intimidade pode ser descrita com detalhes explícitos (sem necessidade de corte)."""
    else:
        regras += """
- Sem consumação em cena; foque em progressão coerente."""
    return prompt_base + "\n" + regras


def gerar_mary_sensorial(level: int = 2, n: int = 2, hair_on: bool = True) -> str:
    """
    Gera 1–3 frases sensoriais sobre Mary.
      level: 0=off, 1=leve, 2=marcado, 3=ousado
      n: quantidade de frases
      hair_on: garante ao menos 1 frase sobre os cabelos (negros, volumosos, levemente ondulados)
    """
    if level <= 0 or n <= 0:
        return ""

    # Base
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
        "O balanço dos quadris de Mary é quase cruel: entra na cabeça e não sai.",
        "Os seios acompanham a passada num movimento suave que acende o ambiente.",
        "O olhar de Mary encosta na pele de quem cruza com ela: quente, demorado, insinuante.",
        "O perfume fica na memória como um toque atrás da nuca.",
    ]

    # Frases específicas de cabelo (negros, volumosos, levemente ondulados)
    hair_leve = [
        "Os cabelos de Mary — negros, volumosos, levemente ondulados — descansam nos ombros e acompanham o passo.",
        "Os cabelos negros, volumosos e levemente ondulados moldam o rosto quando ela vira de leve.",
    ]
    hair_marcado = [
        "Cabelos negros, volumosos, levemente ondulados, fazem um arco quando ela vira o rosto, reforçando o balanço do corpo.",
        "Os cabelos, negros e volumosos, ondulam de leve a cada passada e criam uma moldura hipnótica.",
    ]
    hair_ousado = [
        "Os cabelos negros, volumosos e levemente ondulados deslizam pela clavícula como um toque que fica.",
        "O balanço dos cabelos negros — volumosos, levemente ondulados — marca o compasso do corpo de Mary.",
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

def encontrar_memorias_relevantes(pergunta, buckets):
    """
    Busca memórias relevantes conforme palavra-chave na pergunta do usuário.
    """
    # Palavras comuns que indicam pergunta factual
    keywords = [
        "nome", "integrante", "banda", "integrantes", "profissão", "rotina", "cargo", "ocupação",
        "onde", "quem", "quando", "idade", "universidade", "curso", "história", "grupo"
    ]
    relevantes = []
    pergunta_lc = (pergunta or "").lower()

    # Busca por tags de persona (ex: janio, mary) e keywords
    for tag, items in buckets.items():
        tag_limp = tag.strip("[]")
        # Se o nome/tag aparece na pergunta ou há palavras-chave, considera relevante
        if any(k in pergunta_lc for k in keywords) or tag_limp in pergunta_lc:
            relevantes.extend(items)
    return relevantes


def construir_prompt_com_narrador() -> str:
    memos = carregar_memorias_brutas()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS[0])
    proximo_nome = MOMENTOS.get(mdata.get("proximo", 0), MOMENTOS[0])["nome"]
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")

    # Camada sensorial de Mary (para o 1º parágrafo da cena)
    _sens_on = bool(st.session_state.get("mary_sensorial_on", True))
    _sens_level = int(st.session_state.get("mary_sensorial_level", 2))
    _sens_n = int(st.session_state.get("mary_sensorial_n", 2))
    mary_sens_txt = gerar_mary_sensorial(_sens_level, n=_sens_n) if _sens_on else ""

    # Histórico
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r['role']}: {r['content']}" for r in hist) if hist else "(sem histórico)"
    pergunta_user = hist[-1]["content"] if hist and hist[-1].get("role") == "user" else ""
    
    # Se quiser incluir bloco citacoes, precisa da função encontrar_memorias_relevantes
    bloco_citacoes = ""
    # Se implementar a busca de memórias factuais, descomente essa parte:
    # memorias_fatuais = encontrar_memorias_relevantes(pergunta_user, memos)
    # if memorias_fatuais:
    #     bloco_citacoes = "\n".join([
    #         f"- {m.get('conteudo', '')} (memória registrada em {m.get('timestamp','')})"
    #         for m in memorias_fatuais if m.get("conteudo")
    #     ])
    instrucoes_citacao = ""
    # if bloco_citacoes:
    #     instrucoes_citacao = (
    #         "\n### FATOS OBRIGATÓRIOS PARA RESPONDER A PERGUNTA DO USUÁRIO\n"
    #         "Responda de forma factual e cite explicitamente os dados abaixo na sua resposta. Não invente nem omita informações factuais relacionadas aos personagens da pergunta.\n"
    #         f"{bloco_citacoes}\n"
    #     )

    # CORTE TEMPORAL (até o timestamp da última interação)
    if hist:
        ate_ts = _parse_ts(hist[-1].get("timestamp", "")) if hist else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        ate_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Memória longa Top-K
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

    # Dossiê temporal
    dossie = []
    mary = persona_block_temporal("mary", memos, ate_ts, 8)
    janio = persona_block_temporal("janio", memos, ate_ts, 8)
    if mary:
        dossie.append(mary)
    if janio:
        dossie.append(janio)
    dossie_txt = "\n\n".join(dossie) if dossie else "(sem personas definidas)"

    flag_parallel = bool(st.session_state.get("no_coincidencias", True))

    # MONTAGEM DO PROMPT — peça que as aspas triplas fechem antes do .strip()
    prompt = f"""{instrucoes_citacao}
Você é o Narrador de um roleplay dramático brasileiro, foque em Mary e Jânio. Não repita instruções nem títulos.

### Dossiê (personas)
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
- Todas as cenas devem ser altamente sensoriais e físicas. Detalhe toques, apertos, tremores, suor, respiração, gradação do prazer até o limite do orgasmo e depois.
- As falas de Mary devem ser sensuais, realistas e explícitas, sem linguagem poética ou frases floreadas.
- Mary verbaliza desejo sexual, pede e orienta sexo e prazer sem pudor: frases diretas e sensoriais como "Me fode agora!", "Quero gozar!", "Tira minha roupa!", "Mete fundo!", "Estou molhada!", "Faz eu gritar de prazer!", "Estou gozando!", "Que delícia!", "Vai, não para!", "Me pega desse jeito!", "Põe a mão aqui!", "Goza dentro!", "Porra, tô gozando!", "Mais forte!" etc.
- Não use metáforas, floreios líricos ou reflexões profundas; priorize comandos, reações corporais e escancaradas do tesão e pós-tesão.
- No clímax, **obrigatoriamente inclua frases explosivas e altas de orgasmo**: Mary e Jânio devem gemer, gritar ou falar explicitamente seu prazer ("Estou gozando!", "Ahhh, que delícia!", "Goza dentro de mim!", "Mais forte!", etc.) antes de qualquer relaxamento ou carinho.
- O desfecho da cena sexual sempre inclui reação explosiva: gemidos altos, frases de gozo ditas em voz alta, corpo pulsando, tremores intensos; só depois, podem relaxar exaustos, suados, abraçados, comentando o prazer ("Nunca gozei desse jeito!", "Estou toda mole", "Não sinto as pernas", etc).
- Jamais encerre sem que pelo menos um dos personagens verbalize de forma explícita o orgasmo e a intensidade do prazer.
- Mary reage fisicamente com agarrões, mordidas, pedidos ousados e sinceros, sem vergonha. Nunca escreva desfecho seco ou cômico; sempre finalize mostrando o impacto corporal real, prazeroso e o orgasmo verbalizado.
### Camada sensorial — Mary (OBRIGATÓRIA no 1º parágrafo)
{mary_sens_txt or "- Comece com 1–2 frases curtas sobre o caminhar, olhar, perfume e cabelos (negros, volumosos, levemente ondulados) de Mary; pode mencionar o balanço suave dos seios sob o tecido (sem vulgaridade)."}
- Aplique essa camada ANTES do primeiro diálogo.
- Frases curtas, diretas, físicas; evite metáforas rebuscadas.
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
### Geografia & Montagem
- **Não force coincidências**: se não houver ponte clara (mensagem, convite, "ensaio 18h...", pedido do usuário), mantenha **Mary e Jânio em locais distintos** e utilize **montagem paralela** (A/B).
- **Comece cada bloco** com uma frase que **ancore lugar e hora** (exemplo: "UFES - corredor de Pedagogia, 9h15 - ..." ou "Terminal Laranjeiras, 9h18 - ..."). Não use títulos; escreva essa informação na **primeira frase** do parágrafo.
- **Se montagem paralela** (valor sugerido: {flag_parallel}):
  - Estruture em **2 blocos alternados**: primeiro Mary, depois Jânio (ou vice-versa), cada um em **seu lugar**.
  - Os blocos podem se "responder" por subtexto (mensagens, lembranças, sons à distância), mas **sem co-presença física**.
- **Se houver ponte plausível explícita**, pode convergir para co-presença ao final da cena (de forma plausível), **sem teletransporte**.
- **Sem ponte diegética explícita, um personagem não pode saber, afirmar ou reagir a fatos que só ocorreram no bloco do outro; se houver pressentimento ou ciúme, redija sem afirmar o fato. Exemplos de ponte: mensagem, foto/story, ligação, testemunha, encontro marcado - se existir, mostre isso na cena (exemplo: celular vibra e mostra um story)**.
- **Objetos diegéticos: caso a câmera não se encaixe na situação (encontro, banho, mar, revista), mostre a ação de guardar antes e ignore o objeto até a retomada; não descreva interação física com a câmera nesses contextos**.
### Formato OBRIGATÓRIO da cena
- **Inclua DIÁLOGOS diretos** com travessão (-), intercalados com ação e reação física/visual. Exemplo de travessão: - Ele disse ...
- Garanta **pelo menos 2 falas de Mary e 2 de Jânio** (quando ambos estiverem na cena).
- **Não inclua pensamentos internos em itálico, reflexões internas ou monólogos subjetivos dos personagens.**
- Não escreva blocos finais de créditos, microconquistas, resumos ou ganchos. Apenas narração e interação direta.
- Mostre somente ações, gestos, expressões do ambiente, clima corporal e diálogos.
- Sem títulos de seção, microconquista ou gancho, nem qualquer nota meta ao final.
### Regra de saída
- Narre em **terceira pessoa**; nunca fale com "você".
- Produza uma cena fechada e natural, sem inserir comentários externos ou instruções.
""".strip()
    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# FILTROS DE SAÍDA
# =========================

def render_tail(t: str) -> str:
    if not t:
        return ""
    # remove rótulos meta e blocos <think>
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
        return 3  # explícito
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
    # Se extrapolar o máximo definido, não cortar por padrão (liberar NSFW).
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

# Inicialização dos estados de sessão (inclusive dos templates)
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
    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=3, key="nsfw_max_level")

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
    
        # Inicia o roteiro e já dispara a 1ª etapa (se existir)
        if st.button("Iniciar roteiro", key="btn_iniciar_roteiro"):
            st.session_state.template_ativo = roteiro_escolhido
            st.session_state.etapa_template = 0
            if etapas:
                comando = etapas[0]
                salvar_interacao("user", comando)
                st.session_state.session_msgs.append({"role": "user", "content": comando})
                st.session_state["_trigger_input"] = comando  # dispara geração
    
        # Progresso / próxima etapa
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
                    st.session_state["_trigger_input"] = comando  # dispara geração
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
                    if ok:
                        count += 1
        st.success(f"{count} memórias biográficas reforçadas na memória longa!")

    st.markdown("### 🧩 Histórico no prompt")
    st.slider("Interações do Sheets (N)", 10, 30, value=int(st.session_state.get("n_sheet_prompt", 15)), step=1, key="n_sheet_prompt")



   
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
# Permite que botões do roteiro disparem a geração
if not entrada:
    entrada = st.session_state.pop("_trigger_input", None)


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

    # 1) Salva a entrada e mantém histórico de sessão
    salvar_interacao("user", str(entrada))
    st.session_state.session_msgs.append({"role": "user", "content": str(entrada)})

    # 2) Constrói prompt principal
    prompt = construir_prompt_com_narrador()

    # 3) Histórico curto (somente sessão atual; o prompt já inclui últimas do sheet)
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
        st.error("A chave de API do provedor selecionado não foi definida em st.secrets.")
        st.stop()

    # 5) Mensagens
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

    # 6) Render / Filtro de saída
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

        # 7) Reforço antecipado: memórias que ENTRARAM no prompt (topk + recorrentes)
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
                                placeholder.markdown(_render_visible(resposta_txt) + "▌")
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

        # 10) Exibição final
        placeholder.markdown(visible_txt if visible_txt else "[Sem conteúdo]")

        # 11) Aviso de momento (não bloqueia)
        try:
            viol = viola_momento(visible_txt, int(st.session_state.get("momento", 0)))
            if viol and st.session_state.get("app_bloqueio_intimo", False):
                st.info(f"⚠️ {viol}")
        except Exception:
            pass

        # 12) Validação semântica (entrada do user vs resposta) usando texto visível
        if len(st.session_state.session_msgs) >= 1 and visible_txt and visible_txt != "[Sem conteúdo]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]
            alerta = verificar_quebra_semantica_openai(texto_anterior, visible_txt)
            if alerta:
                st.info(alerta)

        # 13) Salvar resposta SEMPRE (usa o texto visível)
        salvar_interacao("assistant", visible_txt if visible_txt else "[Sem conteúdo]")
        st.session_state.session_msgs.append({"role": "assistant", "content": visible_txt if visible_txt else "[Sem conteúdo]"})

        # 14) Reforço de memórias usadas (pós-resposta)
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





































