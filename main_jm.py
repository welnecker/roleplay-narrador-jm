# roleplay_narrador_jm_clean_messages.py
# ---------------------------------------------------------------------------------
# Objetivo: manter toda a ESTRUTURA DE MODELOS do app original, mas
# ENVIAR APENAS MENSAGENS MÍNIMAS aos modelos (sem prompts auxiliares, fases, etc.).
# - System único (persona Mary)
# - Histórico user/assistant como está
# - Persistência só em "interacoes_jm"
# ---------------------------------------------------------------------------------

import json
import re
import random
from datetime import datetime
from typing import Dict, List, Any
import gspread
import requests
import streamlit as st
from gspread.exceptions import APIError, GSpreadException
from oauth2client.service_account import ServiceAccountCredentials
from huggingface_hub import InferenceClient

# ==== Memória episódica leve (planilha única: interacoes_jm) ====
import re, random
from typing import List, Dict, Optional

def _sheet_to_events_ultima_sessao(ws_values: List[List[str]]) -> List[Dict]:
    """Converte interacoes_jm -> lista de eventos (role=user/assistant) da ÚLTIMA session_id."""
    if not ws_values or len(ws_values) < 2:
        return []
    linhas = [r for r in ws_values[1:] if len(r) >= 6]
    if not linhas:
        return []
    last_sid = (linhas[-1][1] or "").strip()
    sess = [r for r in linhas if (r[1] or "").strip() == last_sid and (r[4] or "").strip().lower() in ("user","assistant")]
    evs = []
    for idx, r in enumerate(sess, start=1):
        evs.append({"turn": idx, "ts": r[0], "sid": r[1], "role": r[4].strip().lower(), "text": (r[5] or "").strip()})
    return evs

# ---- Catálogo de eventos (ampliável) ----
EVENTS = {
    "primeira_noite_motel": {
        "keywords": ["motel status","status motel","quarto do motel","primeira vez","dor inicial","penetra","penetrou"],
        "select": "first_strong",
        "recall": "— Nossa, Jânio… você está tão viril hoje… me lembra a nossa primeira noite no Status."
    },
    "primeiro_beijo_praia": {
        "keywords": ["primeiro beijo","beijou","beijaram","beijo salgado","praia de camburi","areia","calçadão"],
        "select": "first_strong",
        "recall": "— Esse seu jeito agora… me lembra o nosso primeiro beijo na praia."
    },
    "onde_se_conheceram": {
        "keywords": ["primeiro encontro","quando nos conhecemos","a primeira vez que te vi","como te conheci","te vi no"],
        "select": "first",
        "recall": "— Engraçado… às vezes eu ainda sinto o friozinho de quando a gente se conheceu."
    },
    "ciumes_ricardo": {
        "keywords": ["ciúme","ciumes","ciumento","grita","gritou","gritando","bate-boca","discussão","ricardo"],
        "select": "canonical_score_oldest",
        "recall": "— Ei… lembra quando você levantou a voz por ciúmes do Ricardo? A gente combinou de conversar, não gritar."
    },
    "reconciliacao": {
        "keywords": ["desculpa","perdão","me excedi","errei","prometo não gritar","vamos conversar"],
        "select": "after(ciumes_ricardo)",
        "recall": "— Obrigada por ter pedido desculpas aquele dia… eu lembro."
    },
}
PRIORITY = ["primeira_noite_motel","primeiro_beijo_praia","onde_se_conheceram","ciumes_ricardo","reconciliacao"]

def _score_kw(text: str, kws: List[str]) -> int:
    t = text.lower()
    return sum(1 for k in kws if k in t)

def _pick_episode(evs: List[Dict], ev_key: str, base: Optional[Dict]=None) -> Optional[Dict]:
    spec = EVENTS[ev_key]; kws = spec["keywords"]; sel = spec["select"]
    cands = [(e, _score_kw(e["text"], kws)) for e in evs]
    cands = [(e, s) for (e, s) in cands if s > 0]
    if not cands:
        return None
    if sel == "first":
        return sorted((e for e,_ in cands), key=lambda e: e["turn"])[0]
    if sel == "first_strong":
        strong = [e for (e,s) in cands if s >= 2] or [e for (e,s) in cands if s >= 1]
        return sorted(strong, key=lambda e: e["turn"])[0]
    if sel == "canonical_score_oldest":
        cands.sort(key=lambda es: (-es[1], es[0]["turn"]))
        return cands[0][0]
    if sel.startswith("after(") and base:
        dep = sel[6:-1]
        if dep not in EVENTS or "turn" not in base:
            return None
        turn0 = base["turn"]
        after = [(e,s) for (e,s) in cands if e["turn"] > turn0]
        if not after:
            return None
        after.sort(key=lambda es: (es[0]["turn"], -es[1]))
        return after[0][0]
    return None

def _compose_recall(evs: List[Dict]) -> Optional[str]:
    cache: Dict[str, Dict] = {}
    for key in PRIORITY:
        base = cache.get("ciumes_ricardo")
        ep = _pick_episode(evs, key, base=base)
        if ep:
            cache[key] = ep
            if EVENTS[key]["select"].startswith("after(") and not base:
                continue
            return EVENTS[key]["recall"]
    return None

def _should_recall(user_msg: str, base_prob: float = 0.20) -> bool:
    t = (user_msg or "").lower()
    gatilhos = ["motel","status","primeiro beijo","praia","ciúme","ciumes","ricardo","conhecemos"]
    p = base_prob + (0.18 if any(g in t for g in gatilhos) else 0.0)
    return random.random() < min(max(p, 0.0), 0.6)

def _cooldown_ok(min_gap_turns: int = 6) -> bool:
    last = st.session_state.get("last_recall_turn", -9999)
    cur  = len(st.session_state.get("chat", []))
    if cur - last >= min_gap_turns:
        st.session_state["last_recall_turn"] = cur
        return True
    return False

def maybe_inject_spontaneous_recall(answer: str, user_msg: str, ws_values: List[List[str]]) -> str:
    """Opcionalmente prefixa a fala da Mary com uma recordação curta e natural."""
    if not _should_recall(user_msg) or not _cooldown_ok():
        return answer
    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs:
        return answer
    recall = _compose_recall(evs)
    if not recall:
        return answer
    low = (answer or "").lower()
    if any(x in low for x in ["primeira noite","status","primeiro beijo","praia","ciúmes","ricardo","conhecemos"]):
        return answer
    ans = (answer or "").strip()
    return f"{recall}\n\n{ans}" if ans else recall

# ---- Puxão de orelha ao detectar grito/ciúmes no input atual ----
_GRITO_KEYS  = ["grita", "gritou", "gritando", "gritar"]
_CIUMES_KEYS = ["ciúme", "ciumes", "ciumento", "ricardo"]
_UPPER_RE = re.compile(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{3,}!?")

def _parece_grito(texto: str) -> bool:
    t = (texto or "").strip(); tl = t.lower()
    if any(k in tl for k in _GRITO_KEYS): return True
    return bool(_UPPER_RE.search(t)) and "!" in t

def _parece_ciumes(texto: str) -> bool:
    tl = (texto or "").lower()
    return any(k in tl for k in _CIUMES_KEYS)

def _buscar_incidente_ciumes(evs: List[Dict]) -> Optional[Dict]:
    episodios = []
    for e in evs:
        score = 0; tl = e["text"].lower()
        if any(w in tl for w in _GRITO_KEYS):  score += 2
        if any(w in tl for w in _CIUMES_KEYS): score += 2
        if "ricardo" in tl:                    score += 1
        if score > 0:
            episodios.append((e, score))
    if not episodios:
        return None
    episodios.sort(key=lambda es: (-es[1], es[0]["turn"]))
    return episodios[0][0]

def _resumo_curto_incidente(ep: Dict) -> str:
    txt = (ep.get("text") or "").lower()
    partes = []
    if "ricardo" in txt: partes.append("com o Ricardo")
    if any(w in txt for w in ("grita","gritou","gritando")): partes.append("quando você gritou comigo")
    if not partes: partes.append("naquela discussão")
    return " " + " ".join(partes)

def inject_rebuke_if_needed(answer: str, user_msg: str, ws_values: List[List[str]]) -> str:
    if not (_parece_grito(user_msg) or _parece_ciumes(user_msg)):
        return answer
    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs:
        return answer
    ep = _buscar_incidente_ciumes(evs)
    if not ep:
        return answer
    pista = _resumo_curto_incidente(ep)
    prefixo = f"— Você está levantando a voz de novo…{pista}. Vamos conversar sem gritar, por favor."
    ans = (answer or "").strip()
    return f"{prefixo}\n\n{ans}" if ans else prefixo

# ---- Perguntas de memória (“lembra… ?”) → injeta evidências reais no prompt ----
_MEM_TRIGS = ["lembra","lembrar","nossa primeira","primeira vez","quando","naquele dia"]

def is_memoria_query(txt: str) -> bool:
    t = (txt or "").lower()
    return any(k in t for k in _MEM_TRIGS)

def find_evidence_snippets(ws_values: List[List[str]], user_msg: str, max_snips: int = 3) -> List[str]:
    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs:
        return []
    picks: List[str] = []
    ep_motel = _pick_episode(evs,"primeira_noite_motel");      ep_beijo = _pick_episode(evs,"primeiro_beijo_praia")
    ep_conhe = _pick_episode(evs,"onde_se_conheceram");        ep_ciume = _pick_episode(evs,"ciumes_ricardo")
    if ep_motel: picks.append(ep_motel["text"])
    if ep_beijo: picks.append(ep_beijo["text"])
    if ep_conhe: picks.append(ep_conhe["text"])
    if ep_ciume:
        picks.append(ep_ciume["text"])
        ep_recon = _pick_episode(evs,"reconciliacao", base=ep_ciume)
        if ep_recon: picks.append(ep_recon["text"])
    uniq = []
    for p in picks:
        if p not in uniq:
            uniq.append(p)
    return uniq[:max_snips]

def wrap_messages_with_memory_if_needed(messages: List[Dict[str,str]], user_msg: str, ws_values: List[List[str]]) -> List[Dict[str,str]]:
    if not is_memoria_query(user_msg):
        return messages
    snips = find_evidence_snippets(ws_values, user_msg)
    if not snips:
        return messages
    bloco = "Contexto anterior (trechos reais do nosso histórico):\n" + "\n".join(f"• {s}" for s in snips)
    base = [{"role":"user","content": bloco}]
    return base + messages

st.set_page_config(page_title="Narrador JM — Clean Messages", page_icon="🎬")

# ------------------------
# Presets de Modo de Fala
# ------------------------
FALA_PRESETS: Dict[str, Dict[str, str]] = {
    "Oral": {
        "rule": "— Sexy: carga erótica verbal com descrição gráfica; NSFW liberado.",
        "example": "tira minha calcinha; Chupa meu clitóris; me faz gozar; vou gozar....chupa gostoso "
    },
    "Boquete": {
        "rule": "— Sexy: carga erótica verbal com descrição gráfica; NSFW liberado.",
        "example": "tira essa calça; deixa eu chupar seu pau; como é grande e grosso...hummm; goza na minha boca... "
    },
    "cavalga": {
        "rule": "— Sexy: carga erótica verbal com descrição gráfica; NSFW liberado.",
        "example": "deita amor...vou montar em você; fode gostoso; calma..não goza ainda; tá gostoso?.. pega na minha bunda; isso...goza..."
    },
    "quatro": {
        "rule": "— Sexy: carga erótica verbal com descrição gráfica; NSFW liberado.",
        "example": "Vem...mete por trás; porra...que delícia; empurra tudo...humm; calma..não goza ainda; deixa eu rebolar...fica parado; isso...goza...."
    },
    "Ciumenta": {
        "rule": "— Ciumenta: marca território com elegância; perguntas diretas; limites claros; nada de insultos.",
        "example": "Quem é essa piranha?."
    },
}

CARINHOSA_FRASES = [
    '— "Você é uma graça, sabia?"',
    '— "É tão bonitinho quando você fica inseguro."',
    '— "Vem cá… me dá um abraço."',
    '— "Eu gosto de ouvir você assim, sem pressa."',
    '— "Fica perto de mim um pouquinho?"',
    '— "Tá tudo bem. Eu tô aqui."',
    '— "Adoro quando você confia em mim."',
    '— "Deixa eu cuidar de você agora?"',
]

_TRIG_CARINHO = re.compile(
    r"(?i)\b(n[ãa]o sei|talvez|acho que|t[oô] com medo|insegur|desculp|ser[aá] que|pode ser)\b|[?]\s*$"
)

def inject_carinhosa(texto: str, user_text: str, ativo: bool) -> str:
    if not ativo or not texto.strip():
        return texto
    gatilho = bool(_TRIG_CARINHO.search(user_text or "")) or (random.random() < 0.25)
    if not gatilho:
        return texto
    frase = random.choice(CARINHOSA_FRASES)
    # mantém estilo: novo parágrafo curto, 1–2 frases
    sep = "\n\n" if not texto.endswith("\n") else "\n"
    return (texto.rstrip() + f"{sep}{frase}").strip()


def build_fala_block(modos: List[str]) -> str:
    if not modos:
        return ""
    linhas = ["[MODO DE FALA — Mary]", "— Modos ativos: " + ", ".join(modos) + "."]
    for m in modos:
        if m in FALA_PRESETS:
            linhas.append(FALA_PRESETS[m]["rule"])
    linhas.append("— Responda mantendo este(s) tom(ns) em falas e narração de Mary.")
    return "\n".join(linhas)




# ================================
# UI — Roleplay comercial (cards)
# ================================
with st.sidebar:
    st.subheader("Roleplay comercial")

    st.session_state.setdefault("user_name", "")
    st.session_state["user_name"] = st.text_input(
        "Seu nome (como o personagem vai se referir a você)",
        value=st.session_state["user_name"],
        max_chars=40,
    )
    st.session_state.setdefault("scenario_init", "")
    st.session_state.setdefault("plot_init", "")
    st.session_state["scenario_init"] = st.text_area(
        "Cenário inicial",
        value=st.session_state["scenario_init"],
        height=80,
        placeholder="Ex.: Final de tarde na Praia de Camburi; quiosque perto do calçadão…",
    )
    st.session_state["plot_init"] = st.text_area(
        "Enredo inicial",
        value=st.session_state["plot_init"],
        height=80,
        placeholder="Ex.: Mary encontra o usuário após um mal-entendido com Ricardo…",
    )

    # Presets de clima inicial (equilíbrio)
    roleplay_modes = {
        "Cool": "Só conversas cotidianas, curiosidade e aproximação inicial.",
        "Mellow": "Clima levemente sugestivo, insinuações, mas sem abordagem sexual direta.",
        "Standard": "Flerte e química surgem naturalmente, mas nada explícito sem gatilho do usuário.",
        "Real": "Reage de maneira realista ao contexto, histórico e ações do usuário.",
        "Hot": "Flerte aberto, direto e mais ousado já de início, sem ser explícito de cara.",
        "Inner Secret": "Desejos, fetiches e fantasias revelados gradualmente conforme a aproximação.",
        "Explicit": "Sexualidade aberta e explícita permitida desde o início do diálogo, sem restrição de conteúdo."
    }
    st.session_state.setdefault("equilibrio_modo", "Standard")
    modo_eq = st.selectbox("Clima de início", list(roleplay_modes.keys()), index=2)
    st.session_state["equilibrio_modo"] = modo_eq
    st.caption(roleplay_modes.get(modo_eq, ""))  # Mostra a descrição do preset escolhido

with st.sidebar:
    st.markdown("**Modo de fala da Mary**")
    st.session_state.setdefault("fala_mods", [])
    # 5 caixas de seleção (podem combinar)
    mods_escolhidos: List[str] = []
    if st.checkbox("Boquete", value=True, key="fala_Boquete"):
        mods_escolhidos.append("Boquete")
    if st.checkbox("Oral", key="fala_Oral"):
        mods_escolhidos.append("Oral")
    if st.checkbox("cavalga", key="fala_cavalga"):
        mods_escolhidos.append("cavalga")
    if st.checkbox("quatro", key="fala_quatro"):
        mods_escolhidos.append("quatro")
    if st.checkbox("Ciumenta", key="fala_ciumenta"):
        mods_escolhidos.append("Ciumenta")
    if st.checkbox("Carinhosa", key="fala_carinhosa"):
        mods_escolhidos.append("Carinhosa")
    
    st.session_state["fala_mods"] = mods_escolhidos
    
    # (Opcional) dicas rápidas do tom atual
    if mods_escolhidos:
        exemplos = [
            FALA_PRESETS.get(m, {}).get("example")
            for m in mods_escolhidos if FALA_PRESETS.get(m)
        ]
        exemplos = [e for e in exemplos if e]
        if exemplos:
            st.caption("Exemplos de abertura (tom atual): " + " / ".join(exemplos[:3]))


# --------- Filtro: silenciar falas/mensagens de Jânio (robusto) ---------
def _is_quoted_or_bulleted(line: str) -> bool:
    s = line.lstrip()
    return (
        s.startswith('—') or
        s.startswith('"') or s.startswith('“') or
        s.startswith('*"') or s.startswith('*“') or
        s[:2] in ('- ', '* ') or
        (len(s) > 2 and s[0].isdigit() and s[1] in '.)')
    )

# --------- Silenciar falas/mensagens de Jânio (sem trocar o nome) ---------
JANIO_SPEECH = re.compile(r'(?i)^\s*(?:jânio|janio)\s*:\s+.*$')
JANIO_ATTR_QUOTE = re.compile(r'(?i)^\s*—\s*[\"“].*[\"”].*(?:—\s*)?(?:disse|fala|respondeu)\s+j[âa]nio\b.*$')
MSG_HEADER = re.compile(r'(?i)^\s*\**\s*mensagens? de j[âa]nio\s*\**\s*$')

def silenciar_janio(txt: str) -> str:
    if not txt:
        return txt
    out: List[str] = []
    in_msg_block = False
    for line in txt.splitlines():
        raw = line.strip()

        # Bloco "Mensagem de Jânio"
        if MSG_HEADER.match(raw):
            out.append('_Uma notificação de Jânio chega ao celular de Mary._')
            in_msg_block = True
            continue
        if in_msg_block:
            if not raw:   # linha vazia encerra o bloco
                in_msg_block = False
                continue
            if raw.startswith('—') or _is_quoted_or_bulleted(line):
                out.append('_[Conteúdo de Jânio omitido]_')
                continue
            # qualquer outra linha dentro do bloco é omitida
            out.append('_[Conteúdo de Jânio omitido]_')
            continue

        # Linha direta "Jânio: ..." ou citação atribuída a Jânio
        if JANIO_SPEECH.match(line) or JANIO_ATTR_QUOTE.match(line):
            out.append('_[Conteúdo de Jânio omitido]_')
            continue

        # Menções a Jânio (presença/ação) passam normalmente
        out.append(line)
    return "\n".join(out)

# --------- Filtro extra: impedir fala pelo usuário ---------
def silenciar_fala_do_usuario(txt: str) -> str:
    uname = (st.session_state.get("user_name") or "").strip()
    pats: List[re.Pattern] = []
    if uname:
        pats.append(re.compile(rf"(?im)^\s*{re.escape(uname)}\s*:\s*.*$"))
        pats.append(re.compile(rf"(?im)^.*—\s*[\"“].*[\"”]\s*—\s*(?:disse|fala|respondeu)\s+{re.escape(uname)}\b.*$"))
    pats.append(re.compile(r"(?im)^\s*(?:você|voce|usuário|usuario)\s*:\s*.*$"))
    out: List[str] = []
    for line in txt.splitlines():
        if any(p.match(line) for p in pats):
            out.append("_[A fala do usuário é escrita pelo próprio usuário.]_")
            continue
        out.append(line)
    return "\n".join(out)

# Combina filtros (ordem importa)
def apply_filters(txt: str) -> str:
    return silenciar_fala_do_usuario(silenciar_janio(txt))

# =================================================================================
# Config Planilha
# =================================================================================
PLANILHA_ID = (
    st.secrets.get("JM_SHEET_ID")
    or st.secrets.get("SPREADSHEET_ID")
    or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
).strip()
TAB_INTERACOES = "interacoes_jm"

# =================================================================================#
# Modelos (mantidos como no app original do Janio)
# =================================================================================
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
    "Qwen2.5 7B Instruct (HF)": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai-org: GLM-4.5-Air (HF)": "zai-org/GLM-4.5-Air",
    "Mixtral 8x7B Instruct (HF)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "DeepSeek R1 (HF)": "deepseek-ai/DeepSeek-R1",
}

# LM Studio (padrão: base local; pode ser editado no sidebar)
DEFAULT_LMS_BASE_URL = (
    st.secrets.get("LMS_BASE_URL") or "http://127.0.0.1:1234/v1"
).rstrip("/")

@st.cache_data(ttl=15, show_spinner=False)
def _lms_models_dict(base_url: str) -> Dict[str, str]:
    try:
        r = requests.get(f"{base_url}/models", timeout=8)
        r.raise_for_status()
        data = r.json()
        return {f"{m['id']} (LM Studio)": m["id"] for m in data.get("data", []) if m.get("id")}
    except Exception:
        return {"<digite manualmente> (LM Studio)": "llama-3.1-8b-instruct"}

# =================================================================================
# Persona — system mínimo otimizado (com dinâmicas avançadas e filtros de repetição)
# =================================================================================
PERSONA_MARY = """
[IDENTIDADE]
Nome: Mary Massariol (20). Engenharia Civil — UFES. Cabelos negros volumosos, olhos verdes, corpo atlético.
Mora com a mãe, Joselina, ap. 202 — Rua Beethoven, Laranjeiras. Usa moto 500cc.
Traços: inteligente, romântica, gosta de praia e flerte; não tolera grosseria/cantada barata; quer respeito e liberdade.

[CONHECIMENTO E MEMÓRIA]
— Não conhece o usuário previamente; aprende somente pelo diálogo/ações na cena.
— Sem onisciência: nunca descreva pensamentos/intenções do usuário; apenas o que ele diz/faz ou sinais visíveis.
— Mantenha coerência estrita com o histórico (memória recente e eventos passados).

[RELACIONAMENTO / LIMITES]
— Virgem por escolha; só se entrega com confiança/desejo construídos em cena.
— Rejeita traição; entrega total apenas por vontade própria.
— Flertes e avanços seguem a progressão: flerte > toque leve > beijo > carícia > toque íntimo > sexo (consentido).

[CLIMA E CONDUTA]
— Ajuste o tom ao usuário/preset; provoque com leveza e respeito.
— Após cenas intensas, traga respiros (cuidado, humor, cotidiano) e transições naturais de lugar/tempo.
— Não escreva falas do usuário; não ofereça menus/opções; não decida por ele.

[LOCAIS COMUNS]
Clube Serra Bella (balada), Motel Status (privacidade), Café Oregon (encontros), Praia de Camburi (lazer), Enseada do Suá (ousadia).

[RESTRIÇÕES DE ESTILO — OBRIGATÓRIO]
— NÃO usar “Foto:”, “Legenda:” ou formatos de postagem/ensaio fotográfico.
— Linguagem natural, direta e contemporânea; sem metáforas rebuscadas nem comparações acadêmicas.
— Foque em diálogo e reações físicas/emocionais percebidas por Mary (olhar, voz, toque, calor, cheiro pontual).
— No máximo 5 parágrafos por turno; até 2 frases por parágrafo; ~30% mais concisa que o padrão.
— Um traço sensorial/físico por parágrafo; evite listas e repetição.

[CONTINUIDADE]
— Personagens só retornam por gatilho do usuário.
— Marque passagem de tempo/contexto quando necessário (“mais tarde…”, “na praia…”, “novo ambiente…”).
— Sem loops: efeitos são cumulativos e lembrados nas próximas interações.

[REGRAS COMERCIAIS]
— Mary fala e sente apenas por si.
— Trate o usuário pelo nome informado.
— Respeite o cenário/enredo vigentes até novo gatilho claro.
"""

# =================================================================================
# Conector Google Sheets (apenas interacoes_jm)
# =================================================================================
def _load_google_creds_dict() -> dict:
    raw = st.secrets.get("GOOGLE_CREDS_JSON")
    if raw is None:
        raise RuntimeError("Faltando GOOGLE_CREDS_JSON no secrets.")
    creds = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if isinstance(creds.get("private_key"), str):
        creds["private_key"] = creds["private_key"].replace("\\n", "\n")
    return creds

@st.cache_resource(show_spinner=False)
def _open_ws_interacoes():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(_load_google_creds_dict(), scope)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(PLANILHA_ID)
    return sh.worksheet(TAB_INTERACOES)

try:
    WS_INTERACOES = _open_ws_interacoes()
except Exception as e:
    WS_INTERACOES = None
    st.error(f"Não foi possível abrir a aba '{TAB_INTERACOES}'. Verifique permissões/ID. Detalhe: {e}")

def salvar_interacao(ts: str, session_id: str, provider: str, model: str, role: str, content: str):
    if not WS_INTERACOES:
        return
    try:
        WS_INTERACOES.append_row(
            [ts, session_id, provider, model, role, content],
            value_input_option="USER_ENTERED"
        )
    except (APIError, GSpreadException) as e:
        st.warning(f"Falha ao salvar interação: {e}")

# =================================================================================
# Carregamento de histórico persistente (últimas interações)
# =================================================================================

from typing import Tuple

def carregar_ultimas_interacoes(n_min: int = 5) -> list[dict]:
    """Carrega ao menos n_min interações da última sessão registrada na aba interacoes_jm.
    Retorna uma lista de mensagens no formato [{"role": "user|assistant", "content": str}, ...].
    Limita implicitamente a 30 ao aplicar na tela."""
    if not WS_INTERACOES:
        return []
    try:
        valores = WS_INTERACOES.get_all_values()  # [[timestamp, session_id, provider, model, role, content], ...]
        if not valores or len(valores) < 2:
            return []
        linhas = [r for r in valores[1:] if len(r) >= 6]
        if not linhas:
            return []
        # Usa a última session_id registrada na planilha
        last_sid = linhas[-1][1].strip()
        sess = [r for r in linhas if r[1].strip() == last_sid and r[4] in ("user", "assistant")]
        if not sess:
            return []
        recortes = sess[-max(n_min, 1):]
        chat = [{"role": r[4].strip(), "content": (r[5] or "").strip()} for r in recortes]
        return chat
    except Exception:
        return []

def resumir_chat(chat_msgs: list[dict], call_model_func, model_id: str) -> str:
    """
    Usa o modelo LLM para gerar um resumo robusto das mensagens antigas, preservando cenário e contexto.
    """
    texto = "\n".join(
        f"[{m['role']}]: {m['content']}"
        for m in chat_msgs if m['role'] in ('user', 'assistant') and m['content'].strip()
    )
    prompt = (
        "Resuma o diálogo abaixo destacando sempre: 1) o local/situação atual dos personagens, "
        "2) mudanças de ambiente ou cenário (se ocorreram), 3) estado emocional predominante, "
        "e 4) decisões ou fatos relevantes para o desenvolvimento da história. "
        "Seja breve e preserve o tom do roleplay. Se houve saída do motel, viagem, chegada a outro lugar, cite explicitamente.\n\n"
        + texto
    )
    resumo = call_model_func(model_id, [{"role": "user", "content": prompt}])
    return resumo.strip()

# =============================================================================
# Build minimal messages (override) — injeta nome do usuário, cenário e enredo + clima inicial
# =============================================================================
def build_minimal_messages(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    import re

    # 1) Ler inputs da UI (opcionais)
    user_name  = (st.session_state.get("user_name") or "").strip()
    scenario   = (st.session_state.get("scenario_init") or "").strip()
    plot       = (st.session_state.get("plot_init") or "").strip()
    fala_mods  = st.session_state.get("fala_mods") or []
    clima_modo = st.session_state.get("equilibrio_modo", "Standard")

    # 2) Catálogo de climas (mesmo usado no sidebar)
    roleplay_modes = {
        "Cool":        "Só conversas cotidianas, curiosidade e aproximação inicial.",
        "Mellow":      "Clima levemente sugestivo, insinuações, sem abordagem sexual direta.",
        "Standard":    "Flerte e química surgem naturalmente; nada explícito sem gatilho do usuário.",
        "Real":        "Reage de forma realista ao contexto e histórico; ousadia só com sinais claros do usuário.",
        "Hot":         "Flerte aberto e direto, sem ser explícito de cara; respeite limites do contexto.",
        "Inner Secret":"Desejos/fantasias revelados gradualmente conforme a aproximação.",
        "Explicit":    "Sexualidade explícita permitida; ainda assim respeite transição e consentimento."
    }

    # 3) Higienização leve para não inflar o system
    def _clean(s: str, maxlen: int) -> str:
        s = re.sub(r"\s+", " ", s or "").strip()
        return s[:maxlen]

    user_name = _clean(user_name, 80)
    scenario  = _clean(scenario, 600)
    plot      = _clean(plot, 600)

    # 4) Blocos auxiliares
    def build_equilibrio_block(modo: str) -> str:
        desc = roleplay_modes.get(modo, roleplay_modes["Standard"])
        return (
            f"[CLIMA INICIAL — {modo}]\n"
            f"— {desc}\n"
            "— Siga este clima até que o usuário provoque mudança clara."
        )

    def build_fala_block(modos: List[str]) -> str:
        if not modos:
            return ""
        linhas = ["[MODO DE FALA — Mary]", "— Modos ativos: " + ", ".join(modos) + "."]
        for m in modos:
            if m == "Ciumenta":
                linhas.append("— Ciumenta: marca território com elegância; perguntas diretas; nada de insulto.")
            elif m == "Carinhosa":
                linhas.append("— Carinhosa: acolhe, reduz tensão; reforça segurança/afeto.")
            # (demais presets explícitos/NSFW já são tratados pela persona e pelo clima escolhido)
        linhas.append("— Responda mantendo esse(s) tom(ns) nas falas e na narração.")
        return "\n".join(linhas)

    equilibrio_block = build_equilibrio_block(clima_modo)
    fala_block       = build_fala_block(fala_mods)

    extras = []
    if user_name:
        extras.append(f"[USUÁRIO]\n— Trate o usuário pelo nome: {user_name}.")
    if scenario or plot:
        extras.append("[CENÁRIO/ENREDO INICIAL]")
        if scenario: extras.append(f"— Cenário: {scenario}")
        if plot:     extras.append(f"— Enredo: {plot}")

    # 5) Montar o system final (persona já contém as restrições de estilo e o ban a 'Foto/Legenda')
    parts = [equilibrio_block]
    if fala_block:
        parts.append(fala_block)
    parts.append(PERSONA_MARY)            # <- use a versão otimizada que alinhamos
    if extras:
        parts.append("\n".join(extras))
    system_text = "\n\n".join(parts)

    # 6) Chat mínimo (sem mensagens 'system' redundantes; preserva ordem user/assistant)
    mensagens_chat = [m for m in chat if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
    msgs.extend({"role": m["role"], "content": m["content"].strip()} for m in mensagens_chat)

    return msgs

# =================================================================================
# Chamadas por provedor — sem parâmetros extras
# =================================================================================

def call_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
        "HTTP-Referer": st.secrets.get("APP_URL", ""),
        "X-Title": st.secrets.get("APP_TITLE", "Narrador JM"),
    }
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_together(model: str, messages: List[Dict[str, str]]) -> str:
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {st.secrets.get('TOGETHER_API_KEY', '')}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_lmstudio(base_url: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_huggingface(model: str, messages: List[Dict[str, str]]) -> str:
    # Converte chat → prompt simples
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            parts.append(f"[USER]\n{content}\n")
        else:
            parts.append(f"[ASSISTANT]\n{content}\n")
    prompt = "\n".join(parts) + "\n[ASSISTANT]\n"

    client = InferenceClient(api_key=st.secrets.get("HUGGINGFACE_API_KEY", ""))
    # Parâmetros mínimos (sem temperature/top_p etc.)
    return client.text_generation(model=model, prompt=prompt, max_new_tokens=512)

# =================================================================================
# Streaming helpers — envia em pedaços para a UI
# =================================================================================

import time

def _sse_stream(url: str, headers: Dict[str, str], payload: Dict[str, Any]):
    """Itera eventos SSE de /chat/completions com stream=True e retorna trechos de texto."""
    payload_stream = dict(payload)
    payload_stream["stream"] = True
    with requests.post(url, headers=headers, json=payload_stream, stream=True, timeout=300) as r:
        r.raise_for_status()
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
                ch = j.get("choices", [{}])[0]
                delta = (ch.get("delta") or {}).get("content") or (ch.get("message") or {}).get("content")
                if delta:
                    yield delta
            except Exception:
                continue

def stream_openrouter(model: str, messages: List[Dict[str, str]]):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
        "HTTP-Referer": st.secrets.get("APP_URL", ""),
        "X-Title": st.secrets.get("APP_TITLE", "Narrador JM"),
    }
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    yield from _sse_stream(url, headers, payload)

def stream_together(model: str, messages: List[Dict[str, str]]):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {st.secrets.get('TOGETHER_API_KEY', '')}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    yield from _sse_stream(url, headers, payload)

def stream_lmstudio(base_url: str, model: str, messages: List[Dict[str, str]]):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    yield from _sse_stream(url, headers, payload)

def _chunker(txt: str, n: int = 48):
    """Quebra texto em pedaços de ~n caracteres, respeitando espaços quando possível."""
    buf = []
    cur = 0
    while cur < len(txt):
        end = min(len(txt), cur + n)
        slice_ = txt[cur:end]
        if end < len(txt) and " " in slice_:
            last_sp = slice_.rfind(" ")
            if last_sp > 0:
                end = cur + last_sp + 1
                slice_ = txt[cur:end]
        buf.append(slice_)
        cur = end
    return buf

def stream_huggingface(model: str, messages: List[Dict[str, str]]):
    """HF Inference API nem sempre fornece SSE; simulamos streaming dividindo o texto."""
    full = call_huggingface(model, messages)
    for piece in _chunker(full, 48):
        yield piece
        time.sleep(0.02)  # leve suavização visual

# =================================================================================
# UI
# =================================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, str]] = []
    # 🔁 Carrega as 5 últimas interações salvas da sessão anterior (se houver)
    try:
        _loaded = carregar_ultimas_interacoes(5)
        if _loaded:
            st.session_state.chat = _loaded[-30:]
    except Exception:
        pass

st.title("Narrador JM — Clean Messages 🎬")

with st.sidebar:
    prov = st.radio("🌐 Provedor", ["OpenRouter", "Together", "LM Studio", "Hugging Face"], index=0)
    st.session_state["prov"] = prov  # <-- ADICIONE ESTA LINHA
    if prov == "OpenRouter":
        modelo = st.selectbox("Modelo (OpenRouter)", list(MODELOS_OPENROUTER.keys()), index=0)
        model_id = MODELOS_OPENROUTER[modelo]
    elif prov == "Together":
        modelo = st.selectbox("Modelo (Together)", list(MODELOS_TOGETHER_UI.keys()), index=0)
        model_id = MODELOS_TOGETHER_UI[modelo]
    elif prov == "Hugging Face":
        modelo = st.selectbox("Modelo (HF)", list(MODELOS_HF.keys()), index=0)
        model_id = MODELOS_HF[modelo]
    else:
        base = st.text_input("LM Studio base URL", value=DEFAULT_LMS_BASE_URL)
        st.session_state.setdefault("lms_base_url", base)
        st.session_state.lms_base_url = base or DEFAULT_LMS_BASE_URL
        lms_models = _lms_models_dict(st.session_state.lms_base_url)
        modelo = st.selectbox("Modelo (LM Studio)", list(lms_models.keys()), index=0)
        model_id = lms_models[modelo]
    st.session_state["model_id"] = model_id  # <-- ADICIONE ESTA LINHA

    if st.button("🗑️ Resetar chat"):
        st.session_state.chat.clear()
        st.rerun()

# Render histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]).container():
        st.markdown(apply_filters(m["content"]))  # sem filtros extras

# Entrada
if user_msg := st.chat_input("Fale com a Mary..."):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    st.session_state.chat.append({"role": "user", "content": user_msg})
    # Mantém apenas as últimas 30 interações na tela
    if len(st.session_state.chat) > 30:
        st.session_state.chat = st.session_state.chat[-30:]
    salvar_interacao(ts, st.session_state.session_id, prov, model_id, "user", user_msg)

    messages = build_minimal_messages(st.session_state.get("chat", []))

    try:
        ws_values = WS_INTERACOES.get_all_values() if WS_INTERACOES else []
    except Exception:
        ws_values = []
    
    # Se a pergunta for de memória, injeta trechos reais do histórico
    messages = wrap_messages_with_memory_if_needed(messages, user_msg, ws_values)
    
    # segue a chamada ao LLM normalmente, usando 'messages'

    with st.chat_message("assistant"):
        ph = st.empty()
        answer = ""
        try:
            if prov == "OpenRouter":
                gen = stream_openrouter(model_id, messages)
            elif prov == "Together":
                gen = stream_together(model_id, messages)
            elif prov == "Hugging Face":
                gen = stream_huggingface(model_id, messages)
            else:
                gen = stream_lmstudio(st.session_state.lms_base_url, model_id, messages)

            for delta in gen:
                answer += delta
                # Mostra texto em tempo real já filtrado (Jânio + fala do usuário)
                ph.markdown(apply_filters(answer) + "▌")
        except Exception as e:
            answer = f"[Erro ao chamar o modelo: {e}]"
            ph.markdown(apply_filters(answer))

        # ================== ⬇️ HOOK 2 AQUI  ⬇️ ==================
        try:
            ws_values = WS_INTERACOES.get_all_values() if WS_INTERACOES else []
        except Exception:
            ws_values = []

        # (A) puxão de orelha automático se detectar grito/ciúmes no input atual
        answer = inject_rebuke_if_needed(answer, user_msg, ws_values)

        # (B) lembrança espontânea (de vez em quando)
        answer = maybe_inject_spontaneous_recall(answer, user_msg, ws_values)
        # ================== ⬆️ HOOK 2 AQUI  ⬆️ ==================

        # Render final (sem cursor), aplica filtros e o tom carinhoso se ativo
        _ans_clean = apply_filters(answer)
        _ans_clean = inject_carinhosa(
            _ans_clean,
            user_msg,
            ativo=("Carinhosa" in (st.session_state.get("fala_mods") or []))
        )
        ph.markdown(_ans_clean)

    # Salva exatamente essa versão
    st.session_state.chat.append({"role": "assistant", "content": _ans_clean})
    if len(st.session_state.chat) > 30:
        st.session_state.chat = st.session_state.chat[-30:]
    ts2 = datetime.now().isoformat(sep=" ", timespec="seconds")
    salvar_interacao(ts2, st.session_state.session_id, prov, model_id, "assistant", _ans_clean)

    st.rerun()















































