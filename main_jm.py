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
        "keywords": ["motel status","status motel","quarto do motel","primeira vez","dor inicial","penetra","penetrou","penetração","penetrar"],
        "select": "first_strong",
    },
    "primeiro_beijo_praia": {
        "keywords": ["primeiro beijo","beijou","beijaram","beijo salgado","praia de camburi","areia","calçadão","calcadao"],
        "select": "first_strong",
    },
    "onde_se_conheceram": {
        "keywords": ["primeiro encontro","quando nos conhecemos","a primeira vez que te vi","como te conheci","te vi no"],
        "select": "first",
    },
    "ciumes_ricardo": {
        "keywords": ["ciúme","ciumes","ciumento","ciumenta","grita","gritou","gritando","bate-boca","discussão","discussao","ricardo"],
        "select": "canonical_score_oldest",
    },
    "reconciliacao": {
        "keywords": ["desculpa","perdão","perdao","me excedi","errei","prometo não gritar","prometo nao gritar","vamos conversar"],
        "select": "after(ciumes_ricardo)",
    },
    # usado pelo guardião de estado relacional (não entra no PRIORITY)
    "rompimento": {
        "keywords": ["terminamos","rompemos","acabou","dar um tempo","fim do namoro","separamos","separar"],
        "select": "most_recent",
    },
}
PRIORITY = ["primeira_noite_motel","primeiro_beijo_praia","onde_se_conheceram","ciumes_ricardo","reconciliacao"]

def _score_kw(text: str, kws: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in kws if k in t)

def _pick_episode(evs: List[Dict], ev_key: str, base: Optional[Dict]=None) -> Optional[Dict]:
    spec = EVENTS[ev_key]; kws = spec["keywords"]; sel = spec["select"]
    cands = [(e, _score_kw(e.get("text",""), kws)) for e in evs]
    cands = [(e, s) for (e, s) in cands if s > 0]
    if not cands:
        return None

    if sel == "first":
        return min((e for e,_ in cands), key=lambda e: e["turn"])
    if sel == "first_strong":
        strong = [e for (e,s) in cands if s >= 2] or [e for (e,s) in cands if s >= 1]
        return min(strong, key=lambda e: e["turn"])
    if sel == "canonical_score_oldest":
        cands.sort(key=lambda es: (-es[1], es[0]["turn"]))
        return cands[0][0]
    if sel == "most_recent":
        return max((e for e,_ in cands), key=lambda e: e["turn"])
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

# (DEPRECATED) Antigo gerador de recall com frases prontas: mantido por compatibilidade.
# Agora as lembranças são compostas organicamente em maybe_inject_spontaneous_recall().
def _compose_recall(evs: List[Dict]) -> Optional[str]:
    return None

# --- helper: nº de turnos (user/assistant) da ÚLTIMA sessão ---
def _current_turn_from_sheet(ws_values: List[List[str]]) -> int:
    evs = _sheet_to_events_ultima_sessao(ws_values)
    return len(evs)

# prob. baixa + gatilhos; cooldown forte 40 turnos
def _should_recall(user_msg: str, base_prob: float = 0.12) -> bool:
    t = (user_msg or "").lower()
    gatilhos = ["motel","status","primeiro beijo","praia","ciúme","ciumes","ricardo","conhecemos"]
    p = base_prob + (0.18 if any(g in t for g in gatilhos) else 0.0)
    return random.random() < min(max(p, 0.0), 0.6)

def _cooldown_ok(ws_values: List[List[str]], min_gap_turns: int = 40) -> bool:
    cur_turn = _current_turn_from_sheet(ws_values)
    last = st.session_state.get("last_recall_turn", -9999)
    if cur_turn - last >= min_gap_turns:
        st.session_state["last_recall_turn"] = cur_turn
        return True
    return False

def maybe_inject_spontaneous_recall(answer: str, user_msg: str, ws_values: List[List[str]]) -> str:
    """Insere lembrança leve no meio da fala (máx. 1x/40 turnos), sem frase pronta."""
    if not answer or not answer.strip():
        return answer
    if not _should_recall(user_msg) or not _cooldown_ok(ws_values, 40):
        return answer

    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs:
        return answer

        # ep = _pick_episode_respecting_third_party(evs, user_msg)  # ❌ inexistente
    ep = None
    for key in PRIORITY:
        ep = _pick_episode(evs, key)
        if ep:
            break
    if not ep:
        return answer


    trecho = (ep["text"] or "").strip()
    if len(trecho) > 140:
        trecho = trecho[:140].rsplit(" ", 1)[0] + "…"

    variantes = [
        lambda t: f"(Ela lembra de quando {t.lower()})",
        lambda t: f"e por um instante ela lembra de quando {t.lower()}",
        lambda t: f"— Engraçado… isso me faz lembrar de quando {t.lower()}",
    ]
    lembrete = random.choice(variantes)(trecho)

    low = answer.lower()
    if any(x in low for x in ["me faz lembrar", "ela lembra de quando", "engraçado… isso me faz lembrar"]):
        return answer

    partes = re.split(r'([\.!?])(\s+)', answer, maxsplit=2)
    if len(partes) >= 3:
        if len(re.split(r'[\.!?]', answer, maxsplit=2)) >= 3 and random.random() < 0.5:
            head = "".join(partes[:3])
            tail = "".join(partes[3:])
            sub = re.split(r'([\.!?])(\s+)', tail, maxsplit=1)
            if len(sub) >= 3:
                return head + sub[0] + sub[1] + " " + lembrete + sub[2]
        return partes[0] + partes[1] + " " + lembrete + (partes[2] if partes[2].strip() else "") + "".join(partes[3:])
    return answer.strip() + " " + lembrete

# ---- Puxão de orelha ao detectar grito real no input atual ----
_GRITO_KEYS  = ["grita", "gritou", "gritando", "gritar", "berrou", "berrar", "berrando", "aos gritos", "voz alta"]
_CIUMES_KEYS = ["ciúme", "ciumes", "ciumento", "ciumenta"]  # <- sem "ricardo" para evitar falso positivo
_UPPER_RE = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{3,}\b")  # palavra em CAPS (≥3 letras)

def _parece_grito(texto: str) -> bool:
    t = (texto or "").strip()
    tl = t.lower()
    if any(k in tl for k in _GRITO_KEYS):
        return True
    # CAPS + "!" em textos curtos costuma indicar grito
    return bool(_UPPER_RE.search(t)) and "!" in t and len(t) <= 120

def _parece_ciumes(texto: str) -> bool:
    tl = (texto or "").lower()
    return any(k in tl for k in _CIUMES_KEYS)

def _buscar_incidente_ciumes(evs: List[Dict]) -> Optional[Dict]:
    episodios = []
    for e in evs:
        score = 0; tl = (e.get("text") or "").lower()
        if any(w in tl for w in _GRITO_KEYS):  score += 2
        if any(w in tl for w in _CIUMES_KEYS): score += 2
        if "ricardo" in tl:                    score += 1  # só como prioridade (não gatilho)
        if score > 0:
            episodios.append((e, score))
    if not episodios:
        return None
    episodios.sort(key=lambda es: (-es[1], es[0]["turn"]))
    return episodios[0][0]

def inject_rebuke_if_needed(answer: str, user_msg: str, ws_values: List[List[str]]) -> str:
    # Só reage quando há grito real no input atual
    if not _parece_grito(user_msg):
        return answer

    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs or not answer or not answer.strip():
        return answer

    # Contexto suave (se houve discussão anterior)
    pista = ""
    ep = _buscar_incidente_ciumes(evs)
    if ep:
        txt = (ep.get("text") or "").lower()
        pista = " (como combinamos depois daquela discussão" + (" por ciúmes do Ricardo" if "ricardo" in txt else "") + ")"

    # Variações curtas, orgânicas (inline)
    variantes = [
        lambda: f"(Ela respira fundo e pede calma{pista}.)",
        lambda: f"— Calma. Vamos falar baixo{pista}.",
        lambda: f"Ela baixa o tom e sugere conversarem sem elevar a voz{pista}.",
    ]
    lembrete = random.choice(variantes)()

    low = (answer or "").lower()
    if any(x in low for x in ["vamos falar baixo", "pede calma", "sem elevar a voz", "calma. vamos"]):
        return answer

    # Insere após a 1ª frase; se não houver pontuação, anexa no fim
    partes = re.split(r'([\.!?])(\s+)', answer, maxsplit=2)
    if len(partes) >= 3:
        return partes[0] + partes[1] + " " + lembrete + (partes[2] if partes[2].strip() else "") + "".join(partes[3:])
    return answer.strip() + " " + lembrete


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
    # dedup
    uniq: List[str] = []
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
    # ✅ mantém system em primeiro
    return messages[:1] + base + messages[1:]

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
    # no topo do arquivo:
ROLEPLAY_MODES = {
    "Cool": "Só conversas cotidianas, curiosidade e aproximação inicial.",
    "Mellow": "Clima levemente sugestivo, insinuações, sem abordagem sexual direta.",
    "Standard": "Flerte e química surgem naturalmente; nada explícito sem gatilho do usuário.",
    "Real": "Reage de forma realista; ousadia só com sinais claros do usuário.",
    "Hot": "Flerte aberto e direto, sem ser explícito de cara.",
    "Inner Secret": "Desejos/fantasias revelados gradualmente.",
    "Explicit": "Sexualidade explícita permitida; com transição e consentimento."
}

    st.session_state.setdefault("equilibrio_modo", "Standard")
    modo_eq = st.selectbox("Clima de início", list(ROLEPLAY_MODES.keys()), index=2)
    st.caption(ROLEPLAY_MODES.get(modo_eq, ""))

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
Nome: Mary Massariol (20). Engenharia Civil — UFES. Mora com a mãe, Joselina, ap. 202 — Rua Beethoven, Laranjeiras. Usa moto 500cc.
Traços: inteligente, romântica, gosta de praia e flerte; não tolera grosseria/cantada barata; quer respeito e liberdade.

[TRAÇOS FÍSICOS — SENSORIAIS]
— Cabelos negros, volumosos e soltos, que balançam com movimentos suaves.
— Olhos verdes expressivos, que revelam emoções antes das palavras.
— Seios médios e firmes; cintura definida que acentua os quadris largos.
— Coxas grossas e torneadas; bumbum redondo e firme, que move o tecido ao andar.
— Barriga lisa e tonificada; pele dourada pelo sol e levemente perfumada.
— Use apenas UM traço por parágrafo e integre à ação (ex.: “o tecido estica nas coxas quando ela se inclina”); nunca liste.

[CONHECIMENTO E MEMÓRIA]
— Não conhece o usuário previamente; aprende somente pelo diálogo/ações na cena.
— Sem onisciência: nunca descreva pensamentos/intenções do usuário; apenas o que ele diz/faz ou sinais visíveis.
— Mantenha coerência estrita com o histórico (memória recente e eventos passados).

[RELACIONAMENTO / LIMITES]
— Virgem por escolha; só se entrega com confiança/desejo construídos em cena.
— Rejeita traição; entrega total apenas por vontade própria.
— Progressão: flerte > toque leve > beijo > carícia > toque íntimo > sexo (consentido).

[TERCEIROS — PRIORIDADE AO USUÁRIO]
— Mary não inicia cenas românticas/íntimas com terceiros. Se terceiros avançarem, ela desvia com respeito e estabelece limite.
— Menções a “Ricardo” e outros servem a conflito/plot neutro; sem carícias/beijos/avances, salvo pedido explícito do usuário.
— Com o usuário presente, Mary prioriza atenção e respeito a ele.

[CLIMA E CONDUTA]
— Ajuste o tom ao usuário/preset; provoque com leveza e respeito.
— Após cenas intensas, traga respiros (cuidado, humor, cotidiano) e transições naturais de lugar/tempo.
— Não escreva falas do usuário; não ofereça menus/opções; não decida por ele.

[LOCAIS COMUNS]
Clube Serra Bella (balada), Motel Status (privacidade), Café Oregon (encontros), Praia de Camburi (lazer), Enseada do Suá (ousadia).

[STATUS RELACIONAL — CONSISTÊNCIA]
— Nunca declare namoro exclusivo, rompimento ou “dar um tempo” sem evento explícito no histórico.
— Se o usuário pedir rótulos (“terminamos?”, “somos exclusivos?”), peça confirmação ou proponha conversar com calma em cena.

[RESTRIÇÕES DE ESTILO — OBRIGATÓRIO]
— NÃO usar “Foto:”, “Legenda:” ou formatos de postagem/ensaio fotográfico.
— NÃO criar objetos, sons, mensagens, apps, roupas ou acessórios novos não citados antes.
— NÃO inventar simbolismos/metáforas; narre ações e percepções imediatas de Mary.
— Linguagem natural, direta e contemporânea; sem comparações acadêmicas.
— Foque em diálogo e reações físicas/emocionais percebidas por Mary.
— Máx. 5 parágrafos por turno; até 2 frases por parágrafo; ~30% mais concisa que o padrão.
— Um traço sensorial/físico por parágrafo; evite repetição.

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
    clima_modo = (st.session_state.get("equilibrio_modo") or "Standard").strip()

    # 2) Catálogo de climas — usa global ROLEPLAY_MODES se existir; senão, fallback local
    roleplay_modes = globals().get("ROLEPLAY_MODES", {
        "Cool":        "Só conversas cotidianas, curiosidade e aproximação inicial.",
        "Mellow":      "Clima levemente sugestivo, insinuações, sem abordagem sexual direta.",
        "Standard":    "Flerte e química surgem naturalmente; nada explícito sem gatilho do usuário.",
        "Real":        "Reage de forma realista ao contexto e histórico; ousadia só com sinais claros do usuário.",
        "Hot":         "Flerte aberto e direto, sem ser explícito de cara; respeite limites do contexto.",
        "Inner Secret":"Desejos/fantasias revelados gradualmente conforme a aproximação.",
        "Explicit":    "Sexualidade explícita permitida; ainda assim respeite transição e consentimento."
    })

    # ✅ Fallback seguro do modo de clima
    if clima_modo not in roleplay_modes:
        clima_modo = "Standard"

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

    equilibrio_block = build_equilibrio_block(clima_modo)

    # Usa a função GLOBAL build_fala_block (não redefina outra dentro desta função)
    fala_block = build_fala_block(fala_mods) if "build_fala_block" in globals() else ""

    extras = []
    if user_name:
        extras.append(f"[USUÁRIO]\n— Trate o usuário pelo nome: {user_name}.")
    if scenario or plot:
        extras.append("[CENÁRIO/ENREDO INICIAL]")
        if scenario: extras.append(f"— Cenário: {scenario}")
        if plot:     extras.append(f"— Enredo: {plot}")

    # 5) Montar o system final
    parts = [equilibrio_block]
    if fala_block:
        parts.append(fala_block)
    parts.append(PERSONA_MARY)  # persona otimizada
    if extras:
        parts.append("\n".join(extras))
    system_text = "\n\n".join(parts)

    # 6) Chat mínimo (mantém apenas user/assistant; system em primeiro)
    mensagens_chat = [
        m for m in chat
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
    msgs.extend({"role": m["role"], "content": m["content"].strip()} for m in mensagens_chat)

    return msgs
def build_minimal_messages(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    import re

    # 1) Ler inputs da UI (opcionais)
    user_name  = (st.session_state.get("user_name") or "").strip()
    scenario   = (st.session_state.get("scenario_init") or "").strip()
    plot       = (st.session_state.get("plot_init") or "").strip()
    fala_mods  = st.session_state.get("fala_mods") or []
    clima_modo = (st.session_state.get("equilibrio_modo") or "Standard").strip()

    # 2) Catálogo de climas — usa global ROLEPLAY_MODES se existir; senão, fallback local
    roleplay_modes = globals().get("ROLEPLAY_MODES", {
        "Cool":        "Só conversas cotidianas, curiosidade e aproximação inicial.",
        "Mellow":      "Clima levemente sugestivo, insinuações, sem abordagem sexual direta.",
        "Standard":    "Flerte e química surgem naturalmente; nada explícito sem gatilho do usuário.",
        "Real":        "Reage de forma realista ao contexto e histórico; ousadia só com sinais claros do usuário.",
        "Hot":         "Flerte aberto e direto, sem ser explícito de cara; respeite limites do contexto.",
        "Inner Secret":"Desejos/fantasias revelados gradualmente conforme a aproximação.",
        "Explicit":    "Sexualidade explícita permitida; ainda assim respeite transição e consentimento."
    })

    # ✅ Fallback seguro do modo de clima
    if clima_modo not in roleplay_modes:
        clima_modo = "Standard"

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

    equilibrio_block = build_equilibrio_block(clima_modo)

    # Usa a função GLOBAL build_fala_block (não redefina outra dentro desta função)
    fala_block = build_fala_block(fala_mods) if "build_fala_block" in globals() else ""

    extras = []
    if user_name:
        extras.append(f"[USUÁRIO]\n— Trate o usuário pelo nome: {user_name}.")
    if scenario or plot:
        extras.append("[CENÁRIO/ENREDO INICIAL]")
        if scenario: extras.append(f"— Cenário: {scenario}")
        if plot:     extras.append(f"— Enredo: {plot}")

    # 5) Montar o system final
    parts = [equilibrio_block]
    if fala_block:
        parts.append(fala_block)
    parts.append(PERSONA_MARY)  # persona otimizada
    if extras:
        parts.append("\n".join(extras))
    system_text = "\n\n".join(parts)

    # 6) Chat mínimo (mantém apenas user/assistant; system em primeiro)
    mensagens_chat = [
        m for m in chat
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
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
# Guardiões — consistência relacional e limites com terceiros (NPCs)
# =================================================================================

# Se você já definiu estes nomes/constantes em outro ponto, pode remover as duplicatas abaixo.
_THIRD_NAMES = ["ricardo", "rafa", "rafael", "gustavo", "hernando", "hernando cola"]

# Padrões de linguagem para detectar declarações de rompimento/exclusividade
_BREAKUP_PATTERNS = [
    r"\btermin(a mos|ei|ou)\b", r"\bromp(e mos|i)\b", r"\bacab(ou|ei|amos)\b",
    r"\b(se )?paramos\b", r"\b(dar|dei|demos) um tempo\b", r"\bfim do namoro\b"
]
_EXCLUSIVITY_PATTERNS = [
    r"\bestamos juntos\b", r"\bestamos namorando\b", r"\bminh[ao] (namorada|namorado)\b"
]

def _contains_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

# Requer: EVENTS["rompimento"] e _pick_episode(...) já definidos
def _find_breakup(evs: List[Dict]) -> Optional[Dict]:
    return _pick_episode(evs, "rompimento") if evs else None

def guard_relationship_claims(answer: str, ws_values: List[List[str]]) -> str:
    """
    Impede que Mary declare rompimento/exclusividade sem evento explícito no histórico.
    Em vez de negar, reescreve de forma neutra e convida a conversar.
    """
    if not answer or not answer.strip():
        return answer

    evs = _sheet_to_events_ultima_sessao(ws_values)
    if not evs:
        return answer

    has_breakup = _find_breakup(evs) is not None

    # Caso 1: fala em "terminamos/rompemos/acabou..." sem evidência → suaviza
    if _contains_any(answer, _BREAKUP_PATTERNS) and not has_breakup:
        repls = [
            (r"\btermin(a mos|ei|ou)\b", "não quero tratar isso por mensagem"),
            (r"\bromp(e mos|i)\b", "prefiro alinhar isso conversando"),
            (r"\bacab(ou|ei|amos)\b", "não é assim que quero tratar a gente"),
            (r"\b(se )?paramos\b", "melhor conversarmos com calma"),
            (r"\b(dar|dei|demos) um tempo\b", "talvez seja melhor darmos um respiro e conversar"),
            (r"\bfim do namoro\b", "prefiro não rotular nada agora"),
        ]
        fixed = answer
        for pat, sub in repls:
            fixed = re.sub(pat, sub, fixed, flags=re.IGNORECASE)
        return fixed

    # Caso 2 (opcional): afirma exclusividade sem evidência de pedido/aceite → suaviza
    # Se futuramente registrar “pedido de namoro” como evento, cheque aqui antes de suavizar.
    # if _contains_any(answer, _EXCLUSIVITY_PATTERNS) and not has_exclusivity_event:
    #     fixed = re.sub(r"\b(minh[ao] (namorada|namorado))\b", "você", answer, flags=re.IGNORECASE)
    #     fixed = re.sub(r"\bestamos (juntos|namorando)\b", "prefiro ir com calma", fixed, flags=re.IGNORECASE)
    #     return fixed

    return answer


# Sinais de intimidade que não devem ocorrer com terceiros sem pedido explícito do usuário
_INTIMATE_CUES = [
    r"\bbeij(o|ou|ando)\b", r"\bbeija\b", r"\bcarí(ci|cia)\b", r"\bcarinho\b",
    r"\bencosta\b", r"\bapert(a|ou)\b", r"\bm[aã]o (no|na|sobre)\b", r"\btoca(r|)\b", r"\bquadril\b"
]

def guard_third_party_intimacy(answer: str, user_msg: str) -> str:
    """
    Evita intimidade com NPCs (ex.: Ricardo) quando o usuário não pediu isso explicitamente no turno.
    Mantém menções neutras a terceiros e reforça limite/respeito.
    """
    if not answer or not answer.strip():
        return answer

    low_ans = answer.lower()
    if not any(n in low_ans for n in _THIRD_NAMES):
        return answer

    # Libera caso o usuário peça explicitamente cena com terceiro
    if re.search(r"\b(quer(o)?|podemos|vamos)\b.*\b(ricardo|terceir[oa]|a três|a tres|ménage|menage|trisal)\b",
                 (user_msg or "").lower()):
        return answer

    # Se houver gesto/ato íntimo com terceiro → substitui por limite/respeito
    if any(re.search(p, low_ans) for p in _INTIMATE_CUES):
        # Anexa uma linha curta de limite, sem apagar todo o texto.
        return (answer.rstrip() +
                " Mary recua um passo, segura o pulso com gentileza e estabelece limite — sem intimidade com terceiros.")

    return answer

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

        # Guardiões
        answer = guard_relationship_claims(answer, ws_values)
        answer = guard_third_party_intimacy(answer, user_msg)

        # (A) puxão de orelha automático se detectar grito real no input atual
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


