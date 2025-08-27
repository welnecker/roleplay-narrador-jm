# 🎬 Narrador JM — Somente LM Studio (local)
# -------------------------------------------------------------
# Requisitos: pip install streamlit requests
# Como usar:
# 1) Abra o LM Studio → aba "Developer" → clique em "Start Server".
# 2) Carregue/baixe um modelo no LM Studio (ex.: llama-3-8b-lexi-uncensored).
# 3) Rode:  streamlit run app_lmstudio_only.py
# 4) Ajuste a Base URL se necessário (ex.: http://127.0.0.1:1234/v1).
# -------------------------------------------------------------

import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any

import requests
import streamlit as st

# ----------------------------
# CONFIG BÁSICA DO APP
# ----------------------------
st.set_page_config(page_title="Narrador JM — LM Studio", page_icon="🎬", layout="wide")

# ----------------------------
# HELPERS: LM Studio
# ----------------------------
@st.cache_data(ttl=15, show_spinner=False)
def lms_list_models(base_url: str) -> list[str]:
    """Lista modelos expostos pelo servidor do LM Studio (OpenAI-compatible)."""
    try:
        url = base_url.rstrip("/") + "/models"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        j = r.json()
        # LM Studio usa {object:"list", data:[{id:"…"}, …]}
        return [m.get("id") for m in j.get("data", []) if m.get("id")]
    except Exception:
        return []


def stream_chat_lmstudio(*, base_url: str, model: str, messages: List[Dict[str, str]],
                         temperature: float = 0.7, top_p: float = 1.0,
                         max_tokens: int = 1200, timeout: int = 300) -> str:
    """Faz streaming de /v1/chat/completions no servidor do LM Studio."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}

    resposta_txt = ""
    with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        last_update = 0.0
        placeholder = st.empty()
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
            except Exception:
                continue
            delta = (
                j.get("choices", [{}])[0]
                 .get("delta", {})
                 .get("content", "")
            )
            if not delta:
                continue
            resposta_txt += delta
            if time.time() - last_update > 0.08:
                visible = _render_visible(resposta_txt) + "▌"
                placeholder.markdown(visible)
                last_update = time.time()
        # flush final
        placeholder.markdown(_render_visible(resposta_txt))
    return resposta_txt


# ----------------------------
# FORMATAÇÃO DO TEXTO (parágrafos curtos + falas em linhas)
# ----------------------------
_DASHES = re.compile(r"(?:\s*--\s*|\s*–\s*|\s*—\s*)")
_SENT_END = re.compile(r"[.!?…](?=\s|$)")
_UPPER_OR_DASH = re.compile(r"([.!?…])\s+(?=(?:—|[A-ZÁÉÍÓÚÂÊÔÃÕÀÇ0-9]))")


def roleplay_paragraphizer(t: str) -> str:
    if not t:
        return ""
    # 1) Normaliza travessão / fala iniciando linha
    t = _DASHES.sub("\n— ", t)
    # 2) Quebra após pontuação ao surgir nova frase/fala
    t = _UPPER_OR_DASH.sub(r"\1\n", t)
    # 3) Limpa e separa linhas vazias duplicadas
    t = re.sub(r"[ \t]+", " ", t)
    linhas = [ln.strip() for ln in t.splitlines() if ln.strip()]

    out, buf, sent, chars = [], [], 0, 0
    MAX_SENT_PER_PARA = 2
    MAX_CHARS_PER_PARA = 240

    for ln in linhas:
        if ln.startswith("—"):
            if buf:
                out.append(" ".join(buf).strip())
                out.append("")
                buf, sent, chars = [], 0, 0
            out.append(ln)
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

    final = []
    for ln in out:
        if ln == "" and (not final or final[-1] == ""):
            continue
        final.append(ln)
    if final and final[-1] == "":
        final.pop()
    return "\n".join(final).strip()


SCENERY_WORD = re.compile(r"\b(c[ée]u|nuvens?|horizonte|luar|mar|onda?s?|pier|vento|brisa|chuva|garoa|sereno|paisage?m|cen[áa]rio)\b", re.I)


def sanitize_scenery(text: str) -> str:
    if not text:
        return ""
    return SCENERY_WORD.sub("", text)


def sanitize_scenery_preserve_opening(t: str) -> str:
    if not t:
        return ""
    linhas = t.strip().split("\n")
    if not linhas:
        return ""
    primeira = linhas[0].strip()
    resto = "\n".join(linhas[1:]).strip()
    if not resto:
        return primeira
    return primeira + "\n" + sanitize_scenery(resto)


EXPL_PAT = re.compile(r"\b(mamilos?|genit[aá]lia|ere[cç][aã]o|penetra[cç][aã]o|boquete|gozada|gozo|sexo oral|chupar|enfiar)\b", re.I)


def classify_nsfw_level(t: str) -> int:
    if EXPL_PAT.search(t or ""):
        return 3
    if re.search(r"\b(cintura|pesco[cç]o|costas|beijo prolongado|respira[cç][aã]o curta)\b", (t or ""), re.I):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aã]os dadas|beijo)\b", (t or ""), re.I):
        return 1
    return 0


def sanitize_explicit(t: str, max_level: int) -> str:
    lvl = classify_nsfw_level(t)
    # No momento não cortamos; apenas lugar para futura filtragem
    return t if lvl <= max_level else t


def _render_visible(t: str) -> str:
    t = sanitize_scenery_preserve_opening(t)
    t = roleplay_paragraphizer(t)
    t = sanitize_explicit(t, int(st.session_state.get("nsfw_max_level", 0)))
    return t


# ----------------------------
# PROMPT BUILDER (compacto, mas cobre o sidebar)
# ----------------------------
def _fase_label(n: int) -> str:
    nomes = {
        0: "Estranhos",
        1: "Percepção",
        2: "Conhecidos",
        3: "Romance",
        4: "Namoro",
        5: "Compromisso / Encontro definitivo",
    }
    return f"{n} — {nomes.get(int(n), 'Fase')}"


def build_system_prompt(ctx: Dict[str, Any]) -> str:
    modo = ctx.get("modo_resposta", "Narrador padrão")
    estilo = ctx.get("estilo_escrita", "AÇÃO")
    fase = ctx.get("fase", 0)
    nsfw = ctx.get("nsfw_max_level", 0)
    ritmo = ["muito lento", "lento", "médio", "rápido"][int(ctx.get("ritmo_cena", 0))]
    sintoniza = "SIM" if ctx.get("modo_sintonia", True) else "NÃO"
    emocao = ctx.get("app_emocao_oculta", "nenhuma")
    bloquear = "ATIVO" if ctx.get("app_bloqueio_intimo", True) and fase < 5 else "DESATIVADO"

    cabeca = (
        "Você é Mary e responde em 1ª pessoa (eu)." if modo == "Mary (1ª pessoa)"
        else "Você é o Narrador em 3ª pessoa; foque em Mary e Jânio, sem falar com o leitor."
    )

    linhas = [
        cabeca,
        f"Estilo: {estilo}. Ritmo: {ritmo}. Sintonia com parceiro: {sintoniza}. Emoção oculta: {emocao}.",
        f"Fase atual do romance: {_fase_label(fase)} (respeite limites de avanço).",
        f"Nível de calor permitido: {nsfw} (0 a 3).",
        "Nunca descreva cenário/natureza/clima (proibido céu, mar, vento, ondas, luar etc.).",
        "Mantenha parágrafos curtos (máx. 2 frases) e falas com travessão em linhas separadas.",
        "Intercale ação física e diálogo; foque no corpo, reações e toques, sem metáforas ambientais.",
    ]

    if bloquear == "ATIVO":
        linhas.append("Proteção íntima ativa: sem clímax explícito sem liberação direta do usuário.")

    finalizacao = ctx.get("finalizacao_modo", "ponto de gancho")
    if finalizacao == "ponto de gancho":
        linhas.append("Feche com micro-ação de gancho natural (sem concluir a cena).")
    elif finalizacao == "fecho suave":
        linhas.append("Feche com uma frase objetiva, sem cliffhanger.")
    else:
        linhas.append("Mantenha leve suspense, sem ambiguidades clichê.")

    if ctx.get("usar_falas_mary", False):
        linhas.append("Se usar falas fixas de Mary, repita literalmente e intercale com ação corporal.")

    return "\n- ".join([linhas[0]] + linhas[1:])


def build_opening_line(ctx: Dict[str, Any]) -> str:
    tempo = (ctx.get("tempo") or "").strip().rstrip(".")
    lugar = (ctx.get("lugar") or "").strip().rstrip(".")
    figurino = (ctx.get("figurino") or "").strip().rstrip(".")
    pedacos = []
    if tempo:
        pedacos.append(tempo.capitalize())
    mary = "Mary" + (f", {figurino}" if figurino else "")
    pedacos.append(mary)
    if lugar:
        pedacos.append(lugar)
    return (". ".join([p for p in pedacos if p]) + ".").strip(".") + "." if pedacos else ""


def build_messages(ctx: Dict[str, Any], user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": build_system_prompt(ctx)})

    abertura = build_opening_line(ctx)
    if abertura:
        # Pequena exceção para primeira linha objetiva
        msgs.append({"role": "system", "content": f"ABERTURA_SUGERIDA: {abertura}"})

    msgs.append({"role": "user", "content": user_text})
    return msgs


# ----------------------------
# ESTADOS INICIAIS (Session State)
# ----------------------------
DEFAULTS = {
    "modelo_escolhido": "",
    "lms_base_url": "http://127.0.0.1:1234/v1",
    "modo_resposta": "Narrador padrão",
    "estilo_escrita": "AÇÃO",
    "nsfw_max_level": 0,
    "modo_sintonia": True,
    "ritmo_cena": 0,
    "finalizacao_modo": "ponto de gancho",
    "usar_falas_mary": False,
    "fase": 0,
    "no_coincidencias": True,
    "app_bloqueio_intimo": True,
    "app_emocao_oculta": "nenhuma",
    "max_tokens_rsp": 1200,
    "timeout_s": 300,
    "temperature": 0.7,
    "top_p": 1.0,
    # contexto opcional
    "tempo": "",
    "lugar": "",
    "figurino": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ----------------------------
# UI — Sidebar
# ----------------------------
with st.sidebar:
    st.title("🧭 Painel do Roteirista (LM Studio)")

    st.markdown("### 🌐 Servidor LM Studio")
    base_url = st.text_input("Base URL (LM Studio)", value=st.session_state.get("lms_base_url", "http://127.0.0.1:1234/v1"))
    st.session_state.lms_base_url = base_url

    modelos = lms_list_models(base_url)
    if not modelos:
        st.warning("⚠️ Servidor do LM Studio não encontrado ou sem modelos. Abra o LM Studio → Developer → Start Server.")
    modelo_escolhido = st.selectbox("🤖 Modelo de IA (LM Studio)", modelos or ["<digite manualmente>"])
    if modelo_escolhido == "<digite manualmente>":
        modelo_escolhido = st.text_input("Model identifier (LM Studio)", value=st.session_state.get("modelo_escolhido", "llama-3-8b-lexi-uncensored"))
    st.session_state.modelo_escolhido = modelo_escolhido

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros do Modelo")
    st.slider("Temperature", 0.0, 1.5, value=float(st.session_state.get("temperature", 0.7)), step=0.05, key="temperature")
    st.slider("Top-p", 0.0, 1.0, value=float(st.session_state.get("top_p", 1.0)), step=0.05, key="top_p")

    st.markdown("---")
    st.markdown("### ✍️ Estilo & Progresso Dramático")
    st.selectbox("Modo de resposta", ["Narrador padrão", "Mary (1ª pessoa)"], key="modo_resposta")
    st.selectbox("Estilo de escrita", ["AÇÃO", "ROMANCE LENTO", "NOIR"], key="estilo_escrita")
    st.slider("Nível de calor (0=leve, 3=explícito)", 0, 3, value=int(st.session_state.get("nsfw_max_level", 0)), key="nsfw_max_level")
    st.checkbox("Sintonia com o parceiro (modo harmônico)", key="modo_sintonia", value=st.session_state.get("modo_sintonia", True))
    st.select_slider("Ritmo da cena", options=[0,1,2,3], value=int(st.session_state.get("ritmo_cena", 0)), format_func=lambda n: ["muito lento","lento","médio","rápido"][n], key="ritmo_cena")
    st.selectbox("Finalização", ["ponto de gancho", "fecho suave", "deixar no suspense"], key="finalizacao_modo")
    st.checkbox("Usar falas da Mary (fixas)", key="usar_falas_mary", value=st.session_state.get("usar_falas_mary", False))

    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio (apenas Fase)")
    fase_escolhida = st.select_slider("Fase do romance", options=[0,1,2,3,4,5], value=int(st.session_state.get("fase", 0)), format_func=_fase_label, key="fase")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Avançar 1 fase"):
            st.session_state.fase = min(5, int(st.session_state.get("fase", 0)) + 1)
    with col2:
        if st.button("↺ Reiniciar (0)"):
            st.session_state.fase = 0
    st.checkbox("Evitar coincidências forçadas (montagem paralela)", key="no_coincidencias", value=st.session_state.get("no_coincidencias", True))
    st.checkbox("Bloquear avanços íntimos sem ordem", key="app_bloqueio_intimo", value=st.session_state.get("app_bloqueio_intimo", True))
    st.selectbox("🎭 Emoção oculta", ["nenhuma","tristeza","felicidade","tensão","raiva"], key="app_emocao_oculta")

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    st.markdown("---")
    st.markdown("### 🧩 Contexto da Cena (opcional)")
    st.text_input("Tempo (ex.: Domingo de manhã)", key="tempo")
    st.text_input("Lugar (ex.: Jacaraípe / em casa)", key="lugar")
    st.text_input("Figurino (ex.: short jeans e regata)", key="figurino")


# ----------------------------
# UI — Corpo principal
# ----------------------------
st.title("🎬 Narrador JM — LM Studio (somente local)")
st.caption("Use o LM Studio como back-end. O app lista os modelos carregados e envia prompts no formato OpenAI.")

user_input = st.text_area("📝 Direção de cena / o que deve acontecer agora", height=160, placeholder="Ex.: Domingo de manhã. Mary encontra Jânio na academia e pede ajuda no exercício de costas...")

col_run1, col_run2 = st.columns([1,1])
with col_run1:
    run_btn = st.button("Gerar cena com LM Studio", type="primary")
with col_run2:
    clear_btn = st.button("Limpar tela")

if clear_btn:
    st.experimental_rerun()

if run_btn:
    if not user_input.strip():
        st.error("Digite uma direção de cena para gerar a resposta.")
    elif not st.session_state.get("modelo_escolhido"):
        st.error("Selecione ou digite um modelo do LM Studio.")
    else:
        ctx = {
            "modo_resposta": st.session_state.get("modo_resposta"),
            "estilo_escrita": st.session_state.get("estilo_escrita"),
            "nsfw_max_level": st.session_state.get("nsfw_max_level"),
            "modo_sintonia": st.session_state.get("modo_sintonia"),
            "ritmo_cena": st.session_state.get("ritmo_cena"),
            "finalizacao_modo": st.session_state.get("finalizacao_modo"),
            "usar_falas_mary": st.session_state.get("usar_falas_mary"),
            "fase": st.session_state.get("fase"),
            "app_bloqueio_intimo": st.session_state.get("app_bloqueio_intimo"),
            "app_emocao_oculta": st.session_state.get("app_emocao_oculta"),
            # contexto
            "tempo": st.session_state.get("tempo"),
            "lugar": st.session_state.get("lugar"),
            "figurino": st.session_state.get("figurino"),
        }

        messages = build_messages(ctx, user_input)
        with st.status("Enviando para o LM Studio…", expanded=False):
            try:
                _ = stream_chat_lmstudio(
                    base_url=st.session_state.get("lms_base_url", "http://127.0.0.1:1234/v1"),
                    model=st.session_state.get("modelo_escolhido"),
                    messages=messages,
                    temperature=float(st.session_state.get("temperature", 0.7)),
                    top_p=float(st.session_state.get("top_p", 1.0)),
                    max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                    timeout=int(st.session_state.get("timeout_s", 300)),
                )
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code if e.response else ''}: {e}")
            except Exception as e:
                st.error(f"Erro ao conectar/streaming: {e}")

st.markdown("---")
st.markdown("**Dica:** Se nenhum modelo aparecer na lista, abra o LM Studio → ‘Developer’ → ‘Start Server’ e garanta que há pelo menos um modelo carregado. Coloque o mesmo *Model Identifier* mostrado no LM Studio (ex.: `llama-3-8b-lexi-uncensored`).")
