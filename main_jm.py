# 🎭 Narrador JM — LM Studio + Google Sheets (Roleplay)
# -------------------------------------------------------------
# Requisitos:
#   pip install streamlit requests gspread oauth2client
# Secrets necessários (Streamlit):
#   SPREADSHEET_ID = "<ID da planilha>"
#   GOOGLE_CREDS_JSON = """
#     { "type": "service_account", "project_id": "...",
#       "private_key_id": "...",
#       "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
#       "client_email": "...@...iam.gserviceaccount.com", ... }
#   """
# Como usar:
# 1) Abra o LM Studio → aba "Developer" → "Start Server" (ou use um túnel Cloudflare/ngrok).
# 2) Carregue um modelo (ex.: llama-3-8b-lexi-uncensored) e confira o ID em /v1/models.
# 3) Rode:  streamlit run app_lmstudio_sheets_roleplay.py
# 4) No sidebar, defina a Base URL (ex.: http://127.0.0.1:1234/v1 ou https://xxxx.trycloudflare.com/v1).
# -------------------------------------------------------------

from __future__ import annotations
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

# ============================
# Config básica
# ============================
st.set_page_config(page_title="Narrador JM — LM Studio + Sheets", page_icon="🎭", layout="wide")

# Abas da planilha (serão criadas se não existirem)
TAB_INTERACOES = "interacoes_jm"           # cabeçalho: timestamp | role | content | model | base_url
TAB_PERFIL     = "perfil_jm"                # cabeçalho: timestamp | resumo
TAB_MEMORIAS   = "memoria_jm"               # cabeçalho: tipo | conteudo | timestamp
TAB_ML         = "memoria_longa_jm"         # cabeçalho: texto | embedding | tags | timestamp | score
TAB_TEMPLATES  = "templates_jm"             # cabeçalho: template | etapa | texto
TAB_FALAS_MARY = "falas_mary_jm"            # cabeçalho: fala
# ID padrão (fallback) da sua planilha — pode ser sobrescrito via secrets ou sidebar
PLANILHA_ID_PADRAO = "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"

# ============================
# Conexão Google Sheets
# ============================
try:
    import gspread
    from gspread.exceptions import APIError
    from oauth2client.service_account import ServiceAccountCredentials
    _HAS_SHEETS = True
except Exception:
    _HAS_SHEETS = False


def _get_spreadsheet_id() -> str:
    # Pode vir dos secrets, do sidebar, ou cair no ID padrão acima
    sid = st.session_state.get("SPREADSHEET_ID") or st.secrets.get("SPREADSHEET_ID", "").strip()
    return sid or PLANILHA_ID_PADRAO


def conectar_planilha():
    if not _HAS_SHEETS:
        return None
    try:
        creds_raw = st.session_state.get("GS_JSON") or st.secrets.get("GOOGLE_CREDS_JSON", "")
        if not creds_raw:
            st.info("Cole as credenciais JSON no sidebar ou defina GOOGLE_CREDS_JSON em st.secrets.")
            return None
        creds_dict = json.loads(creds_raw)
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sid = _get_spreadsheet_id()
        if not sid:
            st.warning("SPREADSHEET_ID não definido. Informe nos secrets ou no sidebar.")
            return None
        return client.open_by_key(sid)
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
            ws = planilha.add_worksheet(title=name, rows=5000, cols=12)
            # Cabeçalhos padrão por aba
            if name == TAB_INTERACOES:
                ws.append_row(["timestamp", "role", "content", "model", "base_url"])
            elif name == TAB_PERFIL:
                ws.append_row(["timestamp", "resumo"])
            elif name == TAB_MEMORIAS:
                ws.append_row(["tipo", "conteudo", "timestamp"])
            elif name == TAB_ML:
                ws.append_row(["texto", "embedding", "tags", "timestamp", "score"])
            elif name == TAB_TEMPLATES:
                ws.append_row(["template", "etapa", "texto"])
            elif name == TAB_FALAS_MARY:
                ws.append_row(["fala"])
            return ws
        except Exception as e:
            st.warning(f"Falha ao criar aba '{name}': {e}")
            return None


def gs_save_interaction(role: str, content: str, model: str = "", base_url: str = "") -> None:
    ws = _ws(TAB_INTERACOES)  # cria se necessário
    if not ws:
        st.warning("Planilha não conectada (interações).")
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, role, content, model, base_url]
    try:
        ws.append_row(row)
    except Exception as e:
        st.warning(f"Falha ao salvar interação no Sheets: {e}")


def gs_load_last_turns(n_turnos: int = 6) -> List[Dict[str, str]]:
    ws = _ws(TAB_INTERACOES, create_if_missing=False)
    if not ws:
        return []
    try:
        values = ws.get_all_values()
        if not values:
            return []
        header = values[0]
        rows = values[1:]
        # Índices tolerantes (suporta planilhas antigas com 3 colunas)
        def idx(col_name: str, default: int) -> int:
            try:
                return header.index(col_name)
            except ValueError:
                return default
        i_role = idx("role", 1)
        i_content = idx("content", 2)
        # Últimos N turnos = 2*N mensagens (user+assistant)
        total_msgs = max(0, n_turnos) * 2
        rows = rows[-total_msgs:] if total_msgs else []
        msgs: List[Dict[str, str]] = []
        for r in rows:
            try:
                role = (r[i_role] if i_role < len(r) else "user").strip()
                content = r[i_content] if i_content < len(r) else ""
                if role in ("user", "assistant", "system") and content:
                    msgs.append({"role": role, "content": content})
            except Exception:
                continue
        return msgs
    except Exception as e:
        st.warning(f"Falha ao ler interações do Sheets: {e}")
        return []


# ============================
# Texto/Roleplay helpers
# ============================
_DASHES = re.compile(r"(?:\s*--\s*|\s*–\s*|\s*—\s*)")
_SENT_END = re.compile(r"[.!?…](?=\s|$)")
_UPPER_OR_DASH = re.compile(r"([.!?…])\s+(?=(?:—|[A-ZÁÉÍÓÚÂÊÔÃÕÀÇ0-9]))")
SCENERY_WORD = re.compile(r"\b(c[ée]u|nuvens?|horizonte|luar|mar|onda?s?|pier|vento|brisa|chuva|garoa|sereno|paisage?m|cen[áa]rio)\b", re.I)
EXPL_PAT = re.compile(r"\b(mamilos?|genit[aá]lia|ere[cç][aã]o|penetra[cç][aã]o|boquete|gozada|gozo|sexo oral|chupar|enfiar)\b", re.I)


def roleplay_paragraphizer(t: str) -> str:
    if not t:
        return ""
    t = _DASHES.sub("\n— ", t)
    t = _UPPER_OR_DASH.sub(r"\1\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    linhas = [ln.strip() for ln in t.splitlines() if ln.strip()]
    out, buf, sent, chars = [], [], 0, 0
    MAX_SENT_PER_PARA = 2
    MAX_CHARS_PER_PARA = 240
    for ln in linhas:
        if ln.startswith("—"):
            if buf:
                out.append(" ".join(buf).strip()); out.append("")
                buf, sent, chars = [], 0, 0
            out.append(ln)
        else:
            buf.append(ln)
            sent += len(_SENT_END.findall(ln))
            chars += len(ln)
            if sent >= MAX_SENT_PER_PARA or chars >= MAX_CHARS_PER_PARA:
                out.append(" ".join(buf).strip()); out.append("")
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
    return t if lvl <= max_level else t


def _render_visible(t: str) -> str:
    t = sanitize_scenery_preserve_opening(t)
    t = roleplay_paragraphizer(t)
    t = sanitize_explicit(t, int(st.session_state.get("nsfw_max_level", 0)))
    return t


# ============================
# Prompt Builder
# ============================

def _fase_label(n: int) -> str:
    nomes = {0:"Estranhos",1:"Percepção",2:"Conhecidos",3:"Romance",4:"Namoro",5:"Compromisso / Encontro definitivo"}
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
    pedacos.append("Mary" + (f", {figurino}" if figurino else ""))
    if lugar:
        pedacos.append(lugar)
    return (". ".join([p for p in pedacos if p]) + ".").strip(".") + "." if pedacos else ""


def build_messages(ctx: Dict[str, Any], user_text: str, history_msgs: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": build_system_prompt(ctx)})
    if history_msgs:
        n = int(st.session_state.get("history_to_prompt", 6))
        msgs.extend(history_msgs[-n:] if n > 0 else [])
    abertura = build_opening_line(ctx)
    if abertura:
        msgs.append({"role": "system", "content": f"ABERTURA_SUGERIDA: {abertura}"})
    msgs.append({"role": "user", "content": user_text})
    return msgs


# ============================
# LM Studio — API compatível com OpenAI
# ============================

@st.cache_data(ttl=15, show_spinner=False)
def lms_list_models(base_url: str) -> List[str]:
    try:
        url = base_url.rstrip("/") + "/models"
        headers = {"Content-Type": "application/json"}
        if st.session_state.get("cf_client_id") and st.session_state.get("cf_client_secret"):
            headers["CF-Access-Client-Id"] = st.session_state["cf_client_id"]
            headers["CF-Access-Client-Secret"] = st.session_state["cf_client_secret"]
        auth = None
        if st.session_state.get("basic_user") and st.session_state.get("basic_pass"):
            auth = HTTPBasicAuth(st.session_state["basic_user"], st.session_state["basic_pass"])
        r = requests.get(url, timeout=8, headers=headers, auth=auth)
        r.raise_for_status()
        j = r.json()
        return [m.get("id") for m in j.get("data", []) if m.get("id")]
    except Exception:
        return []


def lms_health(base_url: str) -> Tuple[bool, str]:
    try:
        url = base_url.rstrip("/") + "/models"
        headers = {"Content-Type": "application/json"}
        if st.session_state.get("cf_client_id") and st.session_state.get("cf_client_secret"):
            headers["CF-Access-Client-Id"] = st.session_state["cf_client_id"]
            headers["CF-Access-Client-Secret"] = st.session_state["cf_client_secret"]
        auth = None
        if st.session_state.get("basic_user") and st.session_state.get("basic_pass"):
            auth = HTTPBasicAuth(st.session_state["basic_user"], st.session_state["basic_pass"])
        r = requests.get(url, timeout=8, headers=headers, auth=auth)
        r.raise_for_status()
        _ = r.json()
        return True, "Servidor acessível e respondendo em /models."
    except requests.ConnectionError as e:
        return False, f"Sem conexão com {base_url}. {e}"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response else ""
        return False, f"HTTP {code} de {base_url}. {e}"
    except Exception as e:
        return False, f"Falha inesperada ao acessar {base_url}: {e}"


def stream_chat_lmstudio(*, base_url: str, model: str, messages: List[Dict[str, str]],
                         temperature: float = 0.7, top_p: float = 1.0,
                         max_tokens: int = 1200, timeout: int = 300) -> str:
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
    if st.session_state.get("cf_client_id") and st.session_state.get("cf_client_secret"):
        headers["CF-Access-Client-Id"] = st.session_state["cf_client_id"]
        headers["CF-Access-Client-Secret"] = st.session_state["cf_client_secret"]
    auth = None
    if st.session_state.get("basic_user") and st.session_state.get("basic_pass"):
        auth = HTTPBasicAuth(st.session_state["basic_user"], st.session_state["basic_pass"])

    resposta_txt = ""
    with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout, auth=auth) as r:
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
            delta = j.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not delta:
                continue
            resposta_txt += delta
            if time.time() - last_update > 0.07:
                placeholder.markdown(_render_visible(resposta_txt) + "▌")
                last_update = time.time()
        placeholder.markdown(_render_visible(resposta_txt))
    return resposta_txt


# ============================
# Sidebar (único, sem duplicatas)
# ============================
with st.sidebar:
    st.title("🧭 Painel do Roteirista — LM Studio")

    st.markdown("### 🌐 Servidor LM Studio")
    default_base = st.session_state.get("lms_base_url", "http://127.0.0.1:1234/v1")
    base_url_input = st.text_input(
        "Base URL (LM Studio)", value=default_base, key="lms_base_url_input",
        help="Ex.: https://<seu>.trycloudflare.com/v1 ou http://127.0.0.1:1234/v1"
    )
    st.session_state["lms_base_url"] = base_url_input.strip()

    colhb1, colhb2 = st.columns([1,1])
    with colhb1:
        if st.button("Testar conexão", key="btn_test_conn"):
            ok, msg = lms_health(st.session_state["lms_base_url"]) 
            (st.success if ok else st.error)(msg)
            try:
                lms_list_models.clear()
            except Exception:
                pass
    with colhb2:
        st.caption("Se usar túnel, inclua /v1 no final.")

    modelos = lms_list_models(st.session_state["lms_base_url"]) or []
    modelo_escolhido = st.selectbox("🤖 Modelo (LM Studio)", modelos or ["<digite manualmente>"], key="lm_model_select")
    if modelo_escolhido == "<digite manualmente>":
        modelo_escolhido = st.text_input("Model identifier (LM Studio)", value=st.session_state.get("modelo_escolhido", "llama-3-8b-lexi-uncensored"), key="lm_model_identifier_input")
    st.session_state["modelo_escolhido"] = modelo_escolhido

    st.markdown("#### 📋 /v1/models")
    if st.button("Mostrar /v1/models", key="btn_show_models"):
        try:
            url = st.session_state["lms_base_url"].rstrip("/") + "/models"
            headers = {"Content-Type": "application/json"}
            if st.session_state.get("cf_client_id") and st.session_state.get("cf_client_secret"):
                headers["CF-Access-Client-Id"] = st.session_state["cf_client_id"]
                headers["CF-Access-Client-Secret"] = st.session_state["cf_client_secret"]
            auth = None
            if st.session_state.get("basic_user") and st.session_state.get("basic_pass"):
                auth = HTTPBasicAuth(st.session_state["basic_user"], st.session_state["basic_pass"])
            r = requests.get(url, timeout=10, headers=headers, auth=auth)
            r.raise_for_status()
            st.code(json.dumps(r.json(), indent=2), language="json")
        except Exception as e:
            st.error(f"Falha ao consultar /models: {e}")

    st.markdown("### 🔒 Segurança (opcional)")
    st.text_input("CF-Access-Client-Id", key="cf_client_id")
    st.text_input("CF-Access-Client-Secret", type="password", key="cf_client_secret")
    st.text_input("Basic auth - usuário (ngrok)", key="basic_user")
    st.text_input("Basic auth - senha (ngrok)", type="password", key="basic_pass")

    st.markdown("---")
    st.markdown("### 💾 Memória (Google Sheets)")
    st.text_input("Spreadsheet ID (override opcional)", value=_get_spreadsheet_id(), key="SPREADSHEET_ID")
    st.text_area("Credenciais JSON (service account)", value=st.session_state.get("GS_JSON",""), key="GS_JSON", height=120, help="Cole aqui o JSON inteiro do service account. A planilha precisa estar compartilhada com o client_email.")
    st.slider("Interações anteriores no prompt", 0, 20, value=int(st.session_state.get("history_to_prompt", 6)), key="history_to_prompt")
    if st.button("Testar Google Sheets", key="btn_test_gs"):
        # Reconnect com os valores atuais do sidebar
        planilha = conectar_planilha()
        ok = bool(planilha and _ws(TAB_INTERACOES))
        (st.success if ok else st.error)(f"{'OK' if ok else 'Falha'} — ID: {_get_spreadsheet_id() or '(vazio)'}")

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
    st.select_slider("Fase do romance", options=[0,1,2,3,4,5], value=int(st.session_state.get("fase", 0)), format_func=_fase_label, key="fase")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("➕ Avançar 1 fase", key="btn_avanca_fase"):
            st.session_state.fase = min(5, int(st.session_state.get("fase", 0)) + 1)
    with c2:
        if st.button("↺ Reiniciar (0)", key="btn_reset_fase"):
            st.session_state.fase = 0

    st.markdown("---")
    st.markdown("### ⏱️ Comprimento/timeout")
    st.slider("Max tokens da resposta", 256, 2500, value=int(st.session_state.get("max_tokens_rsp", 1200)), step=32, key="max_tokens_rsp")
    st.slider("Timeout (segundos)", 60, 600, value=int(st.session_state.get("timeout_s", 300)), step=10, key="timeout_s")

    st.markdown("---")
    st.markdown("### 🧩 Contexto da Cena (opcional)")
    st.text_input("Tempo (ex.: Domingo de manhã)", key="tempo")
    st.text_input("Lugar (ex.: Jacaraípe / em casa)", key="lugar")
    st.text_input("Figurino (ex.: short jeans e regata)", key="figurino")


# ============================
# Corpo principal — Chat Roleplay
# ============================
st.title("🎭 Narrador JM — Roleplay (LM Studio)")
st.caption("Chat contínuo com memória no Google Sheets (aba interacoes_jm).")

if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, str]] = []

# Render histórico local (desta sessão)
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(_render_visible(m["content"]) if m["role"] == "assistant" else m["content"])

user_text = st.chat_input("O que acontece agora?")
col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("🧹 Limpar chat", key="btn_clear_chat"):
        st.session_state.chat = []
        st.experimental_rerun()
with col_b:
    st.caption(f"🔌 Base URL: {st.session_state.get('lms_base_url', '(não definido)')}")

if user_text:
    # Mostra imediatamente a fala do usuário
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.chat.append({"role": "user", "content": user_text})

    # Contexto
    ctx = {
        "modo_resposta": st.session_state.get("modo_resposta"),
        "estilo_escrita": st.session_state.get("estilo_escrita"),
        "nsfw_max_level": st.session_state.get("nsfw_max_level"),
        "modo_sintonia": st.session_state.get("modo_sintonia"),
        "ritmo_cena": st.session_state.get("ritmo_cena"),
        "finalizacao_modo": st.session_state.get("finalizacao_modo"),
        "usar_falas_mary": st.session_state.get("usar_falas_mary"),
        "fase": st.session_state.get("fase"),
        "app_bloqueio_intimo": st.session_state.get("app_bloqueio_intimo", True),
        "app_emocao_oculta": st.session_state.get("app_emocao_oculta", "nenhuma"),
        "tempo": st.session_state.get("tempo"),
        "lugar": st.session_state.get("lugar"),
        "figurino": st.session_state.get("figurino"),
    }

    # Carrega últimas N interações da planilha
    hist_msgs = gs_load_last_turns(int(st.session_state.get("history_to_prompt", 6)))

    # Monta mensagens
    try:
        messages = build_messages(ctx, user_text, hist_msgs)
    except TypeError:
        base_msgs = [{"role": "system", "content": build_system_prompt(ctx)}]
        base_msgs += hist_msgs
        base_msgs.append({"role": "user", "content": user_text})
        messages = base_msgs

    # Resposta (streaming)
    with st.chat_message("assistant"):
        with st.status("Gerando…", expanded=False):
            try:
                resposta_txt = stream_chat_lmstudio(
                    base_url=st.session_state.get("lms_base_url", "http://127.0.0.1:1234/v1"),
                    model=st.session_state.get("modelo_escolhido", "llama-3-8b-lexi-uncensored"),
                    messages=messages,
                    temperature=float(st.session_state.get("temperature", 0.7)),
                    top_p=float(st.session_state.get("top_p", 1.0)),
                    max_tokens=int(st.session_state.get("max_tokens_rsp", 1200)),
                    timeout=int(st.session_state.get("timeout_s", 300)),
                )
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code if e.response else ''}: {e}")
                resposta_txt = ""
            except Exception as e:
                st.error(f"Erro ao conectar/streaming: {e}")
                resposta_txt = ""

        if resposta_txt:
            st.markdown(_render_visible(resposta_txt))
            # Salva no Sheets (se disponível)
            try:
                gs_save_interaction("user", user_text, st.session_state.get("modelo_escolhido", ""), st.session_state.get("lms_base_url", ""))
                gs_save_interaction("assistant", resposta_txt, st.session_state.get("modelo_escolhido", ""), st.session_state.get("lms_base_url", ""))
            except Exception as e:
                st.warning(f"Falha ao salvar interações: {e}")
            # Atualiza histórico da sessão
            st.session_state.chat.append({"role": "assistant", "content": resposta_txt})

st.markdown("---")
st.markdown("**Dica:** Se nenhum modelo aparecer na lista, abra o LM Studio → Developer → Start Server e carregue um modelo. Se estiver em túnel, cole a URL pública com */v1*.\n**Memória:** mensagens são gravadas em *interacoes_jm*. As outras abas são criadas conforme necessidade.")


# ============================
# Diretrizes (UI) + Memórias (UI) — adicionados ao final para evitar chaves duplicadas
# ============================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📜 Diretrizes de Interação")
    # Carrega uma vez das planilhas, se ainda não houver no estado
    if "diretrizes_text" not in st.session_state:
        try:
            st.session_state.diretrizes_text = sheet_templates_get("diretrizes")
        except Exception:
            st.session_state.diretrizes_text = ""
    st.session_state.diretrizes_text = st.text_area(
        "Diretrizes (aplicadas como system message)",
        value=st.session_state.get("diretrizes_text", ""),
        key="diretrizes_text_ui",
        height=160,
        help="Estas regras serão injetadas como system message em cada geração."
    )
    cgd1, cgd2 = st.columns([1,1])
    with cgd1:
        if st.button("⤵️ Recarregar do Sheets", key="btn_load_diretrizes"):
            st.session_state.diretrizes_text = sheet_templates_get("diretrizes")
            st.experimental_rerun()
    with cgd2:
        if st.button("💾 Salvar no Sheets", key="btn_save_diretrizes"):
            sheet_templates_set("diretrizes", st.session_state.get("diretrizes_text", st.session_state.get("diretrizes_text_ui","")))
            st.success("Diretrizes salvas em templates_jm.")

    st.markdown("---")
    st.markdown("### 🧠 Memória (resumo e fatos)")
    st.slider("Interações para sintetizar", 2, 30, value=int(st.session_state.get("mem_summarize_n", 8)), key="mem_summarize_n")
    st.checkbox("Indexar memórias na aba memoria_longa_jm (requer modelo de embedding)", key="do_index_long", value=bool(st.session_state.get("do_index_long", False)))
    st.text_input("Modelo de embedding (LM Studio)", key="emb_model_override", value=st.session_state.get("emb_model_override", ""), help="Se vazio, tento detectar automaticamente um modelo com 'embedding' pelo /v1/models.")
    cm1, cm2 = st.columns([1,1])
    with cm1:
        if st.button("📄 Gerar RESUMO (perfil_jm)", key="btn_make_resumo"):
            hist = gs_load_last_turns(int(st.session_state.get("mem_summarize_n", 8)))
            if not hist:
                st.error("Sem histórico suficiente em interacoes_jm.")
            else:
                txt = gerar_resumo_perfil(st.session_state.get("lms_base_url","http://127.0.0.1:1234/v1"), st.session_state.get("modelo_escolhido","llama-3-8b-lexi-uncensored"), hist)
                if txt:
                    sheet_perfil_append(txt)
                    st.success("Resumo salvo em perfil_jm.")
    with cm2:
        if st.button("🧩 Extrair MEMÓRIAS (memoria_jm)", key="btn_make_memorias"):
            hist = gs_load_last_turns(int(st.session_state.get("mem_summarize_n", 8)))
            if not hist:
                st.error("Sem histórico suficiente em interacoes_jm.")
            else:
                itens = extrair_memorias(st.session_state.get("lms_base_url","http://127.0.0.1:1234/v1"), st.session_state.get("modelo_escolhido","llama-3-8b-lexi-uncensored"), hist)
                if itens:
                    sheet_memorias_append_many(itens)
                    st.success(f"{len(itens)} memórias salvas em memoria_jm.")

# ============================
# Override: build_messages para injetar diretrizes como system message
# ============================

def build_messages(ctx: Dict[str, Any], user_text: str, history_msgs: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": build_system_prompt(ctx)})
    if history_msgs:
        n = int(st.session_state.get("history_to_prompt", 6))
        msgs.extend(history_msgs[-n:] if n > 0 else [])
    # Diretrizes de interação vindas do sidebar/Sheets
    dtx = st.session_state.get("diretrizes_text") or sheet_templates_get("diretrizes")
    if dtx:
        msgs.append({
            "role": "system",
            "content": f"DIRETRIZES DE INTERAÇÃO (siga estritamente):\n{dtx}"
        })
    abertura = build_opening_line(ctx)
    if abertura:
        msgs.append({"role": "system", "content": f"ABERTURA_SUGERIDA: {abertura}"})
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ============================
# (Opcional) Embeddings — indexar memoria_longa_jm
# ============================

def _auto_embedding_model() -> str:
    if st.session_state.get("emb_model_override"):
        return st.session_state["emb_model_override"].strip()
    try:
        models = lms_list_models(st.session_state.get("lms_base_url","http://127.0.0.1:1234/v1"))
        for m in models:
            if "embed" in m.lower() or "embedding" in m.lower():
                return m
    except Exception:
        pass
    return ""


def lms_embed(base_url: str, model: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    payload = {"model": model, "input": texts}
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if st.session_state.get("cf_client_id") and st.session_state.get("cf_client_secret"):
        headers["CF-Access-Client-Id"] = st.session_state["cf_client_id"]
        headers["CF-Access-Client-Secret"] = st.session_state["cf_client_secret"]
    auth = None
    if st.session_state.get("basic_user") and st.session_state.get("basic_pass"):
        auth = HTTPBasicAuth(st.session_state["basic_user"], st.session_state["basic_pass"])
    r = requests.post(url, json=payload, headers=headers, timeout=30, auth=auth)
    r.raise_for_status()
    j = r.json()
    datas = j.get("data", [])
    return [d.get("embedding", []) for d in datas]

# Quando salvar memórias curtas, opcionalmente indexa
if st.session_state.get("do_index_long") and planilha:
    try:
        ws_mem = _ws(TAB_MEMORIAS, create_if_missing=False)
        ws_long = _ws(TAB_ML, create_if_missing=True)
        if ws_mem and ws_long:
            vals = ws_mem.get_all_values()
            if vals and len(vals) > 1:
                header, rows = vals[0], vals[1:]
                try:
                    i_conteudo = header.index("conteudo")
                except ValueError:
                    i_conteudo = 1
                textos = [r[i_conteudo] for r in rows if len(r) > i_conteudo][:32]
                if textos:
                    emb_model = _auto_embedding_model()
                    if emb_model:
                        embs = lms_embed(st.session_state.get("lms_base_url","http://127.0.0.1:1234/v1"), emb_model, textos)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for txt, vec in zip(textos, embs):
                            ws_long.append_row([txt, json.dumps(vec), "auto", ts, 1.0])
                        st.toast(f"Indexadas {len(embs)} memórias na memoria_longa_jm.")
    except Exception as e:
        st.info(f"Indexação opcional de memória longa não concluída: {e}")

