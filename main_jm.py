# -*- coding: utf-8 -*-
# main_jm.py — build estável
# Requisitos: streamlit, requests, gspread, oauth2client, numpy, openai
# Segredos esperados em st.secrets:
# - OPENAI_API_KEY (para embeddings)
# - OPENROUTER_API_KEY (se usar OpenRouter)
# - TOGETHER_API_KEY (se usar Together)
# - GOOGLE_CREDS_JSON (service account JSON do Google)
# Google Sheet esperado (por key) em SHEET_KEY (ou substitua abaixo):
#   abas: interacoes_jm, memoria_longa_jm, perfil_jm (p/ resumos)

import streamlit as st
import requests
import gspread
import json
import time
import re
import numpy as np
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --- Correção de texto corrompido (mojibake) ---
# Se aparecerem sequências como "Ã", "â", etc., tentamos reparar
# strings que foram renderizadas com encoding errado.
def fix_mojibake(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    # Heurística simples: se há padrões típicos de UTF-8 mal decodificado,
    # tentamos re-interpretar como latin-1 -> utf-8.
    if ("Ã" in s) or ("â" in s) or ("Â" in s):
        try:
            return s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return s
    return s

# ===============
# CONFIG BÁSICA
# ===============
st.set_page_config(page_title="Mary / Jânio — Novela Interativa", page_icon="🌹", layout="wide")
SHEET_KEY = st.secrets.get("SHEET_KEY", "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ===============
# UTIL: PLANILHA
# ===============
@st.cache_resource(show_spinner=False)
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"]) if "GOOGLE_CREDS_JSON" in st.secrets else None
        if not creds_dict:
            st.stop()
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(SHEET_KEY)
        return sh
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

sh = conectar_planilha()

# ---------------
# Abas helpers
# ---------------
@st.cache_data(ttl=15, show_spinner=False)
def get_ws(nome):
    if not sh:
        return None
    try:
        return sh.worksheet(nome)
    except Exception:
        return None

# ===============
# MEMÓRIA LONGA (embeddings)
# ===============
from openai import OpenAI
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def _embedding(texto: str):
    if not _client:
        return None
    try:
        resp = _client.embeddings.create(model="text-embedding-3-small", input=texto)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        st.warning(f"Falha no embedding: {e}")
        return None


def salvar_memoria_longa(texto: str, tags: str = "auto"):
    ws = get_ws("memoria_longa_jm")
    if not ws:
        return
    emb = _embedding(texto) if texto else None
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, texto, tags, json.dumps(emb.tolist() if emb is not None else []), 1]
    try:
        ws.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"Não foi possível salvar memória longa: {e}")


def buscar_memoria_longa(query: str, k: int = 3, limiar: float = 0.75):
    ws = get_ws("memoria_longa_jm")
    if not ws:
        return []
    try:
        vals = ws.get_all_records()
    except Exception:
        return []
    if not vals:
        return []
    qemb = _embedding(query)
    if qemb is None:
        return []
    # cosine
    def cos(a, b):
        a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    scored = []
    for r in vals:
        try:
            emb = json.loads(r.get("embedding_json", "[]"))
            if emb:
                s = cos(qemb, emb)
                if s >= limiar:
                    scored.append((s, r.get("texto", "")))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

# ===============
# INTERAÇÕES (histórico curto)
# ===============

def salvar_interacao(role: str, content: str):
    ws = get_ws("interacoes_jm")
    if not ws:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, role, content], value_input_option="RAW")
    except Exception as e:
        st.warning(f"Falha ao salvar interação: {e}")

@st.cache_data(ttl=10, show_spinner=False)
def carregar_ultimas_interacoes(n=20):
    ws = get_ws("interacoes_jm")
    if not ws:
        return []
    try:
        data = ws.get_all_records()
        return data[-n:]
    except Exception:
        return []

# ===============
# RESUMO (aba perfil_jm) — aparece DEPOIS do histórico
# ===============

def salvar_resumo(resumo: str):
    ws = get_ws("perfil_jm")
    if not ws:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Tenta detectar se a planilha usa o formato antigo (resumo na coluna 7)
        vals = ws.get_all_values()
        if vals and len(vals[0]) >= 7:
            # Formato antigo (7 colunas): timestamp na 6, resumo na 7
            row = ["", "", "", "", "", ts, resumo]
        else:
            # Formato simples (2 colunas): timestamp, resumo
            row = [ts, resumo]
        ws.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"Falha ao salvar resumo: {e}")

@st.cache_data(ttl=20, show_spinner=False)
def carregar_ultimo_resumo():
    ws = get_ws("perfil_jm")
    if not ws:
        return None
    try:
        vals = ws.get_all_values()
        if len(vals) < 2:
            return None
        # Pega a última linha não vazia
        last = vals[-1]
        if len(last) >= 7 and last[6].strip():
            return last[6].strip()
        if len(last) >= 2 and last[1].strip():
            return last[1].strip()
        # Se a última linha estiver vazia, tenta procurar de trás pra frente
        for row in reversed(vals[1:-1]):
            if len(row) >= 7 and row[6].strip():
                return row[6].strip()
            if len(row) >= 2 and row[1].strip():
                return row[1].strip()
    except Exception:
        pass
    return None

# ===============
# REGRAS DE NARRATIVA (Guardrails)
# ===============(ttl=20, show_spinner=False)


# ===============
# REGRAS DE NARRATIVA (Guardrails)
# ===============
VIRGINIDADE_MARY = True  # Mary permanece virgem por vontade própria
JANIO_LIVRE = True       # Jânio pode ter relações com qualquer personagem

REGRAS_MARY = (
    "Mary é virgem e escolhe manter-se casta até o encontro definitivo com Jânio. "
    "Ela pode namorar, beijar, flertar e viver tensão romântica, mas recusa sexo. "
    "Nunca descreva Mary transando, nem algo que implique penetração ou perda da virgindade. "
    "Se a cena ameaçar ultrapassar limites, Mary impõe limites com firmeza e elegância."
)

REGRAS_JANIO = (
    "Jânio não tem essa limitação: ele pode ter relações consensuais com outros personagens, "
    "desde que a cena mantenha o bom gosto e evite descrições gráficas explícitas."
)

# Prompt base
BASE_SYSTEM = f"""
Você é o narrador de uma novela em português do Brasil. Escreva em 3ª pessoa a narração e use 1ª pessoa para falas e pensamentos dos personagens.
Mantenha continuidade com o histórico e memórias fornecidas. Seja sensorial sem apelar para pornografia explícita.

REGRAS FIXAS:
- {REGRAS_MARY if VIRGINIDADE_MARY else ''}
- {REGRAS_JANIO if JANIO_LIVRE else ''}
- Não invente falas do usuário. O usuário só fala quando ele enviar mensagem.
- Evite comandos técnicos ([SFX], (cut), etc.).
""".strip()

# ===============
# MODELOS E CHATS
# ===============

# Listas completas (como na sua lousa)
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
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
}

# Dicionário unificado esperado pelo resto do app
MODELOS = {}
for nome, mid in MODELOS_OPENROUTER.items():
    MODELOS[nome] = {"prov": "openrouter", "id": mid}
for nome, mid in MODELOS_TOGETHER_UI.items():
    MODELOS[nome] = {"prov": "together", "id": mid}

def _stream_openrouter(model_id: str, messages, temperature=0.85, max_tokens=700):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    resposta_txt = ""
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j["choices"][0]["delta"].get("content", "")
                if delta:
                    resposta_txt += delta
                    yield resposta_txt
            except Exception:
                continue


def _stream_together(model_id: str, messages, temperature=0.85, max_tokens=700):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    resposta_txt = ""
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        # Tratar 400 gracefully
        if r.status_code == 400:
            raise RuntimeError(f"Together 400: {r.text}")
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.strip() == "data: [DONE]":
                break
            if raw.startswith("data:"):
                data = raw[5:].strip()
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"].get("content", "")
                    # Perplexity pode devolver <think> … </think>. NÃO remova — exiba completo.
                    if delta:
                        resposta_txt += delta
                        yield resposta_txt
                except Exception:
                    continue


def gerar_resposta(modelo_cfg, historico_msgs):
    prov = modelo_cfg["prov"]
    model_id = modelo_cfg["id"]

    # Recuperar memórias longas relevantes a partir da última entrada do usuário
    ultima_usuario = next((m["content"] for m in reversed(historico_msgs) if m["role"] == "user"), "")
    mem_longas = buscar_memoria_longa(ultima_usuario, k=3, limiar=0.75)
    bloco_memoria = ("\n\nMemórias relevantes:\n" + "\n".join([f"- {t}" for t in mem_longas])) if mem_longas else ""

    system = {"role": "system", "content": BASE_SYSTEM + bloco_memoria}
    msgs = [system] + historico_msgs

    if prov == "openrouter":
        return _stream_openrouter(model_id, msgs)
    elif prov == "together":
        return _stream_together(model_id, msgs)
    else:
        raise ValueError("Provedor desconhecido")

# ===============
# STATE INICIAL
# ===============
if "hist" not in st.session_state:
    # hist guarda apenas user/assistant desta sessão (na tela)
    st.session_state.hist = []
if "modelo_nome" not in st.session_state:
    st.session_state.modelo_nome = list(MODELOS.keys())[0]

# ===============
# UI LATERAL
# ===============
with st.sidebar:
    st.subheader("Config")
    st.session_state.modelo_nome = st.selectbox("Modelo", list(MODELOS.keys()), index=list(MODELOS.keys()).index(st.session_state.modelo_nome))
    st.caption("Trocar de modelo NÃO apaga histórico.")

    st.markdown("---")
    st.subheader("Memória Longa")
    if st.button("Salvar última resposta como memória", use_container_width=True):
        ult = next((m["content"] for m in reversed(st.session_state.hist) if m["role"] == "assistant"), "")
        if ult:
            salvar_memoria_longa(ult, tags="auto")
            st.success("Memória salva.")
        else:
            st.info("Sem resposta para salvar ainda.")

    st.markdown("---")
    st.subheader("Resumo do capítulo")
    if st.button("Gerar e salvar resumo", use_container_width=True):
        # Usa o próprio modelo atual para resumir as últimas interações da sessão
        trecho = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.hist[-10:]]) or "(vazio)"
        user_prompt = (
            "Resuma como capítulo de novela, coeso e elegante, sem pornografia, mantendo tom emocional.\n\n" + trecho
        )
        modelo_cfg = MODELOS[st.session_state.modelo_nome]
        try:
            placeholder = st.empty()
            out = ""
            spinner = st.spinner("Resumindo...")
            with spinner:
                for parcial in gerar_resposta(modelo_cfg, [
                    {"role": "user", "content": user_prompt}
                ]):
                    out = parcial
                    placeholder.markdown(out)
            if out.strip():
                salvar_resumo(out.strip())
                st.success("Resumo salvo na aba perfil_jm.")
        except Exception as e:
            st.error(f"Erro ao resumir: {e}")

# ===============
# HISTÓRICO DA PLANILHA + SESSÃO
# ===============
col_hist, = st.columns(1)
with col_hist:
    st.title("🌹 Mary / 🎸 Jânio — Novela Interativa")

# Render histórico curto recente da planilha (somente exibição)
historico_planilha = carregar_ultimas_interacoes(15)
for m in historico_planilha:
    with st.chat_message(m.get("role", "user")):
        st.markdown(fix_mojibake(m.get("content", "")))

# Render histórico da sessão atual
for m in st.session_state.hist:
    with st.chat_message(m["role"]):
        st.markdown(fix_mojibake(m["content"])) 

# Depois de TUDO, mostra o último resumo
ultimo_resumo = carregar_ultimo_resumo()
if ultimo_resumo:
    with st.chat_message("assistant"):
        st.markdown("### 🧠 Resumo do capítulo anterior\n\n" + fix_mojibake(ultimo_resumo))

# ===============
# ENTRADA DO USUÁRIO
# ===============
entrada = st.chat_input("Digite sua cena / direção narrativa...")
if entrada:
    # salva user na planilha e no estado
    st.chat_message("user").markdown(entrada)
    st.session_state.hist.append({"role": "user", "content": entrada})
    salvar_interacao("user", entrada)

    # montar histórico para envio ao modelo (usa apenas a sessão atual + 6 últimas da planilha p/ contexto)
    contexto_base = []
    for m in historico_planilha[-6:]:
        contexto_base.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    contexto = contexto_base + st.session_state.hist[-8:]

    modelo_cfg = MODELOS[st.session_state.modelo_nome]

    # Gera resposta streaming com fallback
    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        try:
            for parcial in gerar_resposta(modelo_cfg, contexto):
                resposta_txt = parcial
                placeholder.markdown(resposta_txt if resposta_txt.strip() else "[Gerando…]")
            # fallback se veio vazio
            if not resposta_txt.strip():
                # tentativa não-stream
                prov = modelo_cfg["prov"]; model_id = modelo_cfg["id"]
                if prov == "openrouter":
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                else:
                    url = "https://api.together.xyz/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [{"role": "system", "content": BASE_SYSTEM}] + contexto,
                    "temperature": 0.85,
                    "max_tokens": 700,
                    "stream": False,
                }
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    j = r.json()
                    resposta_txt = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    raise RuntimeError(f"Fallback falhou: {r.status_code} - {r.text}")
                placeholder.markdown(resposta_txt or "[Sem conteúdo]")
        except Exception as e:
            placeholder.markdown(f"[Erro: {e}]")
            resposta_txt = f"[Erro: {e}]"

    # Salvar resposta (sempre)
    st.session_state.hist.append({"role": "assistant", "content": resposta_txt})
    salvar_interacao("assistant", resposta_txt)

    # Opcional: salvar pedacinhos em memória longa automaticamente
    if len(resposta_txt) > 300:
        try:
            salvar_memoria_longa(resposta_txt[:1200], tags="auto")
        except Exception:
            pass

# ===============
# FIM
# ===============
