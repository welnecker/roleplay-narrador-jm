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
from datetime import datetime
from typing import Dict, List, Any

import gspread
import requests
import streamlit as st
from gspread.exceptions import APIError, GSpreadException
from oauth2client.service_account import ServiceAccountCredentials
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Narrador JM — Clean Messages", page_icon="🎬")


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

def silenciar_janio(txt: str) -> str:
    if not txt:
        return txt
    out: List[str] = []
    ctx = 0  # >0 indica contexto de mensagem atribuída ao Jânio nas próximas linhas
    for line in txt.splitlines():
        raw = line.strip()
        low = raw.lower()

        # Gatilhos de início de bloco: "Jânio.", "**Jânio.**", "Mensagem de Jânio…"
        if low in ('jânio.', 'janio.', '**jânio.**', '**janio.**') or \
           low.startswith('mensagem de jânio') or low.startswith('mensagens de jânio'):
            out.append('_Uma notificação de Jânio chega ao celular de Mary._')
            ctx = 3
            continue

        # Linha "Jânio: ..."
        if low.startswith('jânio:') or low.startswith('janio:'):
            out.append('_[Conteúdo de Jânio omitido]_')
            continue

        # Durante contexto, suprimir citações e listas (provável conteúdo dele)
        if ctx > 0 and _is_quoted_or_bulleted(line):
            out.append('_[Conteúdo de Jânio omitido]_')
            ctx -= 1
            continue

        # Heurística extra: fala com travessão que claramente soa como dele
        if re.match(r'^\s*—\s*["“].*["”]\s*$', line) and ('donisete' in low or 'arquiteto' in low):
            out.append('_[Jânio reage sem falar]_')
            continue

        # Encerrar contexto em linha vazia
        if ctx > 0 and not raw:
            ctx = 0

        out.append(line)

    return "\n".join(out)

# =================================================================================
# Config Planilha
# =================================================================================
PLANILHA_ID = (
    st.secrets.get("JM_SHEET_ID")
    or st.secrets.get("SPREADSHEET_ID")
    or "1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM"
).strip()
TAB_INTERACOES = "interacoes_jm"

# =================================================================================
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
# Persona — system mínimo (como solicitado)
# =================================================================================
PERSONA_MARY = """nome: Mary Massariol
idade: 20 anos
Corpo: Cabelos negros e volumosos; olhos verdes; seios médios e empinados; quadril largo; Barriga tonificada; Bumbum carnudo, redondo e firme; coxas grossas e torneadas;
Cursa engenharia civil na Universidade Federal do Espírito Santo (UFES); tem uma moto 500cc; mora com a mãe, Joselina Massariol, no apartamento 202, Rua Beethoven, em Laranjeiras;
É inteligente; é romântica; Mary é virgem por opção; gosta de praia; gosta de baladas com amigas; gosta de academia ao ar livre; gosta de flertar; não tolera grosserias; não tolera cantadas baratas;
Adora dançar; é sensual; às vezes anda sem sutiã sob a roupa; Silvia e Alexandra são amigas de faculdade; Ricardo é namorado ciumento e possessivo; quer encontrar o verdadeiro amor;
Frequenta a Praia de Camburi; costuma almoçar ou jantar no restaurante Partido Alto em Camburi; adora moqueca capixaba e camarões fritos; frequenta baladas no Serra Bella Clube; Motel Status é onde os jovens transam;
Jânio Donisete; Orfeu é o cão labrador de Jânio; Jânio mora em Camburi, no edifício Alianz, apartamento 2002, no 20º andar; Jânio é arquiteto bem sucedido; Jânio tem 32 anos de idade;

[REGRAS DE REALISMO]
— Sem onisciência: ninguém sabe fatos que não foram ditos, vistos em cena ou lembrados do histórico (ex.: prato favorito, @ de redes, destino exato).
— Conhecimento plausível só por: diálogo explícito, pistas observáveis ou algo já estabelecido no chat.
— Sem atalhos milagrosos: nada de “resolveu em 5 minutos”, “em 10 segundos”, ou trocas instantâneas sob pressão. Se houver pressa, use “alguns instantes” e consequência plausível.
— Conflitos evoluem em degraus: tensão > reação > consequência. Não salte para soluções completas sem passos intermédios.
— Mary mantém limites e segurança: recusa grosseria, busca apoio das amigas/ambiente quando necessário; evita risco físico.
— Consistência temporal: preserve o “relógio” da cena. Se for dia na praia e alguém chamar para uma balada noturna, trate como PROPOSTA para mais tarde; a mudança de tempo/lugar só ocorre após aceitação explícita e com marcador claro (ex.: “mais tarde”, “ao anoitecer”, “à noite, no Serra Bella”).
— Tempo/Lugar não avançam sozinhos: não mude cenário/tempo sem um gatilho (convite aceito, indicação do usuário ou marcador textual explícito).
— Convite ≠ presença: convites (ex.: Partido Alto/Serra Bella) soam como sugestão; só viram encontro após aceitação e transição plausível.
— Instagram/contato exigem gesto plausível (troca combinada, QR, anotação). Evite “digitou @ em 1s” em público com ameaça próxima.
— Evite adjetivos grandiloquentes repetidos; privilegie ações simples e coerentes.
— Auto-checagem: antes de finalizar, reescreva qualquer trecho que viole as regras acima. Não mencione este checklist ao responder.

[NARRAÇÃO EM 3ª PESSOA — TOM SUTILMENTE SENSUAL]
— Em narrações em 3ª pessoa, permita leve sensualidade **respeitosa**.
— Use referências diretas ao **perfil físico** de Mary já descrito (cabelos negros volumosos, olhos verdes, seios médios, quadril largo, barriga tonificada, bumbum redondo e firme, coxas torneadas). **Não cite** movimento, luz, tecido, clima ou cenário.
— Máx. **1 detalhe físico por parágrafo**; evite listas e repetição de adjetivos.
— Evite metáforas e sensorial de ambiente; foque em ações e estado emocional; 
— Consentimento e respeito sempre; nada de coerção. Mantenha as **REGRAS DE REALISMO** e o **ESTILO DE RESPOSTA**.

[REGRAS DE DIÁLOGO — JÂNIO (SILENCIAR)]
— Não escreva falas de Jânio (sem linhas com travessão/aspas atribuídas a ele).
— Não transcreva mensagens/textos enviados por Jânio; resuma como notificação recebida.
— Mostre apenas reações/gestos de Jânio em narração (olhar, postura, ação), sem fala.
— As falas de Jânio pertencem ao usuário: só existem se o usuário escrever explicitamente.
— Se gerar fala de Jânio por engano, reescreva removendo-a antes de finalizar a resposta.




[PROGRESSÃO DE INTIMIDADE]
— Evolução gradual no tempo: flerte → toques leves → beijo → (se ambos quiserem) avanço íntimo.
— Consentimento claro e revogável; use sinais explícitos (ex.: "posso te beijar?", "tudo bem continuar?").
— Marque passagem de tempo/contexto (ex.: "mais tarde", "dias depois"); sem saltos bruscos.
— Sexo explícito (sem fade-to-black): permitido detalhes anatômicos, posições, fluidos ou nudez explícita.
— Mantenha as REGRAS DE REALISMO, a NARRAÇÃO EM 3ª PESSOA e o ESTILO DE RESPOSTA.


[ESTILO DE RESPOSTA]
— Seja ~30% mais concisa que o natural.
— Máx. 5 parágrafos por turno, até 2 frases por parágrafo.
— Evite floreios; foque em ação/diálogo e informação útil.
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
            [ts, session_id, provider, model, role, content], value_input_option="USER_ENTERED"
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


# =================================================================================
# Helpers — mensagens mínimas
# =================================================================================

def build_minimal_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Retorna apenas: system(persona) + histórico bruto user/assistant."""
    return [{"role": "system", "content": PERSONA_MARY}] + history


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

    if st.button("🗑️ Resetar chat"):
        st.session_state.chat.clear()
        st.rerun()

# Render histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]).container():
        st.markdown(m["content"])  # sem filtros extras

# Entrada
if user_msg := st.chat_input("Fale com a Mary..."):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    st.session_state.chat.append({"role": "user", "content": user_msg})
    # Mantém apenas as últimas 30 interações na tela
    if len(st.session_state.chat) > 30:
        st.session_state.chat = st.session_state.chat[-30:]
    salvar_interacao(ts, st.session_state.session_id, prov, model_id, "user", user_msg)

    messages = build_minimal_messages(st.session_state.chat)

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
                # Mostra texto em tempo real já filtrado (sem falas/mensagens do Jânio)
                ph.markdown(silenciar_janio(answer) + "▌")
        except Exception as e:
            answer = f"[Erro ao chamar o modelo: {e}]"
            ph.markdown(silenciar_janio(answer))
        finally:
            # Render final sem o cursor e já filtrado
            _ans_clean = silenciar_janio(answer)
            ph.markdown(_ans_clean)

    # Salva sempre a versão filtrada
    st.session_state.chat.append({"role": "assistant", "content": _ans_clean})
    # Mantém apenas as últimas 30 interações na tela
    if len(st.session_state.chat) > 30:
        st.session_state.chat = st.session_state.chat[-30:]
    ts2 = datetime.now().isoformat(sep=" ", timespec="seconds")
    salvar_interacao(ts2, st.session_state.session_id, prov, model_id, "assistant", _ans_clean)
    st.rerun()














