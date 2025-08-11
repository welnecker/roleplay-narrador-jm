import streamlit as st
import requests
import gspread
import json
import re
import time
import numpy as np
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# ============================================================
# Configuração básica
# ============================================================
st.set_page_config(page_title="🎬 Narrador JM", page_icon="🎬")

# Chaves esperadas em st.secrets:
# - GOOGLE_CREDS_JSON
# - OPENROUTER_API_KEY
# - TOGETHER_API_KEY
# - OPENAI_API_KEY

# Cliente OpenAI para embeddings SEMPRE via OPENAI_API_KEY
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ============================================================
# Conectar à planilha
# ============================================================
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# ============================================================
# Utilidades de planilha (memórias curtas, interações, resumos)
# ============================================================
def carregar_memorias():
    """Lê a aba 'memorias_jm' e retorna (mem_mary, mem_janio, mem_all)."""
    try:
        aba = planilha.worksheet("memorias_jm")
        registros = aba.get_all_records()
        mem_mary = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[mary]"]
        mem_janio = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() in ("[jânio]", "[janio]")]
        mem_all = [r["conteudo"] for r in registros if r.get("tipo", "").strip().lower() == "[all]"]
        return mem_mary, mem_janio, mem_all
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return [], [], []

def carregar_resumo_salvo():
    """Busca o último resumo (coluna 7) da aba 'perfil_jm'."""
    try:
        aba = planilha.worksheet("perfil_jm")
        valores = aba.col_values(7)
        for val in reversed(valores[1:]):
            if val and val.strip():
                return val.strip()
        return ""
    except Exception as e:
        st.warning(f"Erro ao carregar resumo salvo: {e}")
        return ""

def salvar_resumo(resumo: str):
    """Salva um novo resumo na aba 'perfil_jm' (timestamp na coluna 6, resumo na 7)."""
    try:
        aba = planilha.worksheet("perfil_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        linha = ["", "", "", "", "", timestamp, resumo]
        aba.append_row(linha, value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def salvar_interacao(role: str, content: str):
    """Anexa uma interação na aba 'interacoes_jm'."""
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_jm")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_interacoes(n=20):
    """Carrega as últimas n interações (role, content)."""
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        return registros[-n:] if len(registros) > n else registros
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")
        return []

# ============================================================
# Validações (sintática + semântica com OpenAI Embeddings)
# ============================================================
def resposta_valida(texto: str) -> bool:
    # Heurística simples para detectar “resposta corrompida”
    padroes_invalidos = [
        r"check if.*string", r"#\s?1(\.\d+)+", r"\d{10,}", r"the cmd package",
        r"(111\s?)+", r"#+\s*\d+", r"\bimport\s", r"\bdef\s", r"```", r"class\s"
    ]
    for padrao in padroes_invalidos:
        if re.search(padrao, texto.lower()):
            return False
    return True

def gerar_embedding_openai(texto: str):
    try:
        resp = client_openai.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None:
        return 0.0
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def verificar_quebra_semantica_openai(texto1: str, texto2: str, limite=0.6) -> str:
    e1 = gerar_embedding_openai(texto1)
    e2 = gerar_embedding_openai(texto2)
    if e1 is None or e2 is None:
        return ""
    sim = cosine_similarity(e1, e2)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# ============================================================
# Memória vetorial de longo prazo (Sheets + OpenAI embedding)
# ============================================================
def _mem_longa_sheet():
    return planilha.worksheet("memoria_longa_jm")

def salvar_memoria_longa(texto: str, tags: str = ""):
    """Cria embedding do texto, inicia score=1.0 e salva na 'memoria_longa_jm'."""
    emb = gerar_embedding_openai(texto)
    if emb is None:
        return False
    try:
        aba = _mem_longa_sheet()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row(
            [timestamp, texto, tags or "", json.dumps(emb.tolist()), "1.0"],
            value_input_option="RAW"
        )
        return True
    except Exception as e:
        st.warning(f"Erro ao salvar memória longa: {e}")
        return False

def _decay_score(score: float, rounds: int = 1, fator: float = 0.97):
    return float(score * (fator ** rounds))

def _boost_score(score: float, inc: float = 0.2, max_score: float = 2.0):
    return float(min(max_score, score + inc))

def buscar_memorias_relevantes(query_text: str, top_k: int = 3, min_sim: float = 0.78):
    """
    Retorna até top_k memórias re-ranqueadas por: 0.7*similaridade + 0.3*score.
    """
    q_emb = gerar_embedding_openai(query_text)
    if q_emb is None:
        return []

    try:
        aba = _mem_longa_sheet()
        dados = aba.get_all_records()
    except Exception as e:
        st.warning(f"Erro ao ler memória longa: {e}")
        return []

    candidatos = []
    for r in dados:
        try:
            e = np.array(json.loads(r.get("embedding_json", "[]")), dtype=float)
            if e.size == 0:
                continue
            sim = cosine_similarity(q_emb, e)
            score = float(r.get("score", 1.0))
            rerank = 0.7 * sim + 0.3 * score
            if sim >= min_sim:
                candidatos.append({
                    "timestamp": r.get("timestamp", ""),
                    "texto": r.get("texto", ""),
                    "tags": r.get("tags", ""),
                    "score": score,
                    "sim": sim,
                    "rerank": rerank,
                })
        except Exception:
            continue

    candidatos.sort(key=lambda x: x["rerank"], reverse=True)
    return candidatos[:top_k]

def reforcar_memorias_utilizadas(textos_usados: list[str]):
    """Quando usar memórias no prompt, chama isso pra dar boost nelas e salvar de volta."""
    if not textos_usados:
        return
    try:
        aba = _mem_longa_sheet()
        valores = aba.get_all_values()
        if not valores:
            return
        header = valores[0]
        linhas = valores[1:]
        idx_texto = header.index("texto")
        idx_score = header.index("score")
        for i, linha in enumerate(linhas, start=2):  # cabeçalho na linha 1
            if len(linha) <= max(idx_texto, idx_score):
                continue
            if linha[idx_texto] in textos_usados:
                try:
                    sc = float(linha[idx_score] or "1.0")
                    sc = _boost_score(sc)
                    aba.update_cell(i, idx_score + 1, str(sc))
                except Exception:
                    continue
    except Exception as e:
        st.warning(f"Erro ao reforçar memórias: {e}")

# ============================================================
# Estado estruturado de cena
# ============================================================
if "scene" not in st.session_state:
    st.session_state.scene = {
        "local": "",
        "tempo": "",
        "objetivo_mary": "",
        "objetivo_janio": "",
        "tensao": 0.0,
        "ganchos": [],
        "itens_relevantes": [],
        "intent": {},
    }

def atualizar_scene_state_via_llm(texto_resposta: str):
    """
    Chamada rápida (sem stream) ao provedor atual pedindo um JSON compacto com o estado da cena.
    Usa o provedor/modelo já selecionado.
    """
    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets["TOGETHER_API_KEY"]
        model_to_call = None
        try:
            model_to_call = st.session_state.modelo_escolhido_id
            model_to_call = model_id_for_together(model_to_call)
        except Exception:
            model_to_call = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets["OPENROUTER_API_KEY"]
        model_to_call = st.session_state.get("modelo_escolhido_id", "deepseek/deepseek-chat-v3-0324")

    sys = (
        "Extraia um estado de cena em JSON com as chaves: "
        "local, tempo, objetivo_mary, objetivo_janio, tensao (0..1), ganchos (lista), itens_relevantes (lista). "
        "Se não souber, deixe vazio. Responda APENAS o JSON válido."
    )
    user = f"Texto da cena:\n{texto_resposta}\n\nJSON:"
    try:
        r = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"},
            json={
                "model": model_to_call,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 220,
                "temperature": 0.2,
                "stream": False,
            },
            timeout=60,
        )
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"].strip()
            try:
                parsed = json.loads(txt)
                st.session_state.scene.update({
                    "local": parsed.get("local", st.session_state.scene["local"]),
                    "tempo": parsed.get("tempo", st.session_state.scene["tempo"]),
                    "objetivo_mary": parsed.get("objetivo_mary", st.session_state.scene["objetivo_mary"]),
                    "objetivo_janio": parsed.get("objetivo_janio", st.session_state.scene["objetivo_janio"]),
                    "tensao": parsed.get("tensao", st.session_state.scene["tensao"]),
                    "ganchos": parsed.get("ganchos", st.session_state.scene["ganchos"]),
                    "itens_relevantes": parsed.get("itens_relevantes", st.session_state.scene["itens_relevantes"]),
                })
            except Exception:
                pass
    except Exception:
        pass

# ============================================================
# Provedores e modelos
# ============================================================
MODELOS_OPENROUTER = {
    # === OPENROUTER ===
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

# Together (UI -> mapeamento p/ ID aceito pela API)
MODELOS_TOGETHER_UI = {
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 perplexity-ai (Together)": "perplexity-ai/r1-1776",
}
def model_id_for_together(api_ui_model_id: str) -> str:
    # Corrige para os IDs aceitos pela Together API
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in api_ui_model_id:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if api_ui_model_id.lower().startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return api_ui_model_id

def api_config_for_provider(provider: str):
    if provider == "OpenRouter":
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            st.secrets["OPENROUTER_API_KEY"],
            MODELOS_OPENROUTER,
        )
    else:
        return (
            "https://api.together.xyz/v1/chat/completions",
            st.secrets["TOGETHER_API_KEY"],
            MODELOS_TOGETHER_UI,
        )

# ============================================================
# Construção do prompt (com memórias vetoriais + scene state)
# ============================================================
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    # últimas interações (compacto)
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-10:] if len(registros) > 10 else registros
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
    except Exception:
        registros = []
        texto_ultimas = ""

    # consulta p/ memória vetorial = última fala do usuário
    consulta_mem = ""
    try:
        for r in reversed(registros):
            if r.get("role") == "user":
                consulta_mem = r.get("content", "")
                break
    except Exception:
        pass

    mem_relevantes = buscar_memorias_relevantes(consulta_mem) if consulta_mem else []
    st.session_state._mem_relevantes = mem_relevantes  # para reforço após resposta
    bloco_mem_relev = "\n".join([f"- {m['texto']}" for m in mem_relevantes]) if mem_relevantes else "Nenhuma encontrada para o momento."

    # scene state atual
    scene = st.session_state.get("scene", {})
    bloco_scene = (
        f"Local: {scene.get('local','')}\n"
        f"Tempo: {scene.get('tempo','')}\n"
        f"Objetivo de Mary: {scene.get('objetivo_mary','')}\n"
        f"Objetivo de Jânio: {scene.get('objetivo_janio','')}\n"
        f"Tensão: {scene.get('tensao',0.0)}\n"
        f"Ganchos: {', '.join(scene.get('ganchos',[]))}\n"
        f"Itens relevantes: {', '.join(scene.get('itens_relevantes',[]))}\n"
    )

    regra_intimo = (
        "\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista."
        if st.session_state.get("bloqueio_intimo", False) else ""
    )

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.{regra_intimo}

🎭 Emoção oculta: {emocao}

📖 Capítulo anterior:
{(resumo or 'Nenhum resumo salvo.')}

### 🧠 Memórias (fixas):
Mary:
- {("\n- ".join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {("\n- ".join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {("\n- ".join(mem_all)) if mem_all else 'Nenhuma.'}

### 🧠 Memórias relevantes (recuperação vetorial):
{bloco_mem_relev}

### 🎛️ Estado Atual da Cena
{bloco_scene}

### 📖 Últimas interações (recorte):
{texto_ultimas}

### 🎯 Estilo de prosa
- “Mostre, não conte”: prefira ações e detalhes sensoriais.
- Em cada frase, foque **um** canal sensorial (som/cheiro/tato/visão).
- Falas curtas, com subtexto (≤ 18 palavras).
- Avance o enredo suavemente, sem cortes bruscos.
"""
    return prompt.strip()

# ============================================================
# UI – Cabeçalho e controles
# ============================================================
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Estado inicial
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []
if "provedor_ia" not in st.session_state:
    st.session_state.provedor_ia = "OpenRouter"

# Linha de opções rápidas
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### 📖 Último resumo salvo:")
    st.info(st.session_state.resumo_capitulo or "Nenhum resumo disponível.")
with col2:
    st.markdown("#### ⚙️ Opções")
    st.session_state.bloqueio_intimo = st.checkbox("Bloquear avanços íntimos sem ordem", value=False)
    st.session_state.emocao_oculta = st.selectbox(
        "🎭 Emoção oculta", ["nenhuma", "tristeza", "felicidade", "tensão", "raiva"], index=0
    )

# ============================================================
# Sidebar – Provedor, modelos e resumo
# ============================================================
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    st.session_state.modelo_escolhido_id = modelo_escolhido_id_ui  # armazenar para uso no envio

    st.markdown("---")
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in inter) if inter else ""
            prompt_resumo = (
                "Resuma o seguinte trecho como um capítulo de novela brasileiro, mantendo tom e emoções.\n\n"
                + texto + "\n\nResumo:"
            )

            # escolher o modelo atual também para o resumo
            if provedor == "Together":
                model_id_call = model_id_for_together(modelo_escolhido_id_ui)
            else:
                model_id_call = modelo_escolhido_id_ui

            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_id_call,
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": 0.85,
                },
                timeout=120,
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

    st.caption("Role a tela principal para ver interações anteriores.")

# ============================================================
# Exibir histórico recente (role para cima para ver mais)
# ============================================================
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

# ============================================================
# Envio do usuário + Streaming com “digitação”
# ============================================================
entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    # Salva e mostra a fala do usuário
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    # Construir prompt e histórico
    prompt = construir_prompt_com_narrador()
    historico = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in st.session_state.session_msgs]

    # Penalidades dinâmicas se repetindo demais
    freq_penalty = 0.0
    presence_penalty = 0.0
    try:
        ult_assist = ""
        for m in reversed(st.session_state.session_msgs):
            if m.get("role") == "assistant":
                ult_assist = m.get("content", "")
                break
        if ult_assist:
            sim_rep = cosine_similarity(
                gerar_embedding_openai(ult_assist), gerar_embedding_openai(entrada)
            )
            if sim_rep and sim_rep > 0.92:
                freq_penalty = 0.2
                presence_penalty = 0.4
    except Exception:
        pass

    # Roteia por provedor e ajusta ID do Together quando necessário
    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets["TOGETHER_API_KEY"]
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets["OPENROUTER_API_KEY"]
        model_to_call = st.session_state.modelo_escolhido_id

    payload = {
        "model": model_to_call,
        "messages": [{"role": "system", "content": prompt}] + historico,
        "max_tokens": 900,
        "temperature": 0.85,
        "frequency_penalty": freq_penalty,
        "presence_penalty": presence_penalty,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    # Streaming com “marcapasso” para digitação fluida
    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        buffer_txt = ""
        last_flush = time.time()

        try:
            with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
                    resposta_txt = "[ERRO STREAM]"
                else:
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
                            if delta:
                                buffer_txt += delta
                                now = time.time()
                                if (now - last_flush) > 0.10 or buffer_txt.count(" ") > 20:
                                    resposta_txt += buffer_txt
                                    buffer_txt = ""
                                    last_flush = now
                                    placeholder.markdown(resposta_txt + "▌")
                        except Exception:
                            continue
        except Exception as e:
            st.error(f"Erro no streaming: {e}")
            resposta_txt = "[Erro ao gerar resposta]"

        # Flush final do buffer
        if buffer_txt:
            resposta_txt += buffer_txt
            buffer_txt = ""

        # Validação sintática
        if not resposta_valida(resposta_txt):
            st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar...")
            try:
                regen = requests.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model_to_call,
                        "messages": [{"role": "system", "content": prompt}] + historico,
                        "max_tokens": 900,
                        "temperature": 0.85,
                        "stream": False,
                    },
                    timeout=180,
                )
                if regen.status_code == 200:
                    resposta_txt = regen.json()["choices"][0]["message"]["content"].strip()
                else:
                    st.error(f"Erro ao regenerar: {regen.status_code} - {regen.text}")
            except Exception as e:
                st.error(f"Erro ao regenerar: {e}")

        # Validação semântica (com OpenAI embeddings) entre última entrada do user e resposta
        if len(st.session_state.session_msgs) >= 1 and resposta_txt and resposta_txt != "[ERRO STREAM]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]  # última entrada do user
            alerta = verificar_quebra_semantica_openai(texto_anterior, resposta_txt)
            if alerta:
                st.info(alerta)

        # Finaliza na tela
        placeholder.markdown(resposta_txt or "[Sem conteúdo]")
        salvar_interacao("assistant", resposta_txt)
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt})

        # Reforça memórias usadas nesta rodada (se houver)
        if st.session_state.get("_mem_relevantes"):
            try:
                reforcar_memorias_utilizadas([m['texto'] for m in st.session_state._mem_relevantes])
            except Exception:
                pass

        # Atualiza estado da cena
        atualizar_scene_state_via_llm(resposta_txt)

