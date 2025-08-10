import streamlit as st
import requests
import gspread
import json
import re
import numpy as np
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# --------------------------------------------------------------------
# Configuração inicial
# --------------------------------------------------------------------
st.set_page_config(page_title="Narrador JM", page_icon="🎬")

# --------------------------------------------------------------------
# Conexão com Google Sheets
# --------------------------------------------------------------------
def conectar_planilha():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        # MUITO IMPORTANTE: manter "\\n" -> "\n" exatamente assim
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# --------------------------------------------------------------------
# Utilidades de planilha (abas básicas que você já usa)
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# BLOCO NOVO: Memória longa + Resumos duplos + Estado de cena (plugável)
# --------------------------------------------------------------------
import json as _json_jm
import numpy as _np_jm
from datetime import datetime as _dt_jm

def _get_worksheet_safe(spreadsheet, title, headers=None):
    try:
        return spreadsheet.worksheet(title)
    except Exception:
        try:
            ws = spreadsheet.add_worksheet(title=title, rows=2000, cols=20)
            if headers:
                ws.append_row(headers, value_input_option="RAW")
            return ws
        except Exception:
            return None

def jm_save_memoria_longa(texto: str, tags=None, score: float = 1.0):
    if not texto or not planilha:
        return False
    ws = _get_worksheet_safe(
        planilha, "memoria_longa_jm",
        headers=["id", "texto", "embedding", "tags", "score", "timestamp"]
    )
    if ws is None:
        return False
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        emb = client.embeddings.create(input=texto, model="text-embedding-3-small").data[0].embedding
    except Exception as e:
        st.warning(f"[memoria_longa] Falha ao gerar embedding: {e}")
        return False
    try:
        valores = ws.get_all_values()
        prox_id = len(valores)
        ts = _dt_jm.now().strftime("%Y-%m-%d %H:%M:%S")
        tags_str = ",".join(tags) if isinstance(tags, (list, tuple, set)) else (tags or "[all]")
        ws.append_row([prox_id, texto, _json_jm.dumps(emb), tags_str, score, ts], value_input_option="RAW")
        return True
    except Exception as e:
        st.warning(f"[memoria_longa] Erro ao salvar: {e}")
        return False

def _cos_sim_jm(a: _np_jm.ndarray, b: _np_jm.ndarray) -> float:
    denom = (float(_np_jm.linalg.norm(a)) * float(_np_jm.linalg.norm(b)))
    if denom == 0:
        return 0.0
    return float(a.dot(b) / denom)

def jm_buscar_memoria_longa(query: str, top_k: int = 4, tags_filter=None):
    if not query or not planilha:
        return []
    ws = _get_worksheet_safe(planilha, "memoria_longa_jm")
    if ws is None:
        return []
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        q_emb = _np_jm.array(client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding)
    except Exception as e:
        st.warning(f"[memoria_longa] Falha ao gerar embedding da query: {e}")
        return []
    try:
        registros = ws.get_all_records()
    except Exception as e:
        st.warning(f"[memoria_longa] Falha ao ler planilha: {e}")
        return []
    if tags_filter is not None and not isinstance(tags_filter, (list, tuple, set)):
        tags_filter = [str(tags_filter).strip().lower()]
    candidatos = []
    for r in registros:
        texto = r.get("texto", "").strip()
        emb_str = r.get("embedding", "").strip()
        score = float(r.get("score", 1.0) or 1.0)
        tags_str = r.get("tags", "")
        if tags_filter:
            tags_reg = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
            if not any(t in tags_reg for t in tags_filter):
                continue
        try:
            v = _np_jm.array(_json_jm.loads(emb_str), dtype=float)
        except Exception:
            continue
        sim = _cos_sim_jm(q_emb, v)
        rank = sim * score
        candidatos.append({"texto": texto, "tags": tags_str, "score": score, "sim": sim, "rank": rank})
    candidatos.sort(key=lambda x: x["rank"], reverse=True)
    return candidatos[:top_k]

def jm_salvar_resumo_duplo(episodio: str, tematico: str):
    if not planilha or (not episodio and not tematico):
        return False
    ws = _get_worksheet_safe(planilha, "resumos_jm", headers=["id", "episodio", "tematico", "timestamp"])
    if ws is None:
        return False
    try:
        valores = ws.get_all_values()
        prox_id = len(valores)
        ts = _dt_jm.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([prox_id, episodio or "", tematico or "", ts], value_input_option="RAW")
        return True
    except Exception as e:
        st.warning(f"[resumos_jm] Erro ao salvar: {e}")
        return False

def jm_carregar_ultimo_resumo_duplo():
    if not planilha:
        return {}
    ws = _get_worksheet_safe(planilha, "resumos_jm")
    if ws is None:
        return {}
    try:
        registros = ws.get_all_records()
        if not registros:
            return {}
        ultimo = registros[-1]
        return {
            "episodio": ultimo.get("episodio", "").strip(),
            "tematico": ultimo.get("tematico", "").strip(),
            "timestamp": ultimo.get("timestamp", ""),
        }
    except Exception:
        return {}

def jm_atualizar_scene_state(local="", tempo="", objetivo_mary="", objetivo_janio="", tensao: float = 0.0, ganchos=None, itens_relevantes=None):
    if not planilha:
        return False
    ws = _get_worksheet_safe(
        planilha, "scene_state_jm",
        headers=["id", "local", "tempo", "objetivo_mary", "objetivo_janio", "tensao", "ganchos", "itens_relevantes", "timestamp"]
    )
    if ws is None:
        return False
    try:
        valores = ws.get_all_values()
        prox_id = len(valores)
        ts = _dt_jm.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row(
            [prox_id, local or "", tempo or "", objetivo_mary or "", objetivo_janio or "", float(tensao or 0.0),
             _json_jm.dumps(ganchos or [], ensure_ascii=False), _json_jm.dumps(itens_relevantes or [], ensure_ascii=False), ts],
            value_input_option="RAW"
        )
        return True
    except Exception as e:
        st.warning(f"[scene_state_jm] Erro ao salvar: {e}")
        return False

def jm_carregar_scene_state_ultimo():
    if not planilha:
        return {}
    ws = _get_worksheet_safe(planilha, "scene_state_jm")
    if ws is None:
        return {}
    try:
        registros = ws.get_all_records()
        if not registros:
            return {}
        ultimo = registros[-1]
        try:
            ultimo["ganchos"] = _json_jm.loads(ultimo.get("ganchos") or "[]")
        except Exception:
            ultimo["ganchos"] = []
        try:
            ultimo["itens_relevantes"] = _json_jm.loads(ultimo.get("itens_relevantes") or "[]")
        except Exception:
            ultimo["itens_relevantes"] = []
        return ultimo
    except Exception:
        return {}

# --------------------------------------------------------------------
# Validações (sintática + semântica)
# --------------------------------------------------------------------
def resposta_valida(texto: str) -> bool:
    padroes_invalidos = [
        r"check if.*string", r"#\s?1(\.\d+)+", r"\d{10,}", r"the cmd package",
        r"(111\s?)+", r"#+\s*\d+", r"\bimport\s", r"\bdef\s", r"```", r"class\s"
    ]
    for padrao in padroes_invalidos:
        if re.search(padrao, texto.lower()):
            return False
    return True

client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def gerar_embedding_openai(texto: str):
    try:
        resp = client_openai.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return np.array(resp.data[0].embedding)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def verificar_quebra_semantica_openai(texto1: str, texto2: str, limite=0.6) -> str:
    e1 = gerar_embedding_openai(texto1)
    e2 = gerar_embedding_openai(texto2)
    if e1 is None or e2 is None:
        return ""
    sim = cosine_similarity(e1, e2)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f})."
    return ""

# --------------------------------------------------------------------
# Construção do prompt (com patches plugados)
# --------------------------------------------------------------------
def construir_prompt_com_narrador():
    mem_mary, mem_janio, mem_all = carregar_memorias()
    emocao = st.session_state.get("emocao_oculta", "nenhuma")
    resumo = st.session_state.get("resumo_capitulo", "")

    # últimas 15 interações
    try:
        aba = planilha.worksheet("interacoes_jm")
        registros = aba.get_all_records()
        ultimas = registros[-15:] if len(registros) > 15 else registros
        texto_ultimas = "\n".join(f"{r['role']}: {r['content']}" for r in ultimas)
    except Exception:
        texto_ultimas = ""

    regra_intimo = (
        "\n⛔ Jamais antecipe encontros, conexões emocionais ou cenas íntimas sem ordem explícita do roteirista."
        if st.session_state.get("bloqueio_intimo", False) else ""
    )

    prompt = f"""Você é o narrador de uma história em construção. Os protagonistas são Mary e Jânio.

Sua função é narrar cenas com naturalidade e profundidade. Use narração em 3ª pessoa e falas/pensamentos dos personagens em 1ª pessoa.{regra_intimo}

🎭 Emoção oculta da cena: {emocao}

📖 Capítulo anterior:
{(resumo or 'Nenhum resumo salvo.')}

### 🧠 Memórias:
Mary:
- {("\n- ".join(mem_mary)) if mem_mary else 'Nenhuma.'}

Jânio:
- {("\n- ".join(mem_janio)) if mem_janio else 'Nenhuma.'}

Compartilhadas:
- {("\n- ".join(mem_all)) if mem_all else 'Nenhuma.'}

### 📖 Últimas interações:
{texto_ultimas}"""

    # >>> Patch: enriquecer prompt com memória longa + estado de cena
    try:
        relevantes = jm_buscar_memoria_longa(
            query=(texto_ultimas or st.session_state.get("entrada_atual") or ""),
            top_k=4,
            tags_filter=["[all]"]
        )
        if relevantes:
            bloco_memoria_longa = "\n".join(f"- {r['texto']}" for r in relevantes)
            prompt += f"\n\n### 🧠 Memória longa relevante\n{bloco_memoria_longa}"
    except Exception:
        pass

    try:
        estado = jm_carregar_scene_state_ultimo()
        if estado:
            prompt += (
                f"\n\n### 🎬 Estado de Cena\n"
                f"- Local: {estado.get('local','')}\n"
                f"- Tempo: {estado.get('tempo','')}\n"
                f"- Objetivo Mary: {estado.get('objetivo_mary','')}\n"
                f"- Objetivo Jânio: {estado.get('objetivo_janio','')}\n"
                f"- Tensão: {estado.get('tensao',0)}\n"
                f"- Ganchos: {', '.join(estado.get('ganchos', []))}\n"
                f"- Itens: {', '.join(estado.get('itens_relevantes', []))}\n"
            )
    except Exception:
        pass

    return prompt.strip()

# --------------------------------------------------------------------
# Provedores e modelos (mantendo seus IDs)
# --------------------------------------------------------------------
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

# UI do Together (como você quer ver) -> mapeado para o ID aceito pela API
MODELOS_TOGETHER_UI = {
    "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

def model_id_for_together(api_ui_model_id: str) -> str:
    """Conserta IDs para o endpoint da Together."""
    if "Qwen3-Coder-480B-A35B-Instruct-FP8" in api_ui_model_id:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if api_ui_model_id.lower().startswith("mistralai/mixtral-8x7b-instruct-v0.1"):
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # caso você adicione novos modelos, eles passam direto
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

# --------------------------------------------------------------------
# UI – Cabeçalho e estado inicial
# --------------------------------------------------------------------
st.title("🎬 Narrador JM")
st.subheader("Você é o roteirista. Digite uma direção de cena. A IA narrará Mary e Jânio.")
st.markdown("---")

# Estado inicial
if "resumo_capitulo" not in st.session_state:
    st.session_state.resumo_capitulo = carregar_resumo_salvo()

# Hidratar também a última versão de resumo_duplo (se existir)
try:
    _ultimo_duplo = jm_carregar_ultimo_resumo_duplo()
    if _ultimo_duplo and _ultimo_duplo.get("episodio") and not st.session_state.get("resumo_capitulo"):
        st.session_state.resumo_capitulo = _ultimo_duplo["episodio"]
except Exception:
    pass

if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []

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

# --------------------------------------------------------------------
# Sidebar – Provedor, Modelos, Resumo
# --------------------------------------------------------------------
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
                # Também salva no resumos_jm como "episódio" + rascunho temático
                try:
                    jm_salvar_resumo_duplo(
                        episodio=resumo,
                        tematico="(marque aqui um tema, ex.: tensão crescente entre Mary e Jânio)"
                    )
                except Exception:
                    pass
                st.success("Resumo gerado e salvo com sucesso!")
            else:
                st.error(f"Erro ao resumir: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")

    st.caption("Role a tela principal para ver interações anteriores.")

# --------------------------------------------------------------------
# Mostrar histórico recente (role para cima = mais antigo)
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Envio do usuário + Streaming (OpenRouter/Together)
# --------------------------------------------------------------------
entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    # Salva e mostra a fala do usuário
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})
    st.session_state.entrada_atual = entrada

    # Construir prompt e histórico
    prompt = construir_prompt_com_narrador()
    historico = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in st.session_state.session_msgs]

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
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    # Streaming com digitação
    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
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
                                resposta_txt += delta
                                placeholder.markdown(resposta_txt + "▌")
                        except Exception:
                            continue
        except Exception as e:
            st.error(f"Erro no streaming: {e}")
            resposta_txt = "[Erro ao gerar resposta]"

        # Validação sintática
        if not resposta_valida(resposta_txt):
            st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar...")
            # Tenta regenerar 1x sem stream
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

        # Validação semântica (comparando a última entrada do user com a resposta)
        if len(st.session_state.session_msgs) >= 1 and resposta_txt and resposta_txt != "[ERRO STREAM]":
            texto_anterior = st.session_state.session_msgs[-1]["content"]
            alerta = verificar_quebra_semantica_openai(texto_anterior, resposta_txt)
            if alerta:
                st.info(alerta)

        # Finaliza streaming na tela
        placeholder.markdown(resposta_txt or "[Sem conteúdo]")
        salvar_interacao("assistant", resposta_txt)
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta_txt})

        # >>> Patch: gravar memória longa e snapshot de estado
        try:
            if resposta_txt and len(resposta_txt) > 120:
                jm_save_memoria_longa(
                    texto=resposta_txt,
                    tags=["[all]", "[capitulo]"],
                    score=1.0
                )
        except Exception:
            pass
        try:
            # Preencha com valores reais se quiser snapshot útil a cada resposta
            jm_atualizar_scene_state(
                local="",
                tempo="",
                objetivo_mary="",
                objetivo_janio="",
                tensao=0.5,
                ganchos=[],
                itens_relevantes=[]
            )
        except Exception:
            pass
