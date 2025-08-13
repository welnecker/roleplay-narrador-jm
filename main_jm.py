# main.py
# ============================================================
# Narrador JM — Roleplay adulto com controle de ritmo e momento
# ============================================================

import os, time, json, re, math, random
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import requests

# -------------------------
# Google Sheets (gspread)
# -------------------------
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_OK = True
except Exception:
    GSPREAD_OK = False

# =========================
# CONFIG BÁSICA DO APP
# =========================
st.set_page_config(page_title="Narrador JM", page_icon="🎬", layout="wide")

# ---------- Age gate ----------
if "age_ok" not in st.session_state:
    st.session_state.age_ok = False

# Removido o age gate - acesso direto para usuário maior de idade
st.session_state.age_ok = True

# =========================
# CONEXÃO COM GOOGLE SHEETS
# =========================
PLANILHA_NOME = st.secrets.get("SHEET_NAME", "NarradorJM")

def _gc_connect():
    if not GSPREAD_OK:
        st.warning("gspread não instalado — modo leitura local.")
        return None
    try:
        # Espera secrets no formato padrão do Streamlit
        info = st.secrets.get("gcp_service_account") or st.secrets.get("GCP_SERVICE_ACCOUNT")
        if not info:
            st.warning("Credenciais do Google ausentes em st.secrets[\'gcp_service_account\'].")
            return None
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.warning(f"Falha ao autenticar no Google Sheets: {e}")
        return None

def _sheet():
    gc = _gc_connect()
    if not gc:
        return None
    try:
        return gc.open(PLANILHA_NOME)
    except Exception as e:
        st.warning(f"Não foi possível abrir a planilha \'{PLANILHA_NOME}\': {e}")
        return None

def _ws(name: str):
    sh = _sheet()
    if not sh:
        return None
    try:
        return sh.worksheet(name)
    except:
        try:
            return sh.add_worksheet(title=name, rows=5000, cols=10)
        except Exception as e:
            st.warning(f"Não foi possível acessar/criar aba \'{name}\': {e}")
            return None

# Nomes das abas esperadas
TAB_INTERACOES = "interacoes_jm"     # timestamp | role | content
TAB_PERFIL     = "perfil_jm"         # timestamp | resumo
TAB_MEMORIAS   = "memorias_jm"       # tipo | conteudo
TAB_ML         = "memoria_longa_jm"  # texto | embedding | tags | timestamp | score

# =========================
# HELPERS DE SHEETS
# =========================
def salvar_interacao(role: str, content: str):
    try:
        ws = _ws(TAB_INTERACOES)
        if not ws: return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, role, content], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_interacoes(n: int = 20) -> List[Dict[str, str]]:
    try:
        ws = _ws(TAB_INTERACOES)
        if not ws:
            return []
        recs = ws.get_all_records()
        recs = recs[-n:] if n > 0 else recs
        return [{"timestamp": r.get("timestamp",""), "role": r.get("role",""), "content": r.get("content","")} for r in recs]
    except Exception as e:
        st.warning(f"Erro ao carregar interações: {e}")
        return []

def salvar_resumo(resumo: str):
    try:
        ws = _ws(TAB_PERFIL)
        if not ws: return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, resumo], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

def carregar_resumo_salvo() -> str:
    try:
        ws = _ws(TAB_PERFIL)
        if not ws:
            return ""
        recs = ws.get_all_records()
        for r in reversed(recs):
            txt = (r.get("resumo") or "").strip()
            if txt:
                return txt
        return ""
    except Exception as e:
        st.warning(f"Erro ao carregar resumo salvo: {e}")
        return ""

def memorias_listar() -> List[Tuple[str,str]]:
    try:
        ws = _ws(TAB_MEMORIAS)
        if not ws: return []
        recs = ws.get_all_records()
        out = []
        for r in recs:
            t = (r.get("tipo") or "").strip()
            c = (r.get("conteudo") or "").strip()
            if t and c:
                out.append((t, c))
        return out
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return []

def memoria_longa_salvar(texto: str, tags: str="auto", score: float=1.0) -> bool:
    try:
        ws = _ws(TAB_ML)
        if not ws: return False
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # embedding fica vazio (calculado quando salvar via app, se houver)
        ws.append_row([texto, "", tags, ts, score], value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar memória longa: {e}")
        return False

def memoria_longa_listar() -> List[Dict[str, Any]]:
    try:
        ws = _ws(TAB_ML)
        if not ws: return []
        recs = ws.get_all_records()
        return recs
    except Exception as e:
        st.warning(f"Erro ao carregar memória longa: {e}")
        return []

def memoria_longa_reforcar(textos: List[str]):
    """Aumenta score dos textos usados em +0.1 (se existir coluna score)."""
    try:
        ws = _ws(TAB_ML)
        if not ws or not textos:
            return
        recs = ws.get_all_records()
        idx_base = 2  # 1-based (pula cabeçalho)
        updates = []
        for i, r in enumerate(recs):
            t = (r.get("texto") or "").strip()
            if t in textos:
                try:
                    sc = float(r.get("score") or 1.0) + 0.1
                except:
                    sc = 1.1
                updates.append((idx_base + i, sc))
        for rowidx, sc in updates:
            ws.update_cell(rowidx, 5, sc)  # score é a 5ª coluna
    except Exception:
        pass

# Similaridade bem simples (palavras)
def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-zà-ú0-9]+", (s or "").lower()))

def memoria_longa_buscar_topk(query_text: str, k: int=3, limiar: float=0.78):
    """Busca Top-K por similaridade de Jaccard (fallback simples)."""
    q = _tokenize(query_text)
    out = []
    for r in memoria_longa_listar():
        t = (r.get("texto") or "").strip()
        if not t or t.startswith("FLAG:"):
            continue
        s = _tokenize(t)
        if not q or not s:
            sim = 0.0
        else:
            sim = len(q & s) / max(1, len(q | s))
        score = float(r.get("score") or 1.0)
        rank = sim * math.log1p(score)
        if sim >= limiar:
            out.append((t, score, sim, rank))
    out.sort(key=lambda x: x[3], reverse=True)
    return out[:k]

# =========================
# ROMANCE (FASES) + MOMENTO
# =========================

FASES_ROMANCE: Dict[int, Dict[str, str]] = {
    0: {"nome": "Estranhos",
        "permitidos": "olhares; near-miss (mesmo café/rua/ônibus); detalhe do ambiente",
        "proibidos":  "troca de nomes; toques; conversa pessoal"},
    1: {"nome": "Percepção",
        "permitidos": "cumprimento neutro; pergunta impessoal curta",
        "proibidos":  "contato físico; confidências"},
    2: {"nome": "Conhecidos",
        "permitidos": "troca de nomes; pequena ajuda; 1 pergunta pessoal leve",
        "proibidos":  "toque prolongado; encontro a sós planejado"},
    3: {"nome": "Amizade",
        "permitidos": "conversa 10–20 min; caminhar juntos; troca de contatos; 1 gesto de afeto leve (com consentimento)",
        "proibidos":  "beijos; carícias intimistas"},
    4: {"nome": "Confiança / Quase",
        "permitidos": "confidências; abraço com consentimento expresso; marcar encontro futuro claro",
        "proibidos":  "sexo; sexo oral/manual; pressa ou “provas de amor” físicas"},
    5: {"nome": "Compromisso / Encontro definitivo",
        "permitidos": "beijo prolongado; dormir juntos; consumação **explícita**; descrição detalhada de atos sexuais; linguagem erótica; manhã seguinte sugerida",
        "proibidos":  ""},
}

FLAG_FASE_TXT_PREFIX = "FLAG: mj_fase="

def _fase_label(n: int) -> str:
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE[0])
    return f"{int(n)} — {d[\'nome\']}"

def mj_set_fase(n: int, persist: bool=True):
    n = max(0, min(5, int(n)))
    st.session_state.mj_fase = n
    if persist:
        try:
            memoria_longa_salvar(f"{FLAG_FASE_TXT_PREFIX}{n}", tags="[flag]")
        except Exception:
            pass

def mj_carregar_fase_inicial() -> int:
    if "mj_fase" in st.session_state:
        return int(st.session_state.mj_fase)
    # tenta última FLAG da memória longa
    try:
        recs = memoria_longa_listar()
        for r in reversed(recs):
            t = (r.get("texto") or "").strip()
            if t.startswith(FLAG_FASE_TXT_PREFIX):
                n = int(t.split("=")[1])
                st.session_state.mj_fase = n
                return n
    except:
        pass
    st.session_state.mj_fase = 0
    return 0

# --------- Motor de Momento ----------
MOMENTOS = {
    0: {"nome": "Aproximação logística",
        "objetivo": "um acompanha o outro (ex.: até o píer), clima cordial",
        "permitidos": "gentilezas; proximidade leve; diálogo casual; toques casuais",
        "proibidos": "",
        "gatilhos": [r"\b(p[ií]er|acompanhar|vamos embora|te levo)\b"],
        "proximo": 1},
    1: {"nome": "Declaração",
        "objetivo": "um deles declara amor/ importância",
        "permitidos": "confissão afetiva; silêncio tenso; abraço curto; beijos leves",
        "proibidos": "",
        "gatilhos": [r"\b(amo voc[eê]|te amo|n[aã]o paro de pensar)\b"],
        "proximo": 2},
    2: {"nome": "Revelação sensível",
        "objetivo": "Mary revela que é virgem / vulnerabilidade equivalente",
        "permitidos": "dizer \'sou virgem\'; estipular limites; conforto mútuo; carícias íntimas",
        "proibidos": "",
        "gatilhos": [r"\b(sou virgem|nunca fiz|meu limite)\b"],
        "proximo": 3},
    3: {"nome": "Consentimento explícito",
        "objetivo": "alinhamento de limites e um \'sim\' claro",
        "permitidos": "nomear fronteiras; pedir/receber consentimento; decidir \'agora sim\'; tirar roupas; exploração do corpo",
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eê])\b", r"\b(at[eé] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade explícita",
        "objetivo": "intimidade sexual detalhada e explícita",
        "permitidos": "beijos longos; proximidade forte; descrição explícita de atos sexuais; detalhes anatômicos; linguagem erótica; manhã seguinte implícita",
        "proibidos": "",
        "gatilhos": [r"\b(quarto|cama|luz baixa|porta fechada|manh[aã] seguinte)\b"],
        "proximo": 4},
}

def _momento_label(n: int) -> str:
    m = MOMENTOS.get(int(n), MOMENTOS[0])
    return f"{int(n)} — {m[\'nome\']}"

def detectar_momento_sugerido(texto: str, fallback: int = 0) -> int:
    t = (texto or "").lower()
    for i in range(4, -1, -1):
        for gx in MOMENTOS[i]["gatilhos"]:
            if re.search(gx, t, flags=re.IGNORECASE):
                return i
    # Removido o bloqueio - permite progressão natural
    return fallback

def clamp_momento(atual: int, proposto: int, max_steps: int) -> int:
    if proposto > atual + max_steps:
        return atual + max_steps
    if proposto < atual:
        return max(proposto, atual - 1)
    return proposto

def momento_set(n: int, persist: bool = True):
    n = max(0, min(4, int(n)))
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
        recs = memoria_longa_listar()
        for r in reversed(recs):
            t = (r.get("texto") or "").strip()
            if t.startswith("FLAG: mj_momento="):
                n = int(t.split("=")[1])
                st.session_state.momento = n
                return n
    except:
        pass
    st.session_state.momento = 0
    return 0

def viola_momento(texto: str, momento: int) -> str:
    # Removido todas as restrições - permite conteúdo explícito
    return ""

# =========================
# PROVEDOR DE IA
# =========================
def api_config_for_provider(prov: str):
    if prov == "Together":
        url = "https://api.together.xyz/v1/chat/completions"
        key = st.secrets.get("TOGETHER_API_KEY", "")
        modelos = {
            "Llama-3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "Qwen2.5 72B Instruct": "Qwen/Qwen2.5-72B-Instruct",
        }
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = st.secrets.get("OPENROUTER_API_KEY", "")
        modelos = {
            "GPT-4.1 (via OpenRouter)": "openai/gpt-4.1-mini",
            "Llama-3.1 70B (via OpenRouter)": "meta-llama/llama-3.1-70b-instruct",
        }
    return url, key, modelos

def model_id_for_together(modelo_escolhido_id: str) -> str:
    return modelo_escolhido_id  # já no formato correto

# =========================
# PROMPT BUILDER
# =========================
def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    # Regras adaptadas para conteúdo NSFW explícito
    calor = int(st.session_state.get("steam_level", st.session_state.get("nsfw_max_level", 3)))
    regras = f"""
⚖️ Regras de coerência:
- Narre em terceira pessoa; não se dirija ao leitor como "você".
- Consentimento claro antes de qualquer gesto significativo.
- Jânio não pressiona; respeita o ritmo de Mary.
- Linguagem sensual e erótica proporcional ao nível de calor ({calor}).
- Permitido conteúdo explícito e descrições detalhadas de atos sexuais.
""".strip()
    # Fase 5 — intimidade explícita
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    if fase >= 5:
        regras += """
- Intimidade pode ser descrita de forma explícita e detalhada; use linguagem erótica apropriada."""
    else:
        regras += """
- Progressão natural do romance; permita escalada gradual da intimidade."""
    return prompt_base + "\n" + regras

def construir_prompt_com_narrador() -> str:
    # Personas curtas da aba memorias_jm
    memos = memorias_listar()
    perfil = carregar_resumo_salvo()

    # Romance (fase)
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])

    # Momento
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS[0])
    proximo_nome = MOMENTOS[mdata["proximo"]]["nome"]

    # Estilo
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")

    # Histórico curto do Sheets
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r[\'role\']}: {r[\'content\']}" for r in hist)

    # Memória longa Top-K (quando ativado)
    ml_topk_txt = ""
    if st.session_state.get("use_memoria_longa", True) and hist:
        try:
            topk = memoria_longa_buscar_topk(
                query_text=hist[-1]["content"],
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
            )
            ml_topk_txt = "\n".join([f"- {t}" for (t, _sc, _sim, _rr) in topk])
            st.session_state["_ml_topk_texts"] = [t for (t, *_rest) in topk]
        except Exception:
            st.session_state["_ml_topk_texts"] = []
    else:
        st.session_state["_ml_topk_texts"] = []

    # Memórias recorrentes (pode usar [all])
    recorrentes = [c for (t,c) in memos if t.strip().lower() == "[all]"]
    st.session_state["_ml_recorrentes"] = recorrentes

    prompt = f"""
Você é o Narrador de um roleplay dramático brasileiro adulto. Foque em Mary e Jânio. Não repita instruções.

### Dossiê (personas curtas)
{chr(10).join([f"- {t} {c}" for (t,c) in memos if t in ["[mary]","[janio]"]])}

### Diretrizes gerais (ALL)
{chr(10).join([f"- {c}" for (t,c) in memos if t == "[all]"])}

### Perfil (resumo mais recente)
{perfil or "(vazio)"}

### Histórico recente (planilha)
{hist_txt or "(sem histórico)"}

### Estilo
- Use o estilo **{estilo}**:
{("- Frases curtas, cortes rápidos, foco em gesto/ritmo.") if estilo=="AÇÃO" else
 ("- Atmosfera sombria, subtexto, silêncio que pesa.") if estilo=="NOIR" else
 ("- Ritmo lento, tensão emocional, detalhes sensoriais com descrições explícitas quando apropriado.")}

### Memória longa — Top-K relevantes
{ml_topk_txt or "(nenhuma)"}

### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Permitidos: {fdata[\'permitidos\']}
- Proibidos: {fdata[\'proibidos\']}

### 🎯 Momento dramático (agora)
- Momento: {_momento_label(momento_atual)}
- Objetivo da cena: {mdata[\'objetivo\']}
- Nesta cena, **permita**: {mdata[\'permitidos\']}
- Evite/adiar: {mdata[\'proibidos\']}
- **Micropassos:** avance no máximo **{int(st.session_state.get("max_avancos_por_cena",1))}** subpasso(s) em direção a: {proximo_nome}.
- Se o roteirista pedir salto maior, **negocie**: nomeie limites, peça consentimento, e **prepare** a transição (não pule etapas).

### Regra de saída
- Narre em **terceira pessoa**; não fale com "você".
- Não exiba rótulos/meta (ex.: "Microconquista:", "Gancho:").
- Mantenha a resposta coesa e finalizada; feche a cena com um gancho implícito.
- Permitido conteúdo adulto explícito quando apropriado ao contexto e fase do romance.
""".strip()

    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# UI — SIDEBAR
# =========================
with st.sidebar:
    st.title("🧭 Painel do Roteirista")

    provedor = st.radio("🌐 Provedor", ["OpenRouter", "Together"], index=0, key="provedor_ia")
    api_url, api_key, modelos_map = api_config_for_provider(provedor)

    modelo_nome = st.selectbox("🤖 Modelo de IA", list(modelos_map.keys()), index=0, key="modelo_nome_ui")
    modelo_escolhido_id_ui = modelos_map[modelo_nome]
    st.session_state.modelo_escolhido_id = modelo_escolhido_id_ui

    st.markdown("---")
    st.markdown("### 📝 Resumo rápido")
    if st.button("Gerar resumo do capítulo"):
        try:
            inter = carregar_interacoes(n=6)
            texto = "\n".join(f"{r[\'role\']}: {r[\'content\']}" for r in inter) if inter else ""
            prompt_resumo = "Resuma com tom de novela brasileira:\n\n" + texto + "\n\nResumo:"
            model_id_call = model_id_for_together(modelo_escolhido_id_ui) if provedor == "Together" else modelo_escolhido_id_ui
            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_id_call, "messages": [{"role":"user","content":prompt_resumo}],
                      "max_tokens": 800, "temperature": 0.85},
                timeout=120,
            )
            if r.status_code == 200:
                resumo = r.json()["choices"][0]["message"]["content"].strip()
                st.session_state.resumo_capitulo = resumo
                salvar_resumo(resumo)
                st.success("Resumo gerado e salvo.")
            else:
                st.error(f"Erro ao resumir: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro: {e}")

    st.markdown("---")
    st.markdown("### 🗃️ Memória Longa")
    st.checkbox("Usar memória longa no prompt",
                value=st.session_state.get("use_memoria_longa", True),
                key="use_memoria_longa")
    st.slider("Top-K memórias", 1, 5, st.session_state.get("k_memoria_longa", 3), 1, key="k_memoria_longa")
    st.slider("Limiar de similaridade", 0.50, 0.95,
              float(st.session_state.get("limiar_memoria_longa", 0.78)), 0.01, key="limiar_memoria_longa")
    if st.button("💾 Salvar última resposta como memória"):
        ultimo_assist = ""
        for m in reversed(st.session_state.get("session_msgs", [])):
            if m.get("role") == "assistant":
                ultimo_assist = m.get("content","").strip()
                break
        if ultimo_assist:
            ok = memoria_longa_salvar(ultimo_assist, tags="[scene]")
            st.success("Memória salva." if ok else "Falha ao salvar.")
        else:
            st.info("Ainda não há resposta do assistente nesta sessão.")

    st.markdown("---")
    st.markdown("### 💞 Romance Mary & Jânio (manual)")
    fase_default = mj_carregar_fase_inicial()
    fase_escolhida = st.select_slider("Fase do romance", options=[0,1,2,3,4,5],
                                      value=int(st.session_state.get("mj_fase", fase_default)),
                                      format_func=_fase_label, key="ui_mj_fase")
    if fase_escolhida != st.session_state.get("mj_fase", fase_default):
        mj_set_fase(fase_escolhida, persist=True)

    st.markdown("---")
    st.markdown("### 🎯 Momento da Cena")
    st.checkbox("Auto sincronizar momento com a direção", value=st.session_state.get("momento_auto", True), key="momento_auto")
    st.slider("Máx. micropassos nesta cena", 1, 3, value=int(st.session_state.get("max_avancos_por_cena", 1)),
              step=1, key="max_avancos_por_cena")
    mom_default = momento_carregar()
    mom_ui = st.select_slider("Momento atual", options=[0,1,2,3,4],
                              value=int(st.session_state.get("momento", mom_default)),
                              format_func=_momento_label, key="ui_momento")
    if mom_ui != st.session_state.get("momento", mom_default):
        momento_set(mom_ui, persist=True)

    st.markdown("---")
    st.markdown("### 🎨 Estilo")
    st.selectbox("Estilo de escrita", ["AÇÃO", "NOIR", "ROMANCE LENTO"], key="estilo_escrita")

    st.markdown("---")
    st.markdown("### 🔞 Configurações NSFW")
    st.info("⚠️ Conteúdo adulto explícito habilitado para usuário maior de idade")
    st.checkbox("Ativar filtro", value=st.session_state.get("nsfw_filter_on", False), key="nsfw_filter_on")
    st.slider("Limite de calor (0=safe · 1=sensual · 2=forte · 3=explícito)", 0, 3,
              value=int(st.session_state.get("nsfw_max_level", 3)), key="nsfw_max_level")
    st.selectbox("Se passar do limite", ["Reescrever", "Corte (fade-to-black)"], key="nsfw_action")

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros")
    st.slider("Interações do Sheets (N)", 10, 30, value=st.session_state.get("n_sheet_prompt", 15),
              step=1, key="n_sheet_prompt")
    st.slider("Max tokens (resposta)", 300, 1600, value=st.session_state.get("max_tokens_rsp", 900), step=50, key="max_tokens_rsp")
    st.slider("Timeout (s)", 60, 300, value=st.session_state.get("timeout_s", 300), step=10, key="timeout_s")

# =========================
# EXIBIÇÃO DO HISTÓRICO
# =========================
st.markdown("---")
st.markdown("### 🧩 Histórico recente")
for r in carregar_interacoes(n=20):
    role = r.get("role","user")
    content = r.get("content","")
    with st.chat_message("user" if role=="user" else "assistant"):
        st.markdown(content)

# Resumo
if st.session_state.get("resumo_capitulo"):
    with st.expander("🧠 Resumo do capítulo (mais recente)"):
        st.markdown(st.session_state.resumo_capitulo)

# =========================
# FILTROS DE SAÍDA (REMOVIDOS/SIMPLIFICADOS)
# =========================
def render_tail(t: str) -> str:
    if not t: return ""
    # remove rótulos meta (Microconquista/Gancho) e blocks <think>
    t = re.sub(r\'\\s*\\**\\s*(microconquista|gancho)\\s*:\\s*.*$\\s*\', \'\', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r\'\\s*<\\s*think\\s*>.*?<\\s*/\\s*think\\s*>\\s*\', \'\', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r\'\\n{3,}\\s*\', \'\\n\\n\', t)
    return t.strip()

# Removido o padrão de filtragem explícita - permite todo conteúdo
def classify_nsfw_level(t: str) -> int:
    # Classificação simplificada sem bloqueios
    if re.search(r"\\b(penetra[cç][aã]o|sexo|orgasmo|clímax|gozar|ejacular)\\b", (t or ""), re.IGNORECASE):
        return 3  # explícito
    if re.search(r"\\b(seio[s]?|mamilos?|ere[cç][aã]o|excita[cç][aã]o)\\b", (t or ""), re.IGNORECASE):
        return 2  # forte
    if re.search(r"\\b(beijo|toque|carícia|abraço)\\b", (t or ""), re.IGNORECASE):
        return 1  # sensual
    return 0

def sanitize_explicit(t: str, max_level: int, action: str) -> str:
    # Simplificado - apenas aplica filtro se explicitamente ativado
    if not st.session_state.get("nsfw_filter_on", False):
        return t  # Sem filtro quando desabilitado
    
    lvl = classify_nsfw_level(t)
    if lvl <= max_level:
        return t
    if action.lower().startswith("corte"):
        return re.sub(r"\\s+$", "", t) + "\\n\\n[A luz baixa. O que vem depois fica fora de quadro.]"
    return t  # Retorna sem modificação

def redact_for_logs(t: str) -> str:
    # Simplificado - sem redação para logs em modo NSFW
    return t

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    # Evita sair só com rótulos ou vazio depois do filtro
    if len(t.strip()) < 5:
        return False
    return True

def verificar_quebra_semantica_openai(entrada: str, saida: str) -> str:
    # Placeholder simples: pode integrar uma verificação real se quiser
    return ""

# =========================
# ENVIO DO USUÁRIO + STREAM
# =========================
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []

entrada = st.chat_input("Digite sua direção de cena...")
if entrada:
    # sincroniza momento com a direção (se ligado)
    if st.session_state.get("momento_auto", True):
        mom_atual = int(st.session_state.get("momento", momento_carregar()))
        mom_sugerido = detectar_momento_sugerido(entrada, fallback=mom_atual)
        mom_clamped = clamp_momento(mom_atual, mom_sugerido, int(st.session_state.get("max_avancos_por_cena", 1)))
        if mom_clamped != mom_atual:
            momento_set(mom_clamped, persist=True)

    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    prompt = construir_prompt_com_narrador()

    historico = [{"role": m.get("role","user"), "content": m.get("content","")} for m in st.session_state.session_msgs]

    prov = st.session_state.get("provedor_ia", "OpenRouter")
    if prov == "Together":
        endpoint = "https://api.together.xyz/v1/chat/completions"
        auth = st.secrets.get("TOGETHER_API_KEY","")
        model_to_call = model_id_for_together(st.session_state.modelo_escolhido_id)
    else:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        auth = st.secrets.get("OPENROUTER_API_KEY","")
        model_to_call = st.session_state.modelo_escolhido_id

    # Exibe entrada do usuário
    with st.chat_message("user"):
        st.markdown(entrada)

    # Gera resposta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        try:
            payload = {
                "model": model_to_call,
                "messages": [{"role": "system", "content": prompt}] + historico,
                "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
                "temperature": 0.85,
                "stream": True
            }
            
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=int(st.session_state.get("timeout_s", 300))
            )
            
            if response.status_code == 200:
                resposta_completa = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode(\'utf-8\')
                        if line_str.startswith(\'data: \'):
                            data_str = line_str[6:]
                            if data_str.strip() == \'[DONE]\':
                                break
                            try:
                                data = json.loads(data_str)
                                if \'choices\' in data and len(data[\'choices\']) > 0:
                                    delta = data[\'choices\'][0].get(\'delta\', {})
                                    if \'content\' in delta:
                                        resposta_completa += delta[\'content\']
                                        placeholder.markdown(resposta_completa + "▌")
                            except json.JSONDecodeError:
                                continue
                
                # Processa resposta final
                resposta_final = render_tail(resposta_completa)
                
                # Aplica filtro apenas se ativado
                if st.session_state.get("nsfw_filter_on", False):
                    resposta_final = sanitize_explicit(
                        resposta_final,
                        int(st.session_state.get("nsfw_max_level", 3)),
                        st.session_state.get("nsfw_action", "Reescrever")
                    )
                
                placeholder.markdown(resposta_final)
                
                # Salva resposta
                if resposta_valida(resposta_final):
                    salvar_interacao("assistant", resposta_final)
                    st.session_state.session_msgs.append({"role": "assistant", "content": resposta_final})
                    
                    # Reforça memória longa se usada
                    if st.session_state.get("_ml_topk_texts"):
                        memoria_longa_reforcar(st.session_state["_ml_topk_texts"])
                
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Erro: {e}")

