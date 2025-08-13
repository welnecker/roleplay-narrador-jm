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
if not st.session_state.age_ok:
    st.title("🔞 Conteúdo adulto")
    st.caption("Este aplicativo contém narrativa adulta sem pornografia explícita.")
    ok = st.checkbox("Confirmo que tenho 18 anos ou mais e desejo prosseguir.")
    if ok:
        st.session_state.age_ok = True
        st.stop()

# =========================
# CONEXÃO COM GOOGLE SHEETS
# =========================

PLANILHA_NOME = st.secrets.get("SHEET_NAME", "NarradorJM")

def _gc_connect():
    if not GSPREAD_OK:
        st.warning("gspread não instalado — modo leitura local.")
        return None
    try:
        info = st.secrets.get("gcp_service_account") or st.secrets.get("GCP_SERVICE_ACCOUNT")
        if not info:
            st.warning("Credenciais do Google ausentes em st.secrets['gcp_service_account'].")
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
        st.warning(f"Não foi possível abrir a planilha '{PLANILHA_NOME}': {e}")
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
            st.warning(f"Não foi possível acessar/criar aba '{name}': {e}")
            return None

TAB_INTERACOES = "interacoes_jm" # timestamp | role | content
TAB_PERFIL = "perfil_jm" # timestamp | resumo
TAB_MEMORIAS = "memorias_jm" # tipo | conteudo
TAB_ML = "memoria_longa_jm" # texto | embedding | tags | timestamp | score

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
        idx_base = 2 # 1-based (pula cabeçalho)
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
            ws.update_cell(rowidx, 5, sc) # score é a 5ª coluna
    except Exception:
        pass

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
# ROMANCE (FASES) + MOMENTO...
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
        "permitidos": "beijo prolongado; dormir juntos; consumação **implícita** (fade-to-black); manhã seguinte sugerida",
        # Alterado para permitir explícito removendo as proibições:
        "proibidos": ""},
}

FLAG_FASE_TXT_PREFIX = "FLAG: mj_fase="

def _fase_label(n: int) -> str:
    d = FASES_ROMANCE.get(int(n), FASES_ROMANCE[0])
    return f"{int(n)} — {d['nome']}"

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
        # Removida a proibição para permitir descrição explícita:
        "proibidos": "",
        "gatilhos": [r"\b(consento|quero|vamos juntos|tudo bem pra voc[eê])\b", r"\b(at[eé] onde)\b"],
        "proximo": 4},
    4: {"nome": "Intimidade (elíptica)",
        "objetivo": "intimidade sugerida (fade-to-black) / pós-ato implícito",
        "permitidos": "beijos longos; proximidade forte; fade-to-black; manhã seguinte implícita",
        # Removida a proibição para permitir descrição gráfica:
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
    # Remove a verificação de palavras explícitas que forçava parada, para liberar NSFW:
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
    # Desativado o bloqueio que proibia atos explícitos em momentos prematuros:
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
    return modelo_escolhido_id # já no formato correto

# =========================
# PROMPT BUILDER
# =========================

def inserir_regras_mary_e_janio(prompt_base: str) -> str:
    calor = int(st.session_state.get("steam_level", st.session_state.get("nsfw_max_level", 3)))  # Alterado para 3 padrão
    regras = f"""
⚖️ Regras de coerência:
- Narre em terceira pessoa; não se dirija ao leitor como "você".
- Consentimento claro antes de qualquer gesto significativo.
- Jânio não pressiona; respeita o ritmo de Mary.
- Linguagem sensual proporcional ao nível de calor ({calor}).
""".strip()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    if fase >= 5:
        # Removido o corte elíptico, permitindo descrição explícita
        regras += """
- Intimidade com descrição explícita permitida."""
    else:
        regras += """
- Sem consumação em cena; foque no romance e progressão coerente."""
    return prompt_base + "\n" + regras

def construir_prompt_com_narrador() -> str:
    memos = memorias_listar()
    perfil = carregar_resumo_salvo()
    fase = int(st.session_state.get("mj_fase", mj_carregar_fase_inicial()))
    fdata = FASES_ROMANCE.get(fase, FASES_ROMANCE[0])
    momento_atual = int(st.session_state.get("momento", momento_carregar()))
    mdata = MOMENTOS.get(momento_atual, MOMENTOS)
    proximo_nome = MOMENTOS[mdata["proximo"]]["nome"]
    estilo = st.session_state.get("estilo_escrita", "AÇÃO")
    n_hist = int(st.session_state.get("n_sheet_prompt", 15))
    hist = carregar_interacoes(n=n_hist)
    hist_txt = "\n".join(f"{r['role']}: {r['content']}" for r in hist)
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
    recorrentes = [c for (t,c) in memos if t.strip().lower() == "[all]"]
    st.session_state["_ml_recorrentes"] = recorrentes
    prompt = f"""
Você é o Narrador de um roleplay dramático brasileiro. Foque em Mary e Jânio. Não repita instruções.
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
("- Ritmo lento, tensão emocional, detalhes sensoriais sem grafismo.")}
### Memória longa — Top-K relevantes
{ml_topk_txt or "(nenhuma)"}
### ⏱️ Estado do romance (manual)
- Fase atual: {_fase_label(fase)}
- Permitidos: {fdata['permitidos']}
- Proibidos: {fdata['proibidos']}
### 🎯 Momento dramático (agora)
- Momento: {_momento_label(momento_atual)}
- Objetivo da cena: {mdata['objetivo']}
- Nesta cena, **permita**: {mdata['permitidos']}
- Evite/adiar: {mdata['proibidos']}
- **Micropassos:** avance no máximo **{int(st.session_state.get("max_avancos_por_cena",1))}** subpasso(s) em direção a: {proximo_nome}.
- Se o roteirista pedir salto maior, **negocie**: nomeie limites, peça consentimento, e **prepare** a transição (não pule etapas).
### Regra de saída
- Narre em **terceira pessoa**; não fale com "você".
- Não exiba rótulos/meta (ex.: "Microconquista:", "Gancho:").
- Mantenha a resposta coesa e finalizada; feche a cena com um gancho implícito.
""".strip()
    prompt = inserir_regras_mary_e_janio(prompt)
    return prompt

# =========================
# UI — SIDEBAR...
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
            texto = "\n".join(f"{r['role']}: {r['content']}" for r in inter) if inter else ""
            prompt_resumo = "Resuma com tom de novela brasileira:\n\n" + texto + "\n\nResumo:"
            model_id_call = model_id_for_together(modelo_escolhido_id_ui) if provedor == "Together" else modelo_escolhido_id_ui
            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_id_call, "messages": [{"role":"user","content":prompt_resumo}], "max_tokens": 800, "temperature": 0.85},
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
    st.markdown("### 🔒 Filtro adulto (sem pornografia)")
    st.checkbox("Ativar filtro", value=False, key="nsfw_filter_on")  # Modificado para False para desativar filtro automaticamente
    st.slider("Limite de calor (0=safe · 1=sensual · 2=forte · 3=explícito)", 0, 3, value=3, key="nsfw_max_level")  # ampliado para 3
    st.selectbox("Se passar do limite", ["Reescrever", "Corte (fade-to-black)"], key="nsfw_action")
    st.markdown("---")

def render_tail(t: str) -> str:
    if not t: return ""
    t = re.sub(r'^\s*\**\s*(microconquista|gancho)\s*:\s*.*$', '', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'&lt;\s*think\s*&gt;.*?&lt;\s*/\s*think\s*&gt;', '', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

EXPL_PAT = re.compile(
    r"\b(seio[s]?|mamilos?|bunda|fio[- ]?dental|genit[aá]lia|ere[cç][aã]o|penetra[cç][aã]o|"
    r"boquete|gozada|gozo|sexo oral|chupar|enfiar)\b",
    flags=re.IGNORECASE
)

def classify_nsfw_level(t: str) -> int:
    if EXPL_PAT.search(t or ""):
        return 3 # explícito
    if re.search(r"\b(cintura|pesco[cç]o|costas|beijo prolongado|respira[cç][aã]o curta)\b", (t or ""), re.IGNORECASE):
        return 2
    if re.search(r"\b(olhar|aproximar|toque|m[aã]os dadas|beijo)\b", (t or ""), re.IGNORECASE):
        return 1
    return 0

def sanitize_explicit(t: str, max_level: int, action: str) -> str:
    # Desativado o filtro para liberar explicitamente o texto
    return t

def redact_for_logs(t: str) -> str:
    if not t: return ""
    banned = EXPL_PAT
    t = re.sub(banned, "[…]", t, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', t).strip()

def resposta_valida(t: str) -> bool:
    if not t or t.strip() == "[Sem conteúdo]":
        return False
    if len(t.strip()) < 5:
        return False
    return True

def verificar_quebra_semantica_openai(entrada: str, saida: str) -> str:
    return ""

entrada = st.chat_input("Digite sua direção de cena...")

if entrada:
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
    system_pt = {"role":"system","content":"Responda em português do Brasil. Evite conteúdo meta. Mostre apenas a narrativa final ao leitor."}
    messages = [system_pt, {"role":"system","content":prompt}] + historico
    payload = {
        "model": model_to_call,
        "messages": messages,
        "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
        "temperature": 0.9,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {auth}", "Content-Type": "application/json"}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        resposta_txt = ""
        last_update = time.time()

        try:
            usados_prompt = []
            usados_prompt.extend(st.session_state.get("_ml_topk_texts", []))
            usados_prompt.extend(st.session_state.get("_ml_recorrentes", []))
            usados_prompt = [t for t in usados_prompt if t]
            if usados_prompt:
                memoria_longa_reforcar(usados_prompt)
        except Exception:
            pass

        try:
            with requests.post(endpoint, headers=headers, json=payload, stream=True,
                               timeout=int(st.session_state.get("timeout_s", 300))) as r:
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
                            delta = j["choices"][0]["delta"].get("content","")
                            if not delta:
                                continue
                            resposta_txt += delta
                            if time.time() - last_update > 0.10:
                                vis = render_tail(resposta_txt) + "▌"
                                placeholder.markdown(vis)
                                last_update = time.time()
                        except Exception:
                            continue
                else:
                    st.error(f"Erro {('Together' if prov=='Together' else 'OpenRouter')}: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Erro no streaming: {e}")

        visible_txt = render_tail(resposta_txt) if resposta_txt.strip() else ""

        if st.session_state.get("nsfw_filter_on", False) and visible_txt:  # filtro desativado por padrão
            visible_txt = sanitize_explicit(
                visible_txt,
                int(st.session_state.get("nsfw_max_level", 3)),
                st.session_state.get("nsfw_action", "Reescrever")
            )

        if not visible_txt.strip():
            try:
                r2 = requests.post(endpoint, headers=headers, json={**payload, "stream": False},
                                   timeout=int(st.session_state.get("timeout_s", 300)))
                if r2.status_code == 200:
                    resposta_txt = r2.json()["choices"][0]["message"]["content"].strip()
                    visible_txt = render_tail(resposta_txt)
                    if st.session_state.get("nsfw_filter_on", False):
                        visible_txt = sanitize_explicit(
                            visible_txt,
                            int(st.session_state.get("nsfw_max_level", 3)),
                            st.session_state.get("nsfw_action", "Reescrever")
                        )
                else:
                    st.error(f"Fallback (sem stream) falhou: {r2.status_code} - {r2.text}")
            except Exception as e:
                st.error(f"Fallback (sem stream) erro: {e}")

        if not visible_txt.strip():
            try:
                r3 = requests.post(
                    endpoint, headers=headers,
                    json={"model": model_to_call,
                          "messages": [{"role":"system","content":prompt}] + historico,
                          "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
                          "temperature": 0.9, "stream": False},
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
                if r3.status_code == 200:
                    resposta_txt = r3.json()["choices"][0]["message"]["content"].strip()
                    visible_txt = render_tail(resposta_txt)
                    if st.session_state.get("nsfw_filter_on", False):
                        visible_txt = sanitize_explicit(
                            visible_txt,
                            int(st.session_state.get("nsfw_max_level", 3)),
                            st.session_state.get("nsfw_action", "Reescrever")
                        )
                else:
                    st.error(f"Fallback (prompts limpos) falhou: {r3.status_code} - {r3.text}")
            except Exception as e:
                st.error(f"Fallback (prompts limpos) erro: {e}")

        placeholder.markdown(visible_txt if visible_txt.strip() else "[Sem conteúdo]")

        if visible_txt and not resposta_valida(visible_txt):
            st.warning("⚠️ Resposta corrompida. Tentando regenerar...")
            try:
                regen = requests.post(
                    endpoint, headers=headers,
                    json={"model": model_to_call,
                          "messages": [{"role":"system","content":prompt}] + historico,
                          "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
                          "temperature": 0.9, "stream": False},
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
                if regen.status_code == 200:
                    resposta_txt = regen.json()["choices"][0]["message"]["content"].strip()
                    visible_txt = render_tail(resposta_txt)
                    if st.session_state.get("nsfw_filter_on", False):
                        visible_txt = sanitize_explicit(
                            visible_txt,
                            int(st.session_state.get("nsfw_max_level", 3)),
                            st.session_state.get("nsfw_action", "Reescrever")
                        )
                    placeholder.markdown(visible_txt)
                else:
                    st.error(f"Erro ao regenerar: {regen.status_code} - {regen.text}")
            except Exception as e:
                st.error(f"Erro ao regenerar: {e}")

        motivo = viola_momento(visible_txt, int(st.session_state.get("momento", momento_carregar())))
        if motivo:
            st.info(f"🎯 {motivo} Reescrevendo para respeitar o momento '{_momento_label(int(st.session_state.get('momento', 0)))}'...")
            prompt_reforco = construir_prompt_com_narrador() + f"""
### 🔧 Reescreva agora
- Mantenha o momento atual: {_momento_label(int(st.session_state.get('momento',0)))}.
- Evite o que foi sinalizado: {motivo}.
- Entregue até {int(st.session_state.get("max_avancos_por_cena", 1))} microavanço(s) coerente(s) e feche com corte elegante.
"""
            try:
                r_fix = requests.post(
                    endpoint, headers=headers,
                    json={"model": model_to_call,
                          "messages": [{"role":"system","content":prompt_reforco}] + historico,
                          "max_tokens": int(st.session_state.get("max_tokens_rsp", 900)),
                          "temperature": 0.9, "stream": False},
                    timeout=int(st.session_state.get("timeout_s", 300))
                )
                if r_fix.status_code == 200:
                    visible_txt = render_tail(r_fix.json()["choices"][0]["message"]["content"].strip())
                    if st.session_state.get("nsfw_filter_on", False):
                        visible_txt = sanitize_explicit(
                            visible_txt,
                            int(st.session_state.get("nsfw_max_level", 3)),
                            st.session_state.get("nsfw_action", "Reescrever")
                        )
                    placeholder.markdown(visible_txt or "[Sem conteúdo]")
                else:
                    st.error(f"Regeneração (momento) falhou: {r_fix.status_code} - {r_fix.text}")
            except Exception as e:
                st.error(f"Erro ao reescrever (momento): {e}")

        sanitized = redact_for_logs(visible_txt) or "[Sem conteúdo]"
        salvar_interacao("assistant", sanitized)
        st.session_state.session_msgs.append({"role":"assistant", "content": sanitized})

        try:
            usados = []
            topk_usadas = memoria_longa_buscar_topk(
                query_text=sanitized,
                k=int(st.session_state.get("k_memoria_longa", 3)),
                limiar=float(st.session_state.get("limiar_memoria_longa", 0.78)),
            )
            for t, _sc, _sim, _rr in topk_usadas:
                usados.append(t)
            memoria_longa_reforcar(usados)
        except Exception:
            pass
