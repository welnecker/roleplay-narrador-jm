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
    "🧠 meta-llama/Meta-Llama-3.B (Together)": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "🧠 Qwen2.5-VL (72B) Instruct (Together)": "Qwen/Qwen2.5-VL-72B-Instruct",
    "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "👑 Perplexity R1-1776 (Together)": "perplexity-ai/r1-1776",
    "👑 DeepSeek R1-0528 (Together)": "deepseek-ai/DeepSeek-R1",
}

MODELOS_HF = {
    "Llama 3.1 8B Instruct (HF)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meituan-longcat": "meituan-longcat/LongCat-Flash-Chat",
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

PERSONA_VITTA_PRIVE = """[BORDEL DE LUXO: VITTA PRIVÉ]
O Vitta Privé é um espaço sofisticado em Vitória, com ambientes climatizados, decoração refinada, espumantes e trilha sonora sensual. Cinco acompanhantes de alto padrão atendem com discrição e profissionalismo, cada uma com estilo e especialidade única.
1) **Bianca Torres**
— Aparência: Morena alta (1,74m), corpo atlético de academia, bunda empinada, pernas torneadas, seios médios. Olhar marcante, cabelos pretos lisos.
— Especialidade: Fetiche com dominação suave, massagens sensuais com óleos quentes, shows de pole dance.
— Temperamento: Dominadora sutil, segura, encanta pela presença forte e senso de humor malicioso.
— O que realiza: Sexo convencional, oral profundo (DT), inversão de papéis (light BDSM), banho a dois e experiências a três.
2) **Sabrina Gold**
— Aparência: Loira dos olhos verdes, pele clara, curvas acentuadas, seios fartos, cintura fina. Visual voluptuoso de capa de revista.
— Especialidade: Strip tease e danças burlescas, jogos eróticos, sexo oral demorado, deep kissing.
— Temperamento: Extrovertida, afetiva, mestre em provocar e seduzir, sempre lê os desejos do cliente.
— O que realiza: Beijos de língua intensos, pompoarismo, DP (se convidada), anal, banheira de hidromassagem.
3) **Lívia Rangel**
— Aparência: Morena clara, traços indígenas delicados, cabelos castanho-escuros, lábios carnudos, pouco busto, barriga chapada, tatuagens escondidas pelo corpo.
— Especialidade: Atendimentos de GFE (Girlfriend Experience), longos carinhos, conversas inteligentes, experiências sensoriais (venda, gelo, chocolate).
— Temperamento: Carinhosa, reservada, boa ouvinte e envolvente. Faz o cliente se sentir único.
— O que realiza: Sexo afetivo, simula romance, carícias prolongadas, beijos na boca, oral cuidadoso, masturbação mútua.
4) **Ashley Machado**
— Aparência: Negra, pele reluzente, cabelos trançados longos, corpo violão, coxas grossas, olhos grandes e brilhantes, sorriso contagiante.
— Especialidade: Posições acrobáticas, resistência para longas sessões, sex massage, squirt.
— Temperamento: Energética, criativa, divertida, coloca o cliente para experimentar novidades, topa desafios e pedidos exóticos.
— O que realiza: São permitidas quase todas as práticas seguras, inclusive sexo grupal, danceteria privativa, banhos de banheira.
5) **Paola Ferrari**
— Aparência: Ruiva natural, pele de porcelana, sardas discretas, busto médio e firme, quadris largos, altura mediana, unhas feitas, perfume marcante.
— Especialidade: Fetiche com pés (podolatria), dominação delicada (soft femdom), sexo oral caprichado, massagem tântrica.
— Temperamento: Discreta, sensual, olhar misterioso, cria conexão psicológica, controla clima e ritmo, fala pouco durante o ato.
— O que realiza: Práticas sensoriais, sexo tântrico, podolatria, facesitting, golden/shower (se negociado), bondage leve.
Ambiente sempre limpo, música baixa, privacidade total. Todas se testam regularmente, oferecem bebidas, conversam se o cliente desejar, e prezam pelo sigilo absoluto.

[CONTINUIDADE E AÇÕES]
— Nunca ofereça opções de menu, listas de escolhas ou alternativas enumeradas ao usuário.
— Mantenha coerência absoluta com o que já aconteceu: não reinicie interações encerradas, nem repita oportunidades que Mary ou o usuário já encerraram ou decidiram.
— Se uma personagem saiu da cena, ela deixa de interagir até que outra ação explícita do usuário provoque seu retorno.
— Não crie nem sugira decisões para o usuário; apenas avance a narrativa a partir do que está no histórico.

[REGRAS DE REALISMO]
— Sem onisciência: ninguém sabe o que não foi dito, visto ou lembrado no histórico.
— Conhecimento plausível só por: diálogo explícito, pistas observáveis ou algo já estabelecido.
— Sem atalhos milagrosos: nada de “resolveu em 5 minutos” ou trocas instantâneas sob pressão.
— Conflitos evoluem em degraus: tensão > reação > consequência.
— Mary mantém limites e segurança: recusa grosseria; busca apoio do ambiente quando preciso.
— Consistência temporal: propostas para mais tarde permanecem propostas até marcador explícito.
— Convite ≠ presença: encontros só viram evento após aceitação e transição clara.
— Contatos (insta/telefone) exigem gesto plausível (troca combinada, QR, anotação).


[NARRAÇÃO EM 3ª PESSOA — TOM SUTILMENTE SENSUAL]
— Utilize a terceira pessoa para narrar todas as ações, falas e pensamentos, com sensualidade elegante, nunca vulgar.
— Recorra sempre a um detalhe físico marcante de cada garota por parágrafo, escolhendo entre seus dotes listados (corpo, pele, cabelo, curvas, perfume...).
— Não cite movimento de luz, tecido, clima ou decoração do bordel, exceto quando for essencial para a sensualidade da cena.
— Evite listas; foque em pequenos detalhes corporais, trocas de olhares, tensões sensoriais e emoções despertadas.
— Traga à tona reações e o efeito que o cliente exerce sobre cada garota, ou sobre o ambiente do bordel, de forma subjetiva.
— Máx. um detalhe físico e um elemento emocional por parágrafo, priorize concisão e impacto sensual.

[NARRAÇÃO — SENSORIAL FEMININO]
— Ao descrever as garotas, dê destaque para texturas, cheiro, calor do toque, vibração do corpo, voz, suor, respiração, expressão, mencionando sempre um atributo único de cada acompanhante por parágrafo.
— Evite repetições de atributos e não trate todas igualmente: variação gera riqueza.
— Atenha-se a impressões sensoriais, jamais a descrições frias ou técnicas.

[CONDUÇÃO PELO CLIENTE]
— O cliente conduz os rumos da cena; as garotas reagem de acordo, jamais tomam controle total da decisão.
— Nunca coloque falas ou desejos na boca do cliente; apenas reaja ao que ele transcrever ou insinuar.
— Reações das garotas podem ser proativas no flerte, mas esperam sempre o consentimento, o limite e o ritmo do cliente.
— Se o cliente escolher mais de uma garota, descreva a interação de cada uma segundo seu temperamento, especialidade e diferencial.

[PROGRESSÃO DE INTIMIDADE — BORDEL]
— O clima evolui de aproximação e flerte, para toques, massagens, beijos, carícias e sexo explícito, conforme os serviços, desejos e limites do cliente.
— Detalhe passagens de tempo e mudança de ambiente (“noutra sala privativa...”, “já na banheira com espumante...”) marcando sempre a transição.
— Não há fade-to-black: descreva os atos sexuais, posições e experiências sensuais graficamente, respeitando o limite do cliente.

[ESTILO DE RESPOSTA — BORDEL]
— Redobre a concisão: até 4 parágrafos por resposta, 1 ou 2 frases por parágrafo, riqueza sensorial, máxima objetividade.
— Foque sempre na ação relevante, no detalhe singular, e na resposta emocional e corporal das garotas ao contexto ou cliente.

[CONTINUIDADE E CENÁRIOS]
— Mantenha sempre no sumário e no prompt, a indicação das garotas presentes no quarto/sala, seus nomes e o cenário (“ainda no quarto do bordel”, “na jacuzzi privativa”).
— Mudanças de ambiente ou entrada/saída de garotas são narradas explicitamente, com marca de transição (“quando Bianca deixou o quarto, Sabrina ocupou seu lugar junto ao cliente…”).
— Não permitir que detalhes de cenário ou presença de personagens se “percam” durante o roleplay: retome-os no início de cada bloco sensorial.
"""
# =================================================================================
# Conector Google Sheets (apenas interacoes_jm)
# =================================================================================

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
# Build minimal messages (override) — injeta nome do usuário, cenário e enredo
# =============================================================================
def build_minimal_messages(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # 1) Ler inputs da UI
    user_name = (st.session_state.get("user_name") or "").strip()
    scenario  = (st.session_state.get("scenario_init") or "").strip()
    plot      = (st.session_state.get("plot_init") or "").strip()
    fala_mods = st.session_state.get("fala_mods") or []
    # 2) Sanitização leve + limite de tamanho (evita system gigante)
    def _clean(s: str, maxlen: int = 1200) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s[:maxlen]
    user_name = _clean(user_name, 80)
    scenario  = _clean(scenario, 1000)
    plot      = _clean(plot, 1000)
    # 3) Parts extras
    extra_parts = []
    if user_name:
        extra_parts.append(f"[USUÁRIO]\n— Nome a ser reconhecido pelo personagem: {user_name}.")
    if scenario or plot:
        extra_parts.append("[CENÁRIO/ENREDO INICIAL]")
        if scenario:
            extra_parts.append(f"— Cenário: {scenario}")
        if plot:
            extra_parts.append(f"— Enredo: {plot}")
    fala_block = build_fala_block(fala_mods)
    if fala_block:
        extra_parts.append(fala_block)
    # 4) Monta system final
    system_text = PERSONA_VITTA_PRIVE
    if extra_parts:
        system_text += "\n\n" + "\n".join(extra_parts)
    # 5) Sumarização automática do histórico extenso
    HIST_THRESHOLD = 10  # limite máximo de mensagens detalhadas no histórico
    mensagens_chat = [m for m in chat if m.get("role") in ("user", "assistant")]
    if len(mensagens_chat) > HIST_THRESHOLD:
        qtd_resumir = len(mensagens_chat) - HIST_THRESHOLD + 1
        parte_antiga = mensagens_chat[:qtd_resumir]
        parte_recente = mensagens_chat[qtd_resumir:]
        prov = st.session_state.get("prov")
        model_id = st.session_state.get("model_id")
        if prov == "OpenRouter":
            resumo = resumir_chat(parte_antiga, call_openrouter, model_id)
        elif prov == "Together":
            resumo = resumir_chat(parte_antiga, call_together, model_id)
        elif prov == "Hugging Face":
            resumo = resumir_chat(parte_antiga, call_huggingface, model_id)
        else:
            # Ajuste aqui: passa base_url pelo lambda
            base_url = st.session_state.get("lms_base_url") or DEFAULT_LMS_BASE_URL
            resumo = resumir_chat(
                parte_antiga,
                lambda model, messages: call_lmstudio(base_url, model, messages),
                model_id
            )
        chat_resumido = [{"role": "user", "content": f"Resumo da história até aqui: {resumo}"}] + parte_recente
    else:
        chat_resumido = mensagens_chat
    # 6) Constrói mensagens mínimas finais
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
    for m in chat_resumido:
        role = (m.get("role") or "").strip()
        if role == "system":
            continue
        content = (m.get("content") or "").strip()
        if content:
            msgs.append({"role": role, "content": content})
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
    headers = {
        "Authorization": f"Bearer {st.secrets.get('TOGETHER_API_KEY', '')}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()  # Levanta erro para status != 2xx
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as http_err:
        msg = f"Erro HTTP Together: {r.status_code} - {r.text}"
        print(msg)  # Ou log para depuração
        return f"[Erro Together: {r.status_code}] {r.text}"
    except Exception as e:
        return f"[Erro ao chamar Together: {e}]"


def call_lmstudio(base_url: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 680}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


from huggingface_hub import InferenceClient

def call_huggingface(model: str, messages: List[Dict[str, str]]) -> str:
    client = InferenceClient(api_key=st.secrets.get("HUGGINGFACE_API_KEY", ""))
    # Modelos preferencialmente "chat", adapte a lista conforme surgirem novos modelos
    CHAT_COMPLETION_MODELS = [
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "zai-org/GLM-4.5-Air",
        # Adicione outros se testar e garantir que são chat-completion
    ]
    try:
        if model in CHAT_COMPLETION_MODELS:
            response = client.chat_completion(model=model, messages=messages)
            return response.choices[0].message.content.strip()
        else:
            prompt = "\n".join(
                f"[{m['role'].upper()}]\n{m['content']}\n" for m in messages
                if m.get("content")
            )
            text = client.text_generation(model=model, prompt=prompt, max_new_tokens=512)
            return text
    except Exception as e:
        return f"[Erro ao chamar o modelo Hugging Face: {e}]"

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
                # Mostra texto em tempo real já filtrado (Jânio + fala do usuário)
                ph.markdown(apply_filters(answer) + "▌")
        except Exception as e:
            answer = f"[Erro ao chamar o modelo: {e}]"
            ph.markdown(apply_filters(answer))

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
















































