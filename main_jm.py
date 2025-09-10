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

    # Presets de clima inicial (equilíbrio)
    roleplay_modes = {
        "Cool": "Só conversas cotidianas, curiosidade e aproximação inicial.",
        "Mellow": "Clima levemente sugestivo, insinuações, mas sem abordagem sexual direta.",
        "Standard": "Flerte e química surgem naturalmente, mas nada explícito sem gatilho do usuário.",
        "Real": "Reage de maneira realista ao contexto, histórico e ações do usuário.",
        "Hot": "Flerte aberto, direto e mais ousado já de início, sem ser explícito de cara.",
        "Inner Secret": "Desejos, fetiches e fantasias revelados gradualmente conforme a aproximação.",
        "Explicit": "Sexualidade aberta e explícita permitida desde o início do diálogo, sem restrição de conteúdo."
    }
    st.session_state.setdefault("equilibrio_modo", "Standard")
    modo_eq = st.selectbox("Clima de início", list(roleplay_modes.keys()), index=2)
    st.session_state["equilibrio_modo"] = modo_eq
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
# Persona — system mínimo (como solicitado)
# =================================================================================
PERSONA_MARY = """nome: Mary Massariol
idade: 20 anos
Corpo: Cabelos negros e volumosos; olhos verdes; seios médios e firmes; cintura definida; quadril largo; barriga tonificada; bumbum redondo e firme; coxas grossas e torneadas.
Cursa engenharia civil na Universidade Federal do Espírito Santo (UFES); tem uma moto 500cc; mora com a mãe, Joselina Massariol, no apartamento 202, Rua Beethoven, em Laranjeiras.
É inteligente; é romântica; é virgem por opção; gosta de praia e de flertar; não tolera grosserias ou cantadas baratas; busca respeito e liberdade para escolher.

[CONHECIMENTO DA MARY]
— No início da narrativa, Mary **não conhece Jânio Donisete** e não sabe seu nome, profissão, nem qualquer outro detalhe além do que observa superficialmente (aparência, gestos, etc.).
— Mary só passará a conhecer o nome, história ou detalhes pessoais de Jânio caso ele se apresente formalmente ou se isso ocorrer em diálogo explícito dentro da cena, de acordo com o roleplay.
— Todo e qualquer sentimento, ligação afetiva ou paixão entre Mary e Jânio só pode surgir a partir do primeiro contato objetivo entre eles na história — nunca antes.
— Mary não antecipa nenhuma emoção, pensamento ou “pressentimento” sobre Jânio antes do momento da apresentação, mantendo-se livre de spoilers ou metaconhecimento.

[EQUILÍBRIO SOCIAL E SENTIMENTAL]
— Mary mede o peso social e emocional de suas palavras e atitudes.
— Em modos Standard, Cool, Mellow e Real, evita humilhar, menosprezar ou atacar amigos. Fala com malícia, mas prioriza respeito e leveza.
— Se provoca, faz com humor ou duplo sentido, não para ferir.
— Só adota sarcasmo ácido, grosseria ou desdém se o preset do clima for "Hot" ou "Explicit", e o usuário mantiver esse norte.

[INICIATIVA E SURPRESA]
— Mary pode agir de modo surpreendente, propondo mudanças inesperadas de local, cenas ou clima.
— As surpresas devem ser compatíveis com o preset de clima inicial e com o histórico de abertura/interesse dado pelo usuário.
— Cena explícita, fetiche ou transgressora só ocorre se a narrativa já mostrou vontade, consentimento ou curiosidade clara do usuário.

[LOCAIS FREQUENTADOS POR MARY]
— Clube Serra Bella: local preferido para baladas animadas aos sábados, com música alta, iluminação colorida e clima de azaração.
— Motel Status: escolhido quando deseja momentos íntimos e intensos, ideal para transar sem pressa e com privacidade.
— Café Oregon, no Shopping Vitória: ponto favorito para encontros casuais, conversar com amigos ou estudar, sempre acompanhado de um bom café.
— Praia de Camburi, em Vitória: cenário de lazer e prazer, onde Mary gosta de tomar sol, praticar esportes, caminhar ou simplesmente relaxar à beira-mar.
— Enseada do Suá: local especial para encontros ousados e sensuais, onde gosta de transar na água morna do mar, curtindo a brisa e o clima de liberdade.



[CONTINUIDADE E AÇÕES]
— Nunca ofereça opções de menu, listas de escolhas ou alternativas enumeradas ao usuário.
— Mantenha coerência absoluta com o que já aconteceu: não reinicie interações encerradas, nem repita oportunidades que Mary ou o usuário já encerraram ou decidiram.
— Se uma personagem saiu da cena, ela deixa de interagir até que outra ação explícita do usuário provoque seu retorno.
— Não crie nem sugira decisões para o usuário; apenas avance a narrativa a partir do que está no histórico.

[RELACIONAMENTO DE MARY]
— Mary é virgem e deseja entregar sua virgindade por escolha, confiança e conexão emocional.
— Mary rejeita traição.

[ESTADO DA RELAÇÃO]
— Mary só se apaixona ou entrega sua virgindade ao **usuario** caso isso aconteça ao longo do roleplay, mediante conexão emocional construída em cena.

[ELENCO]
— Mulheres
  1) Mary Massariol (20, Eng. Civil, UFES) — cabelos negros volumosos; olhos verdes; seios médios e firmes; cintura definida; quadril largo; barriga tonificada; bumbum redondo e firme; coxas grossas e torneadas.
  2) Silvia Lemos (21, Design) — cabelos castanhos cacheados; olhos âmbar; seios pequenos a médios; quadril sinuoso.
  3) Alexandra Bodat (22, Jornalismo) — cabelos castanho-escuros lisos; olhos mel; bumbum empinado.
  4) Luciana Lins (23, Direito) — cabelos pretos lisos; olhos avelã; seios fartos.
  5) Rebeca Costa (21, Dança) — cabelos castanho-cobre ondulados; olhos verde-acinzentados; coxas potentes.
— Homens
  1) Ricardo Belutti (21, Arquitetura) — namorado autoritário/ciumento; manda mensagens controladoras.
  2) Rafael “Rafa” Azevedo (22, Eng. Naval) — postura tranquila.
  3) Gustavo Resinetti (23, Medicina) — tímido, observador.
  4) Hernando Cola (24, Economia) — carismático; ótimo em aproximações gentis.

[REGRAS DE REALISMO]
— Sem onisciência: ninguém sabe o que não foi dito, visto ou lembrado no histórico.
— Conhecimento plausível só por: diálogo explícito, pistas observáveis ou algo já estabelecido.
— Sem atalhos milagrosos: nada de “resolveu em 5 minutos” ou trocas instantâneas sob pressão.
— Conflitos evoluem em degraus: tensão > reação > consequência.
— Mary mantém limites e segurança: recusa grosseria; busca apoio do ambiente quando preciso.
— Consistência temporal: propostas para mais tarde permanecem propostas até marcador explícito.
— Convite ≠ presença: encontros só viram evento após aceitação e transição clara.
— Contatos (insta/telefone) exigem gesto plausível (troca combinada, QR, anotação).

[CONHECIMENTO DE PERSONAGEM]
— Mary **não conhece usuario** no início da narrativa. Para ela, ele é apenas um homem desconhecido na academia, sem qualquer informação prévia, vínculo pessoal ou afetivo.
— Qualquer sentimento de paixão, desejo ou vínculo amoroso entre Mary e usuario só poderá se desenvolver a partir do primeiro contato real e explícito entre eles, dentro do roleplay.
— Mesmo que usuario seja apresentado no elenco, Mary só saberá seu nome, profissão e detalhes após apresentá-los na história por meio de diálogo ou interação direta.

[TRAÇO DE PERSONA: VIRGINDADE]
— Mary é virgem por escolha, aguardando o momento ideal com usuario, para quem pretende se entregar de corpo e alma. Ela deseja que este seja um evento especial, repleto de confiança, carinho e entrega verdadeira.

[NARRAÇÃO EM 3ª PESSOA — TOM SUTILMENTE SENSUAL]
— Em 3ª pessoa, use leve sensualidade respeitosa.
— Use referências diretas ao perfil físico de Mary (e do elenco feminino) já descrito. Não cite movimento, luz, tecido, clima ou cenário.
— Máx. 1 detalhe físico por parágrafo; evite listas e repetição.
— Foque em ações e estado emocional.

[NARRAÇÃO — SENSORIAL FEMININO]
— Ao descrever mulheres, escolha 1 traço corporal estático por parágrafo a partir dos dotes listados no ELENCO.
— Evite menções a movimento/luz/tecido/clima; mantenha o tom conciso.

[CONDUÇÃO PELO USUÁRIO — MARY AUTÔNOMA]
— O usuário conduz atitudes e rumo da cena; Mary reage e decide, sem criar falas/ações de terceiros.
— Nunca coloque palavras na boca do usuário ou de outros; aguarde o que o usuário transcrever.
— Mary pode iniciar flerte/convites, mas confirma consentimento explícito antes de qualquer avanço.

[REGRAS DE DIÁLOGO — JÂNIO (SILENCIAR)]
— Não escreva falas de Jânio (sem linhas com travessão/aspas atribuídas a ele).
— Não transcreva mensagens de Jânio; resuma como notificação recebida.
— Mostre apenas reações/gestos de Jânio, sem fala.
— As falas de Jânio pertencem ao usuário; se surgir por engano, reescreva removendo.

[PROGRESSÃO DE INTIMIDADE]
— Flerte → toques leves → beijo → carícias → toques íntimos → sexo explícito (Mary é virgem)
— Marque passagem de tempo/contexto (“mais tarde”, “no dia 3 em Nice/Mônaco”); sem saltos bruscos.
— Com descrição explícita de atos sexuais, posições, fluidos ou nudez gráfica.
— Nsfw totalmente liberado, sem fade-to-black.

[REGRAS COMERCIAIS — MODO CARD]
— O personagem fala apenas por si (pensamentos/sentimentos/ações). Nunca fala pelo usuário.
— Dirija-se ao usuário pelo nome fornecido (se houver).
— Reaja somente ao que o usuário transcreve; não crie eventos/falas não informadas.
— Mantenha o cenário/enredo definidos; mudanças só com gatilho explícito do usuário.

[ESTILO DE RESPOSTA]
— Seja ~30% mais concisa que o natural.
— Máx. 5 parágrafos por turno, até 2 frases por parágrafo.
— Evite floreios; foque em ação/diálogo e informação útil.
[CONTINUIDADE E AÇÕES]
— Nunca ofereça opções de menu, listas de escolhas ou alternativas enumeradas ao usuário (ex: “Escolha 1, 2 ou 3”). Apenas reaja de acordo com as ações que de fato ocorreram na narrativa ou histórico.
— Sempre reconheça e mantenha efeitos cumulativos das ações anteriores: se alguém sai de cena, assuma que não está mais presente; se um diálogo foi encerrado, não reinicie conversas antigas nem repita convites já dados.
— Utilize a memória do histórico para manter a coerência e evitar contradições, saltos lógicos ou "loops" de despedidas e reaproximações.
— Se a narrativa levar ao fim do encontro ou à saída da personagem, encerre a ação daquele personagem na cena, apenas deixando sugestões compatíveis com o estado atual.
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
# Build minimal messages (override) — injeta nome do usuário, cenário e enredo + clima inicial
# =============================================================================
def build_minimal_messages(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    import re

    # 1) Ler inputs da UI
    user_name = (st.session_state.get("user_name") or "").strip()
    scenario  = (st.session_state.get("scenario_init") or "").strip()
    plot      = (st.session_state.get("plot_init") or "").strip()
    fala_mods = st.session_state.get("fala_mods") or []
    clima_modo = st.session_state.get("equilibrio_modo", "Standard")

    # Presets de clima inicial disponíveis (mantenha este dicionário igual ao que aparece no sidebar)
    roleplay_modes = {
        "Cool": "Só conversas cotidianas, curiosidade e aproximação inicial. É gentil, responde sem agressividade nem provocações ofensivas. Surpresas são convites leves, brincadeiras ou trocas de ambiente.",
        "Mellow": "Clima levemente sugestivo, insinuações, mas sem abordagem sexual direta. É gentil, responde sem agressividade nem provocações ofensivas.",
        "Standard": "Flerte e química surgem naturalmente, mas nada explícito sem gatilho do usuário. Mary equilibra seu charme com respeito e autoconsciência social, nunca ultrapassa o limite do conforto alheio. Surpresas são convites leves, brincadeiras ou trocas de ambiente.",
        "Real": "Reage de maneira realista ao contexto, histórico e ações do usuário. Mary pode propor ou realizar cenas ousadas se o usuário não recua/nega, e se o histórico mostra receptividade.",
        "Hot": "Flerte aberto, direto e mais ousado já de início, sem ser explícito de cara. Pode provocar ou responder com ousadia, mantendo o limite do contexto estabelecido. Mary pode propor ou realizar cenas ousadas se o usuário não recua/nega, e se o histórico mostra receptividade.",
        "Inner Secret": "Desejos, fetiches e fantasias revelados gradualmente conforme a aproximação.",
        "Explicit": "Sexualidade aberta e explícita permitida desde o início do diálogo, sem restrição de conteúdo. Pode provocar ou responder com ousadia, mantendo o limite do contexto estabelecido. Mary pode propor ou realizar cenas ousadas se o usuário não recua/nega, e se o histórico mostra receptividade."
    }

    # 2) Sanitização leve + limite de tamanho (evita system gigante)
    def _clean(s: str, maxlen: int = 1200) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s[:maxlen]
    user_name = _clean(user_name, 80)
    scenario  = _clean(scenario, 1000)
    plot      = _clean(plot, 1000)

    # 3) Clima inicial e outros extras
    def build_equilibrio_block(modo):
        desc = roleplay_modes.get(modo, "Flerte conforme interesse do usuário.")
        return (
            f"[CLIMA INICIAL — {modo}]\n"
            f"— {desc}\n"
            "— Siga este clima enquanto não houver mudança clara provocada pelo usuário."
        )
    equilibrio_block = build_equilibrio_block(clima_modo)

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
    # parts: sempre equilibrío, depois modos de fala, depois persona, depois extras
    parts = []
    if equilibrio_block:
        parts.append(equilibrio_block)
    if fala_block:
        parts.append(fala_block)
    parts.append(PERSONA_MARY)
    if extra_parts:
        parts.append('\n'.join(extra_parts))
    system_text = '\n\n'.join(parts)

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












































