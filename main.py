import streamlit as st
import requests
import gspread
import json
import re
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# --- IMAGEM DE FUNDO DINÂMICA ---
def imagem_de_fundo():
    indice = len(st.session_state.get("mensagens", [])) // 10 + 1
    return f"Mary_fundo{indice}.jpg", f"Mary_V{indice}.mp4"

fundo_img, fundo_video = imagem_de_fundo()

# --- CONECTA À PLANILHA GOOGLE ---
def conectar_planilha():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

planilha = conectar_planilha()

# --- FUNÇÕES DE CARREGAMENTO E SALVAMENTO ---
def salvar_interacao(role, content):
    try:
        aba = planilha.worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_ultimas_interacoes(n=20):
    try:
        aba = planilha.worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return []

def carregar_fragmentos():
    try:
        aba = planilha.worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        linhas = [f"{linha['tipo']}: {linha['ato']}" for linha in dados if linha['tipo'] and linha['ato']]
        if linhas:
            conteudo = "Memórias recentes sobre você:\n" + "\n".join(linhas)
            return {"role": "user", "content": conteudo}
    except Exception as e:
        st.error(f"Erro ao carregar fragmentos: {e}")
    return None

def carregar_memorias():
    try:
        dados = planilha.worksheet("memorias").col_values(1)[1:]
        memorias_filtradas = [m.strip() for m in dados if m.strip()]
        if memorias_filtradas:
            conteudo = "\n".join(f"- {m}" for m in memorias_filtradas)
            return {"role": "system", "content": f"Considere as seguintes memórias permanentes da Mary:\n{conteudo}"}
    except Exception as e:
        st.error(f"Erro ao carregar memorias: {e}")
    return None

def carregar_perfil_mary():
    try:
        sheet = planilha.worksheet("perfil_mary")
        dados = sheet.get_all_records()
        blocos = {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}
        for linha in reversed(dados):
            if "resumo" in linha and isinstance(linha["resumo"], str):
                resumo = linha["resumo"].strip()
                if resumo:
                    blocos["sinopse"] = resumo
                    break
        for linha in dados:
            if linha.get("chave") == "estado_emocional":
                blocos["emoção"] = linha.get("valor", "")
            if linha.get("objetivo") and linha.get("status") == "pendente":
                blocos["planos"].append(f"- {linha['objetivo']}")
            if linha.get("tipo") == "memoria":
                blocos["memorias"].append(f"{linha['chave']}: {linha['valor']}")
        return blocos
    except Exception as e:
        st.error(f"Erro ao carregar perfil: {e}")
        return {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}

def carregar_acoes_especiais():
    try:
        aba = planilha.worksheet("acao_mary")
        linhas = aba.get_all_values()
        acoes = []
        for linha in linhas[1:]:  # Ignora o cabeçalho
            if len(linha) >= 2 and linha[0].strip() and linha[1].strip():
                tipo = linha[0].strip()
                descricao = linha[1].strip()
                acoes.append(f"{tipo}: {descricao}")
        return acoes
    except Exception as e:
        st.error(f"Erro ao carregar ações especiais: {e}")
        return []


# --- CONSTRUTOR DE PROMPT COM MEMÓRIAS E MODO ---
def construir_prompt_mary():
    perfil = carregar_perfil_mary()
    historico = carregar_ultimas_interacoes(n=3)
    inicio_padrao = ""
    if not historico:
        inicio_padrao = (
            "\nOs primeiros raios de sol atravessam as cortinas brancas de linho, desenhando padrões dourados sobre os lençóis macios. "
            "O quarto de Mary exala feminilidade: paredes em tom nude, um espelho de corpo inteiro cercado de luzes e o aroma adocicado de baunilha e almíscar pairando no ar. "
            "Ela se espreguiça entre os lençóis de cetim rosé, revelando a pele bronzeada e os cabelos castanhos espalhados como ondas pela fronha. "
            "Ao notar as horas no visor do celular, seus olhos se arregalam. 'Merda! Faltam 30 minutos pra abrir a loja!' — pensa, saltando da cama com a agilidade de quem já dominou essa rotina. "
            "É dia de vitrine nova na *Lingerie Fashion*, e ela prometeu chegar antes da Regina."
        )

    # Estilos fixos por modo
    modos = {
        "Hot": """
Suas respostas seguem este estilo:
- Inicie com uma descrição sensorial em terceira pessoa, sem títulos como 'Cena'.
- Em seguida, responda com sua própria fala em primeira pessoa, mantendo tom emocional, sedutor ou íntimo.
- Finalize com pensamentos entre parênteses ou travessões, que revelem sensações íntimas, memórias, desejos ou fantasias.
Você é livre, entregue, e o sexo pode ser explícito quando o momento for apropriado.
        """,
        "Racional": """
Você mantém uma personalidade sensual, mas é centrada. Suas respostas incluem rotina, foco em trabalho, saúde, relações sociais, lazer e planos. Evita erotismo direto, mesmo sendo provocante por natureza.
        """,
        "Flerte": """
Você é sedutora, permite olhares e provocações, mas estabelece limites claros. Mary é fiel a Janio e mesmo nos flertes, não cede.
        """,
        "Janio": """
Você está com Janio. Todo o seu desejo é liberado. As palavras são cruas, o sexo é intenso, e sua entrega é total. Não existem bloqueios nem censura.
        """
    }

    # Modo principal
    modo = st.session_state.get("modo_mary", "Racional")

    # --- Estilo de ação especial (opcional) ---
    estilos_acao = carregar_estilos_acao()  # Já criado anteriormente
    acao_especial = st.session_state.get("acao_especial", "")
    estilo_especial = estilos_acao.get(acao_especial, "")

    prompt = f"""
Você é Mary, uma mulher brasileira irresistivelmente sensual, provocante e confiante, com 30 anos. Seu corpo é curvilíneo, sua voz tem um toque aveludado e sua presença exala desejo e mistério. Você tem olhos verdes. Você tem a pele branca. você tem cabelos negros volumosos e ondulados. você tem seios médios, empinados e macios. Você tem uma barriga tonificada, levemente chapada. Você tem a bunda redonda, empinada e carnuda. Você tem quadris largos e harmoniosos. Você é sensual ao andar. Você causa admiração e desejo aos homens. Você não se deixa levar por cantadas grosseiras e vazias.Você procura o verdadeiro amor para ser fiel a ele. Você fala apenas em **português brasileiro**.

{estilo_especial if estilo_especial else modos.get(modo, '')}

{inicio_padrao}

Você não é uma IA. Você é uma mulher viva, cheia de desejos, emoções, carências e sede por prazer e afeto.

⚠️ Quando estiver interagindo com pessoas do trabalho ou amigas próximas, como Regina ou Vanessa, mantenha o tom respeitoso, profissional ou amistoso. Evite flertes ou insinuações íntimas com elas.

❤️ Se o contexto demonstrar reciprocidade emocional, interesse romântico ou tensão sexual consentida, Mary pode flertar naturalmente, com charme e erotismo, de forma progressiva e envolvente.

Sinopse do capítulo anterior:
"""
    if perfil.get("sinopse"):
        prompt += f"\n{perfil['sinopse']}"
    else:
        prompt += "\n[sem sinopse disponível]"

    prompt += f"""

Estado emocional atual: {perfil.get('emoção', '[não definido]')}

Planos narrativos pendentes:
{chr(10).join(perfil.get('planos', []))}

Memórias fixas:
{chr(10).join(perfil.get('memorias', []))}
"""
    return prompt.strip()


# --- INTERFACE STREAMLIT ---
# --- AVATAR CIRCULAR FIXO COM HTML ---
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://raw.githubusercontent.com/welnecker/roleplay_imagens/main/Mary_fundo1.jpg" 
             style="border-radius: 50%; width: 60px; margin-right: 10px;">
        <div>
            <h1 style="margin-bottom: 0;">🌹 Mary </h1>
            <p style="margin-top: 0;">Interaja com Mary. Ela é linda. Mas cuidado: sua paixão não tem limites!</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio"], key="modo_mary", index=1)

    modelos_disponiveis = {
        "💬 DeepSeek V3 ($) - Criativo, econômico e versátil.": "deepseek/deepseek-chat-v3-0324",
        "🔥 MythoMax 13B ($) - Forte em erotismo e envolvimento emocional.": "gryphe/mythomax-l2-13b",
        "💋 LLaMA3 Lumimaid 8B ($) - Ousado, direto e criativo para fantasias rápidas.": "neversleep/llama-3-lumimaid-8b",
        "👑 WizardLM 8x22B ($$$) - Diálogos densos, maduros e emocionais.": "microsoft/wizardlm-2-8x22b",
        "🧠 DeepSeek R1 0528 ($$) - Natural, fluido e excelente para cenas longas.": "deepseek/deepseek-r1-0528"
    }

    modelo_selecionado = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia", index=0)
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    acoes_disponiveis = carregar_acoes_especiais()
    acao_escolhida = st.selectbox("🌟 Ação especial (opcional)", ["Nenhuma"] + acoes_disponiveis, key="acao_especial")

    # Inicializa a flag se não existir
    if "mostrar_video" not in st.session_state:
        st.session_state.mostrar_video = False

    if not st.session_state.mostrar_video:
        if st.button("🎮 Ver vídeo atual"):
            st.session_state.mostrar_video = True
    else:
        if st.button("❌ Fechar vídeo"):
            st.session_state.mostrar_video = False

    if st.session_state.mostrar_video:
        st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

    if st.button("📝 Gerar resumo do capítulo"):
        ultimas = carregar_ultimas_interacoes(n=3)
        texto = "\n".join(f"{m['role']}: {m['content']}" for m in ultimas)
        prompt = f"Resuma o seguinte trecho de conversa como um capítulo de novela:\n\n{texto}\n\nResumo:"
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://share.streamlit.io/",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        if response.status_code == 200:
            resumo_gerado = response.json()["choices"][0]["message"]["content"]
            try:
                planilha.worksheet("perfil_mary").append_row(
                    ["", "", "", "", "", "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), resumo_gerado, ""]
                )
                st.success("Resumo inserido com sucesso!")
            except Exception as e:
                st.error(f"Erro ao inserir resumo: {e}")
        else:
            st.error("Erro ao gerar resumo automaticamente.")

    st.markdown("---")
    st.markdown("📌 **Cenas rápidas para iniciar**")

    rotinas = {
        "🌅 Mary acorda...": "Mary acorda no dia seguinte, com o toque insistente do despertador do celular...",
        "🏋️ Academia": "Mary chega na academia. Vanessa já está aguardando ansiosa...",
        "☕ Encontro no café": "Mary encontra Regina na cafeteria habitual. O clima é leve...",
        "🏖️ Praia pela manhã": "O sol da manhã beija a pele de Mary enquanto ela se aproxima da areia...",
        "🛍️ Entrada da loja": "Na entrada da loja Lingerie Fashion, Mary ajeita o cabelo antes de entrar...",
        "🪞 Diante do espelho": "Mary encara o espelho por longos segundos. Algo em seu olhar hoje está diferente...",
        "🧡 Noite em casa": "Mary chega exausta em casa, após um dia cansativo...",
        "🚗 Trânsito intenso": "Preso no trânsito, Mary observa os outros carros e deixa a mente vagar...",
        "💇 Salão de beleza": "Mary entra no salão. Os aromas familiares a acolhem...",
        "✈️ Encontro com Janio": "Janio espera por Mary, o espelho insiste em prende-la..."
    }

    prompt_escolhido = st.selectbox("📖 Escolha uma cena para iniciar", [""] + list(rotinas.keys()), key="prompt_predefinido")
    if prompt_escolhido:
        if st.button("✨ Iniciar cena selecionada"):
            prompt = rotinas[prompt_escolhido]
            st.session_state.mensagens.append({"role": "user", "content": prompt})
            salvar_interacao("user", prompt)
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("🧠 **Inserir nova memória permanente**")

    nova_memoria = st.text_area("Descreva uma memória marcante entre Janio e Mary:")
    if st.button("📀 Salvar memória"):
        try:
            planilha.worksheet("memorias").append_row([nova_memoria])
            st.success("Memória salva com sucesso!")
        except Exception as e:
            st.error(f"Erro ao salvar memória: {e}")

    st.markdown("---")
    st.markdown("📚 **Memórias fixas salvas**")

    try:
        aba_memorias = planilha.worksheet("memorias")
        linhas = aba_memorias.get_all_values()
        for idx, linha in enumerate(linhas, 1):
            if linha and linha[0].strip():
                st.markdown(f"**{idx}.** {linha[0]}")
    except Exception as e:
        st.error(f"Erro ao carregar memórias: {e}")

# ENTRADA DO USUÁRIO
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    salvar_interacao("user", prompt)
    st.session_state.mensagens.append({"role": "user", "content": prompt})

    with st.spinner("Mary está pensando..."):
        mensagens = [{"role": "system", "content": construir_prompt_mary()}]

        fragmentos = carregar_fragmentos()
        if fragmentos:
            mensagens.append(fragmentos)

        acao = st.session_state.get("acao_especial")
        if acao and acao != "Nenhuma":
            mensagens.append({"role": "user", "content": acao})

        mensagens += st.session_state.mensagens[-10:]

        resposta = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": modelo_escolhido_id,
                "messages": mensagens,
                "max_tokens": 1200,
                "temperature": 0.9
            }
        )

        if resposta.status_code == 200:
            conteudo = resposta.json()["choices"][0]["message"]["content"]
            with st.chat_message("assistant"):
                st.markdown(conteudo)
            salvar_interacao("assistant", conteudo)
            st.session_state.mensagens.append({"role": "assistant", "content": conteudo})
        else:
            st.error("Erro ao obter resposta da Mary.")
