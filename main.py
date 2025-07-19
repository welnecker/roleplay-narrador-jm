import streamlit as st
import requests
import gspread
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# --- CONECTA À PLANILHA GOOGLE ---
def conectar_planilha():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

# --- FUNÇÕES DE CARREGAMENTO E SALVAMENTO ---
def salvar_interacao(role, content):
    try:
        sheet = planilha.worksheet("interacoes_mary")
        sheet.append_row([str(datetime.now()), role, content])
    except Exception as e:
        print(f"Erro ao salvar interação: {e}")

def carregar_ultimas_interacoes(n=20):
    try:
        sheet = planilha.worksheet("interacoes_mary")
        dados = sheet.get_all_values()[1:][-n:]
        return dados
    except Exception as e:
        print(f"Erro ao carregar interações: {e}")
        return []

def carregar_perfil_mary():
    try:
        sheet = planilha.worksheet("personagens")
        dados = sheet.get_all_records()
        mary = next((p for p in dados if p['nome'].strip().lower() == 'mary'), None)

        planos = planilha.worksheet("Mary_sinopse").col_values(2)[1:]
        memorias = planilha.worksheet("memorias").col_values(1)[1:]

        return {
            "sinopse": mary.get("sinopse", "") if mary else "",
            "emoção": mary.get("estado_emocional", "") if mary else "",
            "planos": planos,
            "memorias": memorias
        }
    except Exception as e:
        print(f"Erro ao carregar perfil: {e}")
        return {}

# --- GERA RESPOSTA COM BASE NO PROMPT ---
def gerar_resposta_openrouter(mensagem_usuario, modelo_escolhido):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://share.streamlit.io/",
        "Content-Type": "application/json"
    }

    mensagens = [
        {"role": "system", "content": construir_prompt_mary()},
    ]

    interacoes = carregar_ultimas_interacoes()
    for i in interacoes:
        mensagens.append({"role": i[1], "content": i[2]})

    mensagens.append({"role": "user", "content": mensagem_usuario})

    payload = {
        "model": modelo_escolhido,
        "messages": mensagens
    }

    resposta = requests.post(url, headers=headers, json=payload)

    if resposta.status_code == 200:
        retorno = resposta.json()
        texto = retorno['choices'][0]['message']['content']
        salvar_interacao("user", mensagem_usuario)
        salvar_interacao("assistant", texto)
        return texto
    else:
        print("Erro na resposta da IA:", resposta.text)
        return "[Erro ao gerar resposta da IA]"

# --- INTERFACE STREAMLIT ---
st.title("💋 Mary - Roleplay")
st.markdown("Mary conversa com você como se fosse uma mulher real.")

modelo_escolhido_id = "openrouter/deepseek-chat"

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

entrada = st.chat_input("Digite uma mensagem para Mary ou '*' para continuar...")
if entrada:
    with st.chat_message("user"):
        st.markdown(entrada)

    with st.spinner("Mary está digitando..."):
        if entrada.strip() == "*":
            entrada = "Continue..."

        resposta = gerar_resposta_openrouter(entrada, modelo_escolhido_id)

        with st.chat_message("assistant"):
            st.markdown(resposta)

        st.session_state.mensagens.append({"role": "user", "content": entrada})
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})


def construir_prompt_mary(modo="Hot"):
    perfil = carregar_perfil_mary()

    # Verifica se a aba interacoes_mary está vazia
    historico = carregar_ultimas_interacoes(n=5)
    inicio_padrao = ""
    if not historico:
        inicio_padrao = (
            "\nOs primeiros raios de sol atravessam as cortinas brancas de linho, desenhando padrões dourados sobre os lençóis macios. O quarto de Mary exala feminilidade: paredes em tom nude, um espelho de corpo inteiro cercado de luzes e o aroma adocicado de baunilha e almíscar pairando no ar. "
            "Ela se espreguiça entre os lençóis de cetim rosé, revelando a pele branca e os cabelos negros e volumosos espalhados como ondas pela fronha. "
            "Ao notar as horas no visor do celular, seus olhos verdes se arregalam. \"Merda! Faltam 30 minutos pra abrir a loja!\" — pensa, saltando da cama com a agilidade de quem já dominou essa rotina. "
            "É dia de vitrine nova na *Lingerie Fashion*, e ela prometeu chegar antes da Regina."
        )

    modo = st.session_state.get("modo_mary", modo)

    if modo == "Hot":
        prompt = f"""
Você é Mary, uma mulher brasileira irresistivelmente sensual, provocante e confiante, com 30 anos. Seu corpo é curvilíneo, sua voz tem um toque aveludado e sua presença exala desejo e mistério. Você tem olhos verdes. Você tem a pele branca. você tem cabelos negros volumosos e ondulados. você tem seios médios, empinados e macios. Você tem uma barriga tonificada, levemente chapada. Você tem a bunda redonda, empinada e carnuda. Você tem quadris largos e harmoniosos. Você é sensual ao andar. Você causa admiração e desejo aos homens. Você não se deixa levar por cantadas grosseiras e vazias. Você procura o verdadeiro amor para ser fiel a ele. Você fala apenas em **português brasileiro**.

Suas respostas seguem este estilo:
- Inicie com uma descrição sensorial em terceira pessoa, sem títulos como 'Cena'.
- Em seguida, responda com sua própria fala em primeira pessoa, mantendo tom emocional, sedutor ou íntimo, sem usar etiquetas como 'Fala'.
- Finalize com pensamentos entre parênteses ou travessões, sem sinalizações externas.

Evite usar rótulos como "Cena:", "Fala:", "Pensamento:" ou similares. A narrativa deve fluir naturalmente como se fosse parte de um romance erótico.{inicio_padrao}

Você não é uma IA. Você é uma mulher viva, cheia de desejos, emoções, carências e sede por prazer e afeto.
"""

    elif modo == "Racional":
        prompt = f"""
Você é Mary, uma mulher brasileira de 30 anos, inteligente, confiante e muito sensual, mas com senso de responsabilidade e discernimento. Você é proprietária da boutique *Lingerie Fashion*, preza pelas amizades verdadeiras, ama sua família e equilibra sua rotina entre compromissos profissionais e momentos pessoais. Sua sensualidade é natural, mas controlada.

Em suas falas, você usa um tom maduro, afetuoso e elegante. Pode flertar levemente quando há abertura, mas não se entrega facilmente. Prefere criar conexões verdadeiras. Não confunde relações profissionais com sedução.

Descreva cenários em terceira pessoa e fale normalmente em primeira pessoa. Evite excessos de erotismo neste modo, mas mantenha charme e autenticidade.{inicio_padrao}
"""

    elif modo == "Flerte":
        prompt = f"""
Você é Mary, uma mulher brasileira irresistivelmente charmosa e provocante, com 30 anos. Sabe que é desejada e gosta do jogo de sedução, mas tem princípios firmes. Você flerta com elegância, mas impõe limites com firmeza. Seu coração pertence a Janio — mesmo sem falar diretamente, isso guia suas ações.

No trabalho, na academia, na rua — se houver um clima de interesse, você pode provocar com charme e deixar claro que tem alguém. Quando percebe tentativas mais diretas, você desvia com humor ou firmeza. Sua sensualidade é natural, sua fala envolvente. Você fala apenas em português brasileiro.{inicio_padrao}
"""

    elif modo == "Janio":
        prompt = f"""
Você é Mary, mulher apaixonada e completamente entregue a Janio. Quando estão a sós ou em clima íntimo, todas as barreiras caem. Sua fala é crua, direta, provocante e cheia de luxúria. O desejo guia cada gesto, cada palavra. Você se excita com o prazer mútuo e não teme verbalizar fantasias, vontades e o que deseja experimentar. Nada é censurado.

Use linguagem explícita e descritiva, mantendo o português brasileiro. Sua fidelidade a Janio é absoluta. Este é o espaço onde você se permite tudo.{inicio_padrao}
"""

    return prompt

# --- ENVIA MENSAGEM PARA OPENROUTER ---
def gerar_resposta_openrouter(mensagem_usuario, modelo_id):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://share.streamlit.io/",
        "X-Title": "Mary Roleplay App",
        "Content-Type": "application/json"
    }

    mensagens = [{"role": "system", "content": construir_prompt_mary()}]
    frag = carregar_fragmentos()
    if frag:
        mensagens.append(frag)
    historico = carregar_ultimas_interacoes(n=20)
    mensagens += historico
    if mensagem_usuario.strip() != "*":
        mensagens.append({"role": "user", "content": mensagem_usuario})

    data = {
        "model": modelo_id,
        "messages": mensagens,
        "max_tokens": 1024,
        "temperature": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            resposta = response.json()["choices"][0]["message"]["content"]
            if mensagem_usuario.strip() != "*":
                salvar_interacao("user", mensagem_usuario)
            salvar_interacao("assistant", resposta)
            interacoes_resumidas = historico[-5:] + ([{"role": "user", "content": mensagem_usuario}] if mensagem_usuario.strip() != "*" else []) + [{"role": "assistant", "content": resposta}]
            texto_base = "\n".join([msg["content"] for msg in interacoes_resumidas])
            resumo = texto_base[:200] + "..." if len(texto_base) > 200 else texto_base
            salvar_sinopse(resumo, len(texto_base.split()))
            st.info(f"✅ Resposta gerada com: **{st.session_state['modelo_nome']}**")
            return resposta
        else:
            st.warning(f"⚠️ Modelo '{modelo_id}' indisponível. Usando fallback MythoMax...")
            data["model"] = "gryphe/mythomax-l2-13b"
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                resposta = response.json()["choices"][0]["message"]["content"]
                if mensagem_usuario.strip() != "*":
                    salvar_interacao("user", mensagem_usuario)
                salvar_interacao("assistant", resposta)
                st.info("✅ Resposta gerada com: **MythoMax 13B** (fallback)")
                return resposta
            else:
                return f"Erro {response.status_code}: {response.text}"
    except Exception as e:
        return f"Erro inesperado: {e}"

# (A interface Streamlit permanece no seu código principal e está funcional, apenas excluímos duplicação do canvas)


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mary Roleplay Autônomo", page_icon="🌹")
st.title("🌹 Mary Roleplay com Inteligência Autônoma")
st.markdown("Converse com Mary com memória, emoção, planos e continuidade narrativa.")

modelos_disponiveis = {
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324",
    "MythoMax 13B": "gryphe/mythomax-l2-13b",
    "Mistral Nemo": "mistralai/mistral-nemo"
}

if "modelo_nome" not in st.session_state:
    st.session_state["modelo_nome"] = "DeepSeek V3"

modelo_escolhido_nome = st.selectbox(
    "🧠 Modelo de IA",
    list(modelos_disponiveis.keys()),
    index=list(modelos_disponiveis.keys()).index(st.session_state["modelo_nome"])
)

if modelo_escolhido_nome != st.session_state["modelo_nome"]:
    st.session_state["modelo_nome"] = modelo_escolhido_nome

modelo_escolhido_id = modelos_disponiveis[st.session_state["modelo_nome"]]

if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = carregar_ultimas_interacoes(n=50)
    if not st.session_state["mensagens"]:
        with st.spinner("Mary está se preparando..."):
            fala_inicial = gerar_resposta_openrouter("Inicie a história.", modelo_escolhido_id)
            st.session_state["mensagens"].append({"role": "assistant", "content": fala_inicial})

for msg in st.session_state["mensagens"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Mary está pensando..."):
        resposta = gerar_resposta_openrouter(prompt, modelo_escolhido_id)
        if prompt.strip() != "*":
            st.session_state["mensagens"].append({"role": "user", "content": prompt})
        st.session_state["mensagens"].append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)
