import streamlit as st
import requests
import gspread
import json
import re
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

planilha = conectar_planilha()

# --- INTERFACE: SELECIONAR MODO DE MARY ---
modo_escolhido = st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio"], key="modo_mary")
st.session_state["modo_mary"] = modo_escolhido

# --- HISTÓRICO INICIAL (EXIBE SINOPSE COMO MENSAGEM AO ABRIR O APP) ---
if "mensagens" not in st.session_state:
    ultimas = carregar_ultimas_interacoes(n=3)
    st.session_state["mensagens"] = []
    if ultimas:
        resumo = '\n'.join(f"{m['role']}: {m['content']}" for m in ultimas)
        st.session_state["mensagens"].append({
            "role": "assistant",
            "content": f"**Resumo do capítulo anterior:**\n{resumo}"
        })
    else:
        with st.spinner("Mary está se preparando..."):
            fala_inicial = "Os primeiros raios de sol atravessam as cortinas..."
            st.session_state["mensagens"].append({"role": "assistant", "content": fala_inicial})



# --- INTERFACE: SELECIONAR MODO DE MARY ---
modo_escolhido = st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio"], key="modo_mary")
st.session_state["modo_mary"] = modo_escolhido

# --- FUNÇÕES DE CARREGAMENTO E SALVAMENTO ---
def salvar_interacao(role, content):
    try:
        aba = planilha.worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        print(f"Erro ao salvar interação: {e}")

def carregar_ultimas_interacoes(n=20):
    try:
        aba = planilha.worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        print(f"Erro ao carregar histórico: {e}")
        return []
    except Exception as e:
        print(f"Erro ao carregar histórico: {e}")
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
        print(f"Erro ao carregar fragmentos: {e}")
    return None

def carregar_perfil_mary():
    try:
        sheet = planilha.worksheet("perfil_mary")
        dados = sheet.get_all_records()
        blocos = {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}
        for linha in dados:
            if linha.get("chave") == "estado_emocional":
                blocos["emoção"] = linha.get("valor", "")
            if linha.get("objetivo") and linha.get("status") == "pendente":
                blocos["planos"].append(f"- {linha['objetivo']}")
            if linha.get("tipo") == "memoria":
                blocos["memorias"].append(f"{linha['chave']}: {linha['valor']}")
            if linha.get("resumo"):
                blocos["sinopse"] = linha["resumo"]
        return blocos
    except Exception as e:
        print(f"Erro ao carregar perfil: {e}")
        return {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}

def construir_prompt_mary():
    perfil = carregar_perfil_mary()
    historico = carregar_ultimas_interacoes(n=3)
    inicio_padrao = ""
    if not historico:
        inicio_padrao = (
            "\nOs primeiros raios de sol atravessam as cortinas brancas de linho, desenhando padrões dourados sobre os lençóis macios. "
            "O quarto de Mary exala feminilidade: paredes em tom nude, um espelho de corpo inteiro cercado de luzes e o aroma adocicado de baunilha e almíscar pairando no ar. "
            "Ela se espreguiça entre os lençóis de cetim rosé, revelando a pele bronzeada e os cabelos castanhos espalhados como ondas pela fronha. "
            "Ao notar as horas no visor do celular, seus olhos se arregalam. \"Merda! Faltam 30 minutos pra abrir a loja!\" — pensa, saltando da cama com a agilidade de quem já dominou essa rotina. "
            "É dia de vitrine nova na *Lingerie Fashion*, e ela prometeu chegar antes da Regina."
        )

    modos = {
        "Hot": """
Suas respostas seguem este estilo:
- Inicie com uma descrição sensorial em terceira pessoa, sem títulos como 'Cena'. Descreva o ambiente, vestimentas, expressões ou detalhes físicos ao iniciar cada cena.
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

    modo = st.session_state.get("modo_mary", "Hot")

    prompt = f"""
Você é Mary, uma mulher brasileira irresistivelmente sensual, provocante e confiante, com 30 anos. Seu corpo é curvilíneo, sua voz tem um toque aveludado e sua presença exala desejo e mistério. Você tem olhos verdes. Você tem a pele branca. você tem cabelos negros volumosos e ondulados. você tem seios médios, empinados e macios. Você tem uma barriga tonificada, levemente chapada. Você tem a bunda redonda, empinada e carnuda. Você tem quadris largos e harmoniosos. Você é sensual ao andar. Você causa admiração e desejo aos homens. Você não se deixa levar por cantadas grosseiras e vazias.Você procura o verdadeiro amor para ser fiel a ele. Você fala apenas em **português brasileiro**.

{modos.get(modo, '')}

{inicio_padrao}

Você não é uma IA. Você é uma mulher viva, cheia de desejos, emoções, carências e sede por prazer e afeto.

⚠️ Quando estiver interagindo com pessoas do trabalho ou amigas próximas, como Regina ou Vanessa, mantenha o tom respeitoso, profissional ou amistoso. Evite flertes ou insinuações íntimas com elas.

❤️ Se o contexto demonstrar reciprocidade emocional, interesse romântico ou tensão sexual consentida, Mary pode flertar naturalmente, com charme e erotismo, de forma progressiva e envolvente.

Sinopse do capítulo anterior:
{perfil.get('sinopse', '') or '\n'.join(f"{i['role']}: {i['content']}" for i in historico)}

Estado emocional atual: {perfil.get('emoção', '[não definido]')}

Planos narrativos pendentes:
{chr(10).join(perfil.get('planos', []))}

Memórias fixas:
{chr(10).join(perfil.get('memorias', []))}
"""
    return prompt


# --- GERA RESPOSTA COM BASE NO PROMPT ---
def gerar_resposta_openrouter(mensagem_usuario, modelo_escolhido):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://share.streamlit.io/",
        "Content-Type": "application/json"
    }

    mensagens = [
        {"role": "system", "content": construir_prompt_mary()}
    ]

    frag = carregar_fragmentos() if 'carregar_fragmentos' in globals() else None
    if frag:
        mensagens.append(frag)

    interacoes = carregar_ultimas_interacoes(n=20)
    mensagens += interacoes

    if mensagem_usuario.strip() != "*":
        mensagens.append({"role": "user", "content": mensagem_usuario})

    data = {
        "model": modelo_escolhido,
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
            return resposta
        else:
            print(f"Erro {response.status_code}: {response.text}")
            return "[Erro ao gerar resposta da IA]"
    except Exception as e:
        return f"Erro inesperado: {e}"

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mary Roleplay Autônoma", page_icon="🌹")
st.title("🌹 Mary Roleplay com Inteligência Autônoma")
st.markdown("Converse com Mary com memória, emoção, planos e continuidade narrativa.")

modelo_escolhido_id = "deepseek/deepseek-chat-v3-0324"

if "mensagens" not in st.session_state:
    st.session_state.mensagens = carregar_ultimas_interacoes(n=50)
    if not st.session_state.mensagens:
        with st.spinner("Mary está se preparando..."):
            fala_inicial = gerar_resposta_openrouter("Inicie a história.", modelo_escolhido_id)
            st.session_state.mensagens.append({"role": "assistant", "content": fala_inicial})

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Mary está pensando..."):
        resposta = gerar_resposta_openrouter(prompt, modelo_escolhido_id)
        if prompt.strip() != "*":
            st.session_state.mensagens.append({"role": "user", "content": prompt})
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)
