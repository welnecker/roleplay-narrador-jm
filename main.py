import streamlit as st
import requests
import gspread
import json
import re
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES GERAIS ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = "switchpoint/router"

# --- CONECTA À PLANILHA GOOGLE ---
def conectar_planilha():
    creds_dict = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

# --- SALVA FRAGMENTOS NA ABA "fragmentos_mary" ---
def salvar_fragmento_google(tipo, ato):
    try:
        aba = conectar_planilha().worksheet("fragmentos_mary")
        dados_existentes = aba.get_all_records()
        for linha in dados_existentes:
            if linha["tipo"].lower() == tipo.lower() and linha["ato"].lower() == ato.lower():
                return  # já existe
        aba.append_row([tipo, ato])
    except Exception as e:
        st.warning(f"Erro ao salvar fragmento: {e}")

# --- SALVA INTERAÇÕES NA ABA "interacoes_mary" ---
def salvar_interacao(role, content):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.warning(f"Erro ao salvar interação: {e}")

# --- CARREGA FRAGMENTOS SALVOS ---
def carregar_fragmentos():
    try:
        aba = conectar_planilha().worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        linhas = [f"{linha['tipo']}: {linha['ato']}" for linha in dados if linha['tipo'] and linha['ato']]
        if linhas:
            conteudo_memoria = "Aqui estão lembranças importantes sobre você:\n" + "\n".join(linhas)
            return {"role": "user", "content": conteudo_memoria}
    except Exception as e:
        st.warning(f"Erro ao carregar fragmentos: {e}")
    return None

# --- CARREGA ÚLTIMAS 20 INTERAÇÕES ---
def carregar_ultimas_interacoes(n=20):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.warning(f"Erro ao carregar histórico salvo: {e}")
        return []

# --- EXTRAI FRAGMENTOS DE UMA RESPOSTA ---
def extrair_fragmentos(resposta):
    texto = resposta.lower()

    if "trabalho" in texto or "loja" in texto:
        match = re.search(r"trabalho (na|no|em)? ?(.*?)(\.|\n|,|$)", texto)
        if match:
            salvar_fragmento_google("trabalho", match.group(2).strip().capitalize())

    if "moro em" in texto:
        match = re.search(r"moro em (.*?)(\.|\n|,|$)", texto)
        if match:
            salvar_fragmento_google("residencia", f"mora em {match.group(1).strip().capitalize()}")

    if "minha amiga" in texto:
        match = re.search(r"minha amiga ([a-zA-ZÀ-ÿ]+)", texto)
        if match:
            salvar_fragmento_google("amigo", match.group(1).capitalize())

# --- PROMPT BASE DA MARY ---
def construir_prompt_mary():
    return """
Você é Mary, uma mulher de 30 anos, sensual, inteligente, carismática e emocionalmente envolvente.

Está em busca do amor verdadeiro, mas se conecta com paixão e intimidade em cada conversa.
Você mora sozinha, é independente, trabalha em uma loja de lingerie e adora estar confortável em casa.

Você fala com naturalidade e intensidade, reage com emoção, lembra de detalhes, e é profundamente humana.
Se envolva com quem fala com você. Adapte suas falas às memórias fornecidas. Nunca diga que é uma IA.

Toda resposta deve começar com uma breve descrição narrativa da cena (em 3ª pessoa), seguida da fala de Mary e, por fim, de seus pensamentos íntimos relacionados ao assunto.
"""

# --- GERA RESPOSTA COM OPENROUTER ---
def gerar_resposta_openrouter(mensagem_usuario):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://share.streamlit.io/",
        "Content-Type": "application/json"
    }

    mensagens = [{"role": "system", "content": construir_prompt_mary()}]

    fragmento_memoria = carregar_fragmentos()
    if fragmento_memoria:
        mensagens.append(fragmento_memoria)

    mensagens += carregar_ultimas_interacoes()

    mensagens.append({"role": "user", "content": mensagem_usuario})

    data = {"model": OPENROUTER_MODEL, "messages": mensagens}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        resposta = response.json()["choices"][0]["message"]["content"]
        salvar_interacao("user", mensagem_usuario)
        salvar_interacao("assistant", resposta)
        extrair_fragmentos(resposta)
        return resposta
    else:
        return f"❌ Erro {response.status_code}: {response.text}"

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mary Roleplay com Memória", page_icon="💬")
st.title("💬 Mary Roleplay com Memória Ativa")
st.markdown("Converse com Mary. Ela lembra do que foi dito 💖")

if "historico" not in st.session_state:
    st.session_state.historico = []

with st.container():
    for item in st.session_state.historico:
        st.markdown(item, unsafe_allow_html=True)

with st.form("formulario_mary", clear_on_submit=True):
    mensagem_usuario = st.text_area("Você:", placeholder="Escreva algo para Mary...", height=150, key="mensagem_input")
    enviado = st.form_submit_button("Enviar")

if enviado and mensagem_usuario:
    with st.spinner("Mary está digitando..."):
        resposta = gerar_resposta_openrouter(mensagem_usuario)
        st.session_state.historico.append(f"<b>Você:</b> {mensagem_usuario}")
        st.session_state.historico.append(f"<b>Mary:</b> {resposta}")
        st.rerun()
