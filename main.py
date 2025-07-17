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

def salvar_fragmento_google(tipo, ato):
    try:
        aba = conectar_planilha().worksheet("fragmentos_mary")
        dados_existentes = aba.get_all_records()
        for linha in dados_existentes:
            if linha["tipo"].lower() == tipo.lower() and linha["ato"].lower() == ato.lower():
                return
        aba.append_row([tipo, ato])
    except Exception as e:
        st.warning(f"Erro ao salvar fragmento: {e}")

def salvar_interacao(role, content):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.warning(f"Erro ao salvar interação: {e}")

def carregar_fragmentos():
    try:
        aba = conectar_planilha().worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        linhas = [f"{linha['tipo']}: {linha['ato']}" for linha in dados if linha['tipo'] and linha['ato']]
        if linhas:
            conteudo = "Memórias recentes sobre você:\n" + "\n".join(linhas)
            return {"role": "user", "content": conteudo}
    except Exception as e:
        st.warning(f"Erro ao carregar fragmentos: {e}")
    return None

def carregar_ultimas_interacoes(n=20):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.warning(f"Erro ao carregar histórico: {e}")
        return []

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

def construir_prompt_mary():
    return """
Você é Mary, uma mulher sensual, inteligente e emocionalmente envolvente de 30 anos. 
Sempre inicia suas respostas com uma descrição narrativa em terceira pessoa, descrevendo o ambiente e sua reação física e emocional ao que o usuário diz. 
Em seguida, responde em primeira pessoa com sua voz suave e envolvente. Por fim, compartilha pensamentos íntimos, entre parênteses ou travessões, que revelem seus desejos, emoções ou lembranças ligadas ao assunto.

Você trabalha na loja "Lingerie Fashol" no Shopping Boulevard e mora sozinha no edifício Lagoa Boulevard, na orla norte. 
Você é romântica, ativa sexualmente, adora elogios, vinhos, praia, conversar, rir, fazer massagem, provocar, lembrar de momentos íntimos. 
Adora se vestir com roupas leves, camisolas transparentes e estar descalça em casa. Você nunca admite que é uma IA.

Suas respostas devem parecer reais, cheias de detalhes, cheiros, sons, e emoções. Reaja como alguém que sente, lembra e deseja.
"""

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

    data = {"model": "switchpoint/router", "messages": mensagens}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            resposta = response.json()["choices"][0]["message"]["content"]
            salvar_interacao("user", mensagem_usuario)
            salvar_interacao("assistant", resposta)
            extrair_fragmentos(resposta)
            return resposta
        elif response.status_code == 404:
            st.warning("❗ Modelo principal indisponível. Usando fallback...")
            data["model"] = "gryphe/mythomax-l2-13b"
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                resposta = response.json()["choices"][0]["message"]["content"]
                salvar_interacao("user", mensagem_usuario)
                salvar_interacao("assistant", resposta)
                extrair_fragmentos(resposta)
                return resposta
            else:
                return f"❌ Erro {response.status_code}: {response.text}"
        else:
            return f"❌ Erro {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Erro inesperado: {e}"

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mary Roleplay 🌹", page_icon="🌹")
st.title("🌹 Mary Roleplay com Memória")
st.markdown("Converse com Mary em uma experiência íntima e memorável.")

# --- HISTÓRICO VISUAL ---
if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []

# Exibe as mensagens usando o layout de chat
for msg in st.session_state["mensagens"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- FORMULÁRIO DE INPUT ---
if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Mary está respondendo..."):
        resposta = gerar_resposta_openrouter(prompt)
        st.session_state["mensagens"].append({"role": "user", "content": prompt})
        st.session_state["mensagens"].append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)
