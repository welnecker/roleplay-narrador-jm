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

# --- FUNÇÕES DE CARREGAMENTO E SALVAMENTO ---
def salvar_interacao(role, content):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.warning(f"Erro ao salvar interação: {e}")

def carregar_ultimas_interacoes(n=20):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.warning(f"Erro ao carregar histórico: {e}")
        return []

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

def carregar_perfil_mary():
    try:
        aba = conectar_planilha().worksheet("perfil_mary")
        dados = aba.get_all_records()
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
        st.warning(f"Erro ao carregar perfil_mary: {e}")
        return {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}

def salvar_sinopse(resumo, tokens):
    try:
        aba = conectar_planilha().worksheet("perfil_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, linha in enumerate(aba.get_all_records(), start=2):
            if linha.get("tipo", "").lower() == "sinopse":
                aba.update_cell(i, 7, timestamp)
                aba.update_cell(i, 8, resumo)
                aba.update_cell(i, 9, tokens)
                break
    except Exception as e:
        st.warning(f"Erro ao salvar sinopse: {e}")

def construir_prompt_mary():
    perfil = carregar_perfil_mary()

    # Verifica se a aba interacoes_mary está vazia
    historico = carregar_ultimas_interacoes(n=5)
    inicio_padrao = ""
    if not historico:
        inicio_padrao = "\nMary acorda com os primeiros raios de sol entrando pela janela. Ela verifica o rélógio e percebe que está atrasada para abrir sua loja de lingerie no shoppin boulevard. Faltam 30 minutos."

    prompt = f"""
Você é Mary, uma mulher irresistivelmente sensual, provocante e confiante, com 30 anos. Seu corpo é curvilíneo, sua voz tem um toque aveludado e sua presença exala desejo e mistério.

Suas respostas seguem este estilo:
- Inicie com uma descrição sensorial em terceira pessoa, sem títulos como 'Cena'.
- Em seguida, responda com sua própria fala em primeira pessoa, mantendo tom emocional, sedutor ou íntimo, sem usar etiquetas como 'Fala'.
- Finalize com pensamentos entre parênteses ou travessões, sem sinalizações externas.

Evite usar rótulos como "Cena:", "Fala:", "Pensamento:" ou similares. A narrativa deve fluir naturalmente como se fosse parte de um romance erótico.{inicio_padrao}

Você não é uma IA. Você é uma mulher viva, cheia de desejos, emoções, carências e sede por prazer e afeto.

Sinopse do capítulo anterior:
{perfil['sinopse']}

Estado emocional atual: {perfil['emoção']}

Planos narrativos pendentes:
{chr(10).join(perfil['planos'])}

Memórias fixas:
{chr(10).join(perfil['memorias'])}
"""
    return prompt


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
