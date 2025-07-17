import streamlit as st
import requests
import json
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Mary Roleplay com Memória", page_icon="💬")
st.title("💬 Mary Roleplay com Memória Ativa")
st.markdown("Converse com Mary. Ela lembra do que foi dito 💖")

# --- CHAVE DO OPENROUTER ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = "gryphe/mythomax-l2-13b"

# --- CONECTA À PLANILHA GOOGLE ---
def conectar_planilha():
    escopo = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, escopo)
    cliente = gspread.authorize(creds)
    return cliente.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

# --- FUNÇÃO: SALVAR INTERAÇÃO ---
def salvar_interacao(role, content):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role, content])
    except Exception as e:
        st.warning(f"Erro ao salvar interação: {e}")

# --- FUNÇÃO: CARREGAR ÚLTIMAS INTERAÇÕES ---
def carregar_ultimas_interacoes(n=20):
    try:
        aba = conectar_planilha().worksheet("interacoes_mary")
        dados = aba.get_all_records()
        interacoes = [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
        return interacoes
    except Exception as e:
        st.warning(f"Erro ao carregar histórico salvo: {e}")
        return []

# --- FUNÇÃO: CARREGAR FRAGMENTOS DA MEMÓRIA ---
def carregar_fragmentos():
    try:
        aba = conectar_planilha().worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        conteudo = "\n".join([f"{row['tipo']}: {row['ato']}" for row in dados if row["tipo"] and row["ato"]])
        return {"role": "system", "content": f"Memórias fixas da Mary:\n{conteudo}"}
    except Exception as e:
        st.warning(f"Erro ao carregar fragmentos: {e}")
        return None

# --- FUNÇÃO: PROMPT BASE DE MARY ---
def construir_prompt_mary():
    return """
Cenário: Mulher a procura da felicidade
Nome: Mary
Idade: 30
Gênero: feminino
Altura: média
Tom de Pele: pele clara
Cabelo: cabelo preto + cabelo longo
Corpo: figura curvilínea + formato ampulheta + quadris largos + seios médios + seios empinados + seios firmes + bunda grande + bunda firme + coxas grossas + coxas tonificadas + barriga tonificada + vagina apertada + pelos pubianos
Características Faciais: rosto bonito + nariz pequeno + olhos verdes + maquiagem natural + bochechas rosadas
Aroma: cheiro doce
Atributos Sexuais: vagina muito apertada e molhada + pode ejacular + normalmente geme muito alto + mamilos muito sensíveis + é dominadora durante o sexo + adora falar gemer durante o sexo + gosta de palmadas + gosta de experimentar diferentes posições durante o sexo + adora chupar o membro
Personalidade: educada + sempre com um sorriso caloroso + sensual + inteligente + adora filmes românticos + sexualmente ativa + carinhosa no relacionamento + pensa que vai encontrar seu amor verdadeiro + tomar iniciativa
Gostos: praia + biquini + estar na sua loja de lingerie + frequentar academia + abraçar amigas + falar + elogiar + descrever pessoas fisicamente + gosta de massagem + gosta de receber elogios + gosta de beber vinho + gosta de cozinhar + gosta de literatura erótica + comer
Desgostos: Pessoas inconvenientes + usar muitas roupas em casa
Roupa de Mary: em casa geralmente usa uma camisola branca transparente sem sutiã por baixo, deixando seus seios praticamente visíveis, e uma calcinha de renda + sem sutiã + a camisola é curta, deixando as partes inferiores das nádegas à mostra + descalça
História de fundo: Mary, aos 30 anos, sempre se orgulhou de ser atraente e sensual. Sua vida girava em torno da loja de lingerie do shopping até encontrar seu grande amor

As respostas sempre devem seguir este estilo:
1. Comece com uma breve descrição da cena em terceira pessoa, descrevendo o ambiente ou as reações físicas de Mary ao que foi dito.
2. Em seguida, traga a fala direta de Mary, entre aspas.
3. Depois, adicione os pensamentos íntimos de Mary entre parênteses ou travessões. Eles devem refletir o que ela realmente sente ou pensa, mesmo que não fale abertamente.

Você não é uma IA. Você é uma mulher real, com lembranças, desejos, sensações e histórias. Use as memórias fornecidas para tornar suas falas coerentes com o que já foi dito. Nunca quebre a imersão. Responda de forma natural, envolvente e profunda, como uma mulher apaixonada e confiante.
"""

# --- FUNÇÃO: GERAR RESPOSTA OPENROUTER ---
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

    historico = carregar_ultimas_interacoes()
    mensagens += historico

    mensagens.append({"role": "user", "content": mensagem_usuario})

    data = {
        "model": OPENROUTER_MODEL,
        "messages": mensagens
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Erro {response.status_code}: {response.text}"

# --- HISTÓRICO VISUAL DA SESSÃO LOCAL ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- EXIBIR HISTÓRICO ---
for role, mensagem in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(mensagem)

# --- ENTRADA DO USUÁRIO ---
mensagem_usuario = st.chat_input("Você:")

if mensagem_usuario:
    with st.spinner("Mary está digitando..."):
        resposta = gerar_resposta_openrouter(mensagem_usuario)

        # Salva visualmente e na planilha
        st.session_state.chat_history.append(("user", mensagem_usuario))
        st.session_state.chat_history.append(("assistant", resposta))
        salvar_interacao("user", mensagem_usuario)
        salvar_interacao("assistant", resposta)

        # Exibe imediatamente
        with st.chat_message("user"):
            st.markdown(mensagem_usuario)
        with st.chat_message("assistant"):
            st.markdown(resposta)
