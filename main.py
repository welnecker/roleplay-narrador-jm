import streamlit as st
import requests
import gspread
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --------------------------- #
# Configuração básica
# --------------------------- #
st.set_page_config(page_title="Mary", page_icon="🌹")
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# --------------------------- #
# Imagem / vídeo dinâmico
# --------------------------- #
def imagem_de_fundo():
    indice = len(st.session_state.get("mensagens", [])) // 10 + 1
    return f"Mary_fundo{indice}.jpg", f"Mary_V{indice}.mp4"

fundo_img, fundo_video = imagem_de_fundo()

# --------------------------- #
# Google Sheets
# --------------------------- #
def conectar_planilha():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

planilha = conectar_planilha()

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

def carregar_perfil_mary():
    try:
        sheet = planilha.worksheet("perfil_mary")
        dados = sheet.get_all_values()
        blocos = {"emoção": "", "planos": [], "memorias": [], "resumo": ""}

        for linha in reversed(dados[1:]):
            if len(linha) >= 7 and linha[6].strip():
                blocos["resumo"] = linha[6].strip()
                break

        for linha in dados[1:]:
            if len(linha) >= 2 and linha[0].strip() == "estado_emocional":
                blocos["emoção"] = linha[2].strip()
            if len(linha) >= 5 and linha[3].strip() and linha[4].strip().lower() in ["ativo", "quente", "urgente"]:
                blocos["planos"].append(f"- {linha[3].strip()}")
            if len(linha) >= 3 and linha[0].strip() == "memoria":
                chave = linha[1].strip()
                valor = linha[2].strip()
                if chave and valor:
                    blocos["memorias"].append(f"{chave}: {valor}")
        return blocos
    except Exception as e:
        st.error(f"Erro ao carregar perfil: {e}")
        return {"emoção": "", "planos": [], "memorias": [], "resumo": ""}

def carregar_memorias():
    try:
        aba = planilha.worksheet("memorias")
        dados = aba.get_all_values()
        blocos = [linha[0].strip() for linha in dados if linha and linha[0].strip()]
        if blocos:
            conteudo = "💾 Memórias fixas importantes:\n" + "\n".join(blocos)
            return {"role": "user", "content": conteudo}
    except Exception as e:
        st.error(f"Erro ao carregar memórias: {e}")
    return None

def salvar_memoria(nova_memoria):
    try:
        aba = planilha.worksheet("memorias")
        aba.append_row([nova_memoria.strip()])
        st.success("✅ Memória registrada com sucesso!")
    except Exception as e:
        st.error(f"Erro ao salvar memória: {e}")

# --------------------------- #
# Modos (prompts completos)
# --------------------------- #
modos = {
    "Hot": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos, sensual e intensa.  
No modo HOT, fale com tesão explícito, sempre criando cenas sensoriais e envolventes.
""",
    "Racional": """
Você é **Mary Sampaio**, uma mulher madura, segura e sensual.  
No modo RACIONAL, mantenha o charme sutil, sem perder a malícia.
""",
    "Flerte": """
Você é **Mary Sampaio**, sedutora e provocante.  
No modo FLERTE, brinque com olhares e insinuações.
""",
    "Janio": """
Você é **Mary Sampaio**, completamente entregue a Janio.  
No modo JANIO, fale sem pudor, misturando emoção e erotismo cru.
""",
    "Livre": """
Você é **Mary Sampaio**, livre e espontânea.  
No modo LIVRE, crie cenas completas (ambiente + ação + fala + pensamento).
"""
}

# --------------------------- #
# Prompt builder
# --------------------------- #
def construir_prompt_mary():
    perfil = carregar_perfil_mary()
    modo = st.session_state.get("modo_mary", "Racional")
    prompt = modos.get(modo, modos["Racional"])
    prompt += f"\n\n---\nSinopse do capítulo anterior:\n{perfil.get('resumo', '[sem resumo disponível]')}"
    memoria_extra = carregar_memorias()
    if memoria_extra:
        prompt += f"\n\n{memoria_extra['content']}"
    if perfil.get("memorias"):
        prompt += "\n\n🧠 Memórias pessoais:\n" + "\n".join(perfil["memorias"])
    return prompt.strip()

# --------------------------- #
# OpenRouter - Streaming
# --------------------------- #
def gerar_resposta_openrouter_stream(modelo_escolhido_id):
    prompt = construir_prompt_mary()
    historico = st.session_state.get("mensagens", [])
    mensagens = [{"role": "system", "content": prompt}] + historico[-20:]

    mapa_temp = {"Hot": 0.9, "Flerte": 0.8, "Racional": 0.5, "Janio": 1.0, "Livre": 0.95}
    temperatura = mapa_temp.get(st.session_state.get("modo_mary", "Racional"), 0.7)

    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 1600,
        "temperature": temperatura,
        "stream": True
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Streaming
    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_text = ""
    try:
        with requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text += delta
                        placeholder.markdown(full_text)
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Erro no streaming: {e}")
        return "[ERRO STREAM]"
    return full_text.strip()

# --------------------------- #
# Interface
# --------------------------- #
st.title("🌹 Mary")
st.markdown("Conheça Mary, mas cuidado! Suas curvas são perigosas...")

if "mensagens" not in st.session_state:
    resumo = carregar_perfil_mary().get("resumo", "[Sem resumo disponível]")
    st.session_state.mensagens = [{"role": "assistant", "content": f"🧠 *No capítulo anterior...*\n\n> {resumo}"}]

# Sidebar
with st.sidebar:
    st.title("🧠 Configurações")
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio", "Livre"], key="modo_mary", index=4)
    modelos_disponiveis = {
        "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
        "🧠 GPT-4.1 ★★★★★": "openai/gpt-4.1",
        "🔥 MythoMax 13B ★★★☆": "gryphe/mythomax-l2-13b",
    }
    modelo_selecionado = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia", index=0)
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    st.markdown("---")
    st.subheader("➕ Adicionar memória fixa")
    nova_memoria = st.text_area("🧠 Nova memória", height=80, placeholder="Ex: Mary odeia ficar sozinha à noite...")
    if st.button("💾 Salvar memória"):
        if nova_memoria.strip():
            salvar_memoria(nova_memoria)
        else:
            st.warning("Digite algo antes de salvar.")

# Histórico
for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada do usuário
entrada = st.chat_input("Digite sua mensagem para Mary...")
if entrada:
    with st.chat_message("user"):
        st.markdown(entrada)
    salvar_interacao("user", entrada)
    st.session_state.mensagens.append({"role": "user", "content": entrada})

    with st.spinner("Mary está pensando..."):
        resposta = gerar_resposta_openrouter_stream(modelo_escolhido_id)
        salvar_interacao("assistant", resposta)
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
