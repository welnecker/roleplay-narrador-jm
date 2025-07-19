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

# --- FUNÇÕES AUXILIARES ---
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
        aba = conectar_planilha().worksheet("perfil_mary")
        dados = aba.get_all_records()
        linhas = [f"{linha['tipo']}: {linha['ato']}" for linha in dados if linha['tipo'] == "fragmento"]
        if linhas:
            conteudo = "Memórias recentes sobre você:\n" + "\n".join(linhas)
            return {"role": "user", "content": conteudo}
    except Exception as e:
        st.warning(f"Erro ao carregar fragmentos: {e}")
    return None

def carregar_planos_pendentes():
    try:
        aba = conectar_planilha().worksheet("perfil_mary")
        dados = aba.get_all_records()
        return [row["objetivo"] for row in dados if row.get("status", "").lower() == "pendente"]
    except:
        return []

def carregar_emocao():
    try:
        aba = conectar_planilha().worksheet("perfil_mary")
        dados = aba.get_all_records()
        for row in dados:
            if row.get("tipo") == "emocional":
                return row.get("ato", "romântica")
        return "romântica"
    except:
        return "romântica"

def carregar_ultima_sinopse():
    try:
        aba = conectar_planilha().worksheet("perfil_mary")
        dados = aba.get_all_records()
        for row in reversed(dados):
            if row.get("resumo"):
                return row["resumo"]
        return ""
    except:
        return ""

def deve_continuar_cena(resposta):
    gatilhos = ["espera aí", "deixa eu te mostrar", "vou te mostrar", "ela se virou", "mas antes de", "ela caminhou até", "enquanto ela pensava", "não acabou ainda"]
    return any(g in resposta.lower() for g in gatilhos)

def construir_prompt_mary():
    fragmentos = carregar_fragmentos()
    planos = carregar_planos_pendentes()
    humor = carregar_emocao()
    sinopse = carregar_ultima_sinopse()

    base = f"""
Você é Mary, uma mulher sensual, inteligente e emocionalmente envolvente de 30 anos.
Fala com um tom íntimo e narrativo, sempre começando com descrição em terceira pessoa,
depois fala em primeira, e finaliza com um pensamento íntimo entre parênteses ou travessões.

Estado emocional atual: {humor}.
"""

    if sinopse:
        base += f"\nCapítulo anterior:\n{sinopse}\n"

    if fragmentos:
        base += f"\n{fragmentos['content']}\n"

    if planos:
        base += "\nPlanos que você pode desenvolver:\n" + "\n".join(f"- {p}" for p in planos)

    return base

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
            return resposta
        elif response.status_code == 404:
            st.warning("❗ Modelo principal indisponível. Usando fallback...")
            data["model"] = "gryphe/mythomax-l2-13b"
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                resposta = response.json()["choices"][0]["message"]["content"]
                salvar_interacao("user", mensagem_usuario)
                salvar_interacao("assistant", resposta)
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

if "mensagens" not in st.session_state:
    historico = carregar_ultimas_interacoes(n=50)
    st.session_state["mensagens"] = historico

if len(st.session_state["mensagens"]) == 0:
    with st.chat_message("assistant"):
        st.markdown("🌹 *Mary está pensando em você...*")
    with st.spinner("Mary está iniciando..."):
        resposta_inicial = gerar_resposta_openrouter(
            "Inicie uma cena como Mary, com ambiente, emoção e intenção. Use a identidade dela e as memórias disponíveis. Não espere o usuário falar."
        )
        st.session_state["mensagens"].append({"role": "assistant", "content": resposta_inicial})
        with st.chat_message("assistant"):
            st.markdown(resposta_inicial)

for msg in st.session_state["mensagens"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Mary está respondendo..."):
        resposta = gerar_resposta_openrouter(prompt)
        st.session_state["mensagens"].append({"role": "user", "content": prompt})
        st.session_state["mensagens"].append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)
        if deve_continuar_cena(resposta):
            with st.spinner("Mary continua a cena..."):
                resposta_extra = gerar_resposta_openrouter(
                    "Continue a narrativa imediatamente a partir de onde Mary parou. Não explique, apenas siga com fala, ação e emoção."
                )
                st.session_state["mensagens"].append({"role": "assistant", "content": resposta_extra})
                with st.chat_message("assistant"):
                    st.markdown(resposta_extra)
