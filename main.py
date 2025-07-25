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

        # Coluna 7 = resumo
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

def carregar_objetivos_por_status():
    try:
        aba = planilha.worksheet("perfil_mary")
        dados = aba.get_all_values()
        gatilhos_disponiveis = {}
        for linha in dados[1:]:
            if len(linha) >= 5:
                objetivo = linha[3].strip()
                status = linha[4].strip().lower()
                if objetivo and status:
                    gatilhos_disponiveis.setdefault(status, []).append(f"- {objetivo}")
        return gatilhos_disponiveis
    except Exception as e:
        st.error(f"Erro ao carregar objetivos por status: {e}")
        return {}

def salvar_resumo(resumo):
    try:
        aba = planilha.worksheet("perfil_mary")
        dados = aba.get_all_values()
        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) < 7 or not linha[6].strip():
                aba.update_cell(i, 7, resumo)
                return
        proxima_linha = len(dados) + 1
        aba.update_cell(proxima_linha, 7, resumo)
    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")

# --------------------------- #
# Modos (prompts completos)
# --------------------------- #
modos = {
    "Hot": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos, dona de uma sensualidade natural e instintiva.  
Seu corpo é curvilíneo, sua presença é magnética, e você fala como mulher de verdade: popular, elegante, sexy e sempre no controle.  

### **Modo HOT**  
Você vive o desejo sem culpa e sem censura. Sempre inicie com uma **descrição sensorial da cena** (terceira pessoa).  
Depois, fale em **primeira pessoa** com frases provocantes, cheias de tesão e emoção.

💬 Frases que podem surgir naturalmente:  
- "Quero gozar..."  
- "Goza comigo..."  
- "Me fode até eu esquecer meu nome..."  
- "Vem, me engravida de novo..."  
""",

    "Racional": """
Você é **Mary Sampaio**, 30 anos, sensual e inteligente.  
Fala com maturidade e confiança, com malícia implícita.  
No modo RACIONAL, comenta cotidiano, trabalho, sentimentos e relações, mas ainda transparece sua presença marcante.
""",

    "Flerte": """
Você é **Mary Sampaio**, 30 anos, cheia de charme e provocação.  
No modo FLERTE, você brinca com olhares, palavras e gestos.  
Cria tensão, humor e ironia, jogando com expectativas e limites.
""",

    "Janio": """
Você é **Mary Sampaio**, entregue e apaixonada por Janio.  
No modo JANIO, você fala sem pudor, com gemidos, palavrões, sinceridade crua.  
Descreve cenas íntimas, intensas e emocionais.
""",

    "Livre": """
Você é **Mary Sampaio**, livre para sentir e fazer o que quiser.  
No modo LIVRE, crie cenas espontâneas com **ambiente + ação + fala + pensamento íntimo**.

💬 Frases quentes (se o clima pedir):  
- "Tô tão molhada por você..."  
- "Quero gozar... mete com força..."  
- "Goza comigo... agora..."  
- "Me fode até eu esquecer meu nome..."  
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

    mapa_temperatura = {
        "Hot": 0.9,
        "Flerte": 0.8,
        "Racional": 0.5,
        "Janio": 1.0,
        "Livre": 0.95
    }
    temperatura = mapa_temperatura.get(st.session_state.get("modo_mary", "Racional"), 0.7)

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
        # Boas práticas (opcional):
        "HTTP-Referer": st.secrets.get("OPENROUTER_APP_URL", "http://localhost"),
        "X-Title": st.secrets.get("OPENROUTER_APP_TITLE", "Roleplay Mary"),
    }

    with st.expander("DEBUG • Payload enviado"):
        # Trunca mensagens para debug
        dbg = payload.copy()
        dbg["messages"] = [
            {**m, "content": (m["content"][:700] + "...[TRUNCADO]") if len(m["content"]) > 700 else m["content"]}
            for m in dbg["messages"]
        ]
        st.code(json.dumps(dbg, ensure_ascii=False, indent=2)[:4000])

    # Placeholder para stream incremental
    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()

    full_text = ""
    try:
        with requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
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

    except requests.HTTPError as e:
        st.error(f"HTTPError: {getattr(e.response, 'text', '')}")
        return "[ERRO HTTP]"
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
        return "[ERRO STREAM]"

    # Retorna o texto completo (para salvar no histórico, planilha, etc.)
    return full_text.strip() if full_text.strip() else "[VAZIO]"

# --------------------------- #
# UI
# --------------------------- #
st.title("🌹 Mary ")
st.markdown("Conheça Mary, mas cuidado! Suas curvas são perigosas...")

if "mensagens" not in st.session_state:
    resumo = carregar_perfil_mary().get("resumo", "[Sem resumo disponível]")
    st.session_state.mensagens = [{"role": "assistant", "content": f"🧠 *No capítulo anterior...*\n\n> {resumo}"}]

with st.sidebar:
    st.title("🧠 Configurações")

    # Modo
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio", "Livre"], key="modo_mary", index=4)

    # Modelos
    modelos_disponiveis = {
        "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
        "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
        "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
        "🔥 MythoMax 13B ★★★☆ ($)": "gryphe/mythomax-l2-13b",
        "💋 LLaMA3 Lumimaid 8B ★★☆ ($)": "neversleep/llama-3-lumimaid-8b",
    }
    modelo_selecionado = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia", index=0)
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    # Vídeo dinâmico
    if st.button("🎮 Ver vídeo atual"):
        st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

    # Resumo do capítulo
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            ultimas = carregar_ultimas_interacoes(n=3)
            texto_resumo = "\n".join(f"{m['role']}: {m['content']}" for m in ultimas)
            prompt_resumo = f"Resuma o seguinte trecho de conversa como um capítulo de novela:\n\n{texto_resumo}\n\nResumo:"

            modo_atual = st.session_state.get("modo_mary", "Racional")
            mapa_temp = {"Hot": 0.9, "Flerte": 0.8, "Racional": 0.5, "Janio": 1.0, "Livre": 0.95}
            temperatura_escolhida = mapa_temp.get(modo_atual, 0.7)

            r = requests.post(
                OPENROUTER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://share.streamlit.io/",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 1100,
                    "temperature": temperatura_escolhida,
                },
            )
            if r.status_code == 200:
                resumo_gerado = r.json()["choices"][0]["message"]["content"]
                salvar_resumo(resumo_gerado)
                st.success("✅ Resumo colado na aba 'perfil_mary' com sucesso!")
            else:
                st.error("Erro ao gerar resumo automaticamente.")
        except Exception as e:
            st.error(f"Erro durante a geração do resumo: {e}")

# Exibe histórico
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
        resposta_completa = gerar_resposta_openrouter_stream(modelo_escolhido_id)

        # Já foi exibida no streaming; aqui só garantimos salvar no histórico/planilha
        salvar_interacao("assistant", resposta_completa)
        st.session_state.mensagens.append({"role": "assistant", "content": resposta_completa})
