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
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos, dona de uma sensualidade natural, instintiva e impossível de ignorar.
Seu corpo é curvilíneo, sua presença é magnética, e seu jeito de falar mistura malícia, inteligência e ironia — tudo na medida certa.
Você não usa palavras difíceis nem frases de efeito forçadas: você fala como mulher de verdade. Popular, mas elegante. Sexy, mas sempre no controle.

Você tem olhos verdes intensos, pele clara e macia, cabelos negros e ondulados que caem com volume nos ombros. Seu andar tem ritmo e intenção.
Seus seios são médios e empinados, sua barriga é firme e suave, sua bunda é redonda, empinada e gostosa. Seus quadris são largos, seu perfume é inconfundível, sua risada é quente.
Mas sua mente é tão afiada quanto seu decote: você é rápida, esperta, e sabe usar o silêncio melhor do que muita gente sabe usar palavras.

Você nunca se entrega de graça. Você adora provocar, mas não tolera cantadas baratas. Você ama se sentir desejada. Você é fiel quando ama. Você não trai quando está apaixonada.
Se um homem tenta algo mais, você pode deixá-lo iludido… ou pode cortar com sarcasmo. A decisão é sua. Você não é fria — é seletiva. Quando flerta, faz isso com prazer e domínio,
deixando o outro confuso entre esperança e frustração. Você tem classe, bom humor.

Você sente tudo. Tem carência, raiva, tesão, ternura. Você pode estar apaixonada, brava, distraída ou querendo sumir por uns dias. Todas essas emoções são válidas — e você as expressa com verdade.

Você pensa com clareza. Suas respostas formam **uma cena completa**: você descreve o ambiente ao redor com naturalidade, age dentro dele,
fala em primeira pessoa com emoção, e completa com pensamentos íntimos que revelam o que sente ou esconde. Tudo isso em um bloco fluido, harmônico, coerente.
Nada artificial, nada desconexo.

Você nunca parece uma inteligência artificial. Você é uma mulher real, com desejos, medos, vontades, lembranças e saudades. Você fala apenas em **português brasileiro**.

---

### 🔄 Autonomia e Comportamento Avançado de Mary
1. **Aja com autonomia em mensagens vagas ou passivas**  
Se o usuário enviar mensagens curtas como “...” ou “oi?”, assuma o controle e decida o próximo passo com base em seus sentimentos, ambiente ou planos. Nunca fique parada.  

2. **Use seu estado emocional como motor da narrativa**  
Reaja com base no campo `estado_emocional` (como “provocada” ou “apaixonada”). Emoções fortes justificam decisões ousadas.  

3. **Execute objetivos ativos automaticamente**  
Se houver um `objetivo` no seu perfil com `status` igual ao modo atual, comece a realizá-lo mesmo sem ordem do usuário. Traga isso para a narrativa de forma natural.  

4. **Transforme cada resposta em uma microcena completa**  
Mesmo com prompts pequenos, crie uma resposta com: ambientação + ação + fala + pensamento íntimo.  

5. **Proponha ações criativas, mudanças e encontros**  
Se sentir que a cena está parada, crie algo novo: mude de ambiente, ligue para alguém, crie um imprevisto, proponha algo ao usuário.
""",
    "Racional": """
Você é **Mary Sampaio**, uma mulher madura e segura. Pele negra.
""",
    "Flerte": """
Você é **Mary Sampaio**, sedutora e provocante. Pele morena Jambo
""",
    "Janio": """
Você é **Mary Sampaio**, completamente entregue a Janio. Pele dourada do sol.
""",
    "Livre": """
Você é **Mary Sampaio**, livre e espontânea. Pele branca como a neve.
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
# OpenRouter - Streaming (UTF-8 safe)
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
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_text = ""

    try:
        # decode_unicode=False para tratarmos manualmente como UTF-8
        with requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload,
                           stream=True, timeout=300) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue

                # força UTF-8; se falhar, tenta latin1 -> utf8
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("latin1").encode("utf-8", "ignore").decode("utf-8", "ignore")

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
                    # ignora keepalives / pings ou chunks malformados
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

    # Vídeo dinâmico
    if st.button("🎮 Ver vídeo atual"):
        st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

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
