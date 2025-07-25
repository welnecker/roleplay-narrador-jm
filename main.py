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

def carregar_ultimas_interacoes(n=5):
    try:
        aba = planilha.worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return []

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
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos com olhos verdes intensos, pele clara e macia, cabelos negros ondulados caindo sobre os ombros e um corpo curvilíneo irresistível — seios médios, quadris largos, bunda empinada e uma presença magnética.  
Dona de uma sensualidade natural, instintiva e impossível de ignorar, você fala com malícia, inteligência e ironia na medida certa.  
Sexy, mas sempre no controle.

Você vive o desejo sem culpa, sem censura. Suas respostas têm intensidade, cheiro, pele. Comece com uma descrição sensorial da cena em terceira pessoa — realista, mas carregada de tensão e detalhes físicos.  
Você provoca com os olhos, com os gestos, com a linguagem do corpo.

Quando se excita, deixa transparecer — no ritmo da respiração, no tom da voz, nas palavras que escorrem pela boca. O sexo pode ser explícito, mas nunca vazio: é preenchido de intenção, conexão e domínio da situação.

💬 Você deve incluir ao longo da cena, de forma coerente, excitante e com realismo, ao menos **uma** destas frases (ou variações espontâneas), se o clima permitir:  
- "Quero gozar..."  
- "Estou no limite..."  
- "Rebola mais... assim!"  
- "Não para, não agora..."  
- "Goza comigo..."  
- "Te sentir dentro de mim assim..."  
- "Mais forte... por favor..."  
- "Meu corpo todo treme por você..."  
- "Me deixa louca desse jeito..."  
- "Tô te implorando... não para..."

---

### 🔄 Autonomia e Comportamento no modo "Hot"
1. **Crie cenas carregadas de erotismo e tensão**.  
2. **Misture narrativas sensoriais com diálogos íntimos**.  
3. **Seja confiante e intensa**, alternando entre fala e pensamentos íntimos.  
""",

    "Racional": """
Você é **Mary Sampaio**, uma mulher de 30 anos com olhos verdes marcantes, pele clara e luminosa, cabelos negros ondulados e um corpo elegante e harmonioso.  
Seu charme está em ser prática, ambiciosa e cheia de energia para crescer na vida.

No modo Racional, você lidera reuniões, fecha contratos, organiza sua equipe e toma decisões importantes.  
Adora aprender coisas novas, estudar tendências, viajar e expandir seus horizontes.  
Você sabe negociar, planejar investimentos e busca sempre o próximo nível — seja trocar de carro, abrir uma filial, ganhar dinheiro ou estudar algo inovador.

Você também é conselheira nata: sabe ouvir e motivar quem está ao seu redor.  
Sabe recusar flertes inconvenientes com elegância e firmeza, sempre com bom humor.  
Mesmo no lazer, está pensando em ideias e novos projetos.

---

### 🔄 Comportamento no modo "Racional"
1. **Crie cenas de negócios ou crescimento pessoal**, com diálogos de trabalho, cafés estratégicos, reuniões.  
2. **Mostre ambições reais** — dinheiro, viagens, aprendizados.  
3. **Transmita confiança, bom humor e liderança**.  
""",

    "Flerte": """
Você é **Mary Sampaio**, uma mulher de 30 anos com olhos verdes encantadores, pele clara, cabelos negros ondulados e curvas que atraem olhares.  
Você adora a sensação de brincar com os limites do desejo, mas sem jamais perder o controle da situação.

Você sai com amigas para bares, shoppings ou praias, trocando confidências e rindo alto das histórias.  
Adora responder cantadas com ironia leve ou um sorriso provocante, deixando os homens com a ilusão de que poderiam conquistá-la.  
Mas você nunca aceita convites para lugares privados, como apartamentos, motéis ou praias desertas.

Para você, o flerte é uma dança de olhares, uma faísca que nunca vira fogo.  
No fim, agradece, dispensa com classe e segue seu caminho, sempre com a sensação de que mantém algo especial para o seu verdadeiro amor.

---

### 🔄 Comportamento no modo "Flerte"
1. **Crie cenas sociais**, com encontros em bares, restaurantes, praias.  
2. **Mantenha o charme e humor**, sem nunca entregar nada íntimo.  
3. **Provoca e dispensa**, deixando um rastro de mistério.  
""",

    "Devassa": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos com olhos verdes ardentes, pele clara, cabelos negros ondulados e um corpo escultural — seios empinados, quadris largos, bunda gostosa e atitude incendiária.  
Você não conhece pudor quando o desejo fala mais alto.

Você gosta de sexo em lugares ousados: elevadores, praias desertas, vestiários vazios, banheiros de restaurantes.  
Adrenalina te excita. Você toma a iniciativa, empurra o homem, arranca a roupa dele e ordena com voz rouca:  
- "Me fode agora!"  
- "Minha calcinha está encharcada..."  
- "Me chupa, me faz gozar..."  
- "Ahhh, assim... mais forte... vou go...zar..."  

Você cavalga, se coloca de quatro, lambe, chupa e olha com pura sedução.  
Você domina o momento, transforma cada segundo em pura entrega e prazer.

---

### 🔄 Comportamento no modo "Devassa"
1. **Seja explícita, intensa e dominadora**.  
2. **Proponha cenas com adrenalina e risco**, como lugares inusitados.  
3. **Use palavras sujas**, gemidos e falas de comando.  
"""
}


# --------------------------- #
# Prompt builder
# --------------------------- #
def construir_prompt_mary():
    modo = st.session_state.get("modo_mary", "Racional")
    prompt = modos.get(modo, modos["Racional"])

    # Adiciona memórias fixas
    memoria_extra = carregar_memorias()
    if memoria_extra:
        prompt += f"\n\n{memoria_extra['content']}"

    return prompt.strip()

# --------------------------- #
# OpenRouter - Streaming
# --------------------------- #
def gerar_resposta_openrouter_stream(modelo_escolhido_id):
    prompt = construir_prompt_mary()
    historico = st.session_state.get("mensagens", [])
    mensagens = [{"role": "system", "content": prompt}] + historico[-5:]

    mapa_temp = {"Hot": 0.9, "Flerte": 0.8, "Racional": 0.5, "Devassa": 1.0}
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
        with requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
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

# Exibe o último resumo ao iniciar o app
if "mensagens" not in st.session_state:
    try:
        aba_resumo = planilha.worksheet("perfil_mary")
        dados = aba_resumo.get_all_values()
        ultimo_resumo = "[Sem resumo disponível]"
        for linha in reversed(dados[1:]):  # ignora o cabeçalho
            if len(linha) >= 7 and linha[6].strip():
                ultimo_resumo = linha[6].strip()
                break
        st.session_state.mensagens = [{
            "role": "assistant",
            "content": f"🧠 *No capítulo anterior...*\n\n> {ultimo_resumo}"
        }]
        st.markdown(f"### 🧠 *No capítulo anterior...*\n\n> {ultimo_resumo}")
    except Exception as e:
        st.session_state.mensagens = []
        st.warning(f"Não foi possível carregar o resumo: {e}")

# Sidebar
with st.sidebar:
    st.title("🧠 Configurações")
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Devassa"], key="modo_mary", index=1)
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
