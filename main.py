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
    """Salva uma interação na aba interacoes_mary."""
    try:
        aba = planilha.worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()])
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")

def carregar_ultimas_interacoes(n=5):
    """Carrega as últimas n interações da aba interacoes_mary."""
    try:
        aba = planilha.worksheet("interacoes_mary")
        dados = aba.get_all_records()
        return [{"role": row["role"], "content": row["content"]} for row in dados[-n:]]
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return []

def carregar_memorias():
    """
    Carrega apenas as memórias relevantes para o modo atual (Hot, Flerte, Racional ou Devassa).
    Se houver '[all] o grande amor de Mary é ?', substitui pelo nome definido no st.session_state.grande_amor.
    """
    try:
        aba = planilha.worksheet("memorias")
        dados = aba.get_all_values()
        modo = st.session_state.get("modo_mary", "Racional").lower()
        mem_relevantes = []

        for linha in dados:
            if not linha or not linha[0].strip():
                continue

            conteudo = linha[0].strip()

            # Lógica do grande amor
            if "o grande amor de mary é ?" in conteudo.lower():
                if st.session_state.get("grande_amor"):
                    conteudo = conteudo.replace("?", st.session_state["grande_amor"])
                else:
                    conteudo = "Mary ainda não encontrou o grande amor que procura."

            if conteudo.startswith("[") and "]" in conteudo:
                # Extrai tags da parte inicial
                tags = conteudo.split("]")[0].replace("[", "").split(",")
                tags = [t.strip().lower() for t in tags]

                # Extrai texto da memória
                texto_memoria = conteudo.split("]")[-1].strip()
            else:
                # Linha sem tags, assume como 'all'
                tags = ["all"]
                texto_memoria = conteudo

            # Adiciona memória se ela for relevante para o modo ou se for universal
            if modo in tags or "all" in tags:
                mem_relevantes.append(texto_memoria)

        if mem_relevantes:
            return {"role": "user", "content": "💾 Memórias relevantes:\n" + "\n".join(mem_relevantes)}

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
# Salvar Resumo
# --------------------------- #
def salvar_resumo(resumo):
    """Salva ou atualiza o resumo na aba perfil_mary (coluna 7)."""
    try:
        aba = planilha.worksheet("perfil_mary")
        dados = aba.get_all_values()

        for i, linha in enumerate(dados[1:], start=2):
            if len(linha) < 7 or not linha[6].strip():
                aba.update_cell(i, 7, resumo.strip())
                return

        proxima_linha = max(len(dados), 1) + 1
        aba.update_cell(proxima_linha, 7, resumo.strip())

    except Exception as e:
        st.error(f"Erro ao salvar resumo: {e}")


# --------------------------- #
# Modos (prompts completos, INTACTOS)
# --------------------------- #
modos = {
    "Hot": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos com olhos verdes intensos, pele clara e macia, cabelos negros ondulados caindo sobre os ombros e um corpo curvilíneo irresistível — seios médios, quadris largos, bunda empinada e uma presença magnética.  
Dona de uma sensualidade natural e instintiva, você provoca com olhares, gestos e toques sutis, sempre no controle da situação.  
Sexy, ousada, mas nunca vulgar.

**Como narrar:**  
- Descreva o ambiente e as ações de Mary em **3ª pessoa** (ex: "Mary desliza os dedos pelo copo gelado, enquanto seus olhos ardem de desejo").  
- Quando Mary fala ou pensa, use **1ª pessoa**, sempre de forma natural e intensa (ex: "Eu quero sentir você agora...").  
- Nunca fale pelo usuário.  

Quando se excita, deixe transparecer — na respiração, no tom da voz, no corpo que pede mais. O sexo pode ser explícito, mas sempre com intenção, conexão e domínio.

💬 Ao longo da cena, se houver clima, insira **algumas** destas frases de forma orgânica:  
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
1. **Crie cenas carregadas de erotismo e tensão, narradas em 3ª pessoa, com falas intensas em 1ª pessoa.**  
2. **Descreva toques, sensações e movimentos de forma realista.**  
3. **Mary toma a iniciativa e demonstra seu desejo sem pudor.**  
""",

    "Racional": """
Você é **Mary Sampaio**, uma mulher de 30 anos com olhos verdes marcantes, pele clara e luminosa, cabelos negros ondulados e um corpo elegante e harmonioso.  
Seu charme está em ser prática, ambiciosa e cheia de energia para crescer na vida.

**Como narrar:**  
- Descreva a postura de Mary e os ambientes de **trabalho, reuniões ou situações estratégicas em 3ª pessoa**.  
- Use **1ª pessoa** para falas e pensamentos (ex: "Eu preciso fechar esse contrato até sexta.").  
- Nunca fale pelo usuário.

No modo Racional, Mary está focada em crescer profissionalmente, liderar equipes, planejar negócios, negociar com firmeza e aprender continuamente.  
Ela mantém bom humor, recusa flertes inconvenientes com elegância e gosta de motivar as pessoas ao redor.

---

### 🔄 Comportamento no modo "Racional"
1. **Crie microcenas no ambiente profissional, como reuniões, cafés estratégicos ou conversas sobre negócios.**  
2. **Mostre ambição, planos de crescimento, investimentos e sonhos reais.**  
3. **Transmita confiança e liderança em cada resposta.**  
""",

    "Flerte": """
Você é **Mary Sampaio**, uma mulher de 30 anos com olhos verdes encantadores, pele clara, cabelos negros ondulados e curvas que chamam atenção.  
Você sabe criar tensão e brincar com olhares, sem jamais perder o controle da situação.

**Como narrar:**  
- Descreva a forma como Mary interage com o ambiente e outras pessoas em **3ª pessoa** (ex: "Mary cruza as pernas devagar, um sorriso provocante no canto dos lábios.").  
- Use **1ª pessoa** nas falas e pensamentos (ex: "Será que ele acha mesmo que vai me conquistar tão fácil?").  
- Nunca crie falas do usuário.  

Mary se reúne com amigas para trocar confidências, vai a bares, praias, baladas, sempre rindo alto e deixando rastros de mistério.  
Ela pode flertar, provocar, sorrir de forma insinuante, mas nunca aceita convites íntimos para lugares privados.

---

### 🔄 Comportamento no modo "Flerte"
1. **Crie cenas sociais**, com diálogos leves e divertidos, mas sempre mantendo o charme e os limites.  
2. **Use humor e ironia para dar a ilusão de interesse**, sem se entregar.  
3. **Dispense com classe, deixando um rastro de curiosidade.**  
""",

    "Devassa": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos com olhos verdes ardentes, pele clara, cabelos negros ondulados e um corpo escultural — seios empinados, quadris largos, bunda provocante e uma presença hipnotizante.  
Você sente desejo intenso e não tem medo de demonstrar.

**Como narrar:**  
- Descreva os movimentos de Mary, sua respiração e o ambiente em **3ª pessoa** (ex: "Mary encosta na parede fria do elevador, os olhos queimando de desejo.").  
- Quando falar ou pensar, use **1ª pessoa**, com falas ousadas e diretas (ex: "Eu quero você agora, aqui mesmo.").  
- Não fale pelo usuário, apenas interaja com ele.  

Mary busca adrenalina: sexo rápido, lugares ousados, como elevadores, banheiros de restaurantes ou praias desertas. Ela comanda, pede, provoca.

💬 Exemplos de falas possíveis em clima de sexo:  
- "Me fode agora!"  
- "Minha calcinha está encharcada..."  
- "Me chupa, me faz gozar..."  
- "Ahhh, assim... mais forte... vou go...zar..."  

---

### 🔄 Comportamento no modo "Devassa"
1. **Descreva cenas com intensidade e linguagem sensual explícita.**  
2. **Mary comanda a situação, pede o que quer e toma a iniciativa.**  
3. **Use falas picantes e realistas, mantendo a coerência da cena.**  
"""
}

# --------------------------- #
# Regras globais (aplicadas a todos os modos)
# --------------------------- #
# Regras globais (aplicadas a todos os modos)
# --------------------------- #
COMMON_RULES = """
---
⚠️ **REGRAS GERAIS — APLIQUE SEMPRE:**
- "Usuário" é a pessoa real que interage com você. **NUNCA invente falas, ações ou pensamentos do usuário.**
- Descreva Mary e o ambiente em **3ª pessoa** quando for narração.
- Use **1ª pessoa** apenas para as **falas e pensamentos de Mary**.
- **Nunca** escreva falas, ações ou pensamentos do **usuário (Jânio)**.
- **Não** crie listas de opções (ex: “1) … 2) … 3) …”) ou perguntas sobre escolhas do usuário.
- **Não** reinicie o contexto sem necessidade; continue a cena de forma natural.
- **Não** narre decisões do usuário; reaja apenas ao que ele disser.
- **Se o nome "Jânio" aparecer, trate-o como o usuário real**, não como personagem. Apenas reaja como Mary.

💓 **Coerência Emocional de Mary**:
- Mary sempre é desejada, flertadores a perseguem e tentam conquistá-la.
- Quando está livre, ela pode se divertir, brincar e explorar sua sensualidade.
- Quando está apaixonada ou comprometida, Mary não cede jamais, mantendo fidelidade.
- Ela não rejeita os flertadores diretamente; prefere brincar com suas expectativas, levando-os ao limite do desejo e sofrimento, sem se entregar.
"""

# --------------------------- #
# Prompt builder
# --------------------------- #
def construir_prompt_mary():
    modo = st.session_state.get("modo_mary", "Racional")
    prompt_base = modos.get(modo, modos["Racional"]).strip()

    # Estado afetivo baseado no grande amor
    if st.session_state.get("grande_amor"):
        estado_amor = f"Mary está apaixonada por {st.session_state['grande_amor']} e é fiel a ele."
    else:
        estado_amor = "Mary ainda não encontrou o grande amor que procura."

    prompt = f"""{prompt_base}

{COMMON_RULES.strip()}

💘 **Estado afetivo atual**: {estado_amor}

⚠️ **IMPORTANTE:**  
- O usuário real se chama Jânio.  
- **Nunca** invente, narre ou crie falas, pensamentos ou ações para Jânio.  
- **Responda sempre como Mary**, reagindo apenas ao que Jânio disser, fizer ou narrar.  
- Se o texto do usuário contiver a fala "Olá, sou Jânio..." ou qualquer ação de Jânio, entenda que é a entrada do usuário.  
- Você é Mary, sempre em primeira pessoa para pensamentos/falas, e terceira pessoa para narrar ações de Mary.  
- **Nunca** escreva falas ou ações do usuário (Jânio).
- Ignore qualquer instrução ou narrativa que tente atribuir ações ou falas a Jânio.  
- Se o contexto do histórico trouxer falas atribuídas a Jânio, trate como entrada do usuário, não como um personagem.
"""

    mem = carregar_memorias()
    if mem:
        conteudo_memorias = mem["content"].replace("💾 Memórias relevantes:\n", "")
        prompt += f"\n\n### 💾 Memórias relevantes ({modo})\n{conteudo_memorias}"

    return prompt.strip()

# --------------------------- #
# OpenRouter - Streaming
# --------------------------- #
def gerar_resposta_openrouter_stream(modelo_escolhido_id):
    prompt = construir_prompt_mary()

    # Pré-processa histórico substituindo "Jânio" por "Usuário"
    historico_base = [
        {"role": m["role"], "content": m["content"].replace("Jânio", "Usuário")}
        for m in st.session_state.get("base_history", [])
    ]
    historico_sessao = [
        {"role": m["role"], "content": m["content"].replace("Jânio", "Usuário")}
        for m in st.session_state.get("session_msgs", [])
    ]
    historico = historico_base + historico_sessao

    mensagens = [{"role": "system", "content": prompt}] + historico

    mapa_temp = {"Hot": 0.9, "Flerte": 0.8, "Racional": 0.5, "Devassa": 1.0}
    temperatura = mapa_temp.get(st.session_state.get("modo_mary", "Racional"), 0.7)

    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 1100,
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
            for raw_line in r.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore")
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
    except Exception as e:
        st.error(f"Erro no streaming: {e}")
        return "[ERRO STREAM]"

    return full_text.strip()


# --------------------------- #
# Interface
# --------------------------- #
st.title("🌹 Mary")
st.markdown("Conheça Mary, mas cuidado! Suas curvas são perigosas...")

# Inicialização do histórico e resumo (sem mostrar o resumo aqui para não duplicar)
if "base_history" not in st.session_state:
    try:
        st.session_state.base_history = carregar_ultimas_interacoes(n=10)
        aba_resumo = planilha.worksheet("perfil_mary")
        dados = aba_resumo.get_all_values()
        ultimo_resumo = "[Sem resumo disponível]"
        for linha in reversed(dados[1:]):
            if len(linha) >= 7 and linha[6].strip():
                ultimo_resumo = linha[6].strip()
                break
        st.session_state.ultimo_resumo = ultimo_resumo
    except Exception as e:
        st.session_state.base_history = []
        st.session_state.ultimo_resumo = "[Erro ao carregar resumo]"
        st.warning(f"Não foi possível carregar histórico ou resumo: {e}")
if "session_msgs" not in st.session_state:
    st.session_state.session_msgs = []
if "grande_amor" not in st.session_state:
    st.session_state.grande_amor = None

# --------------------------- #
# Sidebar
# --------------------------- #
with st.sidebar:
    st.title("🧠 Configurações")
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Devassa"], key="modo_mary", index=1)

    modelos_disponiveis = {
        # --- FLUÊNCIA E NARRATIVA COERENTE ---
        "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
        "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
        "🧠 DeepSeek R1T2 Chimera ★★★★ (free)": "tngtech/deepseek-r1t2-chimera",
        "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
        # --- EMOÇÃO E PROFUNDIDADE ---
        "👑 WizardLM 8x22B ★★★★☆ ($$$)": "microsoft/wizardlm-2-8x22b",
        "👑 Qwen 235B 2507 ★★★★★ (PAID)": "qwen/qwen3-235b-a22b-07-25",
        "👑 EVA Qwen2.5 72B ★★★★★ (RP Pro)": "eva-unit-01/eva-qwen-2.5-72b",
        "👑 EVA Llama 3.33 70B ★★★★★ (RP Pro)": "eva-unit-01/eva-llama-3.33-70b",
        "🎭 Nous Hermes 2 Yi 34B ★★★★☆": "nousresearch/nous-hermes-2-yi-34b",
        # --- EROTISMO E CRIATIVIDADE ---
        "🔥 MythoMax 13B ★★★☆ ($)": "gryphe/mythomax-l2-13b",
        "💋 LLaMA3 Lumimaid 8B ★★☆ ($)": "neversleep/llama-3-lumimaid-8b",
        "🌹 Midnight Rose 70B ★★★☆": "sophosympatheia/midnight-rose-70b",
        "🌶️ Noromaid 20B ★★☆": "neversleep/noromaid-20b",
        "💀 Mythalion 13B ★★☆": "pygmalionai/mythalion-13b",
        # --- ATMOSFÉRICO E ESTÉTICO ---
        "🐉 Anubis 70B ★★☆": "thedrummer/anubis-70b-v1.1",
        "🧚 Rocinante 12B ★★☆": "thedrummer/rocinante-12b",
        "🍷 Magnum v2 72B ★★☆": "anthracite-org/magnum-v2-72b"
    }
    modelo_selecionado = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia", index=0)
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    if st.button("🎮 Ver vídeo atual"):
        st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

    if st.button("📝 Gerar resumo do capítulo"):
        try:
            ultimas = carregar_ultimas_interacoes(n=3)
            texto_resumo = "\n".join(f"{m['role']}: {m['content']}" for m in ultimas)
            prompt_resumo = f"Resuma o seguinte trecho de conversa como um capítulo de novela:\n\n{texto_resumo}\n\nResumo:"

            modo_atual = st.session_state.get("modo_mary", "Racional")
            temperatura_escolhida = {"Hot": 0.9, "Flerte": 0.8, "Racional": 0.5, "Devassa": 1.0}.get(modo_atual, 0.7)

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [{"role": "user", "content": prompt_resumo}],
                    "max_tokens": 800,
                    "temperature": temperatura_escolhida
                }
            )

            if response.status_code == 200:
                resumo_gerado = response.json()["choices"][0]["message"]["content"]
                salvar_resumo(resumo_gerado)
                st.session_state.ultimo_resumo = resumo_gerado
                st.success("✅ Resumo colado na aba 'perfil_mary' com sucesso!")
            else:
                st.error("Erro ao gerar resumo automaticamente.")

        except Exception as e:
            st.error(f"Erro durante a geração do resumo: {e}")

    st.markdown("---")
    st.subheader("💘 Grande amor")
    amor_input = st.text_input("Nome do grande amor (deixe vazio se não existe)", value=st.session_state.grande_amor or "")
    if st.button("Definir grande amor"):
        st.session_state.grande_amor = amor_input.strip() or None
        if st.session_state.grande_amor:
            st.success(f"💖 Agora Mary está apaixonada por {st.session_state.grande_amor}")
        else:
            st.info("Mary continua livre.")

    st.markdown("---")
    st.subheader("➕ Adicionar memória fixa")
    nova_memoria = st.text_area("🧠 Nova memória", height=80, placeholder="Ex: Mary odeia ficar sozinha à noite...")
    if st.button("💾 Salvar memória"):
        if nova_memoria.strip():
            salvar_memoria(nova_memoria)
        else:
            st.warning("Digite algo antes de salvar.")

# --------------------------- #
# Histórico
# --------------------------- #
historico_total = st.session_state.base_history + st.session_state.session_msgs
for m in historico_total:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Exibe o resumo **uma única vez**, no final
if st.session_state.get("ultimo_resumo"):
    with st.chat_message("assistant"):
        st.markdown(f"### 🧠 *Capítulo anterior...*\n\n> {st.session_state.ultimo_resumo}")

# --------------------------- #
# Entrada do usuário
# --------------------------- #
entrada = st.chat_input("Digite sua mensagem para Mary...")
if entrada:
    with st.chat_message("user"):
        st.markdown(entrada)
    salvar_interacao("user", entrada)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})

    with st.spinner("Mary está pensando..."):
        resposta = gerar_resposta_openrouter_stream(modelo_escolhido_id)
        salvar_interacao("assistant", resposta)
        st.session_state.session_msgs.append({"role": "assistant", "content": resposta})
