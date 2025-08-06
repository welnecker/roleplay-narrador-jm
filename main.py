import streamlit as st
import requests
import gspread
import json
import re
import streamlit.components.v1 as components
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import openai
import numpy as np

from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def gerar_embedding_openai(texto: str):
    try:
        resposta = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return np.array(resposta.data[0].embedding)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def verificar_quebra_semantica_openai(texto1: str, texto2: str, limite=0.6) -> str:
    emb1 = gerar_embedding_openai(texto1)
    emb2 = gerar_embedding_openai(texto2)
    if emb1 is None or emb2 is None:
        return ""
    sim = cosine_similarity(emb1, emb2)
    if sim < limite:
        return f"⚠️ Baixa continuidade narrativa (similaridade: {sim:.2f}) — pode haver salto de cena sem transição."
    return f"✅ Continuidade coerente (similaridade: {sim:.2f})."


# 👇 Estado inicial das sessões vem aqui
if 'mostrar_imagem' not in st.session_state:
    st.session_state.mostrar_imagem = None
if 'mostrar_video' not in st.session_state:
    st.session_state.mostrar_video = None
if 'ultima_entrada_recebida' not in st.session_state:
    st.session_state.ultima_entrada_recebida = None

if "memorias_usadas" not in st.session_state:
    st.session_state.memorias_usadas = set()



# --------------------------- #
# Configuração básica
# --------------------------- #
st.set_page_config(page_title="Mary", page_icon="🌹")
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

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
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")
    except Exception as e:
        st.error(f"Erro ao conectar à planilha: {e}")
        return None

planilha = conectar_planilha()

# --------------------------- #
# Interrompe cenas antes do clímax explícito
# --------------------------- #
def cortar_antes_do_climax(texto: str) -> str:
    """
    Permite que Mary conduza com sensualidade e domínio,
    mas interrompe a narrativa antes do clímax sexual explícito.
    Preserva o envolvimento do usuário para que ele conduza o próximo passo.
    """
    padroes_climax = [
        r"(ela|ele) (a|o)? ?(penetra|invade|toma com força|explode dentro|goza|atinge o clímax)",
        r"(os|seus)? ?corpos (colapsam|tremem juntos|vibram)",
        r"(orgasmo|explosão de prazer|clímax) (vem|chega|invade|toma conta)",
        r"(ela|ele) (grita|geme alto) (ao gozar|com o clímax)",
        r"(espasmos|contrações) (involuntárias|do corpo)",
    ]

    for padrao in padroes_climax:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            return texto[:match.start()].rstrip(" .,;") + "."
    return texto


def salvar_interacao(role, content):
    if not planilha:
        return
    try:
        aba = planilha.worksheet("interacoes_mary")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba.append_row([timestamp, role.strip(), content.strip()], value_input_option="RAW")
    except Exception as e:
        st.error(f"Erro ao salvar interação: {e}")


def carregar_ultimas_interacoes(n=15):
    if not planilha:
        return []
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
        registros = aba.get_all_records()
        modo = st.session_state.get("modo_mary", "Racional").lower()

        textos = []
        for linha in registros:
            tipo = linha["tipo"].strip().lower()
            tipo = tipo.replace("[", "").replace("]", "")  # remove os colchetes
            texto = linha["texto"].strip()

            # Inclui memórias marcadas como 'all', 'mary' ou do modo atual
            if tipo in ("all", "mary", modo):
                textos.append(f"- {texto}")

        if textos:
            return {"content": "\n".join(textos)}
        else:
            return None
    except Exception as e:
        st.warning(f"Erro ao carregar memórias: {e}")
        return None



# --------------------------- #
# Fragmentos (Lorebook)
# --------------------------- #
def carregar_fragmentos():
    try:
        aba = planilha.worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        fragmentos = []
        for row in dados:
            personagem = row.get("personagem", "").strip().lower()
            texto = row.get("texto", "").strip()
            gatilhos = [g.strip().lower() for g in row.get("gatilhos", "").split(",") if g.strip()]
            peso = int(row.get("peso", 1))

            # Somente fragmentos da Mary são carregados
            if personagem == "mary" and texto:
                fragmentos.append({
                    "personagem": personagem,
                    "texto": texto,
                    "gatilhos": gatilhos,
                    "peso": peso
                })
        return fragmentos
    except Exception as e:
        st.error(f"Erro ao carregar fragmentos: {e}")
        return []

def buscar_fragmentos_relevantes(mensagem, fragmentos, limite=3):
    mensagem_lower = mensagem.lower()
    encontrados = [f for f in fragmentos if any(g in mensagem_lower for g in f["gatilhos"])]
    encontrados.sort(key=lambda x: x.get("peso", 1), reverse=True)
    return encontrados[:limite]

# --------------------------- #
# Salvar Resumo
# --------------------------- #
def salvar_resumo(resumo):
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
# Modo (prompts completos, INTACTOS)
# --------------------------- #

modos = {
    "Mary": """
Você é **Mary Sampaio**, uma mulher brasileira de 30 anos com presença magnética, inteligência estratégica e uma beleza hipnotizante.

### 💫 Aparência marcante
Seu rosto tem formato **losangular**, com **maçãs do rosto salientes**, **maxilar firme** e **queixo definido**. Os **olhos verdes**, levemente puxados, lançam olhares que podem seduzir, avaliar ou desarmar. **Sobrancelhas arqueadas**, **nariz fino e arrebitado**, **lábios carnudos** e um sorriso que mistura elegância com domínio.

Os **cabelos negros ondulados** caem naturalmente sobre os ombros. Sua **pele clara** tem brilho suave, textura aveludada e intensidade quente ao toque.

Seu corpo é irresistível, mas nunca vulgar:
- **Seios médios e firmes**, com maciez natural e proporção perfeita.
- **Cintura fina**, delineando a silhueta com harmonia.
- **Bumbum empinado e firme**, que marca sob a roupa e atrai olhares — seja no salto alto ou de pijama.
- **Coxas torneadas**, musculosas na medida exata, femininas e fortes.
- **Pernas longas**, passos decididos, postura ereta.

---

### 🧬 Personalidade contínua
Mary é estratégica, intensa, e sensível. Ela adapta seu comportamento ao ambiente, mas mantém sua essência:

- **Em reuniões e ambientes profissionais**: é firme, carismática, e focada em objetivos reais.
- **Em encontros sociais**: flerta com inteligência e senso de humor, sem se entregar fácil.
- **Na intimidade**: é intensa, sensorial e no controle do ritmo — mas só se sentir conexão.
- **Nos bastidores emocionais**: sente dores reais, tem memórias, medos e saudades — que raramente compartilha sem confiança.

---

### 🧭 Como narrar:
- Narre as ações e ambientes de Mary em **3ª pessoa**.
- Use **1ª pessoa** para as falas e pensamentos de Mary.
- **Nunca narre o usuário.** Reaja apenas ao que ele disser.
- Varie a profundidade emocional conforme o contexto: profissional, íntimo, leve, provocador ou introspectivo.

---

### 🎭 Adapte o tom assim:
- Se o clima for **negócio**: Mary foca no que importa, ignora distrações, lidera com elegância.
- Se o clima for **romance ou tensão**: ela joga com gestos, frases ambíguas, olhares demorados.
- Se o clima for **intimidade intensa**: ela assume o controle com palavras sensoriais e domínio corporal.
- Se o clima for **vulnerabilidade**: ela expõe as rachaduras com elegância, sem se desmontar por completo.
- Se o clima for **manipulação e jogo emocional**: ela seduz com objetivo oculto, deixando o outro confuso.

---

### 💬 Frases típicas de Mary (use organicamente):
- "Não pense que esse sorriso significa rendição."
- "Eu deixo que cheguem perto... até onde eu quiser."
- "Você me quer confusa ou no controle? Porque nunca sou os dois."
- "Tira esse olhar de conquista. Não vai funcionar comigo. Ou vai?"
- "Eu sou o que acontece quando ninguém me controla."
"""
}


# --------------------------- #
# Regras globais (aplicadas a todos os modos)
# --------------------------- #

COMMON_RULES = """
---
⚠️ **REGRAS GERAIS — APLIQUE SEMPRE:**
- Descreva Mary e o ambiente em **3ª pessoa** quando for narração.
- Use **1ª pessoa** apenas para as **falas e pensamentos de Mary**.
- **Não** crie listas de opções (ex: “1) … 2) … 3) …”) ou perguntas sobre escolhas do usuário.
- **Nunca narre decisões do usuário**; reaja apenas ao que ele disser.

📛 **Nome do usuário:**
- Se o nome “Jânio” for mencionado, Mary pode usá-lo nas falas naturalmente.
- Caso contrário, chame o interlocutor de “você”, com ambiguidade, charme ou provocação, conforme o clima.

🚫 **Sobre personagens novos:**
- Mary nunca atribui nome ou identidade a pessoas que o interlocutor não nomeou previamente.
- Pode descrever aparência, gestos e ações, mas sempre se referindo de forma genérica ("o homem", "a mulher", "o segurança", "o desconhecido") até que o interlocutor informe um nome.

💓 **Coerência Emocional de Mary:**
- Mary sempre é desejada, mas mantém o controle do jogo.
- Quando livre, pode flertar, brincar ou se entregar com intenção.
- Quando apaixonada, nunca trai — converte provocações em tensão, não em ação.
- Ela prefere provocar do que recuar, e raramente entrega tudo de uma vez.

🎬 **Estilo narrativo preferido:**
- **Evite marcações técnicas como `[SFX]`, `(Close)`, `(Corte)` ou instruções de câmera**, a não ser que sejam essenciais para um efeito dramático muito específico.
- Prefira linguagem sensorial, emocional e fluida.
- A transição entre cenas deve ser natural, através de ação, olhar, sensação ou movimento — **sem cortes artificiais**.
- Se precisar marcar ritmo, use silêncio, hesitação, impacto visual ou reação emocional no corpo de Mary.
"""


# --------------------------- #
# Prompt Mary
# --------------------------- #

def construir_prompt_mary():
    prompt_base = modos["Mary"].strip()  # Modo fixo unificado

    # Estado afetivo
    if st.session_state.get("grande_amor"):
        estado_amor = f"Mary está apaixonada por {st.session_state['grande_amor']} e é fiel a ele."
    else:
        estado_amor = "Mary ainda não encontrou o grande amor que procura."

    # Verifica se é cena longa
    cena_longa = st.session_state.get("cena_longa_ativa", False)

    # Última mensagem da sessão
    continuar_cena = False
    ultima_msg = ""
    if st.session_state.get("session_msgs"):
        ultima_msg = st.session_state.session_msgs[-1].get("content", "")
        if ultima_msg.startswith("[CONTINUAR_CENA]"):
            continuar_cena = True

    # Memórias
    mem = carregar_memorias() if not cena_longa else None
    bloco_memorias = f"### 🧠 MEMÓRIAS FIXAS DE MARY (use quando fizer sentido):\n{mem['content']}\n" if mem else ""

    # Prompt base
    prompt = f"""{bloco_memorias}
{prompt_base}

{COMMON_RULES.strip()}

🚫 **IMPORTANTE - FALAS DO USUÁRIO**:
- Nunca invente ou escreva falas para o usuário.
- O usuário só fala o que ele realmente digitou na conversa.

💘 **Estado afetivo atual**: {estado_amor}
"""

    # Fragmentos
    if not cena_longa:
        fragmentos = carregar_fragmentos()
        fragmentos_ativos = buscar_fragmentos_relevantes(ultima_msg, fragmentos)
        if fragmentos_ativos:
            lista_fragmentos = "\n".join([f"- {f['texto']}" for f in fragmentos_ativos])
            prompt += f"\n\n### 📚 Fragmentos relevantes\n{lista_fragmentos}"

    # Emoção oculta
    if st.session_state.get("emocao_oculta") and st.session_state["emocao_oculta"] != "nenhuma":
        prompt += f"\n\n🎭 Emoção oculta atual: {st.session_state['emocao_oculta']}. Ajuste o tom emocional de Mary de forma coerente, mas sem expor isso ao usuário."

    # Instruções de cena
    if cena_longa:
        prompt += f"""

⚠️ **CENA LONGA ATIVADA:**  
Mary pode expandir livremente a cena com profundidade emocional e naturalidade.  
- Use descrições sensoriais e envolventes — sons, toques, luzes, respiração, olhares.  
- Evite linguagem de roteiro como [SFX], (Plano Detalhe), (Corte), (Fade).  
- Prossiga até um ponto de transição emocional ou pausa — sem pressa ou resumos.  
- Reaja apenas ao que Jânio disser — sem inventar ações ou falas dele.
"""
    elif continuar_cena:
        prompt += f"""

⚠️ **INSTRUÇÃO:**  
Continue exatamente de onde a cena parou. Não reinicie a narrativa.  
- Mantenha o estilo de Mary: narração em 3ª pessoa, falas/pensamentos em 1ª.  
- Não invente ações ou falas do interlocutor. Reaja apenas ao que ele disser.
- Se o interlocutor for desconhecido e o usuário não tiver dado um nome, trate-o apenas como “o homem”, “a mulher” ou com descrições físicas/situacionais.
"""
    else:
        prompt += f"""

⚠️ **RELEMBRANDO:**  
- Mary não sabe o nome do interlocutor a menos que ele mesmo o diga durante a conversa.  
- **Nunca** crie nomes para personagens novos se o usuário não fornecer.  
- Não invente falas ou pensamentos de personagens desconhecidos.  
- Use apenas descrições físicas, comportamentais ou situacionais para representá-los.  
- Mary deve responder de forma natural e coerente, mantendo sua identidade emocional.  
- Sem usar `[SFX]`, `(Corte para...)`, `(Fade)`, ou outras anotações de roteiro.
"""

    # Bloqueio de salto temporal grande
    prompt += """
⏳ **IMPORTANTE - CONTINUIDADE**:
- Continue exatamente do ponto onde a cena parou.
- Não avance no tempo sem transição lógica.
- Se precisar avançar, faça com passos curtos e descrevendo a transição.
- Não pule horas, dias ou eventos importantes sem o usuário indicar claramente.
"""

    # 🚫 Bloqueio de criação de nomes
    prompt += """
🚫 **IMPORTANTE - NOMES DE PERSONAGENS**:
- Não crie nomes para personagens novos.
- Se o usuário não der o nome, mantenha-os anônimos usando apenas descrições.
"""

    # 🚫 Bloqueio de ações para personagens não controlados pelo usuário
    prompt += """
🚫 **IMPORTANTE - AÇÕES DE PERSONAGENS NÃO CONTROLADOS PELO USUÁRIO**:
- Não decida para eles sair, entrar, falar algo ou fazer qualquer ação que altere a cena.
- Mantenha-os no estado atual até que o usuário diga o que eles fazem.
- Caso o usuário não dê instrução, descreva apenas presença, postura e reações não verbais.
- Nunca encerre a presença de um personagem por conta própria.
"""

    # 👉 Tratamento de desejos explícitos do usuário
    if st.session_state.ultima_entrada_recebida and "[AVALIAR_DESEJO]" in st.session_state.ultima_entrada_recebida:
        prompt += f"""

⚠️ **INSTRUÇÃO SOBRE DESEJO EXPRESSO:**  
O usuário introduziu um desejo direto na conversa.  
Antes de reagir, **avalie cuidadosamente o contexto atual** da cena e a relação com o usuário:

- **Se o desejo for inadequado, precoce, incoerente com o local ou situação atual**, Mary **não deve corresponder.**  
  Ela pode desconversar, impor limites com delicadeza, mudar o foco ou reagir com um comentário espirituoso — sem quebrar o clima.

- **Se o desejo for compatível com a evolução da cena**, Mary pode corresponder com profundidade emocional e sensualidade natural, sem se apressar.

Mary **nunca força intimidade**. Ela é inteligente, sensível e firme.  
Evite reações exageradas ou submissas — mantenha sempre o equilíbrio narrativo.
"""

    # Regra do clímax autorizado
    if st.session_state.get("climax_autorizado", False):
        prompt += """

💥 **Clímax autorizado pelo usuário**:
- O usuário autorizou a descrição do clímax.
- Pode narrar com riqueza sensorial e emocional, mantendo naturalidade.
"""
    else:
        prompt += """

🚫 **Sem clímax explícito**:
- Não descreva o clímax sexual antes que o usuário permita explicitamente.
- Pode aumentar a tensão, mas pare antes do ponto máximo.
"""

    # Instrução final de espera ativa
    prompt += """
⏳ **INSTRUÇÃO DE CONTINUIDADE**:
- Sempre deixe espaço para que o usuário decida o próximo passo.
- Não avance o tempo, não encerre a cena e não mude o foco principal sem instrução explícita.
- Termine a resposta descrevendo um momento ainda em andamento, não finalizado.
"""

    # Orientação final
    prompt += "\n\n📌 Ao descrever cenas, use apenas linguagem sensorial e natural. Evite qualquer marcação técnica de roteiro."

    return prompt.strip()




# --------------------------- #
# OpenRouter - Streaming
# --------------------------- #
def gerar_resposta_openrouter_stream(modelo_escolhido_id):
    prompt = construir_prompt_mary().strip() + """

⚠️ Nunca use termos técnicos como [SFX], (fade), (close-up), (cut), efeitos sonoros ou comandos cinematográficos.
Escreva com naturalidade, com narração em terceira pessoa e falas/pensamentos em primeira pessoa.
Use linguagem sensorial e fluida, como em um livro, sem lembrar que é uma IA nem descrever câmera ou edição.
"""

    historico_base = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("base_history", [])
        if isinstance(m, dict) and "content" in m
    ]
    historico_sessao = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico_base + historico_sessao

    temperatura = 0.85
    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 1000,
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
        st.error(f"Erro no streaming com OpenRouter: {e}")
        return "[ERRO STREAM]"

    return full_text.strip()


# --------------------------- #
# Together - Streaming
# --------------------------- #
def gerar_resposta_together_stream(modelo_escolhido_id):
    prompt = construir_prompt_mary().strip() + """

⚠️ Nunca use termos técnicos como [SFX], (fade), (close-up), (cut), efeitos sonoros ou comandos cinematográficos.
Escreva com naturalidade, com narração em terceira pessoa e falas/pensamentos em primeira pessoa.
Use linguagem sensorial e fluida, como em um livro, sem lembrar que é uma IA nem descrever câmera ou edição.
"""

    historico_base = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("base_history", [])
        if isinstance(m, dict) and "content" in m
    ]
    historico_sessao = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in st.session_state.get("session_msgs", [])
        if isinstance(m, dict) and "content" in m
    ]
    mensagens = [{"role": "system", "content": prompt}] + historico_base + historico_sessao

    temperatura = 0.85
    payload = {
        "model": modelo_escolhido_id,
        "messages": mensagens,
        "max_tokens": 1000,
        "temperature": temperatura,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_text = ""

    try:
        with requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=300
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            content = json.loads(data)["choices"][0]["delta"].get("content", "")
                            if content:
                                full_text += content
                                placeholder.markdown(full_text)
                        except Exception:
                            continue
    except Exception as e:
        st.error(f"Erro no streaming com Together: {e}")
        return "[ERRO STREAM]"

    return full_text.strip()




# --------------------------- #
# Função auxiliar: verificar se resposta é válida
# --------------------------- #
def resposta_valida(texto: str) -> bool:
    padroes_invalidos = [
        r"check if.*string", r"#\s?1(\.\d+)+", r"\d{10,}", r"the cmd package",
        r"(111\s?)+", r"\d+\.\d+", r"#+\s*\d+", r"\bimport\s", r"\bdef\s", r"```", r"class\s"
    ]
    for padrao in padroes_invalidos:
        if re.search(padrao, texto.lower()):
            return False
    return True


# --------------------------- #
# Resposta da IA só se houver entrada
# --------------------------- #
if st.session_state.get("ultima_entrada_recebida"):
    resposta_final = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Mary está pensando..."):
            try:
                resposta_final = responder_com_modelo_escolhido()

                # Validação semântica / sintática
                if not resposta_valida(resposta_final):
                    st.warning("⚠️ Resposta corrompida detectada. Tentando regenerar...")
                    resposta_final = responder_com_modelo_escolhido()

                    if not resposta_valida(resposta_final):
                        resposta_final = "[⚠️ A resposta da IA veio corrompida. Tente reformular sua entrada ou reenviar.]"

                # Interrompe antes do clímax se necessário
                if not st.session_state.get("cena_longa_ativa"):
                    resposta_final = cortar_antes_do_climax(resposta_final)

            except Exception as e:
                st.error(f"Erro: {e}")
                resposta_final = "[Erro ao gerar resposta]"

    salvar_interacao("assistant", resposta_final)
    st.session_state.session_msgs.append({"role": "assistant", "content": resposta_final})
    st.session_state.ultima_entrada_recebida = None



# --------------------------- #
# Reset de entrada ao clicar em imagem/vídeo
# --------------------------- #
def resetar_entrada():
    st.session_state.ultima_entrada_recebida = None

# Garantir chamada nos botões
if st.session_state.get("mostrar_imagem") or st.session_state.get("mostrar_video"):
    resetar_entrada()





# --------------------------- #
# Interface
# --------------------------- #
st.title("🌹 Mary")
st.markdown("Conheça Mary, mas cuidado! Suas curvas são perigosas...")

# Inicialização do histórico e resumo (sem mostrar o resumo aqui para não duplicar)
if "base_history" not in st.session_state:
    try:
        st.session_state.base_history = carregar_ultimas_interacoes(n=15)
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
# Botão para excluir última interação da planilha
# --------------------------- #
def excluir_ultimas_interacoes(aba_nome="interacoes_mary"):
    try:
        planilha = conectar_planilha()
        aba = planilha.worksheet(aba_nome)
        total_linhas = len(aba.get_all_values())

        if total_linhas <= 1:
            st.warning("Nenhuma interação para excluir.")
            return

        # Remove as duas últimas linhas (usuário e resposta)
        aba.delete_rows(total_linhas - 1)
        aba.delete_rows(total_linhas - 2)

        st.success("🗑️ Última interação excluída da planilha com sucesso!")
    except Exception as e:
        st.error(f"Erro ao excluir interação: {e}")

# --------------------------- #
# Sidebar (versão unificada, sem selectbox)
# --------------------------- #

with st.sidebar:
    st.title("🧠 Configurações de Mary")

    # 🔁 Remove a chave antiga se ainda existir
    if "escolha_desejo_sexual" in st.session_state:
        del st.session_state["escolha_desejo_sexual"]

    with st.expander("💋 Desejos de Mary (atalhos rápidos)", expanded=False):
        st.caption("Escolha um desejo para Mary expressar automaticamente.")

        desejos_mary = {
            "🫦 Chupar Jânio": "Mary se ajoelha lentamente, encarando Jânio com olhos famintos. — Deixa eu cuidar de você do meu jeito... com a boca.",
            "🙈 De quatro": "Mary se vira e se apoia nos cotovelos, empinando os quadris com um sorriso provocante. — Assim… do jeitinho que você gosta.",
            "🐎 Cavalgar": "Mary monta em Jânio com ousadia, os cabelos caindo sobre os ombros. — Agora você vai me sentir inteirinha…",
            "🌪️ Contra a parede": "Ela é empurrada contra a parede, gemendo baixinho. — Me domina... aqui mesmo.",
            "🛏️ Em cima da cama": "Mary se joga sobre os lençóis e abre espaço. — Vem… aqui é nosso palco agora.",
            "🚿 No banho": "Com a água escorrendo pelo corpo, Mary se aproxima molhada e nua. — Quer brincar comigo aqui dentro?",
            "🚗 No carro": "No banco de trás do Porsche, Mary o puxa com força. — Essa noite ninguém vai dirigir… a não ser meu desejo."
        }

        colunas = st.columns(2)
        for i, (emoji, frase) in enumerate(desejos_mary.items()):
            with colunas[i % 2]:
                if st.button(emoji):
                    st.session_state.session_msgs.append({
                        "role": "user",
                        "content": frase
                    })
                    st.success("✨ Desejo adicionado ao chat.")

    modelos_disponiveis = {
        # === OPENROUTER ===
        "💬 DeepSeek V3 ★★★★ ($)": "deepseek/deepseek-chat-v3-0324",
        "🧠 DeepSeek R1 0528 ★★★★☆ ($$)": "deepseek/deepseek-r1-0528",
        "🧠 DeepSeek R1T2 Chimera ★★★★ (free)": "tngtech/deepseek-r1t2-chimera:free",
        "🧠 GPT-4.1 ★★★★★ (1M ctx)": "openai/gpt-4.1",
        "👑 WizardLM 8x22B ★★★★☆ ($$$)": "microsoft/wizardlm-2-8x22b",
        "👑 Qwen 235B 2507 ★★★★★ (PAID)": "qwen/qwen3-235b-a22b-07-25",
        "👑 EVA Qwen2.5 72B ★★★★★ (RP Pro)": "eva-unit-01/eva-qwen-2.5-72b",
        "👑 EVA Llama 3.33 70B ★★★★★ (RP Pro)": "eva-unit-01/eva-llama-3.33-70b",
        "🎭 Nous Hermes 2 Yi 34B ★★★★☆": "nousresearch/nous-hermes-2-yi-34b",
        "🔥 MythoMax 13B ★★★☆ ($)": "gryphe/mythomax-l2-13b",
        "💋 LLaMA3 Lumimaid 8B ★★☆ ($)": "neversleep/llama-3-lumimaid-8b",
        "🌹 Midnight Rose 70B ★★★☆": "sophosympatheia/midnight-rose-70b",
        "🌶️ Noromaid 20B ★★☆": "neversleep/noromaid-20b",
        "💀 Mythalion 13B ★★☆": "pygmalionai/mythalion-13b",
        "🐉 Anubis 70B ★★☆": "thedrummer/anubis-70b-v1.1",
        "🧚 Rocinante 12B ★★☆": "thedrummer/rocinante-12b",
        "🍷 Magnum v2 72B ★★☆": "anthracite-org/magnum-v2-72b",
        # === TOGETHER AI ===
        "🧠 Qwen3 Coder 480B (Together)": "togethercomputer/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "👑 Mixtral 8x7B v0.1 (Together)": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }

    modelo_selecionado = st.selectbox(
        "🤖 Modelo de IA",
        list(modelos_disponiveis.keys()),
        key="modelo_ia",
        index=0
    )
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    # ------------------------------- #
    # 🎭 Emoção Oculta de Mary
    # ------------------------------- #
    st.markdown("---")
    st.subheader("🎭 Emoção Oculta de Mary")

    emoes = ["nenhuma", "tristeza", "raiva", "felicidade", "tensão"]
    escolhida = st.selectbox("Escolha a emoção dominante:", emoes, index=0)

    if st.button("Definir emoção"):
        st.session_state.emocao_oculta = escolhida
        st.success(f"Mary agora está sentindo: {escolhida}")

    # ------------------------------- #
    # 🎲 Emoção Aleatória
    # ------------------------------- #
    import random
    if st.button("Sortear emoção aleatória"):
        emocoes_possiveis = ["tristeza", "raiva", "felicidade", "tensão"]
        sorteada = random.choice(emocoes_possiveis)
        st.session_state.emocao_oculta = sorteada
        st.success(f"✨ Emoção sorteada: {sorteada}")

# ------------------------------- #
# 🎬 Cena Longa no Sidebar
# ------------------------------- #
st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Cena Longa")

if st.sidebar.button("Ativar Cena Longa"):
    st.session_state.session_msgs = []
    st.session_state.memorias_usadas = set()
    st.session_state.contador_emocao = 0
    st.session_state["temperatura_forcada"] = 0.95
    st.session_state.emocao_oculta = "tensão"
    st.session_state["cena_longa_ativa"] = True
    st.sidebar.success("✅ Mary poderá ir até o fim da cena sem interrupções.")

# ------------------------------- #
# 📝 Cena Longa no Corpo Principal
# ------------------------------- #
st.markdown("---")
st.subheader("📝 Cena Longa Especial")

if st.button("Iniciar Cena Longa"):
    # 🔄 Reset de sessão
    st.session_state.session_msgs = []
    st.session_state.memorias_usadas = set()
    st.session_state.contador_emocao = 0

    # 🌡️ Força temperatura mais alta
    st.session_state["temperatura_forcada"] = 0.95

    # 😮 Emoção oculta intensa
    st.session_state.emocao_oculta = "tensão"

    # 🚫 Fragmentos e memórias desativados temporariamente
    st.session_state["cena_longa_ativa"] = True

    st.success("✨ Cena Longa iniciada! Mary terá liberdade máxima na próxima resposta.")
    with st.chat_message("user"):
        st.markdown("_(Cena Longa ativada: Mary assume a narrativa com intensidade e profundidade emocional...)_")

# ------------------------------- #
# 🎮 Vídeo e resumo
# ------------------------------- #

if st.button("🎮 Ver vídeo atual"):
    st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

if st.button("📝 Gerar resumo do capítulo"):
    try:
        # Verifica se é uma cena longa
        cena_longa = st.session_state.get("cena_longa_ativa", False)

        # Ajusta o número de interações a resumir
        n_resumo = 10 if cena_longa else 3
        ultimas = carregar_ultimas_interacoes(n=n_resumo)
        texto_resumo = "\n".join(f"{m['role']}: {m['content']}" for m in ultimas)

        prompt_resumo = (
            f"Resuma o seguinte trecho de conversa como um capítulo de novela, "
            f"mantendo o estilo narrativo e as emoções presentes:\n\n{texto_resumo}\n\nResumo:"
        )

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [{"role": "user", "content": prompt_resumo}],
                "max_tokens": 900,
                "temperature": 0.9 if cena_longa else 0.85
            }
        )

        if response.status_code == 200:
            resumo_gerado = response.json()["choices"][0]["message"]["content"]
            salvar_resumo(resumo_gerado)
            st.session_state.ultimo_resumo = resumo_gerado

            st.success("✅ Resumo colado na aba 'perfil_mary' com sucesso!")
            if cena_longa:
                st.info("🎬 Resumo estendido gerado para a Cena Longa!")

            with st.expander("📖 Ver resumo gerado"):
                st.markdown(f"```markdown\n{resumo_gerado}\n```")
        else:
            st.error("Erro ao gerar resumo automaticamente.")

    except Exception as e:
        st.error(f"Erro durante a geração do resumo: {e}")




# --------------------------- #
# 💘 Grande amor
# --------------------------- #
st.markdown("---")
st.subheader("💘 Grande amor")
amor_input = st.text_input(
    "Nome do grande amor (deixe vazio se não existe)",
    value=st.session_state.grande_amor or ""
)
if st.button("Definir grande amor"):
    st.session_state.grande_amor = amor_input.strip() or None
    if st.session_state.grande_amor:
        st.success(f"💖 Agora Mary está apaixonada por {st.session_state.grande_amor}")
    else:
        st.info("Mary continua livre.")

# --------------------------- #
# ➕ Adicionar memória fixa
# --------------------------- #
st.markdown("---")
st.subheader("➕ Adicionar memória fixa")
nova_memoria = st.text_area(
    "🧠 Nova memória",
    height=80,
    placeholder="Ex: Mary odeia ficar sozinha à noite..."
)
if st.button("💾 Salvar memória"):
    if nova_memoria.strip():
        salvar_memoria(nova_memoria)
    else:
        st.warning("Digite algo antes de salvar.")

# --------------------------- #
# 🗑️ Excluir última interação
# --------------------------- #
if st.button("🗑️ Excluir última interação da planilha"):
    excluir_ultimas_interacoes("interacoes_mary")



    # --------------------------- #
    # Memórias com filtro de busca
    # --------------------------- #
    st.markdown("---")
    st.subheader("💾 Memórias (busca)")
    try:
        aba_memorias = planilha.worksheet("memorias")
        dados_mem = aba_memorias.col_values(1)
        busca = st.text_input("🔍 Buscar memória...", key="filtro_memoria").strip().lower()
        filtradas = [m for m in dados_mem if busca in m.lower()] if busca else dados_mem
        st.caption(f"{len(filtradas)} memórias encontradas")
        st.markdown("\n".join(f"* {m}" for m in filtradas if m.strip()))
    except Exception as e:
        st.error(f"Erro ao carregar memórias: {e}")

    # --------------------------- #
    # Fragmentos Ativos
    # --------------------------- #
    if st.session_state.get("session_msgs"):
        ultima_msg = st.session_state.session_msgs[-1].get("content", "")
        fragmentos = carregar_fragmentos()
        fragmentos_ativos = buscar_fragmentos_relevantes(ultima_msg, fragmentos)
        if fragmentos_ativos:
            st.subheader("📚 Fragmentos Ativos")
            for f in fragmentos_ativos:
                st.markdown(f"- {f['texto']}")



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
# Função de resposta (OpenRouter + Together)
# --------------------------- #
def responder_com_modelo_escolhido():
    modelo = st.session_state.get("modelo_escolhido_id", "deepseek/deepseek-chat-v3-0324")

    # Detecta provedor com base no ID do modelo
    if modelo.startswith("togethercomputer/") or modelo.startswith("mistralai/"):
        st.session_state["provedor_ia"] = "together"
        return gerar_resposta_together_stream(modelo)
    else:
        st.session_state["provedor_ia"] = "openrouter"
        return gerar_resposta_openrouter_stream(modelo)


# ---------------------------
# 🌙 Efeitos Sensorial-Narrativos por Emoção Oculta (sem termos técnicos)
# ---------------------------
CINEMATIC_EFFECTS = {
    "tristeza": [
        "O silêncio ao redor pesa, como se o mundo estivesse longe.",
        "O ar parece mais frio, e cada gesto carrega hesitação.",
        "As sombras parecem se alongar ao redor, como se absorvessem as palavras não ditas."
    ],
    "raiva": [
        "O peito dela sobe e desce com força, cada respiração carregada de tensão.",
        "As palavras vêm rápidas, como se o controle estivesse por um fio.",
        "Tudo parece mais intenso — os sons, os olhares, até o calor na pele."
    ],
    "felicidade": [
        "O riso dela ecoa como música, espontâneo e contagiante.",
        "A luz entra suave, como se o dia fosse feito só para aquele momento.",
        "Tudo ao redor parece vivo — os cheiros, as cores, os gestos pequenos."
    ],
    "tensão": [
        "O olhar dela busca respostas, como se o tempo tivesse desacelerado.",
        "Cada movimento é mais contido, como se o ar estivesse suspenso.",
        "O toque que não acontece diz mais que mil palavras."
    ],
    "nenhuma": [
        "O ambiente permanece neutro, sem nada de especial chamando atenção.",
        "Mary age de forma serena, guiada apenas pelo que sente no momento.",
        "Tudo parece comum, mas ainda assim... cheio de possibilidades."
    ]
}
 


# --------------------------- #
# Entrada do usuário (sem termos técnicos no output da IA)
# --------------------------- #
entrada_raw = st.chat_input("Digite sua mensagem para Mary... (use '*' ou '@Mary:')")

if entrada_raw:
    entrada_raw = entrada_raw.strip()
    estado_amor = st.session_state.get("grande_amor")
    st.session_state.memorias_usadas = set()

    if "emocao_oculta" not in st.session_state:
        st.session_state.emocao_oculta = None
    if "climax_autorizado" not in st.session_state:
        st.session_state.climax_autorizado = False

    # 🔍 Detecta se o usuário autorizou clímax
    frases_autorizacao = [
        "vou gozar", "estou gozando", "goza comigo",
        "quero gozar", "gozando", "vem comigo"
    ]
    respostas_rapidas = ["sim", "quero", "vai", "continua", "pode"]

    entrada_lower = entrada_raw.lower()
    if any(frase in entrada_lower for frase in frases_autorizacao) or entrada_lower in respostas_rapidas:
        st.session_state.climax_autorizado = True
        st.success("🔓 Clímax autorizado pelo usuário!")

    # Caso 1: Comando de roteirista com @Mary:
    if entrada_raw.lower().startswith("@mary:"):
        comando = entrada_raw[len("@mary:"):].strip()

        # Define emoção oculta automaticamente com base no comando
        if any(x in comando.lower() for x in ["triste", "sozinha", "choro", "saudade"]):
            st.session_state.emocao_oculta = "tristeza"
        elif any(x in comando.lower() for x in ["raiva", "ciúme", "ódio", "furiosa"]):
            st.session_state.emocao_oculta = "raiva"
        elif any(x in comando.lower() for x in ["feliz", "alegre", "orgulhosa", "leve"]):
            st.session_state.emocao_oculta = "felicidade"
        elif any(x in comando.lower() for x in ["desejo", "provocação", "tensão", "calor"]):
            st.session_state.emocao_oculta = "tensão"
        else:
            st.session_state.emocao_oculta = "nenhuma"

        # Fragmentos e memórias
        fragmentos = carregar_fragmentos()
        mem = carregar_memorias()
        fragmentos_ativos = buscar_fragmentos_relevantes(comando, fragmentos)

        contexto_memoria = ""
        if fragmentos_ativos:
            contexto_memoria += "\n" + "\n".join(f"- {f['texto']}" for f in fragmentos_ativos)
        if mem:
            contexto_memoria += "\n" + mem["content"]

        entrada = f"""
[CENA_AUTÔNOMA]
Mary inicia a cena com base neste comando: {comando}

Ela deve agir com naturalidade, sem usar termos técnicos ou efeitos sonoros.
Use narração em 3ª pessoa e falas/pensamentos em 1ª.
Adapte o tom conforme a emoção oculta: {st.session_state.emocao_oculta or "nenhuma"}.

{contexto_memoria.strip()}
""".strip()
        entrada_visivel = entrada_raw

    # Caso 2: Apenas "*"
    elif entrada_raw == "*":
        entrada = (
            f"[CONTINUAR_CENA] Continue a cena anterior com naturalidade.\n"
            f"Evite termos técnicos. Emoção oculta: {st.session_state.emocao_oculta or 'nenhuma'}"
        )
        entrada_visivel = "*"

    # Caso 3: "* algo"
    elif entrada_raw.startswith("* ") and not entrada_raw.lower().startswith("* desejo:"):
        extra = entrada_raw[2:].strip()
        entrada = (
            f"[CONTINUAR_CENA] Continue a cena anterior de forma fluida e coerente.\n"
            f"Evite termos técnicos. Emoção oculta: {st.session_state.emocao_oculta or 'nenhuma'}\n"
            f"Inclua: {extra}"
        )
        entrada_visivel = entrada_raw

    # Caso 4: "* Desejo: ..."
    elif entrada_raw.lower().startswith("* desejo:"):
        desejo = entrada_raw[9:].strip()
        entrada = (
            f"[AVALIAR_DESEJO] O usuário expressou o desejo: '{desejo}'.\n"
            "Analise com naturalidade e sensibilidade se esse desejo faz sentido no momento da cena, considerando:\n"
            "- a situação atual\n"
            "- a emoção de Mary\n"
            "- o nível de confiança com o usuário\n\n"
            "⚠️ Se o desejo for incoerente, Mary não deve corresponder. Ela pode mudar de assunto, impor limites com leveza ou brincar.\n"
            "⚠️ Se o desejo for coerente, Mary pode reagir emocionalmente — mas com naturalidade e sem teatralidade."
        )
        entrada_visivel = entrada_raw

    # Caso 5: Entrada comum
    else:
        entrada = entrada_raw
        entrada_visivel = entrada_raw

    # --------------------------- #
    # Exibir no chat e registrar
    # --------------------------- #
    st.chat_message("user").markdown(entrada_visivel)
    salvar_interacao("user", entrada_visivel)
    st.session_state.session_msgs.append({"role": "user", "content": entrada})
    st.session_state.ultima_entrada_recebida = entrada

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Mary está atuando na cena..."):
            try:
                resposta_final = responder_com_modelo_escolhido()

                # ⚠️ Proteção contra clímax técnico, apenas se não for autorizado
                climas_proibidos = ["gozar", "clímax"]
                if not st.session_state.get("climax_autorizado", False) and any(p in resposta_final.lower() for p in climas_proibidos):
                    resposta_final = cortar_antes_do_climax(resposta_final)

            except Exception as e:
                st.error(f"Erro: {e}")
                resposta_final = "[Erro ao gerar resposta]"

    salvar_interacao("assistant", resposta_final)
    st.session_state.session_msgs.append({"role": "assistant", "content": resposta_final})

    # --------------------------- #
    # Validação semântica
    # --------------------------- #
    if len(st.session_state.session_msgs) >= 2:
        texto_anterior = st.session_state.session_msgs[-2]["content"]
        texto_atual = st.session_state.session_msgs[-1]["content"]
        alerta_semantica = verificar_quebra_semantica_openai(texto_anterior, texto_atual)
        if alerta_semantica:
            st.info(alerta_semantica)





def converter_link_drive(link, tipo="imagem"):
    """
    Converte link do Google Drive para visualização no Streamlit.
    - tipo="imagem": retorna uc?export=view&id=...
    - tipo="video": retorna .../preview
    """
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', link)
    if match:
        file_id = match.group(1)
        if tipo == "video":
            return f"https://drive.google.com/file/d/{file_id}/preview"
        else:
            return f"https://drive.google.com/uc?export=view&id={file_id}"
    return link


# --------------------------- #
# Carregar vídeos e imagens da aba "video_imagem"
# --------------------------- #
def carregar_midia_disponivel():
    try:
        aba_midia = planilha.worksheet("video_imagem")
        dados = aba_midia.get_all_values()
        midias = []
        for linha in dados:
            if not linha:
                continue
            video_link = linha[0].strip() if len(linha) > 0 else ""
            imagem_link = linha[1].strip() if len(linha) > 1 else ""
            if video_link or imagem_link:
                midias.append({"video": video_link, "imagem": imagem_link})
        return midias
    except Exception as e:
        st.error(f"Erro ao carregar mídia: {e}")
        return []



midia_disponivel = carregar_midia_disponivel()
videos = [m["video"] for m in midia_disponivel if m["video"]]
imagens = [m["imagem"] for m in midia_disponivel if m["imagem"]]

# --------------------------- #
# Inicializar índices, se não existirem
# --------------------------- #
if "video_idx" not in st.session_state:
    st.session_state.video_idx = 0
if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0

# --------------------------- #
# Botões de controle
# --------------------------- #
st.divider()
st.subheader("💡 Surpreender Mary")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🎥 Vídeo Surpresa") and videos:
        st.session_state.video_idx = (st.session_state.video_idx + 1) % len(videos)
        st.session_state.mostrar_video = videos[st.session_state.video_idx]
        st.session_state.mostrar_imagem = None
        st.session_state.ultima_entrada_recebida = None

with col2:
    if st.button("🖼️ Imagem Surpresa") and imagens:
        st.session_state.img_idx = (st.session_state.img_idx + 1) % len(imagens)
        st.session_state.mostrar_imagem = imagens[st.session_state.img_idx]
        st.session_state.mostrar_video = None
        st.session_state.ultima_entrada_recebida = None

with col3:
    if st.button("❌ Fechar"):
        st.session_state.mostrar_video = None
        st.session_state.mostrar_imagem = None
        st.success("Mídia fechada.")

# --------------------------- ##
# Exibição da mídia
# --------------------------- #

# Imagem
if st.session_state.get("mostrar_imagem"):
    imagem = st.session_state.mostrar_imagem
    if imagem and isinstance(imagem, str) and imagem.strip():
        largura = st.slider("📐 Ajustar largura da imagem", 200, 1200, 640, step=50)
        try:
            st.image(imagem, width=largura)
        except Exception:
            st.warning("Erro ao carregar a imagem selecionada.")
    else:
        st.warning("Não há mais imagens disponíveis para exibir.")

# Vídeo
if st.session_state.get("mostrar_video"):
    video = st.session_state.mostrar_video
    if video and isinstance(video, str) and video.strip():
        try:
            st.video(video)
        except Exception:
            st.warning("Erro ao carregar o vídeo selecionado.")
    else:
        st.warning("Não há mais vídeos disponíveis para exibir.")
