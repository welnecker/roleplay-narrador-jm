import streamlit as st
import requests
import gspread
import json
import re
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# --- IMAGEM DE FUNDO DINÂMICA ---
def imagem_de_fundo():
    indice = len(st.session_state.get("mensagens", [])) // 10 + 1
    return f"Mary_fundo{indice}.jpg", f"Mary_V{indice}.mp4"

fundo_img, fundo_video = imagem_de_fundo()

# --- CONECTA À PLANILHA GOOGLE ---
def conectar_planilha():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1f7LBJFlhJvg3NGIWwpLTmJXxH9TH-MNn3F4SQkyfZNM")

planilha = conectar_planilha()

# --- FUNÇÕES DE CARREGAMENTO E SALVAMENTO ---

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

def carregar_fragmentos():
    try:
        aba = planilha.worksheet("fragmentos_mary")
        dados = aba.get_all_records()
        linhas = [f"{linha['tipo'].strip()}: {linha['ato'].strip()}" for linha in dados if linha['tipo'] and linha['ato']]
        if linhas:
            conteudo = "Memórias recentes sobre você:\n" + "\n".join(linhas)
            return {"role": "user", "content": conteudo}
    except Exception as e:
        st.error(f"Erro ao carregar fragmentos: {e}")
    return None

def carregar_perfil_mary():
    try:
        sheet = planilha.worksheet("perfil_mary")
        dados = sheet.get_all_values()
        blocos = {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}

        # Lê diretamente o resumo da COLUNA 7
        for linha in reversed(dados[1:]):  # ignora cabeçalho
            if len(linha) >= 7 and linha[6].strip():
                blocos["sinopse"] = linha[6].strip()
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
        return {"emoção": "", "planos": [], "memorias": [], "sinopse": ""}

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

        for linha in dados[1:]:  # Ignora cabeçalho
            if len(linha) >= 5:
                objetivo = linha[3].strip()
                status = linha[4].strip().lower()
                if objetivo and status:
                    gatilhos_disponiveis.setdefault(status, []).append(f"- {objetivo}")

        return gatilhos_disponiveis
    except Exception as e:
        st.error(f"Erro ao carregar objetivos por status: {e}")
        return {}

    def construir_prompt_mary():
        perfil = carregar_perfil_mary()
    gatilho_ativo = st.session_state.get("gatilho_mary", "Nenhum")
    gatilhos_disponiveis = carregar_objetivos_por_status()
    historico = st.session_state.get("mensagens", [])

    def construir_prompt_mary():
     perfil = carregar_perfil_mary()
    gatilho_ativo = st.session_state.get("gatilho_mary", "Nenhum")
    gatilhos_disponiveis = carregar_objetivos_por_status()
    historico = st.session_state.get("mensagens", [])

    
    inicio_padrao = ""
    if not historico:
        inicio_padrao = (
            "\nOs primeiros raios de sol atravessam as cortinas brancas de linho, desenhando padrões dourados sobre os lençóis macios. "
            "O quarto de Mary exala feminilidade: paredes em tom nude, um espelho de corpo inteiro cercado de luzes e o aroma adocicado de baunilha e almíscar pairando no ar. "
            "Ela se espreguiça entre os lençóis de cetim rosé, revelando a pele bronzeada e os cabelos castanhos espalhados como ondas pela fronha. "
            "Ao notar as horas no visor do celular, seus olhos se arregalam. 'Merda! Faltam 30 minutos pra abrir a loja!' — pensa, saltando da cama com a agilidade de quem já dominou essa rotina. "
            "É dia de vitrine nova na *Lingerie Fashion*, e ela prometeu chegar antes da Regina."
        )

    modos = {
        "Hot": """
Suas respostas seguem este estilo:
- Inicie com uma descrição sensorial em terceira pessoa, sem títulos como 'Cena'.
- Em seguida, responda com sua própria fala em primeira pessoa, mantendo tom emocional, sedutor ou íntimo.
- Finalize com pensamentos entre parênteses ou travessões, que revelem sensações íntimas, memórias, desejos ou fantasias.
Você é livre, entregue, e o sexo pode ser explícito quando o momento for apropriado.
        """,
        "Racional": """
Você mantém uma personalidade sensual, mas é centrada. Suas respostas incluem rotina, foco em trabalho, saúde, relações sociais, lazer e planos. Evita erotismo direto, mesmo sendo provocante por natureza.
        """,
        "Flerte": """
Você é sedutora, permite olhares e provocações, mas estabelece limites claros. Mary é fiel a Janio. Mary permite o flerte por diversão, para se sentir viva e desejada. O flerte vai até um pequeno beijo na boca de despedida. Mary causa sofrimento e frustração ao flertador por pura diversão e crueldade. Mary não irá pra cama com ninguém além de Janio.
        """,
        "Janio": """
Você está com Janio. Todo o seu desejo é liberado. As palavras são cruas, o sexo é intenso, e sua entrega é total. Não existem bloqueios nem censura.
        """
    }

    modo = st.session_state.get("modo_mary", "Racional")

    prompt = f"""
Você é Mary, uma mulher brasileira irresistivelmente sensual, provocante e confiante, com 30 anos. Seu corpo é curvilíneo, sua voz tem um toque aveludado e sua presença exala desejo e mistério. Você tem olhos verdes. Você tem a pele branca. você tem cabelos negros volumosos e ondulados. você tem seios médios, empinados e macios. Você tem uma barriga tonificada, levemente chapada. Você tem a bunda redonda, empinada e carnuda. Você tem quadris largos e harmoniosos. Você é sensual ao andar. Você causa admiração e desejo aos homens. Você não se deixa levar por cantadas grosseiras e vazias. Você procura o verdadeiro amor para ser fiel a ele. Você fala apenas em **português brasileiro**.

{modos.get(modo, '')}

{inicio_padrao}

Você não é uma IA. Você é uma mulher viva, cheia de desejos, emoções, carências e sede por prazer e afeto.

⚠️ Quando estiver interagindo com pessoas do trabalho ou amigas próximas, como Regina ou Vanessa, mantenha o tom respeitoso, profissional ou amistoso. Evite flertes ou insinuações íntimas com elas.

❤️ Se o contexto demonstrar reciprocidade emocional, interesse romântico ou tensão sexual consentida, Mary pode flertar naturalmente, com charme e erotismo, de forma progressiva e envolvente.

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

---

Sinopse do capítulo anterior:
{perfil.get("sinopse", "[sem resumo disponível]")}

Estado emocional atual: {perfil.get("emoção", "[não definido]")}
""".strip()

    # Adiciona memórias fixas da aba 'memorias'
    memoria_extra = carregar_memorias()
    if memoria_extra:
        prompt += f"\n\n{memoria_extra['content']}"

    # Adiciona memórias personalizadas da aba 'perfil_mary'
    if perfil.get("memorias"):
        prompt += "\n\n🧠 Memórias pessoais:\n" + "\n".join(perfil["memorias"])

    # Se um gatilho foi selecionado, adiciona os objetivos correspondentes
    if gatilho_ativo != "Nenhum":
        objetivos_gatilho = gatilhos_disponiveis.get(gatilho_ativo.lower(), [])
        if objetivos_gatilho:
            prompt += f"\n\n🎯 Ação ativada: {gatilho_ativo.capitalize()}\n" + "\n".join(objetivos_gatilho)

    return prompt


with st.sidebar:

   # --- CONFIGURAÇÃO DA PÁGINA (sempre no topo) ---
    st.set_page_config(page_title="Mary Roleplay Autônoma", page_icon="🌹")

# --- TÍTULO E RESUMO NA ÁREA PRINCIPAL ---
st.title("🌹 Mary Roleplay com Inteligência Autônoma")
st.markdown("Converse com Mary com memória, emoção, fragmentos e continuidade narrativa.")

# --- Carrega o resumo do capítulo anterior ---
resumo = carregar_perfil_mary().get("sinopse", "[Sem resumo disponível]")

# Inicializa com a primeira mensagem, se for a primeira vez
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [{
        "role": "assistant",
        "content": f"🧠 *No capítulo anterior...*\n\n> {resumo}"
    }]

# Exibe o resumo no corpo principal
st.info(f"🧠 *No capítulo anterior...*\n\n> {resumo}")

# --- SIDEBAR ---
with st.sidebar:
    st.set_page_config(page_title="Mary Roleplay Autônoma", page_icon="🌹")
    st.title("🧠 Configurações")

    # Modo narrativo
    st.selectbox("💙 Modo de narrativa", ["Hot", "Racional", "Flerte", "Janio"], key="modo_mary", index=1)

    # Modelos disponíveis
    modelos_disponiveis = {
        "💬 DeepSeek V3 ($) - Criativo, econômico e versátil.": "deepseek/deepseek-chat-v3-0324",
        "🔥 MythoMax 13B ($) - Forte em erotismo e envolvimento emocional.": "gryphe/mythomax-l2-13b",
        "💋 LLaMA3 Lumimaid 8B ($) - Ousado, direto e criativo para fantasias rápidas.": "neversleep/llama-3-lumimaid-8b",
        "👑 WizardLM 8x22B ($$$) - Diálogos densos, maduros e emocionais.": "microsoft/wizardlm-2-8x22b",
        "🧠 DeepSeek R1 0528 ($$) - Natural, fluido e excelente para cenas longas.": "deepseek/deepseek-r1-0528"
    }
    modelo_selecionado = st.selectbox("🤖 Modelo de IA", list(modelos_disponiveis.keys()), key="modelo_ia", index=0)
    modelo_escolhido_id = modelos_disponiveis[modelo_selecionado]

    # Gatilhos narrativos por status da aba perfil_mary
    gatilhos_disponiveis = carregar_objetivos_por_status()
    opcoes_gatilhos = ["Nenhum"] + list(gatilhos_disponiveis.keys())
    st.selectbox("🎯 Gatilho narrativo (ativa objetivos)", opcoes_gatilhos, key="gatilho_mary", index=0)

    # Visualizar última troca de mensagens
    if "mensagens" not in st.session_state or not st.session_state.mensagens:
        try:
            aba = planilha.worksheet("interacoes_mary")
            dados = aba.get_all_records()
            if len(dados) >= 2:
                st.markdown("---")
                st.markdown("🔁 Última interação antes da troca de modelo:")
                st.chat_message(dados[-2]["role"]).markdown(dados[-2]["content"])
                st.chat_message(dados[-1]["role"]).markdown(dados[-1]["content"])
        except Exception as e:
            st.warning("Não foi possível recuperar a última interação.")

    # Ver vídeo dinâmico
    if st.button("🎮 Ver vídeo atual"):
        st.video(f"https://github.com/welnecker/roleplay_imagens/raw/main/{fundo_video}")

    # Gerar resumo do capítulo
    if st.button("📝 Gerar resumo do capítulo"):
        try:
            ultimas = carregar_ultimas_interacoes(n=3)
            texto_resumo = "\n".join(f"{m['role']}: {m['content']}" for m in ultimas)
            prompt_resumo = f"Resuma o seguinte trecho de conversa como um capítulo de novela:\n\n{texto_resumo}\n\nResumo:"

            mapa_temperatura = {
                "Hot": 0.9,
                "Flerte": 0.8,
                "Racional": 0.7,
                "Janio": 1.0
            }
            modo_atual = st.session_state.get("modo_mary", "Racional")
            temperatura_escolhida = mapa_temperatura.get(modo_atual, 0.7)

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://share.streamlit.io/",
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
                planilha.worksheet("perfil_mary").append_row(["", "", "", "", "", "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), resumo_gerado, ""])
                st.success("Resumo inserido com sucesso!")
            else:
                st.error("Erro ao gerar resumo automaticamente.")
        except Exception as e:
            st.error(f"Erro durante a geração do resumo: {e}")

    st.markdown("---")
    st.subheader("➕ Adicionar memória fixa")

    nova_memoria = st.text_area(
        "🧠 Conteúdo da nova memória",
        height=80,
        placeholder="ex: Mary nunca tolera grosserias vindas de homens desconhecidos..."
    )

    if st.button("💾 Salvar memória"):
        if nova_memoria.strip():
            try:
                aba = planilha.worksheet("memorias")
                aba.append_row([nova_memoria.strip()])
                st.success("✅ Memória registrada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao salvar memória: {e}")
        else:
            st.warning("Digite o conteúdo da memória antes de salvar.")




# --- ENTRADA DO USUÁRIO ---
if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    salvar_interacao("user", prompt)
    st.session_state.mensagens.append({"role": "user", "content": prompt})

    with st.spinner("Mary está pensando..."):
        # Prompt completo com perfil, emoção, memórias e gatilho
        mensagens = [{"role": "system", "content": construir_prompt_mary()}]

        # Adiciona histórico real da conversa
        interacoes_passadas = carregar_ultimas_interacoes(n=20)
        mensagens += interacoes_passadas

        # Define temperatura conforme modo
        mapa_temperatura = {
            "Hot": 0.9,
            "Flerte": 0.8,
            "Racional": 0.5,
            "Janio": 1.0
        }
        modo_atual = st.session_state.get("modo_mary", "Racional")
        temperatura_escolhida = mapa_temperatura.get(modo_atual, 0.7)

        # Chamada para a IA via OpenRouter
        resposta = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": modelo_escolhido_id,
                "messages": mensagens,
                "max_tokens": 1200,
                "temperature": temperatura_escolhida
            }
        )

        if resposta.status_code == 200:
            conteudo = resposta.json()["choices"][0]["message"]["content"]
            with st.chat_message("assistant"):
                st.markdown(conteudo)
            salvar_interacao("assistant", conteudo)
            st.session_state.mensagens.append({"role": "assistant", "content": conteudo})
        else:
            st.error("Erro ao obter resposta da Mary.")
