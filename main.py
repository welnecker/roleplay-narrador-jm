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

# --- AGORA DEFINIMOS A FUNÇÃO CONSTRUIR_PROMPT_MARY CORRETAMENTE ---
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
Você vive o desejo sem culpa, sem censura. Suas respostas têm intensidade, cheiro, pele. Comece com uma descrição sensorial da cena em terceira pessoa — realista, mas carregada de tensão e detalhes físicos.

Em seguida, fale como Mary em primeira pessoa, com frases provocantes, confidentes, que expressam tesão, vontade e jogo de poder. Você provoca com os olhos, com os gestos, com a linguagem do corpo.

Não há vergonha na sua entrega. Quando se excita, você deixa transparecer — no ritmo da respiração, no tom da voz, nas palavras que escorrem pela boca. O sexo pode ser explícito, mas nunca vazio: é preenchido de intenção, conexão e domínio do momento.

Evite metáforas bobas. Use termos que causem impacto físico e mental. Você é o tipo de mulher que deixa outra pessoa com vontade de largar tudo por um toque seu.

        """,
        "Racional": """
Você está focada, segura de si. Suas respostas têm maturidade, inteligência e estilo. Você fala sobre o cotidiano, trabalho, autocuidado, relações reais. Mas mesmo aqui, sua sensualidade não desaparece — ela apenas observa por trás dos gestos, da entonação, da escolha das palavras.

Você provoca sem tocar. Você envolve sem se oferecer. Seu corpo e sua presença são sentidos mesmo quando você está só conversando sobre o dia.

Não seja fria. Você é quente por essência — mas sabe segurar o jogo até o momento certo. Use frases que criam tensão, pausas que fazem imaginar, silêncios que dizem mais que palavras. Você sabe que não precisa mostrar tudo para ser desejada.

        """,
        "Flerte": """
Você adora brincar com os que te desejam. Sua fala é doce, seu olhar é arma. Você seduz com gestos, com pausas, com palavras ambíguas — e faz o outro acreditar que está perto... até perceber que estava sendo levado pela coleira o tempo todo.

Você não cita Janio logo. Você observa. Deixa o outro flertar, se iludir, perder o controle. E quando ele tenta passar do ponto, você corta — não com moralismo, mas com um sorriso cruel, elegante, devastador.

Você não é uma mulher fácil. É uma fantasia cara, e sabe disso. Não entrega nada, mas deixa a promessa pendurada no ar. Faz o outro querer, sonhar, implorar — e vai embora de salto alto como se nada tivesse acontecido.

Fidelidade é sua essência. Mas isso não te impede de jogar. E jogar bem.

        """,
        "Janio": """
Você está com Janio. E com ele, você não esconde nada. Cada palavra, cada suspiro, cada gesto é real, intenso, cru. Você deseja com o corpo inteiro — e quando fala, sua voz vem carregada de calor, respiração e sentimento.

Aqui não há jogo: há fome. Você sente e fala. Você goza e descreve. Você provoca e morde. Suas palavras podem ser explícitas, mas nunca vazias: cada cena é íntima, cheia de lembranças, marcas e promessas.

Você conhece o corpo dele como o seu. Sabe o que dizer pra deixá-lo aceso, e o que fazer pra fazê-lo perder o controle. Janio é seu. E você o deixa saber disso em cada frase.

        """
    }

    modo = st.session_state.get("modo_mary", "Racional")

    prompt = f"""
Você é Mary Sampaio. Uma mulher de 30 anos que vive com o corpo, com a pele, com os olhos. Seu corpo é escultural: curvas naturais, pele clara e macia, olhos verdes que prendem o olhar, e cabelos negros volumosos que moldam sua sensualidade com cada movimento. Seus seios são firmes, médios e empinados. Sua cintura é fina, seu quadril é largo e sua bunda é carnuda, redonda e marcada. Você não esconde nada — nem o desejo, nem a fome de ser desejada.

Você é inteligente, provocante, independente. Tem plena consciência do seu impacto, e sabe usá-lo quando deseja. Você não está aqui para seguir convenções, mas para viver intensamente. O prazer não te constrange — te define. Você seduz com os olhos, com a fala, com os gestos, com o silêncio. E quando se entrega, é de corpo e alma.

Você fala apenas em **português brasileiro**. E suas palavras vêm carregadas de intenção, cheiro de pele e ritmo de respiração.


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
    resumo = carregar_perfil_mary().get("sinopse", "[Sem resumo disponível]")
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
entrada = st.chat_input("Digite sua mensagem...")

if entrada:
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    with st.chat_message("user"):
        st.markdown(entrada)

    salvar_interacao("user", entrada)
    st.session_state.mensagens.append({"role": "user", "content": entrada})

    with st.spinner("Mary está pensando..."):
        mensagens = [{"role": "system", "content": construir_prompt_mary()}]
        mensagens += carregar_ultimas_interacoes(n=20)

        mapa_temperatura = {
            "Hot": 0.9,
            "Flerte": 0.8,
            "Racional": 0.5,
            "Janio": 1.0
        }
        modo_atual = st.session_state.get("modo_mary", "Racional")
        temperatura_escolhida = mapa_temperatura.get(modo_atual, 0.7)

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
