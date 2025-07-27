import json
import os

# --------------------------- #
# Carregar Lorebook
# --------------------------- #
def carregar_lorebook(arquivo="lorebook_mary_completo.json"):
    """
    Carrega o arquivo JSON com os fragmentos do Lorebook.
    """
    if not os.path.exists(arquivo):
        print(f"[AVISO] Arquivo {arquivo} não encontrado.")
        return []
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o lorebook: {e}")
        return []

# --------------------------- #
# Buscar fragmentos relevantes
# --------------------------- #
def buscar_fragmentos(mensagem, fragmentos, limite=3):
    """
    Busca fragmentos do Lorebook relevantes para a mensagem atual.
    A seleção é baseada na ocorrência de gatilhos definidos no JSON.
    """
    if not mensagem:
        return []

    mensagem_lower = mensagem.lower()
    encontrados = []

    for frag in fragmentos:
        # Verifica se algum gatilho aparece na mensagem
        if any(g.lower() in mensagem_lower for g in frag.get("gatilhos", [])):
            encontrados.append(frag)

    # Ordena os fragmentos por peso (maior relevância primeiro)
    encontrados = sorted(encontrados, key=lambda x: x.get("peso", 1), reverse=True)

    return encontrados[:limite]

# --------------------------- #
# Montar texto dos fragmentos
# --------------------------- #
def montar_fragmentos_texto(fragmentos):
    """
    Constrói um texto formatado com os fragmentos encontrados.
    """
    if not fragmentos:
        return ""
    return "\n".join([f"- {frag['texto']}" for frag in fragmentos])
