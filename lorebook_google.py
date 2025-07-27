# lorebook_google.py
"""
Carrega fragmentos do Lorebook a partir de uma aba do Google Sheets.
Cabeçalho esperado da aba:
    personagem | texto | gatilhos | peso
"""

import json
import os
import time
from typing import List, Dict, Optional

# Fallback local
from lorebook import carregar_lorebook as carregar_lorebook_local

# ---- CONFIG ----
DEFAULT_SHEET_ID = os.getenv("LOREBOOK_SHEET_ID", "")  # defina via env ou st.secrets
DEFAULT_WORKSHEET = os.getenv("LOREBOOK_SHEET_TAB", "fragmentos_mary")
CACHE_TTL_SECONDS = 60  # recarrega a cada 1 minuto

# cache simples em memória
_cache = {"data": None, "ts": 0}


def _get_credentials_from_streamlit():
    """Tenta pegar credenciais do st.secrets['gcp_service_account']."""
    try:
        import streamlit as st
        from google.oauth2.service_account import Credentials

        info = dict(st.secrets["gcp_service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return creds
    except Exception as e:
        print(f"[lorebook_google] Falha ao carregar credenciais do Streamlit: {e}")
        return None


def _get_credentials_from_env():
    """Alternativa: pegar o JSON da conta de serviço via variável de ambiente."""
    try:
        from google.oauth2.service_account import Credentials
        raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
        if not raw:
            return None
        info = json.loads(raw)
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        return Credentials.from_service_account_info(info, scopes=scopes)
    except Exception as e:
        print(f"[lorebook_google] Falha ao carregar credenciais do ENV: {e}")
        return None


def _get_gspread_client():
    """Cria client gspread usando credenciais do Streamlit ou ENV."""
    creds = _get_credentials_from_streamlit() or _get_credentials_from_env()
    if not creds:
        return None

    try:
        import gspread
        return gspread.authorize(creds)
    except Exception as e:
        print(f"[lorebook_google] Erro ao autorizar gspread: {e}")
        return None


def _sheet_to_fragments(rows: List[List[str]]) -> List[Dict]:
    """
    Converte as linhas da planilha em uma lista de dicts:
    personagem | texto | gatilhos | peso
    """
    fragments = []
    header = [h.strip().lower() for h in rows[0]]
    idx_personagem = header.index("personagem")
    idx_texto = header.index("texto")
    idx_gatilhos = header.index("gatilhos")
    idx_peso = header.index("peso")

    for r in rows[1:]:
        if not any(r):  # linha vazia
            continue
        try:
            personagem = r[idx_personagem].strip()
            texto = r[idx_texto].strip()
            gatilhos = [g.strip() for g in r[idx_gatilhos].split(",") if g.strip()]
            peso = int(r[idx_peso])
            fragments.append(
                {
                    "personagem": personagem,
                    "texto": texto,
                    "gatilhos": gatilhos,
                    "peso": peso,
                }
            )
        except Exception as e:
            print(f"[lorebook_google] Linha inválida ignorada: {r} | erro: {e}")
            continue

    return fragments


def carregar_lorebook_google(
    sheet_id: Optional[str] = None,
    worksheet_name: Optional[str] = None,
    fallback_json_path: str = "lorebook_mary_completo.json",
) -> List[Dict]:
    """
    Tenta carregar do Google Sheets. Se falhar, faz fallback para o JSON local.
    Usa cache com TTL (CACHE_TTL_SECONDS).
    """
    global _cache
    now = time.time()

    if (
        _cache["data"] is not None
        and (now - _cache["ts"]) < CACHE_TTL_SECONDS
    ):
        return _cache["data"]

    sheet_id = sheet_id or DEFAULT_SHEET_ID
    worksheet_name = worksheet_name or DEFAULT_WORKSHEET

    if not sheet_id:
        print("[lorebook_google] LOREBOOK_SHEET_ID não definido. Usando fallback local.")
        data = carregar_lorebook_local(fallback_json_path)
        _cache = {"data": data, "ts": now}
        return data

    client = _get_gspread_client()
    if not client:
        print("[lorebook_google] Sem client gspread. Usando fallback local.")
        data = carregar_lorebook_local(fallback_json_path)
        _cache = {"data": data, "ts": now}
        return data

    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(worksheet_name)
        rows = ws.get_all_values()
        data = _sheet_to_fragments(rows)
        _cache = {"data": data, "ts": now}
        return data
    except Exception as e:
        print(f"[lorebook_google] Falhou ao ler Sheets: {e}. Fallback local.")
        data = carregar_lorebook_local(fallback_json_path)
        _cache = {"data": data, "ts": now}
        return data
