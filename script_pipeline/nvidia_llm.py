"""Motor de LLM opcional via API da NVIDIA (build.nvidia.com / integrate.api.nvidia.com),
mesmo contrato JSON-in/JSON-out de `story_structure._call_ollama` -- pedido do usuario
2026-09-12 depois de `cast_characters.py` ter alucinado a aparencia de um personagem
duas vezes seguidas com os modelos Ollama locais (ver memoria de projeto). Um modelo
maior/mais capaz na nuvem e' um caminho alternativo pra extracao de descritores mais
fiel ao roteiro, sem trocar toda a cadeia de --engine.

Selecao: qualquer `--engine`/`model` que comece com "nvidia/" e' roteado pra ca em vez
do Ollama local -- ver `story_structure._call_ollama`, o unico ponto de despacho.

Chave: NVIDIA_API_KEY, lida de variavel de ambiente (carregada de `.env` na raiz do
projeto via python-dotenv se presente -- NUNCA commitada, `.env` esta no .gitignore).
"""
from __future__ import annotations

import json
import os
import re
import time

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def call_nvidia_text(system: str, user: str, model: str, *, log=print, temperature: float = 0.0,
                      max_tokens: int = 8192) -> str | None:
    """Mesma API, mas devolve o CONTEUDO CRU (texto livre), sem tentar
    interpretar como JSON -- pra chamadas de reformatacao/conversao de
    texto (ex.: `prose_to_screenplay.py`), onde o resultado esperado e'
    prosa/roteiro, nao um objeto estruturado. `None` em falha, igual
    `call_nvidia`."""
    _ensure_env_loaded()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        log("[nvidia_llm] NVIDIA_API_KEY nao configurada (.env ou variavel de ambiente).")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log("[nvidia_llm] pacote 'openai' nao instalado (pip install openai).")
        return None

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    for tentativa in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, top_p=0.95, max_tokens=max_tokens, stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            log(f"[nvidia_llm] erro ({type(e).__name__}): {e} (tentativa {tentativa + 1}/3).")
            if tentativa < 2:
                time.sleep(2)
    return None


def call_nvidia(system: str, user: str, model: str, *, log=print, temperature: float = 0.2,
                 max_tokens: int = 4096) -> dict | None:
    """Mesmo contrato de `story_structure._call_ollama`: devolve o JSON do
    content da resposta, ou None em falha (nunca lanca -- quem chama ja
    trata None como "usar o fallback deterministico").

    `model`: nome completo do modelo tal como aparece no catalogo da API
    (ex.: "nvidia/nemotron-3-ultra-550b-a55b") -- o prefixo "nvidia/" NAO e'
    um marcador de roteamento pra remover, e' parte do id real do modelo
    (confirmado contra `client.models.list()`); manda pra API exatamente
    como recebido. `_call_ollama` usa esse MESMO prefixo so pra decidir
    despachar pra ca em vez do Ollama -- coincidencia conveniente, nao
    contrato deste modulo."""
    _ensure_env_loaded()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        log("[nvidia_llm] NVIDIA_API_KEY nao configurada (.env ou variavel de ambiente).")
        return None

    modelo_real = model

    try:
        from openai import OpenAI
    except ImportError:
        log("[nvidia_llm] pacote 'openai' nao instalado (pip install openai).")
        return None

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    # JSON pedido via instrucao no system prompt -- a API da NVIDIA (OpenAI-
    # compatible) nem sempre aceita response_format=json_object dependendo do
    # modelo, entao o contrato aqui e' o MESMO fallback regex que
    # `_call_ollama` ja usa pra respostas que vem com texto em volta do JSON.
    system_com_json = system + "\n\nResponda APENAS com o objeto JSON pedido, sem texto antes ou depois."

    corpo = None
    for tentativa in range(3):
        try:
            completion = client.chat.completions.create(
                model=modelo_real,
                messages=[{"role": "system", "content": system_com_json},
                          {"role": "user", "content": user}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                stream=False,
            )
            corpo = completion.choices[0].message.content or ""
            break
        except Exception as e:
            log(f"[nvidia_llm] erro ({type(e).__name__}): {e} (tentativa {tentativa + 1}/3).")
            if tentativa < 2:
                time.sleep(2)
    if corpo is None:
        return None
    if not corpo.strip():
        log("[nvidia_llm] resposta vazia.")
        return None
    try:
        return json.loads(corpo)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", corpo, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        log(f"[nvidia_llm] resposta nao era JSON valido: {corpo[:300]}")
        return None
