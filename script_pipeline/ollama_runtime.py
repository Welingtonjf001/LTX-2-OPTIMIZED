"""Gerencia somente a residencia dos modelos no servidor Ollama local.

O servidor continua no ar. ``unload_*`` envia ``keep_alive: 0`` para que o
Ollama libere imediatamente RAM/VRAM ocupada pelos modelos carregados.
Todas as funcoes sao best-effort: uma limpeza nunca deve esconder o codigo de
saida verdadeiro do pipeline quando o Ollama ja estiver desligado.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _base_url() -> str:
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def loaded_models(*, timeout: float = 5.0) -> list[str]:
    """Retorna as tags atualmente residentes em memoria, via ``/api/ps``."""
    with urllib.request.urlopen(f"{_base_url()}/api/ps", timeout=timeout) as response:
        payload = json.load(response)
    names: list[str] = []
    for item in payload.get("models", []):
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def unload_model(model: str, *, timeout: float = 15.0) -> bool:
    """Libera uma tag sem parar o servidor Ollama."""
    body = json.dumps({"model": model, "prompt": "", "stream": False,
                       "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        f"{_base_url()}/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return True


def unload_models(models: Iterable[str] | None = None, *, timeout: float = 15.0,
                  log=print) -> list[str]:
    """Libera modelos especificos ou todos os que ``/api/ps`` listar.

    Falhas sao registradas e ignoradas para preservar o resultado da tarefa
    principal. A lista devolvida contem apenas as tags cuja requisicao de
    descarregamento foi aceita.
    """
    try:
        resident = loaded_models(timeout=min(timeout, 5.0))
    except Exception as exc:
        log(f"[ollama] limpeza ignorada: servidor indisponivel ({type(exc).__name__}: {exc}).")
        return []

    if models is None:
        requested = resident
    else:
        wanted = {str(name).strip() for name in models if str(name).strip()}
        wanted_latest = {name if ":" in name else f"{name}:latest" for name in wanted}
        # Nunca mande /api/generate para uma tag ausente: dependendo da versao
        # do Ollama isso poderia carregar o modelo apenas para descarrega-lo.
        requested = [name for name in resident if name in wanted or name in wanted_latest]
    unique = list(dict.fromkeys(requested))
    unloaded: list[str] = []
    for model in unique:
        try:
            unload_model(model, timeout=timeout)
            unloaded.append(model)
            log(f"[ollama] modelo descarregado: {model}")
        except Exception as exc:
            log(f"[ollama] nao foi possivel descarregar {model} "
                f"({type(exc).__name__}: {exc}).")
    if not unique:
        log("[ollama] nenhum modelo carregado.")
    return unloaded


def unload_all(*, timeout: float = 15.0, log=print) -> list[str]:
    """Atalho explicito usado pelo encerramento do pipeline e da WebUI."""
    return unload_models(None, timeout=timeout, log=log)
