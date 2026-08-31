import json
import re
import urllib.request
import urllib.error
from .character_schema import CharacterSpec


SYSTEM = """Interpret the Portuguese character description into JSON only. Keys:
base_figure, gender (female/male/androgynous), age (18-90), height, weight,
muscularity (-1..1), body_shape, skin_tone, hair, outfit, outfit_color, shoes, notes.
Use neutral defaults for unspecified values. Do not invent file paths."""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("A LLM local não retornou JSON válido.")
    return json.loads(match.group(0))


def interpret(description: str, endpoint: str = "http://127.0.0.1:11434", model: str = "qwen2.5:7b") -> CharacterSpec:
    """Call Ollama's chat API. The endpoint can also point to an OpenAI-compatible proxy."""
    base = endpoint.rstrip("/")
    for suffix in ("/api/chat", "/api", "/v1/chat/completions", "/v1"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    url = base + ("/api/chat" if "11434" in base else "/v1/chat/completions")
    if url.endswith("/api/chat"):
        payload = {"model": model, "stream": False, "format": "json", "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": description}]}
    else:
        payload = {"model": model, "temperature": 0.1, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": description}]}
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "11434" in base:
            raise RuntimeError(f"Ollama retornou 404. Confirme o modelo instalado: {model}. Detalhe: {detail}") from exc
        raise RuntimeError(f"Endpoint LLM retornou HTTP {exc.code}: {detail}") from exc
    content = result.get("message", {}).get("content") or result["choices"][0]["message"]["content"]
    return CharacterSpec.from_dict(_extract_json(content))
