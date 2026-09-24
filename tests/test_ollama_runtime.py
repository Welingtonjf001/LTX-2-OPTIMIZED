import io
import json

from script_pipeline import ollama_runtime


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_unload_all_lists_resident_models_and_sets_keep_alive_zero(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if isinstance(request, str):
            return _Response(json.dumps({"models": [
                {"name": "qwen3-vl:30b"},
                {"model": "qwen3.6-35b-a3b:latest"},
            ]}).encode())
        return _Response(b'{"done":true}')

    monkeypatch.setattr(ollama_runtime.urllib.request, "urlopen", fake_urlopen)
    messages = []
    unloaded = ollama_runtime.unload_all(log=messages.append)

    assert unloaded == ["qwen3-vl:30b", "qwen3.6-35b-a3b:latest"]
    payloads = [json.loads(item[0].data) for item in requests[1:]]
    assert payloads == [
        {"model": "qwen3-vl:30b", "prompt": "", "stream": False, "keep_alive": 0},
        {"model": "qwen3.6-35b-a3b:latest", "prompt": "", "stream": False, "keep_alive": 0},
    ]


def test_cleanup_failure_does_not_raise(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(ollama_runtime.urllib.request, "urlopen", unavailable)
    assert ollama_runtime.unload_all(log=lambda _message: None) == []


def test_specific_cleanup_never_loads_an_absent_model(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(b'{"models":[]}')

    monkeypatch.setattr(ollama_runtime.urllib.request, "urlopen", fake_urlopen)
    assert ollama_runtime.unload_models(["qwen3-vl:30b"], log=lambda _message: None) == []
    assert requests == ["http://127.0.0.1:11434/api/ps"]
