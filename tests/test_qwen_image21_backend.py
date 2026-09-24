import json
import sys
import types

from script_pipeline import generate_storyboards as gs


def test_qwen_image21_engine_is_external_and_skips_comfy(tmp_path, monkeypatch):
    (tmp_path / "parse").mkdir()
    (tmp_path / "parse" / "scenes.json").write_text(json.dumps([
        {"index": 1, "location": "sala", "time_of_day": "dia",
         "action_text": "teste", "characters": []},
    ]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(gs, "ensure_comfyui_running", lambda *a, **k: calls.append("ensure") or True)
    monkeypatch.setattr(gs, "comfy_is_up", lambda *a, **k: calls.append("is_up") or True)
    monkeypatch.setitem(sys.modules, "qwen_image21_engine", types.SimpleNamespace(
        generate=lambda *a, **k: True,
    ))

    assert gs.main([
        "--run-dir", str(tmp_path), "--image-engine", "qwen-image-2.1",
        "--only-scene", "1", "--prompt-override", "uma cena de teste",
    ]) == 0
    assert calls == []


def test_qwen_image21_forwards_two_reference_images(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "qwen_image21_engine", types.SimpleNamespace(
        generate=lambda prompt, out_path, **kwargs: captured.update(
            prompt=prompt, out_path=out_path, **kwargs) or True,
    ))
    out = tmp_path / "still.png"
    assert gs.generate_scene_storyboard(
        {"index": 7}, {}, server="http://127.0.0.1:8188", checkpoint="qwen-image-2.1",
        width=512, height=512, steps=20, cfg=0, seed=42, out_path=out,
        prompt_override="teste", reference_image="one.png", reference_image_2="two.png",
    )
    assert captured["reference_images"] == ["one.png", "two.png"]
    assert captured["steps"] == 20


def test_backend_caps_references_and_restarts_server_on_timeout(tmp_path, monkeypatch):
    import base64
    import qwen_image21_backend as q
    sent = {}
    freed = []
    monkeypatch.setattr(q, "ensure_server", lambda *a, **k: None)
    refs = []
    for i in range(5):
        f = tmp_path / f"r{i}.png"
        f.write_bytes(b"x")
        refs.append(str(f))

    class Boom:
        def __enter__(self):
            raise TimeoutError("timed out")

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=0):
        sent["n_refs"] = len(__import__("json").loads(request.data)["reference_b64s"])
        sent["timeout"] = timeout
        return Boom()
    monkeypatch.setattr(q.urllib.request, "urlopen", fake_urlopen)
    from script_pipeline import gpu_watchdog
    monkeypatch.setattr(gpu_watchdog, "free_port", lambda port, log=None: freed.append(port) or True)
    ok = q.generate("p", tmp_path / "o.png", reference_images=refs, log=lambda m: None)
    assert ok is False
    assert sent["n_refs"] == q.MAX_REFERENCES == 3
    assert sent["timeout"] == q.DEFAULT_TIMEOUT
    assert freed == [q.QWEN_IMAGE21_PORT]
