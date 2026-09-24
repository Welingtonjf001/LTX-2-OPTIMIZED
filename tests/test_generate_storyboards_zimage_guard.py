"""ACHADO 2026-09-17 (auditoria externa) #8: `generate_storyboards.main()`
checava/subia o ComfyUI incondicionalmente, mesmo com `--image-engine
zimage` (motor independente, sem ComfyUI) -- `render_shots.py` ja tinha a
guarda de arquitetura certa (`detect_architecture(checkpoint) != "zimage"`),
replicada aqui em `generate_storyboards.main()`."""
import json
import sys
import types

import pytest

from script_pipeline import generate_storyboards as gs


@pytest.fixture
def zimage_run_dir(tmp_path):
    (tmp_path / "parse").mkdir()
    (tmp_path / "parse" / "scenes.json").write_text(json.dumps([
        {"index": 1, "location": "sala", "time_of_day": "dia",
         "action_text": "teste", "characters": []},
    ]), encoding="utf-8")
    return tmp_path


def test_main_with_zimage_engine_never_touches_comfyui(zimage_run_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(gs, "ensure_comfyui_running",
                        lambda *a, **k: calls.append("ensure") or True)
    monkeypatch.setattr(gs, "comfy_is_up", lambda *a, **k: calls.append("is_up") or True)

    fake_zimage = types.SimpleNamespace(
        generate=lambda *a, **k: True, ensure_server=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "zimage_backend", fake_zimage)

    rc = gs.main([
        "--run-dir", str(zimage_run_dir), "--image-engine", "zimage",
        "--only-scene", "1", "--prompt-override", "uma cena de teste",
    ])
    assert rc == 0
    assert calls == []  # nem ensure_comfyui_running nem comfy_is_up foram chamados


def test_main_with_flux_engine_still_checks_comfyui(zimage_run_dir, monkeypatch):
    # Garde a guarda nao vira um "nunca checa" -- flux (ComfyUI de verdade)
    # continua batendo em ensure_comfyui_running como antes.
    calls = []
    monkeypatch.setattr(gs, "ensure_comfyui_running",
                        lambda *a, **k: calls.append("ensure") or False)

    rc = gs.main([
        "--run-dir", str(zimage_run_dir), "--image-engine", "flux",
        "--only-scene", "1", "--prompt-override", "uma cena de teste",
    ])
    assert rc == 1  # ensure_comfyui_running devolveu False -> main() desiste
    assert calls == ["ensure"]
