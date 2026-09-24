# -*- coding: utf-8 -*-
"""Testes sem GPU pra generate_longtake()/build_longtake_workflow() do
minimax_h3_backend.py -- monkeypatch em base_api_longtake() pra nao precisar
do ComfyUI do MiniMax H3 no ar. Ver MEMORIAL 3.117.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import minimax_h3_backend as b  # noqa: E402


@pytest.fixture
def grafo(monkeypatch):
    # ensure_server() faria uma chamada HTTP -- sem servidor no ar, vira
    # no-op aqui. O resto de base_api_longtake() (ler o template do disco,
    # aplicar os overrides de checkpoint) roda de verdade.
    monkeypatch.setattr(b, "ensure_server", lambda *a, **kw: None)
    b._base_api_longtake_cache = None
    return b.base_api_longtake()


def test_template_tem_todos_os_nos_esperados(grafo):
    esperados = [b.N_LT_UNET, b.N_LT_CLIP, b.N_LT_LORA, b.N_LT_SAGEATTN,
                b.N_LT_STEPS, b.N_LT_PROJECT, b.N_LT_EDITOR, b.N_LT_COMBINE, b.N_LT_SAVE]
    for n in esperados:
        assert n in grafo, f"nó {n} ausente do template"
    assert grafo[b.N_LT_SAVE]["class_type"] == "SaveVideo"
    assert grafo[b.N_LT_EDITOR]["class_type"] == "easy multiTrackEditor"
    assert grafo[b.N_LT_PROJECT]["class_type"] == "easy multitrackProject"


def test_um_segmento_shot_sem_referencia(grafo):
    api = b.build_longtake_workflow(
        [{"prompt": "plano unico", "duration_frames": 48}],
        project_name="teste1", seed=7)
    track_data = json.loads(api[b.N_LT_EDITOR]["inputs"]["track_data"])
    segs = track_data["tracks"][0]["segments"]
    assert len(segs) == 1
    assert segs[0]["start_frame"] == 0 and segs[0]["end_frame"] == 48
    assert segs[0]["content"]["continuity_mode"] == "shot"
    assert segs[0]["content"]["task_mode"] == "default"
    assert segs[0]["content"]["images"] == []
    assert track_data["total_length"] == 48
    assert api[b.N_LT_PROJECT]["inputs"]["project_name"] == "teste1"
    assert api[b.N_LT_PROJECT]["inputs"]["seed"] == 7


def test_dois_segmentos_shot_depois_context_com_referencia(grafo):
    api = b.build_longtake_workflow([
        {"prompt": "Use <Picture 1> as reference frame. abre a cena",
         "duration_frames": 48, "ref_images": ["/tmp/ref.png"]},
        {"prompt": "continua o movimento", "duration_frames": 60},
    ])
    track_data = json.loads(api[b.N_LT_EDITOR]["inputs"]["track_data"])
    segs = track_data["tracks"][0]["segments"]
    assert len(segs) == 2
    assert segs[0]["content"]["continuity_mode"] == "shot"
    assert segs[0]["content"]["task_mode"] == "ref"
    assert segs[0]["content"]["images"][0]["file_name"] == "ref.png"
    assert segs[0]["start_frame"] == 0 and segs[0]["end_frame"] == 48
    assert segs[1]["content"]["continuity_mode"] == "context"
    assert segs[1]["content"]["task_mode"] == "default"
    assert segs[1]["content"]["images"] == []
    assert segs[1]["start_frame"] == 48 and segs[1]["end_frame"] == 108
    assert track_data["total_length"] == 108


def test_continuity_mode_explicito_e_respeitado(grafo):
    api = b.build_longtake_workflow([
        {"prompt": "a", "duration_frames": 24, "continuity_mode": "shot"},
        {"prompt": "b", "duration_frames": 24, "continuity_mode": "context_swap"},
        {"prompt": "c", "duration_frames": 24, "continuity_mode": "context"},
    ])
    track_data = json.loads(api[b.N_LT_EDITOR]["inputs"]["track_data"])
    modos = [s["content"]["continuity_mode"] for s in track_data["tracks"][0]["segments"]]
    assert modos == ["shot", "context_swap", "context"]


def test_checkpoint_e_resolucao_aplicados(grafo):
    api = b.build_longtake_workflow(
        [{"prompt": "a", "duration_frames": 48}],
        aspect_ratio="9:16 (Portrait)", megapixels=0.8, steps=12,
        filename_prefix="minha_cena")
    assert api[b.N_LT_UNET]["inputs"]["unet_name"] == b.LONGTAKE_UNET_FILENAME
    assert api[b.N_LT_CLIP]["inputs"]["clip_name"] == b.LONGTAKE_CLIP_FILENAME
    assert api[b.N_LT_EDITOR]["inputs"]["resolution.aspect_ratio"] == "9:16 (Portrait)"
    assert api[b.N_LT_EDITOR]["inputs"]["resolution.megapixels"] == 0.8
    assert api[b.N_LT_STEPS]["inputs"]["steps"] == 12
    assert api[b.N_LT_SAVE]["inputs"]["filename_prefix"] == "minha_cena"


def test_sem_segmentos_recusa(grafo):
    with pytest.raises(ValueError):
        b.build_longtake_workflow([])


def test_duration_zero_recusa(grafo):
    with pytest.raises(ValueError):
        b.build_longtake_workflow([{"prompt": "a", "duration_frames": 0}])


def test_generate_longtake_estagia_referencias_e_limpa(monkeypatch, grafo, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "out.mp4"
    src_output = tmp_path / "comfy_output"
    src_output.mkdir()
    (src_output / "video" / "cena").parent.mkdir(parents=True, exist_ok=True)
    saved_file = src_output / "video_saida.mp4"
    saved_file.write_bytes(b"fake mp4")

    monkeypatch.setattr(b, "COMFY_OUTPUT", str(src_output))
    staged = []

    def fake_stage(path):
        dest = tmp_path / ("staged_" + Path(path).name)
        dest.write_bytes(Path(path).read_bytes())
        staged.append(str(dest))
        return str(dest)

    monkeypatch.setattr(b, "_stage_input", fake_stage)
    monkeypatch.setattr(b, "submit_and_wait",
                        lambda api, **kw: [saved_file.name])

    captured = {}
    orig_build = b.build_longtake_workflow

    def spy_build(segments, **kw):
        captured["segments"] = segments
        return orig_build(segments, **kw)

    monkeypatch.setattr(b, "build_longtake_workflow", spy_build)

    result = b.generate_longtake(
        [{"prompt": "abre a cena", "duration_frames": 48, "ref_images": [str(ref)]},
         {"prompt": "continua", "duration_frames": 48}],
        str(out))

    assert result == str(out)
    assert out.read_bytes() == b"fake mp4"
    # a referencia citada automaticamente no prompt (auto_cite_refs=True)
    assert "<Picture 1>" in captured["segments"][0]["prompt"]
    # arquivos estagiados foram limpos no finally
    for p in staged:
        assert not Path(p).exists()
