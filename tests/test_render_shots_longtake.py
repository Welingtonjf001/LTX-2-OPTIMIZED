# -*- coding: utf-8 -*-
"""Testes sem GPU pro agrupamento 'minimax-longtake' em render_shots.py:
`_minimax_longtake_flush` agrupa em TAKES (cena nova OU framing wide/full
abre take -- ver `_assign_takes`), chama generate_longtake() uma vez por
grupo (2+ planos) ou generate() pro grupo de 1 plano so, e recorta o video
combinado de volta em arquivos por plano. Tudo mockado -- sem ComfyUI, sem
ffmpeg de verdade. Ver MEMORIAL 3.118/3.120.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import script_pipeline.render_shots as rs  # noqa: E402


def _pendente(shot, scene, tmp_path, *, ref_images=None, duration_seconds=2.0,
              framing="close"):
    clip_path = tmp_path / f"shot{shot:03d}.mp4"
    marca = tmp_path / f"shot{shot:03d}.key"
    return {
        "shot": shot, "scene": scene, "framing": framing, "clip_path": clip_path,
        "marca": marca, "chave": f"chave{shot}", "still": tmp_path / f"still{shot}.png",
        "prompt": f"prompt do plano {shot}", "ref_images": ref_images or [],
        "duration_seconds": duration_seconds,
    }


@pytest.fixture(autouse=True)
def sem_gpu(monkeypatch):
    """`_clip_frames` chama ffprobe de verdade -- mocka pra devolver algo fixo
    sem precisar de um mp4 real no disco."""
    monkeypatch.setattr(rs, "_clip_frames", lambda p: 48)


def test_grupo_de_1_plano_usa_generate_normal(tmp_path, monkeypatch):
    import minimax_h3_backend as b
    chamadas = []
    monkeypatch.setattr(b, "generate", lambda prompt, out, **kw: (
        chamadas.append((prompt, out, kw)), Path(out).write_bytes(b"x"))[-1])
    generate_longtake_mock = MagicMock()
    monkeypatch.setattr(b, "generate_longtake", generate_longtake_mock)

    pend = [_pendente(0, scene=1, tmp_path=tmp_path)]
    feitos = rs._minimax_longtake_flush(
        pend, fps=24, seed=1, minimax_aspect_ratio=None, minimax_megapixels=None)

    assert len(chamadas) == 1
    generate_longtake_mock.assert_not_called()
    assert feitos == [{"shot": 0, "still": str(pend[0]["still"]), "clip": str(pend[0]["clip_path"])}]
    assert pend[0]["marca"].exists()


def test_grupo_de_2_planos_usa_generate_longtake_e_recorta(tmp_path, monkeypatch):
    import minimax_h3_backend as b
    chamadas_longtake = []

    def fake_generate_longtake(segments, out_path, **kw):
        chamadas_longtake.append((segments, out_path, kw))
        Path(out_path).write_bytes(b"combined")
        return out_path

    monkeypatch.setattr(b, "generate_longtake", fake_generate_longtake)
    monkeypatch.setattr(b, "generate", MagicMock(side_effect=AssertionError(
        "nao deveria chamar generate() num grupo de 2+")))

    recortes = []

    def fake_extract(src, dest, start, dur, **kw):
        recortes.append((src, dest, round(start, 3), round(dur, 3)))
        Path(dest).write_bytes(b"sub")
        return True

    monkeypatch.setattr(rs, "_extract_subclip", fake_extract)

    pend = [
        _pendente(0, scene=2, tmp_path=tmp_path, ref_images=["/ref/haeun.png"], duration_seconds=2.0),
        _pendente(1, scene=2, tmp_path=tmp_path, duration_seconds=3.0),
    ]
    feitos = rs._minimax_longtake_flush(
        pend, fps=24, seed=10, minimax_aspect_ratio="16:9 (Widescreen)", minimax_megapixels=0.5)

    assert len(chamadas_longtake) == 1
    segments, out_path, kw = chamadas_longtake[0]
    assert len(segments) == 2
    assert segments[0]["continuity_mode"] == "shot"
    assert segments[0]["ref_images"] == ["/ref/haeun.png"]
    assert segments[0]["duration_frames"] == 48  # 2.0s * 24fps
    assert segments[1]["continuity_mode"] == "context"
    assert segments[1]["ref_images"] == []
    assert segments[1]["duration_frames"] == 72  # 3.0s * 24fps
    assert kw["frame_rate"] == 24

    # 2 recortes, o segundo comecando onde o primeiro termina (2.0s)
    assert len(recortes) == 2
    assert recortes[0][2] == 0.0 and recortes[0][3] == 2.0
    assert recortes[1][2] == 2.0 and recortes[1][3] == 3.0

    assert [f["shot"] for f in feitos] == [0, 1]
    assert all(f["clip"] for f in feitos)
    assert all(pend[i]["marca"].exists() for i in range(2))


def test_planos_de_cenas_diferentes_nao_se_misturam(tmp_path, monkeypatch):
    import minimax_h3_backend as b
    grupos_vistos = []

    def fake_generate_longtake(segments, out_path, **kw):
        grupos_vistos.append(len(segments))
        Path(out_path).write_bytes(b"combined")
        return out_path

    monkeypatch.setattr(b, "generate_longtake", fake_generate_longtake)
    monkeypatch.setattr(rs, "_extract_subclip", lambda src, dest, s, d, **kw: (
        Path(dest).write_bytes(b"sub"), True)[-1])

    pend = [
        _pendente(0, scene=1, tmp_path=tmp_path),
        _pendente(1, scene=1, tmp_path=tmp_path),
        _pendente(2, scene=2, tmp_path=tmp_path),
        _pendente(3, scene=2, tmp_path=tmp_path),
    ]
    feitos = rs._minimax_longtake_flush(
        pend, fps=24, seed=1, minimax_aspect_ratio=None, minimax_megapixels=None)

    assert grupos_vistos == [2, 2]
    assert [f["shot"] for f in feitos] == [0, 1, 2, 3]


def test_assign_takes_wide_full_abrem_take_novo():
    # Mesmo padrao de framings do shot_plan real do CERCO EM SEUL (25 planos,
    # 1 cena so): wide nos indices 0, 8, 12, 16 -- 4 takes esperados.
    framings = ["wide", "insert", "insert", "close", "close", "insert", "insert", "medium",
                "wide", "close", "close", "medium",
                "wide", "medium", "insert", "close",
                "wide", "medium", "close", "medium", "close", "medium", "close", "medium", "insert"]
    pend = [{"shot": i, "scene": 1, "framing": f} for i, f in enumerate(framings)]
    rs._assign_takes(pend)
    take_ids = [p["take_id"] for p in pend]
    assert take_ids == (
        [0] * 8 +   # indices 0-7
        [1] * 4 +   # indices 8-11
        [2] * 4 +   # indices 12-15
        [3] * 9     # indices 16-24
    )


def test_assign_takes_cena_nova_sempre_abre_take_mesmo_sem_wide():
    pend = [
        {"shot": 0, "scene": 1, "framing": "close"},
        {"shot": 1, "scene": 1, "framing": "medium"},
        {"shot": 2, "scene": 2, "framing": "close"},
        {"shot": 3, "scene": 2, "framing": "medium"},
    ]
    rs._assign_takes(pend)
    assert [p["take_id"] for p in pend] == [0, 0, 1, 1]


def test_grupo_de_3_com_wide_no_meio_quebra_em_2_takes(tmp_path, monkeypatch):
    import minimax_h3_backend as b
    grupos_vistos = []

    def fake_generate_longtake(segments, out_path, **kw):
        grupos_vistos.append(len(segments))
        Path(out_path).write_bytes(b"combined")
        return out_path

    monkeypatch.setattr(b, "generate_longtake", fake_generate_longtake)
    monkeypatch.setattr(b, "generate", lambda prompt, out, **kw: (
        grupos_vistos.append(1), Path(out).write_bytes(b"x"))[-1])
    monkeypatch.setattr(rs, "_extract_subclip", lambda src, dest, s, d, **kw: (
        Path(dest).write_bytes(b"sub"), True)[-1])

    pend = [
        _pendente(0, scene=1, tmp_path=tmp_path, framing="insert"),
        _pendente(1, scene=1, tmp_path=tmp_path, framing="wide"),  # abre take novo
        _pendente(2, scene=1, tmp_path=tmp_path, framing="close"),
    ]
    feitos = rs._minimax_longtake_flush(
        pend, fps=24, seed=1, minimax_aspect_ratio=None, minimax_megapixels=None)

    # take 0 = so o plano 0 (1 plano -> generate() normal); take 1 = planos 1,2 (generate_longtake)
    assert sorted(grupos_vistos) == [1, 2]
    assert [f["shot"] for f in feitos] == [0, 1, 2]
    assert all(f["clip"] for f in feitos)


def test_falha_no_grupo_marca_todos_como_nao_ok(tmp_path, monkeypatch):
    import minimax_h3_backend as b

    def fake_generate_longtake(segments, out_path, **kw):
        raise RuntimeError("ComfyUI (MiniMax H3) explodiu")

    monkeypatch.setattr(b, "generate_longtake", fake_generate_longtake)
    import script_pipeline.generate_storyboards as sb_real
    monkeypatch.setattr(sb_real, "stop_comfyui", lambda *a, **kw: None)

    pend = [
        _pendente(0, scene=5, tmp_path=tmp_path),
        _pendente(1, scene=5, tmp_path=tmp_path),
    ]
    feitos = rs._minimax_longtake_flush(
        pend, fps=24, seed=1, minimax_aspect_ratio=None, minimax_megapixels=None)

    assert [f["clip"] for f in feitos] == [None, None]
    assert not pend[0]["marca"].exists()
