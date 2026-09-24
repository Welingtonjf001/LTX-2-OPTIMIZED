from pathlib import Path

import pytest

from script_pipeline.motion_director import (
    INTERACTION_CUES,
    _locate_still,
    find_interaction_shots,
    render_conditioned_shot,
    render_conditioned_shot_minimax,
)


def _shot(**kwargs):
    base = {"scene": 1, "position": 1, "subject": "HA-EUN", "co_subject": "SEO-YEON",
           "frames": 65, "beat": "", "fallback": "", "video_prompt": "", "storyboard_prompt": ""}
    base.update(kwargs)
    return base


def test_finds_bump_shot_from_portuguese_fallback():
    plan = {"shots": [_shot(fallback="Ha-eun aparece correndo e quase esbarra nela.")]}
    matches = find_interaction_shots(plan)
    assert len(matches) == 1
    assert matches[0]["intergen_prompt"] == "Two people bump into each other."


def test_finds_bump_shot_from_english_beat():
    plan = {"shots": [_shot(beat="Ha-eun runs into the frame and nearly collides with Seo-yeon.")]}
    matches = find_interaction_shots(plan)
    assert len(matches) == 1
    assert "bump" in matches[0]["intergen_prompt"].lower()


def test_finds_standoff_shot_from_english_beat():
    plan = {"shots": [_shot(beat="Seo-yeon and Min-jun stand facing each other, exchanging challenging glares.")]}
    matches = find_interaction_shots(plan)
    assert len(matches) == 1
    assert "argument" in matches[0]["intergen_prompt"].lower()


def test_skips_shot_without_co_subject():
    plan = {"shots": [_shot(co_subject="", beat="Ha-eun quase esbarra em alguem.")]}
    assert find_interaction_shots(plan) == []


def test_camera_push_boilerplate_does_not_false_match_physical_push():
    # ACHADO 2026-09-18 (rodando de verdade no filme dos piratas): "the
    # camera pushes in slowly" e boilerplate de MOVIMENTO DE CAMERA presente
    # em quase todo shot["video_prompt"] -- nao pode disparar a pista de
    # empurrao fisico. O shot abaixo tem a pista REAL (desafio/confronto)
    # em outro lugar do texto; o casamento deve ser o confronto, nao o push.
    plan = {"shots": [_shot(
        beat="Seo-yeon and Min-jun stand facing each other, exchanging challenging glares.",
        video_prompt="Seo-yeon and Min-jun face off. the camera pushes in slowly.",
    )]}
    matches = find_interaction_shots(plan)
    assert len(matches) == 1
    assert "argument" in matches[0]["intergen_prompt"].lower()


def test_camera_push_alone_does_not_match_anything():
    plan = {"shots": [_shot(video_prompt="A calm hallway. the camera pushes in slowly.")]}
    assert find_interaction_shots(plan) == []


def test_skips_shot_with_no_matching_cue():
    plan = {"shots": [_shot(beat="Two people walk quietly down the hallway.")]}
    assert find_interaction_shots(plan) == []


def test_first_matching_cue_wins_in_declared_order():
    # "luta" (briga) vem ANTES de "abraça" na lista -- um texto com as duas
    # pistas deve casar a primeira, nao a ultima.
    plan = {"shots": [_shot(beat="They fight fiercely, then embrace at the end.")]}
    matches = find_interaction_shots(plan)
    assert "boxing" in matches[0]["intergen_prompt"].lower()


def test_all_cue_prompts_are_nonempty_strings():
    for pattern, prompt in INTERACTION_CUES:
        assert isinstance(prompt, str) and prompt.strip()


# ACHADO 2026-09-18 (primeiro teste real): resolucao nao multipla de 64
# quebra a VAE do guia do union-control depois de MINUTOS de amostragem.

@pytest.mark.parametrize("width,height", [(864, 480), (800, 480), (768, 500)])
def test_render_conditioned_shot_rejects_non_multiple_of_64(width, height):
    shot = {"_width": width, "_height": height, "video_prompt": "x", "frames": 65}
    with pytest.raises(ValueError, match="multiplos de 64"):
        render_conditioned_shot(shot, Path("pose.mp4"), Path("still.png"), Path("out.mp4"))


@pytest.mark.parametrize("width,height", [(768, 512), (1280, 704)])
def test_render_conditioned_shot_accepts_validated_resolutions(width, height, monkeypatch, tmp_path):
    # Nao chega a chamar o LTX de verdade -- so confirma que a validacao de
    # resolucao NAO bloqueia as resolucoes ja validadas em producao (CLAUDE.md).
    import script_pipeline.motion_director as md
    monkeypatch.setattr(md, "REPO_ROOT", tmp_path)  # evita sys.path.insert real ter efeito colateral
    called = {}

    class FakeLtx25Backend:
        @staticmethod
        def generate(*a, **k):
            called["ok"] = True
            return "out.mp4"

    class FakeLtxLoras:
        @staticmethod
        def resolve(name):
            return None, "fake.safetensors"

    monkeypatch.setitem(__import__("sys").modules, "ltx25_backend", FakeLtx25Backend)
    monkeypatch.setitem(__import__("sys").modules, "ltx_loras", FakeLtxLoras)
    shot = {"_width": width, "_height": height, "video_prompt": "x", "frames": 65}
    render_conditioned_shot(shot, Path("pose.mp4"), Path("still.png"), Path("out.mp4"))
    assert called.get("ok") is True


# ACHADO 2026-09-18 (rodando de verdade no filme dos piratas): "position" e
# 0-indexado POR CENA, mas os arquivos de still extraidos numeram por indice
# GLOBAL na lista inteira -- so coincidem na PRIMEIRA cena do roteiro.

def test_find_interaction_shots_attaches_global_index_not_position():
    plan = {"shots": [
        _shot(scene=1, position=0),                                    # sem co_subject nem pista -> nao entra
        _shot(scene=1, position=1, beat="They bump into each other."),  # global 1, position 1 (coincide)
        _shot(scene=2, position=0, beat="They embrace warmly."),        # global 2, position 0 (NAO coincide)
    ]}
    matches = find_interaction_shots(plan)
    assert [m["shot"]["_global_index"] for m in matches] == [1, 2]
    assert [m["shot"]["position"] for m in matches] == [1, 0]


def test_locate_still_uses_global_index_not_position(tmp_path):
    frames_dir = tmp_path / "intermediate" / "_verify_frames"
    frames_dir.mkdir(parents=True)
    # Simula exatamente o caso real: cena 5, position=1, mas indice global=11.
    (frames_dir / "scene05_shot011.png").write_bytes(b"fake")
    assert _locate_still(tmp_path, scene=5, global_index=11) is not None
    assert _locate_still(tmp_path, scene=5, global_index=0) is None  # position-1 antigo apontava aqui, errado


# ACHADO 2026-09-18 (pedido do usuario: "vale para o minimax?") -- caminho
# equivalente pro MiniMax H3, nunca testado com GPU real (ControlNet do H3
# medido em ~500s/passo, ver project_pose_adapter_and_h3_controlnet.md). Este
# teste so trava a integracao de CODIGO: control_video gerado por
# motion_to_h3_controlnet.render_h3_control_video, still como ref_image
# (o H3 nao tem I2V como o LTX), duration_seconds derivado de shot["frames"].

def test_render_conditioned_shot_minimax_wires_control_video_and_ref_image(monkeypatch, tmp_path):
    import numpy as np
    import script_pipeline.motion_director as md

    monkeypatch.setattr(md, "REPO_ROOT", tmp_path)
    called = {}

    def fake_render_h3_control_video(joints, out_path, *, width, height, num_frames, source_fps):
        Path(out_path).write_bytes(b"fake")
        return num_frames, width, height

    class FakeMotionToH3ControlNet:
        render_h3_control_video = staticmethod(fake_render_h3_control_video)

    class FakeMinimaxH3Backend:
        @staticmethod
        def generate(prompt, output_path, **kwargs):
            called.update(kwargs)
            called["prompt"] = prompt
            called["output_path"] = output_path
            return output_path

    monkeypatch.setitem(__import__("sys").modules,
                        "script_pipeline.motion_to_h3_controlnet", FakeMotionToH3ControlNet)
    monkeypatch.setitem(__import__("sys").modules, "minimax_h3_backend", FakeMinimaxH3Backend)

    shot = {"video_prompt": "x", "frames": 68}  # 68/24 = 2.833...s
    joints = np.zeros((10, 2, 22, 3))
    still = tmp_path / "still.png"
    still.write_bytes(b"fake")
    out_path = tmp_path / "scene01_pos00_conditioned.mp4"

    render_conditioned_shot_minimax(shot, joints, still, out_path, width=768, height=480)

    assert called["ref_images"] == [str(still)]
    assert called["control_video"] == str(out_path.with_name(out_path.stem + "_h3control.mp4"))
    # ACHADO 2026-09-18 (validacao com GPU real): "16:9" sozinho e recusado
    # pelo node ResolutionSelector (HTTP 400) -- ele exige o rotulo completo
    # do combo ("16:9 (Widescreen)"). Regressao: nao passar aspect_ratio,
    # deixar no default do backend (que ja tem o rotulo certo).
    assert "aspect_ratio" not in called
    assert called["control_strength"] == 1.0
    assert called["duration_seconds"] == pytest.approx(68 / 24.0)
    assert called["output_path"] == str(out_path)


def test_multiple_shots_each_matched_independently():
    plan = {"shots": [
        _shot(position=1, beat="They embrace warmly."),
        _shot(position=2, co_subject="", beat="She embraces her book."),  # sem co_subject, ignorado
        _shot(position=3, beat="Two friends celebrate and jump together."),
    ]}
    matches = find_interaction_shots(plan)
    assert [m["shot"]["position"] for m in matches] == [1, 3]
