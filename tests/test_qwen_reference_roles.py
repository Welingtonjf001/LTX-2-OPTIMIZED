# -*- coding: utf-8 -*-
"""Testes sem GPU pro papel nomeado de referencia no Qwen-Image-2.1
(pedido do usuario 2026-09-24, depois do CERCO EM SEUL confundir sujeitos em
planos de 2+ personagens): `generate_scene_storyboard()` deve prefixar o
prompt com `<image1> is X.` quando um papel e passado, e `_still_for_shot()`
deve construir esse papel a partir de `subject`/`co_subject` + `cast_
descriptors`, sem mudar nada pra quem nao usa Qwen-Image-2.1.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import script_pipeline.generate_storyboards as sb  # noqa: E402
import script_pipeline.render_shots as rs  # noqa: E402

QWEN_CHECKPOINT = "qwen-image-2.1-gguf-q4.gguf"


def test_generate_scene_storyboard_sem_papel_mantem_comportamento_antigo(monkeypatch):
    captured = {}

    class FakeEngine:
        @staticmethod
        def generate(prompt, out_path, **kw):
            captured.update(kw)
            captured["prompt"] = prompt
            return True

    monkeypatch.setitem(sys.modules, "qwen_image21_engine", FakeEngine)

    ok = sb.generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=QWEN_CHECKPOINT,
        width=512, height=512, steps=10, cfg=1.0, seed=1, out_path=Path("out.png"),
        prompt_override="a scene", reference_image="/a.png", reference_image_2="/b.png")

    assert ok is True
    assert captured["prompt"] == "a scene"
    assert captured["reference_images"] == ["/a.png", "/b.png"]
    assert captured["reference_roles"] is None


def test_generate_scene_storyboard_com_papel_prefixa_e_lista_roles(monkeypatch):
    captured = {}

    class FakeEngine:
        @staticmethod
        def generate(prompt, out_path, **kw):
            captured.update(kw)
            captured["prompt"] = prompt
            return True

    monkeypatch.setitem(sys.modules, "qwen_image21_engine", FakeEngine)

    ok = sb.generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=QWEN_CHECKPOINT,
        width=512, height=512, steps=10, cfg=1.0, seed=1, out_path=Path("out.png"),
        prompt_override="a scene", reference_image="/a.png", reference_image_2="/b.png",
        reference_image_role="SEO-YEON, dark hair", reference_image_2_role="JI-HO",
        extra_references=[("/loc.png", "the location")])

    assert ok is True
    assert captured["reference_images"] == ["/a.png", "/b.png", "/loc.png"]
    assert captured["reference_roles"] == ["SEO-YEON, dark hair", "JI-HO", "the location"]
    # prompt em si nao muda -- o papel entra do lado do backend (roles_prefix),
    # generate_scene_storyboard so encaminha, nao reescreve o texto.
    assert captured["prompt"] == "a scene"


def test_still_for_shot_constroi_papel_a_partir_do_cast(monkeypatch, tmp_path):
    monkeypatch.setattr(sb, "detect_architecture", lambda ck: "qwenimage21")

    captured = {}

    def fake_generate_scene_storyboard(*a, **kw):
        captured.update(kw)
        Path(kw["out_path"]).write_bytes(b"fake")
        return True

    monkeypatch.setattr(sb, "generate_scene_storyboard", fake_generate_scene_storyboard)
    monkeypatch.setattr(rs, "_qwen_emotion_instruction", lambda shot: None)
    monkeypatch.setattr(rs, "_load_repair_notes", lambda out_dir: {})

    shot = {
        "framing": "medium", "subject": "SEO-YEON", "co_subject": "JI-HO",
        "storyboard_prompt": "medium shot of SEO-YEON and JI-HO",
    }
    cast_descriptors = {
        "SEO-YEON": "Dark-brown hair, shoulder-length, mid-30s, black pinstripe suit",
        "JI-HO": "Short black hair, early 40s, dark-gray suit",
    }

    out = rs._still_for_shot(
        shot, 0, out_dir=tmp_path, width=512, height=512, checkpoint=QWEN_CHECKPOINT,
        clip="", vae="", seed=1, reference="/ref_seoyeon.png", reference_2="/ref_jiho.png",
        cast_descriptors=cast_descriptors)

    assert out is not None
    assert captured["reference_image_role"] == "SEO-YEON, Dark-brown hair"
    assert captured["reference_image_2_role"] == "JI-HO, Short black hair"


def test_still_for_shot_sem_cast_descriptors_usa_so_o_nome(monkeypatch, tmp_path):
    monkeypatch.setattr(sb, "detect_architecture", lambda ck: "qwenimage21")
    captured = {}

    def fake_generate_scene_storyboard(*a, **kw):
        captured.update(kw)
        Path(kw["out_path"]).write_bytes(b"fake")
        return True

    monkeypatch.setattr(sb, "generate_scene_storyboard", fake_generate_scene_storyboard)
    monkeypatch.setattr(rs, "_qwen_emotion_instruction", lambda shot: None)
    monkeypatch.setattr(rs, "_load_repair_notes", lambda out_dir: {})

    shot = {"framing": "close", "subject": "HA-EUN", "co_subject": "",
           "storyboard_prompt": "close of HA-EUN"}

    rs._still_for_shot(
        shot, 0, out_dir=tmp_path, width=512, height=512, checkpoint=QWEN_CHECKPOINT,
        clip="", vae="", seed=1, reference="/ref_haeun.png", cast_descriptors=None)

    assert captured["reference_image_role"] == "HA-EUN"
    assert captured["reference_image_2_role"] is None


def test_still_for_shot_engine_nao_qwen_nunca_gera_papel(monkeypatch, tmp_path):
    monkeypatch.setattr(sb, "detect_architecture", lambda ck: "flux")
    captured = {}

    def fake_generate_scene_storyboard(*a, **kw):
        captured.update(kw)
        Path(kw["out_path"]).write_bytes(b"fake")
        return True

    monkeypatch.setattr(sb, "generate_scene_storyboard", fake_generate_scene_storyboard)

    shot = {"framing": "close", "subject": "HA-EUN", "co_subject": "",
           "storyboard_prompt": "close of HA-EUN"}

    rs._still_for_shot(
        shot, 0, out_dir=tmp_path, width=512, height=512, checkpoint="flux-2-klein.safetensors",
        clip="", vae="", seed=1, reference="/ref_haeun.png",
        cast_descriptors={"HA-EUN": "some descriptor"})

    assert captured["reference_image_role"] is None
    assert captured["reference_image_2_role"] is None
