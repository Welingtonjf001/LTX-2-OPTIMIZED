"""Recomendacoes do Qwen-Image-2.1 aplicadas ao pipeline (2026-09-21). Sem GPU e sem servidor."""
import json
import sys
import types
from pathlib import Path

import pytest

from script_pipeline.gate_retry import failed_checks, gate_with_retries, repair_note, write_repair_notes


def test_repair_note_maps_failed_checks_to_corrections():
    entry = {"reasons": ["chatter", "failed checks: forbidden_text_detected, location_match, identity_match"]}
    shot = {"subject": "HA-EUN", "location_id": "LOC_CABIN", "location": "CABINE DE PASSAGEIROS"}
    assert failed_checks(entry) == ["forbidden_text_detected", "location_match", "identity_match"]
    note = repair_note(entry, shot)
    assert "Remove any caption" in note and "CABINE DE PASSAGEIROS" in note and "HA-EUN's face" in note
    assert repair_note({"reasons": ["failed checks: x"], "auditor_error": True}, shot) == ""
    assert repair_note({"reasons": ["failed checks: unknown_check"]}, shot) == ""


def test_write_repair_notes_and_cleanup_on_approval(tmp_path):
    (tmp_path / "shots").mkdir()
    (tmp_path / "parse").mkdir()
    (tmp_path / "parse" / "shot_plan.json").write_text(
        json.dumps({"shots": [{"subject": "A"}, {"subject": "B"}]}), encoding="utf-8")
    (tmp_path / "shots" / "visual_stills_audit.json").write_text(json.dumps({"results": [
        {"shot": 0, "pass": True, "reasons": []},
        {"shot": 1, "pass": False, "reasons": ["failed checks: forbidden_text_detected"]}]}), encoding="utf-8")
    notes = write_repair_notes(tmp_path, "stills", [0, 1])
    assert list(notes) == ["1"]
    assert json.loads((tmp_path / "shots" / "gate_repair_notes.json").read_text(encoding="utf-8")) == notes
    assert write_repair_notes(tmp_path, "video", [1]) == {}
    ok = gate_with_retries(tmp_path, stage="stills", audit_cmd=["--stage", "stills"], regen_cmd=lambda s, seed: ["r"],
                           run_step=lambda n, c, obrigatorio=True: True if c == ["r"] else "--only-shots" in c,
                           blocked_fn=lambda: [1], base_seed=1, max_retries=1,
                           notes_fn=lambda shots: write_repair_notes(tmp_path, "stills", shots), log=lambda m: None)
    assert ok is True and not (tmp_path / "shots" / "gate_repair_notes.json").exists()


def test_emotion_pass_instruction_rules(monkeypatch):
    from script_pipeline import render_shots as rs
    shot = {"framing": "close", "subject": "HA-EUN", "emotion": "com_medo", "line_index": None}
    monkeypatch.delenv("QWEN_EMOTION_PASS", raising=False)
    assert rs._qwen_emotion_instruction(shot) is None                     # desligado por padrao
    monkeypatch.setenv("QWEN_EMOTION_PASS", "1")
    assert "HA-EUN" in rs._qwen_emotion_instruction(shot)
    assert "speaking" in rs._qwen_emotion_instruction(dict(shot, line_index=3))   # nao ocupa a boca
    assert rs._qwen_emotion_instruction(dict(shot, framing="wide")) is None
    assert rs._qwen_emotion_instruction(dict(shot, emotion="neutra")) is None
    assert rs._qwen_emotion_instruction(dict(shot, subject="")) is None


def test_reference_bytes_flattens_transparency(tmp_path):
    import io

    import qwen_image21_backend as q
    from PIL import Image
    rgba = tmp_path / "a.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(rgba)        # 100% transparente, RGB vermelho por baixo
    flat = Image.open(io.BytesIO(q._reference_bytes(rgba)))
    assert flat.mode == "RGB" and flat.getpixel((0, 0)) == (127, 127, 127)
    plain = tmp_path / "b.png"
    plain.write_bytes(b"nao e imagem")
    assert q._reference_bytes(plain) == b"nao e imagem"


def test_generate_rgba_checks_real_alpha_and_edit_keeps_image_size(tmp_path, monkeypatch):
    import qwen_image21_backend as q
    from PIL import Image
    calls = {}

    def gen_rgba(prompt, out_path, **kw):
        calls["prompt"] = prompt
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(out_path)
        return True

    def gen_rgb(prompt, out_path, **kw):
        Image.new("RGB", (64, 64)).save(out_path)
        return True
    monkeypatch.setattr(q, "generate", gen_rgba)
    assert q.generate_rgba("a white jet", tmp_path / "jet.png", log=lambda m: None) is True
    assert "RGBA image with transparency" in calls["prompt"]
    monkeypatch.setattr(q, "generate", gen_rgb)
    assert q.generate_rgba("x", tmp_path / "opaco.png", log=lambda m: None) is False   # sem alfa = reprovado
    src = tmp_path / "src.png"
    Image.new("RGB", (1000, 570)).save(src)
    seen = {}

    def spy(prompt, out_path, **kw):
        seen.update(kw)
        return True
    monkeypatch.setattr(q, "generate", spy)
    assert q.edit(src, "muda", tmp_path / "o.png", extra_references=["ref.png"], log=lambda m: None)
    assert (seen["width"], seen["height"]) == (1000 // 16 * 16, 570 // 16 * 16)
    assert seen["reference_images"] == [str(src), "ref.png"]


def test_turnaround_ref_and_rgba_asset_registration(tmp_path, monkeypatch):
    import qwen_image21_backend as q
    from script_pipeline import production_post as pp
    from script_pipeline import render_shots as rs
    run = tmp_path / "run"
    (run / "characters").mkdir(parents=True)
    (run / "shots" / "stills").mkdir(parents=True)
    tur = run / "characters" / "t.png"
    tur.write_bytes(b"x")
    (run / "characters" / "cast.json").write_text(json.dumps({"HA-EUN": {"turnaround_image": str(tur)}}), encoding="utf-8")
    assert rs._turnaround_for({"subject": "HA-EUN"}, run / "shots" / "stills") == str(tur)
    assert rs._turnaround_for({"subject": "OUTRA"}, run / "shots" / "stills") is None
    proj = tmp_path / "proj"
    (proj / "biblia").mkdir(parents=True)
    (proj / "biblia" / "locacoes.json").write_text(json.dumps({"LOC_SKY": {}}), encoding="utf-8")

    def fake_rgba(prompt, out, **kw):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"x")
        return True
    monkeypatch.setattr(q, "generate_rgba", fake_rgba)
    out = pp.make_rgba_asset(proj, "aircraft", "white jet", register_location="LOC_SKY")
    saved = json.loads((proj / "biblia" / "locacoes.json").read_text(encoding="utf-8"))
    assert saved["LOC_SKY"]["references"]["front"] == str(Path(out).resolve())
    with pytest.raises(ValueError):
        pp.make_rgba_asset(proj, "x", "y", register_location="LOC_INEXISTENTE")


def test_still_for_shot_repairs_by_edit_instead_of_reseeding(tmp_path, monkeypatch):
    import script_pipeline.generate_storyboards as sb
    from script_pipeline import render_shots as rs
    stills = tmp_path / "shots" / "stills"
    stills.mkdir(parents=True)
    (stills / "shot000_close.png").write_bytes(b"velho")
    (stills / "stills.json").write_text(json.dumps({"0": {"file": "shot000_close.png", "key": "outra"}}), encoding="utf-8")
    (tmp_path / "shots" / "gate_repair_notes.json").write_text(json.dumps({"0": "Remove any caption."}), encoding="utf-8")
    seen = {}

    def fake_edit(image, instruction, out, **kw):
        seen.update(image=str(image), instruction=instruction, extra=kw.get("extra_references"))
        Path(out).write_bytes(b"reparado")
        return True

    def no_reseed(*a, **k):
        raise AssertionError("nao deveria resemear")
    monkeypatch.setitem(sys.modules, "qwen_image21_engine", types.SimpleNamespace(edit=fake_edit))
    monkeypatch.setattr(sb, "detect_architecture", lambda ck: "qwenimage21")
    monkeypatch.setattr(rs, "_duplicate_face_check", lambda *a, **k: {})
    monkeypatch.setattr(sb, "generate_scene_storyboard", no_reseed)
    shot = {"framing": "close", "storyboard_prompt": "x", "subject": "A", "co_subject": ""}
    out = rs._still_for_shot(shot, 0, out_dir=stills, width=960, height=544, checkpoint="qwen-image-2.1", clip="",
                             vae="", seed=1, reference="ref.png", log=lambda m: None)
    assert out is not None and out.read_bytes() == b"reparado"
    assert "Fix ONLY these problems" in seen["instruction"] and "Remove any caption." in seen["instruction"]
    assert seen["extra"] == ["ref.png"] and seen["image"].endswith("shot000_close.png")
    man = json.loads((stills / "stills.json").read_text(encoding="utf-8"))
    assert "repair=" in man["0"]["key"] and man["0"]["repair"] == "Remove any caption."


def test_comfy_backend_workflow_wires_refs_and_latent():
    import qwen_image21_comfy_backend as c
    g = c.build_workflow("<image1> and <image2>", ["a.png", "b.png"], width=960, height=544, steps=25, seed=3)
    enc = g["5"]["inputs"]
    assert enc["images.image_1"] == ["img1", 0] and enc["images.image_2"] == ["img2", 0]
    assert g["7"]["inputs"]["latent_image"] == ["6", 0] and g["7"]["inputs"]["cfg"] == 1.0
    ed = c.build_workflow("x", ["a.png"], width=0, height=0, steps=25, seed=3, edit_target=True)
    assert ed["7"]["inputs"]["latent_image"] == ["5", 2] and "6" not in ed
    assert c.roles_prefix(["A", "B."]) == "<image1> is A. <image2> is B. "


def test_pair_by_id_rejects_position_mismatch_and_missing_id(tmp_path):
    from script_pipeline import production_post as pp
    plan = {"shots": [{"id": "S0", "scene": 1}, {"id": "S1", "scene": 2}]}
    edit = {"shots": [{"id": "S1", "duration_frames": 10}, {"id": "S0", "duration_frames": 5}]}
    pares = pp._pair_by_id(plan, edit)
    assert [s["id"] for s, _ in pares] == ["S1", "S0"]   # casado por id, nao por posicao
    with pytest.raises(ValueError):
        pp._pair_by_id(plan, {"shots": [{"id": "SX"}]})


def test_audio_stems_measures_real_duration_without_window(tmp_path, monkeypatch):
    import wave
    from script_pipeline import production_post as pp
    root = tmp_path / "proj"
    (root / "editorial").mkdir(parents=True)
    (root / "audio").mkdir()
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    (run / "dialogue").mkdir()
    wav = run / "dialogue" / "l0.wav"
    with wave.open(str(wav), "wb") as w:
        w.setparams((1, 2, 48000, 48000, "NONE", "x"))   # 1,0 s exato
        w.writeframes(b"\x00\x00" * 48000)
    (root / "editorial" / "timeline.json").write_text(json.dumps(
        {"fps": 24, "duration_frames": 96,
         "shots": [{"id": "S0", "duration_frames": 96, "start_frame": 0}]}), encoding="utf-8")
    (run / "parse" / "shot_plan.json").write_text(json.dumps(
        {"shots": [{"id": "S0", "scene": 1, "line_index": 0}]}), encoding="utf-8")
    (run / "dialogue" / "lines.json").write_text(json.dumps(
        [{"scene_index": 1, "line_index": 0, "audio_path": str(wav), "text": "oi"}]), encoding="utf-8")
    monkeypatch.setattr(pp, "project_for", lambda r: root)
    report = pp.audio_stems(run)
    assert report["dialogos"]["assigned_events"] == 1
    # plano de 96/24=4s, fala de 1s: lead esperado = (4-1)/2 = 1,5s (nao 0s)
    cues = json.loads((root / "audio" / "cues.json").read_text(encoding="utf-8")) if (root / "audio" / "cues.json").exists() else None


def test_qwen_job_translates_raw_slug_to_natural_instruction():
    from script_pipeline.dialogue_tts import _qwen_job
    job = {"id": "x", "text": "oi", "instruct": "com_medo", "output_path": "o.wav"}
    assert _qwen_job(job)["instruct"] == "Speak in a frightened, tense voice, alert, like an urgent warning whispered in a hurry."
    # parentetico real (nao bate com nenhum slug) passa direto, sem traducao
    livre = {"id": "y", "text": "oi", "instruct": "sussurrando, olhando para tras", "output_path": "o.wav"}
    assert _qwen_job(livre)["instruct"] == "sussurrando, olhando para tras"
    vazio = {"id": "z", "text": "oi", "output_path": "o.wav"}
    assert _qwen_job(vazio)["instruct"] is None
