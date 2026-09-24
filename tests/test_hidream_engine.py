"""Sem GPU: wiring do motor HiDream-I1 em generate_storyboards.py."""
from pathlib import Path

from script_pipeline.generate_storyboards import IMAGE_ENGINES, detect_architecture, generate_scene_storyboard


def test_hidream_detected_and_listed():
    assert detect_architecture("hidream-i1-dev-Q4_K_M.gguf") == "hidream"
    assert "hidream" in IMAGE_ENGINES


def test_hidream_builds_quadruple_clip_workflow(monkeypatch, tmp_path):
    import script_pipeline.generate_storyboards as sb
    captured = {}

    def fake_submit(server, workflow, log=print):
        captured["workflow"] = workflow
        return {"outputs": {}}

    def fake_first_output(entry):
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(sb, "submit_and_wait", fake_submit)
    monkeypatch.setattr(sb, "_first_output_image", fake_first_output)
    ok = generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=IMAGE_ENGINES["hidream"]["checkpoint"],
        clip=IMAGE_ENGINES["hidream"]["clip"], vae=IMAGE_ENGINES["hidream"]["vae"],
        width=960, height=544, steps=24, cfg=1.0, seed=1, out_path=tmp_path / "still.png",
        prompt_override="a test prompt", log=lambda m: None)
    assert ok is True
    wf = captured["workflow"]
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wf["2"]["class_type"] == "QuadrupleCLIPLoader"
    assert wf["2"]["inputs"]["clip_name1"] == "clip_l_hidream.safetensors"
    assert wf["2"]["inputs"]["clip_name4"] == "llama_3.1_8b_instruct_fp8_scaled.safetensors"
    assert wf["4"]["inputs"]["clip_l"] == "a test prompt" == wf["4"]["inputs"]["llama"]
    assert wf["7"]["inputs"]["model"] == ["1", 0]


def test_qwen_image_base_detected_and_listed():
    assert detect_architecture("qwen-image-Q4_K_M.gguf") == "qwen-image-base"
    assert detect_architecture("qwen-image-2.1-Q4_K_M.gguf") == "qwenimage21"  # 2.1 nao regride
    assert "qwen-image" in IMAGE_ENGINES


def test_qwen_image_base_builds_single_clip_workflow(monkeypatch, tmp_path):
    import script_pipeline.generate_storyboards as sb
    captured = {}

    def fake_submit(server, workflow, log=print):
        captured["workflow"] = workflow
        return {"outputs": {}}

    def fake_first_output(entry):
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(sb, "submit_and_wait", fake_submit)
    monkeypatch.setattr(sb, "_first_output_image", fake_first_output)
    ok = generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=IMAGE_ENGINES["qwen-image"]["checkpoint"],
        clip=IMAGE_ENGINES["qwen-image"]["clip"], vae=IMAGE_ENGINES["qwen-image"]["vae"],
        width=960, height=544, steps=20, cfg=4.0, seed=1, out_path=tmp_path / "still.png",
        prompt_override="a test prompt", log=lambda m: None)
    assert ok is True
    wf = captured["workflow"]
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wf["2"]["inputs"]["type"] == "qwen_image"
    assert wf["4"]["inputs"]["text"] == "a test prompt"
    assert wf["7"]["inputs"]["cfg"] == 4.0


def test_hidream_wires_img2img_reference_when_given(monkeypatch, tmp_path):
    import script_pipeline.generate_storyboards as sb
    captured = {}

    def fake_submit(server, workflow, log=print):
        captured["workflow"] = workflow
        return {"outputs": {}}

    def fake_first_output(entry):
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return out

    def fake_stage(path, tag):
        return "staged.png"

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    monkeypatch.setattr(sb, "submit_and_wait", fake_submit)
    monkeypatch.setattr(sb, "_first_output_image", fake_first_output)
    monkeypatch.setattr(sb, "_stage_reference", fake_stage)
    ok = generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=IMAGE_ENGINES["hidream"]["checkpoint"],
        clip=IMAGE_ENGINES["hidream"]["clip"], vae=IMAGE_ENGINES["hidream"]["vae"],
        width=960, height=544, steps=24, cfg=1.0, seed=1, out_path=tmp_path / "still.png",
        prompt_override="a test prompt", reference_image=str(ref), log=lambda m: None)
    assert ok is True
    wf = captured["workflow"]
    assert wf["9001"]["class_type"] == "LoadImage" and wf["9001"]["inputs"]["image"] == "staged.png"
    assert wf["9003"]["class_type"] == "VAEEncode" and wf["9003"]["inputs"]["vae"] == ["3", 0]
    assert wf["7"]["inputs"]["latent_image"] == ["9003", 0]
    assert wf["7"]["inputs"]["denoise"] == 0.55


def test_hidream_without_reference_keeps_empty_latent(monkeypatch, tmp_path):
    import script_pipeline.generate_storyboards as sb

    def fake_submit(server, workflow, log=print):
        fake_submit.wf = workflow
        return {"outputs": {}}

    def fake_first_output(entry):
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(sb, "submit_and_wait", fake_submit)
    monkeypatch.setattr(sb, "_first_output_image", fake_first_output)
    ok = generate_scene_storyboard(
        {"index": 0}, {}, server="http://x", checkpoint=IMAGE_ENGINES["hidream"]["checkpoint"],
        clip=IMAGE_ENGINES["hidream"]["clip"], vae=IMAGE_ENGINES["hidream"]["vae"],
        width=960, height=544, steps=24, cfg=1.0, seed=1, out_path=tmp_path / "still.png",
        prompt_override="a test prompt", log=lambda m: None)
    assert ok is True
    assert fake_submit.wf["7"]["inputs"]["latent_image"] == ["6", 0]
    assert fake_submit.wf["7"]["inputs"]["denoise"] == 1.0
    assert "9001" not in fake_submit.wf
