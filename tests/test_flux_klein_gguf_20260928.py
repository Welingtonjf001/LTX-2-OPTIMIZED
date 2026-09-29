"""FLUX.2 Klein GGUF Q8_0 (pedido do usuario 2026-09-28): mesmo checkpoint do motor "flux"
(fp8), so quantizado -- precisa se comportar como "flux" em tudo (referencia de personagem,
condicionamento espacial, LoRA), trocando so o loader do transformer por UnetLoaderGGUF.
Cobre tambem o conserto do extra_model_paths.yaml (G:/models) para flux-krea/flux-kontext.
Sem GPU."""
import json
from pathlib import Path

import pytest

from script_pipeline import generate_storyboards as gs

ROOT = Path(__file__).resolve().parents[1]


def test_gguf_klein_detectado_pelo_nome_e_generico_continua_flux():
    assert gs.detect_architecture("flux-2-klein-9b-Q8_0.gguf") == "flux-klein-gguf"
    assert gs.detect_architecture("flux-2-klein-9b-fp8.safetensors") == "flux"
    # "klein" sem .gguf (hipotetico) nao deve desviar -- so o par nome+extensao decide
    assert gs.detect_architecture("flux-2-klein-9b-fp8.safetensors").startswith("flux")


def test_engine_cadastrado_reaproveita_encoder_e_vae_do_flux():
    assert "flux-klein-gguf" in gs.IMAGE_ENGINES
    fp8, gguf = gs.IMAGE_ENGINES["flux"], gs.IMAGE_ENGINES["flux-klein-gguf"]
    assert gguf["clip"] == fp8["clip"] and gguf["vae"] == fp8["vae"]
    assert gguf["checkpoint"].endswith(".gguf")


def test_template_gguf_e_o_grafo_do_flux_so_com_unetloadergguf():
    assert gs.WORKFLOW_TEMPLATES["flux-klein-gguf"].exists()
    flux = json.loads(gs.WORKFLOW_TEMPLATES["flux"].read_text(encoding="utf-8"))
    gguf = json.loads(gs.WORKFLOW_TEMPLATES["flux-klein-gguf"].read_text(encoding="utf-8"))
    assert set(flux) == set(gguf), "mesmos node IDs -- so o loader do node 1 muda"
    assert gguf["1"]["class_type"] == "UnetLoaderGGUF"
    assert "weight_dtype" not in gguf["1"]["inputs"]
    for node_id in set(flux) - {"1"}:
        assert flux[node_id] == gguf[node_id], f"node {node_id} divergiu do grafo flux"


def test_lora_e_espacial_tratam_gguf_como_flux():
    assert gs.ARCH_MODEL_CONSUMER["flux-klein-gguf"] == gs.ARCH_MODEL_CONSUMER["flux"]


def test_referencia_de_personagem_troca_o_loader_para_gguf(tmp_path, monkeypatch):
    """used_reference_template usa os templates character_flux_reference*.json, que so
    existem em UNETLoader fp8 -- para o motor GGUF o node 1 precisa virar UnetLoaderGGUF
    tambem (achado ao ligar o motor, 2026-09-28)."""
    captured = {}

    def fake_submit_and_wait(server, workflow, log=print):
        captured["workflow"] = workflow
        return None  # aborta cedo (sem GPU); so queremos inspecionar o grafo montado

    monkeypatch.setattr(gs, "submit_and_wait", fake_submit_and_wait)
    from PIL import Image
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), "gray").save(ref)
    scene = {"index": 0, "visual_prompt": "a person standing"}
    gs.generate_scene_storyboard(
        scene, {}, server="http://127.0.0.1:8188", checkpoint="flux-2-klein-9b-Q8_0.gguf",
        width=64, height=64, steps=8, cfg=1.0, seed=1, out_path=tmp_path / "out.png",
        clip="Qwen3-8B-FP8-native-bf16.safetensors", vae="flux2-vae.safetensors",
        guidance=3.5, reference_image=str(ref), log=lambda m: None)
    wf = captured["workflow"]
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wf["1"]["inputs"]["unet_name"] == "flux-2-klein-9b-Q8_0.gguf"


def test_extra_model_paths_mapeia_g_models_para_krea_e_kontext():
    """Achado 2026-09-27: flux-krea/flux-kontext apontavam para arquivos que so existiam
    em G:/models, nunca visiveis ao ComfyUI (nenhum extra_model_paths cobria aquele disco).
    Conserto: secao g_models. So confere que a secao existe e cobre os campos usados por
    esses dois motores -- nao depende do G: estar montado para rodar o teste."""
    import yaml
    cfg = yaml.safe_load((ROOT / "ComfyUI" / "extra_model_paths.yaml").read_text(encoding="utf-8"))
    assert "g_models" in cfg
    secao = cfg["g_models"]
    for campo in ("checkpoints", "diffusion_models", "unet", "text_encoders", "vae"):
        assert campo in secao
