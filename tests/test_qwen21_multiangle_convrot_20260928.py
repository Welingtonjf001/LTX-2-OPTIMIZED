"""Folha de personagem 4-angulos+close via checkpoint dedicado `qwen_image_2.1_int8_convrot.safetensors`
(NAO GGUF) + LoRA Viggle Turbo 4-step -- workflow do usuario "Qwen Image 2.1 Image Edit Viggle 4-Step"
(2026-09-28). Sem GPU: so confere que o grafo API monta certo (class_type/UNETLoader vs UnetLoaderGGUF,
LoRA, ModelSamplingFlux, referencias) contra o que o node schema real do Qwen-Image-2.1 ComfyUI 0.37.0
exige (TextEncodeQwenImage21.images e Autogrow `image_1`..`image_16`, ver comfy_extras/nodes_qwen.py)."""
import qwen_image21_comfy_backend as b


def test_unet_gguf_false_usa_unetloader_simples():
    g = b.build_workflow("p", [], width=512, height=512, steps=4, seed=1, unet="x.safetensors", unet_gguf=False)
    assert g["1"]["class_type"] == "UNETLoader"
    assert g["1"]["inputs"]["unet_name"] == "x.safetensors"
    assert g["1"]["inputs"]["weight_dtype"] == "default"


def test_unet_gguf_true_e_o_padrao_e_mantem_compatibilidade():
    g = b.build_workflow("p", [], width=512, height=512, steps=4, seed=1)
    assert g["1"]["class_type"] == "UnetLoaderGGUF"
    assert "weight_dtype" not in g["1"]["inputs"]


def test_lora_attention_backend_e_model_sampling_flux_encadeiam_o_model():
    g = b.build_workflow("p", [], width=1536, height=1536, steps=4, seed=1, unet_gguf=False,
                         lora=("viggle.safetensors", 1.0), attention_backend="comfy kitchen attention",
                         model_sampling_flux=(0.69, 0.5))
    assert g["10"]["inputs"]["model"] == ["1", 0]
    assert g["11"]["class_type"] == "ModelAttentionBackend"
    assert g["11"]["inputs"] == {"model": ["10", 0], "attention": "comfy kitchen attention"}
    assert g["12"]["class_type"] == "ModelSamplingFlux"
    assert g["12"]["inputs"] == {"model": ["11", 0], "max_shift": 0.69, "base_shift": 0.5,
                                  "width": 1536, "height": 1536}
    assert g["2"]["inputs"]["model"] == ["12", 0]


def test_sem_lora_attention_ou_sampling_o_cache_pega_direto_do_unet():
    g = b.build_workflow("p", [], width=512, height=512, steps=4, seed=1)
    assert g["2"]["inputs"]["model"] == ["1", 0]


def test_cfg_customizavel_mas_default_continua_1():
    g_default = b.build_workflow("p", [], width=512, height=512, steps=4, seed=1)
    assert g_default["7"]["inputs"]["cfg"] == 1.0
    g_custom = b.build_workflow("p", [], width=512, height=512, steps=4, seed=1, cfg=2.5)
    assert g_custom["7"]["inputs"]["cfg"] == 2.5


def test_ate_tres_referencias_viram_image_1_a_3_no_text_encode():
    g = b.build_workflow("p", ["a.png", "b.png", "c.png"], width=512, height=512, steps=4, seed=1)
    enc = g["5"]["inputs"]
    assert enc["images.image_1"] == ["img1", 0]
    assert enc["images.image_2"] == ["img2", 0]
    assert enc["images.image_3"] == ["img3", 0]


def test_generate_character_sheet_convrot_usa_checkpoint_e_lora_certos(monkeypatch):
    captured = {}

    def fake_generate(prompt, out_path, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(b, "generate", fake_generate)
    ok = b.generate_character_sheet_convrot(["ref1.png", "ref2.png"], "out.png", outfit="a red jacket")
    assert ok is True
    assert captured["kwargs"]["unet"] == b.CONVROT_UNET
    assert captured["kwargs"]["unet_gguf"] is False
    assert captured["kwargs"]["lora"] == (b.VIGGLE_TURBO_LORA, 1.0)
    assert captured["kwargs"]["reference_images"] == ["ref1.png", "ref2.png"]
    assert "4 full-body angles" in captured["prompt"]
    assert "Change their outfit to a red jacket" in captured["prompt"]


def test_generate_character_sheet_convrot_limita_a_3_referencias(monkeypatch):
    captured = {}
    monkeypatch.setattr(b, "generate", lambda prompt, out_path, **kw: captured.setdefault("kwargs", kw) or True)
    b.generate_character_sheet_convrot(["a.png", "b.png", "c.png", "d.png"], "out.png")
    assert captured["kwargs"]["reference_images"] == ["a.png", "b.png", "c.png"]
