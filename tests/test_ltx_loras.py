# -*- coding: utf-8 -*-
"""LoRAs de video: catalogo, referencias IC e enxertos no grafo 2.5 -- sem GPU e sem
ComfyUI no ar.

O grafo base (fixture) e o que comfy_workflow_tool.convert() produz do workflow oficial
LTX-2.5_T2V_I2V_Single_Stage_Distilled contra o /object_info real deste ComfyUI: os
testes exercitam os ids e links de verdade, nao um grafo inventado. Regenerar a fixture
quando o workflow oficial mudar (os ids N_* do backend quebram junto).

    .venv/Scripts/python.exe -m pytest tests/test_ltx_loras.py -q
"""
import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ltx25_backend as b  # noqa: E402
import ltx_loras  # noqa: E402
from script_pipeline import ic_references as icr  # noqa: E402

BASE = json.loads((ROOT / "tests" / "fixtures" / "ltx25_base_api_single.json").read_text(encoding="utf-8"))
SEP = "5516:4845"      # LTXVSeparateAVLatent do sampler principal
DECODE = "5518:5538"   # VAEDecodeTiled do video


@pytest.fixture
def grafo(monkeypatch):
    """build_workflow sem servidor: grafo base da fixture e LoRAs 'instalados'."""
    monkeypatch.setattr(b, "base_api", lambda two_stage=False: copy.deepcopy(BASE))
    monkeypatch.setattr(ltx_loras, "installed_path", lambda nome: Path(nome))
    monkeypatch.setattr(ltx_loras, "reference_downscale", lambda nome: 1.0)
    return b


def _links_para(api, link):
    return [(nid, k) for nid, no in api.items() for k, v in no["inputs"].items() if v == link]


# ---------------------------------------------------------------------------
# catalogo
# ---------------------------------------------------------------------------

def test_parse_lora_arg_chave_forca_e_caminho_windows():
    assert ltx_loras.parse_lora_arg("better-human-motion") == ("ltx-2.3-better-human-motion-v2.safetensors", 0.6)
    assert ltx_loras.parse_lora_arg("better-human-motion:0.3") == ("ltx-2.3-better-human-motion-v2.safetensors", 0.3)
    # O ':' do drive nao pode ser lido como separador de forca.
    assert ltx_loras.parse_lora_arg(r"E:\x\meu.safetensors") == ("meu.safetensors", 1.0)
    assert ltx_loras.parse_lora_arg(r"E:\x\meu.safetensors:0.5") == ("meu.safetensors", 0.5)


def test_apply_triggers_uma_vez_e_na_posicao():
    trans = "ltx-2.3-transition-joyfox.safetensors"
    cine = "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors"
    assert ltx_loras.apply_triggers("a cat.", [trans]) == "a cat, zhuanchang"
    assert ltx_loras.apply_triggers("a cat, zhuanchang", [trans]) == "a cat, zhuanchang"
    assert ltx_loras.apply_triggers("a candle", [cine]) == "CINEMAGRAPH_MOTION, a candle"
    assert ltx_loras.apply_triggers("sem gatilho", ["qualquer.safetensors"]) == "sem gatilho"


def test_chaves_unicas_e_tipos_conhecidos():
    assert len(ltx_loras.BY_KEY) == len(ltx_loras.SPECS)
    assert len(ltx_loras.BY_LOCAL) == len(ltx_loras.SPECS)  # ComfyUI lista por nome
    assert all(s.kind in ltx_loras.KINDS for s in ltx_loras.SPECS)


def test_compat_25_union_control_local():
    p = ltx_loras.installed_path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
    if p is None or not ltx_loras.TRANSFORMER_25.exists():
        pytest.skip("LoRA ou transformer 2.5 ausente nesta maquina")
    c = ltx_loras.compat_25(p)
    assert c["faltando_no_25"] == 0 and c["shape_errado"] == 0 and c["alvos_ok"] == 480
    assert c["reference_downscale_factor"] == "2"


# ---------------------------------------------------------------------------
# referencias IC
# ---------------------------------------------------------------------------

def test_msr_alocacao_dois_sujeitos_65_quadros():
    # 9 latentes: o protagonista fica com 5 (quadros 0-32), o segundo com 3 (33-56) e
    # o cenario com o ultimo (57-64).
    assert icr.runs(icr.msr_frame_assignment(2, 65)) == [(0, 33), (1, 24), (2, 8)]


def test_msr_mais_sujeitos_que_latentes_divide_por_quadros():
    corridas = icr.runs(icr.msr_frame_assignment(4, 17))
    assert corridas == [(0, 2), (1, 2), (2, 2), (3, 3), (4, 8)]
    assert sum(n for _, n in corridas) == 17


def test_pick_msr_frame_count_nunca_passa_do_clipe():
    assert icr.pick_msr_frame_count(81) == 65
    assert icr.pick_msr_frame_count(49) == 49
    assert icr.pick_msr_frame_count(20) == 17
    assert icr.pick_msr_frame_count(9) is None


def test_folha_e_guia_msr_no_formato_do_treino(tmp_path):
    retrato = tmp_path / "retrato.png"
    Image.new("RGB", (300, 600), (200, 30, 30)).save(retrato)
    cena = tmp_path / "cena.png"
    Image.new("RGB", (1920, 1080), (30, 120, 30)).save(cena)

    folha = Image.open(icr.build_ingredients_sheet([str(retrato), str(cena)], 960, 544, tmp_path / "f.png"))
    assert folha.size == (960, 544) and folha.getpixel((2, 2)) == (0, 0, 0)  # fundo PRETO

    corridas, quadros = icr.build_msr_guide([str(retrato)], str(cena), 960, 544, 81, tmp_path / "msr")
    assert quadros == 65 and sum(n for _, n in corridas) == 65
    sujeito = Image.open(corridas[0][0])
    assert sujeito.size == (960, 544) and sujeito.getpixel((2, 2)) == (255, 255, 255)  # fundo BRANCO
    assert Image.open(corridas[-1][0]).getpixel((2, 2)) == (30, 120, 30)  # cenario cobre o quadro


def test_shot_ic_spec_ingredients_e_msr_com_referencia(tmp_path):
    folha_personagem = tmp_path / "ANA.png"
    Image.new("RGB", (512, 768), (90, 60, 40)).save(folha_personagem)
    still = tmp_path / "still.png"
    Image.new("RGB", (960, 544), (10, 10, 60)).save(still)
    shot = {"video_prompt": "Ana walks.", "subject": "ANA", "co_subject": ""}
    refs = {"ANA": str(folha_personagem)}

    # A locacao da cena NAO entra por padrao: plano aberto costuma ter outra personagem.
    ic, prompt, chave = icr.shot_ic_spec("ingredients", shot, str(still), refs, str(still),
                                         {"ANA": "ANA is tall. She wears red."},
                                         width=960, height=544, num_frames=81,
                                         work_dir=tmp_path / "ing", lora="ing.safetensors")
    assert ic["frames"][0][1] == 81 and ic["lora"] == "ing.safetensors"
    # Descritor so ate a primeira frase: o video_prompt ja carrega o paragrafo inteiro.
    assert prompt == "Reference sheet: panel 1 shows ANA, ANA is tall. Generated video: Ana walks."
    assert chave.startswith("|ic=ingredients:ing.safetensors:1:1:ANA.png")

    ic, prompt, _ = icr.shot_ic_spec("msr", shot, str(still), refs, None, {},
                                     width=960, height=544, num_frames=81,
                                     work_dir=tmp_path / "msr", lora="msr.safetensors")
    assert sum(n for _, n in ic["frames"]) == 65
    assert prompt == "Image 1: ANA. Image 2: the location and background. Ana walks."


def test_shot_ic_spec_sem_referencia_nao_liga(tmp_path):
    shot = {"video_prompt": "p", "subject": "ANA", "co_subject": ""}
    ic, prompt, chave = icr.shot_ic_spec("ingredients", shot, "still.png", {}, None, {},
                                         width=960, height=544, num_frames=81,
                                         work_dir=tmp_path, lora="x.safetensors")
    assert ic is None and prompt == "p" and chave == ""


# ---------------------------------------------------------------------------
# enxertos no grafo 2.5
# ---------------------------------------------------------------------------

def test_loras_encadeiam_entre_loader_e_guider(grafo):
    api = grafo.build_workflow("p", num_frames=81, loras=[("a.safetensors", 0.6), ("b.safetensors", 0.4)])
    assert api["9500"]["inputs"]["model"] == [grafo.N_UNET, 0]
    assert api["9501"]["inputs"]["model"] == ["9500", 0]
    assert api[grafo.N_CFG_GUIDER]["inputs"]["model"] == ["9501", 0]
    # Ninguem alem do primeiro LoRA pode ler o transformer cru.
    assert _links_para(api, [grafo.N_UNET, 0]) == [("9500", "model")]


def test_ic_lora_guia_no_latente_de_video_e_crop_antes_do_decode(grafo):
    ic = {"lora": "ing.safetensors", "frames": [("C:/x/folha.png", 81)], "guide_strength": 1.0}
    api = grafo.build_workflow("p", num_frames=81, image_path="still.png", ic_lora=ic)
    guider = api[grafo.N_CFG_GUIDER]["inputs"]
    assert guider["model"] == [grafo.N_IC_LOADER, 0]
    guia = api[grafo.N_IC_GUIDE]["inputs"]
    assert guia["latent"] == [grafo.N_IMG2VID, 0]          # depois do primeiro quadro
    assert guia["latent_downscale_factor"] == [grafo.N_IC_LOADER, 1]
    assert api[grafo.N_CONCAT]["inputs"]["video_latent"] == [grafo.N_IC_GUIDE, 2]
    assert guider["positive"] == [grafo.N_IC_GUIDE, 0]
    assert api["9700"]["inputs"]["image"] == "folha.png" and api["9730"]["inputs"]["amount"] == 81
    # Sem o crop, a guia do tamanho do clipe dobraria os quadros decodificados.
    assert api[grafo.N_CROP_GUIDES]["inputs"]["latent"] == [SEP, 0]
    assert api[DECODE]["inputs"]["samples"] == [grafo.N_CROP_GUIDES, 2]


def test_ic_lora_com_audio_mascara_embrulha_a_guia(grafo):
    ic = {"lora": "ing.safetensors", "frames": [("folha.png", 81)]}
    api = grafo.build_workflow("p", num_frames=81, ic_lora=ic, audio_conditioning="fala.wav")
    mascara = api["9402"]["inputs"]
    assert mascara["av_latent"] == [grafo.N_CONCAT, 0]
    assert mascara["positive"] == [grafo.N_IC_GUIDE, 0]
    assert mascara["model"] == [grafo.N_IC_LOADER, 0]
    assert api[grafo.N_CROP_GUIDES]["inputs"]["positive"] == ["9402", 0]


def test_ic_lora_fator_2_com_audio_e_recusado(grafo, monkeypatch):
    monkeypatch.setattr(ltx_loras, "reference_downscale", lambda nome: 2.0)
    ic = {"lora": "union.safetensors", "video": "controle.mp4"}
    with pytest.raises(ValueError, match="fator"):
        grafo.build_workflow("p", num_frames=81, ic_lora=ic, audio_conditioning="fala.wav")
    # Sem trilha condicionada o mesmo IC-LoRA passa.
    api = grafo.build_workflow("p", num_frames=81, ic_lora=ic)
    assert api["9612"]["inputs"]["length"] == 81


def test_ic_lora_two_stage_e_recusado(grafo):
    with pytest.raises(ValueError, match="two-stage"):
        grafo.build_workflow("p", num_frames=81, two_stage=True,
                             ic_lora={"lora": "ing.safetensors", "frames": [("f.png", 81)]})
