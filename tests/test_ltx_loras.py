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


def test_pick_msr_frame_count_guia_no_maximo_um_terco_do_clipe():
    # MEDIDO 2026-09-13: guia de 65 num clipe de 73 virava keyframe (corte no quadro 57).
    assert icr.pick_msr_frame_count(251) == 65   # proporcao do workflow oficial
    assert icr.pick_msr_frame_count(129) == 41
    assert icr.pick_msr_frame_count(73) == 17
    assert icr.pick_msr_frame_count(49) is None  # curto demais: sem MSR
    assert icr.pick_msr_frame_count(9) is None


def test_folha_e_guia_msr_no_formato_do_treino(tmp_path):
    retrato = tmp_path / "retrato.png"
    Image.new("RGB", (300, 600), (200, 30, 30)).save(retrato)
    cena = tmp_path / "cena.png"
    Image.new("RGB", (1920, 1080), (30, 120, 30)).save(cena)

    folha = Image.open(icr.build_ingredients_sheet([str(retrato), str(cena)], 960, 544, tmp_path / "f.png"))
    assert folha.size == (960, 544) and folha.getpixel((2, 2)) == (0, 0, 0)  # fundo PRETO

    # 201 quadros: a guia de 65 so cabe em clipe >= ~3x (MSR_MAX_FRACAO).
    corridas, quadros = icr.build_msr_guide([str(retrato)], str(cena), 960, 544, 201, tmp_path / "msr")
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
    assert sum(n for _, n in ic["frames"]) == 25  # <= 1/3 de 81 (MSR_MAX_FRACAO)
    assert prompt == "Image 1: ANA. Image 2: the location and background. Ana walks."

    # Clipe curto demais para a guia caber em 1/3: sem MSR, so o still.
    ic, prompt, chave = icr.shot_ic_spec("msr", shot, str(still), refs, None, {},
                                         width=960, height=544, num_frames=49,
                                         work_dir=tmp_path / "msr_curto", lora="msr.safetensors")
    assert ic is None and prompt == "Ana walks." and chave == ""


def test_msr_nao_liga_em_plano_aberto(tmp_path):
    # MEDIDO 2026-09-13: still aberto + retratos de corpo inteiro na guia = o video larga
    # o still no quadro 1 (salto 0,39-0,46) nos 3 planos abertos; closes ficaram < 0,015.
    folha = tmp_path / "ANA.png"
    Image.new("RGB", (512, 768), (90, 60, 40)).save(folha)
    still = tmp_path / "still.png"
    Image.new("RGB", (960, 544), (10, 10, 60)).save(still)
    kw = dict(width=960, height=544, num_frames=129, lora="msr.safetensors")
    for enquadramento in ("wide", "full", "insert"):
        shot = {"video_prompt": "p", "subject": "ANA", "co_subject": "", "framing": enquadramento}
        ic, prompt, chave = icr.shot_ic_spec("msr", shot, str(still), {"ANA": str(folha)}, None, {},
                                             work_dir=tmp_path / enquadramento, **kw)
        assert ic is None and prompt == "p" and chave == ""
    shot = {"video_prompt": "p", "subject": "ANA", "co_subject": "", "framing": "close"}
    ic, _, _ = icr.shot_ic_spec("msr", shot, str(still), {"ANA": str(folha)}, None, {},
                                work_dir=tmp_path / "close", **kw)
    assert ic is not None


def test_shot_ic_spec_sem_referencia_nao_liga(tmp_path):
    shot = {"video_prompt": "p", "subject": "ANA", "co_subject": ""}
    ic, prompt, chave = icr.shot_ic_spec("ingredients", shot, "still.png", {}, None, {},
                                         width=960, height=544, num_frames=81,
                                         work_dir=tmp_path, lora="x.safetensors")
    assert ic is None and prompt == "p" and chave == ""


def test_shot_ic_spec_msr25_monta_pic_slots_e_background(tmp_path):
    """MSR 2.5 (slot embedding real, ComfyUI-LTX2.5-MSR) -- MEMORIAL 3.127-3.129: mecanismo
    diferente do 'msr' 2.3 (pseudo-video), devolve um dict no formato que ltx25_backend.
    generate(msr=...) espera, nao o formato 'frames' do ic_lora."""
    ana = tmp_path / "ANA.png"
    Image.new("RGB", (512, 768), (90, 60, 40)).save(ana)
    bea = tmp_path / "BEA.png"
    Image.new("RGB", (512, 768), (40, 90, 60)).save(bea)
    still = tmp_path / "still.png"
    Image.new("RGB", (960, 544), (10, 10, 60)).save(still)
    shot = {"video_prompt": "Ana and Bea walk.", "subject": "ANA", "co_subject": "BEA"}
    refs = {"ANA": str(ana), "BEA": str(bea)}

    ic, prompt, chave = icr.shot_ic_spec("msr25", shot, str(still), refs, None,
                                         {"ANA": "tall", "BEA": "short"},
                                         width=960, height=544, num_frames=81,
                                         work_dir=tmp_path / "msr25", lora="msr25.safetensors")
    assert ic["images"]["pic1"] == str(ana)
    assert ic["images"]["pic2"] == str(bea)
    assert ic["images"]["background"] == str(still)
    assert ic["lora"] == "msr25.safetensors"
    assert ic["reference_frames"] == "33"
    assert "frames" not in ic  # nao e o formato pseudo-video do ic_lora
    assert prompt == "Image 1: ANA, tall. Image 2: BEA, short. Image 3: the location and background. Ana and Bea walk."
    assert chave.startswith("|ic=msr25:msr25.safetensors:1:1:ANA.png:BEA.png")


def test_shot_ic_spec_msr25_ate_4_sujeitos(tmp_path):
    imgs = []
    for nome in ("A", "B", "C", "D", "E"):
        p = tmp_path / f"{nome}.png"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(p)
        imgs.append(p)
    still = tmp_path / "still.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(still)
    # shot_ic_spec so le subject/co_subject (no maximo 2 nomes) -- o teste cobre o
    # truncamento em 4 pics MESMO com menos de 4 refs (nao tem como este pipeline
    # pedir 5 hoje, mas _MSR_SLOTS do backend aceita ate 4 + background).
    shot = {"video_prompt": "p", "subject": "A", "co_subject": "B"}
    ic, _, _ = icr.shot_ic_spec("msr25", shot, str(still), {"A": str(imgs[0]), "B": str(imgs[1])}, None, {},
                                width=64, height=64, num_frames=81,
                                work_dir=tmp_path / "msr25b", lora="m.safetensors")
    assert set(ic["images"].keys()) == {"pic1", "pic2", "background"}


def test_msr25_NAO_bloqueia_enquadramento_aberto(tmp_path):
    """Ao contrario do MSR 2.3 (test_msr_nao_liga_em_plano_aberto acima), o msr25 NAO
    aplica MSR_SEM_ENQUADRAMENTO_ABERTO -- decisao deliberada (mecanismo diferente,
    testado com GPU real em I2V+wide sem reproduzir o vazamento do 2.3, MEMORIAL 3.129)."""
    ana = tmp_path / "ANA.png"
    Image.new("RGB", (512, 768), (90, 60, 40)).save(ana)
    still = tmp_path / "still.png"
    Image.new("RGB", (960, 544), (10, 10, 60)).save(still)
    for enquadramento in ("wide", "full", "insert", "establishing"):
        shot = {"video_prompt": "p", "subject": "ANA", "co_subject": "", "framing": enquadramento}
        ic, prompt, chave = icr.shot_ic_spec("msr25", shot, str(still), {"ANA": str(ana)}, None, {},
                                             width=960, height=544, num_frames=81,
                                             work_dir=tmp_path / enquadramento, lora="m.safetensors")
        assert ic is not None, f"msr25 nao deveria bloquear enquadramento {enquadramento!r}"


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


# ---------------------------------------------------------------------------
# MSR 2.5 (ComfyUI-LTX2.5-MSR) -- MEMORIAL 3.127/3.128, integracao nova
# ---------------------------------------------------------------------------

def test_msr_lora_carrega_e_guia_le_msr_parameters(grafo):
    msr = {"lora": "ltx-2.5-licon-msr-v2.safetensors", "strength": 1.0,
           "images": {"pic1": "C:/x/sujeito1.png", "background": "C:/x/cenario.png"},
           "guide_strength": 0.9, "reference_frames": "33"}
    api = grafo.build_workflow("p", num_frames=81, msr=msr)
    guider = api[grafo.N_CFG_GUIDER]["inputs"]
    assert guider["model"] == [grafo.N_MSR_LOADER, 0]
    assert api[grafo.N_MSR_LOADER]["inputs"]["lora_name"] == "ltx-2.5-licon-msr-v2.safetensors"
    assert api[grafo.N_MSR_LOADER]["inputs"]["strength_model"] == 1.0
    guia = api[grafo.N_MSR_GUIDE]["inputs"]
    assert guia["msr_parameters"] == [grafo.N_MSR_LOADER, 1]
    assert guia["strength"] == 0.9
    assert guia["reference_frames"] == "33"
    # pic1 e background viram LoadImage, pic2/pic3/pic4 ficam de fora (nao pedidos)
    pic1_id = guia["pic1"][0]
    bg_id = guia["background"][0]
    assert api[pic1_id]["inputs"]["image"] == "sujeito1.png"
    assert api[bg_id]["inputs"]["image"] == "cenario.png"
    assert "pic2" not in guia and "pic3" not in guia and "pic4" not in guia
    assert api[grafo.N_CONCAT]["inputs"]["video_latent"] == [grafo.N_MSR_GUIDE, 2]
    assert guider["positive"] == [grafo.N_MSR_GUIDE, 0]
    # crop antes do decode, mesmo padrao do ic_lora
    assert api[grafo.N_CROP_GUIDES]["inputs"]["latent"] == [SEP, 0]
    assert api[DECODE]["inputs"]["samples"] == [grafo.N_CROP_GUIDES, 2]


def test_msr_sem_pic1_e_recusado(grafo):
    with pytest.raises(ValueError, match="pic1"):
        grafo.build_workflow("p", num_frames=81,
                             msr={"lora": "m.safetensors", "images": {"background": "x.png"}})


def test_msr_e_ic_lora_juntos_e_recusado(grafo):
    with pytest.raises(ValueError, match="MESMO ponto"):
        grafo.build_workflow(
            "p", num_frames=81,
            ic_lora={"lora": "ing.safetensors", "frames": [("f.png", 81)]},
            msr={"lora": "m.safetensors", "images": {"pic1": "x.png"}})


def test_msr_two_stage_e_recusado(grafo):
    with pytest.raises(ValueError, match="two-stage"):
        grafo.build_workflow("p", num_frames=81, two_stage=True,
                             msr={"lora": "m.safetensors", "images": {"pic1": "x.png"}})


def test_msr_audio_ref_usa_o_mesmo_audio_vae_do_grafo(grafo):
    msr = {"lora": "m.safetensors",
           "images": {"pic1": "sujeito1.png", "pic2": "sujeito2.png"},
           "audio": {"audio_ref1": "voz1.wav", "audio_ref2": "voz2.wav"}}
    api = grafo.build_workflow("p", num_frames=81, msr=msr)
    guia = api[grafo.N_MSR_GUIDE]["inputs"]
    audio_vae = api["5514:3980"]["inputs"]["audio_vae"]
    enc1_id = guia["audio_ref1"][0]
    enc2_id = guia["audio_ref2"][0]
    assert api[enc1_id]["class_type"] == "LTXVAudioVAEEncode"
    assert api[enc1_id]["inputs"]["audio_vae"] == audio_vae
    load1_id = api[enc1_id]["inputs"]["audio"][0]
    assert api[load1_id]["inputs"]["audio"] == "voz1.wav"
    load2_id = api[api[enc2_id]["inputs"]["audio"][0]]["inputs"]["audio"]
    assert load2_id == "voz2.wav"


def test_msr_sem_audio_nao_adiciona_inputs_de_audio(grafo):
    msr = {"lora": "m.safetensors", "images": {"pic1": "x.png"}}
    api = grafo.build_workflow("p", num_frames=81, msr=msr)
    guia = api[grafo.N_MSR_GUIDE]["inputs"]
    assert "audio_ref1" not in guia and "audio_ref2" not in guia


def test_msr_encadeia_depois_de_loras_comuns(grafo):
    """msr nao pode sobrescrever loras comuns aplicados antes dele (bug do primeiro
    rascunho desta integracao: _apply_msr_lora usava sempre [n_unet, 0] direto)."""
    api = grafo.build_workflow(
        "p", num_frames=81, loras=[("a.safetensors", 0.6)],
        msr={"lora": "m.safetensors", "images": {"pic1": "x.png"}})
    assert api["9500"]["inputs"]["model"] == [grafo.N_UNET, 0]
    assert api[grafo.N_MSR_LOADER]["inputs"]["model"] == ["9500", 0]
    assert api[grafo.N_CFG_GUIDER]["inputs"]["model"] == [grafo.N_MSR_LOADER, 0]
