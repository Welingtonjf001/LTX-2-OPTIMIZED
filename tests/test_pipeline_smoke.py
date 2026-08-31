# -*- coding: utf-8 -*-
"""Suíte rápida, sem GPU e sem rede -- roda em segundos.

Cobre as funções PURAS dos pontos que mais quebraram em silêncio esta sessão:
seleção de saída do ComfyUI, deteccão de corte do doctor, score de prompt,
seleção de índices e resolução de estilo por plano. Não cobre nada que precise
de GPU, Ollama ou ComfyUI no ar -- isso é fumaça manual (MEMORIAL.md §8), não
CI.

    .venv/Scripts/python.exe -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline.prompt_polish import score_prompt  # noqa: E402
from script_pipeline.render_shots import parse_indices, _audio_key  # noqa: E402
from script_pipeline.shot_plan import (  # noqa: E402
    _cobertura_de_fala, emocao_visivel, parse_style_changes, style_for_shot,
)
from video_doctor import (  # noqa: E402
    classify, color_hist, detect_cuts, drop_cut_segments, robust_z,
)


# ---------------------------------------------------------------------------
# video_doctor: cortes vs defeito (MEMORIAL 3.36.1)
# ---------------------------------------------------------------------------

def _hist_solido(cor: tuple) -> np.ndarray:
    """Histograma sintético de uma imagem de cor sólida -- suficiente para
    exercitar compareHist sem decodificar vídeo nenhum."""
    img = np.full((8, 8, 3), cor, dtype=np.uint8)
    return color_hist(img)


def test_detect_cuts_acha_troca_de_plano_persistente():
    # 6 quadros azuis, corte no 3 (persiste ate o fim) -- e a assinatura de um
    # CORTE: a mudanca continua no quadro seguinte tambem. z_thr baixo de
    # proposito: cores solidas sinteticas nao tem o ruido de compressao que
    # calibra o limiar padrao (6.0, medido contra vídeo real em 3.36.1) --
    # aqui o que se testa é a lógica de PERSISTÊNCIA, não a calibração.
    hists = [_hist_solido((200, 50, 50))] * 3 + [_hist_solido((50, 50, 200))] * 3
    metrics = {"_hists": hists}
    cortes = detect_cuts(metrics, z_thr=2.0, min_dist=0.01)
    assert cortes == [3]  # indice t da metrica descreve o frame t (measure()'s +1)


def test_detect_cuts_ignora_pico_que_nao_persiste():
    # Um unico quadro diferente que volta ao normal -- defeito, nao corte.
    # PEGOU UM BUG REAL ao escrever este teste (ver MEMORIAL 3.39): a
    # transicao de ENTRADA (azul->vermelho) e filtrada corretamente pelo
    # d_salto original, mas a de SAIDA (vermelho->azul) usava o proprio
    # quadro anomalo como ancora "antes" e sempre parecia persistir. Corrigido
    # com d_lead: a ancora "antes" tambem precisa ja ser estavel.
    azul, vermelho = _hist_solido((200, 50, 50)), _hist_solido((50, 50, 200))
    hists = [azul, azul, azul, vermelho, azul, azul]
    cortes = detect_cuts({"_hists": hists}, z_thr=2.0, min_dist=0.01)
    assert cortes == []


def test_drop_cut_segments_remove_so_o_que_toca_a_fronteira():
    plano = [
        {"start": 73, "end": 73, "z_residual": 224.4},   # em cima do corte
        {"start": 189, "end": 196, "z_residual": 2.87},  # defeito real, longe
    ]
    mantido, descartado = drop_cut_segments(plano, cuts=[73])
    assert mantido == [plano[1]]
    assert descartado == [plano[0]]


def test_drop_cut_segments_sem_cortes_nao_muda_nada():
    plano = [{"start": 10, "end": 12, "z_residual": 8.0}]
    mantido, descartado = drop_cut_segments(plano, cuts=[])
    assert mantido == plano
    assert descartado == []


def test_robust_z_mediana_zero():
    z = robust_z([5.0, 5.0, 5.0, 5.0, 50.0])
    assert z[-1] > z[0]  # o outlier pontua mais alto


def test_classify_nao_reporta_nada_em_serie_estavel():
    # 20 quadros com residuo constante -- nenhum z-score deve destacar-se.
    n = 20
    metrics = {
        "residual": [3.0] * n, "residual_local": [1.0] * n,
        "residual_local_raw": [1.0] * n, "luma_delta": [0.1] * n,
        "hf_delta": [0.5] * n, "_worst_cell": [[0, 0]] * n,
    }
    assert classify(metrics) == []


# ---------------------------------------------------------------------------
# prompt_polish: as 9 perguntas (MEMORIAL 3.37)
# ---------------------------------------------------------------------------

def test_score_prompt_pipeline_fragmentado_pontua_baixo():
    frag = (
        "Lyra gestures with her crystal staff, her expression urgent as she "
        "speaks to Thoren. A young woman with long silver hair, wearing a "
        "dark blue cloak and holding a crystal staff. the camera pushes in "
        "slowly. cinematic high-fantasy, polished 3D animation, sharp "
        "details, dramatic lighting."
    )
    r = score_prompt(frag)
    assert r["total"] < r["max"]
    assert "beat_final" in r["faltando"]
    assert "audio" in r["faltando"]


def test_score_prompt_completo_pontua_alto():
    cheio = (
        "Lyra touches a glowing rune, turns toward Thoren and says urgently, "
        '"Thoren, as runas despertaram." Lyra closes her mouth. A circular '
        "ancient stone chamber lit by blue moonlight. Cinematic high-fantasy, "
        "polished 3D animation. The camera pushes in slowly while circling. "
        "Audio: wind hums through the chamber; restrained orchestral music "
        "builds. Lyra looks up as the moonlight begins fading."
    )
    r = score_prompt(cheio, has_dialogue=True)
    assert r["total"] >= r["max"] - 2  # tolera 1-2 itens de fronteira


def test_score_prompt_acao_nao_e_punida_por_nao_ter_fala():
    sem_fala = "Thoren grips his sword tightly. cinematic 3D animation."
    r = score_prompt(sem_fala, has_dialogue=False)
    assert "fala_citada" not in r["itens"]


# ---------------------------------------------------------------------------
# render_shots: selecao de indices e chave de reuso por audio (MEMORIAL 3.36.2)
# ---------------------------------------------------------------------------

def test_parse_indices_aceita_lista_e_faixa():
    assert parse_indices("0,2,4", total=6) == [0, 2, 4]
    assert parse_indices("1-3", total=6) == [1, 2, 3]
    assert parse_indices(None, total=4) == [0, 1, 2, 3]


def test_audio_key_muda_com_o_arquivo(tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x" * 100)
    b = tmp_path / "b.wav"
    b.write_bytes(b"x" * 200)
    assert _audio_key(str(a)) != _audio_key(str(b))
    assert _audio_key(None) == "sem-audio"


# ---------------------------------------------------------------------------
# shot_plan: escada de fala e emocao visivel (MEMORIAL 3.36.3/3.36.4)
# ---------------------------------------------------------------------------

def test_cobertura_de_fala_exclui_enquadramento_aberto():
    escada = _cobertura_de_fala(["wide", "medium", "close"])
    assert "wide" not in escada
    assert set(escada) == {"medium", "close"}


def test_cobertura_de_fala_garante_dois_degraus():
    escada = _cobertura_de_fala(["wide"])  # so tem aberto
    assert len(escada) >= 2


def test_emocao_visivel_traduz_slug_conhecido():
    v = emocao_visivel("com_medo")
    assert v and "wide" in v  # "eyes wide and darting..."


def test_emocao_visivel_vazio_e_seguro():
    assert emocao_visivel(None) == ""
    assert emocao_visivel("slug-inexistente-xyz") == ""


def test_parse_style_changes_por_cena_e_por_plano():
    # Semantica de "mudanca de bobina": a marca vale dali PARA A FRENTE ate
    # aparecer outra -- atravessa o fim da cena. Uma marca em 1: nao se limita
    # a cena 1; ela e o estilo vigente ate 3.2 aparecer (docstring de
    # parse_style_changes).
    changes = parse_style_changes("1:nervoso,3.2:intimista")
    assert style_for_shot(1, 1, "classico", changes) == "nervoso"
    assert style_for_shot(3, 1, "classico", changes) == "nervoso"  # antes do 3.2, mas depois do 1
    assert style_for_shot(3, 2, "classico", changes) == "intimista"
    assert style_for_shot(5, 1, "classico", changes) == "intimista"  # vigora ate a proxima marca


def test_parse_style_changes_vazio_devolve_dict_vazio():
    assert parse_style_changes(None) == {}
    assert parse_style_changes("") == {}


# ---------------------------------------------------------------------------
# ltx25_backend: construcao do workflow, sem submeter nada
# ---------------------------------------------------------------------------

def test_build_workflow_nao_toca_rede():
    import ltx25_backend as b
    api = b.build_workflow("a test prompt", width=768, height=512, num_frames=25)
    assert isinstance(api, dict)
    assert b.N_SAVE in api
    assert api[b.N_SAVE]["inputs"]["filename_prefix"] == "ltx25"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
