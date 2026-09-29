"""ACHADO 2026-09-29 (REENTRY WINDOW): `_enforce_descriptor_fidelity` (cast_characters.py) so
media SOBREPOSICAO GERAL de palavras entre a apresentacao do roteiro e o descritor gerado pelo
LLM -- pega invencao TOTAL (a Lyra virando jaqueta de couro), mas nao pegava troca PONTUAL de um
atributo decisivo quando o resto do vocabulario compartilhado (ambos falam de "suit"/"flight"/
"chest") ja inflava a taxa acima do limiar de 20%. Caso real: ANA tinha "long dark hair,
charcoal suit" no roteiro e saiu "short, dark-brown hair... dark-blue suit" no cast.json --
30% de sobreposicao (passou), mas comprimento de cabelo E cor da roupa TROCADOS. Sem GPU/LLM."""
from script_pipeline import cast_characters as cc


def test_ana_comprimento_de_cabelo_e_cor_da_roupa_trocados_e_pego():
    anchor = ("ANA (long dark hair tucked beneath a fitted flight cap, charcoal pressure suit "
              "with blue piping, sealed collar, and chest-mounted oxygen connector) grips the "
              "control yoke with her right hand")
    gerado = ("ANA has short, dark-brown hair cut in a pixie cut, a youthful face with sharp "
              "cheekbones, and a lean build. She wears a dark-blue wool flight suit with "
              "reinforced leather patches on the elbows and a silver nameplate on the chest.")
    resultado = cc._enforce_descriptor_fidelity("ANA", gerado, anchor, log=lambda m: None)
    # a 30% de sobreposicao geral o teste ANTIGO (so ratio < 0.2) deixaria passar;
    # a contradicao de atributo (comprimento/cor) tem que forcar a troca pelo texto original.
    assert resultado != gerado
    assert "long dark hair" in resultado or "long" in resultado


def test_giana_cor_de_cabelo_trocada_e_pega():
    anchor = ("GIANA (short brown hair tucked behind her ears, slate-gray utility suit with "
              "reinforced knees, padded gloves, and a tether clipped to her waist) floats "
              "sideways into the open panel")
    gerado = ("GIANA has long, wavy auburn hair tied back in a practical ponytail, a serene "
              "face with a round shape, and a medium build. She wears a dark-gray jumpsuit "
              "made of durable, fire-retardant material.")
    resultado = cc._enforce_descriptor_fidelity("GIANA", gerado, anchor, log=lambda m: None)
    assert resultado != gerado


def test_midori_cor_de_roupa_trocada_e_pega():
    anchor = ("MIDORI (long, straight dark hair, forest-green flight jacket over a dark shirt, "
              "communications headset and a slim microphone near her cheek) leans toward the "
              "telemetry display")
    gerado = ("MIDORI has shoulder-length, straight black hair with a side part, a mature face "
              "with a square jawline. She wears a white lab coat over a light-blue shirt and "
              "black pants.")
    resultado = cc._enforce_descriptor_fidelity("MIDORI", gerado, anchor, log=lambda m: None)
    assert resultado != gerado


def test_paraphrase_sem_contradicao_nao_e_trocada():
    """Descritor que so reformula (mesma cor, mesmo comprimento) nao deve ser descartado --
    a checagem de atributo e para CONTRADICAO, nao para exigir palavras identicas."""
    anchor = "AKEMI (long black hair, dark-blue jacket) draws her sword"
    gerado = "AKEMI has long, straight black hair and wears a dark-blue tactical jacket with pockets."
    resultado = cc._enforce_descriptor_fidelity("AKEMI", gerado, anchor, log=lambda m: None)
    assert resultado == gerado


def test_cor_substring_nao_conta_como_contradicao():
    """"brown" dentro de "dark-brown" e a MESMA cor com detalhe a mais, nao uma troca."""
    anchor = "TEST (brown hair, red coat) walks in"
    gerado = "TEST has dark-brown hair and wears a red coat."
    resultado = cc._enforce_descriptor_fidelity("TEST", gerado, anchor, log=lambda m: None)
    assert resultado == gerado


def test_hair_length_bucket():
    assert cc._hair_length_bucket("long dark hair") == "long"
    assert cc._hair_length_bucket("short brown hair") == "short"
    assert cc._hair_length_bucket("a pixie cut hair style") == "short"
    assert cc._hair_length_bucket("no mention here") is None


def test_attribute_contradiction_reports_reason():
    anchor = "long dark hair, charcoal suit"
    gerado = "short dark-brown hair, dark-blue suit"
    motivo = cc._attribute_contradiction(anchor, gerado)
    assert motivo is not None
    assert "cabelo" in motivo
