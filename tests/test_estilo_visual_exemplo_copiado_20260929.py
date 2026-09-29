"""ACHADO 2026-09-29 (REENTRY WINDOW): o exemplo de sintaxe no prompt do
`prose_to_screenplay` ("ESTILO VISUAL: polished hand-drawn cel animation, sharp ink
lines") foi copiado literalmente pelo qwen2.5:32b para um roteiro sci-fi que nunca
mencionou nenhum meio visual -- todo still saiu ilustrado em vez de fotorrealista,
derrubando a consistencia facial (ArcFace) contra fotos reais de ator pra perto de
zero em TODOS os personagens. `extract_art_direction` agora descarta o valor quando
ele bate exatamente (case-insensitive) com o exemplo do prompt."""
from script_pipeline.parse_screenplay import extract_art_direction, _ESTILO_VISUAL_EXEMPLO_LITERAL


def test_exemplo_literal_do_prompt_e_descartado():
    texto = f"ESTILO VISUAL: {_ESTILO_VISUAL_EXEMPLO_LITERAL}\n\nINT. CABINE - DIA\n\nAna entra."
    limpo, meio = extract_art_direction(texto)
    assert meio == ""
    assert "ESTILO VISUAL" not in limpo


def test_exemplo_literal_case_insensitive_tambem_e_descartado():
    texto = f"ESTILO VISUAL: {_ESTILO_VISUAL_EXEMPLO_LITERAL.upper()}\n\nINT. CABINE - DIA\n\nAna entra."
    _, meio = extract_art_direction(texto)
    assert meio == ""


def test_estilo_visual_real_e_diferente_do_exemplo_continua_valido():
    texto = "ESTILO VISUAL: watercolor storybook illustration, soft pastel colors\n\nINT. CABINE - DIA\n\nAna entra."
    limpo, meio = extract_art_direction(texto)
    assert meio == "watercolor storybook illustration, soft pastel colors"
    assert "ESTILO VISUAL" not in limpo


def test_sem_linha_estilo_visual_devolve_vazio():
    texto = "INT. CABINE - DIA\n\nAna entra."
    limpo, meio = extract_art_direction(texto)
    assert meio == ""
    assert limpo == texto
