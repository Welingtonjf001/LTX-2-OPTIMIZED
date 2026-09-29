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


def test_meio_sem_evidencia_no_texto_fonte_e_descartado():
    """ACHADO 2026-09-29 (teste real, prompt LTX-2.3 'Fantasia'): o LLM inventou
    'polished 3D animation' -- frase DIFERENTE do exemplo literal, mesma familia de
    alucinacao -- para uma cena que a fonte descrevia como fotorrealista/cinematica,
    sem citar nenhum meio de animacao. Com `source_text` (o texto ANTES da
    reestruturacao), o guard pega isso mesmo sem ser o exemplo exato."""
    fonte = "A cinematic high-fantasy scene in a circular ancient stone chamber. " \
           "Lyra and Thoren face a sealed archway."
    texto = "ESTILO VISUAL: polished 3D animation\n\nINT. CAMARA - NOITE\n\nLyra entra."
    limpo, meio = extract_art_direction(texto, source_text=fonte)
    assert meio == ""
    assert "ESTILO VISUAL" not in limpo


def test_meio_com_evidencia_real_no_texto_fonte_continua_valido():
    """Se a fonte REALMENTE menciona o meio (aqui, 'anime'), o guard nao deve
    descartar -- so pega o caso de invencao sem base nenhuma."""
    fonte = "An anime-style opening scene, hand-drawn cel animation aesthetic."
    texto = "ESTILO VISUAL: hand-drawn cel animation, anime aesthetic\n\nINT. SALA - DIA\n\nMiku entra."
    limpo, meio = extract_art_direction(texto, source_text=fonte)
    assert meio == "hand-drawn cel animation, anime aesthetic"


def test_meio_fotorrealista_sem_termo_animado_nao_precisa_de_evidencia():
    """Um meio puramente fotorrealista (sem termo de animacao/ilustracao) nao cai
    nesta checagem -- so o termo de animacao/ilustracao precisa de base na fonte."""
    fonte = "A quiet dialogue scene between two old friends."
    texto = "ESTILO VISUAL: photorealistic, natural lighting, 35mm film\n\nINT. CAFE - DIA\n\nElas conversam."
    limpo, meio = extract_art_direction(texto, source_text=fonte)
    assert meio == "photorealistic, natural lighting, 35mm film"
