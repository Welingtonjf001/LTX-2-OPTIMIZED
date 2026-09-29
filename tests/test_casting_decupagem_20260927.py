"""Regras da auditoria do CERCO EM SEUL (2026-09-27): figurantes, voz por idade/genero,
enquadramento por conteudo e movimento. Sem GPU e sem LLM."""
from script_pipeline import cast_characters as cc
from script_pipeline import motion_conditioner as mc
from script_pipeline import shot_plan as sp


def test_idade_do_texto():
    assert cc._parse_age("late 20s, athletic") == 28
    assert cc._parse_age("early 40s with a chiseled face") == 42
    assert cc._parse_age("a 7-year-old girl") == 7
    assert cc._parse_age("elderly man with a cane") == 72
    assert cc._parse_age("short dark hair") is None


def test_voz_por_faixa_nunca_infantil_para_adulto():
    pool = ["F01_infantil_rachel", "F02_jovem_expressiva_kristin", "F03_adulta_clara_jodi",
            "F05_madura_nobre_ruth"]
    assert cc._escolhe_voz(pool, "adulto", set()) == "F03_adulta_clara_jodi"
    assert cc._escolhe_voz(pool, "maduro", set()) == "F05_madura_nobre_ruth"
    assert cc._escolhe_voz(pool, None, set()) != "F01_infantil_rachel"
    # voz ja usada perde para outra da mesma faixa
    assert cc._escolhe_voz(pool + ["F04_adulta_suave_amy"], "adulto",
                           {"F03_adulta_clara_jodi"}) == "F04_adulta_suave_amy"


def test_figurantes_sem_llm_juntam_idiomas():
    cenas = [{"action_text": "O presidente entra. O presidente acena.",
              "shot_list": [{"visual": "The President approaches the car"},
                            {"visual": "A waiter drops a tray"}]}]
    extras = cc.detect_extras(cenas, [], engine=None)
    assert "PRESIDENT" in extras
    assert set(extras["PRESIDENT"]["aliases"]) == {"presidente", "president"}
    assert "WAITER" not in extras  # uma aparicao so e fundo, nao elenco


def test_figurante_do_llm_ganha_apelidos_do_outro_idioma(monkeypatch):
    """MEDIDO 2026-09-27: o LLM devolveu so 'Presidente'/'motociclista'; as acoes em
    ingles ('the President', 'the motorcycle rider') nao seriam reconhecidas."""
    import script_pipeline.story_structure as ss
    monkeypatch.setattr(ss, "_call_ollama", lambda *a, **k: {"extras": [
        {"name": "PRESIDENT", "aliases": ["Presidente"], "descriptor": "d", "gender": "male", "age": 65},
        {"name": "MOTORCYCLIST", "aliases": ["motociclista", "terrorista"], "descriptor": "d",
         "gender": "male", "age": 30}]})
    cenas = [{"action_text": "O Presidente entra. O motociclista foge. O terrorista cai.",
              "shot_list": [{"visual": "The President approaches the car"},
                            {"visual": "The motorcycle rider flees"},
                            {"visual": "Ha-eun tackles the terrorist"},
                            {"visual": "The President waves"}]}]
    extras = cc.detect_extras(cenas, [], engine="x")
    assert {"Presidente", "president"} <= {a if a == "Presidente" else a.lower() for a in extras["PRESIDENT"]["aliases"]}
    moto = {a.lower() for a in extras["MOTORCYCLIST"]["aliases"]}
    assert {"motociclista", "terrorista", "motorcycle rider", "terrorist"} <= moto


def test_familias_de_papel_nao_juntam_papeis_diferentes():
    # juntar terrorista e motociclista e decisao de ENREDO (LLM), nunca da tabela
    assert "motorcycle rider" not in cc._FAMILIAS_DE_PAPEL.get("terrorist", [])
    assert "terrorist" not in cc._FAMILIAS_DE_PAPEL.get("motociclista", [])
    assert "president" in cc._FAMILIAS_DE_PAPEL["presidente"]


def test_personagens_citados_em_ordem_com_figurantes():
    cast = {"HA-EUN": {}, "TERRORIST": {"aliases": ["motorcycle rider"]},
            "VOZ": {"on_screen": False}}
    assert cc.personagens_citados("The motorcycle rider flees as Ha-eun watches; VOZ fala",
                                  cast) == ["TERRORIST", "HA-EUN"]


def test_alias_com_fronteira_de_palavra():
    sp.ALIASES = {"PRESIDENT": ["President"], "TERRORIST": ["motorcycle rider"]}
    try:
        assert sp._character_in("Sirens clear the way for the presidential convoy",
                                ["PRESIDENT"]) == ""
        assert sp._character_in("The President approaches the car", ["PRESIDENT"]) == "PRESIDENT"
        assert sp._other_character_in("Ha-eun spots the motorcycle rider fleeing",
                                      ["HA-EUN", "TERRORIST"], "HA-EUN") == "TERRORIST"
        assert sp._character_in("Ha eun runs", ["HA-EUN"]) == "HA-EUN"
    finally:
        sp.ALIASES = {}


def test_enquadramento_sem_sujeito_pelo_conteudo():
    assert sp._enquadre_sem_sujeito("The sedan explodes, a fireball engulfs the entrance") == "wide"
    assert sp._enquadre_sem_sujeito("Glass shatters, the crowd panics") == "wide"
    assert sp._enquadre_sem_sujeito("A new reflection appears in the glass") == "insert"


def test_direcao_de_arte_so_visual():
    arte = ("Ação/suspense contemporâneo, luz dura de dia claro, câmera nervosa e ágil, cortes "
            "rápidos em picos de tensão. Trilha percussiva tensa. Sirenes, vidro quebrando, tiros, "
            "gritos da multidão. Diálogos em")
    assert sp._arte_visual(arte) == "Ação/suspense contemporâneo, luz dura de dia claro"


def test_close_de_fala_e_reacao_sem_sorriso_em_acao():
    v = sp._video_prompt(action="Ha-eun spots the rider removing his helmet", movement="static",
                         descriptor="", look="Ação/suspense contemporâneo", quote=None,
                         subject="HA-EUN", emotion="alegre", framing="close", speaking=True)
    assert "helmet" not in v
    assert "reacting to the action" in v
    assert "no smile" in v and "smiling" not in v


def test_verbos_de_acao_nao_caem_em_idle():
    casos = {
        "Min-jun kicks the weapon away and handcuffs the terrorist": "restrain",
        "Ji-ho leads the President to safety through the smoke": "escort",
        "Ha-eun takes down the terrorist on a pedestrian crossing": "takedown",
        "Ha-eun and Min-jun leap over tables in pursuit": "pursue",
        "Ji-ho examines the buildings": "turn",
    }
    for beat, esperado in casos.items():
        prim, _ = mc.infer_primitive(beat, "X", "Y", framing="medium", speaking=False)
        assert prim == esperado, (beat, prim)
