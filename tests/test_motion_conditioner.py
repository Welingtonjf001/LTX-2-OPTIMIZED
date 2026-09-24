import pytest

from script_pipeline.motion_conditioner import (
    MIN_SEPARATION_M,
    _meeting_point,
    apply_motion_score,
    build_motion_score,
    infer_primitive,
)


def test_motion_score_preserves_actor_and_blocking():
    plan = {"fps": 24, "style": "classico", "screen_sides": {"ANA": "left", "BIA": "right"},
            "shots": [{"scene": 1, "subject": "ANA", "co_subject": "BIA", "seconds": 3,
                       "beat": "ANA walks toward BIA and shakes her hand", "video_prompt": "Ana acts."}]}
    score = build_motion_score(plan)
    command = score["commands"][0]
    assert command["actor"] == "ANA"
    assert command["partner"] == "BIA"
    assert command["primitive"] == "contact"
    assert score["scenes"][0]["formation"]["ANA"]["anchor_m"][0] < 0
    assert score["scenes"][0]["formation"]["BIA"]["anchor_m"][0] > 0


def test_apply_adds_motion_without_removing_decupagem_prompt():
    plan = {"shots": [{"scene": 1, "subject": "ANA", "seconds": 2,
                        "beat": "ANA nods", "video_prompt": "Ana listens."}]}
    applied = apply_motion_score(plan, build_motion_score(plan))
    assert applied["shots"][0]["video_prompt"].startswith("Ana listens.")
    assert "motion constraint" in applied["shots"][0]["video_prompt"].lower()


def test_apply_is_idempotent():
    plan = {"shots": [{"scene": 1, "subject": "ANA", "seconds": 2,
                        "beat": "ANA nods", "video_prompt": "Ana listens."}]}
    once = apply_motion_score(plan, build_motion_score(plan))
    twice = apply_motion_score(once, build_motion_score(once))
    assert twice["shots"][0]["video_prompt"] == once["shots"][0]["video_prompt"]
    assert twice["shots"][0]["video_prompt"].lower().count("motion constraint") == 1


# Casos medidos no plano real dos piratas (2026-09-17) que a v1 errava.
def test_word_boundaries_and_lemmas_from_real_plan():
    assert infer_primitive("Seo-yeon gestures towards the burning ship", "SEO-YEON", "")[0] == "gesture"
    assert infer_primitive("Ji-ho holds a spyglass up to his eye", "JI-HO", "")[0] == "reach"
    # Olhar (turn) vence postura (gesture) pela ordem das regras: o que a
    # tomada pede e a direcao do olhar, a postura e o estado de partida.
    assert infer_primitive("Min-jun leans against the railing, looking out at the horizon",
                           "MIN-JUN", "")[0] == "turn"
    # "touching" descreve os cascos dos navios, nao o ator: nao pode virar reach.
    beat = ("Seo-yeon and Min-jun stand facing each other across the narrow gap between the "
            "two ships' decks. The two ships sail side by side, hulls almost touching.")
    assert infer_primitive(beat, "SEO-YEON", "MIN-JUN")[0] == "turn"
    assert infer_primitive("she returns to the table", "ANA", "")[0] == "idle"


def test_shot_without_subject_is_environment_even_if_fallback_mentions_walking():
    plan = {"shots": [{"scene": 1, "subject": "", "seconds": 2, "framing": "wide",
                        "beat": "", "fallback": "Ana walks across the deck while Bia hurries in.",
                        "video_prompt": "Wide deck."}]}
    command = build_motion_score(plan)["commands"][0]
    assert command["primitive"] == "environment"
    assert "The subject" not in command["motion_prompt"]


def test_speaking_close_up_never_asks_for_body_gesture():
    plan = {"shots": [{"scene": 1, "subject": "ANA", "seconds": 4, "framing": "close",
                        "line_index": 0, "beat": "Ana gestures wildly and walks to the door",
                        "video_prompt": "Close on Ana."}]}
    command = build_motion_score(plan)["commands"][0]
    assert command["primitive"] == "speak"
    assert "no hand gestures" in command["motion_prompt"]
    # Em plano medio a mesma fala pode gesticular: a restricao e do close.
    plan["shots"][0]["framing"] = "medium"
    assert build_motion_score(plan)["commands"][0]["primitive"] == "locomote"


def test_locomote_without_partner_has_no_self_destination():
    plan = {"shots": [{"scene": 1, "subject": "ANA", "seconds": 2,
                        "beat": "ANA walks to the window", "video_prompt": "x"}]}
    target = build_motion_score(plan)["commands"][0]["target"]
    assert target["mode"] == "short_path"
    assert target["destination"] is None


@pytest.mark.parametrize(("beat", "expected", "phrase"), [
    ("HA-EUN and MIN-JUN chase the terrorist through the crowd", "pursue", "sprint after"),
    ("HA-EUN tackles the terrorist to the ground", "takedown", "controlled tackle"),
    ("SEO-YEON pulls the president away from the falling sign", "reach", "reach toward"),
    ("JI-HO shields the president from the attack", "protect", "protective position"),
    ("The rooftop gunman fires repeatedly toward the car", "fire", "fire controlled shots"),
])
def test_action_beats_never_degrade_to_idle(beat, expected, phrase):
    primitive, instruction = infer_primitive(beat, "ACTOR", "PARTNER")
    assert primitive == expected
    assert phrase in instruction


def test_pursuit_pair_keeps_shared_heading_instead_of_converging():
    plan = {"screen_sides": {"HA-EUN": "left", "MIN-JUN": "right"}, "shots": [
        {"scene": 1, "subject": "HA-EUN", "co_subject": "MIN-JUN", "seconds": 3,
         "beat": "HA-EUN and MIN-JUN chase the terrorist through the crowd"}
    ]}
    command = build_motion_score(plan)["commands"][0]
    assert command["primitive"] == "pursue"
    assert command["target"]["mode"] == "short_path"
    assert command["target"]["destination"] is None


# Achado 2026-09-17: dois agentes reais (ANA e BIA) andando um em direcao ao
# outro cruzaram e trocaram de lado de tela inteiro, terminando MAIS LONGE
# (2,75 m) do que a distancia inicial (2,4 m) -- porque o destino de cada um
# era a ancora ORIGINAL do outro, nao um ponto de encontro.
def test_meeting_point_is_between_anchors_and_respects_separation():
    ponto = _meeting_point([-1.2, 0.0], [1.2, 0.0], MIN_SEPARATION_M)
    assert ponto[0] < 0  # fica do lado de QUEM vai andar (own_anchor), nao cruza pro outro lado
    assert abs(ponto[0] - (-MIN_SEPARATION_M / 2)) < 1e-9


def test_two_actors_approaching_each_other_converge_without_crossing():
    plan = {"fps": 24, "screen_sides": {"ANA": "left", "BIA": "right"},
            "shots": [{"scene": 1, "subject": "ANA", "co_subject": "BIA", "seconds": 3,
                       "beat": "ANA walks toward BIA", "video_prompt": "x"},
                      {"scene": 1, "subject": "BIA", "co_subject": "ANA", "seconds": 3,
                       "beat": "BIA walks toward ANA", "video_prompt": "x"}]}
    score = build_motion_score(plan)
    ana_dest = score["commands"][0]["target"]["destination"]
    bia_dest = score["commands"][1]["target"]["destination"]
    # ANA (ancora negativa) para do lado NEGATIVO do ponto de encontro; BIA do
    # lado POSITIVO -- nunca cruzam pro lado de tela da outra.
    assert ana_dest[0] < 0
    assert bia_dest[0] > 0
    separacao_final = abs(ana_dest[0] - bia_dest[0])
    assert abs(separacao_final - MIN_SEPARATION_M) < 1e-6
