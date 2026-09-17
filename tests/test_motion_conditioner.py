from script_pipeline.motion_conditioner import apply_motion_score, build_motion_score, infer_primitive


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
