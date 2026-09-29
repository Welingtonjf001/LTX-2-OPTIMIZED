"""ACHADO 2026-09-29 (REENTRY WINDOW, pedido do usuario para baratear o gate visual): a
chamada de DECISAO do gate (`_evaluation_prompt` em `visual_continuity_audit.audit()`) nunca
recebe imagem -- so o JSON da percepcao + o contrato do plano, texto puro (`_ollama_json(model,
..., [], ...)`, terceiro argumento e a lista de imagens, vazia). Rodar essa chamada no mesmo VLM
de 30B da percepcao pagava o preco de um modelo de visao para uma tarefa so de texto. Separado:
percepcao continua em qwen3-vl:30b (unica que precisa ver a imagem), decisao passa para um modelo
de texto rapido ja validado nesta maquina (qwen2.5:32b-instruct-q4_K_M)."""
import script_pipeline.run_decupagem as run_decupagem
import script_pipeline.visual_continuity_audit as visual_continuity_audit


def test_modelo_de_decisao_e_diferente_do_de_percepcao():
    assert visual_continuity_audit.DEFAULT_MODEL != visual_continuity_audit.DEFAULT_PERCEPTION_MODEL


def test_modelo_de_percepcao_continua_o_vlm_de_visao():
    assert visual_continuity_audit.DEFAULT_PERCEPTION_MODEL == "qwen3-vl:30b"


def test_modelo_de_decisao_e_um_modelo_de_texto_ja_validado():
    assert visual_continuity_audit.DEFAULT_MODEL == "qwen2.5:32b-instruct-q4_K_M"


def test_run_decupagem_expoe_os_mesmos_defaults_no_cli():
    parser = run_decupagem.build_parser() if hasattr(run_decupagem, "build_parser") else None
    if parser is None:
        import argparse
        import inspect
        # Sem um builder exportado, confere direto nas strings de default do modulo fonte.
        src = inspect.getsource(run_decupagem)
        assert '"--visual-audit-model", default="qwen2.5:32b-instruct-q4_K_M"' in src
        assert '"--visual-perception-model", default="qwen3-vl:30b"' in src
        return
    args = parser.parse_args(["--run-dir", "x"])
    assert args.visual_audit_model == "qwen2.5:32b-instruct-q4_K_M"
    assert args.visual_perception_model == "qwen3-vl:30b"


def test_chamada_de_decisao_nunca_recebe_imagens(monkeypatch, tmp_path):
    """audit() so passa imagens na chamada de PERCEPCAO -- a de decisao recebe lista vazia.
    Trava esse contrato: se algum dia alguem "otimizar" e passar as imagens de novo pra
    decisao, o modelo de texto rapido nao teria como processa-las."""
    import json as _json
    calls = []

    def fake_ollama_json(model, prompt, images, *, timeout=180):
        calls.append({"model": model, "images": list(images)})
        if model == visual_continuity_audit.DEFAULT_PERCEPTION_MODEL:
            return {"image_received": True, "target_images_seen": 1, "reference_images_seen": 0,
                    "targets": [{"location_description": "x", "framing": "wide",
                                 "primary_people_description": [], "visible_objects": [],
                                 "text_detected": False, "visible_text": [], "text_type": "none",
                                 "aircraft_visible": False, "aircraft_state": "n/a",
                                 "ground_or_runway_visible": False, "wheels_visible": False,
                                 "wheels_touching_surface": False}],
                    "identity_matches": [], "same_location_across_targets": True,
                    "same_primary_people_across_targets": True, "same_location_as_reference": True,
                    "primary_person_visible_in_every_target": False,
                    "reference_subject_prominent_in_every_target": False}
        return {"image_received": True, "target_images_seen": 1, "visual_pass": True,
                "location_match": True, "framing_match": True, "subjects_match": True,
                "speaker_remains_visible": True, "persistent_objects_match": True,
                "aircraft_state_match": True, "forbidden_text_detected": False,
                "identity_match": True, "confidence": 0.9, "reasons": []}

    monkeypatch.setattr(visual_continuity_audit, "_ollama_json", fake_ollama_json)

    run = tmp_path
    (run / "parse").mkdir()
    (run / "shots" / "stills").mkdir(parents=True)
    plan = {"shots": [{"index": 0, "framing": "wide", "line_index": None}]}
    (run / "parse" / "shot_plan.json").write_text(_json.dumps(plan), encoding="utf-8")
    (run / "shots" / "stills" / "stills.json").write_text(
        _json.dumps({"0": {"file": "shot000_wide.png"}}), encoding="utf-8")
    (run / "shots" / "stills" / "shot000_wide.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    visual_continuity_audit.audit(run, stage="stills")

    assert len(calls) == 2
    perception_call = next(c for c in calls if c["model"] == visual_continuity_audit.DEFAULT_PERCEPTION_MODEL)
    decision_call = next(c for c in calls if c["model"] == visual_continuity_audit.DEFAULT_MODEL)
    assert perception_call["images"], "a chamada de percepcao precisa receber a imagem"
    assert decision_call["images"] == [], "a chamada de decisao NAO deve receber imagem"
