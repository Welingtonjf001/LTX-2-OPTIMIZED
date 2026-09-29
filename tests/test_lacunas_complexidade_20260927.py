"""Auditoria geral 2026-09-27 (MEMORIAL 3.134): lacunas do roteiro, complexidade por
plano, emocao unica para todo caminho que escreve prompt, e as reconstrucoes de
prompt que perdiam o parceiro. Sem GPU e sem LLM."""
from script_pipeline import screenplay_gaps as sg
from script_pipeline import shot_complexity as sc
from script_pipeline import shot_plan as sp

ELENCO = {
    "SEO-YEON": {"descriptor": "Dark-brown hair, black pinstripe suit", "line_count": 2,
                 "voice": {"gender": "female", "gender_guessed": False}},
    "JI-HO": {"descriptor": "Short black hair, dark-gray suit, vest", "line_count": 2,
              "voice": {"gender": "male", "gender_guessed": False}},
}
ELENCO_COM_EXTRA = {**ELENCO, "PRESIDENT": {
    "descriptor": "silver-gray hair, navy suit, red tie", "extra": True, "aliases": ["President"],
    "line_count": 0}}


def _cena(itens, dialogo=()):
    return [{"index": 1, "location": "Seoul avenue", "time_of_day": "day", "action_text": "",
             "characters": ["SEO-YEON", "JI-HO"], "shot_list": list(itens), "dialogue": list(dialogo)}]


def test_pessoa_agindo_sem_cadastro_e_critica():
    cenas = _cena([{"type": "action", "visual": "The President approaches the armored sedan",
                    "actor": None}])
    lacunas, _ = sg.audit_deterministic(cenas, ELENCO)
    assert any(l["tipo"] == "pessoa_sem_cadastro" and l["gravidade"] == "critica" for l in lacunas)
    lacunas, _ = sg.audit_deterministic(cenas, ELENCO_COM_EXTRA)
    assert not any(l["tipo"] == "pessoa_sem_cadastro" for l in lacunas)


def test_acao_de_contato_sem_alvo_e_tiro_sem_direcao():
    cenas = _cena([{"type": "action", "visual": "Ji-ho shields him with his body", "actor": "JI-HO"},
                   {"type": "action", "visual": "Seo-yeon fires her gun to cover the retreat",
                    "actor": "SEO-YEON"}])
    lacunas, _ = sg.audit_deterministic(cenas, ELENCO)
    tipos = [(l["tipo"], l["item"]) for l in lacunas]
    assert ("alvo", 0) in tipos
    assert ("alvo", 1) in tipos


def test_fala_que_repete_a_acao_vizinha_e_fala_sobre_acao():
    cenas = _cena(
        [{"type": "action", "visual": "Ji-ho leads the President to safety through the smoke",
          "actor": "JI-HO"}, {"type": "dialogue", "line_index": 0}],
        [{"character": "JI-HO", "text": "Por aqui!",
          "beat_visual": "Ji-ho leads the President to safety through the smoke."}])
    lacunas, _ = sg.audit_deterministic(cenas, ELENCO_COM_EXTRA)
    tipos = {l["tipo"] for l in lacunas if l["item"] == 1}
    assert {"repeticao", "fala_sobre_acao"} <= tipos


def test_mudanca_de_estado_vira_continuidade_a_manter():
    cenas = _cena([{"type": "action", "visual": "The sedan explodes, a fireball engulfs the entrance",
                    "actor": None}])
    _, estados = sg.audit_deterministic(cenas, ELENCO)
    assert estados and estados[0]["item"] == 0


def test_voz_adivinhada_de_quem_fala_e_lacuna():
    elenco = {"HA-EUN": {"descriptor": "ponytail, tactical suit", "line_count": 2,
                         "voice": {"gender": "male", "gender_guessed": True}}}
    lacunas, _ = sg.audit_deterministic(_cena([]), elenco)
    assert any(l["tipo"] == "voz" for l in lacunas)


def _plano(**kw):
    base = {"beat": "", "framing": "medium", "seconds": 3.0, "movement": "static",
            "subject": "", "co_subject": "", "line_index": None,
            "motion_conditioning": {"primitive": "idle"}}
    base.update(kw)
    return base


def test_complexidade_contato_curto_e_complexo():
    p = _plano(beat="Min-jun kicks the weapon away and handcuffs the terrorist", subject="MIN-JUN",
               co_subject="TERRORIST", seconds=1.6, motion_conditioning={"primitive": "restrain"})
    a = sc.score_shot(p, {})
    assert a["nivel"] == "complexa" and a["contato"]
    assert any("PREVIS" in r for r in sc.recomendacao(a, "ltx"))
    assert any("union-control" in r for r in sc.recomendacao(a, "ltx"))
    assert any("MiniMax" in r for r in sc.recomendacao(a, "minimax-longtake"))


def test_close_de_fala_pontua_so_o_rosto_e_sinaliza_acao_fora():
    p = _plano(beat="Ji-ho shields the President with his body as the sedan explodes.",
               subject="JI-HO", framing="close", line_index=1,
               motion_conditioning={"primitive": "speak"})
    a = sc.score_shot(p, ELENCO_COM_EXTRA)
    assert a["nivel"] == "simples"
    assert a["acao_fora_do_quadro"]
    assert any("nao aparece neste close" in r for r in sc.recomendacao(a, "ltx"))


def test_explosao_em_plano_aberto_e_media():
    p = _plano(beat="The sedan explodes, a fireball engulfs the hotel entrance", framing="wide",
               motion_conditioning={"primitive": "environment"})
    assert sc.score_shot(p, {})["nivel"] == "media"


def test_emocao_leve_em_acao_detectada_pelo_texto_sem_direcao_de_arte():
    v = sp.emocao_para_video("alegre", falando=True,
                             contexto="Ha-eun tackles the terrorist onto the taxi hood")
    assert "no smile" in v
    # fora de acao segue a versao de FALA (sorriso contido, sem riso -- MEMORIAL 3.85)
    assert sp.emocao_para_video("alegre", falando=True, contexto="") == sp.EMOCAO_VISIVEL_FALA["alegre"]


def test_camera_llm_nao_derruba_o_parceiro(monkeypatch):
    sp.ALIASES = {}
    cena = {"index": 1, "location": "avenue", "time_of_day": "day", "action_text": "",
            "characters": ["JI-HO", "SEO-YEON"],
            "shot_list": [{"type": "action", "visual": "Ji-ho and Seo-yeon run to the car",
                           "actor": "JI-HO"}], "dialogue": []}
    plano = sp.plan_all([cena], None, descriptors={"JI-HO": "gray suit", "SEO-YEON": "pinstripe suit"})
    shot = plano["shots"][-1]
    assert shot["co_subject"] == "SEO-YEON" and "pinstripe suit" in shot["video_prompt"]
    novo = next(m for m in sp.CAMERA_STYLE_VOCAB[shot["style"]]["movements"] if m != shot["movement"])
    monkeypatch.setattr(sp, "_call_ollama_camera",
                        lambda *a, **k: {"choices": [{"position": shot["position"], "movement": novo}]})
    sp.enrich_camera_style(plano, engine="x", log=lambda m: None)
    assert "pinstripe suit" in shot["video_prompt"]
    assert "pinstripe suit" in shot["storyboard_prompt"]


def test_verify_output_avisa_gates_nao_aprovados(tmp_path):
    import json
    from script_pipeline.verify_output import _check_gates
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "visual_stills_audit.json").write_text(json.dumps(
        {"status": "blocked", "rejected": 3, "auditor_errors": [4], "evaluated": 9, "expected": 10}))
    (tmp_path / "shots" / "clip_identity_audit.json").write_text(json.dumps(
        {"shot001": {"alerta": True}, "shot002": {"alerta": False}}))
    report, findings = {"checks": []}, []
    _check_gates(tmp_path, report, findings)
    assert findings and findings[0]["code"] == "GATES_NOT_APPROVED"
    assert "3 reprovado" in findings[0]["detail"] and "1 clipe" in findings[0]["detail"]
    report, findings = {"checks": []}, []
    (tmp_path / "shots" / "visual_stills_audit.json").write_text(json.dumps({"status": "ok"}))
    (tmp_path / "shots" / "visual_video_audit.json").write_text(json.dumps({"status": "ok"}))
    (tmp_path / "shots" / "clip_identity_audit.json").write_text("{}")
    _check_gates(tmp_path, report, findings)
    assert not findings


def test_verify_output_avisa_gate_ausente_ou_ilegivel(tmp_path):
    """Achado de revisao externa 2026-09-29: gate AUSENTE (arquivo nunca gerado) ou
    ILEGIVEL (JSON corrompido) nao pode virar silencio -- antes esses dois casos eram
    ignorados (`continue`) e o verification.json ficava indistinguivel de "tudo aprovado"."""
    import json
    from script_pipeline.verify_output import _check_gates
    (tmp_path / "shots").mkdir()
    # nenhum arquivo de gate/identidade existe -- os dois devem virar AUSENTE
    report, findings = {"checks": []}, []
    _check_gates(tmp_path, report, findings)
    assert findings and findings[0]["code"] == "GATES_NOT_APPROVED"
    assert "AUSENTE" in findings[0]["detail"]
    status_por_check = {c["check"]: c["status"] for c in report["checks"]}
    assert status_por_check["gate_stills"] == "missing"
    assert status_por_check["gate_video"] == "missing"
    assert status_por_check["identity_audit"] == "missing"

    # JSON corrompido -- deve virar ILEGIVEL, nao "sem alerta"
    (tmp_path / "shots" / "clip_identity_audit.json").write_text("{nao e json valido")
    report, findings = {"checks": []}, []
    _check_gates(tmp_path, report, findings)
    status_por_check = {c["check"]: c["status"] for c in report["checks"]}
    assert status_por_check["identity_audit"] == "unreadable"
    assert "ILEGIVEL" in findings[0]["detail"]


def test_render_scenes_acao_leva_aparencia_e_fala_sobre_acao_vira_reacao(tmp_path):
    from script_pipeline.render_scenes import build_render_plan
    cenas = [{"index": 1, "location": "avenue", "time_of_day": "day", "action_text": "",
              "art_direction": "action thriller", "characters": ["JI-HO"],
              "shot_list": [{"type": "action", "visual": "The President approaches the car",
                             "actor": None}, {"type": "dialogue", "line_index": 0}],
              "dialogue": [{"character": "JI-HO", "text": "Abaixem!", "emotion": "alegre",
                            "beat_visual": "Ji-ho tackles the gunman to the ground"}]}]
    jobs = build_render_plan(cenas, ELENCO_COM_EXTRA, {}, tmp_path)
    acao, fala = jobs[0]["prompt"], jobs[1]["prompt"]
    assert "navy suit" in acao
    assert "tackles" not in fala and "reacts urgently" in fala
    assert "no smile" in fala
