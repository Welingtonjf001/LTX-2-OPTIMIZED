# -*- coding: utf-8 -*-
"""Itens da auditoria de scripts de 2026-09-13 que dao para testar sem GPU."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline import shot_plan as sp  # noqa: E402


def test_fala_alegre_nao_pede_boca_aberta():
    # VISTO: "alegre" -> "open smile" fez o LTX gargalhar a fala (shot005, sync -0,29).
    fala = sp.emocao_visivel_fala("alegre")
    assert "open smile" not in fala and "not laughing" in fala
    assert "open smile" in sp.emocao_visivel("alegre")  # plano sem fala continua igual


def test_video_prompt_de_fala_usa_tabela_sem_boca():
    p = sp._video_prompt(action="SEO-YEON shakes her head, amused", movement="static",
                         descriptor="", look="", quote=None, emotion="alegre",
                         framing="close", speaking=True)
    assert "open smile" not in p and "shakes her head" not in p
    assert "not laughing" in p


def test_close_tira_gesto_de_corpo_e_medium_mantem():
    acao = "SEO-YEON smirks, arms crossed"
    close = sp._video_prompt(action=acao, movement="static", descriptor="", look="",
                             quote=None, framing="close", speaking=False)
    medium = sp._video_prompt(action=acao, movement="static", descriptor="", look="",
                              quote=None, framing="medium", speaking=False)
    assert "arms crossed" not in close and "smirks" in close
    assert "arms crossed" in medium


def test_fala_em_plano_medio_tira_risada():
    p = sp._video_prompt(action="MIN-JAE laughs, looking at the map", movement="static",
                         descriptor="", look="", quote=None, framing="medium", speaking=True)
    assert "laugh" not in p.lower() and "looking at the map" in p


def test_tts_reaproveita_fala_com_chave_igual(tmp_path, monkeypatch):
    from script_pipeline import synthesize_dialogue as sd
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    (run / "characters").mkdir()
    (run / "parse" / "scenes.json").write_text(json.dumps([{"index": 1, "dialogue": [
        {"character": "ANA", "text": "Oi.", "emotion": "calma"},
        {"character": "ANA", "text": "Tchau.", "emotion": "alegre"}]}]), encoding="utf-8")
    (run / "characters" / "cast.json").write_text(json.dumps({"ANA": {"voice": {}}}), encoding="utf-8")
    chamadas = []

    def falso(jobs, engine, log):
        chamadas.append([j["id"] for j in jobs])
        out = []
        for j in jobs:
            Path(j["output_path"]).write_bytes(b"x")
            out.append({"id": j["id"], "ok": True, "output_path": j["output_path"], "engine_used": "falso"})
        return out

    import script_pipeline.dialogue_tts as dt
    monkeypatch.setattr(dt, "synthesize_batch", falso)
    monkeypatch.setattr(sd, "_wav_duration_sec", lambda p: 1.0)
    assert sd.main(["--run-dir", str(run)]) == 0
    assert chamadas[-1] == ["scene01_line00", "scene01_line01"]
    assert sd.main(["--run-dir", str(run)]) == 0
    assert chamadas[-1] == ["scene01_line00", "scene01_line01"] and len(chamadas) == 1  # nada refeito
    cenas = json.loads((run / "parse" / "scenes.json").read_text(encoding="utf-8"))
    cenas[0]["dialogue"][1]["emotion"] = "tristeza"
    (run / "parse" / "scenes.json").write_text(json.dumps(cenas), encoding="utf-8")
    assert sd.main(["--run-dir", str(run)]) == 0
    assert chamadas[-1] == ["scene01_line01"]  # so a fala com emocao nova


def test_lora_ab_parse_config():
    from script_pipeline.lora_ab import _parse_config
    assert _parse_config("msr:ic=msr") == {"rotulo": "msr", "loras": [], "ic": "msr"}
    c = _parse_config("combo:better-human-motion:0.6+cameraman-v2")
    assert c["loras"] == [("better-human-motion", 0.6), ("cameraman-v2", None)]
