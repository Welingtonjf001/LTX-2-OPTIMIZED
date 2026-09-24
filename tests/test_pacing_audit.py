from script_pipeline.pacing_audit import audit, summary


def test_flags_too_short_action_wide_and_too_long_static_close():
    plan = {"shots": [
        {"position": 0, "framing": "wide", "seconds": 1.5,
         "video_prompt": "the agent runs and dives across the avenue"},
        {"position": 1, "framing": "close", "seconds": 7.0, "video_prompt": "a quiet stare"},
        {"position": 2, "framing": "insert", "seconds": 1.5, "video_prompt": "a red light blinking"},
    ]}
    report = audit(plan)
    assert report["status"] == "warnings"
    types = {(f["shot"], f["type"]) for f in report["findings"]}
    assert (0, "too_short") in types   # acao pede >= 2x o minimo do wide (1.5 -> 3.0)
    assert (1, "too_long") in types    # close estatico acima de 5s
    assert 2 not in {f["shot"] for f in report["findings"]}   # dentro da faixa do insert


def test_dialogue_shots_are_skipped_speech_split_owns_them():
    plan = {"shots": [{"position": 0, "framing": "close", "seconds": 30.0, "line_index": 2}]}
    assert audit(plan)["status"] == "ok"


def test_summary_lists_each_finding():
    plan = {"shots": [{"position": 5, "framing": "insert", "seconds": 6.0, "video_prompt": ""}]}
    report = audit(plan)
    text = summary(report)
    assert "plano 5" in text and "1 aviso" in text
