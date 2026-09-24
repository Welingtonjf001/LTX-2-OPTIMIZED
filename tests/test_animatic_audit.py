from script_pipeline.animatic_audit import expected_cues


def test_expected_cues_center_dialogue_in_reserved_margin():
    plan = {"shots": [
        {"index": 0, "scene": 1, "line_index": None, "seconds": 2.0},
        {"index": 1, "scene": 1, "line_index": 0, "seconds": 5.6},
    ]}
    dialogue = [{"scene_index": 1, "line_index": 0, "ok": True,
                 "audio_path": "voice.wav", "duration_sec": 5.0}]
    cue = expected_cues(plan, dialogue)[0]
    assert abs(cue["expected_start"] - 2.3) < 1e-9
    assert abs(cue["margin_before"] - 0.3) < 1e-9
    assert abs(cue["margin_after"] - 0.3) < 1e-9
