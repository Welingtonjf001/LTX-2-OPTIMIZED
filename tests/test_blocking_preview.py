import json

from script_pipeline.blocking_preview import _checks, build_contact_sheet


def _plan(commands, formation=None):
    formation = formation or {"A": {"anchor_m": [-1.2, 0.0], "screen_side": "left", "facing": "toward_center"},
                              "B": {"anchor_m": [1.2, 0.0], "screen_side": "right", "facing": "toward_center"}}
    return {"scenes": [{"scene": 1, "formation": formation, "commands": commands}]}


def test_flags_overlapping_anchors():
    formation = {"A": {"anchor_m": [-1.2, 0.0], "screen_side": "left", "facing": "x"},
                "B": {"anchor_m": [-1.2, 0.0], "screen_side": "left", "facing": "x"}}
    avisos = _checks(_plan([], formation))
    assert any("mesma" in a and "ancora" in a for a in avisos)


def test_flags_partnered_primitive_without_partner():
    avisos = _checks(_plan([{"shot_index": 3, "actor": "A", "partner": None, "primitive": "pursue"}]))
    assert any("plano 3" in a and "exige parceiro" in a for a in avisos)


def test_flags_pursue_same_screen_side():
    formation = {"A": {"anchor_m": [-1.2, 0.0], "screen_side": "left", "facing": "x"},
                "B": {"anchor_m": [-1.2, 1.35], "screen_side": "left", "facing": "x"}}
    avisos = _checks(_plan([{"shot_index": 1, "actor": "A", "partner": "B", "primitive": "pursue"}], formation))
    assert any("MESMO lado de tela" in a for a in avisos)


def test_no_avisos_for_clean_plan():
    avisos = _checks(_plan([{"shot_index": 0, "actor": "A", "partner": "B", "primitive": "pursue"}]))
    assert avisos == []  # A e B estao em lados opostos (formation padrao)


def test_build_contact_sheet_writes_png_and_report(tmp_path):
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    plan = _plan([{"shot_index": 0, "duration_s": 1.5, "actor": "A", "partner": "B",
                  "primitive": "pursue", "instruction": "chase"}])
    (run / "parse" / "motion_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    out, avisos = build_contact_sheet(run)
    assert out.exists() and out.name == "blocking_preview.png"
    assert (run / "shots" / "blocking_preview_avisos.json").exists()
    assert avisos == []
