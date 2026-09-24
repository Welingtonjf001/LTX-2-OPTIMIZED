"""Regressoes da auditoria do gate visual (2026-09-19) e das pecas novas:
divisao de fala longa, laco de regeneracao, referencia de locacao, manifesto
parcial e master de audio. Sem GPU e sem Ollama."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from script_pipeline import visual_continuity_audit as vca
from script_pipeline.gate_retry import format_shot_spec, gate_with_retries
from script_pipeline.location_master import dependents, expand_blocked, invalidate_bad_refs
from script_pipeline.render_shots_stage import merge_manifest
from script_pipeline.speech_split import choose_boundaries, dialogue_entry, slice_wav, split_long_speech

FFMPEG = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
needs_ffmpeg = pytest.mark.skipif(not Path(FFMPEG).exists(), reason="ffmpeg ausente")


def _target(loc="airplane cabin", people="a woman, face clearly visible", **kw):
    base = dict(location_description=loc, framing="close-up", primary_people_description=people,
                visible_objects=[], text_detected=False, visible_text=[], text_type="none",
                aircraft_visible=False, aircraft_state="absent", ground_or_runway_visible=False,
                wheels_visible=False, wheels_touching_surface=False)
    base.update(kw)
    return base


def _per(targets, **kw):
    p = dict(image_received=True, target_images_seen=len(targets), targets=targets,
             identity_matches=[True], reference_subject_prominent_in_every_target=True)
    p.update(kw)
    return p


def _lock(perception, **kw):
    args = dict(aircraft_required=False, aircraft_must_be_visible=False, identity_required=False)
    args.update(kw)
    return vca._apply_perception_locks({}, perception, **args)


# ---- bugs do gate ----------------------------------------------------------

def test_location_drift_in_one_frame_blocks():
    per = _per([_target(), _target(), _target("Airport tarmac")])
    assert _lock(per, location_id="LOC_CABIN")["location_match"] is False


def test_location_all_frames_ok():
    per = _per([_target()] * 3)
    assert _lock(per, location_id="LOC_CABIN")["location_match"] is True


def test_contract_supplied_location_keywords_generalize():
    per = _per([_target("a school hallway with lockers")] * 3)
    ok = _lock(per, location_id="LOC_CORREDOR", location_keywords=["hallway"])
    bad = _lock(_per([_target("a school hallway"), _target("a beach")]),
                location_id="LOC_CORREDOR", location_keywords=["hallway"])
    assert ok["location_match"] is True and bad["location_match"] is False


def test_declared_drift_fields_are_enforced():
    per = _per([_target()] * 3, same_location_across_targets=False, same_primary_people_across_targets=False)
    raw = _lock(per, location_id="LOC_OUTRO")
    assert raw["location_match"] is False and raw["subjects_match"] is False


def test_cropped_hair_does_not_hide_the_speaker():
    per = _per([_target(people="a woman with cropped black hair, face clearly visible")] * 3)
    raw = _lock(per, identity_required=True, framing="close", speaker_required=True, expected_refs=1)
    assert raw["speaker_remains_visible"] is True


@pytest.mark.parametrize("phrase", ["partially visible at the edge", "standing in the background",
                                    "cropped at the frame edge", "her back to the camera"])
def test_real_occlusion_still_blocks(phrase):
    per = _per([_target(people=f"a woman {phrase}")] * 3)
    raw = _lock(per, identity_required=True, framing="close", speaker_required=True, expected_refs=1)
    assert raw["speaker_remains_visible"] is False


def test_identity_needs_one_boolean_per_reference():
    per = _per([_target()] * 3, identity_matches=[True])
    assert _lock(per, identity_required=True, expected_refs=2, framing="close")["identity_match"] is False
    per2 = _per([_target()] * 3, identity_matches=[True, True])
    assert _lock(per2, identity_required=True, expected_refs=2, framing="close")["identity_match"] is True


def test_numeric_face_decides_close_identity_when_available():
    per = _per([_target()] * 3)
    low = _lock(per, identity_required=True, expected_refs=1, framing="close",
                face={"frames": 3, "frames_with_face": 3, "min_similarity": 0.10})
    assert low["identity_match"] is False
    high = _lock(per, identity_required=True, expected_refs=1, framing="close",
                 face={"frames": 3, "frames_with_face": 3, "min_similarity": 0.80})
    assert high["identity_match"] is True
    # o booleano do VLM reprovava closes corretos (Ji-ho 0,57): com o rosto medido, o numero decide
    vlm_no = _lock(_per([_target()] * 3, identity_matches=[False]), identity_required=True, expected_refs=1,
                   framing="close", face={"frames": 3, "frames_with_face": 3, "min_similarity": 0.57})
    assert vlm_no["identity_match"] is True
    # sem medida numerica (insightface ausente), o VLM decide
    sem = _lock(_per([_target()] * 3, identity_matches=[False]), identity_required=True, expected_refs=1,
                framing="close", face=None)
    assert sem["identity_match"] is False


def test_speaker_needs_a_face_in_every_frame():
    per = _per([_target()] * 3)
    raw = _lock(per, identity_required=True, framing="close", speaker_required=True, expected_refs=1,
                face={"frames": 3, "frames_with_face": 2, "min_similarity": 0.7})
    assert raw["speaker_remains_visible"] is False
    sem_foto = _lock(per, speaker_required=True, face={"frames": 3, "frames_with_face": 1, "min_similarity": None},
                     framing="close")
    assert sem_foto["speaker_remains_visible"] is False


# ---- relatorio mesclado ----------------------------------------------------

def test_status_partial_when_plan_not_fully_audited():
    res = [{"shot": 0, "pass": True}]
    assert vca.build_status(res, [0, 1]) == "partial"
    assert vca.build_status(res + [{"shot": 1, "pass": True}], [0, 1]) == "ok"
    assert vca.build_status(res + [{"shot": 1, "pass": False}], [0, 1]) == "blocked"


def test_merge_replaces_by_shot_and_keeps_the_rest():
    old = [{"shot": 0, "pass": False}, {"shot": 1, "pass": True}]
    fresh = [{"shot": 0, "pass": True}]
    merged = vca.merge_results(old, fresh)
    assert [(r["shot"], r["pass"]) for r in merged] == [(0, True), (1, True)]


def _fake_run(tmp_path, n=2, with_index=False):
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    (run / "shots" / "stills").mkdir(parents=True)
    (run / "characters").mkdir()
    shots = []
    for i in range(n):
        s = {"scene": 1, "position": i, "framing": "wide", "location_id": "LOC_X",
             "video_prompt": "x", "subject": None, "co_subject": None}
        if with_index:
            s["index"] = i
        shots.append(s)
    (run / "parse" / "shot_plan.json").write_text(json.dumps({"shots": shots}), encoding="utf-8")
    manifest = {}
    for i in range(n):
        f = run / "shots" / "stills" / f"shot{i:03d}_wide.png"
        f.write_bytes(b"x")
        manifest[str(i)] = {"file": f.name}
    (run / "shots" / "stills" / "stills.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _stub_ollama(monkeypatch, fail_first_perception=False):
    calls = {"n": 0}

    def fake(model, prompt, images, *, timeout=180):
        calls["n"] += 1
        if prompt.startswith("Perform neutral pixel observation"):
            return _per([_target("a place")], same_location_as_reference=True)
        return {"image_received": True, "target_images_seen": 1, "visual_pass": True, "location_match": True,
                "framing_match": True,
                "subjects_match": True, "persistent_objects_match": True, "forbidden_text_detected": False,
                "reasons": []}
    monkeypatch.setattr(vca, "_ollama_json", fake)
    monkeypatch.setattr(vca, "face_check", lambda *a, **k: None)
    return calls


def test_plans_without_index_are_audited_by_position(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, n=2, with_index=False)
    _stub_ollama(monkeypatch)
    report = vca.audit(run, stage="stills")
    assert report["status"] == "ok" and report["audited"] == 2 and report["expected"] == 2


def test_partial_reaudit_does_not_erase_the_full_report(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, n=2)
    _stub_ollama(monkeypatch)
    vca.audit(run, stage="stills")
    (run / "shots" / "stills" / "shot000_wide.png").unlink()  # plano 0 passa a falhar
    report = vca.audit(run, stage="stills", only_shots={0})
    assert report["expected"] == 2 and report["audited"] == 2
    assert report["status"] == "blocked"
    assert vca.blocked_shots(run, "stills") == [0]


def test_isolated_auditor_error_does_not_hide_other_shots(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, n=3)
    _stub_ollama(monkeypatch)
    (run / "shots" / "stills" / "shot000_wide.png").unlink()
    report = vca.audit(run, stage="stills")
    assert report["audited"] == 3  # antes: o primeiro erro interrompia o laco
    assert vca.blocked_shots(run, "stills") == [0]


# ---- laco de regeneracao ---------------------------------------------------

def test_format_shot_spec():
    assert format_shot_spec([7, 0, 2, 3, 4]) == "0,2-4,7"
    assert format_shot_spec([]) == ""


def test_regen_loop_fixes_then_passes(tmp_path):
    state = {"blocked": [3, 4], "calls": []}

    def run_step(nome, cmd, obrigatorio=True):
        state["calls"].append(cmd)
        if "--only-shots" in cmd and "--stage" in cmd:  # reauditoria
            state["blocked"] = []
            return True
        return not state["blocked"] if "--stage" in cmd else True

    ok = gate_with_retries(tmp_path, stage="stills", audit_cmd=["-m", "x", "--stage", "stills"],
                           regen_cmd=lambda spec, seed: ["regen", spec, str(seed)], run_step=run_step,
                           blocked_fn=lambda: list(state["blocked"]), base_seed=1234, max_retries=2,
                           log=lambda m: None)
    assert ok is True
    regen = [c for c in state["calls"] if c and c[0] == "regen"]
    assert regen == [["regen", "3-4", str(1234 + 7919)]]
    info = json.loads((tmp_path / "shots" / "visual_gate_stills_retries.json").read_text(encoding="utf-8"))
    assert info["approved"] is True and info["rounds"][0]["shots"] == [3, 4]


def test_regen_loop_gives_up_after_max_retries(tmp_path):
    def run_step(nome, cmd, obrigatorio=True):
        return False if "--stage" in cmd else True
    ok = gate_with_retries(tmp_path, stage="video", audit_cmd=["--stage", "video"],
                           regen_cmd=lambda spec, seed: ["regen", spec], run_step=run_step,
                           blocked_fn=lambda: [5], base_seed=1, max_retries=2, log=lambda m: None)
    assert ok is False
    info = json.loads((tmp_path / "shots" / "visual_gate_video_retries.json").read_text(encoding="utf-8"))
    assert len(info["rounds"]) == 2 and info["unresolved"] == [5]


def test_regen_loop_skips_when_no_identifiable_shot(tmp_path):
    ok = gate_with_retries(tmp_path, stage="video", audit_cmd=["--stage", "video"],
                           regen_cmd=lambda s, seed: ["regen"], run_step=lambda n, c, obrigatorio=True: False,
                           blocked_fn=lambda: [], base_seed=1, max_retries=3, log=lambda m: None)
    assert ok is False
    info = json.loads((tmp_path / "shots" / "visual_gate_video_retries.json").read_text(encoding="utf-8"))
    assert info["rounds"] == []


# ---- referencia de locacao -------------------------------------------------

def test_bad_reference_is_dropped_and_dependents_are_redone(tmp_path):
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    (run / "shots").mkdir()
    plan = {"shots": [{"index": i, "scene": 4, "location_id": "LOC_SKY" if i >= 27 else "LOC_CABIN"}
                      for i in range(33)]}
    (run / "parse" / "shot_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run / "shots" / "location_refs.json").write_text(json.dumps(
        {"LOC_SKY": "x/shot027_wide.png", "LOC_CABIN": "x/shot006_wide.png"}), encoding="utf-8")
    assert invalidate_bad_refs(run, [27, 40]) == {"LOC_SKY": 27}
    refs = json.loads((run / "shots" / "location_refs.json").read_text(encoding="utf-8"))
    assert "LOC_SKY" not in refs and "LOC_CABIN" in refs
    (run / "shots" / "location_refs.json").write_text(json.dumps({"LOC_SKY": "x/shot027_wide.png"}), encoding="utf-8")
    assert expand_blocked(run, [27]) == [27, 28, 29, 30, 31, 32]
    assert dependents(plan, {"LOC_SKY": 27}) == [28, 29, 30, 31, 32]


# ---- manifesto parcial -----------------------------------------------------

def test_partial_manifest_merges_instead_of_replacing():
    existing = [{"id": f"scene01_shot{i:03d}", "video_path": f"a{i}"} for i in range(4)]
    novos = [{"id": "scene01_shot001", "video_path": "novo"}]
    merged = merge_manifest(existing, novos)
    assert [c["id"][-3:] for c in merged] == ["000", "001", "002", "003"]
    assert merged[1]["video_path"] == "novo"


# ---- fala longa ------------------------------------------------------------

def test_boundaries_balanced_and_capped():
    b = choose_boundaries(14.0, 6.0, [])
    segs = [y - x for x, y in zip(b, b[1:])]
    assert len(segs) == 3 and max(segs) <= 6.0 and min(segs) >= 1.6


def test_boundaries_snap_to_silence_within_budget():
    b = choose_boundaries(12.0, 6.5, [5.6, 6.3])
    assert 6.3 in b or 5.6 in b  # encaixou num silencio proximo do corte ideal (6.0)
    assert max(y - x for x, y in zip(b, b[1:])) <= 6.5


def test_short_speech_untouched_and_long_split_contiguous():
    plan = {"shots": [
        {"scene": 1, "line_index": 0, "seconds": 3.0, "frames": 73, "index": 0},
        {"scene": 1, "line_index": 1, "seconds": 15.0, "frames": 361, "index": 1},
        {"scene": 1, "line_index": None, "seconds": 2.0, "frames": 49, "index": 2},
    ]}
    durs = {(1, 0): 2.4, (1, 1): 14.4}
    out = split_long_speech(plan, durs, {}, max_seconds=6.0, folga=0.6,
                            frames_fn=lambda s, f: round(s * f), fps=24.0, gap_fn=lambda w: [])
    speech = [s for s in out["shots"] if s.get("audio_window")]
    assert len(speech) == 3 and all(s["line_index"] == 1 for s in speech)
    assert speech[0]["audio_window"][0] == 0.0 and speech[-1]["audio_window"][1] == 14.4
    for a, b in zip(speech, speech[1:]):
        assert a["audio_window"][1] == b["audio_window"][0]
    assert [s["index"] for s in out["shots"]] == list(range(len(out["shots"])))
    assert out["total_shots"] == len(out["shots"]) == 5


def test_split_disabled_and_no_real_duration():
    plan = {"shots": [{"scene": 1, "line_index": 1, "seconds": 15.0, "frames": 361}]}
    assert split_long_speech(plan, {(1, 1): 14.4}, {}, max_seconds=0, folga=0.6,
                             frames_fn=lambda s, f: 1, fps=24.0) is plan
    out = split_long_speech(plan, {}, {}, max_seconds=6.0, folga=0.6, frames_fn=lambda s, f: 1, fps=24.0)
    assert len(out["shots"]) == 1  # sem duracao real de TTS: nao divide


@needs_ffmpeg
def test_slice_wav_and_dialogue_entry(tmp_path):
    wav = tmp_path / "fala.wav"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
                    str(wav)], check=True)
    shot = {"scene": 1, "line_index": 0, "audio_window": [1.0, 2.5]}
    path, dur = dialogue_entry(shot, {(1, 0): (str(wav), 4.0)})
    assert dur == 1.5 and Path(path).parent.name == "_slices"
    probe = subprocess.run([str(Path(FFMPEG).with_name("ffprobe.exe")), "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    assert abs(float(probe) - 1.5) < 0.05
    assert dialogue_entry({"scene": 1, "line_index": 0}, {(1, 0): (str(wav), 4.0)})[0] == str(wav)


# ---- master de audio -------------------------------------------------------

@needs_ffmpeg
def test_master_audio_hits_target_and_true_peak(tmp_path):
    from script_pipeline.assemble_final import master_audio, measure_loudness
    src = tmp_path / "loud.mp4"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=6",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=6", "-af", "volume=14dB,alimiter=limit=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    antes = measure_loudness(src)
    assert float(antes["input_i"]) > -14.0  # de fato "alto demais"
    out = tmp_path / "mastered.mp4"
    res = master_audio(src, out, log=lambda m: None)
    assert res is not None and out.exists()
    depois = res["depois"]
    assert abs(float(depois["input_i"]) - (-16.0)) < 1.5
    assert float(depois["input_tp"]) <= -1.0


# ---- segunda rodada (corrida real do Voo 702, 2026-09-19) -------------------

def test_shot_set_accepts_ranges_from_the_retry_loop():
    assert vca._shot_set("0-2,5,9-10") == {0, 1, 2, 5, 9, 10}
    assert vca._shot_set(None) is None


def test_diegetic_text_passes_but_captions_and_unknown_text_block():
    exit_sign = _per([_target(text_type="diegetic_instrument", visible_text=["EXIT"], text_detected=True)])
    assert _lock(exit_sign)["forbidden_text_detected"] is False
    caption = _per([_target(text_type="subtitle_caption", visible_text=["hello"])])
    assert _lock(caption)["forbidden_text_detected"] is True
    other = _per([_target(text_type="other", visible_text=["xq"])])
    assert _lock(other)["forbidden_text_detected"] is True
    # o passe de decisao NAO pode reprovar texto diegetico por conta propria
    assert vca._apply_perception_locks({"forbidden_text_detected": True}, exit_sign, aircraft_required=False,
                                       aircraft_must_be_visible=False, identity_required=False
                                       )["forbidden_text_detected"] is False


def test_persistent_objects_need_one_visible_not_the_whole_inventory():
    inv = ["single central aisle", "3-3 seating", "overhead bins", "forward galley", "right bin 19"]
    per = _per([_target(visible_objects=["blue seats", "overhead compartments", "windows"])])
    raw = _lock(per, framing="wide", persistent_objects=inv)
    assert raw["persistent_objects_match"] is True  # 'overhead' casa
    none = _per([_target("a beach", visible_objects=["sand", "palm tree"])])
    assert _lock(none, framing="wide", persistent_objects=inv)["persistent_objects_match"] is False
    assert _lock(none, framing="close", persistent_objects=inv)["persistent_objects_match"] is True


def test_reference_images_counted_by_the_model_do_not_fail_the_gate():
    raw = {"image_received": True, "target_images_seen": 2, "visual_pass": True,
           "location_match": True, "framing_match": True, "subjects_match": True,
           "persistent_objects_match": True, "forbidden_text_detected": False}
    ok, _ = vca._decision(raw, expected_targets=1, aircraft_required=False, speaker_required=False,
                          identity_required=False)
    assert ok is True
    raw["target_images_seen"] = 0
    assert vca._decision(raw, expected_targets=1, aircraft_required=False, speaker_required=False,
                         identity_required=False)[0] is False


@pytest.mark.parametrize("field", ["visual_pass", "framing_match"])
def test_global_or_framing_failure_cannot_be_approved(field):
    raw = {"image_received": True, "target_images_seen": 1, "visual_pass": True,
           "location_match": True, "framing_match": True, "subjects_match": True,
           "persistent_objects_match": True, "forbidden_text_detected": False}
    raw[field] = False
    ok, reasons = vca._decision(raw, expected_targets=1, aircraft_required=False,
                                 speaker_required=False, identity_required=False)
    assert ok is False and f"failed checks: {field}" in " ".join(reasons)


def test_regen_brake_when_gate_blocks_most_of_the_film(tmp_path):
    calls = []
    ok = gate_with_retries(tmp_path, stage="stills", audit_cmd=["--stage", "stills"],
                           regen_cmd=lambda spec, seed: calls.append(spec) or ["regen"],
                           run_step=lambda n, c, obrigatorio=True: False,
                           blocked_fn=lambda: list(range(41)), base_seed=1, max_retries=2,
                           total_shots=68, log=lambda m: None)
    assert ok is False and calls == []


def test_identity_by_face_only_in_close_shots():
    bad = _per([_target()] * 3, identity_matches=[False, False])
    med = _lock(bad, identity_required=True, expected_refs=2, framing="medium",
                face={"frames": 3, "frames_with_face": 3, "min_similarity": 0.05})
    assert med["identity_match"] is True and med["subjects_match"] is True  # rosto pequeno: sem veredito por rosto
    close = _lock(bad, identity_required=True, expected_refs=2, framing="close")
    assert close["identity_match"] is False


def test_cabin_interior_is_not_blocked_by_on_ground_guess():
    cabin = _per([_target(aircraft_visible=True, aircraft_state="on_ground")] * 3)
    ok = _lock(cabin, aircraft_required=True, location_id="LOC_CABIN")
    assert ok["aircraft_state_match"] is True
    apron = _per([_target(aircraft_visible=True, aircraft_state="on_ground", ground_or_runway_visible=True)] * 3)
    assert _lock(apron, aircraft_required=True, location_id="LOC_CABIN")["aircraft_state_match"] is False
    # exterior continua exigindo aviao em voo
    ext = _per([_target("sky", aircraft_visible=True, aircraft_state="on_ground")])
    assert _lock(ext, aircraft_required=True, aircraft_must_be_visible=True,
                 location_id="LOC_SKY")["aircraft_state_match"] is False


def test_truncated_json_is_retried_once(monkeypatch):
    calls = {"n": 0}

    def once(model, prompt, images, *, timeout=180, retry=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("Unterminated string")
        return {"ok": True}
    monkeypatch.setattr(vca, "_ollama_json_once", once)
    assert vca._ollama_json("m", "p", []) == {"ok": True} and calls["n"] == 2


def test_subjects_are_not_named_by_the_decision_pass_without_face_verdict():
    crowd = _per([_target(people="many passengers seated")])
    # sem foto nominal e/ou fora de close: "expected no subjects but observed passengers" nao reprova
    raw = vca._apply_perception_locks({"subjects_match": False}, crowd, aircraft_required=False,
                                      aircraft_must_be_visible=False, identity_required=False, framing="wide")
    assert raw["subjects_match"] is True


def test_close_identity_needs_only_the_subject_reference():
    per = _per([_target()] * 3, identity_matches=[True, False])  # co-sujeito fora do quadro
    raw = _lock(per, identity_required=True, expected_refs=2, framing="close")
    assert raw["identity_match"] is True
    per2 = _per([_target()] * 3, identity_matches=[False, True])
    assert _lock(per2, identity_required=True, expected_refs=2, framing="close")["identity_match"] is False


def test_sky_has_no_location_reference(tmp_path):
    run = tmp_path
    (run / "shots").mkdir()
    ref = run / "shots" / "shot033_wide.png"
    ref.write_bytes(b"x")
    (run / "shots" / "location_refs.json").write_text(json.dumps({"LOC_SKY": str(ref), "LOC_CABIN": str(ref)}))
    assert vca._location_reference(run, {"location_id": "LOC_SKY", "framing": "wide"}) is None
    assert vca._location_reference(run, {"location_id": "LOC_CABIN", "framing": "wide"}) == ref


def test_extra_targets_listed_for_reference_photos_are_not_drift_in_a_still():
    per = _per([_target(), _target("someone's portrait")], same_primary_people_across_targets=False,
               same_location_across_targets=False)
    raw = _lock(per, location_id="LOC_CABIN", expected_targets=1)
    assert raw["subjects_match"] is True and raw["location_match"] is True
    # em clipe (3 quadros) a deriva declarada continua reprovando
    per3 = _per([_target()] * 3, same_primary_people_across_targets=False)
    assert _lock(per3, location_id="LOC_CABIN", expected_targets=3)["subjects_match"] is False


def test_retry_persists_per_shot_seed(tmp_path):
    ok = gate_with_retries(tmp_path, stage="stills", audit_cmd=["--stage", "stills"],
                           regen_cmd=lambda spec, seed: ["regen"],
                           run_step=lambda n, c, obrigatorio=True: True if c == ["regen"] else "--only-shots" in c,
                           blocked_fn=lambda: [3, 5], base_seed=1000, max_retries=1, log=lambda m: None)
    assert ok is True
    seeds = json.loads((tmp_path / "shots" / "seed_overrides.json").read_text(encoding="utf-8"))
    assert seeds == {"3": 1000 + 7919, "5": 1000 + 7919}


def test_animatic_uses_the_manifest_still_not_the_first_alphabetical_file(tmp_path):
    from script_pipeline.render_shots import still_candidates
    (tmp_path / "shot002_close.png").write_bytes(b"velho")   # sobra de um plano anterior
    (tmp_path / "shot002_ots.png").write_bytes(b"atual")
    (tmp_path / "stills.json").write_text(json.dumps({"2": {"file": "shot002_ots.png"}}), encoding="utf-8")
    assert still_candidates(tmp_path, 2)[0].name == "shot002_ots.png"
    # sem manifesto (ou arquivo do manifesto ausente) cai no glob
    (tmp_path / "stills.json").write_text("{}", encoding="utf-8")
    assert still_candidates(tmp_path, 2)[0].name == "shot002_close.png"


# ---- "corrija tudo" (2026-09-20) --------------------------------------------

def test_cockpit_never_satisfies_the_cabin_location():
    per = _per([_target("aircraft cabin interior next to the cockpit door"), _target("cockpit with two pilots")])
    assert _lock(per, location_id="LOC_CABIN", expected_targets=2)["location_match"] is False
    ok = _per([_target("airplane cabin interior")] * 3)
    assert _lock(ok, location_id="LOC_CABIN")["location_match"] is True
    assert _lock(_per([_target("passenger cabin with seat rows")]), location_id="LOC_COCKPIT")["location_match"] is False


def test_generic_words_do_not_satisfy_the_inventory():
    inv = ["passengers seated in the same rows throughout the flight", "right bin 19"]
    cockpit = _per([_target("cockpit", visible_objects=["pilots seated on the right", "instrument panel"])])
    assert _lock(cockpit, framing="wide", persistent_objects=inv)["persistent_objects_match"] is False
    cabin = _per([_target("cabin", visible_objects=["passengers", "seats"])])
    assert _lock(cabin, framing="wide", persistent_objects=inv)["persistent_objects_match"] is True


def test_warnings_flag_marginal_face_and_low_resolution(tmp_path):
    from PIL import Image
    small = tmp_path / "s.png"
    Image.new("RGB", (384, 256)).save(small)
    w = vca._entry_warnings([small], {"min_similarity": 0.38})
    assert any("marginal" in x for x in w) and any("baixa resolucao" in x for x in w)
    big = tmp_path / "b.png"
    Image.new("RGB", (960, 544)).save(big)
    assert vca._entry_warnings([big], {"min_similarity": 0.6}) == []


def test_location_dependents_follow_position_not_source_index():
    plan = {"shots": [{"index": 14 + i, "scene": 2, "location_id": "LOC_SKY"} for i in range(4)]}
    # plano recortado: index-fonte 14..17, mas os arquivos sao shot000..shot003
    assert dependents(plan, {"LOC_SKY": 1}) == [2, 3]


def test_visual_diagnostic_refuses_video_and_final(monkeypatch):
    import sys
    from script_pipeline import run_decupagem as rd
    monkeypatch.setattr("script_pipeline.ollama_runtime.unload_all", lambda *a, **k: None)
    for ate in ("render", "final"):
        monkeypatch.setattr(sys, "argv", ["run_decupagem", "--run-dir", "x", "--ate", ate, "--visual-diagnostic"])
        with pytest.raises(SystemExit) as exc:
            rd.main()
        assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", ["run_decupagem", "--run-dir", "x", "--ate", "animatic",
                                      "--visual-diagnostic", "--no-visual-audit"])
    with pytest.raises(SystemExit) as exc:
        rd.main()
    assert exc.value.code == 2  # diagnostico roda o gate; nao combina com desligar o gate


@pytest.mark.parametrize("extra", [["--no-visual-audit"], ["--visual-unresolved", "continue"]])
def test_final_refuses_visual_gate_bypass(monkeypatch, extra):
    import sys
    from script_pipeline import run_decupagem as rd
    monkeypatch.setattr("script_pipeline.ollama_runtime.unload_all", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_decupagem", "--run-dir", "x", *extra])
    with pytest.raises(SystemExit) as exc:
        rd.main()
    assert exc.value.code == 2


def test_clip_identity_strict_fails_delivery_on_alert(monkeypatch, tmp_path):
    from script_pipeline import clip_identity_audit as audit
    monkeypatch.setattr(audit, "audit_run", lambda *_a, **_k: {"shot": {"alerta": True}})
    assert audit.main(["--run-dir", str(tmp_path), "--strict"]) == 1
    assert audit.main(["--run-dir", str(tmp_path)]) == 0


def test_prune_moves_stale_stills_and_manifest_entries(tmp_path):
    from script_pipeline.render_shots import prune_stale_stills
    for name in ("shot000_close.png", "shot000_wide.png", "shot001_ots.png", "shot007_close.png", "shot002_x.png"):
        (tmp_path / name).write_bytes(b"x")
    man = {"0": {"file": "shot000_wide.png"}, "1": {"file": "shot001_ots.png"}, "7": {"file": "shot007_close.png"}}
    (tmp_path / "stills.json").write_text(json.dumps(man), encoding="utf-8")
    moved = prune_stale_stills(tmp_path, 3, log=lambda m: None)
    assert moved == 2  # shot000_close (manifesto aponta outro) e shot007 (indice fora do plano)
    assert (tmp_path / "_stale" / "shot000_close.png").exists() and (tmp_path / "_stale" / "shot007_close.png").exists()
    assert (tmp_path / "shot002_x.png").exists()  # indice valido sem entrada de manifesto: preservado
    assert sorted(json.loads((tmp_path / "stills.json").read_text(encoding="utf-8"))) == ["0", "1"]


def test_spatial_audit_deterministic_locks(monkeypatch, tmp_path):
    from script_pipeline import spatial_audit as sa
    from script_pipeline import visual_continuity_audit as v
    img = tmp_path / "a.png"
    from PIL import Image
    Image.new("RGB", (960, 544)).save(img)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 64)).save(ref)
    caption = [{"text_kind": "caption_or_watermark", "visible_text": "HELLO"}]
    locks, _ = sa.deterministic_locks({"framing": "wide"}, "stills", [img], caption, tmp_path)
    assert locks and "texto artificial" in locks[0]
    sign = [{"text_kind": "diegetic_sign_or_instrument", "visible_text": "EXIT"}]
    assert sa.deterministic_locks({"framing": "wide"}, "stills", [img], sign, tmp_path)[0] == []
    monkeypatch.setattr(v, "face_check", lambda frames, refs: {"frames": 1, "frames_with_face": 1, "min_similarity": 0.1})
    locks, _ = sa.deterministic_locks({"framing": "close", "references": [str(ref)]}, "stills", [img], sign, tmp_path)
    assert locks and "similaridade facial" in locks[0]
    locks, _ = sa.deterministic_locks({"framing": "wide", "references": [str(ref)]}, "stills", [img], sign, tmp_path)
    assert locks == []


def test_single_subject_prompt_only_removes_the_partner_clause():
    from script_pipeline.spatial_pipeline import _single_subject_prompt
    prompt = ("Cinematic frame with natural lighting. Aircraft cabin, JI-HO with a calm smile in a navy uniform. "
              "Two crew with SEO-YEON, black bob and round glasses. positioned on the left. Fixed layout.")
    out = _single_subject_prompt(prompt, "JI-HO")
    assert "with natural lighting" in out          # contexto anterior preservado
    assert "calm smile in a navy uniform" in out   # descritor do proprio sujeito preservado
    assert "SEO-YEON" not in out                   # so a clausula do parceiro saiu
    assert "positioned on the left" in out and out.startswith("Single-person close-up of JI-HO only.")


def test_regen_failure_is_not_silently_reaudited(tmp_path):
    """Auditoria 2026-09-21: regen crashando nao pode virar reauditoria do still velho."""
    calls = []

    def run_step(n, c, obrigatorio=True):
        calls.append(c)
        return False   # audit inicial reprova e o regen sempre falha
    ok = gate_with_retries(tmp_path, stage="stills", audit_cmd=["--stage", "stills"],
                           regen_cmd=lambda s, seed: ["regen"], run_step=run_step,
                           blocked_fn=lambda: [2], base_seed=1, max_retries=2, log=lambda m: None)
    assert ok is False
    reauditorias = [c for c in calls if "--only-shots" in c]
    assert reauditorias == []   # nunca reauditou: a regeneracao nunca teve sucesso
    info = json.loads((tmp_path / "shots" / "visual_gate_stills_retries.json").read_text(encoding="utf-8"))
    assert all(r.get("regen_failed") for r in info["rounds"])


# ---- bug do cabecalho "Cena N -" nao separando local/periodo (2026-09-23) ----

def test_cena_style_heading_splits_time_of_day():
    from script_pipeline.parse_screenplay import parse_structure
    texto = "CENA 1 - AVENIDA DE SEUL, ENTRADA DE UM HOTEL - DIA\n\nA multidao corre.\n"
    scenes = parse_structure(texto)
    assert scenes[0].location == "AVENIDA DE SEUL, ENTRADA DE UM HOTEL"
    assert scenes[0].time_of_day.lower() == "dia"


def test_storyboard_prompt_lowercases_shouting_location():
    from script_pipeline.shot_plan import _storyboard_prompt
    prompt = _storyboard_prompt(framing="wide", angle="eye", subject="", location="AVENIDA DE SEUL",
                                time_of_day="day", look="", descriptor="", screen_side=None)
    assert "AVENIDA DE SEUL" not in prompt and "avenida de seul" in prompt
