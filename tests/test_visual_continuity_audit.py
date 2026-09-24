from script_pipeline.visual_continuity_audit import _apply_perception_locks, _decision


def _base_result():
    return {
        "image_received": True,
        "target_images_seen": 1,
        "visual_pass": True,
        "location_match": True,
        "framing_match": True,
        "subjects_match": True,
        "speaker_remains_visible": True,
        "persistent_objects_match": True,
        "aircraft_state_match": True,
        "forbidden_text_detected": False,
        "identity_match": True,
        "reasons": [],
    }


def test_grounded_aircraft_cannot_be_approved_by_decision_pass():
    perception = {
        "image_received": True,
        "target_images_seen": 1,
        "targets": [{
            "aircraft_state": "on_ground",
            "ground_or_runway_visible": True,
            "wheels_touching_surface": True,
            "text_detected": False,
        }],
    }
    locked = _apply_perception_locks(_base_result(), perception,
                                     aircraft_required=True, aircraft_must_be_visible=True,
                                     identity_required=False)
    ok, _ = _decision(locked, expected_targets=1, aircraft_required=True,
                      speaker_required=False, identity_required=False)
    assert not ok
    assert locked["aircraft_state_match"] is False


def test_detected_text_cannot_be_hidden_by_decision_pass():
    perception = {
        "image_received": True,
        "target_images_seen": 1,
        "targets": [{
            "aircraft_state": "absent",
            "ground_or_runway_visible": False,
            "wheels_touching_surface": False,
            "text_detected": True,
            "text_type": "subtitle_caption",
        }],
    }
    locked = _apply_perception_locks(_base_result(), perception,
                                     aircraft_required=False, aircraft_must_be_visible=False,
                                     identity_required=False)
    ok, _ = _decision(locked, expected_targets=1, aircraft_required=False,
                      speaker_required=False, identity_required=False)
    assert not ok
    assert locked["forbidden_text_detected"] is True


def test_diegetic_instrument_text_is_allowed():
    perception = {
        "image_received": True,
        "target_images_seen": 1,
        "targets": [{
            "aircraft_state": "absent",
            "ground_or_runway_visible": False,
            "wheels_touching_surface": False,
            "text_detected": True,
            "text_type": "diegetic_instrument",
        }],
    }
    locked = _apply_perception_locks(_base_result(), perception,
                                     aircraft_required=False, aircraft_must_be_visible=False,
                                     identity_required=False)
    assert locked["forbidden_text_detected"] is False


def test_close_with_matching_reference_may_crop_inventory():
    perception = {
        "image_received": True,
        "target_images_seen": 1,
        "targets": [{
            "location_description": "Airplane cabin interior with passengers",
            "aircraft_state": "airborne",
            "ground_or_runway_visible": False,
            "wheels_touching_surface": False,
            "text_type": "none",
        }],
        "identity_matches": [True],
        "primary_person_visible_in_every_target": True,
        "reference_subject_prominent_in_every_target": True,
    }
    raw = _base_result()
    raw.update(subjects_match=False, persistent_objects_match=False)
    locked = _apply_perception_locks(
        raw, perception, aircraft_required=True, aircraft_must_be_visible=False,
        identity_required=True, location_id="LOC_CABIN", framing="close",
        speaker_required=False,
    )
    ok, _ = _decision(locked, expected_targets=1, aircraft_required=True,
                      speaker_required=False, identity_required=True)
    assert ok


def test_dialogue_close_blocks_partially_visible_background_speaker():
    perception = {
        "image_received": True,
        "target_images_seen": 3,
        "targets": [{
            "location_description": "Airplane cabin interior",
            "framing": "Medium shot",
            "primary_people_description": "Multiple passengers; referenced person partially visible",
            "aircraft_state": "airborne",
            "ground_or_runway_visible": False,
            "wheels_touching_surface": False,
            "text_type": "none",
        }] * 3,
        "identity_matches": [True],
        "reference_subject_prominent_in_every_target": True,
    }
    raw = _base_result()
    locked = _apply_perception_locks(
        raw, perception, aircraft_required=True, aircraft_must_be_visible=False,
        identity_required=True, location_id="LOC_CABIN", framing="close",
        speaker_required=True,
    )
    ok, _ = _decision(locked, expected_targets=3, aircraft_required=True,
                      speaker_required=True, identity_required=True)
    assert not ok
    assert locked["speaker_remains_visible"] is False


def test_location_reference_mismatch_cannot_be_approved_by_decision_pass():
    perception = {
        "image_received": True,
        "target_images_seen": 1,
        "targets": [{"location_description": "airplane cabin", "text_type": "none"}],
        "same_location_as_reference": False,
    }
    locked = _apply_perception_locks(
        _base_result(), perception, aircraft_required=False,
        aircraft_must_be_visible=False, identity_required=False,
        location_id="LOC_CABIN", framing="medium",
        location_reference_required=True,
    )
    ok, _ = _decision(locked, expected_targets=1, aircraft_required=False,
                      speaker_required=False, identity_required=False)
    assert not ok
    assert locked["location_match"] is False


def test_evaluation_prompt_allows_diegetic_environmental_text():
    from script_pipeline.visual_continuity_audit import _evaluation_prompt
    prompt = _evaluation_prompt({"framing": "wide"}, stage="stills", target_count=1, ref_count=0,
                                perception={})
    assert "does NOT by itself fail" in prompt
    assert "diegetic_instrument" in prompt
    # a regra antiga (rejeitar QUALQUER texto ilegivel) nao deve mais aparecer sozinha
    assert "Reject invented subtitles, captions, watermarks, logos or illegible text." not in prompt
    assert "large/prominent lettering that dominates the frame" in prompt
