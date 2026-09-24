import copy
from pathlib import Path

import pytest

from script_pipeline.production_project import (
    create_project, read, write, restore_source, conform_plan, export_otio, source_ledger,
)
from script_pipeline.parse_screenplay import parse_structure
from script_pipeline.shot_plan import plan_all


SCRIPT = '''### CENA 1: SALA (0:00 - 0:20)

**INT. SALA - DIA**

A porta abre.

**ANA**
(calma)
*Bom dia.*

A porta fecha.
'''


def prepare(tmp_path):
    project, run = tmp_path / 'film', tmp_path / 'run'
    script = create_project(project, run, SCRIPT, 'Test', 20)
    scenes = [s.to_dict() for s in parse_structure(Path(script).read_text(encoding='utf-8'))]
    write(run / 'parse/scenes.json', scenes)
    restore_source(run)
    scenes = read(run / 'parse/scenes_enriched.json')
    plan = plan_all(scenes, None, style_name='classico', fps=24)
    write(run / 'parse/shot_plan.json', plan)
    return project, run, plan


def test_source_order_and_literal_dialogue():
    scenes, _ = source_ledger(SCRIPT)
    assert [u['type'] for u in scenes[0]['units']] == ['action', 'dialogue', 'action']
    assert scenes[0]['dialogue'][0]['text'] == 'Bom dia.'
    assert scenes[0]['target_seconds'] == 20


def test_duration_coverage_and_otio(tmp_path):
    project, run, _ = prepare(tmp_path)
    conform_plan(run)
    edit = read(project / 'editorial/timeline.json')
    assert edit['duration_frames'] == 480
    assert read(project / 'editorial/cobertura.json')['complete']
    assert len({s['id'] for s in edit['shots']}) == len(edit['shots'])
    export_otio(project)
    import opentimelineio as otio
    timeline = otio.adapters.read_from_file(str(project / 'editorial/timeline.otio'))
    assert timeline.duration().to_seconds() == 20


def test_changed_prompt_invalidates_approval(tmp_path):
    project, run, plan = prepare(tmp_path)
    conform_plan(run)
    edit = read(project / 'editorial/timeline.json')
    for s in edit['shots']:
        s['approval'] = 'approved'
    write(project / 'editorial/timeline.json', edit)
    write(run / 'parse/shot_plan.json', plan)
    conform_plan(run)
    assert all(s['approval'] == 'approved' for s in read(project / 'editorial/timeline.json')['shots'])
    plan['shots'][0]['video_prompt'] += ' Rain.'
    write(run / 'parse/shot_plan.json', plan)
    conform_plan(run)
    assert read(project / 'editorial/timeline.json')['shots'][0]['approval'] == 'pending'


def test_missing_action_blocks_production(tmp_path):
    project, run, plan = prepare(tmp_path)
    plan['shots'] = [s for s in plan['shots'] if s['type'] != 'action']
    write(run / 'parse/shot_plan.json', plan)
    with pytest.raises(ValueError):
        conform_plan(run)


def test_enrichment_cannot_rewrite_speech(tmp_path):
    project, run, _ = prepare(tmp_path)
    data = read(run / 'parse/scenes_enriched.json')
    data[0]['dialogue'][0]['text'] = 'Rewritten'
    write(run / 'parse/scenes_enriched.json', data)
    with pytest.raises(ValueError, match='Falas alteradas'):
        restore_source(run)


def test_restore_reuses_aligned_parse_visuals_without_second_llm(tmp_path, monkeypatch):
    project, run = tmp_path / 'film', tmp_path / 'run'
    script = create_project(project, run, SCRIPT, 'Test', 20)
    scenes = [s.to_dict() for s in parse_structure(Path(script).read_text(encoding='utf-8'))]
    scenes[0]['shot_list'] = [
        {'type': 'action', 'visual': 'The door opens.', 'actor': None},
        {'type': 'dialogue', 'line_index': 0, 'visual': 'Ana greets the room.', 'actor': 'ANA'},
        {'type': 'action', 'visual': 'The door closes.', 'actor': None},
    ]
    write(run / 'parse/scenes_enriched.json', scenes)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError('aligned parse must not call the LLM again')

    monkeypatch.setattr('script_pipeline.story_structure._call_ollama', unexpected_call)
    restore_source(run, engine='any-local-model')
    restored = read(run / 'parse/scenes_enriched.json')[0]
    assert [item['visual'] for item in restored['shot_list']] == [
        'The door opens.', 'Ana greets the room.', 'The door closes.'
    ]


def test_long_script_preserves_120_scenes():
    script = '\n\n'.join(f'INT. SALA {i} - DIA\n\nA porta abre.\n\nANA\nBom dia número {i}.\n\nA porta fecha.' for i in range(120))
    scenes, _ = source_ledger(script)
    assert len(scenes) == 120
    assert sum(len(s['dialogue']) for s in scenes) == 120
    assert sum(len(s['units']) for s in scenes) == 360


def test_coverage_checks_identity_not_just_count(tmp_path):
    project, run, plan = prepare(tmp_path)
    for shot in plan['shots']:
        if shot.get('source_unit_id'):
            shot['source_unit_id'] = 'invented'
            break
    write(run / 'parse/shot_plan.json', plan)
    with pytest.raises(ValueError, match='Cobertura incompleta'):
        conform_plan(run)


def test_generation_budget_and_attempts(tmp_path):
    from script_pipeline.production_project import reserve_generation
    project, run, _ = prepare(tmp_path)
    shot = {'id': 's1', 'seconds': 3}
    for _ in range(3):
        reserve_generation(run, shot)
    with pytest.raises(ValueError, match='tentativas'):
        reserve_generation(run, shot)
    with pytest.raises(ValueError, match='Orçamento'):
        reserve_generation(run, {'id': 's2', 'seconds': 100})


def test_approval_requires_matching_real_take(tmp_path):
    from script_pipeline.production_project import validate_timeline
    project, run, _ = prepare(tmp_path)
    conform_plan(run)
    edit = read(project / 'editorial/timeline.json')
    validate_timeline(edit)
    edit['shots'][0]['approval'] = 'approved'
    with pytest.raises(ValueError, match='Tomada ausente'):
        validate_timeline(edit)


def test_audio_stems_are_equal_length_and_report_missing(tmp_path):
    from script_pipeline.production_post import audio_stems
    import wave
    project, run, _ = prepare(tmp_path)
    conform_plan(run)
    write(project / 'audio/cues.json', {'foley': [{'media': None, 'start': 0, 'duration': 1, 'text': 'door'}]})
    report = audio_stems(run)
    assert len(report['foley']['missing']) == 1
    assert report['foley']['silent']
    for name in report:
        with wave.open(str(project / 'audio/stems' / (name + '.wav'))) as wav:
            assert wav.getnframes() == 20 * 48000
            assert wav.getnchannels() == 2


def test_aircraft_rule_is_opt_in_for_generic_scripts(tmp_path):
    from script_pipeline.continuity_audit import build_report
    run = tmp_path / 'generic_run'
    (run / 'parse').mkdir(parents=True)
    write(run / 'parse/shot_plan.json', {'shots': [{
        'index': 0, 'scene': 1, 'location_id': 'LOC_ROOM', 'framing': 'wide',
        'subject': 'ANA', 'co_subject': '', 'storyboard_prompt': 'same room, table and lamp',
        'video_prompt': 'ANA crosses the room', 'continuity': {'prompt': 'same room'}
    }]})
    report = build_report(run)
    assert report['status'] == 'ok'
    assert not any(item['type'] == 'aircraft_state_drift' for item in report['blocking'])


def test_aircraft_rule_does_not_leak_from_one_shot_to_another(tmp_path):
    from script_pipeline.continuity_audit import build_report
    run = tmp_path / 'mixed_run'
    (run / 'parse').mkdir(parents=True)
    write(run / 'parse/shot_plan.json', {'shots': [
        {
            'index': 0, 'scene': 1, 'location_id': 'LOC_SKY', 'framing': 'wide',
            'subject': '', 'co_subject': '',
            'storyboard_prompt': 'the same twin-engine passenger jet remains airborne',
            'video_prompt': 'the jet keeps flying',
            'continuity': {'aircraft_contract': 'same aircraft airborne'},
        },
        {
            'index': 1, 'scene': 2, 'location_id': 'LOC_ROOM', 'framing': 'wide',
            'subject': 'ANA', 'co_subject': '', 'storyboard_prompt': 'same room',
            'video_prompt': 'ANA crosses the room', 'continuity': {'prompt': 'same room'},
        },
    ]})
    report = build_report(run)
    assert report['status'] == 'ok'
    assert not any(item.get('shot') == 1 and item['type'] == 'aircraft_state_drift'
                   for item in report['blocking'])


def test_continuity_audit_detects_secondary_missing_from_planned_shots(tmp_path):
    from script_pipeline.continuity_audit import build_report
    run = tmp_path / 'run'
    (run / 'parse').mkdir(parents=True)
    write(run / 'parse/shot_plan.json', {'shots': [{
        'index': 0, 'scene': 1, 'location_id': 'LOC_ROOM', 'framing': 'wide',
        'subject': 'ANA', 'co_subject': '', 'storyboard_prompt': 'same room',
        'video_prompt': 'ANA crosses the room', 'continuity': {'prompt': 'same room'},
        'continuity_contract': {
            'location_id': 'LOC_ROOM', 'scene_characters': ['ANA', 'BIA'],
            'persistent_objects': [],
        },
    }]})
    report = build_report(run)
    assert report['status'] == 'blocked'
    assert any(item['type'] == 'secondary_character_missing'
               and item['characters'] == ['BIA'] for item in report['blocking'])


def test_continuity_audit_is_opt_in_when_project_has_no_location_contracts(tmp_path):
    """Achado 2026-09-21: roteiro avulso sem projeto mestre/location bible nunca preenche
    location_id/continuity no shot_plan -- block_on_missing bloqueava TODO plano em silencio."""
    from script_pipeline.continuity_audit import build_report
    run = tmp_path / 'run_sem_biblia'
    (run / 'parse').mkdir(parents=True)
    write(run / 'parse/shot_plan.json', {'shots': [
        {'index': 0, 'scene': 1, 'framing': 'wide', 'subject': 'LYRA', 'co_subject': '',
         'storyboard_prompt': 'a stone chamber', 'video_prompt': 'LYRA touches a rune'},
        {'index': 1, 'scene': 1, 'framing': 'close', 'subject': 'THOREN', 'co_subject': '',
         'storyboard_prompt': 'the same chamber', 'video_prompt': 'THOREN raises his shield'},
    ]})
    report = build_report(run)
    assert report['status'] == 'ok'
    assert report['blocking'] == []
