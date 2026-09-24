import copy
import json

import numpy as np
import pytest

from script_pipeline.world_store import WorldStore, apply_operations
from script_pipeline.camera_geometry import project, y_up_to_z_up, keyframe_indices
from script_pipeline.spatial_pipeline import demo_spec, prepare
from script_pipeline.spatial_conditioning import wire_spatial
from script_pipeline.spatial_planner import (
    active_entities_for_shot, camera_for_shot, state_for_shot, validate_spatial_shot,
)


@pytest.fixture
def world(tmp_path):
    spec = demo_spec()
    with WorldStore(tmp_path) as store:
        version = store.register_asset(spec['location_id'],spec['location'])
        state = dict(schema_version=1,coordinates='Z_UP_METERS',location_id=spec['location_id'],
                     location_asset=version,entities=spec['entities'])
        initial = store.create(state)
        yield store,initial,spec['shots'][2]['event']


def test_transfer_atomic_idempotent_and_persistent(world):
    store,initial,event = world
    final = store.apply(initial,event)
    assert store.apply(initial,event) == final
    assert store.get(initial)['entities']['BOOK_01']['attachment']['entity_id'] == 'ANA'
    assert store.get(final)['entities']['BOOK_01']['attachment']['entity_id'] == 'PEDRO'
    with WorldStore(store.root.parent) as reopened:
        assert reopened.get(final) == store.get(final)


def test_event_id_cannot_change_parent(world):
    store,initial,event = world
    final = store.apply(initial,event)
    with pytest.raises(ValueError,match='reused'):
        store.apply(final,event)


def test_failed_batch_rolls_back(world):
    store,initial,event = world
    broken = copy.deepcopy(event)
    broken['operations'].append(dict(op='transfer',entity_id='BOOK_01',to='GHOST',socket='left_hand',**{'from':'PEDRO'}))
    with pytest.raises(ValueError,match='attachment'):
        store.apply(initial,broken)
    assert store.db.execute('SELECT COUNT(*) FROM continuity_events').fetchone()[0] == 0
    assert store.db.execute('SELECT COUNT(*) FROM scene_states').fetchone()[0] == 1


def test_unknown_operation_rejected(world):
    store,initial,_ = world
    with pytest.raises(ValueError):
        apply_operations(store.get(initial),[{'op':'erase','entity_id':'ANA'}])


def test_nan_rejected(world):
    store,initial,_ = world
    with pytest.raises(ValueError):
        apply_operations(store.get(initial),[dict(op='set',entity_id='ANA',field='position',
            expected=[-.48,1,0],value=[float('nan'),1,0])])


def test_camera_only_changes_binding(world):
    store,initial,_ = world
    cameras = [s['camera'] for s in demo_spec()['shots']]
    a = store.bind('A',initial,initial,cameras[0])
    b = store.bind('B',initial,initial,cameras[1])
    assert a['initial'] == b['initial']
    assert a['camera'] != b['camera']
    assert store.db.execute('SELECT COUNT(*) FROM scene_states').fetchone()[0] == 1


def test_perspective_center_and_behind_camera():
    camera = dict(position=[0,-5,1],target=[0,0,1],lens_mm=36,sensor_mm=36,width=768,height=512)
    xy,depth,visible = project([[0,0,1],[1,0,1],[0,-6,1]],camera)
    np.testing.assert_allclose(xy[0],[384,256])
    assert xy[1,0] > xy[0,0]
    assert visible.tolist() == [True,True,False]
    assert np.isnan(xy[2]).all()


def test_coordinate_conversion_preserves_handedness():
    converted = y_up_to_z_up(np.eye(3))
    assert np.linalg.det(converted) == 1
    np.testing.assert_allclose(y_up_to_z_up([0,1,0]),[0,0,1])


@pytest.mark.parametrize('frames',[9,24,72,120,121,480])
def test_keyframe_destination_survives_editorial_trim(frames):
    generated,index = keyframe_indices(frames)
    assert generated >= frames and (generated-1)%8 == 0
    assert index < frames and index%8 == 0


def test_prepare_four_shots_share_state_and_return(tmp_path):
    shots = prepare(tmp_path,demo_spec())
    assert len(shots) == 4
    assert shots[0]['binding']['initial'] == shots[1]['binding']['initial']
    assert shots[2]['binding']['final'] == shots[3]['binding']['initial']
    assert shots[0]['camera'] == shots[3]['camera']
    plan = json.loads((tmp_path/'world/shot_bindings.json').read_text())
    assert plan['duration_frames'] == 480
    assert prepare(tmp_path,demo_spec())[2]['binding'] == shots[2]['binding']


def test_spatial_adapter_rejects_unsupported_backend():
    with pytest.raises(ValueError,match='Klein'):
        wire_spatial({}, {}, architecture='flux1',stage_image=lambda p:p)


def test_spatial_reference_mode_keeps_character_latent(monkeypatch):
    import script_pipeline.spatial_conditioning as conditioning
    monkeypatch.setattr(conditioning, 'validate_bundle', lambda bundle: 'beauty.png')
    workflow = {
        'latent': {'class_type':'EmptyLatentImage',
                   'inputs':{'width':384,'height':256}},
        'character_ref': {'class_type':'ReferenceLatent',
                          'inputs':{'conditioning':['prompt',0], 'latent':['char_latent',0]}},
        'sampler': {'class_type':'KSampler',
                    'inputs':{'positive':['character_ref',0],
                              'latent_image':['latent',0], 'denoise':1.0}},
        'decode': {'class_type':'VAEDecode',
                   'inputs':{'samples':['sampler',0], 'vae':['vae',0]}},
    }
    bundle = {'camera':{'width':384,'height':256}, 'fingerprint':'abc'}
    wire_spatial(workflow, bundle, architecture='flux',
                 stage_image=lambda path:path, denoise=.65, mode='reference')
    assert workflow['sampler']['inputs']['positive'] == ['character_ref',0]
    assert workflow['character_ref']['inputs']['conditioning'] == ['spatial_ref',0]
    assert workflow['spatial_ref']['inputs']['conditioning'] == ['prompt',0]
    assert workflow['sampler']['inputs']['latent_image'] == ['latent',0]
    assert workflow['sampler']['inputs']['denoise'] == 1.0


def test_spatial_img2img_mode_uses_blocking_latent(monkeypatch):
    import script_pipeline.spatial_conditioning as conditioning
    monkeypatch.setattr(conditioning, 'validate_bundle', lambda bundle: 'beauty.png')
    workflow = {
        'latent': {'class_type':'EmptyLatentImage',
                   'inputs':{'width':384,'height':256}},
        'sampler': {'class_type':'KSampler',
                    'inputs':{'positive':['prompt',0],
                              'latent_image':['latent',0], 'denoise':1.0}},
        'decode': {'class_type':'VAEDecode',
                   'inputs':{'samples':['sampler',0], 'vae':['vae',0]}},
    }
    bundle = {'camera':{'width':384,'height':256}, 'fingerprint':'abc'}
    wire_spatial(workflow, bundle, architecture='flux',
                 stage_image=lambda path:path, denoise=.72, mode='img2img')
    assert workflow['sampler']['inputs']['latent_image'] == ['spatial_encode',0]
    assert workflow['sampler']['inputs']['denoise'] == .72


def test_stale_control_cannot_reuse_cache(tmp_path):
    from script_pipeline.spatial_conditioning import validate_bundle
    from script_pipeline.scene_composer import file_hash
    p = tmp_path/'beauty.png'
    p.write_bytes(b'old')
    bundle = {'schema_version':1,'state_hash':'state','files':{'beauty':str(p)},'hashes':{'beauty':file_hash(p)}}
    p.write_bytes(b'new')
    with pytest.raises(ValueError,match='stale'):
        validate_bundle(bundle)


def test_gate_approval_invalidated_by_media_change(tmp_path):
    from script_pipeline.production_project import write
    from script_pipeline.spatial_audit import require_approval, target_hash
    media = tmp_path/'still.png'
    media.write_bytes(b'original')
    shot = {'id':'A','binding':{'initial':'s1'},'stills':{'initial':str(media),'final':str(media)}}
    write(tmp_path/'world/shot_bindings.json',{'shots':[shot]})
    write(tmp_path/'world/audit_stills.json',{'shots':[{'shot_id':'A','approved':True,'target_hash':target_hash(shot,'stills')}]})
    require_approval(tmp_path,'stills')
    media.write_bytes(b'regenerated')
    with pytest.raises(ValueError,match='stale'):
        require_approval(tmp_path,'stills')


def test_spatial_still_key_changes_with_state_and_camera():
    from script_pipeline.render_shots import _still_key
    shot = {'storyboard_prompt':'same prompt','spatial':{'fingerprint':'a'}}
    a = _still_key(shot,None,768,512)
    shot['spatial']['fingerprint'] = 'b'
    assert _still_key(shot,None,768,512) != a


def test_decupagem_ui_emits_spatial_cli_flags(tmp_path):
    from decupagem_ui import _argv
    spec = tmp_path/'scene.json'
    spec.write_text('{}', encoding='utf-8')
    command = _argv(tmp_path, '', 'classico', '', 768, 512, 'stills',
                    'qwen2.5:32b-instruct-q4_K_M', False,
                    spatial_on=True, spatial_spec=str(spec), spatial_denoise=.9)
    assert '--spatial-spec' in command
    assert command[command.index('--spatial-denoise') + 1] == '0.9'
    assert '--reuse-plan' in command


def test_decupagem_ui_rejects_spatial_without_flux(tmp_path):
    from decupagem_ui import _argv
    spec = tmp_path/'scene.json'
    spec.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='flux'):
        _argv(tmp_path, '', 'classico', '', 768, 512, 'stills',
              'qwen2.5:32b-instruct-q4_K_M', False, motor_img='sd35',
              spatial_on=True, spatial_spec=str(spec))


def test_close_camera_targets_subject_and_uses_portrait_lens():
    entities = {'ANA': {'kind':'character','position':[1,2,0]}}
    base = dict(position=[0,-5,1.7],target=[0,2,1.4],lens_mm=38,
                sensor_mm=36,width=768,height=512,near=.05,far=30)
    shot = {'id':'A','framing':'close','subject':'ANA','screen_side':'right'}
    camera = camera_for_shot(base, shot, entities)
    planned = dict(shot, camera=camera, active_entities=['ANA'])
    assert camera['target'] == [1.0,2.0,1.55]
    assert camera['lens_mm'] == 72
    assert camera != base
    assert validate_spatial_shot(planned, entities)


def test_tight_shot_hides_other_entities_without_mutating_world():
    entities = {
        'ANA': {'kind':'character','position':[0,1,0],'present':True},
        'PEDRO': {'kind':'character','position':[1,1,0],'present':True},
        'CART': {'kind':'prop','position':[0,3,0],'present':True},
    }
    active = active_entities_for_shot({'framing':'close','subject':'ANA'}, entities,
                                      location_cast=['ANA','PEDRO','CART'])
    state = {'entities':entities}
    rendered = state_for_shot(state, active)
    assert active == ['ANA']
    assert rendered['entities']['ANA']['present'] is True
    assert rendered['entities']['PEDRO']['present'] is False
    assert state['entities']['PEDRO']['present'] is True


def test_open_shot_uses_location_cast_only():
    entities = {
        'PILOT': {'kind':'character','position':[0,1,0]},
        'PASSENGER': {'kind':'character','position':[0,9,0]},
    }
    assert active_entities_for_shot({'framing':'wide'}, entities,
                                    location_cast=['PILOT']) == ['PILOT']


def test_medium_shot_uses_only_named_blocking_participants():
    entities = {
        'A': {'kind':'character','position':[0,1,0]},
        'B': {'kind':'character','position':[1,1,0]},
        'C': {'kind':'character','position':[2,1,0]},
    }
    shot = {'framing':'medium','subject':'A','co_subject':'B'}
    assert active_entities_for_shot(shot, entities,
                                    location_cast=['A','B','C']) == ['A','B']


def test_spatial_close_prompt_removes_explicit_partner_and_is_idempotent():
    from script_pipeline.spatial_pipeline import _single_subject_prompt
    source = ("close-up. COPILOTO, blonde hair. with PILOTO, black hair and navy uniform. "
              "positioned on the left. interior, cockpit. Scene roster: COPILOTO, PILOTO.")
    once = _single_subject_prompt(source, 'COPILOTO')
    twice = _single_subject_prompt(once, 'COPILOTO')
    assert 'with PILOTO' not in once
    assert 'positioned on the left' in once
    assert 'Scene roster: COPILOTO, PILOTO' in once
    assert twice == once


def test_spatial_ui_preserves_existing_plan(tmp_path):
    from decupagem_ui import _argv
    spec = tmp_path/'scene.json'
    spec.write_text('{}', encoding='utf-8')
    command = _argv(tmp_path, '', 'classico', '', 768, 512, 'animatic',
                    'qwen2.5:32b-instruct-q4_K_M', False,
                    spatial_on=True, spatial_spec=str(spec))
    assert '--reuse-plan' in command
