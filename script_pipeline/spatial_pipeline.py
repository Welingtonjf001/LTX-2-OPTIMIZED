"""Persistent spatial production: prepare, blocking, stills and LTX video.

python -m script_pipeline.spatial_pipeline --project PATH --demo --stage all
Custom projects use --spec FILE; no implicit invented coordinates on migration.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from script_pipeline.world_store import WorldStore, digest
from script_pipeline.production_project import read, write
from script_pipeline.camera_geometry import keyframe_indices
from script_pipeline.scene_composer import compose, file_hash


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = 'C:/ffmpeg/bin/ffmpeg.exe'
FFPROBE = 'C:/ffmpeg/bin/ffprobe.exe'


def demo_spec():
    location = {'boxes':[
        {'id':'FLOOR','position':[0,1,-.1],'size':[8,10,.2],'color':[.34,.36,.37]},
        {'id':'BACK_WALL','position':[0,4,1.65],'size':[8,.15,3.3],'color':[.69,.67,.60]},
        {'id':'LEFT_WALL','position':[-4,1,1.65],'size':[.15,6,3.3],'color':[.57,.59,.56]},
        {'id':'TEAL_DOOR','position':[1.8,3.90,1.1],'size':[1.1,.12,2.2],'color':[.035,.28,.26]},
        {'id':'DOOR_HANDLE','position':[1.45,3.8,1.0],'size':[.05,.1,.12],'color':[.65,.55,.2]},
        {'id':'BENCH','position':[-2.7,3.1,.45],'size':[1.8,.60,.18],'color':[.28,.14,.05]},
        {'id':'BENCH_LEG_A','position':[-3.3,3.1,.2],'size':[.12,.45,.4],'color':[.07,.07,.07]},
        {'id':'BENCH_LEG_B','position':[-2.1,3.1,.2],'size':[.12,.45,.4],'color':[.07,.07,.07]},
    ]}
    entities = {
        'ANA':{'kind':'character','position':[-.48,1,0],'yaw':0,'wardrobe_id':'ANA_RED',
               'color':[.55,.035,.025],'skin':[.70,.43,.28],'present':True},
        'PEDRO':{'kind':'character','position':[.48,1,0],'yaw':0,'wardrobe_id':'PEDRO_BLUE',
                 'color':[.025,.09,.46],'skin':[.51,.28,.15],'present':True},
        'BOOK_01':{'kind':'prop','position':[0,1,1],'color':[.92,.63,.025],'size':[.16,.10,.23],
                   'attachment':{'entity_id':'ANA','socket':'right_hand'},'present':True},
    }
    camera = dict(position=[0,-5.6,1.9],target=[0,1.25,1.15],lens_mm=38,sensor_mm=36,
                  width=768,height=512,near=.05,far=30)
    medium = dict(camera,position=[-1.5,-3.9,1.8],target=[0,1,1.25],lens_mm=48)
    transfer = dict(camera,position=[.7,-3.7,1.65],target=[0,1,1.2],lens_mm=45)
    appearance = ('Cinematic live-action photograph, two young adult colleagues in a quiet university corridor. '
        'Ana, a woman with dark hair and a red jacket and dark trousers, stands on the left. '
        'Pedro, a man with short dark hair and a blue jacket and dark trousers, stands on the right. '
        'A small yellow hardback book. Beige walls, one teal door in the back right, wooden bench in the back left. '
        'Natural faces and hands, realistic fabric, soft daylight. Preserve the exact layout, camera and positions of the 3D reference. ')
    return {'schema_version':1,'location_id':'CORRIDOR_A','location':location,'entities':entities,'fps':24,
        'shots':[
            {'id':'SH_001','camera':camera,'seconds':5,'action':'Ana holds the yellow book in her right hand. Pedro listens. Both remain in place.',
             'prompt':appearance+'Ana holds the yellow book near her waist. Wide two shot.', 'event':None},
            {'id':'SH_002','camera':medium,'seconds':5,'action':'Ana looks at Pedro and slightly nods while holding the yellow book. Fixed camera.',
             'prompt':appearance+'Ana still holds the yellow book. Medium two shot.', 'event':None},
            {'id':'SH_003','camera':transfer,'seconds':5,'action':'Ana gently passes the yellow book from her right hand into Pedro left hand. Pedro accepts it. Both remain in place. Fixed camera.',
             'prompt':appearance+'They are standing close together. The yellow book is between them at waist height.',
             'event':{'id':'EV_TRANSFER','source_unit_id':'UNIT_TRANSFER',
                      'operations':[{'op':'transfer','entity_id':'BOOK_01','from':'ANA','to':'PEDRO','socket':'left_hand'}]}},
            {'id':'SH_004','camera':camera,'seconds':5,'action':'Pedro holds the yellow book in his left hand. Ana has empty hands. They exchange a small smile. Static wide shot.',
             'prompt':appearance+'Pedro now holds the yellow book in his left hand. Ana has empty hands. Wide two shot.', 'event':None}
        ]}


def prepare(project, spec, *, qwen=False):
    from script_pipeline.production_project import create_project
    project = Path(project).resolve()
    run = project/'renders'/'spatial_run'
    if not (project/'projeto.json').exists():
        script = 'INT. UNIVERSITY CORRIDOR - DAY\n\n' + '\n\n'.join(s['action'] for s in spec['shots'])
        create_project(project,run,script,'Spatial continuity pilot',sum(s['seconds'] for s in spec['shots']))
    write(project/'world/spec.json',spec)
    frames_total = 0
    seen = set()
    shots = []
    with WorldStore(project) as store:
        default_location_id = spec.get('location_id', 'LOCATION_A')
        default_location = spec.get('location', {'boxes': []})
        entities = spec.get('entities', {})
        location_cast = spec.get('location_cast', {})
        version = store.register_asset(default_location_id, default_location)
        state = {'schema_version':1,'coordinates':'Z_UP_METERS','location_id':default_location_id,
                 'location_asset':version,'entities':entities}
        current = store.create(state)
        for index, shot in enumerate(spec['shots']):
            if shot['id'] in seen:
                raise ValueError('Duplicate shot ID')
            seen.add(shot['id'])
            initial = current
            location_id = shot.get('location_id', default_location_id)
            location = shot.get('location', default_location)
            if location_id != store.get(current)['location_id']:
                version = store.register_asset(location_id, location)
                moved = dict(store.get(current), location_id=location_id, location_asset=version)
                current = store.create(moved)
                initial = current
            # Visibility is a rendering view of the semantic state. It does not
            # mutate the continuing world, so a close-up cannot make everybody
            # disappear from later shots or move props between locations.
            from script_pipeline.spatial_planner import active_entities_for_shot, state_for_shot
            active = active_entities_for_shot(
                shot, entities,
                location_cast=(location_cast.get(location_id, []) if location_cast else None))
            initial = store.create(state_for_shot(store.get(current), active))
            event = shot.get('event')
            if event:
                if qwen:
                    from script_pipeline.scene_state_planner import propose_event
                    cached_path = project/'world'/(event['id']+'_qwen.json')
                    key = digest({'state':current,'action':shot['action'],'event':event})
                    cached = read(cached_path,{})
                    if cached.get('key') == key:
                        event = cached['event']
                    else:
                        proposed = propose_event(store.get(current),shot['action'],event['id'],event['source_unit_id'])
                        # The pilot has a known screenplay contract; no creative LLM changes permitted.
                        if proposed['operations'] != event['operations']:
                            raise ValueError(f'Qwen proposal differs from explicit pilot contract: {proposed}')
                        event = proposed
                        write(cached_path,{'key':key,'event':event})
                current = store.apply(current,event)
            final = store.create(state_for_shot(store.get(current), active))
            editorial = round(shot['seconds']*spec['fps'])
            generated,end_index = keyframe_indices(editorial)
            binding = store.bind(shot['id'],initial,final,shot['camera'],
                                 editorial_frames=editorial,frames=generated,end_index=end_index)
            shots.append(dict(shot,index=index,binding=binding,start_frame=frames_total,
                              references=shot.get('references',spec.get('references',[]))))
            frames_total += editorial
    write(project/'world/shot_bindings.json',{'fps':spec['fps'],'duration_frames':frames_total,'shots':shots})
    write(run/'production.json',{'project_dir':str(project)})
    return shots


def blocking(project):
    plan = read(project/'world/shot_bindings.json')
    with WorldStore(project) as store:
        for shot in plan['shots']:
            binding = shot['binding']
            shot['controls'] = {}
            for endpoint in ('initial','final'):
                print(f'[blocking] {shot["id"]} {endpoint}',flush=True)
                shot['controls'][endpoint] = compose(store,binding[endpoint],shot['camera'])
    write(project/'world/shot_bindings.json',plan)


def stills(project, *, denoise=.65):
    from script_pipeline import generate_storyboards as sb
    from script_pipeline.ollama_runtime import unload_models
    unload_models()
    import ltx25_backend
    ltx25_backend.ensure_server()
    plan = read(project/'world/shot_bindings.json')
    engine = sb.IMAGE_ENGINES['flux']
    directory = project/'renders'/'spatial_stills'
    directory.mkdir(parents=True,exist_ok=True)
    for shot in plan['shots']:
        shot['stills'] = {}
        for endpoint in ('initial','final'):
            bundle = shot['controls'][endpoint]
            state = json.loads((project/'world/snapshots'/f'{bundle["state_hash"]}.json').read_text(encoding='utf-8'))
            attachments = {k:v.get('attachment') for k,v in state['entities'].items() if v['kind']=='prop'}
            prompt = shot['prompt'] + ' Exact object ownership: ' + json.dumps(attachments) + '.'
            references = shot.get('references',[])
            if len(references) > 2:
                raise ValueError('Current Klein adapter accepts at most two identity references')
            recipe = {'bundle':bundle['fingerprint'],'prompt':prompt,'denoise':denoise,'engine':engine,
                      'references':{p:file_hash(p) for p in references},
                      'seed':8472,'adapter':file_hash(Path(__file__).with_name('spatial_conditioning.py'))}
            key = digest(recipe)
            dest = directory/f'{key}.png'
            meta = directory/f'{key}.json'
            old = read(meta,{})
            if not dest.exists() or old.get('image_hash') != file_hash(dest):
                print(f'[still] {shot["id"]} {endpoint}',flush=True)
                ok = sb.generate_scene_storyboard({'index':shot['index']},{},server='http://127.0.0.1:8188',
                    checkpoint=engine['checkpoint'],width=shot['camera']['width'],height=shot['camera']['height'],
                    steps=8,cfg=1,seed=8472,out_path=dest,clip=engine['clip'],vae=engine['vae'],
                    guidance=engine['guidance'],prompt_override=prompt,control_bundle=bundle,spatial_denoise=denoise,
                    reference_image=references[0] if references else None,
                    reference_image_2=references[1] if len(references)>1 else None)
                if not ok:
                    raise RuntimeError(f'Still generation failed: {shot["id"]}')
                write(meta,{'recipe':recipe,'image_hash':file_hash(dest),'approval':'pending'})
            shot['stills'][endpoint] = str(dest)
            write(project/'world/shot_bindings.json',plan)
    return plan


def video(project, *, allow_unapproved=False):
    from script_pipeline.spatial_audit import require_approval
    if not allow_unapproved:
        require_approval(project,'stills')
    import ltx25_backend
    from script_pipeline.ollama_runtime import unload_models
    unload_models()
    ltx25_backend.ensure_server()
    plan = read(project/'world/shot_bindings.json')
    directory = project/'renders'/'spatial_video'
    directory.mkdir(parents=True,exist_ok=True)
    for shot in plan['shots']:
        binding,camera = shot['binding'],shot['camera']
        recipe = {'binding':binding,'prompt':shot['action'],'stills':{k:file_hash(p) for k,p in shot['stills'].items()},
                  'variant':ltx25_backend.DEFAULT_VARIANT,'seed':8472+shot['index'],'fps':plan['fps'],
                  'backend':file_hash(ltx25_backend.__file__)}
        key = digest(recipe)
        raw,trimmed = directory/f'{key}_raw.mp4',directory/f'{key}.mp4'
        meta = directory/f'{key}.json'
        if not trimmed.exists() or read(meta,{}).get('video_hash') != file_hash(trimmed):
            print(f'[video] {shot["id"]}: {binding["frames"]} generated frames',flush=True)
            ltx25_backend.generate(shot['action']+' Natural small movements. Continuous shot, steady camera. No dialogue.',
                str(raw),width=camera['width'],height=camera['height'],num_frames=binding['frames'],
                frame_rate=plan['fps'],seed=8472+shot['index'],image_path=shot['stills']['initial'],
                keyframes=[(shot['stills']['final'],binding['end_index'],.8)],
                disable_audio=True,two_stage=False,log_cb=lambda m:print(m,flush=True),timeout=3600)
            subprocess.run([FFMPEG,'-y','-v','error','-i',str(raw),'-an','-frames:v',str(binding['editorial_frames']),
                            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(trimmed)],check=True)
            write(meta,{'recipe':recipe,'video_hash':file_hash(trimmed),'approval':'pending'})
        shot['video'] = str(trimmed)
        write(project/'world/shot_bindings.json',plan)
    concat = directory/'concat.txt'
    concat.write_text(''.join("file '"+s['video'].replace('\\','/').replace("'", "'\\''")+"'\n" for s in plan['shots']),encoding='utf-8')
    name = 'spatial_20s.mp4' if plan['duration_frames']/plan['fps'] == 20 else 'spatial_preview.mp4'
    final = project/'entregas'/name
    final.parent.mkdir(exist_ok=True)
    subprocess.run([FFMPEG,'-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(final)],check=True)
    probe = json.loads(subprocess.check_output([FFPROBE,'-v','error','-count_frames','-show_streams','-show_format',
                                               '-of','json',str(final)],text=True))
    frames = int(probe['streams'][0]['nb_read_frames'])
    if frames != plan['duration_frames']:
        raise RuntimeError(f'Final frame count mismatch: {frames} != {plan["duration_frames"]}')
    write(project/'entregas'/'verification.json',{'file':str(final),'sha256':file_hash(final),'frames':frames,
        'duration_seconds':float(probe['format']['duration']),'fps':plan['fps'],
        'visual_approval':'bypassed_for_diagnostic' if allow_unapproved else 'pending'})
    from script_pipeline.spatial_report import build_report
    build_report(project)
    print(f'[complete] {final}: {frames} frames',flush=True)


def _single_subject_prompt(prompt, subject):
    """Remove the planner's explicit partner clause from a spatial close-up.

    The continuity paragraph may continue to name the scene roster; that is
    useful context.  The visual clause ``with NAME, descriptor`` is different:
    it explicitly orders a second foreground person and forces FLUX to widen a
    requested close-up.  Shot planning emits that clause immediately before
    ``positioned ...``, which gives us a bounded, deterministic removal.
    """
    prefix = (f"Single-person close-up of {subject} only. No second person visible; "
              "preserve the canonical face and the fixed environment behind the subject. ")
    # O "with" removido e o MAIS PROXIMO antes de ". positioned": o `.*?` simples comecava no
    # primeiro "with" do prompt (ex.: "... with natural lighting.") e apagava tudo ate la.
    prompt = re.sub(r"\s+with\s+(?:(?!\swith\s).)*?\.\s+(?=positioned\b)", " ", prompt,
                    count=1, flags=re.IGNORECASE | re.DOTALL)
    return prompt if prompt.startswith(prefix) else prefix + prompt


def attach_to_run(run, spec_path, *, denoise=.65):
    """Strict opt-in bridge to existing decupagem plans; every stable shot ID must resolve."""
    from script_pipeline.production_project import project_for
    run = Path(run)
    project = project_for(run)
    if project is None:
        raise ValueError('Spatial mode requires a persistent production project')
    plan = read(run/'parse/shot_plan.json')
    spec = read(spec_path)
    expected = [s.get('id') for s in plan['shots']]
    given = [s['id'] for s in spec['shots']]
    if not all(expected) or len(set(expected)) != len(expected) or set(expected) != set(given):
        raise ValueError('Spatial spec must cover every unique stable shot ID in shot_plan.json')
    spec_by_id = {s['id']:s for s in spec['shots']}
    spec['shots'] = [dict(spec_by_id[s['id']],seconds=s['seconds']) for s in plan['shots']]
    spec['fps'] = plan['fps']
    prepare(project,spec)
    blocking(project)
    resolved = {s['id']:s for s in read(project/'world/shot_bindings.json')['shots']}
    for shot in plan['shots']:
        spatial = resolved[shot['id']]
        bundle = spatial['controls']['initial']
        tight = str(shot.get('framing', '')).casefold() in {'close', 'extreme_close'}
        shot['spatial'] = {'control_bundle':str(Path(bundle['files']['beauty']).with_name('control_bundle.json')),
                           'fingerprint':bundle['fingerprint'],'denoise':float(denoise),
                           'mode':'reference' if tight else 'img2img',
                           'adapter_version':2,
                           'binding':spatial['binding']}
        if tight and shot.get('subject'):
            shot['co_subject'] = ''
            shot['storyboard_prompt'] = _single_subject_prompt(
                shot['storyboard_prompt'], shot['subject'])
        # Changing-state endpoints must use the temporal runner until the standard stage consumes both stills.
        if spatial['binding']['initial'] != spatial['binding']['final']:
            raise ValueError('Temporal spatial events require spatial_pipeline; standard decupagem supports static spatial bindings')
    write(run/'parse/shot_plan.json',plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',required=True)
    ap.add_argument('--spec')
    ap.add_argument('--demo',action='store_true')
    ap.add_argument('--qwen',action='store_true')
    ap.add_argument('--stage',choices=['prepare','blocking','stills','audit-stills','video','audit-video','all'],default='all')
    ap.add_argument('--denoise',type=float,default=.65)
    ap.add_argument('--allow-unapproved',action='store_true',
                    help='gera um vídeo diagnóstico mesmo com still gate bloqueado; nunca aprova a produção')
    args = ap.parse_args()
    project = Path(args.project).resolve()
    try:
        if args.stage in ('prepare','all'):
            spec = demo_spec() if args.demo else read(args.spec) if args.spec else read(project/'world/spec.json')
            if not spec:
                ap.error('Provide --demo or --spec for initial preparation')
            prepare(project,spec,qwen=args.qwen)
        if args.stage in ('blocking','all'):
            blocking(project)
        if args.stage in ('stills','all'):
            stills(project,denoise=args.denoise)
        if args.stage in ('audit-stills','all'):
            from script_pipeline.spatial_audit import audit
            if audit(project,'stills')['status'] != 'approved':
                raise RuntimeError('Spatial still gate blocked production; inspect world/audit_stills.json')
        if args.stage in ('video','all'):
            video(project,allow_unapproved=args.allow_unapproved)
        if args.stage in ('audit-video','all'):
            from script_pipeline.spatial_audit import audit
            report = audit(project,'video')
            verification = read(project/'entregas/verification.json',{})
            verification['visual_approval'] = report['status']
            write(project/'entregas/verification.json',verification)
            if report['status'] != 'approved':
                raise RuntimeError('Spatial video gate rejected the test; output retained for diagnosis')
    finally:
        from script_pipeline.ollama_runtime import unload_models
        unload_models()


if __name__ == '__main__':
    main()
