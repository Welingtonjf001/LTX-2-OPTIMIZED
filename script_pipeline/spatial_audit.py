"""Pixel gate for spatial projects. Perception is independent of desired scene text."""
import json
import os
import subprocess
from pathlib import Path

from script_pipeline.production_project import read, write
from script_pipeline.scene_composer import file_hash
from script_pipeline.world_store import digest


def targets_for(shot, stage):
    return list(dict.fromkeys(shot['stills'].values())) if stage == 'stills' else [shot['video']]


def target_hash(shot, stage):
    return digest({'binding':shot['binding'],'contract':shot.get('prompt'),
                   'auditor':file_hash(__file__),'targets':{p:file_hash(p) for p in targets_for(shot,stage)}})


def require_approval(project, stage):
    plan = read(project/'world/shot_bindings.json')
    report = read(project/'world'/f'audit_{stage}.json',{})
    entries = {r['shot_id']:r for r in report.get('shots',[])}
    for shot in plan['shots']:
        entry = entries.get(shot['id'],{})
        if entry.get('approved') is not True or entry.get('target_hash') != target_hash(shot,stage):
            raise ValueError(f'{stage}: missing, rejected or stale approval for {shot["id"]}')


def _nominal_references(project, shot):
    """Fotos nominais do sujeito do plano (se o spec as fornece e existem em disco)."""
    refs = [Path(str(x)) for x in (shot.get('references') or [])
            if str(x).lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and Path(str(x)).is_file()]
    return refs[:1]


def deterministic_locks(shot, stage, images, frames, project):
    """Travas que o LLM de decisao nao pode contornar (mesma politica do gate principal):
    texto inventado (legenda/marca d'agua) reprova; em close, o rosto tem que parecer com a
    foto nominal (ArcFace, so aperta). Devolve (bloqueios, avisos)."""
    from script_pipeline.visual_continuity_audit import face_check, _entry_warnings, FACE_IDENTITY_THRESHOLD
    locks = []
    for i, f in enumerate(frames):
        kind = str(f.get('text_kind', 'none')).casefold()
        if kind.startswith('caption') or (kind == 'other' and str(f.get('visible_text', '')).strip()
                                          and str(f.get('visible_text', '')).strip().casefold() != 'none'):
            locks.append(f'frame {i}: texto artificial ({kind}: {str(f.get("visible_text", ""))[:40]})')
    face = None
    if str(shot.get('framing', '')).casefold() in ('close', 'extreme_close'):
        refs = _nominal_references(project, shot)
        if refs:
            face = face_check([Path(p) for p in images], refs)
            sim = (face or {}).get('min_similarity')
            if sim is not None and sim < FACE_IDENTITY_THRESHOLD:
                locks.append(f'similaridade facial {sim} < {FACE_IDENTITY_THRESHOLD} com a foto nominal')
    return locks, _entry_warnings([Path(p) for p in images], face)


def audit(project, stage, *, model='qwen3-vl:30b'):
    from script_pipeline.visual_continuity_audit import _ollama_json
    from script_pipeline.ollama_runtime import unload_models
    from script_pipeline.generate_storyboards import stop_comfyui, comfy_is_up
    if comfy_is_up('http://127.0.0.1:8188'):
        stop_comfyui()
    plan = read(project/'world/shot_bindings.json')
    report_path = project/'world'/f'audit_{stage}.json'
    old_report = read(report_path,{})
    previous = {s['shot_id']:s for s in old_report.get('shots',[])} if old_report.get('model') == model else {}
    report = {'stage':stage,'model':model,'shots':[],'status':'partial'}
    directory = project/'world'/f'audit_frames_{stage}'
    directory.mkdir(parents=True,exist_ok=True)
    try:
        for shot in plan['shots']:
            key = target_hash(shot,stage)
            if previous.get(shot['id'],{}).get('target_hash') == key:
                report['shots'].append(previous[shot['id']])
                continue
            images = [Path(p) for p in targets_for(shot,stage)]
            if stage == 'video':
                images = []
                duration = shot['binding']['editorial_frames']/plan['fps']
                for i,t in enumerate([0,duration*.25,duration*.5,duration*.75,max(0,duration-.1)]):
                    path = directory/f'{shot["id"]}_{i}.png'
                    subprocess.run([os.environ.get('LTX_FFMPEG','C:/ffmpeg/bin/ffmpeg.exe'),'-y','-v','error','-ss',str(t),'-i',shot['video'],
                                    '-frames:v','1',str(path)],check=True)
                    images.append(path)
            print(f'[audit {stage}] {shot["id"]}',flush=True)
            perception = _ollama_json(model,
                'Describe only the attached images, one entry per image, in order. Do not infer invisible objects. '
                'Return JSON {"frames":[{"people_count":integer,"people":[{"clothing":"...","screen_position":"...",'
                '"appearance":"...","held_objects":"..."}],"room":"...","key_objects":["..."],'
                '"visible_text":"...","text_kind":"none|diegetic_sign_or_instrument|caption_or_watermark|other",'
                '"defects":"..."}]}. Describe colors and any object changing hands. Keep lists short. '
                'Do not assume continuity between images. There are '+str(len(images))+' images.',images)
            frames = perception.get('frames',[])
            if len(frames) != len(images):
                raise ValueError('Auditor did not perceive every target')
            states = {}
            for endpoint in ('initial','final'):
                states[endpoint] = read(project/'world/snapshots'/f'{shot["binding"][endpoint]}.json')['entities']
            from PIL import Image
            import numpy as np
            expected_counts = []
            for endpoint in ('initial','final'):
                bundle = shot['controls'][endpoint]
                visible_ids = set(np.unique(np.asarray(Image.open(bundle['files']['instances']))).tolist())
                expected_counts.append(sum(1 for index,eid in bundle['mask_entities'].items()
                    if int(index) in visible_ids and states[endpoint].get(eid,{}).get('kind') in ('character','extra')))
            decision = _ollama_json(model,
                'Judge observed evidence against this scene contract. The observations, not expectations, are facts. '
                'Return JSON {"approved":boolean,"reasons":[strings],"identity_assessment":"...","spatial_assessment":"..."}. '
                'Require the people and garment colors described by the shot contract, '
                'the same location across frames, and no artificial caption. Exact limb pose and face identity are not '
                'proven by outfit; record limitations. A cropped background element is permitted. A held-object transfer '
                'is required only for an initial/final owner change; if not visible, reject it as unverified. '
                'A held object must not clearly belong to the wrong person. State the actual observed garment colors and object holder. '
                +json.dumps({'stage':stage,'observations':perception,'expected_states':states,
                             'shot_contract':shot['prompt'],'visible_people_endpoint_counts':expected_counts}),[])
            count_ok = True
            if expected_counts[0] == expected_counts[1]:
                count_ok = all(f.get('people_count') == expected_counts[0] for f in frames)
            approved = decision.get('approved') is True and count_ok
            locks, warnings = deterministic_locks(shot, stage, images, frames, project)
            if locks:
                approved = False
            report['shots'].append({'shot_id':shot['id'],'target_hash':key,'approved':approved,
                                    'locks':locks,'warnings':warnings,
                                    'perception':perception,'decision':decision})
            write(report_path,report)
        report['status'] = 'approved' if all(x['approved'] for x in report['shots']) else 'blocked'
        write(report_path,report)
        return report
    finally:
        unload_models([model])
