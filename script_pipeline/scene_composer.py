"""Content-addressed Blender blocking with auditable depth and instance masks."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from script_pipeline.world_store import digest
from script_pipeline.camera_geometry import project, validate_camera


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compose(store, state_hash, camera, *, blender=None):
    validate_camera(camera)
    state = store.get(state_hash)
    location = store.asset(state['location_id'], state['location_asset'])
    worker = Path(__file__).with_name('blender_scene_worker.py')
    key = digest({'state':state_hash,'camera':camera,'worker':file_hash(worker),
                  'composer':file_hash(__file__), 'geometry':file_hash(Path(__file__).with_name('camera_geometry.py'))})
    out = store.root.parent / 'vfx' / 'blocking' / key
    manifest = out/'control_bundle.json'
    if manifest.exists():
        cached = json.loads(manifest.read_text(encoding='utf-8'))
        if all(Path(p).exists() and file_hash(p) == cached['hashes'][name] for name,p in cached['files'].items()):
            return cached
    out.mkdir(parents=True, exist_ok=True)
    job = {'output':str(out.resolve()),'state':state,'camera':camera,'location':location}
    job_path = out/'job.json'
    job_path.write_text(json.dumps(job),encoding='utf-8')
    executable = blender or shutil.which('blender')
    if not executable:
        raise RuntimeError('Blender executable not found')
    with (out/'blender.log').open('w',encoding='utf-8') as log:
        subprocess.run([executable,'--background','--factory-startup','--python',str(worker),'--',str(job_path)],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    import OpenEXR
    with OpenEXR.File(str(out/'passes.exr'), separate_channels=True) as exr:
        # Blender 5 writes multipart EXR; channels() alone reads only beauty (part 0).
        channels = {k:v for i in range(len(exr.parts)) for k,v in exr.channels(i).items()}
        def channel(suffix):
            matches = [v.pixels for k,v in channels.items() if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected EXR channel {suffix}; got {list(channels)}')
            return matches[0]
        depth = channel('.Depth.Z').copy()
        mask_suffix = '.Object Index.X' if any(k.endswith('.Object Index.X') for k in channels) else '.IndexOB.X'
        masks = np.rint(channel(mask_suffix)).astype(np.uint16)
        normals = np.stack([channel('.Normal.'+axis) for axis in ('X','Y','Z')],axis=-1)
    np.save(out/'depth_m.npy', depth)
    np.save(out/'normal_world.npy', normals)
    Image.fromarray(masks).save(out/'instance_ids.png')
    near, far = camera.get('near',.05), camera.get('far',100)
    preview = np.where(np.isfinite(depth), 1-np.clip((depth-near)/(far-near),0,1),0)
    Image.fromarray((preview*65535).astype(np.uint16)).save(out/'depth_preview.png')
    metadata = json.loads((out/'blender_metadata.json').read_text(encoding='utf-8'))
    import cv2
    from script_pipeline.pose_video import COCO18_COLORS, COCO18_EDGES
    pose = np.zeros((camera['height'],camera['width'],3),dtype=np.uint8)
    for joints in metadata.get('skeletons',{}).values():
        xy, _, visible = project(joints,camera)
        for i,(a,b) in enumerate(COCO18_EDGES):
            if visible[a] and visible[b]:
                color = tuple(int(c) for c in COCO18_COLORS[i][::-1])
                cv2.line(pose,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,3,cv2.LINE_AA)
        for i,p in enumerate(xy):
            if visible[i]:
                cv2.circle(pose,tuple(np.rint(p).astype(int)),3,tuple(int(c) for c in COCO18_COLORS[i][::-1]),-1)
    Image.fromarray(cv2.cvtColor(pose,cv2.COLOR_BGR2RGB)).save(out/'pose.png')
    errors = []
    for landmark in metadata['landmarks'].values():
        xy, _, _ = project([landmark['world']],camera)
        if np.isfinite(xy).all():
            errors.append(float(np.linalg.norm(xy[0]-landmark['pixel'])))
    if errors and max(errors) > .1:
        raise ValueError(f'Blender/camera projection mismatch: {max(errors)} pixels')
    files = {n:str((out/f).resolve()) for n,f in {
        'beauty':'beauty.png','depth':'depth_m.npy','depth_preview':'depth_preview.png',
        'normal':'normal_world.npy','instances':'instance_ids.png','scene':'scene.blend','pose':'pose.png'}.items()}
    bundle = {'schema_version':1,'state_hash':state_hash,'camera':camera,'fingerprint':key,'files':files,
              'hashes':{n:file_hash(p) for n,p in files.items()},'mask_entities':metadata['mask_entities'],
              'depth_convention':'Blender Z pass, camera ray distance in metres; background may be infinite',
              'normal_space':'world','max_projection_error_px':max(errors,default=0)}
    bundle['pose_occlusion_policy'] = 'Frustum clipped; hidden joints are retained, not a visibility-segmented pose'
    manifest.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    return bundle
