"""Explicit capability: FLUX.2 Klein RGB img2img + independent identity references.

This is not depth ControlNet. Depth/IDs remain available for auditing and future adapters.
"""
from pathlib import Path


def validate_bundle(bundle):
    from script_pipeline.scene_composer import file_hash
    if bundle.get('schema_version') != 1 or not bundle.get('state_hash'):
        raise ValueError('Invalid control bundle')
    for key, path in bundle['files'].items():
        if not Path(path).is_file() or file_hash(path) != bundle['hashes'].get(key):
            raise ValueError(f'Missing or stale spatial input: {key}')
    return bundle['files']['beauty']


def wire_spatial(workflow, bundle, *, architecture, stage_image, denoise=.65,
                 mode='img2img'):
    if architecture != 'flux':
        raise ValueError('Spatial RGB adapter currently supports FLUX.2 Klein only')
    if not 0 < denoise <= 1:
        raise ValueError('Spatial denoise must be in (0,1]')
    if mode not in {'img2img', 'reference'}:
        raise ValueError("Spatial mode must be 'img2img' or 'reference'")
    beauty = validate_bundle(bundle)
    samplers = [n for n in workflow.values() if n['class_type'] == 'KSampler']
    if len(samplers) != 1:
        raise ValueError('Spatial adapter requires exactly one KSampler')
    sampler = samplers[0]['inputs']
    latent = workflow[sampler['latent_image'][0]]['inputs']
    width, height = latent['width'], latent['height']
    camera = bundle['camera']
    if abs(width/height-camera['width']/camera['height']) > .001:
        raise ValueError('Spatial camera and output aspect ratios differ')
    vae = next(n['inputs']['vae'] for n in workflow.values() if n['class_type'] == 'VAEDecode')
    workflow['spatial_image'] = {'class_type':'LoadImage','inputs':{'image':stage_image(beauty)}}
    workflow['spatial_resize'] = {'class_type':'ImageScale','inputs':{'image':['spatial_image',0],
        'upscale_method':'lanczos','width':width,'height':height,'crop':'disabled'}}
    workflow['spatial_encode'] = {'class_type':'VAEEncode','inputs':{'pixels':['spatial_resize',0],'vae':vae}}
    positive = sampler['positive']
    workflow['spatial_ref'] = {'class_type':'ReferenceLatent','inputs':{
        'conditioning':positive,'latent':['spatial_encode',0]}}
    # Wide/environment shots need the blocking image as their starting canvas so
    # walls, furniture and screen direction remain fixed.  In a close-up that
    # same canvas contains a deliberately crude proxy head; using it as img2img
    # overwhelms the real character reference.  Reference-only mode still feeds
    # the camera/environment latent into conditioning, while the normal empty
    # latent and the character ReferenceLatent remain responsible for the face.
    if mode == 'img2img':
        sampler['positive'] = ['spatial_ref',0]
        sampler['latent_image'] = ['spatial_encode',0]
        sampler['denoise'] = denoise
    else:
        # ReferenceLatent is an ordered chain.  Character templates already
        # put one or two identity references on ``positive``.  Appending the
        # environment after them made the room/proxy dominate the face and
        # even widened close-ups.  Insert the spatial latent at the base of
        # that chain so character identity remains the last, strongest cue.
        cursor = positive
        innermost = None
        seen = set()
        while isinstance(cursor, list) and len(cursor) == 2 and cursor[0] in workflow:
            node_id = cursor[0]
            if node_id in seen:
                raise ValueError('Cycle in ReferenceLatent conditioning chain')
            seen.add(node_id)
            node = workflow[node_id]
            if node.get('class_type') != 'ReferenceLatent':
                break
            innermost = node
            cursor = node['inputs']['conditioning']
        if innermost is None:
            sampler['positive'] = ['spatial_ref',0]
        else:
            workflow['spatial_ref']['inputs']['conditioning'] = cursor
            innermost['inputs']['conditioning'] = ['spatial_ref',0]
