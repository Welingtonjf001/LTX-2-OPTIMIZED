"""Run only in Blender: --background --factory-startup --python FILE -- JOB.json.

Two job kinds share the same scene builder:
- still (default): Cycles beauty + multilayer EXR passes + scene.blend, consumed by scene_composer;
- sequence (job['sequence'] = list of states): low-resolution Workbench frames of the moving
  mannequins, consumed by previs_spec (3D previs of the shot's motion).
The mannequin is built from its COCO-18 skeleton (mannequin_poses), so a pose change moves the
mesh, the exported skeleton and the hand sockets together.
"""
import json
import math
import sys
from pathlib import Path


def main():
    import bpy
    from mathutils import Matrix, Vector
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from script_pipeline.camera_geometry import socket_position
    from script_pipeline import mannequin_poses as mp
    job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    sequence = job.get('sequence')
    if sequence:
        # Workbench: segundos para dezenas de quadros; cor chapada do material basta para ler
        # posicao, direcao e contato no previs.
        scene.render.engine = 'BLENDER_WORKBENCH'
        scene.display.shading.light = 'STUDIO'
        scene.display.shading.color_type = 'MATERIAL'
        scene.display.shading.show_shadows = True
    else:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 12
    scene.render.use_compositing = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 12
    scene.world.color = (.22, .22, .22)
    materials = {}

    def material(name, color):
        key = (name, tuple(round(c, 4) for c in color))
        if key in materials:
            return materials[key]
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = .65
        materials[key] = mat
        return mat

    def finish(obj, name, mat, index, created):
        obj.name = name
        obj.data.materials.append(mat)
        obj.pass_index = index
        created.append(obj)
        return obj

    def box(name, position, size, mat, index, created):
        bpy.ops.mesh.primitive_cube_add(size=1, location=position)
        obj = bpy.context.object
        obj.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = obj.modifiers.new('Soft edges', 'BEVEL')
        bevel.width, bevel.segments = .025, 2
        return finish(obj, name, mat, index, created)

    def ellipsoid(name, center, axes, radii, mat, index, created):
        """Esfera escalada nos eixos (x, y, z) dados -- cabeca, torso, mao e pe seguem a pose."""
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=center)
        obj = bpy.context.object
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Matrix((axes[0], axes[1], axes[2])).transposed().to_quaternion()
        obj.scale = radii
        for poly in obj.data.polygons:
            poly.use_smooth = True
        return finish(obj, name, mat, index, created)

    def limb(name, a, b, radius, mat, index, created):
        direction = Vector(b) - Vector(a)
        if direction.length < 1e-4:
            return None
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=direction.length,
                                            location=(Vector(a) + Vector(b)) / 2)
        obj = bpy.context.object
        obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
        return finish(obj, name, mat, index, created)

    def frame_axes(right, up):
        """Eixos ortonormais (x = esquerda anatomica, y = costas, z = cima) a partir da
        direita anatomica e de um 'cima' aproximado. O rosto fica em -y, como no referencial
        local do personagem."""
        z = Vector(up).normalized()
        x = (-Vector(right))
        x = (x - z * x.dot(z)).normalized()
        y = z.cross(x)
        return x, y, z

    def mannequin(eid, entity, index, created):
        mat = material(eid, entity['color'])
        joints = [Vector(p) for p in mp.world_joints(entity)]
        skin = material(eid + '_skin', entity.get('skin', [.65, .39, .24]))
        pants = material(eid + '_pants', [.035, .045, .06])
        hair = material(eid + '_hair', [.035, .018, .012])
        mid_hip = (joints[mp.R_HIP] + joints[mp.L_HIP]) / 2
        neck = joints[mp.NECK]
        body_right = joints[mp.R_SHOULDER] - joints[mp.L_SHOULDER]
        bx, by, bz = frame_axes(body_right, neck - mid_hip)
        ellipsoid(eid + '_torso', (neck + mid_hip) / 2 - bz * .02, (bx, by, bz), (.24, .16, .36),
                  mat, index, created)
        head = (joints[mp.R_EAR] + joints[mp.L_EAR]) / 2
        hx, hy, hz = frame_axes(joints[mp.R_EAR] - joints[mp.L_EAR], head - neck)
        ellipsoid(eid + '_head', head - hz * .02, (hx, hy, hz), (.135, .12, .18), skin, index, created)
        ellipsoid(eid + '_hair', head + hy * .025 + hz * .07, (hx, hy, hz), (.14, .12, .105),
                  hair, index, created)
        ellipsoid(eid + '_nose', joints[mp.NOSE], (hx, hy, hz), (.025, .045, .035), skin, index, created)
        for eye in (mp.R_EYE, mp.L_EYE):
            ellipsoid(eid + '_eye', joints[eye], (hx, hy, hz), (.012, .01, .014), hair, index, created)
        limb(eid + '_neck', neck, head - hz * .12, .055, skin, index, created)
        for hip, knee, ankle in ((mp.R_HIP, mp.R_KNEE, mp.R_ANKLE), (mp.L_HIP, mp.L_KNEE, mp.L_ANKLE)):
            limb(eid + '_thigh', joints[hip], joints[knee], .09, pants, index, created)
            limb(eid + '_shin', joints[knee], joints[ankle], .075, pants, index, created)
            ellipsoid(eid + '_knee', joints[knee], (bx, by, bz), (.085, .085, .085), pants, index, created)
            # Pe aponta para o rosto do corpo (-y do torso), achatado no chao.
            forward = -by
            ellipsoid(eid + '_shoe', joints[ankle] + forward * .075 - bz * .04,
                      (bx, by, bz), (.10, .19, .08), pants, index, created)
        for shoulder, elbow, wrist in ((mp.R_SHOULDER, mp.R_ELBOW, mp.R_WRIST),
                                       (mp.L_SHOULDER, mp.L_ELBOW, mp.L_WRIST)):
            limb(eid + '_upperarm', joints[shoulder], joints[elbow], .075, mat, index, created)
            limb(eid + '_forearm', joints[elbow], joints[wrist], .06, mat, index, created)
            ellipsoid(eid + '_hand', joints[wrist], (bx, by, bz), (.065, .06, .085), skin, index, created)
        return [list(p) for p in joints], list(head - hz * .02)

    def build_state(state, created, start_index=100):
        masks, landmarks, skeletons = {}, {}, {}
        for index, (eid, entity) in enumerate(sorted(state['entities'].items()), start_index):
            if not entity.get('present', True):
                continue
            masks[str(index)] = eid
            if entity['kind'] == 'prop':
                mat = material(eid, entity['color'])
                attachment = entity.get('attachment')
                pos = list(socket_position(state['entities'][attachment['entity_id']], attachment['socket'])) \
                    if attachment else entity['position']
                box(eid, pos, entity.get('size', [.18, .14, .24]), mat, index, created)
                landmarks[eid] = list(pos)
                continue
            skeletons[eid], landmarks[eid] = mannequin(eid, entity, index, created)
        return masks, landmarks, skeletons

    static = []
    masks = {}
    for i, item in enumerate(job['location']['boxes'], 1):
        box(item['id'], item['position'], item['size'], material(item['id'], item['color']), i, static)
        masks[str(i)] = item['id']
    for pos, energy, size in [((0,-2,5),900,5), ((-3,2,4),650,4), ((3,3,4),650,4)]:
        bpy.ops.object.light_add(type='AREA', location=pos)
        light = bpy.context.object
        light.data.energy, light.data.size = energy,size
        light.rotation_euler = (Vector((0,1,1))-light.location).to_track_quat('-Z','Y').to_euler()
    camera = job['camera']
    bpy.ops.object.camera_add(location=camera['position'])
    scene.camera = bpy.context.object
    scene.camera.rotation_euler = (Vector(camera['target'])-scene.camera.location).to_track_quat('-Z','Y').to_euler()
    scene.camera.data.lens, scene.camera.data.sensor_width = camera['lens_mm'], camera['sensor_mm']
    scene.camera.data.sensor_fit = 'HORIZONTAL'
    scene.camera.data.clip_start, scene.camera.data.clip_end = camera.get('near', .05), camera.get('far',100)
    scene.render.resolution_x, scene.render.resolution_y = camera['width'],camera['height']
    scene.render.resolution_percentage = 100

    if sequence:
        if hasattr(scene.render.image_settings, 'media_type'):
            scene.render.image_settings.media_type = 'IMAGE'
        scene.render.image_settings.file_format = 'PNG'
        skeleton_frames = []
        for frame, state in enumerate(sequence):
            created = []
            _, _, skeletons = build_state(state, created)
            skeleton_frames.append(skeletons)
            scene.render.filepath = str(out / f'frame_{frame:04d}.png')
            bpy.ops.render.render(write_still=True)
            for obj in created:
                bpy.data.objects.remove(obj, do_unlink=True)
        (out/'sequence_metadata.json').write_text(json.dumps({'frames':len(sequence),'skeletons':skeleton_frames,
            'blender_version':bpy.app.version_string},indent=2),encoding='utf-8')
        return

    state = job['state']
    entity_masks, landmarks, skeletons = build_state(state, [])
    masks.update(entity_masks)
    layer = scene.view_layers[0]
    layer.use_pass_z = layer.use_pass_object_index = layer.use_pass_normal = True
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.filepath = str(out/'passes.exr')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    bpy.ops.render.render(write_still=True)
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'PNG'
    bpy.data.images['Render Result'].save_render(str(out/'beauty.png'), scene=scene)
    from bpy_extras.object_utils import world_to_camera_view
    projection = {}
    for eid, pos in landmarks.items():
        p = world_to_camera_view(scene, scene.camera, Vector(pos))
        projection[eid] = {'world':pos,'pixel':[p.x*camera['width'],(1-p.y)*camera['height']], 'depth':p.z}
    (out/'blender_metadata.json').write_text(json.dumps({'mask_entities':masks,'landmarks':projection,'skeletons':skeletons,
        'blender_version':bpy.app.version_string},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
