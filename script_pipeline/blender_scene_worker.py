"""Run only in Blender: --background --factory-startup --python FILE -- JOB.json."""
import json
import math
import sys
from pathlib import Path


def main():
    import bpy
    from mathutils import Vector
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from script_pipeline.camera_geometry import socket_position
    job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.use_compositing = False
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 12
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 12
    scene.world.color = (.22, .22, .22)
    masks = {}

    def material(name, color):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = .65
        return mat

    def finish(obj, name, mat, index):
        obj.name = name
        obj.data.materials.append(mat)
        obj.pass_index = index
        return obj

    def box(name, position, size, mat, index):
        bpy.ops.mesh.primitive_cube_add(size=1, location=position)
        obj = bpy.context.object
        obj.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = obj.modifiers.new('Soft edges', 'BEVEL')
        bevel.width, bevel.segments = .025, 2
        return finish(obj, name, mat, index)

    def sphere(name, position, scale, mat, index):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=position)
        obj = bpy.context.object
        obj.scale = scale
        for poly in obj.data.polygons:
            poly.use_smooth = True
        return finish(obj, name, mat, index)

    def limb(name, a, b, radius, mat, index):
        direction = Vector(b) - Vector(a)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=direction.length,
                                           location=(Vector(a) + Vector(b)) / 2)
        obj = bpy.context.object
        obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
        return finish(obj, name, mat, index)

    for i, item in enumerate(job['location']['boxes'], 1):
        box(item['id'], item['position'], item['size'], material(item['id'], item['color']), i)
        masks[str(i)] = item['id']

    state = job['state']
    landmarks = {}
    skeletons = {}
    for index, (eid, entity) in enumerate(sorted(state['entities'].items()), 100):
        if not entity.get('present', True):
            continue
        masks[str(index)] = eid
        mat = material(eid, entity['color'])
        if entity['kind'] == 'prop':
            attachment = entity.get('attachment')
            pos = socket_position(state['entities'][attachment['entity_id']], attachment['socket']) if attachment else entity['position']
            box(eid, pos, entity.get('size', [.18, .14, .24]), mat, index)
            landmarks[eid] = list(pos)
            continue
        x, y, z = entity['position']
        angle = math.radians(entity.get('yaw', 0))
        def point(dx, dy, dz):
            return [x + math.cos(angle)*dx-math.sin(angle)*dy,
                    y + math.sin(angle)*dx+math.cos(angle)*dy, z+dz]
        # COCO-18 order, in world coordinates; same dimensions used to build the mesh.
        skeletons[eid] = [point(*p) for p in [
            (0,-.12,1.68),(0,0,1.45),(.23,0,1.4),(.31,0,1.15),(.30,-.08,1),
            (-.23,0,1.4),(-.31,0,1.15),(-.30,-.08,1),(.12,0,.95),(.13,0,.53),(.13,0,.13),
            (-.12,0,.95),(-.13,0,.53),(-.13,0,.13),(.046,-.113,1.71),(-.046,-.113,1.71),
            (.12,0,1.70),(-.12,0,1.70)]]
        skin = material(eid + '_skin', entity.get('skin', [.65, .39, .24]))
        pants = material(eid + '_pants', [.035, .045, .06])
        hair = material(eid + '_hair', [.035, .018, .012])
        torso = sphere(eid+'_torso', point(0, 0, 1.18), (.24, .16, .36), mat, index)
        torso.rotation_euler.z = angle
        sphere(eid+'_head', point(0, 0, 1.68), (.135, .12, .18), skin, index)
        sphere(eid+'_hair', point(0, .025, 1.77), (.14, .12, .105), hair, index)
        # Facing local -Y, two visible eyes and a nose give the renderer a facial orientation.
        for side in (-1, 1):
            sphere(eid+'_eye', point(side*.046, -.113, 1.71), (.012,.01,.014), hair,index)
            limb(eid+'_leg', point(side*.12,0,.95), point(side*.13,0,.13), .09,pants,index)
            sphere(eid+'_shoe', point(side*.13,-.075,.09), (.10,.19,.08),pants,index)
            limb(eid+'_upperarm', point(side*.23,0,1.40), point(side*.31,0,1.15),.075,mat,index)
            limb(eid+'_forearm', point(side*.31,0,1.15), point(side*.30,-.08,1.00),.06,mat,index)
            sphere(eid+'_hand', point(side*.30,-.08,1.00), (.065,.06,.085),skin,index)
        sphere(eid+'_nose',point(0,-.124,1.67),(.025,.045,.035),skin,index)
        landmarks[eid] = point(0,0,1.68)
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
