"""Blender CLI bridge: reusable cabin blocking, repeatable cameras and multilayer passes.

Run with Blender --background --python this_file -- --project PATH [--build-cabin].
For existing scenes, pass a .blend before --python and a --job JSON camera definition.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def main():
    import bpy
    from mathutils import Vector
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--build-cabin', action='store_true')
    ap.add_argument('--job')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    root = Path(args.project)
    out = root / 'assets/locacoes/LOC_CABIN_3D'
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if args.build_cabin:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        def material(name, color):
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*color, 1)
            return mat
        shell = material('Cabin warm ivory', (.7, .72, .73))
        blue = material('Seats navy blue', (.04, .1, .19))
        gray = material('Aisle carpet', (.12, .14, .16))
        def box(name, location, size, mat, mask=1):
            bpy.ops.mesh.primitive_cube_add(size=1, location=location)
            obj = bpy.context.object
            obj.name, obj.dimensions = name, size
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            obj.data.materials.append(mat)
            obj.pass_index = mask
            return obj
        box('Floor', (0, 12.5, -.06), (4.7, 29, .12), gray)
        for side in (-1, 1):
            box(f'Wall {side}', (side * 2.35, 12.5, 1.4), (.12, 29, 2.8), shell)
            for row in range(1, 31):
                y = (row - 1) * .82
                box(f'Overhead bin {side} row {row}', (side * 1.7, y, 2.25), (1.0, .78, .5), shell, 3)
                for seat in range(3):
                    x = side * (.75 + seat * .5)
                    box(f'Seat {side} {row} {seat}', (x, y, .55), (.45, .5, .16), blue, 2)
                    box(f'Seatback {side} {row} {seat}', (x, y + .25, 1.05), (.45, .12, 1.0), blue, 2)
        for y in (-1.6, 25.8):
            for x in (-1.3, 1.3):
                box('Galley secured cart', (x, y, .55), (.6, .6, 1.1), shell, 4)
        for y in range(0, 25, 4):
            bpy.ops.object.light_add(type='AREA', location=(0, y, 2.65))
            bpy.context.object.data.energy = 120
            bpy.context.object.data.shape = 'RECTANGLE'
            bpy.context.object.data.size = 3
        bpy.ops.object.camera_add(location=(0, -2.7, 1.65))
        scene.camera = bpy.context.object
        direction = Vector((0, 14, 1.5)) - scene.camera.location
        scene.camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        scene.camera.data.lens = 22
    if args.job:
        job = json.loads(Path(args.job).read_text(encoding='utf-8'))
        camera = job.get('camera', {})
        if 'position' in camera:
            scene.camera.location = camera['position']
        if 'target' in camera:
            scene.camera.rotation_euler = (Vector(camera['target']) - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.camera.data.lens = camera.get('lens_mm', scene.camera.data.lens)
        scene.frame_set(job.get('frame', 1))
        out = root / 'vfx' / job['shot_id']
        out.mkdir(parents=True, exist_ok=True)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.render.resolution_x, scene.render.resolution_y = 960, 544
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.filepath = str(out / 'passes.exr')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    bpy.ops.render.render(write_still=True)
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'PNG'
    bpy.data.images['Render Result'].save_render(str(out / 'beauty.png'), scene=scene)
    (out / 'manifest.json').write_text(json.dumps({'scene_file': str(out / 'scene.blend'),
        'beauty': str(out / 'beauty.png'), 'depth_and_object_masks': str(out / 'passes.exr'),
        'purpose': 'Spatial blocking reference; not a photoreal final render.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
