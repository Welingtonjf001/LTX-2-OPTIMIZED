"""Build the opt-in spatial spec for the Voo 702 emergency pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

from script_pipeline.production_project import read, write


def _location(scene):
    if scene == 1:
        return 'LOC_COCKPIT', [
            {'id':'COCKPIT_FLOOR','position':[0,2,-.08],'size':[4.6,6.5,.16],'color':[.15,.17,.19]},
            {'id':'COCKPIT_CEILING','position':[0,2,2.75],'size':[4.4,6.2,.12],'color':[.58,.58,.54]},
            {'id':'SIDE_WALL_L','position':[-2.15,2,1.35],'size':[.12,6.2,2.7],'color':[.52,.53,.51]},
            {'id':'SIDE_WALL_R','position':[2.15,2,1.35],'size':[.12,6.2,2.7],'color':[.52,.53,.51]},
            {'id':'INSTRUMENT_PANEL','position':[0,3.65,1.25],'size':[3.7,.35,1.25],'color':[.055,.07,.075]},
            {'id':'RADAR_L','position':[-.85,3.43,1.38],'size':[.65,.05,.45],'color':[.04,.28,.18]},
            {'id':'RADAR_R','position':[.85,3.43,1.38],'size':[.65,.05,.45],'color':[.26,.055,.035]},
            {'id':'CENTER_PEDESTAL','position':[0,2.25,.72],'size':[.62,2.1,.72],'color':[.075,.085,.09]},
            {'id':'PILOT_SEAT_L','position':[-.82,2.05,.65],'size':[.68,.75,1.3],'color':[.12,.14,.15]},
            {'id':'PILOT_SEAT_R','position':[.82,2.05,.65],'size':[.68,.75,1.3],'color':[.12,.14,.15]},
            {'id':'WINDSHIELD_L','position':[-.95,4.02,2.05],'size':[1.7,.08,.9],'color':[.16,.34,.47]},
            {'id':'WINDSHIELD_R','position':[.95,4.02,2.05],'size':[1.7,.08,.9],'color':[.16,.34,.47]},
        ]
    if scene == 4:
        return 'LOC_SKY', [
            {'id':'SKY_FLOOR','position':[0,5,0],'size':[30,30,.2],'color':[.12,.16,.25]},
            {'id':'STORM_CLOUD_A','position':[-4,9,5],'size':[8,4,5],'color':[.20,.23,.29]},
            {'id':'STORM_CLOUD_B','position':[5,13,7],'size':[9,5,5],'color':[.12,.15,.20]},
        ]
    cabin = [
        {'id':'CABIN_FLOOR','position':[0,8,0],'size':[5,20,.15],'color':[.18,.20,.22]},
        {'id':'CABIN_CEILING','position':[0,8,2.85],'size':[5,20,.12],'color':[.82,.82,.78]},
        {'id':'CABIN_WALL_L','position':[-2.5,8,1.5],'size':[.12,20,3],'color':[.68,.68,.64]},
        {'id':'CABIN_WALL_R','position':[2.5,8,1.5],'size':[.12,20,3],'color':[.68,.68,.64]},
        {'id':'CABIN_AISLE','position':[0,8,.12],'size':[1.1,20,.03],'color':[.10,.12,.14]},
        {'id':'OVERHEAD_BIN_L','position':[-2.0,8,2.35],'size':[.65,18,.65],'color':[.72,.72,.69]},
        {'id':'OVERHEAD_BIN_R','position':[2.0,8,2.35],'size':[.65,18,.65],'color':[.72,.72,.69]},
        {'id':'FORWARD_GALLEY','position':[0,-1,1.2],'size':[4,.8,2.4],'color':[.45,.46,.44]},
        {'id':'AFT_GALLEY','position':[0,18,1.2],'size':[4,.8,2.4],'color':[.45,.46,.44]},
        {'id':'ROW19_BIN','position':[1.8,10,2.4],'size':[1.0,.9,.45],'color':[.75,.72,.60]},
    ]
    for row, y in enumerate(range(2, 17, 2), 1):
        cabin.extend([
            {'id':f'SEAT_{row:02d}_L','position':[-1.35,y,.62],
             'size':[1.25,1.05,1.25],'color':[.13,.20,.31]},
            {'id':f'SEAT_{row:02d}_R','position':[1.35,y,.62],
             'size':[1.25,1.05,1.25],'color':[.13,.20,.31]},
            {'id':f'WINDOW_{row:02d}_L','position':[-2.43,y,1.55],
             'size':[.05,.72,.58],'color':[.20,.43,.58]},
            {'id':f'WINDOW_{row:02d}_R','position':[2.43,y,1.55],
             'size':[.05,.72,.58],'color':[.20,.43,.58]},
        ])
    return 'LOC_CABIN', cabin


def build(run, output):
    run = Path(run).resolve()
    plan = read(run/'parse/shot_plan.json')
    if not plan or not plan.get('shots'):
        raise ValueError('shot_plan.json is missing or empty')
    from script_pipeline.spatial_planner import camera_for_shot, active_entities_for_shot, validate_spatial_shot
    shots = []
    entities = {}
    for name, position, color, kind in [
        ('PILOTO',[-.8,2.0,0],[.12,.16,.30],'character'),
        ('COPILOTO',[.8,2.0,0],[.16,.18,.22],'character'),
        ('JI-HO',[-1.0,3.0,0],[.08,.14,.30],'character'),
        ('SEO-YEON',[1.0,3.0,0],[.08,.14,.30],'character'),
        ('HA-EUN',[-1.0,12.0,0],[.28,.48,.70],'character'),
        ('MIN-JUN',[1.0,12.0,0],[.08,.14,.30],'character'),
    ]:
        entities[name] = {'kind':kind,'position':position,'yaw':0,'wardrobe_id':name+'_UNIFORM',
                          'color':color,'skin':[.55,.30,.18],'present':False}
    entities['CART_FRONT'] = {'kind':'prop','position':[0,1,1],'color':[.35,.35,.38],'size':[.6,.6,1.1], 'present':False}
    entities['CART_AFT'] = {'kind':'prop','position':[0,14,1],'color':[.35,.35,.38],'size':[.6,.6,1.1], 'present':False}
    location_cast = {
        'LOC_COCKPIT':['PILOTO','COPILOTO'],
        'LOC_CABIN':['JI-HO','SEO-YEON','HA-EUN','MIN-JUN','CART_FRONT','CART_AFT'],
        'LOC_SKY':[],
    }
    for index, shot in enumerate(plan['shots']):
        scene = int(shot.get('scene', 1))
        lid, boxes = _location(scene)
        seconds = float(shot.get('seconds') or shot.get('editorial_seconds') or 3.0)
        base_camera = {'position':[0,-5,1.6],'target':[0,3,1.3],'lens_mm':38,'sensor_mm':36,
                  'width':int(plan.get('width',768) or 768),'height':int(plan.get('height',512) or 512),
                  'near':.05,'far':60}
        if scene == 1:
            base_camera.update(position=[0,-3,1.72],target=[0,2.6,1.35],lens_mm=42)
        elif scene == 4:
            base_camera.update(position=[0,-10,4],target=[0,6,3],lens_mm=50)
        planning = dict(shot)
        planning['screen_side'] = (plan.get('screen_sides') or {}).get(shot.get('subject',''), '')
        action_text = ' '.join(str(shot.get(k) or '') for k in ('beat','storyboard_prompt','video_prompt')).casefold()
        if lid == 'LOC_CABIN' and ('cart' in action_text or 'carrinho' in action_text):
            planning['active_props'] = [
                'CART_FRONT' if shot.get('subject') in {'JI-HO','SEO-YEON'} else 'CART_AFT']
        camera = camera_for_shot(base_camera, planning, entities)
        active = active_entities_for_shot(planning, entities, location_cast=location_cast.get(lid, []))
        spatial_shot = {'id':shot['id'],'scene':scene,'seconds':seconds,'camera':camera,
                      'action':shot.get('video_prompt') or shot.get('storyboard_prompt') or 'Emergency turbulence action.',
                      'prompt':('Cinematic realistic emergency flight sequence. The same aircraft and fixed spatial layout. '
                                + (shot.get('storyboard_prompt') or shot.get('video_prompt') or '')),
                      'location_id':lid,'location':{'boxes':boxes},'active_entities':active,
                      'framing':shot.get('framing'),'subject':shot.get('subject',''),'event':None}
        validate_spatial_shot(spatial_shot, entities)
        shots.append(spatial_shot)
    spec = {'schema_version':1,'title':'VOO 702 - emergência de turbulência','fps':float(plan.get('fps',24)),
            'location_id':'LOC_CABIN','location':{'boxes':[]},'entities':entities,'location_cast':location_cast,
            'shots':shots,'source_run':str(run)}
    write(output,spec)
    return spec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    spec=build(args.run,Path(args.output))
    print(f'[voo702-spatial] {len(spec["shots"])} shots -> {args.output}')


if __name__ == '__main__':
    main()
