"""Reproducible, CPU-only media inspection of the spatial pilot."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw
from script_pipeline.production_project import read, write


def build_report(project):
    project = Path(project)
    plan = read(project/'world/shot_bindings.json')
    tiles, observations = [], []
    for shot in plan['shots']:
        video = shot.get('video')
        if not video:
            continue
        capture = cv2.VideoCapture(video)
        frames,changes = [],[]
        previous = None
        while True:
            ok,frame = capture.read()
            if not ok:
                break
            small = cv2.resize(frame,(192,128)).astype(np.float32)
            if previous is not None:
                changes.append(float(np.abs(small-previous).mean()))
            previous = small
            frames.append(frame)
        capture.release()
        if not frames:
            raise ValueError(f'Video cannot be decoded: {video}')
        for label,index in [('start',0),('middle',len(frames)//2),('end',len(frames)-1)]:
            thumb = Image.fromarray(cv2.cvtColor(frames[index],cv2.COLOR_BGR2RGB)).resize((384,256))
            tile = Image.new('RGB',(384,284),'#111111')
            tile.paste(thumb,(0,28))
            ImageDraw.Draw(tile).text((8,8),f'{shot["id"]} / {label} / frame {index}',fill='white')
            tiles.append(tile)
        observations.append({'shot_id':shot['id'],'frames':len(frames),
            'mean_frame_change_0_255':float(np.mean(changes)),
            'identical_transition_fraction':float(np.mean(np.array(changes)<.05)),
            'state_initial':shot['binding']['initial'],'state_final':shot['binding']['final'],
            'projection_error_px':shot['controls']['initial']['max_projection_error_px']})
    if tiles:
        sheet = Image.new('RGB',(1152,284*len(observations)))
        for i,tile in enumerate(tiles):
            sheet.paste(tile,((i%3)*384,(i//3)*284))
        sheet.save(project/'entregas/contact_sheet.jpg',quality=92)
    report = {'shots':observations,'note':'Frame differences prove pixel changes, not correct action or identity.'}
    write(project/'entregas/media_report.json',report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    print(json.dumps(build_report(parser.parse_args().project),indent=2))
