import cv2
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'outputs/decupagem_06_palace_emerald'
VIDEO = RUN/'final/movie.mp4'
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)

def frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t*fps))
    ok, im = cap.read()
    if not ok:
        raise RuntimeError(t)
    return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

def sheet(times, name, width=432, cols=3):
    height=round(width*480/864)
    grid=Image.new('RGB',(width*cols,(height+26)*((len(times)+cols-1)//cols)), '#202020')
    draw=ImageDraw.Draw(grid)
    for i,t in enumerate(times):
        x,y=(i%cols)*width,(i//cols)*(height+26)
        grid.paste(frame(t).resize((width,height)),(x,y+26))
        draw.text((x+8,y+6),f'{t:06.2f}s | frame {round(t*fps)}',fill='white')
    grid.save(OUT/name)

for n,times in enumerate([list(range(0,20,2)),list(range(20,40,2)),list(range(40,57,2))]):
    sheet(times, f'panorama_{n+1}.jpg')
clips=[]
elapsed=0
for line in (RUN/'intermediate/concat_list.txt').read_text(encoding='utf-8-sig').splitlines():
    path=line[6:-1]
    info=json.loads(subprocess.check_output(['C:/ffmpeg/bin/ffprobe.exe','-v','error','-show_streams','-of','json',path],text=True))
    v=next(s for s in info['streams'] if s['codec_type']=='video')
    seconds=float(v['duration'])
    clips.append(dict(file=path,start=elapsed,end=elapsed+seconds,frames=int(v['nb_frames'])))
    elapsed+=seconds
(OUT/'timeline.json').write_text(json.dumps(clips,indent=2),encoding='utf-8')
for i,c in enumerate(clips):
    ts=[c['start']+min(.2,(c['end']-c['start'])/4), (c['start']+c['end'])/2,c['end']-.15]
    sheet(ts,f'plano_{i:02d}.jpg',width=576)
print(json.dumps(clips,indent=2))
