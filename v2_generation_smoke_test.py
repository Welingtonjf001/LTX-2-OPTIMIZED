import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")

import music_maker_ui_v2 as ui

audio = r"E:\Users\home\Documents\LTX-2-OPTIMIZED\v2_smoke\quem_2s.wav"
photos = r"E:\Users\home\Documents\LTX-2-OPTIMIZED\v2_smoke\photos"

ui.v2_status("SMOKE TEST: preparando uma cena com foto existente", "Teste", 0)
ui.slice_photo_music(
    audio,
    photos,
    "Cinematic performance, subtle natural motion, realistic lighting",
    25,
    49,
    False,
)

scene = ui.SCENES_DATA[0]
if not scene:
    raise RuntimeError("A preparação da cena não produziu SCENES_DATA[0]")

prompts = [None] * 20
audios = [None] * 20
starts = [None] * 20
lasts = [None] * 20
prompts[0] = scene["prompt"]
audios[0] = scene["audio_path"]
starts[0] = scene["photo_path"]

ui.v2_status("SMOKE TEST: iniciando geração LTX", "Gerando vídeo", 5)
ui.process_chain_generation(
    prompts, audios, starts, lasts,
    ui.DEFAULT_CHECKPOINT, ui.DEFAULT_GEMMA, ui.DEFAULT_UPSAMPLER,
    8, 25, 768, 512, 49, 12345, False, False, 0, 2, False,
)
ui.v2_status(f"SMOKE TEST concluído: {ui.LATEST_VIDEO_PATH}", "Teste concluído", 100)
print("RESULT=", ui.LATEST_VIDEO_PATH, flush=True)
