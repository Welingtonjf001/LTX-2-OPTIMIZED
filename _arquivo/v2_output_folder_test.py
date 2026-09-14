import os
import music_maker_ui_v2 as ui

ui.MUSIC_AUDIO_PATH = r"E:\Users\home\Documents\LTX-2-OPTIMIZED\v2_smoke\quem_2s.wav"
audio = r"E:\Users\home\Documents\LTX-2-OPTIMIZED\v2_smoke\quem_2s.wav"
image = r"E:\Users\home\Documents\LTX-2-OPTIMIZED\v2_smoke\photos\reference.png"
ui.SCENES_DATA[0] = {"audio_path": audio, "video_path": None}

audios, images = ui.create_generation_folder(
    ["folder organization test"] + [None] * 19,
    [audio] + [None] * 19,
    [image] + [None] * 19,
)
print(ui.CURRENT_RUN_DIR)
print(audios[0])
print(images[0])
for name in ("input", "audio", "scenes", "frames", "lipsync", "intermediate", "final", "logs"):
    assert os.path.isdir(os.path.join(ui.CURRENT_RUN_DIR, name)), name
assert os.path.isfile(os.path.join(ui.CURRENT_RUN_DIR, "generation_manifest.json"))
print("OUTPUT_FOLDER_TEST=OK")
