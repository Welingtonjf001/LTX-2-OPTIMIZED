import os
import music_maker_ui_v2 as ui

video = os.path.abspath("scene_1_20260721_111658.mp4")
audio = os.path.abspath(os.path.join("v2_smoke", "quem_2s.wav"))
output = os.path.abspath("scene_1_wav2lip_scene_test.mp4")

result = ui.run_wav2lip(video, audio, output)
print(f"RESULT={result}", flush=True)
print(ui.CURRENT_LOG[-6000:], flush=True)
