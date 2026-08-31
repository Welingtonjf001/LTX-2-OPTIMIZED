import music_maker_ui as ui
audio = r"E:\Users\home\Documents\quem.mp3"
result = ui.slice_audio(audio, "cinematic music video, consistent singer", 24, 225, True)
print(result[0])
print("SCENE_COUNT", sum(1 for item in ui.SCENES_DATA if item))
for idx, item in enumerate(ui.SCENES_DATA):
    if item and item.get("lyrics_text"):
        print("LYRIC_SCENE", idx + 1, item["lyrics_text"][:300])
        break
else:
    print("NO_LYRICS_IN_SCENES")
