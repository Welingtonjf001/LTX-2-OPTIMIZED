import json, sys

api = json.load(open("_api_25_t2v_single.json", encoding="utf-8"))

NEGATIVE = ("subtitles, captions, written text, logos, watermark, extra characters, duplicate characters, "
            "merged faces, identity swapping, costume changes, wrong speaker, both characters moving their "
            "lips simultaneously, overlapping dialogue, background voices, inaccurate lip sync, deformed "
            "hands, distorted faces, flickering, abrupt cuts, camera teleportation, gibberish speech")

POSITIVE = ("""A cinematic live-action western confrontation on an empty frontier street at golden hour. Keep exactly two characters with fixed appearances and positions. Clara stands on the left, a sharp-eyed female gunslinger in her early thirties, wearing a dusty beige hat, red neckerchief, long brown coat and a rifle held low beside her leg. Cole stands on the right, a rugged male outlaw with dark stubble, a black hat, weathered leather coat and one hand hovering near his holstered revolver. The camera holds a tense medium two-shot and performs a very slow dolly forward. Dust crosses the street, a wooden sign creaks, distant horses snort and a lonely harmonica plays softly. Spoken dialogue is exclusively in Brazilian Portuguese with distinct voices, precise lip sync and no overlapping speech. Clara narrows her eyes and says evenly, \u201cLargue o rev\u00f3lver, Cole. Esta cidade n\u00e3o teme voc\u00ea.\u201d Clara closes her mouth. Cole tilts his hat and answers in a low rough voice, \u201cN\u00e3o vim pela cidade. Vim pelo ouro do trem.\u201d Cole becomes silent. Clara tightens her grip on the rifle and says, \u201cEnt\u00e3o escolheu o trilho errado. Meu rifle nunca erra.\u201d Clara closes her mouth. Cole gives a restrained smile, his spurs shifting in the dust, and replies, \u201cVeremos ao meio-dia... se voc\u00ea chegar at\u00e9 l\u00e1.\u201d A distant clock bell rings once. No gunfire, subtitles or written text.""")

api["5508"]["inputs"]["value"] = POSITIVE
api["5509"]["inputs"]["value"] = NEGATIVE
api["5511"]["inputs"]["value"] = 24       # fps
api["5512"]["inputs"]["value"] = 15       # duration seconds -> 361 frames
api["5014:5506"]["inputs"]["value"] = False  # use image input = False (pure T2V)
api["5514:3059"]["inputs"]["width"] = 768
api["5514:3059"]["inputs"]["height"] = 512
api["4852"]["inputs"]["filename_prefix"] = "screenplay_western_2_5"

json.dump(api, open("_api_25_western.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote _api_25_western.json")
