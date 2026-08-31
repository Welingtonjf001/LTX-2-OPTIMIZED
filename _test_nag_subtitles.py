# -*- coding: utf-8 -*-
"""Controlled A/B test: does NAG actually remove the baked-in subtitles?

Background (why the negative prompt alone never worked):
the LTX-2.5 *distilled* workflow samples at **CFG = 1**. Classifier-free
guidance at scale 1 reduces to `uncond + 1.0 * (cond - uncond) == cond`, so the
negative branch cancels out exactly -- the negative prompt is a mathematical
no-op, no matter what it says. Confirmed against our own graph
(`CFGGuider.cfg == 1`) and corroborated by Lightricks' own answer on the
LTX-2 HF discussions ("use Kijai's LTXVModelPatchNAG ... allows you to inject
negative conditioning" for the distilled model).

NAG (Normalized Attention Guidance) applies the negative conditioning inside
attention instead of through the CFG formula, so it still bites at CFG 1. Node
`LTX2_NAG` patches MODEL -> MODEL and takes the negative CONDITIONING as
`nag_cond_video`.

Both arms use the SAME seed (RandomNoise.noise_seed = 42 as shipped) and the
same prompt, so any difference in the output is attributable to NAG.
"""
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

SERVER = "http://127.0.0.1:8188"
BASE_API = "_api_25_t2v_single.json"

NEGATIVE = (
    "subtitles, captions, subtitle bar, closed captions, burned-in text, written text, "
    "on-screen text, text overlay, letters, words, logos, watermark, signature, timestamp, "
    "UI, OSD, extra characters, duplicate characters, merged faces, identity swapping, "
    "costume changes, wrong speaker, both characters moving their lips simultaneously, "
    "overlapping dialogue, background voices, inaccurate lip sync, deformed hands, "
    "distorted faces, flickering, abrupt cuts, camera teleportation, gibberish speech"
)

WESTERN = (
    "A cinematic live-action western confrontation on an empty frontier street at golden hour. "
    "Keep exactly two characters with fixed appearances and positions. Clara stands on the left, "
    "a sharp-eyed female gunslinger in her early thirties, wearing a dusty beige hat, red neckerchief, "
    "long brown coat and a rifle held low beside her leg. Cole stands on the right, a rugged male outlaw "
    "with dark stubble, a black hat, weathered leather coat and one hand hovering near his holstered "
    "revolver. The camera holds a tense medium two-shot and performs a very slow dolly forward. Dust "
    "crosses the street, a wooden sign creaks, distant horses snort and a lonely harmonica plays softly. "
    "Spoken dialogue is exclusively in Brazilian Portuguese with distinct voices, precise lip sync and no "
    "overlapping speech. Clara narrows her eyes and says evenly, “Largue o revólver, Cole. Esta cidade "
    "não teme você.” Clara closes her mouth. Cole tilts his hat and answers in a low rough voice, "
    "“Não vim pela cidade. Vim pelo ouro do trem.” Cole becomes silent. Clara tightens her grip on "
    "the rifle and says, “Então escolheu o trilho errado. Meu rifle nunca erra.” Clara closes her "
    "mouth. Cole gives a restrained smile, his spurs shifting in the dust, and replies, “Veremos ao "
    "meio-dia... se você chegar até lá.” A distant clock bell rings once. No gunfire, subtitles or "
    "written text."
)

# The full 40-term negative above spreads its embedding budget across identity,
# lip-sync, camera and quality concepts. NAG guides *away from* whatever this
# encodes, so diluting it across 40 concepts weakens the anti-text signal
# specifically. This variant encodes text/caption artifacts ONLY, to spend the
# whole negative embedding on the one artifact we're trying to kill.
FOCUSED_NEGATIVE = (
    "subtitles, subtitle bar, captions, closed captions, burned-in subtitles, "
    "burned-in text, on-screen text, text overlay, caption box, lower third, "
    "written words, letters, typography, watermark, timestamp"
)

NAG_NODE_ID = "9001"
CFG_GUIDER = "5516:4828"
UNET_LOADER = "5004:5569"


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def build_api(prefix, seconds, use_nag, nag_scale=11.0, focused=False):
    api = json.load(open(BASE_API, encoding="utf-8"))

    # Structural fixes (schema drift after the ComfyUI core update; see MEMORIAL.md).
    api["5004:5545"]["inputs"]["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    inp = api["5014:4990"]["inputs"]
    inp["scale_method"] = "lanczos"
    for k in ("width", "height", "crop"):
        inp.pop(k, None)
    inp["resize_type"] = "scale dimensions"
    inp["resize_type.width"] = 512
    inp["resize_type.height"] = 512
    inp["resize_type.crop"] = "center"
    api["2004"]["inputs"]["image"] = "4d84bf57_i2v_input.png"
    api["4852"]["inputs"]["format"] = "auto"
    api["4852"]["inputs"]["format.codec"] = "auto"

    api["5508"]["inputs"]["value"] = WESTERN
    api["5509"]["inputs"]["value"] = FOCUSED_NEGATIVE if focused else NEGATIVE
    api["5511"]["inputs"]["value"] = 24
    api["5512"]["inputs"]["value"] = seconds
    api["5014:5506"]["inputs"]["value"] = False
    api["5514:3059"]["inputs"]["width"] = 768
    api["5514:3059"]["inputs"]["height"] = 512
    api["4852"]["inputs"]["filename_prefix"] = prefix

    if use_nag:
        negative_link = api[CFG_GUIDER]["inputs"]["negative"]
        api[NAG_NODE_ID] = {
            "class_type": "LTX2_NAG",
            "inputs": {
                "model": api[CFG_GUIDER]["inputs"]["model"],
                "nag_scale": nag_scale,
                "nag_alpha": 0.25,
                "nag_tau": 2.5,
                "nag_cond_video": negative_link,
                "inplace": True,
            },
        }
        api[CFG_GUIDER]["inputs"]["model"] = [NAG_NODE_ID, 0]

    return api


def run(label, api, timeout=5400):
    print(f"[{label}] submitting...", flush=True)
    try:
        res = post_json(f"{SERVER}/prompt", {"prompt": api, "client_id": str(uuid.uuid4())})
    except urllib.error.HTTPError as e:
        print(f"[{label}] REJECTED: {e.read().decode('utf-8', 'replace')[:2500]}", file=sys.stderr, flush=True)
        return None
    pid = res.get("prompt_id")
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get_json(f"{SERVER}/history/{pid}")
        if pid in hist:
            e = hist[pid]
            st = e.get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                files = [it["filename"] for _n, o in (e.get("outputs") or {}).items()
                         for _k, items in o.items()
                         for it in (items if isinstance(items, list) else [])
                         if isinstance(it, dict) and it.get("filename")]
                print(f"[{label}] DONE in {time.time()-t0:.1f}s -> {files}", flush=True)
                return files
            if st.get("status_str") == "error":
                print(f"[{label}] FAILED: {json.dumps(st.get('messages', []))[:2500]}", file=sys.stderr, flush=True)
                return None
        time.sleep(5)
    print(f"[{label}] TIMEOUT", file=sys.stderr, flush=True)
    return None


if __name__ == "__main__":
    arms = sys.argv[1:] or ["baseline", "nag"]
    for arm in arms:
        if arm == "baseline":
            run("baseline", build_api("nagtest_baseline", 15, use_nag=False))
        elif arm == "nag":
            run("nag", build_api("nagtest_nag", 15, use_nag=True))
        elif arm.startswith("focus"):
            # focus<scale>: focused text-only negative, given NAG scale
            scale = float(arm[5:])
            run(f"focus{scale:g}",
                build_api(f"nagtest_focus{scale:g}".replace(".", "_"), 15,
                          use_nag=True, nag_scale=scale, focused=True))
        elif arm.startswith("nag"):
            scale = float(arm[3:])
            run(f"nag{scale:g}", build_api(f"nagtest_nag{scale:g}".replace(".", "_"), 15,
                                           use_nag=True, nag_scale=scale))
