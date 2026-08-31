# -*- coding: utf-8 -*-
"""Where does a single continuous LTX-2.5 take break?

Longest we had actually generated was 433 frames (18s). The screenplay UI caps
max_clip_seconds at 15 with the note "clipes longos travam o upsampler" -- a
limit inherited from the 2.3 native path, never verified on the 2.5 ComfyUI
route. This measures it: same prompt (the user's fantasy scene, which carries an
extended ~20s speech, i.e. exactly the case that motivated the question), same
seed, only the length changes.

Frame counts must be 1 + a multiple of 8:
  30s @24fps -> 721,  45s @24fps -> 1081
"""
import sys
sys.path.insert(0, ".")
import time
import ltx25_backend as b

FANTASY = (
    "A cinematic high-fantasy scene in a circular ancient stone chamber illuminated by blue moonlight "
    "and glowing golden runes. Keep exactly two characters with fixed identities throughout the shot. "
    "On the left stands Lyra, a young female mage with long silver hair, a dark-blue cloak and a crystal "
    "staff. On the right stands Thoren, a broad-shouldered male warrior with short brown hair, worn steel "
    "armor, a round shield and a sword. A huge dragon shadow moves behind a sealed stone archway. The "
    "camera begins in a medium two-shot and slowly pushes closer while circling slightly. Wind passes "
    "through the chamber, stones vibrate, magical energy hums and restrained orchestral music builds "
    "underneath. Spoken dialogue is exclusively in natural Brazilian Portuguese with accurate lip sync "
    "and no overlapping speech. Lyra touches a glowing rune, turns toward Thoren and says urgently, "
    "\u201cThoren, as runas despertaram. O drag\u00e3o sentiu nossa presen\u00e7a. Desde ontem, ap\u00f3s o terremoto na "
    "cidade perdida, as correntes que prendiam os portais da chama intensa se quebraram. Subiram centenas "
    "deles da escurid\u00e3o, famintos, incans\u00e1veis.\u201d Lyra closes her mouth. Thoren raises his shield, watches "
    "the archway and answers in a deep steady voice, \u201cMantenha o portal aberto. Eu contenho as chamas.\u201d "
    "Thoren closes his mouth. Lyra looks up as the moonlight begins fading and says with shortened breath, "
    "\u201cA lua est\u00e1 desaparecendo... restam poucos segundos!\u201d Lyra becomes silent. Thoren plants his feet, "
    "points his sword toward the archway and declares with rising force, \u201c\u00c9 o bastante. Pela \u00faltima "
    "estrela, avan\u00e7amos!\u201d The portal flashes at the final word. No subtitles or on-screen text."
)

CASES = {"17s": 409, "30s": 721, "45s": 1081, "60s": 1441}

if __name__ == "__main__":
    for label in (sys.argv[1:] or ["30s", "45s"]):
        frames = CASES[label]
        print(f"\n=== {label} ({frames} frames) ===", flush=True)
        t0 = time.time()
        try:
            out = b.generate(
                FANTASY, f"outputs_25/longtake_{label}.mp4",
                width=768, height=512, num_frames=frames, frame_rate=24.0, seed=42,
                variant="distilled", log_cb=lambda m: print(m, flush=True), timeout=10800,
            )
            print(f"[{label}] OK em {time.time()-t0:.0f}s -> {out}", flush=True)
        except Exception as e:
            print(f"[{label}] FALHOU em {time.time()-t0:.0f}s: {e}", flush=True)
