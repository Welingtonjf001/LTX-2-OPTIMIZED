# -*- coding: utf-8 -*-
"""Ciclo completo do storyplay25, chamando as MESMAS funcoes que os botoes da UI
chamam (do_parse -> do_storyboard -> do_video), com a cena da Lyra em 30s -- acima
do limiar de legendas medido (MEMORIAL.md 3.11)."""
import sys, time
sys.path.insert(0, ".")
import storyplay25 as sp

CENA = (
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

print("=== 1. PARSE ===", flush=True)
t0 = time.time()
summary, _log, pick = sp.do_parse(CENA, "gemma4")
print(summary[:1200], flush=True)
print(f"[parse] {time.time()-t0:.0f}s", flush=True)
if not sp.STATE["scenes"]:
    raise SystemExit("parse nao produziu cenas")
idx = sp.STATE["scenes"][0]["index"]

print("\n=== 2. STORYBOARD (um quadro por plano) ===", flush=True)
t0 = time.time()
gallery, status, _log = sp.do_storyboard(
    idx, "flux-2-klein-9b-fp8.safetensors", "Qwen3-8B-FP8-native-bf16.safetensors",
    "flux2-vae.safetensors", 768, 512, 8, 1234)
print(status, f"({time.time()-t0:.0f}s)", flush=True)
for g in gallery: print("   ", g[1], "->", g[0], flush=True)

print("\n=== 3. VIDEO 30s COM KEYFRAMES ===", flush=True)
t0 = time.time()
video, status, _log = sp.do_video(idx, 30, 24, 768, 512, "distilled", 0.8, 42, True)
print(status, f"({time.time()-t0:.0f}s)", flush=True)
print("video:", video, flush=True)
print("E2E_OK" if video else "E2E_FALHOU", flush=True)
