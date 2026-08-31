# -*- coding: utf-8 -*-
"""Experimento de atribuicao: QUAL fator fez o clipe T2V ganhar da decupagem?

O clipe de referencia (prompt do usuario, T2V continuo, sem still, seed 77) foi
julgado "o melhor de todos". Ele muda CINCO coisas ao mesmo tempo em relacao a
cadeia de decupagem. Tres bracos, um fator alterado por vez, todo o resto
identico (seed 77, 960x544, distilled, mesmas flags de servidor):

  A) mesmo prompt + STILL a forca 1.0 (o proprio frame 0 do clipe de
     referencia, entao a composicao inicial e IDENTICA -- so mede o custo da
     ancora dura de imagem, que e como a decupagem condiciona)
  B) mesmo conteudo DIVIDIDO em 2 clipes + concat (mede o custo da
     continuidade quebrada: identidade, luz e audio atraves de um corte)
  C) prompt ESTILO PIPELINE ANTIGO (fragmentos, sem falas, sem audio, sem
     beat, 5/8 no score) em clipe continuo (mede so a qualidade do prompt)
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, r'E:\Users\home\Documents\LTX-2-OPTIMIZED')
os.environ["LTX_COMFY_EXTRA_ARGS"] = "--disable-dynamic-vram"
os.environ["LTX_COMFY_CACHE_NONE"] = "1"

import ltx25_backend  # noqa: E402

OUT = r'E:\Users\home\Documents\LTX-2-OPTIMIZED\outputs\decupagem\lyra\final'
FFMPEG = "C:/ffmpeg/bin/ffmpeg.exe"

IDENT = (
    "Keep exactly two characters with fixed identities throughout the shot. "
    "On the left stands Lyra, a young female mage with long silver hair, a "
    "dark-blue cloak and a crystal staff. On the right stands Thoren, a "
    "broad-shouldered male warrior with short brown hair, worn steel armor, a "
    "round shield and a sword. "
)
CENARIO = (
    "A cinematic high-fantasy scene in a circular ancient stone chamber "
    "illuminated by blue moonlight and glowing golden runes. "
)
SOM = (
    "Wind passes through the chamber, stones vibrate, magical energy hums and "
    "restrained orchestral music builds underneath. Spoken dialogue is "
    "exclusively in natural Brazilian Portuguese with accurate lip sync and "
    "no overlapping speech. "
)
FALA_LYRA = ("\u201cThoren, as runas despertaram. O drag\u00e3o sentiu nossa "
             "presen\u00e7a.Subiram centenas deles, da escurid\u00e3o, "
             "famintos, incans\u00e1veis.\u201d")
FALA_THOREN = "\u201cMantenha o portal aberto. Eu contenho suas chamas.\u201d"

PROMPT_REF = (
    CENARIO + IDENT +
    "A huge dragon shadow moves behind a sealed stone archway. The camera "
    "begins in a medium two-shot and slowly pushes closer while circling "
    "slightly. " + SOM +
    "Lyra touches a glowing rune, turns toward Thoren and says urgently, "
    + FALA_LYRA + " Lyra closes her mouth. Thoren raises his shield, watches "
    "the archway and answers in a deep steady voice, " + FALA_THOREN +
    " Thoren closes his mouth. Lyra looks up as the moonlight begins fading."
)

# B1/B2: o MESMO conteudo, repartido onde a decupagem cortaria (troca de
# falante). Cada metade repete cenario+identidades+som, como todo clipe da
# decupagem precisa fazer.
PROMPT_B1 = (
    CENARIO + IDENT +
    "A huge dragon shadow moves behind a sealed stone archway. The camera "
    "begins in a medium two-shot and slowly pushes closer while circling "
    "slightly. " + SOM +
    "Lyra touches a glowing rune, turns toward Thoren and says urgently, "
    + FALA_LYRA + " Lyra closes her mouth."
)
PROMPT_B2 = (
    CENARIO + IDENT +
    "A huge dragon shadow moves behind a sealed stone archway. The camera "
    "holds a close two-shot. " + SOM +
    "Thoren raises his shield, watches the archway and answers in a deep "
    "steady voice, " + FALA_THOREN + " Thoren closes his mouth. Lyra looks up "
    "as the moonlight begins fading."
)

# C: o estilo do pipeline ANTIGO -- fragmentos colados, sem fala citada, sem
# desenho de som, sem beat final, sem trava de identidade (score 5/8).
PROMPT_C = (
    "Lyra gestures with her crystal staff, her expression urgent as she "
    "speaks to Thoren. Thoren grips his sword, his face set in determination. "
    "A young woman with long silver hair, wearing a dark blue cloak and "
    "holding a crystal staff. A broad-shouldered warrior with short brown "
    "hair, wearing worn steel armor and holding a round shield and sword. "
    "the camera pushes in slowly. cinematic high-fantasy, polished 3D "
    "animation, sharp details, dramatic lighting, natural cinematic lighting, "
    "balanced composition."
)


def gera(nome, prompt, frames, image=None):
    out = os.path.join(OUT, nome)
    t0 = time.time()
    kw = dict(width=960, height=544, num_frames=frames, frame_rate=24.0,
              seed=77, log_cb=lambda m: print(f"[{nome}] {m}", flush=True),
              timeout=3600)
    if image:
        kw.update(image_path=image, image_strength=1.0)
    ltx25_backend.generate(prompt, out, **kw)
    print(f"[{nome}] OK em {time.time()-t0:.0f}s", flush=True)
    return out


# --- A: ancora de still a 1.0, composicao inicial identica ao de referencia
gera("atrib_A_still.mp4", PROMPT_REF, 457,
     image=os.path.join(OUT, "pu_frame0.png"))

# --- B: dois clipes + concat
b1 = gera("atrib_B1.mp4", PROMPT_B1, 265)
b2 = gera("atrib_B2.mp4", PROMPT_B2, 193)
lista = os.path.join(OUT, "_atrib_b.txt")
with open(lista, "w", encoding="utf-8") as f:
    f.write(f"file '{b1}'\nfile '{b2}'\n")
subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", lista, "-c:v", "libx264", "-crf", "16",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                os.path.join(OUT, "atrib_B_2clipes.mp4")], check=True)
print("[atrib_B] concat ok", flush=True)

# --- C: prompt fragmentado, clipe continuo
gera("atrib_C_prompt_antigo.mp4", PROMPT_C, 457)

print("EXPERIMENTO COMPLETO", flush=True)
