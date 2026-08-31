import time, ltx25_backend as B

PROMPT = ("A beautiful teenage princess with fair skin, expressive green eyes, and black hair, "
          "wearing a long blue dress and an emerald necklace; a medium shot at sunset with the "
          "camera slowly circling her as she sings and expresses herself on the balcony of a "
          "crystal castle.")

B.ensure_server()
t0 = time.time()
out = B.generate(PROMPT, "outputs/_diag/princesa_twostage.mp4",
                 width=640, height=352,          # estagio 2 dobra -> 1280x704 final
                 num_frames=465, frame_rate=24.0, seed=7,
                 audio_conditioning="outputs/_diag/musica.wav",
                 two_stage=True, timeout=7200)
print(f"[ts] TWO-STAGE em {time.time()-t0:.0f}s -> {out}", flush=True)
print("TS_FIM", flush=True)
