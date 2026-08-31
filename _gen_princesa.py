import os, time, subprocess, ltx25_backend as B

PROMPT = ("A beautiful teenage princess with fair skin, expressive green eyes, and black hair, "
          "wearing a long blue dress and an emerald necklace; a medium shot at sunset with the "
          "camera slowly circling her as she sings and expresses herself on the balcony of a "
          "crystal castle.")
WAV = "outputs/_diag/musica.wav"

def vram():
    try:
        o = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.split("\n")[0].strip()
        return f"{int(o)/1024:.1f}GB"
    except Exception:
        return "?"

B.ensure_server()
print(f"[gen] VRAM antes: {vram()}", flush=True)

# Sondagem: mesma resolucao alvo, poucos frames -- se estourar aqui, estoura no alvo.
t0 = time.time()
try:
    B.generate(PROMPT, "outputs/_diag/probe_1280.mp4",
               width=1280, height=704, num_frames=121, frame_rate=24.0, seed=7, timeout=1800)
    print(f"[gen] SONDAGEM 1280x704/121f OK em {time.time()-t0:.0f}s, VRAM pico ~{vram()}", flush=True)
    res = (1280, 704)
except Exception as e:
    print(f"[gen] SONDAGEM 1280x704 FALHOU: {str(e).splitlines()[0]}", flush=True)
    print("[gen] caindo para 960x544 (padrao do workflow oficial)", flush=True)
    res = (960, 544)

W, H = res
t0 = time.time()
out = B.generate(PROMPT, "outputs/_diag/princesa_final.mp4",
                 width=W, height=H, num_frames=465, frame_rate=24.0, seed=7,
                 audio_conditioning=WAV, timeout=7200)
print(f"[gen] FINAL {W}x{H}/465f em {time.time()-t0/1:.0f}s -> {out}", flush=True)
print("GEN_FIM", flush=True)
