"""O LTX obedece a vocabulario de camera? Mesma cena, mesmo seed, so a
especificacao muda. Testa os dois eixos mais basicos da gramatica: ESCALA
(wide<->close) e ALTURA (contra-plongee<->plongee)."""
import time, ltx25_backend as B

SUBJECT = ("A teenage princess with fair skin, green eyes and black hair, wearing a long "
           "blue dress, on the stone balcony of a crystal castle at sunset")

SPECS = {
    "0_baseline":   "",
    "1_wide":       "extreme wide establishing shot, she is small within a vast frame, the whole castle and landscape visible",
    "2_closeup":    "tight close-up on her face filling the frame, shallow depth of field, background blurred",
    "3_lowangle":   "low angle shot, the camera is near the ground looking steeply up at her, she towers over the frame",
    "4_highangle":  "high angle shot, the camera is far above looking steeply down at her, she is seen from overhead",
}

B.ensure_server()
for tag, spec in SPECS.items():
    prompt = f"{SUBJECT}. {spec}." if spec else f"{SUBJECT}."
    t0 = time.time()
    try:
        B.generate(prompt, f"outputs/_diag/cam_{tag}.mp4",
                   width=640, height=352, num_frames=97, frame_rate=24.0, seed=99,
                   timeout=2400)
        print(f"[cam] {tag}: OK em {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[cam] {tag}: FALHOU -> {str(e).splitlines()[0]}", flush=True)
print("CAM_FIM", flush=True)
