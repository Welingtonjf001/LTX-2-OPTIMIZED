"""Teste A/B do LoRA "Multi-Ref Character Storyboard V2": mesma imagem de
referencia (rosto da espadachim, extraido do beat de close-up da cena wuxia),
mesmo prompt (uma cena NOVA, nao a que gerou a referencia), mesma seed --
so a forca do LoRA muda (0.0 = base, 1.0 = LoRA). O texto e codificado UMA
vez e reaproveitado nas duas gerações (a etapa cara, ~150s por prompt) --
so o estagio 2 (LoRA+sampler) roda duas vezes.
"""
import os
import sys
import time

sys.path.insert(0, ".")
import lora_storyboard_backend as lsb

REF_IMAGE = "outputs/continuous/wuxia_23_voz/frames/chunk_002_last.png"
PROMPT = ("Cinematic wuxia film still, the same young swordswoman now walking "
          "through a sunlit bamboo forest clearing at dawn, medium shot, "
          "confident stride, gentle morning light, live action footage.")
NEGATIVE = lsb.DEFAULT_NEGATIVE
OUT_DIR = "outputs/continuous/_lora_ab"
SEED = 777
LORA_NAME = "ltx_2.3_Multi-Ref Character Storyboard_V2.safetensors"


def log(msg):
    print(f"[ab] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    staged = lsb.backend.stage_input_image(REF_IMAGE)
    image_filename = os.path.basename(staged)
    log(f"referencia -> {image_filename}")

    run_id = "ab" + str(int(time.time()))
    pos_filename = f"lora_ab_pos_{run_id}"
    neg_filename = f"lora_ab_neg_{run_id}"
    embeddings_dir = os.path.join(lsb.backend.COMFY_ROOT, "models", "embeddings")
    pos_path = os.path.join(embeddings_dir, f"{pos_filename}.safetensors")
    neg_path = os.path.join(embeddings_dir, f"{neg_filename}.safetensors")

    log("codificando texto UMA vez (reaproveitado nas duas geracoes)...")
    lsb._encode_prompts_native(prompt=PROMPT, negative=NEGATIVE,
                                pos_path=pos_path, neg_path=neg_path, log_cb=log)
    if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
        raise SystemExit("codificacao nao gravou os arquivos esperados.")

    results = {}
    for label, strength in (("sem_lora", 0.0), ("com_lora", 1.0)):
        log(f"gerando {label} (lora_strength={strength})...")
        stage2 = lsb.build_stage2_video(
            lora_name=LORA_NAME, lora_strength=strength,
            pos_filename=pos_filename, neg_filename=neg_filename,
            width=768, height=512, num_frames=97, frame_rate=24.0,
            steps=8, cfg=3.0, seed=SEED,
            image_filename=image_filename, image_strength=0.9,
        )
        files = lsb.backend.submit_and_wait(stage2, log_cb=log, expect_node=lsb.SAVE_NODE_ID)
        if not files:
            raise SystemExit(f"{label}: ComfyUI nao produziu arquivo.")
        src = os.path.join(lsb.backend.COMFY_ROOT, "output", files[0])
        dest = os.path.join(OUT_DIR, f"{label}.mp4")
        import shutil
        shutil.copy2(src, dest)
        results[label] = dest
        log(f"{label} -> {dest}")

    for p in (pos_path, neg_path):
        try:
            os.remove(p)
        except OSError:
            pass

    log(f"pronto: {results}")


if __name__ == "__main__":
    main()
