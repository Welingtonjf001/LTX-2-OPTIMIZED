"""Passo 0 do teste em vídeo real: o VAE do LTX 2.5 faz ida e volta sem estragar
o vídeo, rodando FORA do Windows (Linux, Python 3.13, torch 2.11, RTX 4070)?

Sem isso, nenhum preditor treinado sobre latentes pode ser avaliado: o
decoder congelado é a régua. Mede a qualidade da reconstrução (PSNR por quadro)
e salva o latente, que vira dado de treino nos passos seguintes.

Carrega o VAE pelo NÚCLEO do ComfyUI deste repositório (sem servidor, sem
custom node), do mesmo jeito que o `VAELoader` faz: state dict + metadados
do safetensors, que definem a configuração do VAE 2.5.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def ler_video(caminho: Path, max_frames: int | None):
    import av
    quadros = []
    with av.open(str(caminho)) as c:
        fps = float(c.streams.video[0].average_rate)
        for f in c.decode(video=0):
            quadros.append(f.to_ndarray(format="rgb24"))
            if max_frames and len(quadros) >= max_frames:
                break
    return np.stack(quadros), fps


def gravar_video(caminho: Path, quadros: np.ndarray, fps: float):
    import av
    with av.open(str(caminho), mode="w") as c:
        s = c.add_stream("libx264", rate=round(fps))
        s.width, s.height, s.pix_fmt = quadros.shape[2], quadros.shape[1], "yuv420p"
        s.options = {"crf": "12"}
        for q in quadros:
            for pkt in s.encode(av.VideoFrame.from_ndarray(q, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="pasta do ComfyUI (só o código)")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="corta o clipe; o VAE exige 1 + múltiplo de 8 quadros")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # comfy.cli_args lê sys.argv na importação: não pode ver os argumentos deste script.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, args.comfy)
    import torch
    import comfy.sd
    import comfy.utils

    quadros, fps = ler_video(Path(args.clip), args.max_frames)
    t, h, w, _ = quadros.shape
    t_valido = 1 + 8 * ((t - 1) // 8)
    if t_valido != t:
        print(f"cortando de {t} para {t_valido} quadros (1 + múltiplo de 8)")
        quadros = quadros[:t_valido]
        t = t_valido
    print(f"clipe: {t} quadros {w}x{h} a {fps:.2f} fps")

    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    del sd
    print(f"VAE carregado em {time.time() - t0:.1f}s | device {vae.device} | dtype {vae.vae_dtype} "
          f"| compressão espacial {vae.spacial_compression_decode()} temporal {vae.temporal_compression_decode()}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pixels = torch.from_numpy(quadros).float() / 255.0          # [T,H,W,C]

    # O ComfyUI executa todo node dentro de torch.inference_mode(). Fora dele, o
    # decode em blocos morre com "Inplace update to inference tensor outside
    # InferenceMode" no process_output -- MEDIDO 2026-09-13 na primeira execução
    # deste script (o encode passou, o decode quebrou).
    t0 = time.time()
    with torch.inference_mode():
        latente = vae.encode(pixels)
    t_enc = time.time() - t0
    print(f"encode: {t_enc:.1f}s -> latente {tuple(latente.shape)} {latente.dtype}")

    # Mesmos parâmetros padrão do node VAEDecodeTiled (tile 512, overlap 64,
    # temporal 64/8), convertidos para unidades de latente como o node faz.
    comp = vae.spacial_compression_decode()
    tcomp = vae.temporal_compression_decode() or 1
    tile_t = max(2, 64 // tcomp)
    overlap_t = max(1, min(tile_t // 2, 8 // tcomp))
    t0 = time.time()
    with torch.inference_mode():
        img = vae.decode_tiled(latente, tile_x=512 // comp, tile_y=512 // comp, overlap=64 // comp,
                               tile_t=tile_t, overlap_t=overlap_t)
    t_dec = time.time() - t0
    if img.ndim == 5:
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    rec = (img.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    print(f"decode: {t_dec:.1f}s -> {tuple(rec.shape)}")

    n = min(len(rec), len(quadros))
    por_quadro = [psnr(quadros[i], rec[i]) for i in range(n)]
    pico = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0

    torch.save(latente.cpu(), out / "latente.pt")
    gravar_video(out / "reconstruido.mp4", rec[:n], fps)
    try:
        from PIL import Image
        meio = n // 2
        lado = np.concatenate([quadros[meio], rec[meio]], axis=1)
        Image.fromarray(lado).save(out / "quadro_meio_original_vs_reconstruido.png")
    except Exception as e:  # imagem é conveniência, não pode derrubar a medição
        print(f"(sem PNG lado a lado: {e})")

    resumo = {
        "clipe": args.clip, "quadros": int(t), "resolucao": [int(w), int(h)],
        "latente": list(latente.shape), "encode_s": round(t_enc, 1), "decode_s": round(t_dec, 1),
        "quadros_reconstruidos": int(len(rec)),
        "psnr_medio": round(float(np.mean(por_quadro)), 2),
        "psnr_min": round(float(np.min(por_quadro)), 2),
        "psnr_primeiro": round(por_quadro[0], 2), "psnr_ultimo": round(por_quadro[-1], 2),
        "vram_pico_gib": round(pico, 2),
    }
    (out / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
