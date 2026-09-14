"""Passo 3 do teste em vídeo real: decodificar com o decoder CONGELADO os latentes
previstos por `train_latent_dynamics.py`, e medir no espaço de imagem.

Duas comparações, para separar as fontes de erro:
  previsto decodificado  vs  latente REAL decodificado  -> erro só do preditor
  latente real decodificado  vs  clipe original          -> perda do VAE (a régua,
                                                            31,2 dB no ida e volta)
Os pisos (congelado e linear) são decodificados do mesmo jeito, como referência.

O VAE é causal: o 1º quadro latente representa um quadro real sozinho e foi
separado na extração (`primeiro`). Ele é recolocado na frente antes de
decodificar, senão o decoder recebe uma sequência que não é a que ele espera.
Mapeamento: latente completo i>=1 cobre os quadros 1+8(i-1) .. 8i; a previsão
começa no latente completo 1+ctx, isto é, no quadro 1+8·ctx.

Saída: PSNR por modelo na região prevista e um mp4 com os painéis lado a lado
(real | pisos | modelos) para julgar a olho.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ler_video(caminho: Path) -> np.ndarray:
    import av
    with av.open(str(caminho)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def gravar_video(caminho: Path, quadros: np.ndarray, fps: int = 24):
    import av
    with av.open(str(caminho), mode="w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = quadros.shape[2], quadros.shape[1], "yuv420p"
        s.options = {"crf": "18"}
        for q in quadros:
            for pkt in s.encode(av.VideoFrame.from_ndarray(q, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def rotular(quadro: np.ndarray, texto: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(quadro)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8 + 7 * len(texto), 18], fill=(0, 0, 0))
        d.text((4, 3), texto, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return quadro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--latentes", required=True, help="pasta da extração (manifesto + .pt)")
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--previsoes", required=True, help="out-dir do train_latent_dynamics")
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--escala", type=float, default=0.5, help="reduz os painéis do mp4")
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]
    sys.path.insert(0, args.comfy)
    import torch
    import comfy.sd
    import comfy.utils

    prev_dir = Path(args.previsoes)
    arquivos = sorted(prev_dir.glob(f"previsto_*_seq{args.seq}.pt"))
    if not arquivos:
        raise SystemExit(f"nenhum previsto_*_seq{args.seq}.pt em {prev_dir}")
    previsoes = {a.stem.removeprefix("previsto_").removesuffix(f"_seq{args.seq}"): torch.load(a) for a in arquivos}
    ref = next(iter(previsoes.values()))
    rel, ctx = ref["rel"], ref["ctx"]
    real = ref["real"].float()                                  # [C, ctx+k, H, W]
    k = real.shape[1] - ctx

    manif = {json.loads(l)["rel"]: json.loads(l) for l in
             (Path(args.latentes) / "manifesto.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    primeiro = torch.load(Path(args.latentes) / manif[rel]["arquivo"])["primeiro"].float()  # [C,1,H,W]

    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    del sd
    comp = vae.spacial_compression_decode()
    tcomp = vae.temporal_compression_decode() or 1

    def decodificar(seq_sem_primeiro: torch.Tensor) -> np.ndarray:
        z = torch.cat([primeiro, seq_sem_primeiro], dim=1).unsqueeze(0)
        tile_t = max(2, 64 // tcomp)
        with torch.inference_mode():   # ver vae_roundtrip.py
            img = vae.decode_tiled(z, tile_x=512 // comp, tile_y=512 // comp, overlap=64 // comp,
                                   tile_t=tile_t, overlap_t=max(1, min(tile_t // 2, 8 // tcomp)))
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        return (img.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)

    q, v = real[:, ctx - 1], real[:, ctx - 1] - real[:, ctx - 2]
    passos = torch.arange(1, k + 1, dtype=real.dtype).view(1, k, 1, 1)
    candidatos = {
        "real_latente": real,
        "piso_congelado": torch.cat([real[:, :ctx], q.unsqueeze(1).expand(-1, k, -1, -1)], 1),
        "piso_linear": torch.cat([real[:, :ctx], q.unsqueeze(1) + passos * v.unsqueeze(1)], 1),
    }
    for nome, d in previsoes.items():
        candidatos[nome] = torch.cat([real[:, :ctx], d["previsto"].float()], 1)

    decod = {}
    for nome, z in candidatos.items():
        decod[nome] = decodificar(z)
        print(f"decodificado {nome}: {decod[nome].shape}", flush=True)

    original = ler_video(Path(args.clips_dir) / rel)
    ini = 1 + 8 * ctx                     # primeiro quadro da região prevista
    fim = min(len(original), *(len(x) for x in decod.values()))
    print(f"\nclipe {rel} | contexto {ctx} latentes | previsão de {k} latentes = quadros {ini}..{fim - 1} "
          f"({(fim - ini) / 24:.1f}s)\n")

    linhas = {}
    for nome, quadros in decod.items():
        vs_real = np.mean([psnr(quadros[i], decod["real_latente"][i]) for i in range(ini, fim)])
        vs_orig = np.mean([psnr(quadros[i], original[i]) for i in range(ini, fim)])
        linhas[nome] = {"psnr_vs_latente_real_decod": round(float(vs_real), 2),
                        "psnr_vs_original": round(float(vs_orig), 2)}
        print(f"{nome:>18}: vs latente real decodificado {vs_real:6.2f} dB | vs original {vs_orig:6.2f} dB")

    (prev_dir / f"decod_seq{args.seq}.json").write_text(
        json.dumps({"rel": rel, "ctx": ctx, "k": k, "quadros_previstos": [ini, fim], "psnr": linhas},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    ordem = ["real_latente", "piso_congelado", "piso_linear"] + sorted(previsoes)
    paineis = []
    for i in range(fim):
        fileira = []
        for nome in ordem:
            quadro = decod[nome][i]
            if args.escala != 1.0:
                from PIL import Image
                h, w = quadro.shape[:2]
                quadro = np.asarray(Image.fromarray(quadro).resize((int(w * args.escala) // 2 * 2,
                                                                     int(h * args.escala) // 2 * 2)))
            rot = nome + ("" if i >= ini else " (contexto)")
            fileira.append(rotular(quadro, rot))
        paineis.append(np.concatenate(fileira, axis=1))
    gravar_video(prev_dir / f"comparacao_seq{args.seq}.mp4", np.stack(paineis))
    print(f"\nvídeo lado a lado: {prev_dir / f'comparacao_seq{args.seq}.mp4'}")


if __name__ == "__main__":
    main()
