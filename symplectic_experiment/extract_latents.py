"""Passo 1 do teste em vídeo real: latentes do VAE do LTX 2.5 para os clipes do
professor (vídeos já gerados pelo LTX 2.5 neste repositório).

Formato MEDIDO no ida e volta (vae_roundtrip.py, 2026-09-13): um clipe de 121
quadros 960x544 vira latente (1, 128, 16, 17, 30) -- 128 canais, compressão
temporal 8, espacial 32. Reconstrução com PSNR médio 31,2 dB, que é a RÉGUA de
qualquer preditor avaliado pelo decoder congelado.

Decisões de dados:
- O 1º quadro latente é DESCARTADO da sequência de dinâmica. O VAE é causal: ele
  codifica o primeiro quadro real sozinho, e os demais comprimem 8 quadros cada.
  A estatística é outra, e tratá-lo como um estado sucessivo misturaria dois
  processos. Ele é salvo à parte (`primeiro`) caso seja útil depois.
- Cada latente leva o nome da CORRIDA de origem (`grupo`). Muitos clipes são o
  mesmo plano regerado em corridas diferentes; separar treino e teste por
  corrida evita que um lado veja o gêmeo do outro.
- fp16 em disco; retoma de onde parou (pula o que já existe).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def ler_video(caminho: Path):
    import av
    with av.open(str(caminho)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def grupo_da_corrida(rel: Path) -> str:
    partes = rel.parts
    if partes and partes[0] == "outputs" and len(partes) > 2:     # outputs/decupagem/<corrida>/...
        return partes[2]
    if partes and partes[0] == "outputs_25" and len(partes) > 1:  # outputs_25/<teste>/...
        return partes[1]
    return partes[0] if partes else "desconhecido"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--clips-dir", required=True, help="raiz dos clipes (caminhos relativos preservados)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limite", type=int, default=None, help="só os N primeiros (teste)")
    args = ap.parse_args()

    raiz = Path(args.clips_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifesto = out / "manifesto.jsonl"

    sys.argv = [sys.argv[0]]   # comfy.cli_args lê sys.argv na importação
    sys.path.insert(0, args.comfy)
    import torch
    import comfy.sd
    import comfy.utils

    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    del sd

    clipes = sorted(raiz.rglob("*.mp4"))
    if args.limite:
        clipes = clipes[: args.limite]
    feitos = {json.loads(l)["rel"] for l in manifesto.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if manifesto.exists() else set()
    print(f"{len(clipes)} clipes, {len(feitos)} já extraídos", flush=True)

    t_total = time.time()
    for i, clipe in enumerate(clipes):
        rel = clipe.relative_to(raiz)
        if str(rel) in feitos:
            continue
        try:
            quadros = ler_video(clipe)
            t = 1 + 8 * ((len(quadros) - 1) // 8)
            quadros = quadros[:t]
            h, w = quadros.shape[1:3]
            if h % 32 or w % 32 or t < 17:
                print(f"  [{i + 1}] pulado {rel}: {len(quadros)}q {w}x{h}", flush=True)
                continue
            pixels = torch.from_numpy(quadros).float() / 255.0
            t0 = time.time()
            with torch.inference_mode():   # fora dele o VAE em blocos quebra (ver vae_roundtrip.py)
                lat = vae.encode(pixels)[0]  # [128, T, H, W]
            dt = time.time() - t0
            nome = str(rel).replace("/", "__").replace("\\", "__").removesuffix(".mp4") + ".pt"
            torch.save({"primeiro": lat[:, :1].half().clone(),
                        "sequencia": lat[:, 1:].half().clone()}, out / nome)
            reg = {"rel": str(rel), "arquivo": nome, "grupo": grupo_da_corrida(rel),
                   "quadros": int(t), "latente": list(lat.shape), "passos_dinamica": int(lat.shape[1] - 1),
                   "encode_s": round(dt, 2)}
            with manifesto.open("a", encoding="utf-8") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            print(f"  [{i + 1}/{len(clipes)}] {reg['grupo']} {rel.name}: {t}q -> {tuple(lat.shape)} "
                  f"em {dt:.1f}s", flush=True)
        except Exception as e:  # um clipe ruim não pode derrubar os outros 192
            print(f"  [{i + 1}] FALHOU {rel}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)

    regs = [json.loads(l) for l in manifesto.read_text(encoding="utf-8").splitlines() if l.strip()]
    grupos = {}
    for r in regs:
        grupos.setdefault(r["grupo"], 0)
        grupos[r["grupo"]] += r["passos_dinamica"]
    print(f"\n{len(regs)} latentes | {sum(r['passos_dinamica'] for r in regs)} passos de dinâmica "
          f"| {len(grupos)} corridas | {time.time() - t_total:.0f}s", flush=True)
    for g, n in sorted(grupos.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>5} passos  {g}")


if __name__ == "__main__":
    main()
