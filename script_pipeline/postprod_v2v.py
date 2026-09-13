"""Estagio [7b], opcional: pos-producao por IC-LoRA V2V sobre os clipes ja mixados.

Roda DEPOIS do mix e ANTES da montagem: o audio de cada clipe ja esta pronto e e so
recolocado sobre a imagem refeita (ltx25_backend.generate_v2v, audio="remux"), com a
mesma duracao -- o sincronismo nao tem como mudar aqui.

  --deblur    Lightricks Deblur 2.5: recupera nitidez de take com desfoque/derretimento.
  --upscale   Lightricks Pixel-Spatial-Upscaler 2.5: re-renderiza a x2.

Upscale e tudo ou nada: o filme precisa de UMA resolucao. Clipe que falhar no upscale
generativo sobe x2 por lanczos no ffmpeg, para a montagem nao misturar tamanhos.

CLI: python -m script_pipeline.postprod_v2v --run-dir DIR [--deblur] [--upscale]
Le e reescreve <run>/intermediate/mixed_clips.json (copia em mixed_clips_pre_post.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASSOS = {
    # nome: (chave do catalogo, escala, prompt de apoio)
    "deblur": ("deblur-2.5", 1, "sharp, in focus, clean detailed footage"),
    "upscale": ("pixel-upscaler-2.5", 2, "high resolution, fine detail"),
}


def _prompt_do_plano(run_dir: Path, clip_id: str) -> str:
    try:
        plan = json.loads((run_dir / "parse" / "shot_plan.json").read_text(encoding="utf-8"))["shots"]
        return plan[int(clip_id.rsplit("shot", 1)[1])]["video_prompt"]
    except (OSError, ValueError, KeyError, IndexError):
        return ""


def _lanczos_x2(src: str, dest: Path) -> str | None:
    import os
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", src, "-vf", "scale=iw*2:ih*2:flags=lanczos",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dest)],
                       capture_output=True, text=True)
    return str(dest) if r.returncode == 0 and dest.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--deblur", action="store_true")
    ap.add_argument("--deblur-strength", type=float, default=1.0)
    ap.add_argument("--deblur-guide-strength", type=float, default=1.0)
    ap.add_argument("--upscale", action="store_true")
    ap.add_argument("--upscale-strength", type=float, default=1.0)
    ap.add_argument("--upscale-guide-strength", type=float, default=1.0)
    ap.add_argument("--only", default=None, help="ids de clipe separados por virgula (teste)")
    args = ap.parse_args(argv)

    import ltx25_backend
    import ltx_loras
    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    work = run_folder.subdir(run_dir, "intermediate")
    manifest_path = work / "mixed_clips.json"
    log = lambda m: run_folder.append_log(run_dir, m)  # noqa: E731
    pedidos = [(n, getattr(args, f"{n}_strength"), getattr(args, f"{n}_guide_strength"))
               for n in PASSOS if getattr(args, n)]
    if not pedidos:
        log("postprod_v2v: nenhum passo pedido (--deblur/--upscale); nada a fazer.")
        return 0
    for nome, _, _ in pedidos:
        local = ltx_loras.BY_KEY[PASSOS[nome][0]].local
        if ltx_loras.installed_path(local) is None:
            log(f"postprod_v2v: LoRA de {nome} nao instalado -- python ltx_loras.py download --set gated")
            return 1

    backup = work / "mixed_clips_pre_post.json"
    if not backup.exists():
        shutil.copy2(manifest_path, backup)
    clips = json.loads(backup.read_text(encoding="utf-8"))  # sempre parte do mix, nunca de um post anterior
    so = set(args.only.split(",")) if args.only else None

    falhas = 0
    for clip in clips:
        fonte = clip.get("mixed_video_path")
        if not fonte or (so and clip["id"] not in so):
            continue
        atual = fonte
        prompt = _prompt_do_plano(run_dir, clip["id"])
        for nome, forca, guia in pedidos:
            chave, escala, apoio = PASSOS[nome]
            destino = work / f"{clip['id']}_{nome}.mp4"
            marca = destino.with_suffix(".key")
            assinatura = f"{Path(atual).name}|{forca:g}|{guia:g}"
            if destino.exists() and marca.exists() and marca.read_text(encoding="utf-8") == assinatura:
                log(f"{clip['id']}: {nome} ja feito com a mesma forca, reaproveitando")
                atual = str(destino)
                continue
            try:
                ltx25_backend.generate_v2v(
                    f"{prompt} {apoio}".strip(), atual, str(destino),
                    lora=ltx_loras.BY_KEY[chave].local, strength=forca, guide_strength=guia,
                    audio="remux", scale=escala, log_cb=lambda m: log(f"    [{nome}] {m}"))
                marca.write_text(assinatura, encoding="utf-8")
                log(f"{clip['id']}: {nome} ok (forca {forca:g}, guia {guia:g}) -> {destino.name}")
                atual = str(destino)
            except Exception as e:
                falhas += 1
                log(f"{clip['id']}: {nome} FALHOU ({type(e).__name__}: {str(e)[:200]})")
                if nome == "upscale":
                    reserva = _lanczos_x2(atual, destino)
                    if reserva:
                        log(f"{clip['id']}: upscale por lanczos x2 para manter a resolucao do filme")
                        atual = reserva
        clip["mixed_video_path"] = atual

    manifest_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"postprod_v2v: {', '.join(n for n, _, _ in pedidos)} aplicado(s); {falhas} falha(s).")
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
