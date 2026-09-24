"""A/B reprodutivel de LoRA de video: mesmo plano, mesma seed, com e sem.

Pedido da auditoria de 2026-09-13. Os testes de LoRA ate aqui foram scripts avulsos no
scratchpad e nunca compararam com e sem na mesma seed -- por isso "o better-human-motion
nao fez nada" nem "o MSR ajudou" estao provados (MEMORIAL §3.84).

Para cada plano pedido de uma corrida existente, renderiza a VARIANTE BASE (sem LoRA/IC)
e uma variante por configuracao, todas com o mesmo still, prompt, audio, seed e variante
do LTX. Depois monta uma folha lado a lado (quadros a 10/50/90%) e mede, sem julgar:
  - sync (SyncNet LSE-C) nos planos de fala;
  - identidade vs still (ArcFace) e corte interno (guide_leak_audit).
O veredito continua sendo do usuario, vendo a folha e os clipes.

CLI:
  python -m script_pipeline.lora_ab --run-dir DIR --shots 1,3 \
      --config "bhm:better-human-motion:0.6" --config "msr:ic=msr" \
      [--ltx-variant distilled] [--seed 1234]

Formato de --config: "rotulo:chave[:forca]" para LoRA comum (repetivel com "+", ex.
"combo:better-human-motion:0.6+cameraman-v2:0.8"), ou "rotulo:ic=msr|ingredients".
Saida: <run>/lora_ab/<rotulo>/shotNNN.mp4, <run>/lora_ab/folha_shotNNN.png, relatorio.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse_config(txt: str) -> dict:
    rotulo, resto = txt.split(":", 1)
    if resto.startswith("ic="):
        return {"rotulo": rotulo, "loras": [], "ic": resto[3:]}
    loras = []
    for parte in resto.split("+"):
        chave, _, forca = parte.partition(":")
        loras.append((chave, float(forca) if forca else None))
    return {"rotulo": rotulo, "loras": loras, "ic": None}


def _folha(videos: dict[str, Path], saida: Path) -> None:
    import cv2
    import numpy as np
    linhas = []
    for rotulo, v in videos.items():
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        quadros = []
        for f in (0.1, 0.5, 0.9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
            ok, fr = cap.read()
            fr = fr if ok else np.zeros((272, 480, 3), np.uint8)
            fr = cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1])))
            quadros.append(fr)
        cap.release()
        linha = np.hstack(quadros)
        cv2.putText(linha, rotulo, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        linhas.append(linha)
    largura = max(l.shape[1] for l in linhas)
    linhas = [np.pad(l, ((0, 0), (0, largura - l.shape[1]), (0, 0))) for l in linhas]
    cv2.imwrite(str(saida), np.vstack(linhas))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shots", required=True, help="indices dos planos, ex. 1,3")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--ltx-variant", default="distilled",
                    help="distilled por padrao: no w4a8 o LoRA e requantizado (MEMORIAL 3.78)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    args = ap.parse_args(argv)

    import shutil
    run = Path(args.run_dir).resolve()
    base_dir = run / "lora_ab"
    configs = [{"rotulo": "base", "loras": [], "ic": None}] + [_parse_config(c) for c in args.config]
    env = os.environ.copy()
    env["LTX25_VARIANT"] = args.ltx_variant
    env.setdefault("LTX_COMFY_EXTRA_ARGS", "--disable-dynamic-vram")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    for cfg in configs:
        # Cada variante e uma COPIA rasa da corrida (parse/dialogue/stills por link), para o
        # cache de clipe de uma nao servir a outra e a corrida original nao ser tocada.
        destino = base_dir / cfg["rotulo"]
        (destino / "shots" / "clips").mkdir(parents=True, exist_ok=True)
        for sub in ("parse", "dialogue", "characters"):
            if (run / sub).exists() and not (destino / sub).exists():
                shutil.copytree(run / sub, destino / sub)
        if not (destino / "shots" / "stills").exists():
            shutil.copytree(run / "shots" / "stills", destino / "shots" / "stills")
        cmd = [sys.executable, "-u", "-m", "script_pipeline.render_shots_stage", "--run-dir", str(destino),
               "--videos-only", "--only-shots", args.shots, "--width", str(args.width),
               "--height", str(args.height), "--seed", str(args.seed)]
        for chave, forca in cfg["loras"]:
            cmd += ["--video-lora", f"{chave}:{forca}" if forca is not None else chave]
        if cfg["ic"]:
            cmd += ["--ic-reference", cfg["ic"]]
        print(f"=== [{cfg['rotulo']}] {' '.join(cmd[4:])}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env)

    from script_pipeline.guide_leak_audit import audit_clip
    from script_pipeline.syncnet_audit import syncnet_score
    from script_pipeline.consistency_audit import face_embedding
    import numpy as np
    import cv2
    plano = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))["shots"]
    falas = {}
    if (run / "dialogue" / "lines.json").exists():
        for e in json.loads((run / "dialogue" / "lines.json").read_text(encoding="utf-8")):
            if e.get("ok"):
                falas[(e["scene_index"], e["line_index"])] = e["audio_path"]
    relatorio = {}
    for i in [int(x) for x in args.shots.split(",") if x.strip()]:
        videos = {c["rotulo"]: base_dir / c["rotulo"] / "shots" / "clips" / f"shot{i:03d}.mp4" for c in configs}
        videos = {k: v for k, v in videos.items() if v.exists()}
        if not videos:
            continue
        _folha(videos, base_dir / f"folha_shot{i:03d}.png")
        from script_pipeline.render_shots import still_candidates
        still = next(iter(still_candidates(run / "shots" / "stills", i)), None)
        e_still = face_embedding(str(still)) if still else None
        wav = falas.get((plano[i].get("scene"), plano[i].get("line_index")))
        relatorio[f"shot{i:03d}"] = {}
        for rotulo, v in videos.items():
            m = {"corte": audit_clip(str(v))}
            if wav:
                m["syncnet"] = syncnet_score(str(v), wav)
            if e_still is not None:
                cap = cv2.VideoCapture(str(v))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 2) / 2))
                ok, fr = cap.read()
                cap.release()
                if ok:
                    tmp = base_dir / f"_meio_{rotulo}_{i:03d}.png"
                    cv2.imwrite(str(tmp), fr)
                    e = face_embedding(str(tmp))
                    tmp.unlink(missing_ok=True)
                    m["identidade_vs_still"] = None if e is None else round(float(np.dot(e, e_still)), 3)
            relatorio[f"shot{i:03d}"][rotulo] = m
            print(f"shot{i:03d} [{rotulo}]: {json.dumps(m, ensure_ascii=False)}", flush=True)
    (base_dir / "relatorio.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"folhas e relatorio em {base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
