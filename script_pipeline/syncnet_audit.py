"""Sincronia labial medida pelo SyncNet (LSE-C / LSE-D), o avaliador padrao da area.

Pedido da auditoria de 2026-09-13. O `lipsync_audit` correlaciona movimento do terco
inferior do rosto com o volume do audio: e um proxy, e mediu mal rosto a 3/4 e fala
curta com muita expressao -- foi o que deixou em aberto LongCat x LTX. O LatentSync ja
instalado traz o avaliador de verdade (`eval/syncnet`, `checkpoints/auxiliary/
syncnet_v2.model`, detector `sfd_face.pth`); este modulo so o chama no ambiente conda
dele, em subprocesso (o SyncNet precisa do torch/python_speech_features de la).

Leitura dos numeros (convencao do LatentSync/Wav2Lip):
  - `conf` (LSE-C): maior = melhor. Video real bem sincronizado fica ~6-9; abaixo de ~3
    e sincronia fraca.
  - `min_dist` (LSE-D): menor = melhor.
  - `av_offset`: deslocamento em quadros (25 fps) em que o audio casa melhor com a boca;
    |offset| > 2 indica audio adiantado/atrasado.

O video e remuxado a 25 fps com o wav da fala (o SyncNet le o audio do proprio mp4) num
diretorio temporario sem espacos -- o codigo do LatentSync monta comandos ffmpeg sem aspas.

CLI:  python -m script_pipeline.syncnet_audit --video CLIPE.mp4 [--audio FALA.wav]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATENTSYNC_ROOT = os.environ.get("LATENTSYNC_ROOT", r"E:\Users\home\Documents\LatentSync")
LATENTSYNC_PYTHON = os.environ.get("LATENTSYNC_PYTHON",
                                   os.path.join(LATENTSYNC_ROOT, ".conda_env", "python.exe"))
FFMPEG = os.environ.get("LTX_FFMPEG", "ffmpeg")
LIMIAR_CONF = 3.0

_HELPER = r'''
import json, os, sys, shutil
sys.path.insert(0, os.getcwd())
import torch
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
video, work = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = SyncNetEval(device=dev)
net.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
det_dir = os.path.join(work, "detect")
det = SyncNetDetector(device=dev, detect_results_dir=det_dir)
det(video_path=video, min_track=int(sys.argv[3]))
crops = sorted(os.listdir(os.path.join(det_dir, "crop"))) if os.path.isdir(os.path.join(det_dir, "crop")) else []
res = []
for c in crops:
    off, dist, conf = net.evaluate(video_path=os.path.join(det_dir, "crop", c), temp_dir=os.path.join(work, "tmp"))
    res.append({"av_offset": int(off), "min_dist": float(dist), "conf": float(conf)})
print("SYNCNET_JSON=" + json.dumps(res))
'''


def syncnet_score(video: str, audio: str | None = None, *, min_track: int = 25,
                  timeout: int = 900) -> dict:
    """{"conf", "min_dist", "av_offset", "tracks", "motivo"}; conf=None quando nao mediu."""
    vazio = {"conf": None, "min_dist": None, "av_offset": None, "tracks": 0}
    if not os.path.isfile(LATENTSYNC_PYTHON):
        return {**vazio, "motivo": f"python do LatentSync nao encontrado: {LATENTSYNC_PYTHON}"}
    work = tempfile.mkdtemp(prefix="syncnet_")
    try:
        entrada = os.path.join(work, "in.mp4")
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", video]
        if audio:
            cmd += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        cmd += ["-r", "25", "-c:v", "libx264", "-crf", "16", "-c:a", "aac", entrada]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(entrada):
            return {**vazio, "motivo": f"ffmpeg falhou: {r.stderr[-300:]}"}
        helper = os.path.join(work, "helper.py")
        Path(helper).write_text(_HELPER, encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        r = subprocess.run([LATENTSYNC_PYTHON, helper, entrada, work, str(min_track)],
                           cwd=LATENTSYNC_ROOT, capture_output=True, text=True, timeout=timeout,
                           env=env, encoding="utf-8", errors="replace")
        linha = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("SYNCNET_JSON=")), None)
        if linha is None:
            return {**vazio, "motivo": f"SyncNet falhou: {((r.stderr or '') + (r.stdout or ''))[-400:]}"}
        faixas = json.loads(linha.split("=", 1)[1])
        if not faixas:
            return {**vazio, "motivo": "nenhum rosto rastreado por tempo suficiente"}
        # Vale o rosto com MAIOR confianca: em plano com dois rostos, quem fala e o sincronizado.
        melhor = max(faixas, key=lambda f: f["conf"])
        return {**melhor, "tracks": len(faixas), "motivo": None}
    except subprocess.TimeoutExpired:
        return {**vazio, "motivo": f"SyncNet passou de {timeout}s"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None, help="wav da fala; sem ele usa o audio do proprio mp4")
    ap.add_argument("--min-track", type=int, default=25)
    args = ap.parse_args(argv)
    print(json.dumps(syncnet_score(args.video, args.audio, min_track=args.min_track), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
