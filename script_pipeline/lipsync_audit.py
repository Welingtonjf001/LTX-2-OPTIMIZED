"""Auditoria de qualidade do lip-sync aplicado -- pedido do usuario 2026-09-09
depois de assistir ao filme "Palacio de Esmeralda" e perguntar se a
sincronizacao era auditada. Ate aqui, `lipsync_scenes.py` so verificava se o
processo TECNICO rodou sem excecao (fallback LatentSync -> Wav2Lip quando um
dos dois falha) -- nunca se o resultado realmente sincronizou.

METODO: proxy leve, sem depender de landmarks faciais completos (a maquina
ja tem insightface/buffalo_l instalado para `consistency_audit.py`, que so
devolve o BBOX do rosto, nao pontos da boca). Recorta o TERCO INFERIOR do
bbox do maior rosto detectado por quadro amostrado (regiao boca/queixo) e
mede o MOVIMENTO frame-a-frame ali (diferenca media de pixel em escala de
cinza). Correlaciona esse sinal de movimento com o ENVELOPE RMS do audio,
os dois reamostrados pro mesmo eixo de tempo.

Intuicao: um lip-sync real faz a boca se mexer quando ha som e ficar quieta
no silencio -- correlacao ALTA (perto de +1). Um clipe cuja boca se move sem
relacao com a fala (sync ruim, ou o video original sem sync nenhum usado
como fallback) mede correlacao BAIXA ou proxima de zero. NAO e um detector
de "a fala X foi pronunciada certo" -- e um proxy de correlacao temporal
movimento-som, mensuravel sem reconhecimento de fala nem landmarks completos.

Roda por padrao dentro de `lipsync_scenes.py` (so nos clipes que TEM audio
de fala) e grava o relatorio em `lipsync/lipsync_audit.json` -- nunca
bloqueia a corrida: uma falha ou score baixo so fica registrado no log e no
relatorio, para revisao humana, do mesmo jeito que `consistency_audit.py` ja
faz para consistencia facial dos stills.

CLI standalone, para auditar um clipe avulso:
    python -m script_pipeline.lipsync_audit --video CLIPE.mp4 --audio FALA.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _mouth_motion_signal(video_path: str, sample_every: int = 2) -> tuple[np.ndarray, float, int]:
    """(sinal de movimento por quadro amostrado, fps efetivo da amostragem,
    quantos quadros amostrados tinham rosto detectavel)."""
    import cv2

    from script_pipeline.consistency_audit import _get_app

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([], dtype=np.float32), 0.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    app = _get_app()

    prev_crop = None
    motions: list[float] = []
    idx = 0
    face_hits = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            faces = app.get(frame)
            if faces:
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
                h = y2 - y1
                # Terco inferior do rosto (boca/queixo) -- recorte generoso,
                # nao exige landmark exato da boca.
                my1 = y1 + int(h * 0.6)
                crop = frame[max(0, my1):y2, max(0, x1):x2]
                if crop.size:
                    crop = cv2.resize(crop, (64, 64))
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if prev_crop is not None:
                        motions.append(float(np.abs(crop_gray - prev_crop).mean()))
                    prev_crop = crop_gray
                    face_hits += 1
        idx += 1
    cap.release()
    sampled_fps = fps / sample_every if motions else 0.0
    return np.array(motions, dtype=np.float32), sampled_fps, face_hits


def _audio_rms_envelope(audio_path: str, n_frames: int, frame_rate: float) -> np.ndarray:
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if n_frames <= 0 or len(y) == 0 or frame_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    hop = max(1, int(sr / frame_rate))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if len(rms) == 0:
        return np.zeros(n_frames, dtype=np.float32)
    x_old = np.linspace(0, 1, len(rms))
    x_new = np.linspace(0, 1, n_frames)
    return np.interp(x_new, x_old, rms).astype(np.float32)


def audit_lipsync(video_path: str, audio_path: str, *, log=print) -> dict:
    """Devolve {"score": float|None, "faces_detected": int, "frames_sampled": int,
    "motivo": str|None}. `score` e None quando nao da pra medir (rosto
    raramente detectavel, audio/video vazios) -- NAO alega falha de sync
    nesse caso, so reporta que nao pode medir, mesma filosofia de
    `consistency_audit.check_consistency` para stills sem rosto."""
    try:
        motion, sampled_fps, face_hits = _mouth_motion_signal(video_path)
    except Exception as e:
        return {"score": None, "faces_detected": 0, "frames_sampled": 0,
                "motivo": f"erro lendo video: {type(e).__name__}: {e}"}
    if len(motion) < 5 or face_hits < 5:
        return {"score": None, "faces_detected": face_hits, "frames_sampled": len(motion),
                "motivo": "rosto detectavel em poucos quadros -- sem base para medir"}
    try:
        audio_env = _audio_rms_envelope(audio_path, len(motion), sampled_fps)
    except Exception as e:
        return {"score": None, "faces_detected": face_hits, "frames_sampled": len(motion),
                "motivo": f"erro lendo audio: {type(e).__name__}: {e}"}
    if len(audio_env) != len(motion) or audio_env.std() < 1e-6 or motion.std() < 1e-6:
        return {"score": None, "faces_detected": face_hits, "frames_sampled": len(motion),
                "motivo": "audio ou movimento sem variacao -- correlacao indefinida"}
    corr = float(np.corrcoef(motion, audio_env)[0, 1])
    if not np.isfinite(corr):
        return {"score": None, "faces_detected": face_hits, "frames_sampled": len(motion),
                "motivo": "correlacao nao numerica (nan/inf)"}
    return {"score": corr, "faces_detected": face_hits, "frames_sampled": len(motion), "motivo": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args(argv)
    resultado = audit_lipsync(args.video, args.audio)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
