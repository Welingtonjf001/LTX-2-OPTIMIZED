"""Renderiza esqueletos 3D (InterGen/Inter-X) como video estilo OpenPose (COCO-18)
sobre fundo preto -- o formato que os IC-LoRA/ControlNet de pose costumam esperar
(LTX Union-Control e o MiniMax H3 Fun ControlNet Union usam a mesma convencao
visual: linhas coloridas por membro, sem eixos/legenda, fundo preto).

Fonte unica para os dois motores: nenhum dos dois exige um preprocessador de pose
de verdade (DWPose/OpenPose) porque JA TEMOS as juntas 3D (saida do InterGen ou do
Inter-X, ver utils/hhi_visualization.py) -- so falta projetar numa camera e desenhar.

Os primeiros 22 joints do InterGen (SMPL, HumanML3D) e do Inter-X (SMPL-X, so usa
os 22 primeiros dos 55) tem a MESMA ordem (SMPL-X e SMPL + maos/rosto anexados) --
por isso um so mapeamento SMPL_TO_COCO18 serve para as duas fontes.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

# Ordem SMPL/SMPL-X (primeiros 22 joints, identica nas duas fontes):
# 0 pelvis, 1 l_hip, 2 r_hip, 3 spine1, 4 l_knee, 5 r_knee, 6 spine2, 7 l_ankle,
# 8 r_ankle, 9 spine3, 10 l_foot, 11 r_foot, 12 neck, 13 l_collar, 14 r_collar,
# 15 head, 16 l_shoulder, 17 r_shoulder, 18 l_elbow, 19 r_elbow, 20 l_wrist, 21 r_wrist

# COCO-18 (formato classico do OpenPose/controlnet_aux draw_bodypose):
# 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow, 7 LWrist,
# 8 RHip, 9 RKnee, 10 RAnkle, 11 LHip, 12 LKnee, 13 LAnkle,
# 14 REye, 15 LEye, 16 REar, 17 LEar
# Olhos/orelhas do SMPL-X (indices 23/24) ficam FORA dos 22 joints do corpo
# base -- o InterGen (SMPL puro, 22 joints) nao os tem. Para o mapeamento
# funcionar com as duas fontes, olhos/orelhas reusam a cabeca (15): limbo de
# comprimento zero, nao aparece desenhado -- perde precisao de rosto, mas o
# alvo aqui e guia de CORPO, nao rosto (ver docstring do modulo).
SMPL_TO_COCO18 = [15, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 15, 15, 15, 15]

COCO18_EDGES = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Paleta padrao OpenPose (BODY_25/COCO18), UMA COR POR JUNTA (18 entradas) --
# e a mesma tabela que o controlnet_aux/DWPose usa (`util.py::draw_bodypose`),
# por isso e reaproveitada tambem no adaptador do MiniMax H3
# (motion_to_h3_controlnet.py so muda resolucao/fps/contagem de quadros, chama
# esta mesma funcao de desenho). Valores em RGB (a convencao publicada) --
# `_rgb_to_bgr_tuple` converte na hora de desenhar, porque cv2 espera BGR.
# Os primeiros 17 membros (COCO18_EDGES) reusam `COCO18_COLORS[limb_idx]`, a
# MESMA indexacao da referencia.
COCO18_COLORS = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
], dtype=np.uint8)


def _rgb_to_bgr_tuple(rgb) -> tuple[int, int, int]:
    """cv2 desenha em BGR; COCO18_COLORS guarda RGB (a convencao publicada
    do OpenPose/controlnet_aux) -- ACHADO 2026-09-17 (auditoria externa):
    passar a cor RGB direto pro cv2 sem inverter fazia um segmento
    pescoco-ombro-direito, [255,0,0]=vermelho na tabela, sair AZUL no video
    (o buffer e escrito como bgr24 pro ffmpeg)."""
    r, g, b = (int(c) for c in rgb)
    return (b, g, r)


def smpl_to_coco18(joints_smpl: np.ndarray) -> np.ndarray:
    """(..., >=22, 3) -> (..., 18, 3). So usa os 22 primeiros joints do corpo."""
    if joints_smpl.shape[-2] < 22:
        raise ValueError(f"Esperado >=22 joints (SMPL/SMPL-X), recebido {joints_smpl.shape[-2]}")
    return joints_smpl[..., SMPL_TO_COCO18, :]


def project_orthographic(points3d: np.ndarray, azimuth_deg: float = 0.0,
                         elevation_deg: float = 0.0) -> np.ndarray:
    """(...,3) mundo Y-up -> (...,2) camera, ortografica. Roda em Y (azimute)
    e depois em X (elevacao) antes de descartar Z. Convencao: x direita, y CIMA
    em coordenadas de camera (inverta o sinal soh na hora de rasterizar)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    x, y, z = points3d[..., 0], points3d[..., 1], points3d[..., 2]
    xr = x * np.cos(az) - z * np.sin(az)
    zr = x * np.sin(az) + z * np.cos(az)
    yr = y * np.cos(el) - zr * np.sin(el)
    return np.stack((xr, yr), axis=-1)


def fit_transform(points2d: np.ndarray, width: int, height: int, margin: float = 0.12):
    """Bounding box FIXO (calculado uma vez sobre todos os quadros/atores) para
    a cena inteira nao 're-enquadrar' a cada quadro -- controle de pose com
    camera "pulando" de escala quebraria a coerencia temporal do ControlNet."""
    finite = points2d[np.isfinite(points2d).all(axis=-1)]
    if finite.size == 0:
        raise ValueError("Nenhum ponto valido para enquadrar")
    low, high = finite.min(axis=0), finite.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    usable_w, usable_h = width * (1 - 2 * margin), height * (1 - 2 * margin)
    scale = min(usable_w / span[0], usable_h / span[1])
    center = (low + high) / 2

    def to_pixels(p2d: np.ndarray) -> np.ndarray:
        px = (p2d[..., 0] - center[0]) * scale + width / 2
        py = height / 2 - (p2d[..., 1] - center[1]) * scale  # y da imagem cresce p/ baixo
        return np.stack((px, py), axis=-1)

    return to_pixels


def resample_time(joints: np.ndarray, source_fps: float, target_fps: float,
                   target_frames: int | None = None) -> np.ndarray:
    """(actors, T, J, 3) -> (actors, T', J, 3), interpolacao linear de posicao
    preservando a DURACAO (T'/target_fps ~= T/source_fps), a menos que
    target_frames seja passado explicitamente."""
    a, t = joints.shape[0], joints.shape[1]
    if target_frames is None:
        target_frames = max(1, round(t * target_fps / source_fps))
    if t == 1:
        return np.repeat(joints, target_frames, axis=1)
    src_idx = np.linspace(0, t - 1, target_frames)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    frac = (src_idx - lo).reshape(1, -1, 1, 1)
    return joints[:, lo] * (1 - frac) + joints[:, hi] * frac


def draw_coco18_frame(canvas: np.ndarray, points2d_per_actor: list[np.ndarray],
                      thickness: int = 4, radius: int = 5) -> None:
    """Desenha N atores (cada um (18,2) em pixels) sobre `canvas` (H,W,3) uint8.
    Requer cv2 -- import local para nao forcar a dependencia em quem so usa o
    resto do modulo (resample/projecao) em ambiente sem opencv."""
    import cv2

    for points in points2d_per_actor:
        valid = np.isfinite(points).all(axis=-1)
        for limb_idx, (i, j) in enumerate(COCO18_EDGES):
            if not (valid[i] and valid[j]):
                continue
            p1, p2 = points[i], points[j]
            if np.allclose(p1, p2):
                continue  # limbo degenerado (ex.: orelha == olho no SMPL-X)
            color = _rgb_to_bgr_tuple(COCO18_COLORS[limb_idx])
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     color, thickness, cv2.LINE_AA)
        for k in range(18):
            if not valid[k]:
                continue
            # cor por junta (nao branco fixo) -- mesma convencao da referencia
            # OpenPose/controlnet_aux, ver COCO18_COLORS.
            cv2.circle(canvas, tuple(points[k].astype(int)), radius,
                      _rgb_to_bgr_tuple(COCO18_COLORS[k]), -1, cv2.LINE_AA)


def render_pose_video(joints: np.ndarray, out_path: str | Path, *, width: int, height: int,
                      fps: float = 24.0, source_fps: float = 30.0,
                      num_frames: int | None = None, azimuth_deg: float = 0.0,
                      elevation_deg: float = 0.0, margin: float = 0.12) -> int:
    """joints: (T,J,3) um ator OU (T,A,J,3) A atores -- MESMA convencao de
    `hhi_visualization.recover_hhi` (tempo primeiro, depois ator). Nao hah
    deteccao automatica entre (T,A,J,3) e (A,T,J,3): adivinhar pela magnitude
    dos eixos falha em clipes curtos (ex.: T=3,A=2 e indistinguivel de T=2,A=3).
    J>=22 assume SMPL/SMPL-X e mapeia pra COCO18; J==18 usa direto. Retorna o
    numero de quadros escritos."""
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError(f"fps precisa ser finito e positivo, recebido {fps}")
    if not (np.isfinite(source_fps) and source_fps > 0):
        raise ValueError(f"source_fps precisa ser finito e positivo, recebido {source_fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height precisam ser positivos, recebido {width}x{height}")
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 3:
        joints = joints[:, None]  # (T,J,3) -> (T,1,J,3)
    if joints.ndim != 4:
        raise ValueError(f"Esperado joints com 3 ou 4 dims, recebido shape {joints.shape}")
    if joints.shape[-1] != 3:
        raise ValueError(f"Ultima dimensao precisa ser 3 (x,y,z), recebido {joints.shape[-1]}")
    if joints.shape[0] == 0:
        raise ValueError("joints com T=0 (nenhum quadro)")
    actors = joints.transpose(1, 0, 2, 3)  # (T,A,J,3) -> (A,T,J,3)

    if actors.shape[-2] == 18:
        coco = actors
    else:
        coco = smpl_to_coco18(actors)

    coco = resample_time(coco, source_fps=source_fps, target_fps=fps, target_frames=num_frames)
    n_actors, n_frames = coco.shape[0], coco.shape[1]

    points2d_all = project_orthographic(coco, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    to_pixels = fit_transform(points2d_all.reshape(-1, 2), width, height, margin=margin)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(out_path),
    ]
    # stderr CAPTURADO (nao DEVNULL) -- diagnostico de dimensao/encoder
    # invalidos so aparece ali; descartar deixava "ffmpeg falhou (codigo N)"
    # sem pista nenhuma do motivo (achado da auditoria externa, risco
    # adicional listado junto aos achados P1/P2).
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        for f in range(n_frames):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            per_actor = [to_pixels(points2d_all[a, f]) for a in range(n_actors)]
            draw_coco18_frame(canvas, per_actor)
            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
        _, stderr = proc.communicate()
    if proc.returncode != 0:
        detalhe = stderr.decode("utf-8", errors="replace")[-2000:] if stderr else ""
        raise RuntimeError(f"ffmpeg falhou (codigo {proc.returncode}) gerando {out_path}:\n{detalhe}")
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joints", help=".npy com juntas 3D: (T,J,3) um ator ou (T,A,J,3) A atores, J=18/22/55")
    ap.add_argument("out", help="video de saida (.mp4)")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--source-fps", type=float, default=30.0,
                    help="fps da fonte (InterGen ~20, Inter-X/hhi ~30 -- conferir motion_process.py/train_comp_v6.py)")
    ap.add_argument("--num-frames", type=int, default=None, help="forca contagem de quadros (senao preserva duracao)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="graus, rotacao da camera em torno do eixo Y")
    ap.add_argument("--elevation", type=float, default=0.0, help="graus, rotacao da camera em torno do eixo X")
    args = ap.parse_args()

    joints = np.load(args.joints)
    n = render_pose_video(joints, args.out, width=args.width, height=args.height,
                          fps=args.fps, source_fps=args.source_fps,
                          num_frames=args.num_frames, azimuth_deg=args.azimuth,
                          elevation_deg=args.elevation)
    print(f"[pose_video] {n} quadros -> {args.out}")


if __name__ == "__main__":
    main()
