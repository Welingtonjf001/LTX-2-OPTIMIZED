"""Wrapper fino sobre pose_video.py pras restricoes proprias do MiniMax H3:
contagem de quadros 17*n+5 (compressao temporal do VAE de video do H3) e
largura/altura multiplas de 32. O desenho do esqueleto (SMPL/SMPL-X -> COCO18,
projecao, cores OpenPose) e o MESMO de script_pipeline/pose_video.py -- o H3
Fun ControlNet Union foi treinado com mapas de pose no formato OpenPose
clássico (mesma convencao do DWPose/controlnet_aux), entao nao ha desenho
proprio aqui, so o ajuste de grade temporal/espacial.

O peso de controle (minimax_h3_fun_controlnet_union_pruned_*.safetensors) entra
no grafo do ComfyUI via ModelPatchLoader -> MiniMaxH3FunControlNetApply
(control_video), nao aqui -- este script so produz o control_video.mp4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from script_pipeline.pose_video import render_pose_video
except ImportError:
    # ACHADO 2026-09-17 (auditoria externa) #7: `python script_pipeline/
    # motion_to_h3_controlnet.py` (execucao direta, nao "-m") coloca so
    # script_pipeline/ no sys.path, nao a raiz do repo -- "script_pipeline"
    # nao e visivel como pacote de dentro dele mesmo. Insere a raiz (pai
    # deste arquivo) e tenta de novo; quem roda via "-m" nunca cai aqui.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from script_pipeline.pose_video import render_pose_video

H3_FPS = 24.0


def h3_frame_count(n_raw: int) -> int:
    """Replica EXATAMENTE a expressao do node 131 (ComfyMathExpression) do
    workflow oficial (`video_minimax_h3_r2v.json`):

        max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17

    NAO e o 17n+5 mais PROXIMO -- e o proximo 17n+5 >= n_raw (arredonda pra
    CIMA, minimo 5). ACHADO 2026-09-17 (auditoria externa): a versao antiga
    (`nearest_h3_frame_count`) arredondava pro mais proximo -- pra 108
    quadros pedidos dava 107, mas o node de verdade pede 124 pro mesmo `a`.
    O backend completa a diferenca REPETINDO O ULTIMO QUADRO
    (`MiniMaxH3FunControlPatch._fit_frames`), entao a pose ficava parada por
    ate ~17 quadros (~0,7s a 24fps) sem o adaptador saber disso -- silencioso,
    nao um erro."""
    n_raw = max(5, n_raw)
    return n_raw + (5 - (n_raw % 17)) % 17


def round_to_multiple_of_32(value: int) -> int:
    return max(32, round(value / 32) * 32)


def render_h3_control_video(joints: np.ndarray, out_path: str | Path, *, width: int, height: int,
                            duration_s: float | None = None, num_frames: int | None = None,
                            source_fps: float = 30.0, azimuth_deg: float = 0.0,
                            elevation_deg: float = 0.0) -> tuple[int, int, int]:
    """Retorna (n_frames, width_ajustado, height_ajustado) realmente escritos."""
    if num_frames is None:
        if duration_s is None:
            raise ValueError("Passe --duration ou --num-frames")
        desired = round(duration_s * H3_FPS)
    else:
        desired = num_frames
    h3_frames = h3_frame_count(desired)
    h3_width = round_to_multiple_of_32(width)
    h3_height = round_to_multiple_of_32(height)

    n = render_pose_video(joints, out_path, width=h3_width, height=h3_height,
                          fps=H3_FPS, source_fps=source_fps, num_frames=h3_frames,
                          azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    return n, h3_width, h3_height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joints", help=".npy (T,J,3) ou (T,A,J,3) -- mesma convencao do pose_video.py")
    ap.add_argument("out", help="control_video de saida (.mp4)")
    ap.add_argument("--width", type=int, required=True, help="arredondado para multiplo de 32")
    ap.add_argument("--height", type=int, required=True, help="arredondado para multiplo de 32")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", type=float, help="segundos; contagem de quadros arredonda pra CIMA pro proximo 17n+5 a 24fps (mesma formula do node 131 do workflow oficial)")
    group.add_argument("--num-frames", type=int, help="contagem desejada; arredondada pra CIMA pro proximo 17n+5")
    ap.add_argument("--source-fps", type=float, default=30.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--elevation", type=float, default=0.0)
    args = ap.parse_args()

    joints = np.load(args.joints)
    n, w, h = render_h3_control_video(
        joints, args.out, width=args.width, height=args.height,
        duration_s=args.duration, num_frames=args.num_frames,
        source_fps=args.source_fps, azimuth_deg=args.azimuth, elevation_deg=args.elevation)
    print(f"[motion_to_h3_controlnet] {n} quadros (17n+5) @ {w}x{h} @ {H3_FPS}fps -> {args.out}")


if __name__ == "__main__":
    main()
