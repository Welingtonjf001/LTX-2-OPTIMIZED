"""Identidade da FOTO + movimento do VIDEO -> sequencia de malhas SMPL-X.

    conda run -n mhr python animar_corpo.py <smplx_params.npz> <hmr4d_results.pt> <saida> [--frames N]

Por que isto funciona sem rigging nenhum: o SMPL-X e' um modelo PARAMETRICO --
os pesos de skinning fazem parte do modelo. Dar `betas` (identidade) e
`body_pose` (pose por frame) ja devolve a malha deformada. Nao ha osso a
amarrar, nao ha transferencia de peso, nao ha IK.

As duas pontas ja falam a mesma lingua:
- `mhr_to_smplx.py` (foto -> SAM 3D Body -> MHR -> SMPL-X) devolve `betas`;
- o GVHMR grava `smpl_params_global` com `body_pose` e `global_orient` por frame.

Trocar so os `betas` mantendo a pose = o corpo da pessoa da foto executando a
danca do video.

ATENCAO: rode no ambiente conda `mhr` com PYTHONNOUSERSITE=1 (ver mhr_to_smplx.py).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

SMPLX_DIR = Path(os.environ.get(
    "SMPLX_DIR", r"E:\Users\home\Documents\GVHMR\inputs\checkpoints\body_models"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("betas_npz", help="smplx_params.npz vindo de mhr_to_smplx.py")
    ap.add_argument("motion_pt", help="hmr4d_results.pt do GVHMR")
    ap.add_argument("saida")
    ap.add_argument("--frames", type=int, default=8, help="quantos frames amostrar")
    a = ap.parse_args()

    import smplx
    import trimesh

    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)

    ident = np.load(a.betas_npz)
    betas = torch.as_tensor(ident["betas"], dtype=torch.float32)   # [1, 10]
    print(f"identidade da foto: betas {tuple(betas.shape)}")

    res = torch.load(a.motion_pt, map_location="cpu", weights_only=False)
    mov = res["smpl_params_global"]
    T = mov["body_pose"].shape[0]
    idx = np.linspace(0, T - 1, min(a.frames, T)).round().astype(int)
    print(f"movimento do video: {T} frames, amostrando {len(idx)}")

    # `flat_hand_mean` tem que casar com o que o GVHMR usou (False em
    # bridge/export_3d.py); divergir aqui deforma as maos em silencio.
    modelo = smplx.create(
        model_path=str(SMPLX_DIR), model_type="smplx", gender="neutral",
        use_pca=False, flat_hand_mean=False, num_betas=10, batch_size=len(idx),
    )

    saida = modelo(
        betas=betas.expand(len(idx), -1),                      # identidade FIXA
        global_orient=mov["global_orient"][idx].float(),       # movimento do video
        body_pose=mov["body_pose"][idx].float(),
        transl=mov["transl"][idx].float() if "transl" in mov else None,
    )
    V = saida.vertices.detach().cpu().numpy()
    print(f"malhas geradas: {V.shape}  (frames, vertices, 3)")

    faces = modelo.faces
    for n, i in enumerate(idx):
        trimesh.Trimesh(V[n], faces, process=False).export(str(out / f"frame_{n:03d}.ply"))

    # prova de que ANIMA: deslocamento medio entre frames consecutivos
    desloc = [float(np.linalg.norm(V[k + 1] - V[k], axis=1).mean()) for k in range(len(V) - 1)]
    np.savez_compressed(out / "vertices.npz", vertices=V, faces=faces, frames=idx)
    rel = {
        "frames_amostrados": idx.tolist(),
        "vertices_por_frame": int(V.shape[1]),
        "deslocamento_medio_entre_frames_m": [round(d, 4) for d in desloc],
    }
    (out / "animacao.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print("deslocamento medio por frame (m):", [round(d, 3) for d in desloc])
    print("pronto ->", out)


if __name__ == "__main__":
    main()
