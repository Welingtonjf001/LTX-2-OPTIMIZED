"""Ponte: saida do SAM 3D Body (MHR) -> parametros SMPL-X.

    conda run -n mhr python mhr_to_smplx.py <entrada.npz | pasta> <saida>

IMPORTANTE -- este script roda no ambiente conda `mhr`, NAO no Python global,
e exige `PYTHONNOUSERSITE=1`. Os dois ambientes sao Python 3.12 e compartilham
o mesmo user site; sem a variavel, o env importa o torch 2.4 do global em vez do
proprio 2.13, e o `pymomentum` foi compilado contra o 2.13. A falha e' silenciosa.

Fonte dos modelos:
- MHR: assets do repo `facebookresearch/MHR` (Apache 2.0)
- SMPL-X: os arquivos ja instalados pelo GVHMR. Sao de licenca ACADEMICA (MPI),
  entao o resultado desta conversao herda essa restricao -- ver LICENCAS.md.
  O rig MHR cru, sem passar por aqui, nao tem esse problema.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LTX = Path(__file__).resolve().parent
MHR_REPO = LTX / "MHR"
CONV = MHR_REPO / "tools" / "mhr_smpl_conversion"
SMPLX_DIR = Path(os.environ.get(
    "SMPLX_DIR", r"E:\Users\home\Documents\GVHMR\inputs\checkpoints\body_models"))

sys.path.insert(0, str(MHR_REPO))
sys.path.insert(0, str(CONV))


def main(entrada: str, saida: str) -> None:
    import smplx
    from conversion import Conversion
    from mhr.mhr import MHR

    # a build de pymomentum instalada e' CPU; forcar cuda aqui so falharia.
    # Tem que ser torch.device, nao string -- o MHR._create_model le `device.type`.
    device = torch.device("cpu")
    # O conversor referencia seus assets por caminho RELATIVO
    # ('./assets/subsampled_vertex_indices.npy'), entao so funciona com o
    # diretorio de trabalho dentro dele. Entrada e saida ja foram resolvidas
    # para absoluto antes da troca.
    origem = Path(entrada).resolve()
    saida = str(Path(saida).resolve())
    os.chdir(CONV)
    arquivos = sorted(origem.glob("*.npz")) if origem.is_dir() else [origem]
    if not arquivos:
        raise SystemExit(f"nenhum .npz em {origem}")

    out_dir = Path(saida)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"MHR: carregando assets de {MHR_REPO / 'assets'}")
    mhr_model = MHR.from_files(folder=MHR_REPO / "assets", lod=1, device=device)
    print(f"SMPL-X: {SMPLX_DIR}")
    smplx_model = smplx.create(
        model_path=str(SMPLX_DIR), model_type="smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=len(arquivos),
    ).to(device)

    entradas, nomes = [], []
    for f in arquivos:
        z = np.load(f)
        faltando = [k for k in ("mhr_model_params", "pred_vertices", "pred_cam_t") if k not in z]
        if faltando:
            print(f"  [pulado] {f.name}: faltam {faltando}")
            continue
        entradas.append(z)
        nomes.append(f.stem)
    if not entradas:
        raise SystemExit("nenhum .npz com os campos necessarios")

    print(f"convertendo {len(entradas)} corpo(s)...")
    conv = Conversion(mhr_model=mhr_model, smpl_model=smplx_model, method="pytorch")
    res = conv.convert_sam3d_output_to_smpl(
        sam3d_outputs=entradas,
        return_smpl_meshes=True,
        return_smpl_parameters=True,
        return_smpl_vertices=False,
        return_fitting_errors=True,
    )

    relatorio = {"convertidos": nomes, "erros_de_ajuste": None}
    if getattr(res, "result_errors", None) is not None:
        e = res.result_errors
        e = e.detach().cpu().numpy() if torch.is_tensor(e) else np.asarray(e)
        relatorio["erros_de_ajuste"] = [float(v) for v in np.ravel(e)]
        print("erro de ajuste por corpo:", relatorio["erros_de_ajuste"])

    params = getattr(res, "result_parameters", None)
    if params is not None:
        arrays = {}
        for k, v in (params.items() if isinstance(params, dict) else []):
            arrays[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        if arrays:
            np.savez_compressed(out_dir / "smplx_params.npz", **arrays)
            relatorio["parametros"] = {k: list(v.shape) for k, v in arrays.items()}
            print("parametros SMPL-X salvos:", relatorio["parametros"])

    malhas = getattr(res, "result_meshes", None)
    if malhas:
        for nome, m in zip(nomes, malhas):
            m.export(str(out_dir / f"{nome}_smplx.ply"))
        relatorio["malhas"] = [f"{n}_smplx.ply" for n in nomes]

    (out_dir / "conversao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print("pronto ->", out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
