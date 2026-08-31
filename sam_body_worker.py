"""Headless SAM 3D Body inference for the unified local WebUI.

Grava DOIS artefatos por imagem:

- `sam_body_NN.obj`  — malha estatica, para o visualizador da UI e para levar
  ao Blender/Hunyuan como geometria.
- `sam_body_NN.npz`  — o RIG: 127 juntas com rotacao, coordenadas de junta,
  parametros de pose/forma/maos/expressao e a camera estimada.

Por que o `.npz` importa: o OBJ nao carrega esqueleto nem pesos. A versao
anterior deste worker gravava apenas `pred_vertices` + `faces`, ou seja,
descartava exatamente a parte que um pipeline de DANCA precisa -- o
ChoreoEngine produz movimento (rotacao/posicao de junta ao longo do tempo) e
nao tem onde aplica-lo se o corpo vier como malha morta. Medido numa imagem de
teste: o resultado traz `pred_global_rots` (127, 3, 3) e `pred_joint_coords`
(127, 3), alem de `body_pose_params` (133,), `shape_params` (45,),
`hand_pose_params` (108,) e `mhr_model_params` (204,).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SAM_REPO = ROOT / "sam-3d-body"
sys.path.insert(0, str(SAM_REPO))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

# O SAM carrega SEM detector de pessoa ("No human detector is used..."), e nesse
# modo ele ajusta um corpo a IMAGEM INTEIRA -- medido: numa foto 720x1280 com
# varias pessoas a caixa devolvida foi [0, 0, 720, 1280] e o resultado nao
# corresponde a ninguem. Como o ChoreoEngine ja tem um detector ONNX rodando na
# GPU (o mesmo do DWPose), reaproveitamos ele e passamos a caixa pronta.
CHOREO = Path(os.environ.get("CHOREO_ROOT", r"E:\Users\home\Documents\ChoreoEngine"))


def carregar_detector():
    """Detector de pessoa do ChoreoEngine, ou None se indisponivel."""
    try:
        if str(CHOREO) not in sys.path:
            sys.path.insert(0, str(CHOREO))
        from choreo.extract.pose2d import PersonDetector

        return PersonDetector()
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] detector de pessoa indisponivel ({exc}); "
              f"o SAM vai ajustar o corpo ao quadro inteiro", file=sys.stderr)
        return None

# Tudo o que compoe o rig ou a calibracao. Salvo cru, sem reinterpretar: quem
# consome (retarget, conversao para SMPL-X, Blender) decide a convencao.
CAMPOS_RIG = (
    "pred_global_rots",     # (J, 3, 3) rotacao global por junta -- o esqueleto posado
    "pred_joint_coords",    # (J, 3)    posicao 3D por junta
    "pred_keypoints_3d",    # (K, 3)    keypoints 3D
    "pred_keypoints_2d",    # (K, 2)    reprojecao na imagem
    "global_rot",           # (3,)      orientacao global do corpo
    "body_pose_params",     # (133,)    pose articular
    "hand_pose_params",     # (108,)
    "expr_params",          # (72,)
    "shape_params",         # (45,)     identidade (equivalente aos betas)
    "scale_params",         # (28,)
    "mhr_model_params",     # (204,)    vetor parametrico completo do MHR
    "pred_cam_t",           # (3,)      translacao de camera
    "pred_pose_raw",        # (266,)    pose crua (rot global 6D + pose continua)
    "bbox",
    "lhand_bbox",
    "rhand_bbox",
)
# Os campos acima cobrem, como superconjunto, o formato dos exemplos oficiais em
# MHR/tools/mhr_smpl_conversion/data/sam3d_body_outputs/ -- conferido campo a
# campo, para o `.npz` deste worker entrar direto no conversor MHR -> SMPL-X.


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as output:
        for x, y, z in vertices:
            output.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for a, b, c in faces:
            output.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_rig(path: Path, resultado: dict, faces: np.ndarray) -> dict:
    """Salva o rig num `.npz` e devolve um resumo legivel."""
    arrays: dict[str, np.ndarray] = {"faces": np.asarray(faces)}
    resumo: dict[str, list[int]] = {}
    for campo in CAMPOS_RIG:
        valor = resultado.get(campo)
        if valor is None:
            continue
        arr = np.asarray(valor)
        arrays[campo] = arr
        resumo[campo] = list(arr.shape)
    # `focal_length` e' escalar; entra separado para nao virar array 0-d confuso
    if resultado.get("focal_length") is not None:
        arrays["focal_length"] = np.asarray(float(resultado["focal_length"]))
        resumo["focal_length"] = []
    arrays["pred_vertices"] = np.asarray(resultado["pred_vertices"])
    resumo["pred_vertices"] = list(arrays["pred_vertices"].shape)
    np.savez_compressed(path, **arrays)
    return resumo


def main(image_folder: str, output_folder: str) -> None:
    image_dir, output_dir = Path(image_folder), Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = SAM_REPO / "sam_3d_body" / "models"
    model, cfg = load_sam_3d_body(
        str(model_dir / "model.ckpt"), device="cuda", mhr_path=str(model_dir / "mhr_model.pt")
    )
    estimator = SAM3DBodyEstimator(model, cfg)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in extensions]
    if not images:
        raise RuntimeError("Nenhuma imagem de referência encontrada.")

    detector = carregar_detector()

    manifesto = []
    for index, image in enumerate(images, start=1):
        bboxes, n_detectadas = None, None
        if detector is not None:
            import cv2

            quadro = cv2.imread(str(image))
            caixas = detector(quadro)
            n_detectadas = int(len(caixas))
            if n_detectadas:
                # a pessoa de maior score e' o personagem; as outras entram no
                # manifesto so como contagem, para a foto errada nao passar batido
                caixas = caixas[np.argsort(-caixas[:, 4])]
                bboxes = caixas[:1, :4].astype(np.float32)

        results = estimator.process_one_image(str(image), bboxes=bboxes)
        if not results:
            raise RuntimeError(f"O SAM não conseguiu estimar um corpo em {image.name}.")
        r = results[0]
        obj = output_dir / f"sam_body_{index:02d}.obj"
        npz = output_dir / f"sam_body_{index:02d}.npz"
        write_obj(obj, r["pred_vertices"], estimator.faces)
        resumo = write_rig(npz, r, estimator.faces)
        manifesto.append({
            "imagem": image.name,
            "pessoas_no_quadro": n_detectadas,
            "caixa_usada": (bboxes[0].tolist() if bboxes is not None else "quadro inteiro"),
            "corpos_estimados": len(results),
            "malha": obj.name,
            "rig": npz.name,
            "formas": resumo,
        })

    (output_dir / "sam_body.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
