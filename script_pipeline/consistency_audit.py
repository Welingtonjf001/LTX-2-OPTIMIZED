"""Auditoria automatica de consistencia visual entre stills, via embedding
facial (insightface/buffalo_l, ArcFace). Nasceu de um pedido do usuario depois
de comparar 4 motores de storyboard a olho (2026-09-03) -- ate aqui a unica
forma de saber "quem manteve melhor a Lyra" era inspecao visual.

VALIDADO antes de integrar (nao so instalado -- medido): comparando os stills
da propria cena Lyra/Thoren (mesmo personagem em dois planos vs personagens
diferentes), a similaridade de cosseno do embedding ficou em 0,85-0,97 para o
MESMO personagem e 0,10-0,12 para personagens DIFERENTES -- separacao limpa o
bastante pra qualquer limiar entre 0,2 e 0,5 funcionar. Ver MEMORIAL.md 3.53.

Uso principal: `check_consistency()`, chamada de dentro de
`render_shots.py::_still_for_shot` depois de cada still com referencia --
se a similaridade cair abaixo do limiar, tenta de novo com seed diferente
(ate `max_retries`), fica com o MELHOR resultado entre as tentativas.

CLI standalone, pra auditar um still_dir inteiro depois do fato:
    python -m script_pipeline.consistency_audit --stills-dir DIR \
        --manifest DIR/stills.json --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_APP = None  # cache: o modelo (buffalo_l) carrega uma vez, nao por chamada


def _get_app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(640, 640))
    return _APP


def face_embedding(image_path: str):
    """Devolve o embedding normalizado do MAIOR rosto na imagem, ou None se
    nenhum rosto foi detectado (still de wide/insert sem close o bastante,
    por exemplo -- nao e erro, so nao ha o que comparar)."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    app = _get_app()
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def face_similarity(path_a: str, path_b: str) -> float | None:
    """Similaridade de cosseno entre os embeddings dos rostos principais de
    duas imagens. None se algum dos dois nao tem rosto detectavel -- CHAMADOR
    decide o que fazer (nao pressupor falha: still wide legitimamente pode
    nao ter rosto grande o bastante)."""
    import numpy as np

    ea = face_embedding(path_a)
    eb = face_embedding(path_b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))


def check_consistency(still_path: str, reference_path: str, *, threshold: float = 0.35) -> tuple[bool, float | None]:
    """(ok, score). ok=True quando score >= threshold OU quando nao da pra
    medir (sem rosto em algum dos dois -- nao bloqueia geracao por um caso
    que a ferramenta nao sabe avaliar, so reporta score=None)."""
    score = face_similarity(still_path, reference_path)
    if score is None:
        return True, None
    return score >= threshold, score


def audit_directory(stills_dir: Path, manifest: dict, *, threshold: float = 0.35, log=print) -> dict:
    """Audita um diretorio de stills JA GERADO contra as referencias que o
    proprio manifesto registrou (reaproveita `reference` se o manifesto de
    render_shots.py passar a gravar isso -- ver nota em render_shots.py).
    Devolve um relatorio {indice: {"score":..., "ok":...}}."""
    relatorio = {}
    for idx, entry in manifest.items():
        still = stills_dir / entry.get("file", "")
        ref = entry.get("reference")
        if not ref or not still.exists():
            continue
        ok, score = check_consistency(str(still), ref, threshold=threshold)
        relatorio[idx] = {"file": entry.get("file"), "score": score, "ok": ok, "reference": ref}
        marca = "OK" if ok else "ABAIXO DO LIMIAR"
        log(f"  plano {idx}: {entry.get('file')} vs {Path(ref).name} -> "
            f"{'sem rosto detectavel' if score is None else f'{score:.3f}'} ({marca})")
    return relatorio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stills-dir", required=True)
    ap.add_argument("--manifest", default=None, help="stills.json (default: <stills-dir>/stills.json)")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--out", default=None, help="salva o relatorio em JSON")
    args = ap.parse_args()

    stills_dir = Path(args.stills_dir)
    manifest_path = Path(args.manifest) if args.manifest else stills_dir / "stills.json"
    if not manifest_path.exists():
        print(f"{manifest_path} nao existe.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    relatorio = audit_directory(stills_dir, manifest, threshold=args.threshold)
    n_ok = sum(1 for r in relatorio.values() if r["ok"])
    print(f"\n{n_ok}/{len(relatorio)} plano(s) com referencia dentro do limiar {args.threshold}.")
    if args.out:
        Path(args.out).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"relatorio -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
