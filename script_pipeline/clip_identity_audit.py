"""Identidade do personagem ENTRE CLIPES, medida por embedding facial.

Item 10 do MEMORIAL §7 ("identidade entre cenas, sem ferramenta de medida") -- a
ferramenta ja existia para STILLS (`consistency_audit`, insightface/ArcFace, validado em
§3.53: mesmo personagem 0,85-0,97, personagens diferentes 0,10-0,12). Aqui ela olha os
CLIPES: o still pode estar certo e o video derivar (o rosto muda no meio do plano, ou o
MSR/IC troca a pessoa).

Para cada clipe com personagem: amostra quadros a 15/50/85% da duracao, pega o maior
rosto de cada um e mede
  - `vs_still`: o rosto do clipe contra o rosto do still do proprio plano (deriva dentro
    do plano);
  - `vs_personagem`: contra o centroide dos rostos do mesmo personagem em TODOS os clipes
    (deriva entre planos);
  - `quadros_min`: a pior similaridade entre os quadros amostrados e o still.
Abaixo de 0,35 (o limiar ja validado) marca o plano.

CLI: python -m script_pipeline.clip_identity_audit --run-dir DIR
Escreve <run>/shots/clip_identity_audit.json. So avisa; nunca bloqueia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIMIAR = 0.35


def _rostos_do_clipe(video: str, fracoes=(0.15, 0.5, 0.85)) -> list:
    import cv2
    from script_pipeline.consistency_audit import _get_app
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    app = _get_app()
    embs = []
    for f in fracoes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * f))))
        ok, frame = cap.read()
        if not ok:
            continue
        faces = app.get(frame)
        if faces:
            faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            embs.append(faces[0].normed_embedding)
    cap.release()
    return embs


def _still_do_plano(run: Path, clip_id: str) -> Path | None:
    num = clip_id.rsplit("shot", 1)[-1]
    from script_pipeline.render_shots import still_candidates
    achados = still_candidates(run / "shots" / "stills", int(num))
    return achados[0] if achados else None


def audit_run(run: Path, *, log=print) -> dict:
    from script_pipeline.consistency_audit import face_embedding
    clips = json.loads((run / "scenes" / "clips.json").read_text(encoding="utf-8"))
    medidas = {}
    for c in clips:
        if not c.get("character") or not c.get("video_path") or not Path(c["video_path"]).exists():
            continue
        embs = _rostos_do_clipe(c["video_path"])
        if not embs:
            medidas[c["id"]] = {"personagem": c["character"], "framing": c.get("framing"),
                                "rostos": 0, "motivo": "sem rosto detectavel nos quadros amostrados"}
            continue
        media = np.mean(embs, axis=0)
        media = media / (np.linalg.norm(media) or 1.0)
        still = _still_do_plano(run, c["id"])
        e_still = face_embedding(str(still)) if still else None
        medidas[c["id"]] = {
            "personagem": c["character"], "framing": c.get("framing"), "rostos": len(embs),
            "_emb": media,
            "vs_still": None if e_still is None else round(float(np.dot(media, e_still)), 3),
            "quadros_min": None if e_still is None else round(float(min(np.dot(e, e_still) for e in embs)), 3),
        }
    por_personagem: dict[str, list] = {}
    for m in medidas.values():
        if "_emb" in m:
            por_personagem.setdefault(m["personagem"], []).append(m["_emb"])
    centroides = {}
    for p, lst in por_personagem.items():
        ctr = np.mean(lst, axis=0)
        centroides[p] = ctr / (np.linalg.norm(ctr) or 1.0)
    relatorio = {}
    for cid, m in medidas.items():
        emb = m.pop("_emb", None)
        if emb is not None and len(por_personagem.get(m["personagem"], [])) > 1:
            m["vs_personagem"] = round(float(np.dot(emb, centroides[m["personagem"]])), 3)
        valores = [v for v in (m.get("vs_still"), m.get("quadros_min"), m.get("vs_personagem")) if v is not None]
        m["alerta"] = bool(valores) and min(valores) < LIMIAR
        relatorio[cid] = m
        if "motivo" in m:
            log(f"  {cid} ({m['personagem']}): {m['motivo']}")
        else:
            log(f"  {cid} ({m['personagem']}, {m.get('framing')}): vs_still={m.get('vs_still')} "
                f"pior_quadro={m.get('quadros_min')} vs_personagem={m.get('vs_personagem')}"
                f"{'  <- IDENTIDADE DERIVOU' if m['alerta'] else ''}")
    return relatorio


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="retorna erro se qualquer clipe tiver alerta de identidade")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    relatorio = audit_run(run)
    report_path = run / "shots" / "clip_identity_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    n = sum(1 for m in relatorio.values() if m.get("alerta"))
    print(f"clip_identity_audit: {n}/{len(relatorio)} clipe(s) com identidade abaixo de {LIMIAR}")
    return 1 if args.strict and n else 0


if __name__ == "__main__":
    raise SystemExit(main())
