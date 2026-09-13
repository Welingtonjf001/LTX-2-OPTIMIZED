"""Auditoria de VAZAMENTO DE GUIA: corte seco dentro de UM clipe.

VISTO 2026-09-13 (corrida teste_webui_loras, w4a8 + IC-LoRA MSR): nos planos de 73
quadros com dois sujeitos, o video copiava a sequencia de referencia do MSR quadro a
quadro -- plano medio das duas personagens (as imagens de sujeito) e, no quadro 57,
corte seco para o cenario (o grupo final da guia). O video_doctor apontava como
anomalia (z_glob 100-270); e um CORTE, e a mesma familia da folha do Ingredients
virando o proprio video (MEMORIAL 3.78): a guia age como keyframe.

Um clipe unico nao tem corte legitimo. Usa video_doctor.detect_cuts (mudanca que
PERSISTE) sobre os histogramas do clipe; qualquer corte achado e suspeita de guia
vazando.

CLI: python -m script_pipeline.guide_leak_audit --run-dir DIR   (ou --video X.mp4)
Escreve <run>/shots/guide_leak_audit.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Salto do still (quadro 0) para o quadro 1. detect_cuts exige dois quadros estaveis ANTES
# do corte e nunca ve o quadro 1. MEDIDO 2026-09-13 (MSR com guia de 17 quadros num plano
# de 73): 1-correl 0,0->1 = 0,45 e ~0,000 dali em diante -- o video largou o still no
# primeiro quadro e virou o plano medio dos retratos da guia. Movimento real fica < 0,01.
SALTO_INICIAL = 0.15


def audit_clip(video: str, *, z_thr: float = 6.0) -> dict:
    import cv2
    from video_doctor import color_hist, detect_cuts
    cap = cv2.VideoCapture(video)
    hists = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        hists.append(color_hist(fr))
    cap.release()
    if len(hists) < 5:
        return {"frames": len(hists), "cortes": [], "vazamento": False}
    cortes = detect_cuts({"_hists": hists}, z_thr=z_thr)
    salto0 = 1.0 - float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))
    if salto0 > SALTO_INICIAL and 1 not in cortes:
        cortes = [1] + list(cortes)
    return {"frames": len(hists), "cortes": cortes, "vazamento": bool(cortes),
            "salto_still": round(salto0, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir")
    ap.add_argument("--video")
    args = ap.parse_args(argv)
    if args.video:
        print(json.dumps(audit_clip(args.video), ensure_ascii=False))
        return 0
    run = Path(args.run_dir)
    rel = {}
    for clip in sorted((run / "shots" / "clips").glob("shot*.mp4")):
        r = audit_clip(str(clip))
        rel[clip.stem] = r
        marca = f"CORTE em {r['cortes']} -- guia vazando?" if r["vazamento"] else "ok"
        print(f"{clip.stem}: {r['frames']} quadros, {marca}")
    (run / "shots" / "guide_leak_audit.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    n = sum(r["vazamento"] for r in rel.values())
    print(f"guide_leak_audit: {n}/{len(rel)} clipe(s) com corte interno")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
