"""Importa foto de referência EXTERNA (gerada por outra ferramenta, ex. ChatGPT/
DALL-E) para um personagem do cast, no formato que `character_sheet.py` produz
nativamente: rosto detectável, enquadramento cintura/peito para cima, sem
overlay.

Sem isso, quem tivesse uma foto pronta precisava recortar/redimensionar à mão
e editar `cast.json` na unha. Este script faz as duas coisas:

1. Detecta o MAIOR rosto (insightface/buffalo_l, mesmo backend do
   `consistency_audit.py` -- reprova cedo se o ArcFace não vai achar esse
   rosto depois, em vez de descobrir isso só quando o `check_consistency`
   falhar silenciosamente no meio da decupagem).
2. Recorta ao redor do rosto com a margem de um "plano médio, cintura/peito
   para cima" (a mesma moldura que `character_sheet.py::_reference_prompt`
   pede ao gerar), redimensiona para a maior dimensão configurada e salva
   como PNG RGB em `<run-dir>/characters/refs/`.

Grava o resultado em `cast.json[nome]["reference_image"]` (mesmo campo que
`cast_characters.build_cast` e `character_sheet.py::apply_to_cast` usam --
`render_shots.py`/`generate_storyboards.py` não sabem a origem da imagem).

CLI:
    python -m script_pipeline.import_reference --run-dir DIR \
        --character "Nome do Personagem" --image caminho/foto.png \
        [--character "Outro Nome" --image caminho/foto2.png ...]

Um par --character/--image por personagem, na ordem em que aparecem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Moldura do character_sheet.py: "medium shot, framed from the waist up" --
# rosto ocupando uma fração pequena do quadro, não um crop apertado de
# passaporte. Fator multiplicado pela altura do bbox do rosto para decidir
# até onde o crop desce/sobe/alarga.
_TOP_MARGIN = 1.1      # do topo do bbox para cima (cabelo/topo da cabeça)
_BOTTOM_MARGIN = 3.4   # do topo do bbox para baixo (desce até peito/cintura)
_SIDE_MARGIN = 1.7     # da largura do bbox para cada lado
_MAX_DIM = 960          # maior dimensão de saída -- mesma ordem de grandeza
                        # dos candidatos gerados pelo character_sheet.py (960x544)


def _detect_face_bbox(image_path: str):
    """(x0, y0, x1, y1) do maior rosto, ou None. Reusa o mesmo detector do
    consistency_audit.py para que 'detectável aqui' implique 'detectável
    depois' -- os dois passam pelo buffalo_l com os mesmos parâmetros."""
    import cv2
    import numpy as np
    from script_pipeline.consistency_audit import _get_app

    # cv2.imread fails on Windows when the run path contains accents; decode
    # the bytes so external references remain importable in e.g. CÉU_TURBULENTO.
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"não abriu como imagem: {image_path}")
    app = _get_app()
    faces = app.get(img)
    if not faces:
        return None, img
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    bbox = faces[0].bbox
    return tuple(float(v) for v in bbox), img


def import_reference_photo(image_path: str, out_path: Path, *, log=print) -> dict:
    """Recorta/normaliza uma foto externa para o enquadramento de referência
    do pipeline. Devolve {"ok": bool, "path": str|None, "face_detected": bool,
    "note": str}. NÃO levanta em rosto ausente -- grava mesmo assim (o
    chamador decide se aceita uma referência sem rosto detectável), mas
    avisa, porque `check_consistency` vai devolver None pra esse par depois."""
    from PIL import Image
    import numpy as np

    bbox, img_cv = _detect_face_bbox(image_path)
    h_img, w_img = img_cv.shape[:2]

    img = Image.open(image_path).convert("RGB")
    # BUGFIX auditoria 2026-09-16 (A11): so o ramo COM rosto criava a pasta de
    # destino antes de salvar. Na primeira importacao de uma foto sem rosto
    # detectavel (pasta `characters/refs/` ainda nao existe), o comportamento
    # prometido no docstring -- "grava mesmo assim, so avisa" -- terminava em
    # FileNotFoundError em vez do aviso.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bbox is None:
        # Sem rosto detectável: não inventa crop, só normaliza tamanho/canal
        # e deixa claro no retorno -- a foto pode ainda servir de referência
        # de corpo/figurino, mas não vai alimentar o ArcFace.
        img.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
        img.save(out_path)
        return {"ok": True, "path": str(out_path), "face_detected": False,
                "note": "nenhum rosto detectado -- salvo sem recorte, "
                        "consistency_audit não vai comparar por rosto."}

    x0, y0, x1, y1 = bbox
    fw, fh = x1 - x0, y1 - y0
    top = max(0, y0 - fh * _TOP_MARGIN)
    bottom = min(h_img, y0 + fh * _BOTTOM_MARGIN)
    cx = (x0 + x1) / 2
    left = max(0, cx - fw * _SIDE_MARGIN / 2 - fw / 2 * (_SIDE_MARGIN - 1))
    right = min(w_img, cx + fw * _SIDE_MARGIN / 2 + fw / 2 * (_SIDE_MARGIN - 1))
    # PIL usa RGB (imagem já convertida acima); bbox veio do cv2 (mesma
    # imagem, mesmas coordenadas -- cv2 só afeta ordem de canal, não geometria).
    crop = img.crop((int(left), int(top), int(right), int(bottom)))
    crop.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    log(f"[import_reference] {Path(image_path).name}: rosto detectado, "
        f"recorte {crop.size[0]}x{crop.size[1]} -> {out_path.name}")
    return {"ok": True, "path": str(out_path), "face_detected": True, "note": ""}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--character", action="append", default=[], dest="characters",
                     help="nome do personagem (repetível, pareado por posição com --image)")
    ap.add_argument("--image", action="append", default=[], dest="images",
                     help="caminho da foto externa (repetível, pareado por posição com --character)")
    args = ap.parse_args(argv)

    if len(args.characters) != len(args.images):
        ap.error(f"--character ({len(args.characters)}) e --image ({len(args.images)}) "
                  "precisam vir em pares na mesma quantidade.")
    if not args.characters:
        ap.error("nenhum --character/--image informado.")

    run_dir = Path(args.run_dir).resolve()
    cast_path = run_dir / "characters" / "cast.json"
    if not cast_path.exists():
        ap.error(f"cast.json não existe em {cast_path} -- rode o estágio de cast primeiro.")
    cast = json.loads(cast_path.read_text(encoding="utf-8"))

    refs_dir = run_dir / "characters" / "refs"
    resultado = {}
    for name, image_path in zip(args.characters, args.images):
        if name not in cast:
            print(f"[import_reference] AVISO: '{name}' não está em cast.json -- pulando "
                  f"(nomes existentes: {', '.join(cast.keys())}).")
            continue
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        out_path = refs_dir / f"{safe_name}_external.png"
        info = import_reference_photo(image_path, out_path)
        resultado[name] = info
        if info["ok"]:
            cast[name]["reference_image"] = info["path"]
            cast[name]["reference_source"] = "external"
            if not info["face_detected"]:
                print(f"[import_reference] AVISO: '{name}' sem rosto detectável em "
                      f"{image_path} -- confira antes de gerar stills.")

    cast_path.write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[import_reference] cast.json atualizado ({len(resultado)} personagem(ns)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
