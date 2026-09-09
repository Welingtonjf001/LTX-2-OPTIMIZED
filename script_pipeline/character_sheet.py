"""Fase A da etapa de consistência de personagem (2026-09-08, pedido do
usuário depois de assistir ao filme e apontar drift de rosto/acessório).

PROBLEMA: `render_shots.py` já documenta que "o primeiro still de cada
personagem vira reference_image dos stills seguintes" -- mas esse primeiro
still é só o que aconteceu de sair primeiro na ordem dos planos, gerado uma
ÚNICA vez, sem nenhum critério de qualidade. Se aquela geração específica saiu
mediana (ângulo estranho, expressão ruim, traço levemente fora do descritor),
o filme INTEIRO herda esse desvio, porque é a partir dela que todo o resto é
comparado (via consistency_audit) e/ou visualmente ancorado (I2V).

SOLUÇÃO (Fase A, sem LoRA -- ltx-trainer exige Linux+triton e 80GB
recomendado, inviável nesta máquina, ver MEMORIAL 3.72): gerar um pequeno
LOTE de candidatos por personagem (mesma descrição, seeds diferentes) e
escolher o mais CENTRAL -- o que tem maior similaridade facial MÉDIA aos
outros candidatos, não o primeiro que saiu. Intuição: se 4 gerações do mesmo
prompt produzem um cacho e um outlier, o cacho é o que o modelo "quer" gerar
consistentemente para aquele descritor: é mais fácil ancorar o resto do filme
nele do que no outlier.

Roda ANTES do estágio de stills normal (`render_shots_stage.py`), gera pra
`<run-dir>/characters/sheet_candidates/`, e devolve {nome: path_escolhido}
para o chamador decidir o que fazer com isso (hoje: gravar em
`cast.json[nome]["reference_image"]`, substituindo o que `build_cast` teria
deixado vazio).

CLI standalone:
    python -m script_pipeline.character_sheet --run-dir DIR [--n-candidates 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LogFn = None  # tipagem só documental, log=print por padrão


def _reference_prompt(name: str, descriptor: str) -> str:
    """Prompt do candidato: retrato NEUTRO -- sem ação, sem enquadramento de
    cena, sem cenário específico. É uma folha de referência, não um plano do
    filme; qualquer viés de pose/luz aqui contaminaria a comparação de
    similaridade entre candidatos (dois candidatos podem ter o MESMO rosto e
    ainda assim medir baixo se um está de perfil e o outro de frente)."""
    partes = [
        "medium shot, framed from the waist up, eye-level angle.",
        f"{name}, {descriptor}" if descriptor else name,
        "neutral standing pose, facing camera directly, calm relaxed expression.",
        "plain neutral studio background, soft even lighting, no props, no other people.",
    ]
    return " ".join(p for p in partes if p)


def generate_character_candidates(
    cast: dict, run_dir: Path, *, n_candidates: int = 4, image_engine: str = "flux",
    width: int = 960, height: int = 544, seed_base: int = 1009, log=print,
    lora_name: str = "", lora_strength: float = 0.8,
) -> dict:
    """Gera `n_candidates` retratos por personagem e escolhe o medoid (maior
    similaridade facial MÉDIA aos outros candidatos do mesmo personagem).

    Devolve {nome: {"reference_image": path, "score": float|None,
    "candidates": [paths]}} -- `score` é None quando não deu pra comparar
    (menos de 2 candidatos com rosto detectável; o still fica com o
    PRIMEIRO candidato gerado nesse caso, sem alegar escolha melhor)."""
    from script_pipeline import generate_storyboards as sb
    from script_pipeline.consistency_audit import face_embedding
    import numpy as np

    engine = sb.IMAGE_ENGINES[image_engine]
    server = "http://127.0.0.1:8188"
    sb.ensure_comfyui_running(server, wait_seconds=180, watch_stalls=False)

    out_dir = Path(run_dir) / "characters" / "sheet_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    resultado = {}
    for idx_nome, (name, info) in enumerate(cast.items()):
        descriptor = info.get("descriptor", "")
        prompt = _reference_prompt(name, descriptor)
        candidatos = []
        for i in range(n_candidates):
            seed = seed_base + idx_nome * 1000 + i * 101
            out_path = out_dir / f"{name}_{i}.png"
            ok = sb.generate_scene_storyboard(
                {"index": 0}, {}, server=server, checkpoint=engine["checkpoint"],
                width=width, height=height, steps=engine["steps"], cfg=engine["cfg"],
                seed=seed, out_path=out_path, clip=engine["clip"], vae=engine["vae"],
                guidance=engine["guidance"], prompt_override=prompt, log=log,
                lora_name=lora_name, lora_strength=lora_strength)
            if ok and out_path.exists():
                candidatos.append(out_path)
        if not candidatos:
            log(f"[character_sheet] {name}: nenhum candidato gerado -- pulando.")
            continue

        embeddings = {}
        for c in candidatos:
            emb = face_embedding(str(c))
            if emb is not None:
                embeddings[c] = emb

        if len(embeddings) < 2:
            escolhido = candidatos[0]
            score = None
            log(f"[character_sheet] {name}: rosto detectavel em menos de 2 "
                f"candidatos ({len(embeddings)}/{len(candidatos)}) -- usando "
                f"o primeiro gerado, sem escolha por similaridade.")
        else:
            escolhido, melhor_media = None, -2.0
            for c, emb in embeddings.items():
                sims = [float(np.dot(emb, outro)) for oc, outro in embeddings.items() if oc != c]
                media = sum(sims) / len(sims)
                if media > melhor_media:
                    escolhido, melhor_media = c, media
            score = melhor_media
            log(f"[character_sheet] {name}: {len(embeddings)}/{len(candidatos)} "
                f"com rosto detectavel -- escolhido {escolhido.name} "
                f"(similaridade media aos outros: {score:.3f}).")

        resultado[name] = {
            "reference_image": str(escolhido),
            "score": score,
            "candidates": [str(c) for c in candidatos],
        }
    return resultado


def apply_to_cast(cast: dict, resultado: dict) -> dict:
    """Grava `reference_image` escolhido de volta no dicionario de cast (o
    mesmo formato que `cast_characters.build_cast` produz) -- SUBSTITUI
    qualquer `reference_image` que já existisse (essa é a etapa que decide
    isso agora, não o acaso do primeiro still)."""
    for name, info in resultado.items():
        if name in cast:
            cast[name]["reference_image"] = info["reference_image"]
            cast[name]["reference_sheet_score"] = info["score"]
    return cast


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-candidates", type=int, default=4)
    ap.add_argument("--image-engine", default="flux", choices=["flux", "sd35", "sdxl"])
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--apply", action="store_true",
                    help="grava o resultado em characters/cast.json (sem isto, so gera e reporta)")
    from script_pipeline.generate_storyboards import available_loras_images
    ap.add_argument("--lora", default="", choices=[""] + available_loras_images(),
                    help="LoRA opcional aplicado aos candidatos (models/loras_images/). "
                         "Sem isto, nenhum LoRA.")
    ap.add_argument("--lora-strength", type=float, default=0.8)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cast_path = run_dir / "characters" / "cast.json"
    cast = json.loads(cast_path.read_text(encoding="utf-8"))

    resultado = generate_character_candidates(
        cast, run_dir, n_candidates=args.n_candidates, image_engine=args.image_engine,
        width=args.width, height=args.height,
        lora_name=args.lora, lora_strength=args.lora_strength)

    report_path = run_dir / "characters" / "sheet_report.json"
    report_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[character_sheet] relatorio em {report_path}")

    if args.apply:
        cast = apply_to_cast(cast, resultado)
        cast_path.write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[character_sheet] cast.json atualizado com as referencias escolhidas.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
