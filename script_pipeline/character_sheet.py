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


# LIMIAR DE ACEITACAO -- achado da avaliacao visual 2026-09-17: a sheet do
# Palacio Esmeralda aceitou Mei-Li com score 0.475 (o candidato mais "central"
# do lote, mas o lote inteiro ja tinha drift de roupa/idade entre si -- o
# medoid so garante que o ESCOLHIDO parece com os OUTROS candidatos do MESMO
# lote, nunca que o lote inteiro bate com o descritor pedido). Abaixo disto,
# o lote e tratado como baixa confianca e um segundo lote roda com o prompt
# reforcado antes de aceitar.
CONSISTENCY_SCORE_THRESHOLD = 0.55


def _reference_prompt(name: str, descriptor: str, *, reinforce: bool = False) -> str:
    """Prompt do candidato: retrato NEUTRO -- sem ação, sem enquadramento de
    cena, sem cenário específico. É uma folha de referência, não um plano do
    filme; qualquer viés de pose/luz aqui contaminaria a comparação de
    similaridade entre candidatos (dois candidatos podem ter o MESMO rosto e
    ainda assim medir baixo se um está de perfil e o outro de frente).

    `reinforce`: segunda tentativa depois de um lote com score baixo -- repete
    o descritor como uma restricao explicita de figurino/penteado/epoca, em
    vez de uma frase solta entre outras. Duas doses do MESMO texto pesam mais
    no condicionamento do que uma, e a segunda vem em tom de restricao
    ('must match', 'no deviation'), nao de descricao."""
    partes = [
        "medium shot, framed from the waist up, eye-level angle.",
        f"{name}, {descriptor}" if descriptor else name,
        "neutral standing pose, facing camera directly, calm relaxed expression.",
        "plain neutral studio background, soft even lighting, no props, no other people.",
    ]
    if reinforce and descriptor:
        partes.append(
            f"Wardrobe, hairstyle, accessories and era must exactly match this "
            f"description, with no deviation and no generic or modern clothing "
            f"substituted in: {descriptor}")
    return " ".join(p for p in partes if p)


def generate_character_candidates(
    cast: dict, run_dir: Path, *, n_candidates: int = 4, image_engine: str = "flux",
    width: int = 960, height: int = 544, seed_base: int = 1009, log=print,
    lora_name: str = "", lora_strength: float = 0.8, force: bool = False,
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

    # ACHADO 2026-09-17 (auditoria externa, risco adicional): o ComfyUI subia
    # incondicionalmente aqui, ANTES do laco que checa `reference_image` por
    # personagem -- com um elenco inteiro ja tendo foto real (import_reference.py)
    # e sem --force, nao havia NADA pra gerar, mas o ComfyUI subia (custo de
    # minutos) mesmo assim. Filtra quem precisa de geracao ANTES de tocar no
    # ComfyUI/servidor; se ninguem precisa, nem sobe.
    pendentes = {name: info for name, info in cast.items()
                if force or not info.get("reference_image")}
    for name, info in cast.items():
        ref = info.get("reference_image")
        if ref and not Path(ref).exists():
            log(f"[character_sheet] {name}: AVISO -- reference_image aponta pra "
                f"{ref!r}, que nao existe mais no disco. "
                + ("Sera regenerada (--force)." if name in pendentes else
                   "Rode com --force pra regenerar, ou os stills seguintes vao falhar."))
    if not pendentes:
        log("[character_sheet] todo o elenco ja tem reference_image -- nada a gerar "
            "(use --force pra regenerar mesmo assim). ComfyUI nao foi iniciado.")
        return {}

    # Z-Image-Turbo e Qwen-Image-2.1 usam servidores HTTP proprios;
    # generate_scene_storyboard sobe/fala com o backend selecionado sozinho.
    if image_engine not in {"zimage", "qwen-image-2.1"}:
        sb.ensure_comfyui_running(server, wait_seconds=180, watch_stalls=False)

    out_dir = Path(run_dir) / "characters" / "sheet_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _gerar_lote(name, idx_nome, descriptor, *, reinforce, sufixo, seed_offset):
        prompt = _reference_prompt(name, descriptor, reinforce=reinforce)
        candidatos = []
        for i in range(n_candidates):
            seed = seed_base + idx_nome * 1000 + seed_offset + i * 101
            out_path = out_dir / f"{name}_{sufixo}{i}.png"
            ok = sb.generate_scene_storyboard(
                {"index": 0}, {}, server=server, checkpoint=engine["checkpoint"],
                width=width, height=height, steps=engine["steps"], cfg=engine["cfg"],
                seed=seed, out_path=out_path, clip=engine["clip"], vae=engine["vae"],
                guidance=engine["guidance"], prompt_override=prompt, log=log,
                lora_name=lora_name, lora_strength=lora_strength)
            if ok and out_path.exists():
                candidatos.append(out_path)
        return candidatos

    def _escolher_medoid(candidatos):
        embeddings = {}
        for c in candidatos:
            emb = face_embedding(str(c))
            if emb is not None:
                embeddings[c] = emb
        if len(embeddings) < 2:
            return (candidatos[0], None) if candidatos else (None, None)
        escolhido, melhor_media = None, -2.0
        for c, emb in embeddings.items():
            sims = [float(np.dot(emb, outro)) for oc, outro in embeddings.items() if oc != c]
            media = sum(sims) / len(sims)
            if media > melhor_media:
                escolhido, melhor_media = c, media
        return escolhido, melhor_media

    resultado = {}
    for idx_nome, (name, info) in enumerate(cast.items()):
        # Fix #3 acoplado a fotos externas (pedido do usuario 2026-09-17,
        # roteiro dos piratas): character-sheet virou padrao com 2+
        # personagens (ver run_decupagem.py), e sem este guard ela
        # SOBRESCREVIA a foto de referencia REAL que import_reference.py ja
        # tinha importado para cada personagem -- apply_to_cast() troca
        # `reference_image` sem perguntar. Uma foto real ja resolve o
        # problema que a sheet existe pra resolver (ancora de identidade
        # estavel); gerar candidatos sinteticos e substitui-la e regressao,
        # nao melhoria. So mexe em quem AINDA nao tem `reference_image`.
        if info.get("reference_image") and not force:
            log(f"[character_sheet] {name}: ja tem reference_image "
                f"({Path(info['reference_image']).name}) -- pulando (--force sobrescreve).")
            continue
        descriptor = info.get("descriptor", "")
        candidatos = _gerar_lote(name, idx_nome, descriptor, reinforce=False,
                                 sufixo="", seed_offset=0)
        if not candidatos:
            log(f"[character_sheet] {name}: nenhum candidato gerado -- pulando.")
            continue

        escolhido, score = _escolher_medoid(candidatos)
        if score is None:
            log(f"[character_sheet] {name}: rosto detectavel em menos de 2 "
                f"candidatos -- usando o primeiro gerado, sem escolha por similaridade.")
        else:
            log(f"[character_sheet] {name}: escolhido {escolhido.name} "
                f"(similaridade media aos outros: {score:.3f}).")

            # Fix #2 (avaliacao visual 2026-09-17): score baixo e sinal de que o
            # LOTE inteiro variou demais (roupa/idade/estilo), nao so o
            # candidato descartado -- um segundo lote com o descritor repetido
            # como restricao costuma convergir melhor que escolher o "menos
            # ruim" do primeiro.
            if score < CONSISTENCY_SCORE_THRESHOLD and descriptor:
                log(f"[character_sheet] {name}: score {score:.3f} abaixo do limiar "
                    f"({CONSISTENCY_SCORE_THRESHOLD}) -- gerando um segundo lote com "
                    "prompt reforcado.")
                candidatos2 = _gerar_lote(name, idx_nome, descriptor, reinforce=True,
                                          sufixo="r", seed_offset=500)
                if candidatos2:
                    escolhido2, score2 = _escolher_medoid(candidatos2)
                    if score2 is not None and score2 > score:
                        log(f"[character_sheet] {name}: segundo lote melhorou "
                            f"({score:.3f} -> {score2:.3f}) -- usando o reforcado.")
                        escolhido, score = escolhido2, score2
                        candidatos = candidatos2
                    else:
                        log(f"[character_sheet] {name}: segundo lote nao melhorou "
                            f"(mantendo o primeiro, score {score:.3f}).")
                if score < CONSISTENCY_SCORE_THRESHOLD:
                    log(f"[character_sheet] {name}: AVISO -- ainda abaixo do limiar "
                        f"depois do reforco ({score:.3f}); revise "
                        f"characters/sheet_candidates/{name}_*.png antes de confiar no still.")

        resultado[name] = {
            "reference_image": str(escolhido),
            "score": score,
            "candidates": [str(c) for c in candidatos],
            "needs_review": bool(score is not None and score < CONSISTENCY_SCORE_THRESHOLD),
        }
    return resultado


def apply_to_cast(cast: dict, resultado: dict) -> dict:
    """Grava `reference_image` escolhido de volta no dicionario de cast (o
    mesmo formato que `cast_characters.build_cast` produz). So sobrescreve
    quem esta em `resultado` -- personagens com `reference_image` (foto
    externa importada, ou sheet anterior) ja saem de `generate_character_
    candidates` pulados (ver o guard la), entao nunca chegam aqui a menos
    que `--force` tenha sido passado."""
    for name, info in resultado.items():
        if name in cast:
            cast[name]["reference_image"] = info["reference_image"]
            cast[name]["reference_sheet_score"] = info["score"]
            cast[name]["reference_needs_review"] = info.get("needs_review", False)
    return cast


def generate_turnaround_sheets(cast: dict, run_dir: Path, *, force: bool = False, seed_base: int = 4021,
                               log=print) -> dict:
    """Folha de 3 vistas (frente, tres-quartos, costas) de cada personagem COM foto de referencia.

    MEDIDO 2026-09-21 (Qwen-Image-2.1, 1536x864, 74 s): uniforme e penteado consistentes nas tres vistas
    a partir de UMA foto de rosto. Serve de segunda referencia dos planos que mostram o corpo (wide/full/
    medium/OTS): a foto de rosto sozinha vaza a roupa da foto (camisa bordo, blazer verde) -- a folha
    fixa o uniforme do roteiro. Grava `characters/turnaround/NOME.png` e `cast[nome]["turnaround_image"]`."""
    import qwen_image21_engine as q
    out_dir = Path(run_dir) / "characters" / "turnaround"
    out_dir.mkdir(parents=True, exist_ok=True)
    feitos = {}
    for idx, (name, info) in enumerate(cast.items()):
        ref = info.get("reference_image")
        if not ref or not Path(ref).exists():
            log(f"[turnaround] {name}: sem foto de referencia; pulando")
            continue
        dest = out_dir / f"{name}.png"
        if dest.exists() and info.get("turnaround_image") == str(dest) and not force:
            feitos[name] = str(dest)
            continue
        outfit = (info.get("descriptor") or "the outfit described for the character").strip()
        prompt = q.TURNAROUND_PROMPT.format(outfit=outfit)
        if q.generate(prompt, dest, width=1536, height=864, seed=seed_base + idx, reference_images=[ref], log=log):
            info["turnaround_image"] = str(dest)
            feitos[name] = str(dest)
        else:
            log(f"[turnaround] {name}: falhou")
    return feitos


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-candidates", type=int, default=4)
    ap.add_argument("--image-engine", default="flux",
                    choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext", "zimage", "qwen-image-2.1", "hidream", "qwen-image"])
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--apply", action="store_true",
                    help="grava o resultado em characters/cast.json (sem isto, so gera e reporta)")
    from script_pipeline.generate_storyboards import available_loras_images
    ap.add_argument("--lora", default="", choices=[""] + available_loras_images(),
                    help="LoRA opcional aplicado aos candidatos (models/loras_images/). "
                         "Sem isto, nenhum LoRA.")
    ap.add_argument("--lora-strength", type=float, default=0.8)
    ap.add_argument("--force", action="store_true",
                    help="gera candidatos mesmo para personagem que ja tem reference_image "
                         "(foto externa importada, ou sheet anterior) e substitui. Sem isto, "
                         "quem ja tem foto e pulado -- ela e a ancora de identidade mais forte "
                         "que existe, gerar uma sintetica por cima seria regressao.")
    ap.add_argument("--turnaround", action="store_true",
                    help="em vez de candidatos, gera a folha de 3 vistas de cada personagem com foto "
                         "(Qwen-Image-2.1) e grava turnaround_image no cast.json")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cast_path = run_dir / "characters" / "cast.json"
    cast = json.loads(cast_path.read_text(encoding="utf-8"))
    if args.turnaround:
        feitos = generate_turnaround_sheets(cast, run_dir, force=args.force)
        cast_path.write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[turnaround] {len(feitos)} folha(s): {sorted(feitos)}")
        return 0 if feitos or not cast else 1

    resultado = generate_character_candidates(
        cast, run_dir, n_candidates=args.n_candidates, image_engine=args.image_engine,
        width=args.width, height=args.height, force=args.force,
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
