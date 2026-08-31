"""Orquestrador da variante DECUPADA. Não substitui nada; declara um caminho.

POR QUE UM ORQUESTRADOR SEPARADO

Havia dois caminhos fazendo coisas sobrepostas sem que nenhum documento dissesse
isso -- que é a pior das configurações possíveis, porque a escolha entre eles
acontecia por acidente de qual comando alguém digitou. Aqui o caminho decupado
vira explícito, nomeado e rodável de ponta a ponta.

O QUE ELE REUSA E O QUE ELE TROCA

Reusa, sem tocar: parse_screenplay, cast_characters, synthesize_dialogue,
lipsync_scenes [6], mix_audio [7], assemble_final [8], verify_output [9].
Troca só o estágio [5], por `render_shots_stage`, e acrescenta três etapas novas
ANTES dele: story_structure, emotion_director e shot_plan.

    [1] parse            reusado
    [2] cast             reusado
    [E] emotion_director NOVO   emoção por fala + casting de voz
    [4] TTS              reusado -- e agora é ENTRADA da decupagem, não saída
    [S] story_structure  NOVO   função dramática e continuidade entre cenas
    [P] shot_plan        NOVO   a decupagem: enquadramento, ângulo, duração
    [5-D] render_shots   TROCA  um clipe por PLANO, still fixando o enquadramento
    [6] lipsync          reusado
    [7] mix_audio        reusado
    [8] assemble_final   reusado
    [9] verify_output    reusado

A ORDEM NÃO É ARBITRÁRIA

TTS vem ANTES da decupagem porque a duração de um plano de diálogo é a duração
da fala. MEDIDO em §3.25: estimando por contagem de caracteres, o plano da fala
mais longa da cena saiu com 9,07s para uma fala de 14,80s -- cortava metade. E
emotion_director vem antes do TTS porque trocar a emoção MUDA a duração (uma
fala foi de 8,85s para 10,19s ao virar "excitada"), o que mudaria a decupagem
depois de pronta.

CLI:
    python -m script_pipeline.run_decupagem --run-dir DIR --script ROTEIRO.txt \\
        [--style intimista] [--style-changes "1:nervoso,1.10:intimista"] \\
        [--width 960 --height 544] [--ate animatic]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

# Onde é seguro parar. `animatic` é o ponto de revisão barato: tudo decidido,
# nada renderizado em vídeo ainda.
# A ORDEM DESTA LISTA E O CONTRATO DE `--ate`: `ate()` compara INDICES, entao
# ela precisa bater com a ordem de execucao em main(). MEDIDO 2026-08-27: com
# "animatic" listado ANTES de "stills", `--ate animatic` -- justamente o ponto
# de revisao barato que a documentacao manda usar primeiro -- pulava o estagio
# dos stills, e o rascunho saia sem nenhuma imagem. Ao mexer aqui, confira
# contra a sequencia de `if ate(...)` la embaixo.
PARADAS = ["parse", "cast", "emocao", "tts", "estrutura", "plano", "stills",
           "animatic", "render", "lipsync", "mix", "final"]


def passo(nome: str, cmd: list, *, obrigatorio: bool = True) -> bool:
    print(f"\n{'='*70}\n[{nome}] {' '.join(str(c) for c in cmd[2:6])}...\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u", *cmd], cwd=str(ROOT))
    if r.returncode != 0:
        # Etapa opcional que falha não derruba a corrida: sem Ollama, por
        # exemplo, a estrutura sai só com a camada determinística e a decupagem
        # cai na cobertura neutra -- pior, mas utilizável.
        marca = "FALHOU" if obrigatorio else "falhou (opcional; seguindo)"
        print(f"[{nome}] {marca} (codigo {r.returncode})", flush=True)
        return not obrigatorio
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline com decupagem, ponta a ponta")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--script", help="roteiro .txt (so na primeira vez)")
    ap.add_argument("--style", default="classico")
    ap.add_argument("--style-changes", default=None,
                    help='troca de estilo, vigora ate a proxima marca. Por CENA ("3:tenso") ou por TOMADA dentro da cena ("1.10:intimista" = da tomada 10 da cena 1 em diante, atravessando o fim da cena). O numero da tomada e o que aparece na tabela do plano. Ex.: "1:classico,1.10:intimista,2:tenso"')
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--engine", default="qwen3.6-35b-a3b:latest")
    ap.add_argument("--language", default="pt")
    ap.add_argument("--ate", default="final", choices=PARADAS,
                    help="para depois desta etapa (use 'animatic' para revisar antes do video)")
    ap.add_argument("--image-engine", default="flux", choices=["flux", "sd35", "sdxl"],
                    help="motor dos STILLS (nao do video, que e sempre o LTX). "
                         "flux e o padrao e o que melhor obedece enquadramento e "
                         "lado de tela; sd35 carrega em ~1 min contra ~4 e cabe em "
                         "~12 GB de VRAM contra ~24, ao custo de enquadramento pior "
                         "e sem imagem de referencia por personagem.")
    ap.add_argument("--recast", action="store_true",
                    help="deixa o emotion_director reescolher a VOZ de cada personagem")
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    limite = PARADAS.index(args.ate)

    def ate(nome: str) -> bool:
        return PARADAS.index(nome) <= limite

    if ate("parse"):
        if not (run / "parse" / "scenes.json").exists():
            if not args.script:
                print("[run_decupagem] primeira execucao exige --script", file=sys.stderr)
                return 1
            if not passo("1 parse", ["-m", "script_pipeline.parse_screenplay",
                                     "--script", args.script, "--run-dir", str(run),
                                     "--enrich-engine", args.engine]):
                return 1
        else:
            print("[1 parse] ja feito, reaproveitando")

    if ate("cast") and not (run / "characters" / "cast.json").exists():
        # --engine: o descritor visual sai do mesmo motor que o resto da cadeia.
        # Sem ele o descritor e um recorte do texto de acao, que em roteiro de
        # prosa corrida devolve enredo em vez de aparencia -- e esse texto vai
        # colado em TODO still e TODO clipe. Ver cast_characters.
        if not passo("2 cast", ["-m", "script_pipeline.cast_characters",
                                "--run-dir", str(run), "--engine", args.engine]):
            return 1

    if ate("emocao"):
        cmd = ["-m", "script_pipeline.emotion_director", "--run-dir", str(run),
               "--engine", args.engine]
        if args.recast:
            cmd.append("--recast")
        passo("E emocao", cmd, obrigatorio=False)

    if ate("tts") and not (run / "dialogue" / "lines.json").exists():
        if not passo("4 tts", ["-m", "script_pipeline.synthesize_dialogue",
                               "--run-dir", str(run), "--language", args.language]):
            return 1

    if ate("estrutura") and not (run / "parse" / "story_structure.json").exists():
        cenas = run / "parse" / "scenes_enriched.json"
        if not cenas.exists():
            cenas = run / "parse" / "scenes.json"
        passo("S estrutura", ["-m", "script_pipeline.story_structure",
                              "--scenes", str(cenas), "--engine", args.engine,
                              "--out", str(run / "parse" / "story_structure.json")],
              obrigatorio=False)

    if ate("plano"):
        cenas = run / "parse" / "scenes_enriched.json"
        if not cenas.exists():
            cenas = run / "parse" / "scenes.json"
        cmd = ["-m", "script_pipeline.shot_plan", "--scenes", str(cenas),
               "--structure", str(run / "parse" / "story_structure.json"),
               "--dialogue", str(run / "dialogue" / "lines.json"),
               "--cast", str(run / "characters" / "cast.json"),
               "--style", args.style, "--fps", str(args.fps),
               "--out", str(run / "parse" / "shot_plan.json")]
        if args.style_changes:
            cmd += ["--style-changes", args.style_changes]
        if not passo("P decupagem", cmd):
            return 1

    if ate("stills"):
        if not passo("5-D stills", ["-m", "script_pipeline.render_shots_stage",
                                    "--run-dir", str(run), "--width", str(args.width),
                                    "--height", str(args.height), "--stills-only",
                                    "--image-engine", args.image_engine]):
            return 1

    # O animatic vem DEPOIS dos stills e ANTES do vídeo: é o único ponto em que
    # dá para ver a cena inteira montada sem ter gastado GPU com difusão.
    if ate("animatic"):
        # Quando o rascunho E o destino pedido, a falha dele e a falha da
        # corrida: nao ha etapa posterior para compensar, e anunciar "parado no
        # rascunho" apontando para um arquivo inexistente manda o usuario
        # revisar o vazio. Numa corrida ate o video ele segue opcional -- ali o
        # animatic e conveniencia, nao entrega.
        alvo = args.ate == "animatic"
        ok = passo("R rascunho", ["-m", "script_pipeline.render_shots",
                                  "--plan", str(run / "parse" / "shot_plan.json"),
                                  "--out", str(run / "shots"),
                                  "--dialogue", str(run / "dialogue" / "lines.json"),
                                  "--animatic", "--width", str(args.width)],
                   obrigatorio=not alvo)
        if alvo:
            if not ok or not (run / "shots" / "animatic.mp4").exists():
                print(file=sys.stderr)
                print("[run_decupagem] o rascunho NAO foi gerado. Confira o "
                      "estagio '5-D stills' acima: sem still nao ha animatic.",
                      file=sys.stderr)
                return 1
            print(f"\n[run_decupagem] parado no rascunho: {run / 'shots' / 'animatic.mp4'}")
            print("Revise e rode de novo sem --ate para seguir para o video.")
            return 0

    if ate("render"):
        # REINICIAR ANTES DO VIDEO nao e higiene, e o que separa 8,5 min de 55.
        # O ComfyUI que gerou os stills esta com o alocador moldado para o FLUX,
        # e o LTX 2.5 nao cabe nesse molde. Ver stop_comfyui().
        from script_pipeline.generate_storyboards import stop_comfyui
        stop_comfyui(log=lambda m: print(f"[reinicio] {m}", flush=True))

        # A passada de VIDEO nao gera imagem -- ela le os stills pelo manifesto,
        # por indice. O motor vai junto so para as duas passadas descreverem a
        # mesma corrida no log; nao ha decisao pendurada nele aqui.
        if not passo("5-D video", ["-m", "script_pipeline.render_shots_stage",
                                   "--run-dir", str(run), "--width", str(args.width),
                                   "--height", str(args.height), "--videos-only",
                                   "--fps", str(args.fps),
                                   "--image-engine", args.image_engine]):
            return 1

    if ate("lipsync"):
        passo("6 lipsync", ["-m", "script_pipeline.lipsync_scenes",
                            "--run-dir", str(run)], obrigatorio=False)
    if ate("mix"):
        passo("7 mix", ["-m", "script_pipeline.mix_audio",
                        "--run-dir", str(run)], obrigatorio=False)
    if ate("final"):
        if not passo("8 montagem", ["-m", "script_pipeline.assemble_final",
                                    "--run-dir", str(run)]):
            return 1
        passo("9 verificacao", ["-m", "script_pipeline.verify_output",
                                "--run-dir", str(run)], obrigatorio=False)

    print(f"\n[run_decupagem] concluido em {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
