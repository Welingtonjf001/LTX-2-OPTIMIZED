"""Orquestrador da variante DECUPADA. Não substitui nada; declara um caminho.

POR QUE UM ORQUESTRADOR SEPARADO

Havia dois caminhos fazendo coisas sobrepostas sem que nenhum documento dissesse
isso -- que é a pior das configurações possíveis, porque a escolha entre eles
acontecia por acidente de qual comando alguém digitou. Aqui o caminho decupado
vira explícito, nomeado e rodável de ponta a ponta.

O QUE ELE REUSA E O QUE ELE TROCA

Reusa, sem tocar: parse_screenplay, cast_characters, synthesize_dialogue,
lipsync_scenes [6], mix_audio [7], assemble_final [8], verify_output [9].
Troca só o estágio [5], por `render_shots_stage`, e acrescenta quatro etapas
novas ANTES dele: story_structure, emotion_director, shot_plan e (opt-in)
character_sheet.

    [1] parse            reusado
    [2] cast             reusado
    [E] emotion_director NOVO   emoção por fala + casting de voz
    [4] TTS              reusado -- e agora é ENTRADA da decupagem, não saída
    [S] story_structure  NOVO   função dramática e continuidade entre cenas
    [P] shot_plan        NOVO   a decupagem: enquadramento, ângulo, duração
    [C] character_sheet  NOVO, opt-in (--character-sheet) -- referência de
                          personagem escolhida por centralidade facial entre
                          N candidatos, em vez do primeiro still que sair
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
import os
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
PARADAS = ["parse", "cast", "emocao", "tts", "estrutura", "plano", "sheet",
           "stills", "animatic", "render", "lipsync", "mix", "final"]


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
    from script_pipeline.generate_storyboards import IMAGE_ENGINES
    ap.add_argument("--image-engine", default="flux", choices=sorted(IMAGE_ENGINES),
                    help="motor dos STILLS (video e escolhido por --video-engine). "
                         "flux e o padrao e o que melhor obedece enquadramento e "
                         "lado de tela; sd35 carrega em ~1 min contra ~4 e cabe em "
                         "~12 GB de VRAM contra ~24, ao custo de enquadramento pior "
                         "e sem imagem de referencia por personagem. flux-krea/"
                         "flux-kontext = FLUX.1, ver MEMORIAL 3.48.")
    ap.add_argument("--recast", action="store_true",
                    help="deixa o emotion_director reescolher a VOZ de cada personagem")
    ap.add_argument("--video-engine", default="ltx", choices=["ltx", "minimax"],
                    help="motor de VIDEO (nao confundir com --engine, que e o LLM de "
                         "estrutura). ltx = LTX 2.5, condicionado pela fala sintetizada "
                         "(TTS + lipsync + mix rodam normalmente). minimax = MiniMax H3 -- "
                         "fala e lip-sync NATIVOS a partir do texto do video_prompt; os "
                         "estagios 6/7 (lipsync/mix) viram passthrough pra esses planos, "
                         "ja que o audio ja vem pronto no clipe.")
    ap.add_argument("--ltx-variant", default="distilled",
                    choices=["distilled", "dev", "gguf-q6k"],
                    help="variante do checkpoint LTX 2.5 (so importa com "
                         "--video-engine ltx). MEDIDO 2026-09-06, mesma cena/seed: "
                         "distilled (bf16, padrao) 13min17s -- gguf-q6k 4min25s, "
                         "~3x mais rapido. dev = CFG real, mais lento, negative "
                         "prompt funciona (nao comparado neste teste).")
    ap.add_argument("--minimax-variant", default="fp8int8",
                    choices=["fp8int8", "w4a8", "gguf-q4km"],
                    help="variante do checkpoint MiniMax H3 (so importa com "
                         "--video-engine minimax). MEDIDO 2026-09-06, mesma "
                         "cena/seed, turbo 4 passos: fp8int8 (padrao) 11min6s, "
                         "sempre mais rapido que w4a8 (11min44s) e com "
                         "checkpoints menos comprimidos -- gguf-q4km 16min26s, "
                         "o mais lento dos tres, so vale se VRAM for o limite.")
    ap.add_argument("--consistency-threshold", type=float, default=None,
                    help="auditoria automatica de consistencia facial dos STILLS "
                         "(insightface) -- ver MEMORIAL 3.53. Sem isto, desligado.")
    ap.add_argument("--consistency-max-retries", type=int, default=2)
    ap.add_argument("--tts-engine", default=None, choices=["auto", "xtts", "qwen", "fish"],
                    help="motor de VOZ (nao confundir com --engine, LLM, nem --video-engine). "
                         "Sem isto, usa o default do synthesize_dialogue.py (\"auto\", XTTS/Qwen "
                         "-- ver MEMORIAL 3.55). \"fish\" precisa do servidor do fish-speech ja "
                         "no ar (START_API.ps1); nao descartamos xtts de proposito, escolha "
                         "explicita ate o fish amadurecer mais.")
    ap.add_argument("--camera-llm", action="store_true",
                    help="deixa o LLM (--engine) refinar movimento de camera/luz por "
                         "plano, dentro do vocabulario permitido pelo Estilo escolhido "
                         "-- ver CAMERA_STYLE_VOCAB em shot_plan.py. Opt-in: sem isto, "
                         "so a decupagem deterministica de sempre.")
    ap.add_argument("--character-sheet", action="store_true",
                    help="Fase A da consistencia de personagem (MEMORIAL 3.72): gera N "
                         "retratos candidatos por personagem (mesmo descritor do cast, "
                         "seeds diferentes) e escolhe o MEDOID -- maior similaridade "
                         "facial media aos outros candidatos -- como reference_image, em "
                         "vez do primeiro still que acontecer de sair no estagio normal. "
                         "Opt-in: sem isto, comportamento de sempre (primeiro still de "
                         "perto vira referencia). Roda antes dos stills, so precisa do "
                         "cast.json pronto.")
    ap.add_argument("--character-sheet-candidates", type=int, default=4,
                    help="quantos retratos candidatos gerar por personagem (--character-sheet).")
    from script_pipeline.generate_storyboards import available_loras_images
    ap.add_argument("--lora", default="", choices=[""] + available_loras_images(),
                    help="LoRA opcional aplicado aos STILLS e a character-sheet "
                         "(models/loras_images/, checkpoints de IMAGEM apenas -- "
                         "nao confundir com os LoRAs de video em models/loras/). "
                         "Sem isto, nenhum LoRA (comportamento de sempre).")
    ap.add_argument("--lora-strength", type=float, default=0.8)
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
        cmd_tts = ["-m", "script_pipeline.synthesize_dialogue",
                  "--run-dir", str(run), "--language", args.language]
        # Sem --tts-engine (None), cai no default do proprio synthesize_dialogue.py
        # ("auto" -- ver MEMORIAL 3.55). So passa a flag quando pedido explicito,
        # pra nao fixar aqui uma escolha que o modulo chamado ja decide sozinho.
        if args.tts_engine:
            cmd_tts += ["--engine", args.tts_engine]
        if not passo("4 tts", cmd_tts):
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
        if args.camera_llm:
            cmd += ["--camera-llm", "--engine", args.engine]
        if not passo("P decupagem", cmd):
            return 1

    if ate("sheet") and args.character_sheet:
        if (run / "characters" / "sheet_report.json").exists():
            print("[C character-sheet] ja feito, reaproveitando")
        else:
            from script_pipeline import gpu_watchdog
            gpu_watchdog.sweep_garbage(run, log=print)
            cmd_sheet = ["-m", "script_pipeline.character_sheet",
                        "--run-dir", str(run), "--apply",
                        "--n-candidates", str(args.character_sheet_candidates),
                        "--image-engine", args.image_engine,
                        "--width", str(args.width), "--height", str(args.height)]
            if args.lora:
                cmd_sheet += ["--lora", args.lora, "--lora-strength", str(args.lora_strength)]
            # Opcional: falhar aqui (Ollama fora do ar nao afeta isto, mas GPU/
            # ComfyUI podem) nao deve derrubar a corrida -- sem sheet, cai no
            # comportamento de sempre (primeiro still vira referencia).
            passo("C character-sheet", cmd_sheet, obrigatorio=False)

    if ate("stills"):
        from script_pipeline import gpu_watchdog
        gpu_watchdog.sweep_garbage(run, log=print)
        cmd_stills = ["-m", "script_pipeline.render_shots_stage",
                     "--run-dir", str(run), "--width", str(args.width),
                     "--height", str(args.height), "--stills-only",
                     "--image-engine", args.image_engine]
        if args.consistency_threshold is not None:
            cmd_stills += ["--consistency-threshold", str(args.consistency_threshold),
                           "--consistency-max-retries", str(args.consistency_max_retries)]
        if args.lora:
            cmd_stills += ["--lora", args.lora, "--lora-strength", str(args.lora_strength)]
        if not passo("5-D stills", cmd_stills):
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
        from script_pipeline import gpu_watchdog
        gpu_watchdog.sweep_garbage(run, log=print)

        # REINICIAR ANTES DO VIDEO nao e higiene, e o que separa 8,5 min de 55.
        # O ComfyUI que gerou os stills esta com o alocador moldado para o FLUX,
        # e o LTX 2.5 nao cabe nesse molde. Ver stop_comfyui().
        from script_pipeline.generate_storyboards import stop_comfyui
        stop_comfyui(log=lambda m: print(f"[reinicio] {m}", flush=True))

        # MESMA disciplina, agora pro servidor do MiniMax H3 (8189).
        # MEDIDO 2026-09-02: rodando 4 corridas em sequencia sem NUNCA reiniciar
        # esse servidor (ele fica de pe entre corridas -- `ensure_server()` so
        # confere se a porta ja responde e reaproveita), um plano de 4,15s
        # travou 22min34s no PASSO 0 da amostragem antes de destravar sozinho e
        # terminar normal -- mesma assinatura de trava do DynamicVRAM/aimdo ja
        # documentada pro LTX (MEMORIAL 3.41/3.42), so que nunca medida aqui pro
        # MiniMax. `stop_comfyui()` mata por PORTA (netstat+taskkill), nao pelo
        # handle em memoria de `minimax_h3_backend.shutdown_server()` -- que so
        # funciona DENTRO do mesmo processo que subiu o servidor, e aqui cada
        # estagio e um subprocesso novo. `minimax_h3_backend.generate()` sobe o
        # servidor de novo sozinho quando precisar (~30s de boot, medido) --
        # muito mais barato que os 22 min perdidos.
        if args.video_engine == "minimax":
            stop_comfyui(port=8189, log=lambda m: print(f"[reinicio-minimax] {m}", flush=True))

        # `passo()` roda o subprocesso com o `os.environ` ATUAL herdado (nao
        # passa `env=` proprio) -- setar aqui e o bastante pro estagio de video
        # (um processo Python novo) ler a variante certa quando importar
        # ltx25_backend/minimax_h3_backend, que le a variante uma vez na carga
        # do modulo. Sempre seta as duas: so a que bate com --video-engine tem
        # efeito, a outra fica ociosa sem custo.
        os.environ["LTX25_VARIANT"] = args.ltx_variant
        os.environ["MINIMAX_H3_VARIANT"] = args.minimax_variant

        # A passada de VIDEO nao gera imagem -- ela le os stills pelo manifesto,
        # por indice. O motor vai junto so para as duas passadas descreverem a
        # mesma corrida no log; nao ha decisao pendurada nele aqui.
        if not passo("5-D video", ["-m", "script_pipeline.render_shots_stage",
                                   "--run-dir", str(run), "--width", str(args.width),
                                   "--height", str(args.height), "--videos-only",
                                   "--fps", str(args.fps),
                                   "--image-engine", args.image_engine,
                                   "--engine", args.video_engine]):
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
