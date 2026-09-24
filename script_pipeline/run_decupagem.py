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
import json
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
PARADAS = ["parse", "cast", "emocao", "tts", "estrutura", "plano", "motion", "sheet",
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


def _gate_step(nome: str, cmd: list, obrigatorio: bool = True) -> bool:
    """passo() que libera o ComfyUI antes do gate: o Qwen3-VL ocupa ~22 GB e nao
    cabe ao lado do FLUX/LTX residente (o Ollama cairia em CPU parcial, muito lento)."""
    if any("visual_continuity_audit" in str(c) for c in cmd):
        from script_pipeline.generate_storyboards import stop_comfyui
        stop_comfyui(log=lambda m: print(f"[gate] {m}", flush=True))
        # Motores de imagem externos ficam residentes na 3090; libere-os antes
        # do Qwen3-VL, que também usa a GPU para o gate visual.
        try:
            import zimage_backend
            from script_pipeline import gpu_watchdog
            gpu_watchdog.free_port(zimage_backend.ZIMAGE_PORT, log=lambda m: print(f"[gate] {m}", flush=True))
            import qwen_image21_engine
            for port in qwen_image21_engine.all_ports():
                gpu_watchdog.free_port(port, log=lambda m: print(f"[gate] {m}", flush=True))
            # Fish Speech (TTS) tambem fica residente (~20 GB) e disputa com o Qwen3-VL do
            # gate -- mesmo achado do estagio de video (ver render_shots.py, 2026-09-22).
            from script_pipeline.dialogue_tts import FISH_API_URL
            fish_port = int(FISH_API_URL.rsplit(":", 1)[-1].split("/")[0])
            gpu_watchdog.free_port(fish_port, log=lambda m: print(f"[gate] {m}", flush=True))
        except Exception as exc:
            print(f"[gate] aviso: nao consegui liberar um servidor externo de imagem ({exc})", flush=True)
    return passo(nome, cmd, obrigatorio=obrigatorio)


def _visual_gate(run, args, *, stage: str, regen_base: list, nome: str) -> bool:
    """Gate visual com regeneracao dos reprovados (ver gate_retry.py)."""
    from script_pipeline.gate_retry import gate_with_retries
    from script_pipeline.visual_continuity_audit import blocked_shots
    audit_cmd = ["-m", "script_pipeline.visual_continuity_audit",
                 "--run-dir", str(run), "--stage", stage,
                 "--model", args.visual_audit_model,
                 "--perception-model", args.visual_perception_model]
    ok = gate_with_retries(
        run, stage=stage, audit_cmd=audit_cmd,
        regen_cmd=lambda spec, seed: regen_base + ["--only-shots", spec],  # semente via seed_overrides.json
        run_step=lambda n, cmd, obrigatorio=True: _gate_step(f"{nome} {n}", cmd, obrigatorio),
        blocked_fn=lambda: blocked_shots(run, stage),
        base_seed=1234, max_retries=max(0, args.visual_max_retries),
        notes_fn=((lambda shots: __import__("script_pipeline.gate_retry", fromlist=["x"])
                   .write_repair_notes(run, stage, shots)) if stage == "stills" else None),
        total_shots=len(json.loads((Path(run) / 'parse' / 'shot_plan.json').read_text(encoding='utf-8')).get('shots', [])),
        expand_fn=((lambda shots: __import__("script_pipeline.location_master", fromlist=["x"])
                    .expand_blocked(run, shots)) if stage == "stills" else None),
        log=lambda m: print(m, flush=True))
    if not ok and args.visual_unresolved == "continue":
        print(f"[run_decupagem] AVISO: gate {stage} com planos reprovados; seguindo "
              "(--visual-unresolved continue). Ver shots/visual_gate_*_retries.json.",
              file=sys.stderr)
        return True
    return ok


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
    ap.add_argument("--spatial-spec", default=None,
                    help="JSON de blocking 3D por ID de plano; exige projeto persistente e FLUX Klein")
    ap.add_argument("--spatial-denoise", type=float, default=.65,
                    help="força da transformação RGB do blocking espacial")
    ap.add_argument("--reuse-plan", action="store_true",
                    help="preserva o shot_plan existente; obrigatório quando um spec espacial já foi criado para seus IDs")
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
    ap.add_argument("--video-engine", default="ltx", choices=["ltx", "minimax", "longcat", "wan"],
                    help="motor de VIDEO (nao confundir com --engine, que e o LLM de "
                         "estrutura). ltx = LTX 2.5, condicionado pela fala sintetizada "
                         "(TTS + lipsync + mix rodam normalmente). minimax = MiniMax H3 -- "
                         "fala e lip-sync NATIVOS a partir do texto do video_prompt; os "
                         "estagios 6/7 (lipsync/mix) viram passthrough pra esses planos, "
                         "ja que o audio ja vem pronto no clipe. wan = Wan 2.2 TI2V 5B, MESMO "
                         "ComfyUI do LTX (8188) -- video SEMPRE MUDO, TTS+lipsync+mix rodam "
                         "normalmente (como no ltx). Validado com GPU real 2026-09-18.")
    ap.add_argument("--ltx-variant", default="w4a8-v10",
                    choices=["w4a8-v10", "distilled", "dev", "gguf-q6k"],
                    help="variante do checkpoint LTX 2.5 (so importa com "
                         "--video-engine ltx). Padrao w4a8-v10 desde 2026-09-12: "
                         "destilado em 4 bits, ~1,6x mais rapido que o distilled "
                         "(bf16) com o modelo ja carregado, qualidade julgada "
                         "maior em 3 de 3 comparacoes, validado nesta cadeia com "
                         "I2V + fala (MEMORIAL 3.77). distilled = bf16, padrao "
                         "anterior. dev = CFG real, mais lento, negative prompt "
                         "funciona.")
    ap.add_argument("--minimax-variant", default="fp8int8",
                    choices=["fp8int8", "w4a8", "gguf-q4km"],
                    help="variante do checkpoint MiniMax H3 (so importa com "
                         "--video-engine minimax). MEDIDO 2026-09-06, mesma "
                         "cena/seed, turbo 4 passos: fp8int8 (padrao) 11min6s, "
                         "sempre mais rapido que w4a8 (11min44s) e com "
                         "checkpoints menos comprimidos -- gguf-q4km 16min26s, "
                         "o mais lento dos tres, so vale se VRAM for o limite.")
    ap.add_argument("--minimax-ref-audio", action="store_true",
                    help="MiniMax H3: manda o WAV do TTS ja sintetizado para cada fala como "
                         "ref_audios (timbre/cadencia reais -- MEMORIAL 3.74). So com "
                         "--video-engine minimax. Opt-in, validado so com uma fala isolada.")
    ap.add_argument("--minimax-no-still", action="store_true",
                    help="MiniMax H3: NAO manda o still do plano como referencia -- so o TEXTO "
                         "guia a identidade (a sheet do personagem, se existir, continua indo, "
                         "pra nao perder a ancora ENTRE planos). So com --video-engine minimax.")
    ap.add_argument("--minimax-chain-max-seconds", type=float, default=None,
                    help="MiniMax H3: planos mais longos que isto viram sub-planos curtos "
                         "encadeados por ultimo-frame + sheet, em vez de uma chamada longa "
                         "instavel (MEDIDO: ~16s+ ja trava 30min+). 6.0 e o ponto de partida "
                         "validado. Sem isto (padrao), sempre uma chamada so. So com "
                         "--video-engine minimax.")
    ap.add_argument("--ltx-no-still", action="store_true",
                    help="LTX 2.5: NAO manda o still como image_path -- T2V puro, so texto. "
                         "So com --video-engine ltx.")
    ap.add_argument("--ltx-chain-max-seconds", type=float, default=None,
                    help="LTX 2.5: planos mais longos que isto viram sub-planos encadeados por "
                         "ultimo-frame (mesmo principio do continuous_chain.py). Sem isto "
                         "(padrao), sempre uma chamada so. So com --video-engine ltx.")
    ap.add_argument("--consistency-threshold", type=float, default=0.35,
                    help="auditoria automatica de consistencia facial dos STILLS "
                         "(insightface) -- ver MEMORIAL 3.53. Fix #3 (avaliacao visual "
                         "2026-09-17): agora GATE por padrao, nao so relatorio -- um "
                         "still abaixo do limiar contra a reference_image do personagem "
                         "e regerado (ate --consistency-max-retries vezes) antes de "
                         "aceitar o de maior score. 0.35 e o mesmo limiar do "
                         "clip_identity_audit (ArcFace). --no-consistency-check desliga.")
    ap.add_argument("--consistency-max-retries", type=int, default=2)
    ap.add_argument("--no-consistency-check", action="store_true",
                    help="desliga o gate de consistencia facial dos stills (volta ao "
                         "comportamento antigo: gera uma vez, nao compara com a referencia).")
    ap.add_argument("--visual-audit-model", default="qwen3-vl:30b",
                    help="LLM que julga a percepcao visual contra o contrato e bloqueia "
                         "still/video fora dele. Padrao: qwen3-vl:30b. Diferente da auditoria textual e do "
                         "InsightFace: verifica locacao, objetos, falante, texto inventado "
                         "e regras opt-in como a aeronave do Voo 702.")
    ap.add_argument("--visual-perception-model", default="qwen3-vl:30b",
                    help="VLM usado somente para descrever pixels sem ver o contrato. "
                         "O mesmo Qwen3-VL faz uma segunda chamada textual para decidir. "
                         "Separar as chamadas evita confirmacao do prompt.")
    ap.add_argument("--visual-max-retries", type=int, default=2,
                    help="quando o gate visual reprova planos, refaz SO eles com outra seed e "
                         "reaudita, ate N rodadas (0 = comportamento antigo: so bloqueia). "
                         "Ver gate_retry.py.")
    ap.add_argument("--visual-unresolved", default="block", choices=["block", "continue"],
                    help="o que fazer com planos que seguem reprovados depois das rodadas: "
                         "block (padrao) para a corrida; continue segue e deixa o registro "
                         "em shots/visual_gate_*_retries.json.")
    ap.add_argument("--max-speech-seconds", type=float, default=6.0,
                    help="fala mais longa que isto vira varios planos de fala (mesmo falante "
                         "em close, audio recortado em silencio). 0 desliga. Ver speech_split.py.")
    ap.add_argument("--no-master-audio", action="store_true",
                    help="nao normaliza o audio final (-16 LUFS / -1,5 dBTP) na montagem.")
    ap.add_argument("--room-tone-db", type=float, default=None,
                    help="ambiencia continua sob o filme inteiro, em dB (ex.: -46). "
                         "Esconde as emendas de audio entre planos. Desligado por padrao.")
    ap.add_argument("--visual-diagnostic", action="store_true",
                    help="modo diagnostico: RODA o gate visual e grava o relatorio, mas nunca "
                         "bloqueia nem regenera, e a corrida para em no maximo `animatic` "
                         "(video, lipsync e montagem sao recusados). Para inspecionar um "
                         "resultado reprovado sem aprova-lo. Diferente de --no-visual-audit, "
                         "que nem roda o gate.")
    ap.add_argument("--no-visual-audit", action="store_true",
                    help="desliga explicitamente os gates visuais multimodais antes do "
                         "video e antes do lipsync/mix. Nao recomendado para producao.")
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
    ap.add_argument("--motion-conditioning", action="store_true",
                    help="compila a decupagem em movimento por personagem: acrescenta um "
                         "motion_prompt curto ao LTX/MiniMax e salva parse/motion_plan.json "
                         "como contrato para um adaptador MotionBricks/retarget. Nao exige "
                         "o runtime 3D e e opt-in para preservar prompts ja validados.")
    ap.add_argument("--character-sheet", action="store_true",
                    help="Forca a Fase A da consistencia de personagem (MEMORIAL 3.72) mesmo "
                         "com 1 personagem so no cast. Normalmente desnecessario: com 2+ "
                         "personagens ela liga sozinha (ver --no-character-sheet).")
    ap.add_argument("--no-character-sheet", action="store_true",
                    help="Desliga a Fase A mesmo com 2+ personagens no cast. Volta ao "
                         "comportamento antigo (primeiro still de perto vira referencia). "
                         # BUGFIX (auditoria externa 2026-09-16, achado #3): com 2+
                         # personagens no cast, o primeiro still de perto travar como
                         # referencia (design de sempre) e o que produz identidade
                         # "roubada" entre coadjuvantes -- gera N candidatos e escolhe
                         # o medoid por padrao nesse caso; opt-out explicito aqui.
                         "Use se character_sheet.py estiver indisponivel/lento demais.")
    ap.add_argument("--character-sheet-candidates", type=int, default=4,
                    help="quantos retratos candidatos gerar por personagem (--character-sheet).")
    from script_pipeline.generate_storyboards import available_loras_images
    ap.add_argument("--lora", default="", choices=[""] + available_loras_images(),
                    help="LoRA opcional aplicado aos STILLS e a character-sheet "
                         "(models/loras_images/, checkpoints de IMAGEM apenas -- "
                         "nao confundir com os LoRAs de video em models/loras/). "
                         "Sem isto, nenhum LoRA (comportamento de sempre).")
    ap.add_argument("--lora-strength", type=float, default=0.8)
    # LoRAs de VIDEO do LTX 2.5 (pedido do usuario 2026-09-12) -- so na passada de video,
    # nunca nos stills. Catalogo, forcas, gatilhos e compatibilidade 2.3->2.5 em
    # ltx_loras.py; montagem das referencias IC em script_pipeline/ic_references.py.
    ap.add_argument("--video-lora", action="append", default=[], metavar="CHAVE[:FORCA]",
                    help="LoRA comum na passada de VIDEO (repita para empilhar), ex.: "
                         "better-human-motion:0.6. Nao confundir com --lora, que e dos stills.")
    ap.add_argument("--ic-reference", default="off", choices=["off", "ingredients", "msr"],
                    help="IC-LoRA de referencia no video: ingredients = folha com a character "
                         "sheet + locacao; msr = sequencia MSR V2 (sujeitos + cenario). Opt-in; "
                         "custo de tokens maior (a guia entra no contexto do clipe).")
    ap.add_argument("--ic-lora", default=None, help="troca o IC-LoRA padrao do modo")
    ap.add_argument("--ic-strength", type=float, default=1.0)
    ap.add_argument("--ic-guide-strength", type=float, default=1.0)
    # Lip-sync e pos-producao por IC-LoRA V2V (pedido do usuario 2026-09-13, depois de
    # o filme de teste perder sincronia). Tudo opt-in, forcas com o padrao do model card.
    ap.add_argument("--lipsync-engine", default="auto",
                    choices=["auto", "latentsync", "wav2lip", "dubit", "none"],
                    help="auto = LatentSync, Wav2Lip de reserva (como sempre); dubit = IC-LoRA "
                         "DubIt do LTX; none = sem lip-sync (fica a boca que o LTX gerou)")
    ap.add_argument("--dialogue-framing", default="auto", choices=["auto", "close", "livre"],
                    help="enquadramento dos planos de FALA: auto = so close abaixo de 704 px de "
                         "altura (onde o rosto passa de 300 px e o lip-sync aprova); close = sempre; "
                         "livre = escada do estilo. extreme_close nunca, em nenhum modo.")
    ap.add_argument("--dubit-strength", type=float, default=1.0)
    ap.add_argument("--dubit-guide-strength", type=float, default=1.0)
    ap.add_argument("--dubit-audio", default="congelar", choices=["congelar", "gerar"])
    ap.add_argument("--post-deblur", action="store_true",
                    help="pos-producao: Deblur 2.5 em todo clipe mixado (audio intacto)")
    ap.add_argument("--post-deblur-strength", type=float, default=1.0)
    ap.add_argument("--post-strip-subtitles", action="store_true",
                    help="Fix #5 (avaliacao visual 2026-09-17): corta a faixa de legenda "
                         "queimada do LTX distilled/w4a8-v10 (CFG=1, negative prompt sem "
                         "efeito) de TODOS os clipes uniformemente -- ver postprod_v2v.py "
                         "e strip_subtitles.py. So ligue depois de confirmar visualmente "
                         "que ha legenda nesta corrida; alternativa que evita em vez de "
                         "reparar: --ltx-variant dev.")
    ap.add_argument("--post-strip-subtitles-keep", type=float, default=None,
                    help="fracao da altura a manter (padrao calibrado no LTX 2.3 -- "
                         "confira com strip_subtitles.py --measure antes de confiar no 2.5).")
    ap.add_argument("--post-upscale", action="store_true",
                    help="pos-producao: Pixel-Upscaler 2.5, filme inteiro a x2 (bem mais lento)")
    ap.add_argument("--post-upscale-strength", type=float, default=1.0)
    # Musica de fundo continua sob o filme inteiro (pedido do usuario
    # 2026-09-12): os motores de video so geram trilha nas cenas de dialogo
    # (audio_conditioning), entao o filme montado tem trechos sem musica
    # nenhuma. Misturada em assemble_final.py DEPOIS da concatenacao -- uma
    # trilha continua, nao reiniciada a cada corte. `--music-path` explicito
    # ganha de `--music-dir` (sorteia um arquivo da pasta).
    ap.add_argument("--music-path", default=None,
                    help="arquivo de musica para tocar sob o filme inteiro")
    ap.add_argument("--music-dir", default=None,
                    help="pasta de musicas -- sorteia um arquivo se --music-path nao for dado")
    ap.add_argument("--music-volume", type=float, default=None,
                    help="nivel da musica antes do ducking (0-1, padrao 0.18)")
    args = ap.parse_args()

    if args.visual_diagnostic:
        if args.no_visual_audit:
            ap.error("--visual-diagnostic roda o gate; nao combine com --no-visual-audit")
        if PARADAS.index(args.ate) > PARADAS.index("animatic"):
            ap.error("--visual-diagnostic nao libera video nem montagem: use --ate animatic "
                     "(ou antes). Um resultado reprovado so pode ser inspecionado, nunca entregue.")
        args.visual_unresolved = "continue"
        args.visual_max_retries = 0

    # Entrega final e fail-closed: os dois switches abaixo existem para
    # diagnósticos/previews, nunca para declarar um filme de produção pronto.
    # Em 2026-09-22 os gates deixaram planos sem aprovação, mas a corrida foi
    # concluída ao usar continue. Faça a exceção explícita parando antes do final.
    if args.ate == "final" and args.no_visual_audit:
        ap.error("--no-visual-audit não pode ser usado com --ate final; pare em --ate animatic para diagnóstico")
    if args.ate == "final" and args.visual_unresolved == "continue":
        ap.error("--visual-unresolved continue não pode ser usado com --ate final; use --ate animatic para diagnóstico")

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

    from script_pipeline.production_project import restore_source, conform_plan, sync_media
    restore_source(run, engine=args.engine)

    if ate("cast") and not (run / "characters" / "cast.json").exists():
        # --engine: o descritor visual sai do mesmo motor que o resto da cadeia.
        # Sem ele o descritor e um recorte do texto de acao, que em roteiro de
        # prosa corrida devolve enredo em vez de aparencia -- e esse texto vai
        # colado em TODO still e TODO clipe. Ver cast_characters.
        if not passo("2 cast", ["-m", "script_pipeline.cast_characters",
                                "--run-dir", str(run), "--engine", args.engine]):
            return 1

    # Perfis de producao sao opt-in por projeto. O Voo 702 recebe aqui suas
    # fotos nominais, locacoes e contrato de aeronave; nenhum outro roteiro
    # herda a regra de permanecer em voo.
    from script_pipeline.voo702_setup import configure_if_matching
    if configure_if_matching(run):
        print("[perfil] Voo 702 aplicado: referencias nominais e continuidade especifica.",
              flush=True)

    if ate("emocao"):
        cmd = ["-m", "script_pipeline.emotion_director", "--run-dir", str(run),
               "--engine", args.engine]
        if args.recast:
            cmd.append("--recast")
        passo("E emocao", cmd, obrigatorio=False)

    # Sempre roda: o synthesize_dialogue guarda chave por fala (texto + emocao + voz +
    # motor) e so refaz o que mudou. VISTO 2026-09-13: pular quando lines.json existia
    # fazia a emocao dirigida pelo [E] nunca chegar a voz ("Rode synthesize_dialogue de
    # novo" ficava so no log).
    if ate("tts"):
        from script_pipeline.production_project import apply_voice_overrides
        apply_voice_overrides(run)
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
        if args.reuse_plan and (run / "parse" / "shot_plan.json").exists():
            print("[P decupagem] shot_plan existente preservado (--reuse-plan)")
        else:
            cenas = run / "parse" / "scenes_enriched.json"
            if not cenas.exists():
                cenas = run / "parse" / "scenes.json"
            cmd = ["-m", "script_pipeline.shot_plan", "--scenes", str(cenas),
               "--structure", str(run / "parse" / "story_structure.json"),
               "--dialogue", str(run / "dialogue" / "lines.json"),
               "--cast", str(run / "characters" / "cast.json"),
               "--style", args.style, "--fps", str(args.fps),
               "--height", str(args.height), "--dialogue-framing", args.dialogue_framing,
               "--out", str(run / "parse" / "shot_plan.json")]
            if args.style_changes:
                cmd += ["--style-changes", args.style_changes]
            if args.video_engine == "minimax":
            # BUGFIX (achado critico do proprio usuario, corrigido manualmente
            # antes desta sessao de fixes -- ver MEMORIAL): sem isto o MiniMax
            # H3 gera fala NATIVA a partir so do video_prompt, sem a fala real
            # embutida como texto -- "nao tem um unico audio correto", ele
            # inventa palavras. --include-quotes bota a fala literal no
            # video_prompt. LTX nao precisa (usa audio_conditioning, que ja
            # carrega o WAV real do TTS) -- so liga aqui pra minimax.
                cmd += ["--include-quotes"]
            if args.camera_llm:
                cmd += ["--camera-llm", "--engine", args.engine]
            cmd += ["--max-speech-seconds", str(args.max_speech_seconds)]
            if not passo("P decupagem", cmd):
                return 1
            conform_plan(run)
        # Relatorio de ritmo (nunca bloqueia): planos de acao longos/curtos demais para o
        # enquadramento. So mede -- ver script_pipeline/pacing_audit.py.
        passo("P-pacing ritmo", ["-m", "script_pipeline.pacing_audit", "--run-dir", str(run)],
              obrigatorio=False)

    if args.spatial_spec and ate("stills"):
        if args.image_engine != "flux":
            raise ValueError("--spatial-spec requires --image-engine flux")
        from script_pipeline.spatial_pipeline import attach_to_run
        attach_to_run(run, args.spatial_spec, denoise=args.spatial_denoise)

    # MotionBricks e um gerador de movimento esqueletico, nao um endpoint de
    # video. Esta etapa produz ao mesmo tempo o condicionamento textual que os
    # dois motores aceitam hoje e o score espacial que um adaptador 3D pode
    # executar depois. Roda depois da decupagem porque depende de sujeito,
    # co-sujeito, lado de tela e beat ja resolvidos.
    # Toda geração de vídeo recebe o contrato semântico de movimento. O flag
    # continua útil para produzir/inspecionar o score antes da etapa de vídeo,
    # mas não deixa mais a rota de entrega renderizar uma perseguição como idle.
    motion_required_for_video = PARADAS.index(args.ate) >= PARADAS.index("render")
    if ate("motion") and (args.motion_conditioning or motion_required_for_video):
        plan_path = run / "parse" / "shot_plan.json"
        if not plan_path.exists():
            print("[M movimento] shot_plan.json ausente; nao ha decupagem para condicionar", file=sys.stderr)
            return 1
        print("[M movimento] contrato semântico obrigatório para vídeo; "
              "compile uma vez no shot_plan e preserve sua cobertura.", flush=True)
        if not passo("M movimento", ["-m", "script_pipeline.motion_conditioner",
                                     "--plan", str(plan_path), "--apply"],
                     obrigatorio=motion_required_for_video):
            print("[run_decupagem] render bloqueado: não foi possível compilar os movimentos.",
                  file=sys.stderr)
            return 1
        # Pre-visualizacao BARATA (sem GPU de difusao) do blocking/posicionamento resolvido --
        # pedido do usuario depois do CERCO EM SEUL sair ruim com still/video reais: pegar erro de
        # POSICIONAMENTO (personagens sobrepostos, perseguicao no mesmo lado de tela, primitiva sem
        # parceiro) ANTES de gastar still ou video. So relata; nunca bloqueia.
        if (run / "parse" / "motion_plan.json").exists():
            from script_pipeline.blocking_preview import build_contact_sheet
            try:
                _, avisos = build_contact_sheet(run)
                if avisos:
                    print(f"[blocking-preview] {len(avisos)} aviso(s) de posicionamento -- "
                          f"veja shots/blocking_preview.png e shots/blocking_preview_avisos.json:",
                          flush=True)
                    for aviso in avisos:
                        print(f"  - {aviso}", flush=True)
                else:
                    print("[blocking-preview] sem avisos geometricos.", flush=True)
            except Exception as exc:
                print(f"[blocking-preview] aviso: nao consegui gerar o contact sheet ({exc})",
                      flush=True)

    # Parse, direção emocional, estrutura e câmera já terminaram de usar o
    # Ollama. Liberar o LLM antes de subir FLUX/SD evita que dois modelos
    # disputem a mesma VRAM durante folhas e stills. A auditoria visual carrega
    # seu VLM novamente depois que as imagens estiverem prontas.
    if ate("sheet"):
        from script_pipeline.ollama_runtime import unload_all
        unload_all(log=print)

    quer_sheet = args.character_sheet or args.no_character_sheet
    if ate("sheet") and not args.no_character_sheet and not quer_sheet:
        cast_path = run / "characters" / "cast.json"
        if cast_path.exists():
            try:
                n_personagens = len(json.loads(cast_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                n_personagens = 0
            if n_personagens >= 2:
                print(f"[C character-sheet] {n_personagens} personagens no cast -- "
                      "ligando por padrao (--no-character-sheet desliga)")
                quer_sheet = True

    if ate("sheet") and quer_sheet and not args.no_character_sheet:
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
        if not args.no_consistency_check and args.consistency_threshold is not None:
            cmd_stills += ["--consistency-threshold", str(args.consistency_threshold),
                           "--consistency-max-retries", str(args.consistency_max_retries)]
        if args.lora:
            cmd_stills += ["--lora", args.lora, "--lora-strength", str(args.lora_strength)]
        if not passo("5-D stills", cmd_stills):
            return 1
        from script_pipeline.production_project import sync_assets
        sync_assets(run)

        # PORTAO DE REVISAO (2026-09-10, pedido do usuario depois de dois
        # defeitos reais -- rosto duplicado num still de 2 personagens e
        # drift de estilo -- passarem direto pro video sem aviso nenhum).
        # So reporta: nunca interrompe a corrida sozinho, quem decide se
        # revisa a galeria antes de --ate render e a pessoa. Ver
        # storyboard_audit.py.
        from script_pipeline.storyboard_audit import build_report, format_summary
        relatorio_stills = build_report(run)
        print(format_summary(relatorio_stills))
        if relatorio_stills:
            (run / "shots" / "storyboard_audit.json").write_text(
                json.dumps(relatorio_stills, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_visual_audit:
            if not _visual_gate(run, args, stage="stills", regen_base=cmd_stills,
                                nome="5-E Qwen3-VL stills"):
                print("[run_decupagem] video bloqueado: o Qwen3-VL nao aprovou os pixels dos stills.",
                      file=sys.stderr)
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
        if ok and (run / "shots" / "animatic.mp4").exists():
            ok = passo("R-A auditoria do animatic",
                       ["-m", "script_pipeline.animatic_audit", "--run-dir", str(run)],
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
        cmd_video = ["-m", "script_pipeline.render_shots_stage",
                    "--run-dir", str(run), "--width", str(args.width),
                    "--height", str(args.height), "--videos-only",
                    "--fps", str(args.fps),
                    "--image-engine", args.image_engine,
                    "--engine", args.video_engine]
        if args.video_engine == "minimax" and args.minimax_ref_audio:
            cmd_video.append("--minimax-ref-audio")
        if args.video_engine == "minimax" and args.minimax_no_still:
            cmd_video.append("--minimax-no-still")
        if args.video_engine == "minimax" and args.minimax_chain_max_seconds:
            cmd_video += ["--minimax-chain-max-seconds", str(args.minimax_chain_max_seconds)]
        if args.video_engine == "ltx" and args.ltx_no_still:
            cmd_video.append("--ltx-no-still")
        if args.video_engine == "ltx" and args.ltx_chain_max_seconds:
            cmd_video += ["--ltx-chain-max-seconds", str(args.ltx_chain_max_seconds)]
        if args.video_engine == "ltx":
            for valor in args.video_lora:
                cmd_video += ["--video-lora", valor]
            if args.ic_reference != "off":
                cmd_video += ["--ic-reference", args.ic_reference,
                              "--ic-strength", str(args.ic_strength),
                              "--ic-guide-strength", str(args.ic_guide_strength)]
                if args.ic_lora:
                    cmd_video += ["--ic-lora", args.ic_lora]
        if not passo("5-D video", cmd_video):
            return 1
        sync_media(run)
        if not args.no_visual_audit:
            if not _visual_gate(run, args, stage="video", regen_base=cmd_video,
                                nome="5-F Qwen3-VL clipes"):
                print("[run_decupagem] lipsync e montagem bloqueados: o Qwen3-VL nao aprovou "
                      "inicio, meio e fim dos clipes.", file=sys.stderr)
                return 1
        # So avisa, nunca interrompe: corte seco dentro de um clipe unico e a guia do
        # IC-LoRA agindo como keyframe (VISTO 2026-09-13 com MSR no w4a8). Ver
        # guide_leak_audit.py.
        if args.video_engine == "ltx" and args.ic_reference != "off":
            passo("5-D auditoria de guia", ["-m", "script_pipeline.guide_leak_audit",
                                            "--run-dir", str(run)], obrigatorio=False)

    if ate("lipsync"):
        cmd_lip = ["-m", "script_pipeline.lipsync_scenes", "--run-dir", str(run),
                   "--engine", args.lipsync_engine]
        if args.lipsync_engine == "dubit":
            cmd_lip += ["--dubit-strength", str(args.dubit_strength),
                        "--dubit-guide-strength", str(args.dubit_guide_strength),
                        "--dubit-audio", args.dubit_audio]
        passo("6 lipsync", cmd_lip, obrigatorio=False)
    if ate("mix"):
        passo("7 mix", ["-m", "script_pipeline.mix_audio",
                        "--run-dir", str(run)], obrigatorio=False)
        if args.post_deblur or args.post_upscale or args.post_strip_subtitles:
            cmd_post = ["-m", "script_pipeline.postprod_v2v", "--run-dir", str(run)]
            if args.post_deblur:
                cmd_post += ["--deblur", "--deblur-strength", str(args.post_deblur_strength)]
            if args.post_upscale:
                cmd_post += ["--upscale", "--upscale-strength", str(args.post_upscale_strength)]
            if args.post_strip_subtitles:
                cmd_post += ["--strip-subtitles"]
                if args.post_strip_subtitles_keep:
                    cmd_post += ["--strip-subtitles-keep", str(args.post_strip_subtitles_keep)]
            passo("7b pos-producao", cmd_post, obrigatorio=False)
    if ate("final"):
        # Identidade é auditada depois do lipsync/mix (último estágio que pode
        # substituir clipes) e antes da montagem. Alertas não podem chegar à
        # entrega; ausência de rosto em plano aberto permanece não conclusiva.
        if not passo("8a identidade (gate)", ["-m", "script_pipeline.clip_identity_audit",
                                               "--run-dir", str(run), "--strict"]):
            print("[run_decupagem] montagem bloqueada: identidade visual reprovada.",
                  file=sys.stderr)
            return 1
        cmd_montagem = ["-m", "script_pipeline.assemble_final", "--run-dir", str(run)]
        if args.music_path:
            cmd_montagem += ["--music-path", args.music_path]
        if args.music_dir:
            cmd_montagem += ["--music-dir", args.music_dir]
        if args.music_volume is not None:
            cmd_montagem += ["--music-volume", str(args.music_volume)]
        if args.no_master_audio:
            cmd_montagem += ["--no-master"]
        if args.room_tone_db is not None:
            cmd_montagem += ["--room-tone-db", str(args.room_tone_db)]
        if not passo("8 montagem", cmd_montagem):
            return 1
        from script_pipeline.production_project import project_for
        if project_for(run):
            from script_pipeline.production_post import conform_movie
            print('[projeto] conformando prévia editorial e exportando stems', flush=True)
            conform_movie(run)
        # Relatorios de continuidade (2026-09-13, auditoria de scripts). So MEDEM: nada e
        # reescrito. Identidade do personagem entre clipes (ArcFace) e defeito temporal no
        # filme montado (video_doctor, com deteccao de corte ligada -- as fronteiras entre
        # planos nao viram falso positivo, MEMORIAL 3.36.1). O reparo continua manual,
        # revisando as tiras antes (MEMORIAL 3.19).
        filme = run / "final" / "movie.mp4"
        if filme.exists():
            passo("8c doctor", [str(ROOT / "video_doctor.py"), "analyze", str(filme),
                                "--plan", str(run / "final" / "doctor_plan.json"),
                                "--previews", str(run / "final" / "doctor_previews")],
                  obrigatorio=False)
        if not passo("9 verificacao (gate de entrega)", ["-m", "script_pipeline.verify_output",
                                                          "--run-dir", str(run), "--strict"]):
            print("[run_decupagem] arquivo montado, mas reprovado na verificação técnica; "
                  "não é uma entrega aprovada.", file=sys.stderr)
            return 1

    print(f"\n[run_decupagem] concluido em {run}")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        # Vale para sucesso, gate bloqueado, erro e Ctrl+C. O servidor Ollama
        # permanece ligado; apenas os modelos residentes deixam RAM/VRAM.
        from script_pipeline.ollama_runtime import unload_all
        unload_all(log=print)
    raise SystemExit(exit_code)
