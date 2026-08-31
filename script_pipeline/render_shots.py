"""Executa um `shot_plan`: para cada plano, gera o STILL e depois o VÍDEO
condicionado por ele. É a ponte que faltava entre decupagem e imagem.

POR QUE DOIS ESTÁGIOS POR PLANO, E NÃO UM PROMPT SÓ

MEDIDO 2026-08-26 (MEMORIAL.md §3.21): o LTX trata vocabulário de câmera como
DESTINO, não como estado. Pedindo close-up, o frame 0 sai em plano geral e só o
frame 96 chega ao close-up. Num plano de 3 segundos ele nunca chega.

Então o enquadramento tem de estar fixado ANTES da difusão de vídeo começar:

    storyboard_prompt --> FLUX --> still (fixa enquadramento, ângulo, lado)
                                     |
                                     v  imagem de condicionamento, strength 1.0
    video_prompt ------------------> LTX --> clipe (só movimento e ação)

Isto depende de duas correções anteriores para funcionar: `strength` do
`LTXVImgToVideoInplace`, que vinha zerado e desligava o condicionamento inteiro
(§3.16), e o encadeamento de keyframes (§3.12). Sem elas, a imagem seria
ignorada e o still não serviria para nada.

CONSISTÊNCIA DE PERSONAGEM

O primeiro still de cada personagem vira `reference_image` dos stills seguintes
daquele personagem. O caminho de referência do FLUX foi validado neste projeto
(mesmo prompt e seed produzem pessoa diferente SEM a referência e a pessoa
certa COM ela). É mais forte que repetir descritor em texto, e é o que ataca a
troca de figurino entre planos.

CLI:
    python -m script_pipeline.render_shots --run RUN_DIR [--width 960 --height 544]
                                           [--limit 3] [--stills-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FFMPEG = "C:/ffmpeg/bin/ffmpeg.exe"
FFPROBE = "C:/ffmpeg/bin/ffprobe.exe"


def parse_indices(spec: str | None, total: int) -> list:
    """"0,2-5" -> [0,2,3,4,5]. Vazio/None -> todos.

    Existe porque `--limit N` so pega os N PRIMEIROS, e o plano caro raramente
    e o ultimo. MEDIDO 2026-08-27: numa cena de 6 planos o unico fora do
    envelope da placa era o de indice 1 (593 frames, contra 337 do maior ja
    fechado) -- com --limit nao havia como gerar os outros cinco sem passar por
    ele. Os indices sao os do PLANO, os mesmos que aparecem na tabela do
    shot_plan e no nome do still."""
    if not spec or not spec.strip():
        return list(range(total))
    fora = []
    for parte in spec.replace(";", ",").split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            fora.extend(range(int(a), int(b) + 1))
        else:
            fora.append(int(parte))
    return [i for i in sorted(set(fora)) if 0 <= i < total]


def _still_key(shot: dict, reference: str | None, width: int, height: int,
               checkpoint: str = "") -> str:
    """Chave de cache de um still: tudo o que muda a IMAGEM.

    Existe porque reaproveitar por NOME DE ARQUIVO é o mesmo defeito que já
    mordeu nos clipes (§3.25) -- e aqui seria pior, porque invisível. Trocar de
    estilo, editar um descritor do cast ou mudar o enquadramento reescreve o
    `storyboard_prompt`, mas o arquivo continua se chamando `shot003_ots.png`.
    O still velho seria reusado e a personagem apareceria com o figurino
    antigo, sem nenhum sinal de que algo ficou para trás.

    A referência entra pelo NOME, não pelo conteúdo: ela é sempre um still
    deste mesmo lote, e se ele mudou a chave dele já mudou junto."""
    material = "|".join([
        shot.get("storyboard_prompt", ""),
        shot.get("framing", ""),
        shot.get("angle", ""),
        str(shot.get("screen_side")),
        Path(reference).name if reference else "",
        f"{width}x{height}",
        # O MOTOR entra na chave: FLUX e SD 3.5 desenham o mesmo prompt de jeitos
        # muito diferentes, e sem isto trocar de motor reusaria o still do outro
        # em silencio -- que e a mesma classe de defeito que esta chave existe
        # para impedir.
        checkpoint or "",
    ])
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def _audio_key(wav: str | None) -> str:
    """Identidade do audio de condicionamento, para a chave de reuso do clipe.

    Tamanho em bytes basta: o wav do TTS e reescrito inteiro a cada sintese, e
    qualquer mudanca de texto, voz, emocao ou velocidade muda a duracao."""
    if not wav:
        return "sem-audio"
    try:
        return str(Path(wav).stat().st_size)
    except OSError:
        return "sem-audio"


def _load_manifest(stills_dir: Path) -> dict:
    p = stills_dir / "stills.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(stills_dir: Path, man: dict) -> None:
    json.dump(man, open(stills_dir / "stills.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def _clip_frames(path: Path) -> int | None:
    """Frames de um mp4, ou None se ilegível."""
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def _still_for_shot(shot: dict, idx: int, *, out_dir: Path, width: int, height: int,
                    checkpoint: str, clip: str, vae: str, seed: int,
                    reference: str | None, log=print,
                    steps: int = 8, cfg: float = 1.0, guidance: float = 3.5) -> Path | None:
    import script_pipeline.generate_storyboards as sb

    out = out_dir / f"shot{idx:03d}_{shot['framing']}.png"
    chave = _still_key(shot, reference, width, height, checkpoint)
    man = _load_manifest(out_dir)
    guardado = man.get(str(idx)) or {}
    if out.exists() and guardado.get("key") == chave:
        log(f"  still ja existe e bate com o prompt, reaproveitando: {out.name}")
        return out
    if out.exists():
        log(f"  still existente foi feito com outro prompt/enquadramento; refazendo")
    # `scene` é só o que generate_scene_storyboard usa para nomear a saída --
    # o prompt vem inteiro de prompt_override, que é o ponto.
    ok = sb.generate_scene_storyboard(
        {"index": idx}, {}, server="http://127.0.0.1:8188",
        checkpoint=checkpoint, width=width, height=height, steps=steps, cfg=cfg,
        seed=seed + idx, out_path=out, clip=clip, vae=vae, guidance=guidance,
        prompt_override=shot["storyboard_prompt"], reference_image=reference,
        art_directed=bool(shot.get("art_direction")), log=log)
    if not (ok and out.exists()):
        return None
    # Só registra depois de a imagem existir: manifesto apontando para arquivo
    # que não saiu faria o próximo run pular a geração e falhar mais adiante.
    man[str(idx)] = {"file": out.name, "key": chave,
                     "prompt": shot["storyboard_prompt"][:300]}
    _save_manifest(out_dir, man)
    return out


def render(plan: dict, out_dir: Path, *, width: int, height: int, fps: float,
           checkpoint: str, clip: str, vae: str, seed: int, limit: int | None,
           stills_only: bool, use_reference: bool, videos_only: bool = False,
           steps: int | None = None, cfg: float | None = None,
           guidance: float | None = None, only_shots: str | None = None,
           dialogue: dict | None = None, audio_conditioning: bool = True,
           log=print) -> list:
    """AGRUPE POR MODELO, NÃO POR PLANO.

    MEDIDO 2026-08-26: alternando still e vídeo dentro do mesmo laço, cada plano
    forçava carregar FLUX (encoder Qwen3-8B + DiT + VAE), descarregar, carregar
    LTX (Gemma4-12B + DiT de 40 GB + dois VAEs), descarregar -- ~90 GB de leitura
    de disco POR PLANO, com `--cache-none` no ComfyUI. Os tempos por clipe foram
    321s, 477s, 1275s: degradação por fragmentação, não por conteúdo.

    Por isso a execução é em duas passadas: `--stills-only` gera todos os stills
    com UMA carga do FLUX, e `--videos-only` gera todos os vídeos com UMA carga
    do LTX. Duas cargas no total em vez de vinte."""
    import ltx25_backend
    import script_pipeline.generate_storyboards as sb

    stills_dir = out_dir / "stills"
    clips_dir = out_dir / "clips"
    stills_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # O ESTAGIO DE VIDEO EXIGE --cache-none NESTA PLACA. Nao e preferencia:
    # MEDIDO 2026-08-27, mesmo clipe de 145 frames a 960x544, servidor limpo nos
    # dois casos --
    #     sem  --cache-none: encalha a 24,2 GB de 24,5, 100% de uso, sem sair do
    #                        estagio de carga do encoder
    #     com  --cache-none: passa da carga com 15,9 GB e chega a amostrar
    # O encoder do LTX 2.5 tem 25 GB e o transformer 40 GB; sem a flag o ComfyUI
    # ainda segura resultados intermediarios e nao sobra espaco para a troca.
    #
    # A cadeia de decupagem caia nisso porque quem sobe o servidor e o estagio
    # dos STILLS, que roda antes e nao pede a flag -- e os dois `ensure` voltam
    # cedo quando a porta ja responde. Ver MEMORIAL 3.30.
    if not stills_only:
        os.environ["LTX_COMFY_CACHE_NONE"] = "1"
    if not sb.comfy_is_up("http://127.0.0.1:8188"):
        sb.ensure_comfyui_running("http://127.0.0.1:8188", log=log)

    # Amostragem por MOTOR. `steps=8` era fixo no codigo, e 8 e numero de modelo
    # DESTILADO: apontar este render para o SD 3.5, que usa CFG real, devolveria
    # imagem crua sem nenhum aviso. Quem passou o valor explicitamente manda.
    padroes = sb.engine_defaults(sb.detect_architecture(checkpoint))
    passos = steps if steps is not None else padroes["steps"]
    escala_cfg = cfg if cfg is not None else padroes["cfg"]
    escala_guidance = guidance if guidance is not None else padroes["guidance"]

    todos = plan["shots"][:limit] if limit else plan["shots"]
    # Pares (indice ORIGINAL, plano). O indice tem de sobreviver ao filtro: ele
    # nomeia o still, e a chave do manifesto e a posicao no shot_plan. Renumerar
    # aqui faria o plano 2 escrever por cima do still do plano 0.
    escolhidos = set(parse_indices(only_shots, len(todos)))
    shots = [(i, s) for i, s in enumerate(todos) if i in escolhidos]
    if only_shots and len(shots) != len(todos):
        log(f"[render] {len(shots)} de {len(todos)} plano(s): "
            f"{sorted(i for i, _ in shots)}")
    # Referência por personagem: o primeiro still de alguém guia os próximos.
    refs: dict[str, str] = {}
    feitos = []

    for i, shot in shots:
        t0 = time.time()
        sujeito = shot.get("subject") or ""
        ref = refs.get(sujeito) if (use_reference and sujeito) else None
        log(f"\n[plano {i} · {len(shots)} na fila] cena {shot['scene']} · {shot['style']} · "
            f"{shot['framing']}/{shot['angle']}/{shot['movement']} · "
            f"{shot['seconds']}s ({shot['frames']}f)"
            f"{' · ref=' + Path(ref).name if ref else ''}")

        if videos_only:
            # Manifesto primeiro: com hash no cache, o glob pode achar um still
            # de outra versao do prompt que ficou no disco.
            man = _load_manifest(stills_dir)
            reg = man.get(str(i)) or {}
            cand = stills_dir / reg["file"] if reg.get("file") else None
            if cand and cand.exists():
                still = cand
            else:
                achados = sorted(stills_dir.glob(f"shot{i:03d}_*.png"))
                still = achados[0] if achados else None
            if still is None:
                log(f"  sem still para o plano {i}; rode --stills-only antes")
                continue
        else:
            still = _still_for_shot(shot, i, out_dir=stills_dir, width=width * 2,
                                    height=height * 2, checkpoint=checkpoint, clip=clip,
                                    vae=vae, seed=seed, reference=ref, log=log,
                                    steps=passos, cfg=escala_cfg, guidance=escala_guidance)
        if still is None:
            log(f"  still falhou; pulando o plano {i}")
            continue
        # Só vira referência quem foi enquadrado perto o bastante para mostrar
        # o rosto -- um wide como referência ensinaria o cenário, não a pessoa.
        if use_reference and sujeito and sujeito not in refs and \
                shot["framing"] in ("close", "extreme_close", "medium", "ots"):
            refs[sujeito] = str(still)

        if stills_only:
            feitos.append({"shot": i, "still": str(still), "clip": None})
            continue

        # A fala DESTE plano, quando ha. E ela que o LTX recebe como trilha de
        # referencia -- ver o docstring de audio_conditioning em ltx25_backend.
        wav_cond = None
        if audio_conditioning and dialogue:
            entrada = dialogue.get((shot.get("scene"), shot.get("line_index")))
            if entrada and entrada[0] and Path(entrada[0]).exists():
                wav_cond = entrada[0]

        clip_path = clips_dir / f"shot{i:03d}.mp4"
        marca = clips_dir / f"shot{i:03d}.key"
        chave = f"{shot['frames']}|{_audio_key(wav_cond)}"
        if clip_path.exists():
            # Reaproveitar exige que o clipe CORRESPONDA ao plano atual, não só
            # que exista. MEDIDO 2026-08-26: depois que o TTS trocou as durações
            # estimadas pelas reais, cinco clipes ficaram com a contagem de
            # frames antiga e foram reusados assim mesmo -- o mux tentou ajustar
            # a voz a uma duração que o vídeo não tinha, o ffmpeg errou, e a
            # montagem saiu com 47,6s de imagem contra 40s de áudio. Clipe
            # obsoleto é pior que clipe ausente: passa em silêncio.
            atual = _clip_frames(clip_path)
            # A chave inclui o AUDIO, nao so os frames. Com condicionamento por
            # fala, dois clipes com a mesma contagem podem ter sido gerados de
            # audios diferentes -- e o comentario acima ja registra que clipe
            # obsoleto passa em silencio, que e o pior modo de falhar.
            antiga = marca.read_text(encoding="utf-8").strip() if marca.exists() else None
            if atual == shot["frames"] and antiga == chave:
                log(f"  clipe ja existe e bate com o plano, reaproveitando: {clip_path.name}")
                feitos.append({"shot": i, "still": str(still), "clip": str(clip_path)})
                continue
            if atual != shot["frames"]:
                log(f"  clipe existente tem {atual}f mas o plano pede {shot['frames']}f; refazendo")
            else:
                log(f"  clipe existente foi gerado com outro audio; refazendo")
        try:
            ltx25_backend.generate(
                shot["video_prompt"], str(clip_path),
                width=width, height=height, num_frames=shot["frames"],
                frame_rate=fps, seed=seed + i,
                image_path=str(still), image_strength=1.0,
                # Sem isto o LTX 2.5 inventa a trilha sozinho e gera VOZ
                # propria, que depois briga com o TTS por baixo da mixagem.
                # Com isto ele constroi o som em volta da fala real, que e o
                # que o caminho por FALA ja fazia (render_scenes.py:550).
                audio_conditioning=wav_cond,
                # O log do backend ia para o LIXO. Foi assim que o estagio de
                # video ficou sem diagnostico: qual variante carregou, se o
                # encoder foi para a CPU, quanto tempo em cada fase -- nada
                # disso saia. Mesma licao da secao 3.28: silenciar o caminho de
                # sucesso e razoavel, o de falha nao, e aqui os dois estavam
                # silenciados. Prefixado para nao se confundir com o log do
                # proprio render_shots.
                log_cb=lambda m: log(f"    [ltx25] {m}"), timeout=2400)
            log(f"  clipe OK em {time.time()-t0:.0f}s -> {clip_path.name}"
                f"{' (som condicionado pela fala)' if wav_cond else ''}")
            marca.write_text(chave, encoding="utf-8")
            feitos.append({"shot": i, "still": str(still), "clip": str(clip_path)})
        except Exception as e:
            log(f"  clipe FALHOU: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            feitos.append({"shot": i, "still": str(still), "clip": None})
    return feitos


# --------------------------------------------------------------------------
# rascunho
# --------------------------------------------------------------------------
# Resolução e teto de frames do modo rascunho. O objetivo não é ver o filme --
# é conferir se enquadramento, ordem, eixo e figurino fecham ANTES de gastar
# uma hora. Um plano de 1 segundo já mostra tudo isso.
DRAFT_W, DRAFT_H, DRAFT_MAX_FRAMES = 512, 288, 25


def load_dialogue(path) -> dict:
    """(cena, fala) -> (caminho do wav, duracao)."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for e in json.load(open(path, encoding="utf-8")):
        if e.get("ok") and e.get("audio_path"):
            out[(e["scene_index"], e["line_index"])] = (e["audio_path"],
                                                        float(e.get("duration_sec") or 0))
    return out


def animatic(plan: dict, stills_dir: Path, out_path: Path, *, dialogue: dict | None = None,
             log=print) -> Path | None:
    """Animatic: cada still exibido pela duração do seu plano, com a voz por cima.

    É o rascunho que NÃO usa GPU. Valida tudo o que a decupagem decide --
    enquadramento, ordem, ritmo, eixo, figurino -- e mais uma coisa que só a voz
    revela: se a fala CABE no plano. Um plano curto demais corta a última
    sílaba, e isso não aparece em nenhuma inspeção de still.

    É o mesmo animatic que uma produção monta antes de filmar, pela mesma razão:
    descobrir o erro de montagem quando ele ainda é barato.
    """
    dialogue = dialogue or {}
    entradas, filtros, audios = [], [], []
    t = 0.0
    for i, shot in enumerate(plan["shots"]):
        achados = sorted(stills_dir.glob(f"shot{i:03d}_*.png"))
        if not achados:
            log(f"[animatic] sem still para o plano {i}; pulando")
            continue
        entradas += ["-loop", "1", "-t", f"{shot['seconds']:.3f}", "-i", str(achados[0])]
        chave = (shot.get("scene"), shot.get("line_index"))
        if chave in dialogue:
            wav, _ = dialogue[chave]
            audios.append((len(entradas) // 2, wav, t))   # índice provisório
        t += shot["seconds"]
    if not entradas:
        log("[animatic] nenhum still encontrado.")
        return None

    n_video = len(entradas) // 6
    for wav, _, _ in [(a[1], 0, 0) for a in audios]:
        entradas += ["-i", wav]

    # Vídeo: escala uniforme e concatena.
    v = "".join(f"[{k}:v]scale={plan.get('draft_w', 960)}:-2,setsar=1[v{k}];"
                for k in range(n_video))
    v += "".join(f"[v{k}]" for k in range(n_video)) + f"concat=n={n_video}:v=1:a=0[vout]"
    filtros.append(v)

    cmd = [FFMPEG, "-y", "-v", "error"] + entradas
    if audios:
        # Cada fala entra atrasada até o início do seu plano.
        a = "".join(f"[{n_video + j}:a]adelay={int(off * 1000)}|{int(off * 1000)}[a{j}];"
                    for j, (_, _, off) in enumerate(audios))
        a += "".join(f"[a{j}]" for j in range(len(audios)))
        a += f"amix=inputs={len(audios)}:dropout_transition=0:normalize=0[aout]"
        filtros.append(a)
        cmd += ["-filter_complex", ";".join(filtros), "-map", "[vout]", "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-filter_complex", ";".join(filtros), "-map", "[vout]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(plan.get("fps", 24)),
            "-crf", "20", str(out_path)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[animatic] falhou: {r.stderr[-400:]}")
        return None
    log(f"[animatic] pronto -> {out_path}  ({t:.1f}s, {n_video} planos"
        f"{', ' + str(len(audios)) + ' falas' if audios else ''})")
    return out_path


# --------------------------------------------------------------------------
# voz: lip-sync e mux
# --------------------------------------------------------------------------
def mux_audio(clip: Path, wav: str | None, out: Path, clip_seconds: float,
              fps: float = 24.0, log=print) -> Path:
    """Coloca a fala dentro do clipe, CENTRADA na folga.

    `shot_plan` dimensiona o plano como duracao_da_fala + FALA_FOLGA_S. Aqui a
    folga e dividida em duas: metade antes, metade depois. Encostar a fala no
    inicio faria o corte cair em cima da ultima silaba -- que e exatamente o
    defeito que a folga existe para evitar.

    Sem fala (plano de acao), entra silencio do tamanho do clipe: o concat
    precisa que TODOS os trechos tenham faixa de audio, ou os que tem somem."""
    # O clipe final é CONFORMADO à duração do plano, sempre.
    #
    # MEDIDO 2026-08-26: o lip-sync reencoda com a duração do ÁUDIO, não do
    # vídeo, e devolve contagem de frames diferente da que entrou -- 249->252,
    # 121->116 e, no pior caso, 41->30. Com `-c:v copy` não há como recuperar:
    # copiar preserva o que veio, e o que veio já está errado. Reencodando com
    # `tpad` (congela o último frame) mais `-t`, todo clipe sai exatamente com
    # `clip_seconds`, que é o que a montagem pressupõe.
    #
    # O áudio também é normalizado para 44100/estéreo: o `concat` junta streams
    # de origens diferentes, e parâmetro divergente entre eles fazia o encoder
    # recusar quadros ("Error submitting audio frame to the encoder").
    # `-r` FORCADO: o lip-sync devolve 25 fps, nao 24 (MEDIDO 2026-08-26), e sem
    # fixar a taxa o `-t` em segundos vira contagem de frames errada -- 10,375s a
    # 25 fps sao 260 frames onde o plano pede 249.
    #
    # `tpad=stop_duration` ADICIONA essa duracao, nao completa ate ela. Por isso
    # o valor aqui e uma folga generosa: quem define o tamanho final e o `-t`.
    pad = f"tpad=stop_mode=clone:stop_duration={clip_seconds:.3f}"
    if wav and Path(wav).exists():
        dur = _audio_seconds(wav)
        atraso = max(0.0, (clip_seconds - dur) / 2.0)
        cmd = [FFMPEG, "-y", "-v", "error", "-i", str(clip), "-i", wav,
               "-filter_complex",
               f"[0:v]{pad}[v];"
               f"[1:a]adelay={int(atraso*1000)}|{int(atraso*1000)},"
               f"aresample=44100,aformat=channel_layouts=stereo,"
               f"apad=whole_dur={clip_seconds:.3f}[a]",
               "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
               "-r", f"{fps}", "-vsync", "cfr",
               "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
               "-t", f"{clip_seconds:.4f}", str(out)]
    else:
        cmd = [FFMPEG, "-y", "-v", "error", "-i", str(clip),
               "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
               "-filter_complex", f"[0:v]{pad}[v]",
               "-map", "[v]", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
               "-r", f"{fps}", "-vsync", "cfr",
               "-c:a", "aac", "-ar", "44100", "-ac", "2",
               "-t", f"{clip_seconds:.4f}", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  mux falhou ({r.stderr[-200:]}); mantendo clipe mudo")
        return clip
    return out


def _audio_seconds(path: str) -> float:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def apply_voice(feitos: list, plan: dict, dialogue: dict, out_dir: Path, *,
                engine: str = "auto", do_lipsync: bool = True, log=print) -> list:
    """Para cada plano com fala: lip-sync no clipe e a voz muxada dentro dele.

    Plano de acao passa direto, com faixa muda -- nada a sincronizar. Lip-sync
    que falha nao derruba o plano: fica o clipe original com a voz por cima, que
    ainda e melhor que clipe mudo."""
    from tensorxx_ge.lipsync import apply_lipsync, lipsync_status

    ls_dir = out_dir / "lipsync"
    fin_dir = out_dir / "final"
    ls_dir.mkdir(parents=True, exist_ok=True)
    fin_dir.mkdir(parents=True, exist_ok=True)
    disponivel = lipsync_status().available if do_lipsync else False
    if do_lipsync and not disponivel:
        log("[voz] nenhum motor de lip-sync disponivel; so o audio sera muxado.")

    saida = []
    for f in feitos:
        clip = f.get("clip")
        if not clip:
            saida.append(f)
            continue
        i = f["shot"]
        shot = plan["shots"][i]
        wav = None
        chave = (shot.get("scene"), shot.get("line_index"))
        if chave in dialogue:
            wav = dialogue[chave][0]

        base = Path(clip)
        if wav and disponivel:
            alvo = ls_dir / f"shot{i:03d}_synced.mp4"
            # TERCEIRA etapa cara a ganhar reaproveitamento (depois de stills e
            # clipes). A chave e o par (clipe de entrada, wav): se nenhum dos
            # dois mudou, o resultado do lip-sync tambem nao muda. Sem isto,
            # reprocessar so o mux forcava LatentSync de novo nos cinco planos.
            marca = ls_dir / f"shot{i:03d}.key"
            chave_ls = f"{_clip_frames(base)}|{Path(wav).stat().st_size}"
            if alvo.exists() and marca.exists() and                     marca.read_text(encoding="utf-8").strip() == chave_ls:
                log(f"  plano {i}: lip-sync ja feito, reaproveitando")
                base = alvo
                final = mux_audio(base, wav, fin_dir / f"shot{i:03d}.mp4",
                                  shot["frames"] / float(plan.get("fps", 24.0)),
                                  fps=float(plan.get("fps", 24.0)), log=log)
                saida.append({**f, "final": str(final), "lipsync": True})
                continue
            # O log do lip-sync e capturado, nao descartado: silenciar o caminho
            # de sucesso e razoavel, o de FALHA nao -- sem ele, "falhou" nao diz
            # nada e o diagnostico comeca do zero.
            linhas: list[str] = []
            r = apply_lipsync(str(base), wav, str(alvo), work_dir=str(ls_dir),
                              engine=engine, log=linhas.append)
            if r is not None:
                base = Path(r)
                marca.write_text(chave_ls, encoding="utf-8")
                log(f"  plano {i}: lip-sync ok")
            else:
                motivo = " | ".join(linhas[-3:]) if linhas else "sem detalhe"
                log(f"  plano {i}: lip-sync falhou ({motivo[:220]}); "
                    f"seguindo com o clipe original")

        # frames/fps, NAO shot["seconds"]. MEDIDO 2026-08-26: `seconds` e a
        # duracao PEDIDA e `frames` a que o modelo permite (1 + multiplo de 8),
        # e as duas divergem -- 3,47s pedidos viram 81 frames, que sao 3,375s.
        # Conformando por `seconds` o clipe saia com 84 frames e a montagem
        # acumulava erro a cada plano.
        alvo_s = shot["frames"] / float(plan.get("fps", 24.0))
        final = mux_audio(base, wav, fin_dir / f"shot{i:03d}.mp4", alvo_s,
                          fps=float(plan.get("fps", 24.0)), log=log)
        saida.append({**f, "final": str(final), "lipsync": bool(wav and disponivel)})
    return saida


def concat(feitos: list, out_path: Path, log=print) -> Path | None:
    # Prefere o clipe FINAL (com voz e lip-sync); cai no bruto se nao houver.
    clipes = [f.get("final") or f.get("clip") for f in feitos
              if f.get("final") or f.get("clip")]
    if not clipes:
        log("[render_shots] nenhum clipe para concatenar.")
        return None
    lista = out_path.parent / "_concat.txt"
    lista.write_text("".join(f"file '{Path(c).as_posix()}'\n" for c in clipes), encoding="utf-8")
    # Reencode em vez de -c copy: os clipes vêm de gerações distintas e podem
    # divergir em parâmetros de stream; copy falha ou produz vídeo com glitch.
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lista), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "16", "-c:a", "aac", "-b:a", "192k",
                        str(out_path)], capture_output=True, text=True)
    lista.unlink(missing_ok=True)
    if r.returncode != 0:
        log(f"[render_shots] concat falhou: {r.stderr[-300:]}")
        return None
    log(f"[render_shots] montagem final -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Executa um shot_plan: still + video por plano")
    ap.add_argument("--run", help="pasta de run (usa parse/shot_plan.json)")
    ap.add_argument("--plan", help="caminho direto do shot_plan.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image-engine", default="flux", choices=["flux", "sd35", "sdxl"],
                    help='motor de imagem dos stills. flux = melhor adesao a enquadramento e lado de tela e aceita imagem de referencia, mas carrega ~4 min e encosta no teto de VRAM da 3090. sd35 = carrega em ~1 min, cabe em ~12 GB e amostra mais rapido, mas obedece menos o enquadramento e NAO tem referencia. --checkpoint/--clip/--vae explicitos continuam ganhando disto.')
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="passos de amostragem; sem isto usa o padrao do motor")
    ap.add_argument("--cfg", type=float, default=None,
                    help="CFG; sem isto usa o padrao do motor (o FLUX Klein ignora e usa guidance)")
    ap.add_argument("--limit", type=int, default=None, help="renderiza so os N primeiros planos")
    ap.add_argument("--only-shots", default=None,
                    help="renderiza SO estes planos, por indice do shot_plan: \"0,2-5\". Diferente de --limit, que so pega os primeiros.")
    ap.add_argument("--stills-only", action="store_true",
                    help="so os stills (UMA carga do FLUX) -- rode isto primeiro")
    ap.add_argument("--videos-only", action="store_true",
                    help="so os videos, usando os stills ja em disco (UMA carga do LTX)")
    ap.add_argument("--animatic", action="store_true",
                    help="RASCUNHO sem GPU: stills na duracao planejada + voz. "
                         "Valida enquadramento, ordem, ritmo e se a fala cabe no plano.")
    ap.add_argument("--draft", action="store_true",
                    help=f"video rascunho: {DRAFT_W}x{DRAFT_H}, no maximo "
                         f"{DRAFT_MAX_FRAMES} frames por plano")
    ap.add_argument("--dialogue", default=None, help="dialogue/lines.json (voz)")
    ap.add_argument("--no-lipsync", action="store_true",
                    help="muxa a voz mas nao roda lip-sync (mais rapido)")
    ap.add_argument("--lipsync-engine", default="auto",
                    choices=["auto", "latentsync", "wav2lip"])
    ap.add_argument("--no-reference", action="store_true",
                    help="nao usa o primeiro still de um personagem como referencia dos demais")
    ap.add_argument("--no-audio-conditioning", action="store_true",
                    help="nao passa a fala do TTS ao LTX como trilha de referencia. "
                         "Sem o condicionamento o modelo inventa o som sozinho e "
                         "gera VOZ propria, que depois disputa com o TTS na mixagem.")
    args = ap.parse_args()

    plan_path = Path(args.plan) if args.plan else Path(args.run) / "parse" / "shot_plan.json"
    if not plan_path.exists():
        print(f"[render_shots] nao encontrei {plan_path}", file=sys.stderr)
        return 1
    plan = json.load(open(plan_path, encoding="utf-8"))
    out_dir = Path(args.out) if args.out else plan_path.parent.parent / "shots"
    out_dir.mkdir(parents=True, exist_ok=True)

    dlg_path = args.dialogue
    if not dlg_path and args.run:
        c = Path(args.run) / "dialogue" / "lines.json"
        dlg_path = str(c) if c.exists() else None
    dialogo = load_dialogue(dlg_path)

    if args.animatic:
        plan["draft_w"] = args.width
        alvo = animatic(plan, out_dir / "stills", out_dir / "animatic.mp4",
                        dialogue=dialogo)
        return 0 if alvo else 1

    if args.draft:
        args.width, args.height = DRAFT_W, DRAFT_H
        for sh in plan["shots"]:
            sh["frames"] = min(sh["frames"], DRAFT_MAX_FRAMES)
        print(f"[render_shots] RASCUNHO: {DRAFT_W}x{DRAFT_H}, "
              f"max {DRAFT_MAX_FRAMES} frames/plano")

    print(f"[render_shots] {len(plan['shots'])} plano(s), estilo base {plan['style']}, "
          f"{args.width}x{args.height}")
    # O motor decide checkpoint, encoders e amostragem de uma vez. Passar
    # --checkpoint sozinho continua valendo: quem foi explicito manda.
    from script_pipeline.generate_storyboards import engine_defaults
    _motor = engine_defaults(args.image_engine)
    if not args.checkpoint:
        args.checkpoint = _motor["checkpoint"]
    if args.clip is None:
        args.clip = _motor["clip"]
    if args.vae is None:
        args.vae = _motor["vae"]

    feitos = render(plan, out_dir, width=args.width, height=args.height, fps=args.fps,
                    checkpoint=args.checkpoint, clip=args.clip, vae=args.vae,
                    seed=args.seed, limit=args.limit, stills_only=args.stills_only,
                    videos_only=args.videos_only, steps=args.steps, cfg=args.cfg,
                    only_shots=args.only_shots, dialogue=dialogo,
                    audio_conditioning=not args.no_audio_conditioning,
                    use_reference=not args.no_reference)
    if not args.stills_only and dialogo:
        print(f"[render_shots] voz: {len(dialogo)} fala(s) para muxar"
              f"{', com lip-sync' if not args.no_lipsync else ''}")
        feitos = apply_voice(feitos, plan, dialogo, out_dir,
                             engine=args.lipsync_engine, do_lipsync=not args.no_lipsync)
    json.dump(feitos, open(out_dir / "render_shots.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if not args.stills_only:
        concat(feitos, out_dir / "montagem.mp4")
    print(f"\n[render_shots] {sum(1 for f in feitos if f.get('clip'))} clipe(s), "
          f"{sum(1 for f in feitos if f.get('still'))} still(s) em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
