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
               checkpoint: str = "", lora_name: str = "", lora_strength: float = 0.8,
               seed: int = 0, steps: int = 0, cfg: float = 0.0, guidance: float = 0.0,
               weight_dtype: str = "", consistency_threshold: float | None = None) -> str:
    """Chave de cache de um still: tudo o que muda a IMAGEM.

    Existe porque reaproveitar por NOME DE ARQUIVO é o mesmo defeito que já
    mordeu nos clipes (§3.25) -- e aqui seria pior, porque invisível. Trocar de
    estilo, editar um descritor do cast ou mudar o enquadramento reescreve o
    `storyboard_prompt`, mas o arquivo continua se chamando `shot003_ots.png`.
    O still velho seria reusado e a personagem apareceria com o figurino
    antigo, sem nenhum sinal de que algo ficou para trás.

    BUGFIX auditoria 2026-09-16 (A03): a referência entrava só pelo NOME do
    arquivo, não pelo conteúdo -- trocar a character sheet ou importar outra
    foto MANTENDO o mesmo caminho (ex.: `characters/fulano/ref.png`, que é
    como `cast_characters.py`/`import_reference.py` salvam) devolvia
    exatamente o still anterior, sem nenhum sinal de que a referência mudou.
    Agora usa `_audio_key` (hash de conteúdo) na referência. Seed, steps, cfg,
    guidance, weight_dtype e o limiar de consistência também faltavam: ajustar
    qualidade/nitidez ou ligar a auditoria de consistência reaproveitava o
    still gerado com os parâmetros antigos."""
    material = "|".join([
        shot.get("storyboard_prompt", ""),
        shot.get("framing", ""),
        shot.get("angle", ""),
        str(shot.get("screen_side")),
        _audio_key(reference),
        f"{width}x{height}",
        # O MOTOR entra na chave: FLUX e SD 3.5 desenham o mesmo prompt de jeitos
        # muito diferentes, e sem isto trocar de motor reusaria o still do outro
        # em silencio -- que e a mesma classe de defeito que esta chave existe
        # para impedir.
        checkpoint or "",
        # LoRA muda o traço/identidade do mesmo jeito que trocar de motor --
        # sem isto, ligar/desligar um LoRA (ou trocar a força) reaproveitaria
        # o still antigo em silêncio.
        f"{lora_name}@{lora_strength}" if lora_name else "",
        f"seed={seed}", f"steps={steps}", f"cfg={cfg}", f"guidance={guidance}",
        weight_dtype or "",
        f"consist={consistency_threshold}" if consistency_threshold is not None else "",
    ])
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def _audio_key(wav: str | None) -> str:
    """Identidade de CONTEUDO de um arquivo (audio ou still) usado na chave de
    reuso do clipe/still.

    BUGFIX auditoria 2026-09-16 (A02): media so pelo TAMANHO (st_size) colidia
    arquivos diferentes -- WAVs PCM de mesma duracao (mesma frase, voz trocada)
    ou dois stills PNG de tamanho parecido frequentemente tem o MESMO numero de
    bytes. MEDIDO: substituir o conteudo de um wav mantendo o tamanho preservava
    a chave antiga, e o clipe velho era reaproveitado. Hash SHA-1 do conteudo
    inteiro -- os arquivos aqui sao curtos (uma fala, um still), entao o custo e
    desprezivel perto de uma geracao de video."""
    if not wav:
        return "sem-audio"
    try:
        h = hashlib.sha1()
        with open(wav, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 20), b""):
                h.update(bloco)
        return h.hexdigest()[:16]
    except OSError:
        return "sem-audio"


def _extrair_ultimo_frame(video_path: str, out_path: Path) -> bool:
    """Ultimo frame decodificado de um clipe, pra alimentar o proximo elo do
    encadeamento MiniMax (ver `_minimax_chain_generate`). Porte de
    `_test_minimax_duration_cap.py::extrair_ultimo_frame`."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame))


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


def _apply_freeze(clip_path: Path, *, extra_seconds: float = 0.6, log=print) -> None:
    """Segura o ULTIMO FRAME do clipe por `extra_seconds` extra, no fim.

    Efeito "post" do enriquecimento de camera (ver shot_plan.enrich_camera_
    style / POST_EFFECTS): nem LTX nem MiniMax tem como "congelar no meio" da
    propria geracao -- e propriedade do CLIPE JA PRONTO, nao do prompt. Usa o
    filtro `tpad` do ffmpeg (`stop_mode=clone` repete o ultimo frame) e
    reescreve o arquivo no lugar via um temporario, mesmo padrao do resto
    deste modulo. Falha aqui NAO derruba o plano -- o clipe sem o freeze ainda
    e um clipe utilizavel, so sem o acento; ver a mesma filosofia em
    apply_camera_style (silencioso, acabamento nao e estrutura)."""
    tmp = clip_path.with_suffix(".freeze_tmp.mp4")
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(clip_path),
           "-vf", f"tpad=stop_mode=clone:stop_duration={extra_seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        log(f"  freeze: ffmpeg falhou ({r.stderr.strip()[-200:]}); clipe segue sem o efeito.")
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(clip_path)
    log(f"  freeze aplicado: +{extra_seconds}s segurando o ultimo frame.")


def _duplicate_face_check(still_path: Path, *, log=print, idx: int) -> dict:
    """Roda `consistency_audit.detect_duplicate_faces` no still ja pronto --
    barato (so mais uma passada de deteccao facial, sem gerar nada de novo)
    e roda SEMPRE, mesmo sem `consistency_threshold` -- pedido do usuario
    2026-09-10 depois de achar um still com dois personagens onde o segundo
    saiu com a cara clonada do primeiro (`check_consistency` nao pega isso,
    so olha o MAIOR rosto contra uma referencia). Nunca bloqueia a geracao,
    so grava o resultado no manifesto para a UI/relatorio avisarem."""
    try:
        from script_pipeline.consistency_audit import detect_duplicate_faces

        resultado = detect_duplicate_faces(str(still_path))
    except Exception as e:
        log(f"  auditoria de duplicidade facial falhou: {type(e).__name__}: {e}")
        return {"duplicate_flag": None, "duplicate_faces_detected": None}
    if resultado["flagged"]:
        pares = ", ".join(f"{i}-{j} ({score:.3f})" for i, j, score in resultado["duplicate_pairs"])
        log(f"  ALERTA plano {idx}: {resultado['faces_detected']} rosto(s) detectado(s), "
            f"PAR(ES) SUSPEITO(S) DE DUPLICIDADE: {pares} -- confira se dois personagens "
            f"saíram com a mesma cara antes de gerar vídeo.")
    return {"duplicate_flag": resultado["flagged"], "duplicate_faces_detected": resultado["faces_detected"]}


def _still_for_shot(shot: dict, idx: int, *, out_dir: Path, width: int, height: int,
                    checkpoint: str, clip: str, vae: str, seed: int,
                    reference: str | None, reference_2: str | None = None, log=print,
                    steps: int = 8, cfg: float = 1.0, guidance: float = 3.5,
                    weight_dtype: str = "default",
                    consistency_threshold: float | None = None,
                    consistency_max_retries: int = 2,
                    lora_name: str = "", lora_strength: float = 0.8) -> Path | None:
    import script_pipeline.generate_storyboards as sb

    out = out_dir / f"shot{idx:03d}_{shot['framing']}.png"
    # A chave de cache tem de incluir a 2a referencia -- sem isso, um plano
    # gerado ANTES do fix (so uma referencia) seria reaproveitado como se
    # nada tivesse mudado, mesmo com o co_subject novo disponivel agora.
    chave = _still_key(shot, reference, width, height, checkpoint, lora_name, lora_strength,
                       seed=seed, steps=steps, cfg=cfg, guidance=guidance,
                       weight_dtype=weight_dtype, consistency_threshold=consistency_threshold)
    if reference_2:
        # Conteúdo, não nome -- mesmo raciocínio do BUGFIX A03 acima.
        chave = f"{chave}|ref2={_audio_key(reference_2)}"
    man = _load_manifest(out_dir)
    guardado = man.get(str(idx)) or {}
    if out.exists() and guardado.get("key") == chave:
        log(f"  still ja existe e bate com o prompt, reaproveitando: {out.name}")
        return out
    if out.exists():
        log(f"  still existente foi feito com outro prompt/enquadramento; refazendo")

    def _gerar_uma_vez(dest: Path, seed_usado: int) -> bool:
        # `scene` é só o que generate_scene_storyboard usa para nomear a saída --
        # o prompt vem inteiro de prompt_override, que é o ponto.
        return sb.generate_scene_storyboard(
            {"index": idx}, {}, server="http://127.0.0.1:8188",
            checkpoint=checkpoint, width=width, height=height, steps=steps, cfg=cfg,
            seed=seed_usado, out_path=dest, clip=clip, vae=vae, guidance=guidance,
            weight_dtype=weight_dtype,
            prompt_override=shot["storyboard_prompt"], reference_image=reference,
            reference_image_2=reference_2,
            art_directed=bool(shot.get("art_direction")), log=log,
            lora_name=lora_name, lora_strength=lora_strength)

    # AUDITORIA DE CONSISTENCIA (2026-09-03, pedido do usuario -- ver MEMORIAL
    # 3.53): so faz sentido com referencia (nada pra comparar sem ela) e com
    # limiar explicito (None = comportamento de sempre, sem custo extra pra
    # quem nao pediu). Gera ate `consistency_max_retries` tentativas, fica com
    # a de MAIOR similaridade ao rosto de referencia -- nao so a primeira que
    # passar, porque a diferenca entre "0,36 e 0,52" ainda importa.
    if reference and consistency_threshold is not None:
        from script_pipeline.consistency_audit import check_consistency

        melhor_path, melhor_score, melhor_seed = None, None, None
        for tentativa in range(consistency_max_retries + 1):
            seed_tentativa = seed + idx + tentativa * 7919
            candidato = out if tentativa == 0 else out.with_suffix(f".tentativa{tentativa}.png")
            if not _gerar_uma_vez(candidato, seed_tentativa):
                continue
            ok_cons, score = check_consistency(str(candidato), reference, threshold=consistency_threshold)
            log(f"  consistencia (tentativa {tentativa}, seed {seed_tentativa}): "
                f"{'sem rosto detectavel' if score is None else f'{score:.3f}'} "
                f"{'(dentro do limiar)' if ok_cons else '(ABAIXO do limiar)'}")
            if melhor_score is None or (score is not None and score > (melhor_score or -1)):
                melhor_path, melhor_score, melhor_seed = candidato, score, seed_tentativa
            if ok_cons:
                break
        if melhor_path is None:
            return None
        if melhor_path != out:
            melhor_path.replace(out)
        # Limpa as tentativas descartadas (a vencedora ja foi movida pra `out`).
        for tentativa in range(1, consistency_max_retries + 1):
            sobra = out.with_suffix(f".tentativa{tentativa}.png")
            if sobra.exists():
                sobra.unlink(missing_ok=True)
        if not out.exists():
            return None
        dup = _duplicate_face_check(out, log=log, idx=idx)
        man[str(idx)] = {"file": out.name, "key": chave,
                         "prompt": shot["storyboard_prompt"][:300],
                         "reference": reference, "reference_2": reference_2,
                         "consistency_score": melhor_score,
                         "seed": melhor_seed, **dup}
        _save_manifest(out_dir, man)
        return out

    ok = _gerar_uma_vez(out, seed + idx)
    if not (ok and out.exists()):
        return None
    # Só registra depois de a imagem existir: manifesto apontando para arquivo
    # que não saiu faria o próximo run pular a geração e falhar mais adiante.
    dup = _duplicate_face_check(out, log=log, idx=idx)
    man[str(idx)] = {"file": out.name, "key": chave,
                     "prompt": shot["storyboard_prompt"][:300],
                     "reference": reference, "reference_2": reference_2, **dup}
    _save_manifest(out_dir, man)
    return out


def render(plan: dict, out_dir: Path, *, width: int, height: int, fps: float,
           checkpoint: str, clip: str, vae: str, seed: int, limit: int | None,
           stills_only: bool, use_reference: bool, videos_only: bool = False,
           steps: int | None = None, cfg: float | None = None,
           guidance: float | None = None, only_shots: str | None = None,
           dialogue: dict | None = None, audio_conditioning: bool = True,
           weight_dtype: str | None = None,
           consistency_threshold: float | None = None, consistency_max_retries: int = 2,
           character_sheets: dict[str, str] | None = None,
           lora_name: str = "", lora_strength: float = 0.8,
           engine: str = "ltx",
           minimax_aspect_ratio: str | None = None, minimax_megapixels: float | None = None,
           minimax_turbo: bool = True, minimax_ref_audio: bool = False,
           minimax_no_still: bool = False, minimax_chain_max_seconds: float | None = None,
           ltx_no_still: bool = False, ltx_chain_max_seconds: float | None = None,
           video_loras: list[tuple[str, float]] | None = None,
           ic_mode: str = "off", ic_lora: str | None = None,
           ic_strength: float = 1.0, ic_guide_strength: float = 1.0,
           cast_descriptors: dict[str, str] | None = None,
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
    if engine == "minimax":
        import minimax_h3_backend
    # LongCat (2026-09-13): motor SO de planos de fala; os planos sem fala do mesmo run
    # seguem no LTX. `engine="longcat"` = fala no LongCat, acao no LTX.
    if engine == "longcat":
        import longcat_video_backend

    # BUGFIX auditoria 2026-09-16 (A17): o modo combinado (nem stills_only nem
    # videos_only) gera os stills no MESMO laço em que troca de servidor para o
    # MiniMax H3 -- e a troca acontece ANTES do laço, entao qualquer still ainda
    # nao cacheado tenta submeter ao ComfyUI do FLUX (8188) depois dele ja ter
    # sido derrubado. O orquestrador de producao (`run_decupagem.py`) sempre usa
    # duas passadas (--stills-only depois --videos-only) e nunca bate nisso; so a
    # chamada direta/avulsa (CLI ou função) expõe a combinação perigosa.
    if engine == "minimax" and not stills_only and not videos_only:
        raise ValueError(
            "engine=minimax exige duas passadas: chame render() com stills_only=True "
            "primeiro (todos os stills, com o ComfyUI do FLUX no ar) e depois "
            "videos_only=True (troca para o ComfyUI do MiniMax H3). O modo combinado "
            "derrubaria o 8188 antes de gerar os stills que ainda faltam.")

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
    # BUGFIX 2026-09-02: nada garantia o ComfyUI (FLUX/SD3.5, 8188) no ar antes
    # da passada de STILLS -- so a passada de video tinha um `ensure`. Na
    # pratica quase sempre "funcionava" porque outro estagio anterior (ex.:
    # cast_characters com imagem de referencia) ja tinha subido o servidor por
    # acidente; numa corrida sem isso, `_still_for_shot` -> generate_scene_
    # storyboard -> submit_and_wait quebra com ConnectionRefusedError. Preciso
    # de FLUX no ar sempre que vamos gerar QUALQUER still (stills_only OU o
    # modo combinado antigo, nao so videos_only).
    # Z-Image-Turbo (pedido do usuario 2026-09-17) nao usa ComfyUI -- roda no
    # servidor HTTP proprio de zimage_backend.py, que generate_scene_storyboard
    # sobe sozinho quando precisa. So sobe o ComfyUI do FLUX aqui para as
    # outras arquiteturas de still.
    if ((stills_only or not videos_only) and sb.detect_architecture(checkpoint) != "zimage"
            and not sb.comfy_is_up("http://127.0.0.1:8188")):
        sb.ensure_comfyui_running("http://127.0.0.1:8188", log=log)

    # Motor de video decide qual instancia de ComfyUI sobe -- as duas (LTX 2.5
    # na 8188, MiniMax H3 na 8189) sao processos INDEPENDENTES mas disputam a
    # MESMA 3090 fisica; minimax_h3_backend.py e explicito que nunca devem
    # rodar as duas ao mesmo tempo. O estagio de stills sempre usa FLUX na
    # 8188 (generate_storyboards.py), entao aqui, na troca pro estagio de
    # video, e o unico lugar que sabe qual das duas o plano pediu.
    if not stills_only:
        if engine == "minimax":
            if sb.comfy_is_up("http://127.0.0.1:8188"):
                log("[render] motor=minimax: derrubando o ComfyUI do LTX 2.5 (8188) antes de subir o do MiniMax H3 (8189).")
                sb.stop_comfyui(8188, log=log)
            # minimax_h3_backend.generate() sobe o proprio servidor sozinho
            # (ensure_server, dentro de submit_and_wait) -- nao precisa de
            # nada aqui alem de garantir que o 8188 nao ficou no ar.
        else:
            os.environ["LTX_COMFY_CACHE_NONE"] = "1"
            if not sb.comfy_is_up("http://127.0.0.1:8188"):
                sb.ensure_comfyui_running("http://127.0.0.1:8188", log=log)

    # Amostragem por MOTOR. `steps=8` era fixo no codigo, e 8 e numero de modelo
    # DESTILADO: apontar este render para o SD 3.5, que usa CFG real, devolveria
    # imagem crua sem nenhum aviso. Quem passou o valor explicitamente manda.
    #
    # BUGFIX 2026-09-02: `detect_architecture(checkpoint)` devolve a ARQUITETURA
    # ("flux1"), nao o nome do MOTOR ("flux-krea"/"flux-kontext") -- os dois
    # motores FLUX.1 compartilham arquitetura mas tem steps/guidance diferentes,
    # e `engine_defaults("flux1")` nem existe (KeyError). Resolve pelo nome do
    # ARQUIVO primeiro (bate exato com um motor conhecido); só cai pra
    # architecture-guessing se o checkpoint nao for de nenhum motor catalogado.
    padroes = None
    nome_checkpoint = Path(checkpoint).name
    for _eng, _cfg in sb.IMAGE_ENGINES.items():
        if _cfg.get("checkpoint") == nome_checkpoint:
            padroes = sb.engine_defaults(_eng)
            break
    if padroes is None:
        padroes = sb.engine_defaults(sb.detect_architecture(checkpoint))
    passos = steps if steps is not None else padroes["steps"]
    escala_cfg = cfg if cfg is not None else padroes["cfg"]
    escala_guidance = guidance if guidance is not None else padroes["guidance"]
    escala_weight_dtype = weight_dtype if weight_dtype is not None else padroes.get("weight_dtype", "default")

    todos = plan["shots"][:limit] if limit else plan["shots"]
    # Pares (indice ORIGINAL, plano). O indice tem de sobreviver ao filtro: ele
    # nomeia o still, e a chave do manifesto e a posicao no shot_plan. Renumerar
    # aqui faria o plano 2 escrever por cima do still do plano 0.
    escolhidos = set(parse_indices(only_shots, len(todos)))
    shots = [(i, s) for i, s in enumerate(todos) if i in escolhidos]
    if engine == "longcat" and not stills_only:
        # LTX (8188) e LongCat (8190) disputam a mesma 3090: todos os planos SEM fala
        # primeiro, depois os de fala, para trocar de servidor uma vez so. O indice
        # original viaja junto, entao still/clipe/manifesto nao mudam de nome.
        shots.sort(key=lambda par: (par[1].get("line_index") is not None, par[0]))
    if only_shots and len(shots) != len(todos):
        log(f"[render] {len(shots)} de {len(todos)} plano(s): "
            f"{sorted(i for i, _ in shots)}")
    # Referência por personagem: o primeiro still de alguém guia os próximos --
    # A MENOS que exista uma character sheet (cast.json, `reference_image`,
    # gerada pela aba dedicada -- ver decupagem_ui.py). Pré-semear com ela faz
    # TODO plano do personagem, desde o primeiro, usar a "verdade canônica" em
    # vez de deixar o primeiro close que calhar de sair virar a referência --
    # é a rota padrão de consistência pedida pelo usuário 2026-09-03 (MEMORIAL
    # 3.54): gera-se a cena 0/sheet UMA vez, com curadoria, e todo o resto
    # clona dela em vez de cada plano reinventar o personagem. Como a chave já
    # existe, a checagem "sujeito not in refs" mais abaixo nunca a sobrescreve
    # com um still comum -- a sheet fica fixa a corrida inteira.
    refs: dict[str, str] = dict(character_sheets or {})
    # Referência por LOCAÇÃO (nova): mesma ideia, mas para cenário -- sem ela,
    # todo plano SEM sujeito (wide/insert/estabelecimento) nunca tinha imagem
    # de referência nenhuma, e o cenário derivava plano a plano dentro da
    # MESMA cena (achado do usuário, 2026-09-03: "não mantém consistência de
    # locação"). Chave e o índice da cena (`shot["scene"]") -- é o único
    # identificador de lugar que sobrevive até aqui; `location` em si só
    # existe no texto da cena, não em cada plano (ver shot_plan.py). O
    # primeiro plano WIDE/FULL da cena (o enquadramento que mais mostra
    # cenário, simétrico ao critério de personagem que usa close/medium)
    # vira a referência dos planos seguintes sem sujeito.
    location_refs: dict[int, str] = {}
    feitos = []

    for i, shot in shots:
        t0 = time.time()
        sujeito = shot.get("subject") or ""
        cena_id = shot.get("scene")
        ref = refs.get(sujeito) if (use_reference and sujeito) else None
        # BUGFIX (auditoria externa 2026-09-16, achado #9): este fallback
        # disparava mesmo com um SUJEITO nomeado que so ainda nao tinha
        # still-ancora proprio -- um still de LOCACAO (as vezes o rosto de
        # OUTRO personagem, ou nenhum rosto) virava a "referencia de
        # identidade" dele em silencio. MEDIDO: o score de consistencia de
        # Mei-Li saiu negativo (~-0.03) num plano cuja referencia registrada
        # era, na real, um still de Xiao-Lan/cenario. Location_ref so faz
        # sentido pra plano SEM sujeito (continuidade de cenario); com
        # sujeito e sem still proprio ainda, o certo e ficar SEM referencia
        # (texto puro) -- errado e melhor que uma referencia errada that
        # parece certa.
        if ref is None and use_reference and not sujeito:
            ref = location_refs.get(cena_id)
        # SEGUNDA referencia (opcao B, 2026-09-10): o plano tem um segundo
        # personagem NOMEADO no proprio texto (`co_subject`, de shot_plan.py)
        # -- so entra se ja existe still de referencia pra ele, senao nao ha
        # o que passar. Sem `ref` (o primeiro personagem) o template dual nao
        # e usado mesmo com co_ref presente -- ReferenceLatent encadeia a
        # partir do primeiro.
        co_sujeito = shot.get("co_subject") or ""
        co_ref = refs.get(co_sujeito) if (use_reference and ref and co_sujeito) else None
        log(f"\n[plano {i} · {len(shots)} na fila] cena {shot['scene']} · {shot['style']} · "
            f"{shot['framing']}/{shot['angle']}/{shot['movement']} · "
            f"{shot['seconds']}s ({shot['frames']}f)"
            f"{' · ref=' + Path(ref).name if ref else ''}"
            f"{' · ref2=' + Path(co_ref).name if co_ref else ''}")

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
                # BUGFIX auditoria 2026-09-16 (A08): sem registrar a falha,
                # este plano desaparecia de `feitos` (nao so do vídeo) -- e
                # `render_shots_stage.build_clips_manifest` so entende "falhou"
                # o que ESTA no manifesto com ok=False; o que nunca chega nem
                # aparece. Um plano faltando podia deixar `ok == len(manifesto)`
                # trivialmente verdadeiro (inclusive 0 == 0) e marcar o estagio
                # como concluido com o filme incompleto.
                feitos.append({"shot": i, "still": None, "clip": None})
                continue
        else:
            # BUGFIX (achado testando o Fix #3, avaliacao visual 2026-09-17): o
            # gate de consistencia facial e um checagem de ROSTO -- sem sentido
            # (e sem chance de passar) num plano SEM sujeito, onde `ref` e a
            # foto de LOCACAO (fallback de `location_refs`, ver acima) ou o
            # enquadramento e `insert` ("no face in frame" por definicao do
            # proprio prompt). MEDIDO: insert e wide sem sujeito gastavam as 3
            # geracoes (1 + 2 retries) do gate, sempre abaixo do limiar (0.07 a
            # 0.23), porque nao ha rosto de personagem pra comparar -- so o
            # tempo de GPU. So aplica o gate quando ha um SUJEITO de verdade.
            gate_consistencia = consistency_threshold if sujeito else None
            still = _still_for_shot(shot, i, out_dir=stills_dir, width=width,
                                    height=height, checkpoint=checkpoint, clip=clip,
                                    vae=vae, seed=seed, reference=ref, reference_2=co_ref, log=log,
                                    steps=passos, cfg=escala_cfg, guidance=escala_guidance,
                                    weight_dtype=escala_weight_dtype,
                                    consistency_threshold=gate_consistencia,
                                    consistency_max_retries=consistency_max_retries,
                                    lora_name=lora_name, lora_strength=lora_strength)
        if still is None:
            log(f"  still falhou; pulando o plano {i}")
            # BUGFIX auditoria 2026-09-16 (A08): ver comentario equivalente no
            # ramo `videos_only` acima.
            feitos.append({"shot": i, "still": None, "clip": None})
            continue
        # Só vira referência quem foi enquadrado perto o bastante para mostrar
        # o rosto -- um wide como referência ensinaria o cenário, não a pessoa.
        if use_reference and sujeito and sujeito not in refs and \
                shot["framing"] in ("close", "extreme_close", "medium", "ots"):
            refs[sujeito] = str(still)
        # Espelho pra locação: só vira referência o plano ABERTO o bastante
        # pra mostrar cenário (o inverso do critério de personagem acima).
        if use_reference and cena_id is not None and cena_id not in location_refs and \
                shot["framing"] in ("wide", "full"):
            location_refs[cena_id] = str(still)

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
        # BUGFIX auditoria 2026-09-16 (A04): seed, fps e resolucao faltavam na
        # chave -- mudar qualquer um dos tres muda o CLIPE (composicao,
        # movimento amostrado) mas nao o numero de frames nem o audio, entao o
        # cache antigo nunca via a diferenca.
        chave += f"|seed={seed}|fps={fps}|{width}x{height}"
        # O STILL e o quadro 0 do I2V: still refeito com clipe velho e o mesmo modo de falha
        # silenciosa. VISTO 2026-09-13 (teste_close_v2): os closes novos foram gerados e os
        # 3 clipes de fala antigos, de plano medio, foram reaproveitados. Invalida uma vez
        # o cache de corridas antigas (chave nova), o que e o comportamento certo.
        chave += f"|still={_audio_key(str(still))}"
        # BUGFIX 2026-09-15: o MOTOR de video (ltx/minimax/longcat) faltava na
        # chave pra minimax -- so longcat entrava (linha abaixo, ja existia).
        # MEDIDO: trocar --video-engine ltx por --video-engine minimax no MESMO
        # run-dir reaproveitou os 14 clipes do LTX em silencio ("clipe ja existe
        # e bate com o plano"), sem gerar um unico frame no MiniMax -- o mesmo
        # modo de falha silenciosa que o comentario abaixo ja resolvia so pro
        # longcat. `engine` sempre entra agora, independente de qual for.
        chave += f"|engine={engine}"
        # BUGFIX 2026-09-16: o TEXTO do video_prompt faltava na chave. Pro LTX
        # isso quase nunca importa sozinho (o audio_conditioning real dirige a
        # boca, nao o texto), mas pro MiniMax H3 e onde a FALA em si mora
        # (comentario mais abaixo: "fala nativa a partir do texto do prompt").
        # MEDIDO: religar --include-quotes no shot_plan (pra corrigir o
        # MiniMax inventando fala) e sozinho NAO bastava -- os clipes velhos
        # batiam com a chave antiga (frames+still+engine, tudo igual) e eram
        # reaproveitados com a fala inventada de antes. Um hash curto do
        # proprio texto fecha essa lacuna pros dois motores.
        import hashlib
        chave += f"|prompt={hashlib.sha1(shot['video_prompt'].encode('utf-8')).hexdigest()[:12]}"
        if engine == "minimax":
            # Mesmo bug de cache silencioso: sem isto, ligar/desligar
            # --minimax-no-still ou --minimax-chain-max-seconds reaproveitaria
            # o clipe velho (frames/still/prompt/engine iguais) em vez de
            # regenerar do jeito novo.
            chave += f"|nostill={int(minimax_no_still)}|chain={minimax_chain_max_seconds or 0}"
            # BUGFIX auditoria 2026-09-16 (A04): aspect_ratio, megapixels, turbo,
            # ref_audio e a VARIANTE de checkpoint (fp8int8/w4a8/gguf-q4km, ver
            # CLAUDE.md) mudam o clipe do MiniMax H3 e nao entravam na chave.
            chave += (f"|ar={minimax_aspect_ratio or ''}|mp={minimax_megapixels or 0}"
                      f"|turbo={int(minimax_turbo)}|refaudio={int(minimax_ref_audio)}"
                      f"|variant={os.environ.get('MINIMAX_H3_VARIANT', '')}")
        if engine == "ltx":
            chave += f"|nostill={int(ltx_no_still)}|chain={ltx_chain_max_seconds or 0}"
            # BUGFIX auditoria 2026-09-16 (A04): a VARIANTE do transformer 2.5
            # (distilled/dev/gguf-q6k/w4a8-v10, CLAUDE.md) e o modo two-stage
            # mudam o clipe e nao entravam na chave -- trocar LTX25_VARIANT no
            # mesmo run-dir reaproveitaria o clipe da variante anterior.
            chave += (f"|variant={os.environ.get('LTX25_VARIANT', '')}"
                      f"|twostage={os.environ.get('LTX25_TWO_STAGE', '')}")
        if engine == "longcat" and wav_cond:
            # Mesmo plano, outro motor: sem isto um clipe LTX existente seria reaproveitado.
            chave += f"|longcat={os.environ.get('LONGCAT_VARIANT', '1.5')}"
        elif engine == "longcat":
            # BUGFIX auditoria 2026-09-16 (A04): plano de ACAO com engine=longcat
            # cai no ramo LTX mais abaixo (so falas vao pro LongCat de verdade) --
            # sem isto a variante do LTX 2.5 nao entrava na chave destes clipes.
            chave += (f"|variant={os.environ.get('LTX25_VARIANT', '')}"
                      f"|twostage={os.environ.get('LTX25_TWO_STAGE', '')}")
        # LoRAs de video e IC-LoRA (2026-09-12) mudam o CLIPE do mesmo jeito que trocar o
        # audio -- entram na chave, senao ligar um LoRA reaproveitaria o clipe velho em
        # silencio. So quando ligados: corrida sem eles mantem a chave antiga e o cache.
        prompt_video, ic_spec = shot["video_prompt"], None
        if engine != "minimax":
            if video_loras:
                chave += "|loras=" + ",".join(f"{n}@{s:g}" for n, s in video_loras)
            if ic_mode and ic_mode != "off":
                from script_pipeline import ic_references
                ic_spec, prompt_video, chave_ic = ic_references.shot_ic_spec(
                    ic_mode, shot, str(still), refs, location_refs.get(cena_id),
                    cast_descriptors, width=width, height=height, num_frames=shot["frames"],
                    work_dir=clips_dir / f"shot{i:03d}_ic", lora=ic_lora,
                    strength=ic_strength, guide_strength=ic_guide_strength)
                chave += chave_ic
                if ic_spec is None:
                    if ic_mode == "msr" and shot.get("framing") in ic_references.MSR_SEM_ENQUADRAMENTO_ABERTO:
                        log(f"  IC-LoRA (msr): plano {shot.get('framing')} -- sem guia, so o still "
                            f"(retratos da guia fariam o video largar o still no quadro 1)")
                    else:
                        log(f"  IC-LoRA ({ic_mode}): plano sem personagem com referencia ou curto "
                            f"demais para a guia -- so o still")
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
            #
            # BUGFIX auditoria 2026-09-16 (A04): comparar `atual` contra
            # `shot["frames"]` so faz sentido pro LTX -- e a grade 8k+1 que o
            # shot_plan usou pra calcular esse numero. MiniMax H3 (5+17k a
            # 24fps) e LongCat (4k+1 a 25fps, normalizado na montagem) NUNCA
            # batem exatamente com `shot["frames"]`, entao esta checagem
            # regenerava o clipe TODA corrida pra esses dois motores -- ou,
            # pior, um clipe do motor errado com contagem parecida por
            # coincidencia passaria pela MESMA checagem fraca. Guarda o par
            # (chave, frames REALMENTE produzidos) no marcador e compara com
            # o proprio arquivo, nao com o alvo de outro motor.
            marcador = {}
            if marca.exists():
                try:
                    marcador = json.loads(marca.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    marcador = {}
            antiga, frames_gravados = marcador.get("key"), marcador.get("frames")
            if antiga == chave and atual is not None and atual == frames_gravados:
                log(f"  clipe ja existe e bate com o plano, reaproveitando: {clip_path.name}")
                feitos.append({"shot": i, "still": str(still), "clip": str(clip_path)})
                continue
            if antiga == chave and atual != frames_gravados:
                log(f"  clipe existente tem {atual}f mas o registro esperava {frames_gravados}f "
                    "(arquivo truncado/incompleto?); refazendo")
            else:
                log(f"  clipe existente foi gerado com outro plano/audio/config; refazendo")
        def _minimax_chain_generate(refs_minimax, sheet_ref, audio_refs_minimax, out_path, log):
            """Plano longo dividido em sub-planos curtos e ENCADEADOS: cada
            sub-plano usa a sheet (identidade) e o ULTIMO FRAME do sub-plano
            anterior (continuidade de movimento/cenario) -- mesmo principio do
            `continuous_chain.py` pro LTX. Nasceu como teste isolado
            (`_test_minimax_duration_cap.py`, 2026-09-03, MEMORIAL 3.47/3.49):
            MEDIDO la que dividir evita o teto de ~16s onde o plano inteiro
            fica instavel (30min+ sem terminar). Pedido do usuario 2026-09-16:
            expor isso como opcao de producao, nao so teste manual.

            NAO tenta dividir a FALA -- cada sub-plano recebe o MESMO
            video_prompt inteiro (a fala completa do plano). Pra dialogo, isso
            e aceitavel so quando o prompt ja carrega a fala inteira (a boca
            sincroniza no sub-plano que ela cair, o resto fica em silencio ou
            repetindo a articulacao -- REVISAR visualmente antes de confiar
            pra producao continua)."""
            import math
            from script_pipeline.assemble_final import concat_videos

            duracao_total = shot["frames"] / fps
            n_partes = max(1, math.ceil(duracao_total / minimax_chain_max_seconds))
            seg_seconds = round(duracao_total / n_partes, 2)
            log(f"    [minimax_h3-chain] {duracao_total:.1f}s dividido em {n_partes} "
                f"sub-plano(s) de ~{seg_seconds}s cada")

            work_dir = clip_path.parent / f"{clip_path.stem}_chain"
            work_dir.mkdir(parents=True, exist_ok=True)
            partes, frame_anterior = [], None
            for j in range(n_partes):
                # BUGFIX auditoria 2026-09-16 (A05): `(refs_j + [frame_anterior])[:2]`
                # mantinha os dois primeiros elementos de `refs_minimax` (still do
                # plano + sheet) e DESCARTAVA o frame_anterior, que e justamente o
                # elo de continuidade que este encadeamento existe para dar --
                # MEDIDO: a segunda chamada em diante nunca recebia o ultimo frame.
                # A partir do segundo sub-plano a prioridade e explicita: sheet
                # (identidade, se existir) + frame_anterior (continuidade) --
                # o still do PLANO INTEIRO nao faz mais sentido a partir daqui,
                # o frame anterior ja e o quadro 0 mais atual.
                if frame_anterior:
                    refs_j = [r for r in [sheet_ref, frame_anterior] if r] or [frame_anterior]
                else:
                    refs_j = list(refs_minimax)
                sub_path = work_dir / f"sub{j:02d}.mp4"
                minimax_h3_backend.generate(
                    shot["video_prompt"], str(sub_path),
                    ref_images=refs_j[:2] or None, ref_audios=audio_refs_minimax,
                    aspect_ratio=minimax_aspect_ratio or minimax_h3_backend.DEFAULT_ASPECT,
                    megapixels=minimax_megapixels if minimax_megapixels is not None else minimax_h3_backend.DEFAULT_MEGAPIXELS,
                    duration_seconds=seg_seconds, seed=seed + i + j * 101, turbo=minimax_turbo,
                    log_cb=lambda m, j=j: log(f"    [minimax_h3-chain sub{j}] {m}"), timeout=1800)
                partes.append(str(sub_path))
                frame_path = work_dir / f"sub{j:02d}_last.png"
                if _extrair_ultimo_frame(str(sub_path), frame_path):
                    frame_anterior = str(frame_path)
                else:
                    log(f"    [minimax_h3-chain] sub{j:02d}: nao consegui extrair "
                        f"ultimo frame -- proximo sub-plano perde a continuidade "
                        f"de movimento (mantem so a sheet).")
            # BUGFIX auditoria 2026-09-16 (A07): concat_videos devolve False em
            # falha e o retorno era ignorado -- o caller registrava `out_path`
            # como clipe pronto mesmo quando ele nao foi (re)escrito (arquivo
            # ausente ou, pior, sobra de uma corrida anterior). Levanta, para
            # cair no mesmo `except` que ja marca o plano como FALHOU.
            if not concat_videos(partes, out_path, work_dir=work_dir / "intermediate", log=log):
                raise RuntimeError(f"concat dos {len(partes)} sub-planos do encadeamento MiniMax falhou")

        def _split_audio(wav: str, seg_seconds: list, out_dir: Path) -> list:
            """Corta `wav` em pedacos consecutivos, um por segmento do
            encadeamento -- ver `_ltx_chain_generate`/BUGFIX A06. Segmento que
            cai inteiramente depois do fim do audio recebe None (sem
            condicionamento): melhor um sub-plano sem trilha de referencia do
            que repetir o WAV inteiro fora de hora (o LTX inventaria voz por
            cima da fala real nos sub-planos seguintes)."""
            total = None
            try:
                r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", wav], capture_output=True, text=True)
                total = float(r.stdout.strip())
            except (ValueError, OSError):
                pass
            out, t = [], 0.0
            for j, dur in enumerate(seg_seconds):
                inicio = t
                t += dur
                if total is not None and inicio >= total:
                    out.append(None)
                    continue
                dest = out_dir / f"audio_seg{j:02d}.wav"
                r = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", wav,
                                    "-ss", f"{inicio:.3f}", "-t", f"{dur:.3f}", str(dest)],
                                   capture_output=True, text=True)
                out.append(str(dest) if r.returncode == 0 and dest.exists() else None)
            return out

        def _ltx_chain_generate(image_inicial, out_path, log):
            """Mesmo principio do `_minimax_chain_generate`, pro LTX 2.5:
            divide o plano em sub-clipes curtos, cada um usando o ULTIMO frame
            do anterior como `image_path` (I2V) -- e exatamente o mecanismo
            de `continuous_chain.py` (ja validado, 2.3/2.5), só que aplicado
            DENTRO de um plano da decupagem em vez de fora, orquestrado por
            script standalone. Pedido do usuario 2026-09-16."""
            import math
            from script_pipeline.assemble_final import concat_videos

            duracao_total = shot["frames"] / fps
            n_partes = max(1, math.ceil(duracao_total / ltx_chain_max_seconds))
            # BUGFIX auditoria 2026-09-16 (A06): a versao anterior usava o MESMO
            # `shot["frames"] // n_partes` (arredondado pro grid 8k+1) em TODOS
            # os segmentos, descartando o resto INTEIRO uma vez por segmento --
            # MEDIDO: um plano de 401f em 3 partes perdia 14f (~0,58s a 24fps).
            # Agora so o ultimo segmento absorve o resto (perda de no maximo 8f,
            # a granularidade do proprio grid), e a diferenca final e logada em
            # vez de ficar silenciosa.
            base = shot["frames"] // n_partes
            seg_frames, restante = [], shot["frames"]
            for j in range(n_partes):
                alvo = restante if j == n_partes - 1 else base
                seg_frames.append(max(9, ((alvo - 1) // 8) * 8 + 1))
                restante -= alvo
            total_real = sum(seg_frames)
            if total_real != shot["frames"]:
                log(f"    [ltx25-chain] grade 8k+1 por segmento: {shot['frames']}f pedidos, "
                    f"{total_real}f reais ({total_real - shot['frames']:+d}f)")
            log(f"    [ltx25-chain] {duracao_total:.1f}s dividido em {n_partes} "
                f"sub-plano(s): {[f'{f}f' for f in seg_frames]}")

            work_dir = clip_path.parent / f"{clip_path.stem}_chain"
            work_dir.mkdir(parents=True, exist_ok=True)
            # BUGFIX auditoria 2026-09-16 (A06): so o PRIMEIRO sub-plano recebia
            # `wav_cond` inteiro como audio_conditioning; os seguintes iam sem
            # nada e o LTX 2.5 inventava voz propria por cima da fala real
            # (mesmo defeito que audio_conditioning existe pra evitar, ver
            # comentario mais abaixo). Corta o WAV pelo tempo de cada segmento.
            wavs_seg = _split_audio(wav_cond, [f / fps for f in seg_frames], work_dir) \
                if wav_cond else [None] * n_partes

            partes, imagem_atual = [], image_inicial
            for j in range(n_partes):
                sub_path = work_dir / f"sub{j:02d}.mp4"
                ltx25_backend.generate(
                    prompt_video, str(sub_path),
                    width=width, height=height, num_frames=seg_frames[j],
                    frame_rate=fps, seed=seed + i + j * 101,
                    image_path=imagem_atual, image_strength=1.0,
                    loras=video_loras or None, ic_lora=ic_spec,
                    audio_conditioning=wavs_seg[j],
                    log_cb=lambda m, j=j: log(f"    [ltx25-chain sub{j}] {m}"), timeout=2400)
                partes.append(str(sub_path))
                frame_path = work_dir / f"sub{j:02d}_last.png"
                if _extrair_ultimo_frame(str(sub_path), frame_path):
                    imagem_atual = str(frame_path)
                else:
                    log(f"    [ltx25-chain] sub{j:02d}: nao consegui extrair ultimo "
                        f"frame -- proximo sub-plano perde a continuidade.")
            # BUGFIX auditoria 2026-09-16 (A07): ver comentario no encadeamento MiniMax acima.
            if not concat_videos(partes, out_path, work_dir=work_dir / "intermediate", log=log):
                raise RuntimeError(f"concat dos {len(partes)} sub-planos do encadeamento LTX falhou")

        try:
            if engine == "minimax":
                # MiniMax H3 fala e sincroniza labios NATIVAMENTE a partir do
                # texto do prompt (o proprio video_prompt do plano ja carrega
                # a fala, quando o roteiro/decoupage a colocou la) -- nao ha
                # audio_conditioning aqui, e o TTS/lipsync/mix do pipeline
                # ficam sem o que fazer para estes planos (ver aviso no
                # decupagem_ui.py e no CLI). `duration_seconds` e arredondado
                # pro grid do proprio modelo (5+17k a 24fps), NAO o 8k+1 do
                # LTX que o shot_plan usou pra calcular `shot["frames"]" --
                # a duracao final pode divergir um pouco do planejado.
                # MEDIDO 2026-09-02: um plano de 16,6s (401f) estourou 2400s
                # (40 min) sem terminar -- bem alem de qualquer duracao ja
                # validada pro MiniMax H3 (MEMORIAL 3.41 ja registrava 10s
                # como estresse). 3600s da mais folga sem mascarar uma trava
                # de verdade (a trava do 3.51/§3.[novo] destravou sozinha em
                # 1727s -- 3600s cobre esse caso e o dobro dele).
                # Rota de consistência padrão (MEMORIAL 3.54): quando o
                # personagem tem sheet canônica, manda os DOIS refs que o
                # grafo do MiniMax aceita -- o still DESTE plano (pose/
                # enquadramento) e a sheet (identidade -- a mesma imagem que
                # já ancorou o still na etapa de FLUX, reforçada aqui de novo
                # no motor de vídeo). Sem sheet, cai no still sozinho, igual
                # sempre foi.
                # --minimax-no-still (pedido do usuario 2026-09-16): "e se
                # usarmos so os descritivos?" -- deixa de passar o still deste
                # plano como referencia, so o TEXTO (video_prompt, que ja
                # carrega o descritor do personagem) guia a identidade. NAO
                # remove a sheet quando ela existe (a sheet e a ancora entre
                # PLANOS diferentes do mesmo personagem -- tirar ela tambem
                # devolveria exatamente o drift que o still existe pra evitar;
                # ver discussao no chat). Sem sheet nem still, e T2V puro.
                refs_minimax = [] if minimax_no_still else ([str(still)] if still else [])
                sheet_do_sujeito = (character_sheets or {}).get(sujeito)
                if sheet_do_sujeito and sheet_do_sujeito not in refs_minimax:
                    refs_minimax.append(sheet_do_sujeito)
                # ref_audios (MEMORIAL 3.74, 2026-09-08): opt-in -- so testado
                # ate agora com UMA fala isolada, nao com a cadeia de producao
                # inteira. `wav_cond` (calculado acima, mesma fala que o LTX
                # usaria como audio_conditioning) e o timbre/cadencia REAIS
                # do TTS pra ESTE plano -- reusa em vez de recalcular.
                audio_refs_minimax = [wav_cond] if (minimax_ref_audio and wav_cond) else None
                duracao_plano = shot["frames"] / fps
                if minimax_chain_max_seconds and duracao_plano > minimax_chain_max_seconds:
                    _minimax_chain_generate(refs_minimax, sheet_do_sujeito, audio_refs_minimax,
                                            clip_path, lambda m: log(m))
                else:
                    minimax_h3_backend.generate(
                        shot["video_prompt"], str(clip_path),
                        ref_images=refs_minimax[:2] or None,
                        ref_audios=audio_refs_minimax,
                        aspect_ratio=minimax_aspect_ratio or minimax_h3_backend.DEFAULT_ASPECT,
                        megapixels=minimax_megapixels if minimax_megapixels is not None else minimax_h3_backend.DEFAULT_MEGAPIXELS,
                        duration_seconds=duracao_plano,
                        seed=seed + i, turbo=minimax_turbo,
                        log_cb=lambda m: log(f"    [minimax_h3] {m}"), timeout=3600)
            elif engine == "longcat" and wav_cond:
                # Plano de FALA no LongCat-Avatar 1.5 (MEMORIAL 3.83): boca gerada junto com
                # o video a partir do wav do TTS -- o lip-sync depois e opcional. Os planos
                # sem fala vieram antes (ordenacao acima) no LTX; aqui a 8188 sai do ar uma
                # vez. O modelo e treinado a 25 fps com quadros 4k+1; a montagem normaliza
                # para 24 fps.
                if sb.comfy_is_up("http://127.0.0.1:8188"):
                    log("[render] motor=longcat: derrubando o ComfyUI do LTX 2.5 (8188) antes do LongCat (8190).")
                    sb.stop_comfyui(8188, log=log)
                nf = max(17, int(round(shot["frames"] / fps * 25)))
                nf = ((nf - 1 + 3) // 4) * 4 + 1
                longcat_video_backend.generate(
                    shot["video_prompt"], str(clip_path), image_path=str(still),
                    audio_path=wav_cond, num_frames=nf, seed=seed + i, fps=25.0,
                    log_cb=lambda m: log(f"    {m}"))
            else:
                if engine == "longcat" and not sb.comfy_is_up("http://127.0.0.1:8188"):
                    longcat_video_backend.stop_server(log_cb=log)
                    os.environ["LTX_COMFY_CACHE_NONE"] = "1"
                    sb.ensure_comfyui_running("http://127.0.0.1:8188", log=log)
                # --ltx-no-still (pedido do usuario 2026-09-16, mesmo espirito
                # do --minimax-no-still): ltx25_backend.generate ja aceita
                # image_path=None (T2V puro) -- so nunca era exposto aqui, o
                # caminho da decupagem sempre ancorava no still (I2V).
                imagem_ltx = None if ltx_no_still else str(still)
                duracao_ltx = shot["frames"] / fps
                if ltx_chain_max_seconds and duracao_ltx > ltx_chain_max_seconds:
                    _ltx_chain_generate(imagem_ltx, clip_path, lambda m: log(m))
                else:
                    ltx25_backend.generate(
                        prompt_video, str(clip_path),
                        width=width, height=height, num_frames=shot["frames"],
                        frame_rate=fps, seed=seed + i,
                        image_path=imagem_ltx, image_strength=1.0,
                        loras=video_loras or None, ic_lora=ic_spec,
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
                f"{' (boca gerada pelo LongCat 1.5)' if (engine == 'longcat' and wav_cond) else ''}"
                f"{' (som condicionado pela fala)' if (engine == 'ltx' and wav_cond) else ''}"
                f"{' (fala nativa do MiniMax H3)' if engine == 'minimax' else ''}")
            if "freeze" in (shot.get("post_effects") or []):
                _apply_freeze(clip_path, log=log)
            marca.write_text(json.dumps({"key": chave, "frames": _clip_frames(clip_path)}),
                             encoding="utf-8")
            feitos.append({"shot": i, "still": str(still), "clip": str(clip_path)})
        except Exception as e:
            log(f"  clipe FALHOU: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            feitos.append({"shot": i, "still": str(still), "clip": None})
            # MEDIDO 2026-09-02: um TimeoutError no MiniMax H3 nao mata o
            # servidor -- a geracao orfa continua rodando por tras, e o
            # PROXIMO plano ve a porta "up" mas presa, tenta subir de novo e
            # colide ("Port 8189 is already in use"), derrubando um plano que
            # nao tinha nada de errado. Depois de QUALQUER falha no motor
            # minimax, forca reinicio antes do proximo plano -- mesma
            # disciplina do stop_comfyui(8189) que o run_decupagem.py agora
            # faz entre corridas, so que aqui e DENTRO da mesma corrida.
            if engine == "minimax":
                log("  reiniciando o servidor do MiniMax H3 antes do proximo plano (falha pode ter deixado geracao orfa presa na porta).")
                sb.stop_comfyui(8189, log=log)
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
            # + tamanho do clipe: um clipe refeito (still novo) com a mesma contagem de
            # quadros herdava o lip-sync do clipe velho (VISTO 2026-09-13).
            chave_ls = f"{_clip_frames(base)}|{base.stat().st_size}|{Path(wav).stat().st_size}"
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
    from script_pipeline.generate_storyboards import IMAGE_ENGINES as _IMAGE_ENGINES_CLI
    ap.add_argument("--image-engine", default="flux", choices=sorted(_IMAGE_ENGINES_CLI),
                    help='motor de imagem dos stills. flux = melhor adesao a enquadramento e lado de tela e aceita imagem de referencia, mas carrega ~4 min e encosta no teto de VRAM da 3090. sd35 = carrega em ~1 min, cabe em ~12 GB e amostra mais rapido, mas obedece menos o enquadramento e NAO tem referencia. flux-krea/flux-kontext = FLUX.1, ver MEMORIAL 3.48. --checkpoint/--clip/--vae explicitos continuam ganhando disto.')
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
    # BUGFIX 2026-09-03 (mesmo do render_shots_stage.py, MEMORIAL 3.52): sem
    # isto, flux-kontext carregava o bf16 de 23,8 GB cru, sem o cast fp8_e4m3fn
    # que precisa pra caber.
    weight_dtype = _motor.get("weight_dtype", "default")

    feitos = render(plan, out_dir, width=args.width, height=args.height, fps=args.fps,
                    checkpoint=args.checkpoint, clip=args.clip, vae=args.vae,
                    seed=args.seed, limit=args.limit, stills_only=args.stills_only,
                    videos_only=args.videos_only, steps=args.steps, cfg=args.cfg,
                    only_shots=args.only_shots, dialogue=dialogo,
                    audio_conditioning=not args.no_audio_conditioning,
                    use_reference=not args.no_reference, weight_dtype=weight_dtype)
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
