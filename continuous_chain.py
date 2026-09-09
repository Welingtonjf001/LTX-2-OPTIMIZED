"""Camada 0 do plano de video continuo por minutos (MEMORIAL.md, secao "Video
continuo por encadeamento" -- ver tambem CLAUDE.md, que so documenta 2.3/2.5
por clipe curto).

Generaliza para fora do roteiro/screenplay o mecanismo de chain-continuity ja
validado em duas UIs deste repo (music_maker_ui_v2.py::process_chain_generation
e script_pipeline/render_scenes.py::render_job/extract_last_frame): gera um
clipe LTX 2.3, extrai o ULTIMO frame decodificado, usa esse frame como imagem
de condicionamento (--image PATH 0 STRENGTH, ou seja VideoConditionByLatentIndex
substituindo o latente inicial) do clipe seguinte, e assim por diante. Nao
treina nada -- e pura orquestracao de script (Camada 2 no vocabulario da
analise Gemini registrada no MEMORIAL). Drift de identidade entre chunks nao e
resolvido aqui; se aparecer, a proxima etapa e um LoRA de personagem, nao mexer
neste script.

CLI:
  python continuous_chain.py --run-dir outputs/continuous/teste01 \
    --prompt "a woman walking through a rain-soaked neon street, steady tracking shot" \
    --chunks 6 --seconds-per-chunk 5.0 --width 896 --height 512 --fps 24 --seed 1234

  # prompt por chunk (uma linha por chunk; define o numero de chunks):
  python continuous_chain.py --run-dir outputs/continuous/teste01 \
    --prompt-file prompts_teste01.txt --seconds-per-chunk 5.0

Retomavel: se outputs/continuous/<run>/clips/chunk_XXX.mp4 ja existe, o chunk
NAO e regerado -- so o frame final e reextraido para alimentar o proximo. Isso
segue o mesmo principio de retomada do decupagem_ui (nao recomeca do zero numa
corrida interrompida).

Ao final, imprime o comando video_doctor sugerido com --cuts ja calculado a
partir das fronteiras reais de cada chunk -- MEMORIAL.md ja registrou que sem
isso o doctor confunde fronteira de encadeamento com defeito temporal.

FALA E SOM AMBIENTE (2026-09-02): por padrao os chunks saiam mudos no 2.3
(o `music_to_video` so decodifica audio quando ha `--audio-input-path` OU
`LTX_GENERATE_AMBIENT_AUDIO=1`) e com ambiente generico no 2.5. Dois jeitos
de dar som:

  1. --prompt/--prompt-file (sem fala): agora liga som AMBIENTE por padrao
     (LTX_GENERATE_AMBIENT_AUDIO=1 no 2.3; o 2.5 ja gera ambiente sozinho).
     --no-ambient-audio desliga, volta ao silencio de antes.

  2. --beats-file arquivo.json: um beat pode ter FALA de verdade, sintetizada
     via XTTS (script_pipeline/dialogue_tts.py, mesmo motor do pipeline de
     roteiro) e passada como --audio-input-path -- o video e GERADO em cima do
     audio real, nao dublado depois (mesmo motivo do render_scenes.py: sem
     isso o LTX inventa uma voz propria). A duracao do chunk vira a duracao da
     fala sintetizada (arredondada pra grade 8k+1), nao --seconds-per-chunk.
     Formato de cada item da lista:
       {"prompt": "...",                          # igual ao --prompt-file
        "speech_text": "...", (opcional)           # fala literal, sintetizada
        "speaker": "M01_jovem_leve_andy",           # pasta em xtts/webui/speakers/01_vozes_emotivas/
        "language": "zh-cn", (opcional)             # default --speech-language
        "emotion": "15_calma.wav"} (opcional)        # default --speech-emotion
     Beat sem "speech_text" usa --seconds-per-chunk e ambiente (regra 1).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "script_pipeline"))

DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
LTX_PIPELINE_MODULE = os.environ.get("LTX_PIPELINE_MODULE", "ltx_pipelines.music_to_video")

# Mesmo diretorio de vozes emotivas que o pipeline de roteiro usa (CLAUDE.md:
# speakers/01_vozes_emotivas/<ID>/<NN>_<emocao>.wav, 6s+ por amostra, fala nao
# canto). Hardcoded de proposito -- e o mesmo caminho absoluto ja hardcoded em
# script_pipeline/dialogue_tts.py (XTTS_SPEAKER_DIR), nao um symlink do repo.
EMOTIVE_VOICES_DIR = Path(r"E:\Users\home\Documents\xtts\webui\speakers\01_vozes_emotivas")


def normalize_ltx_frames(value) -> int:
    """Encaixa na grade 8k+1 do VAE do LTX -- mesma formula do render_scenes."""
    raw = max(9, int(round(value)))
    return 8 * max(1, round((raw - 1) / 8)) + 1


def extract_last_frame(video_path: str, output_path: Path) -> bool:
    """Mesmo frame-grab via OpenCV ja validado em render_scenes.py/
    music_maker_ui_v2.py. Duplicado aqui de proposito (convencao deste repo:
    ver CLAUDE.md sobre as 17 UIs duplicadas nao consolidadas) para manter
    este script sem dependencia do modulo screenplay."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, frame = cap.read()
    cap.release()
    if not success:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def extract_tail_clip(video_path: str, num_frames: int, output_path: Path) -> bool:
    """Experimento de memoria de movimento (vs. extract_last_frame): recorta os
    ULTIMOS num_frames pixels do chunk anterior num mp4 pequeno, em vez de um
    frame estatico. Passado como --video-head (ltx_pipelines.utils.helpers.
    video_head_conditionings) o VAE encoda os frames JUNTOS, entao o latente
    resultante carrega direcao/velocidade/fase do gesto -- uma foto isolada nao
    informa se a pessoa estava andando, parando ou comecando um salto (e o
    ponto de partida desta ideia, ver conversa no MEMORIAL). Ainda nao medido
    se isso reduz a descontinuidade na fronteira -- e o proprio experimento."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if frame_count <= 0:
        cap.release()
        return False
    start = max(0, frame_count - num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path.exists()


def _resolve_speaker_wav(speaker: str, emotion: str) -> str:
    """Aceita um .wav absoluto direto, ou o nome de uma pasta de voz em
    EMOTIVE_VOICES_DIR (ex: "M01_jovem_leve_andy" + emotion "15_calma.wav")."""
    if speaker.lower().endswith(".wav") and (os.path.isabs(speaker) or Path(speaker).exists()):
        return speaker
    return str(EMOTIVE_VOICES_DIR / speaker / emotion)


def _wav_duration_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def _ensure_stereo(audio_path: str, work_dir: Path) -> str:
    """Mesmo conserto de render_scenes.py::_ensure_stereo -- ltx_pipelines exige
    audio de 2 canais, e o XTTS entrega mono. -ac 2 duplica o canal (nao
    remixa/perde energia como aformat=channel_layouts=stereo -- MEMORIAL.md
    3.36.2 documenta essa diferenca)."""
    import soundfile as sf

    info = sf.info(audio_path)
    if info.channels >= 2:
        return audio_path
    work_dir.mkdir(parents=True, exist_ok=True)
    stereo_path = work_dir / (Path(audio_path).stem + "_stereo.wav")
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", audio_path, "-ac", "2", str(stereo_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not stereo_path.exists():
        raise RuntimeError(f"Falha ao converter {audio_path} para estereo: {result.stderr}")
    return str(stereo_path)


def synthesize_beat_speech(beats: list[dict], *, language: str, emotion: str,
                            tts_engine: str, work_dir: Path, log) -> None:
    """Sintetiza a fala de todo beat com "speech_text", em lote (mesma API do
    pipeline de roteiro: script_pipeline/dialogue_tts.synthesize_batch). Muta
    `beats` in place, acrescentando "audio_path" (wav estereo) e "duration_sec"
    aos beats com fala -- essa duracao passa a REGER o num_frames do chunk
    (mesma dependencia do CLAUDE.md: "emocao muda a duracao da fala; a duracao
    da fala define a duracao do plano")."""
    from script_pipeline.dialogue_tts import synthesize_batch

    speech_dir = work_dir / "speech"
    jobs = []
    indices = []
    for i, beat in enumerate(beats):
        text = beat.get("speech_text")
        if not text:
            continue
        speaker = beat.get("speaker") or "M01_jovem_leve_andy"
        beat_emotion = beat.get("emotion") or emotion
        jobs.append({
            "id": f"beat{i:03d}",
            "text": text,
            "language_code": beat.get("language", language),
            "xtts_speaker_wav": _resolve_speaker_wav(speaker, beat_emotion),
            "output_path": str(speech_dir / f"beat{i:03d}.wav"),
        })
        indices.append(i)
    if not jobs:
        return
    log(f"sintetizando {len(jobs)} fala(s) via TTS ({tts_engine})...")
    results = synthesize_batch(jobs, engine=tts_engine, log=log)
    for i, result in zip(indices, results):
        if not result.get("ok"):
            log(f"beat {i}: TTS falhou ({result.get('error')}) -- vira chunk mudo/ambiente.")
            continue
        stereo = _ensure_stereo(result["output_path"], speech_dir)
        beats[i]["audio_path"] = stereo
        beats[i]["duration_sec"] = _wav_duration_seconds(stereo)
        log(f"beat {i}: fala sintetizada, {beats[i]['duration_sec']:.1f}s -> {stereo}")


def _run_streamed(command: list[str], *, env: dict, log) -> int:
    log(f"Comando: {' '.join(command)}")
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        log(line.rstrip("\n"))
    process.wait()
    return process.returncode


def render_chunk(  # noqa: PLR0913
    *, prompt: str, output_path: Path, checkpoint: str, gemma_root: str, upsampler: str,
    width: int, height: int, fps: int, num_frames: int, steps: int, seed: int,
    chain_image: str | None, chain_strength: float,
    end_image: str | None, end_image_strength: float,
    negative_prompt: str | None, log,
    audio_input_path: str | None = None, ambient_audio: bool = False,
    video_head: str | None = None, video_head_frames: int = 0, video_head_strength: float = 0.8,
) -> bool:
    command = [
        sys.executable, "-u", "-m", LTX_PIPELINE_MODULE,
        "--distilled-checkpoint-path", checkpoint,
        "--gemma-root", gemma_root,
        "--spatial-upsampler-path", upsampler,
        "--prompt", prompt,
        "--output-path", str(output_path),
        "--width", str(width), "--height", str(height),
        "--num-frames", str(num_frames), "--frame-rate", str(fps),
        "--num-inference-steps", str(steps), "--seed", str(seed),
        "--quantization", "fp8-cast",
    ]
    if negative_prompt:
        command += ["--negative-prompt", negative_prompt]
    # frame_idx 0 -> VideoConditionByLatentIndex, substitui o latente de abertura
    # (I2V puro). E o mesmo mecanismo de render_scenes.py, so que aqui encadeado
    # sem storyboard nem roteiro por tras -- so o frame anterior mesmo.
    # video_head ocupa o MESMO slot (latent_idx=0) com um trecho real em vez de
    # imagem unica -- nao combine os dois (--motion-history-frames desliga
    # chain_image sozinho, ver loop principal).
    if video_head:
        command += ["--video-head", video_head, str(video_head_frames), str(video_head_strength)]
    elif chain_image:
        command += ["--image", chain_image, "0", str(chain_strength)]
    if end_image:
        command += ["--image", end_image, str(max(1, num_frames - 1)), str(end_image_strength)]
    if audio_input_path:
        # Condiciona a GERACAO no audio real (fala sintetizada ou trilha), em vez
        # de dublar depois -- mesma razao do render_scenes.py: sem isso o LTX
        # inventa uma voz propria e o resultado nao bate com a fala pretendida.
        command += ["--audio-input-path", audio_input_path]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090, mesma convencao do resto do repo
    env.setdefault("LTX_TRANSFORMER_GPU_MEMORY", "18GiB")
    env.setdefault("LTX_TRANSFORMER_CPU_MEMORY", "32GiB")
    env.setdefault("LTX_TEXT_ENCODER_GPU_MEMORY", "4GiB")
    env.setdefault("LTX_UPSAMPLER_GPU_MEMORY", "18GiB")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    if ambient_audio and not audio_input_path:
        # So o 2.3 nativo (ltx_pipelines.music_to_video) precisa disto -- ele so
        # decodifica o latente de audio quando ha --audio-input-path OU esta env
        # var (render_scenes.py ja documenta essa regra). O 2.5 (ltx_pipelines_25)
        # ignora a variavel e gera ambiente sozinho de qualquer jeito.
        env["LTX_GENERATE_AMBIENT_AUDIO"] = "1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    code = _run_streamed(command, env=env, log=log)
    if code != 0:
        log(f"FALHOU (codigo {code}).")
        return False
    if not output_path.exists():
        log("subprocesso terminou OK mas nao criou o arquivo de saida.")
        return False
    return True


def load_beats(args) -> list[dict]:
    """Devolve uma lista de beats {"prompt": str, "speech_text": str|None, ...}.
    --beats-file (JSON) e o unico jeito de pedir fala de verdade; --prompt/
    --prompt-file continuam existindo, sem fala (so ambiente, regra 1 do
    docstring do modulo)."""
    if args.beats_file:
        raw = json.loads(Path(args.beats_file).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise SystemExit(f"{args.beats_file} precisa ser uma lista JSON nao vazia de beats.")
        beats = []
        for item in raw:
            if not item.get("prompt"):
                raise SystemExit(f"beat sem \"prompt\": {item}")
            beats.append(dict(item))
        return beats
    if args.prompt_file:
        lines = [ln.strip() for ln in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln and not ln.startswith("#")]
        if not prompts:
            raise SystemExit(f"{args.prompt_file} nao tem nenhuma linha de prompt.")
        return [{"prompt": p} for p in prompts]
    if not args.prompt:
        raise SystemExit("Passe --prompt, --prompt-file ou --beats-file.")
    return [{"prompt": args.prompt} for _ in range(args.chunks)]


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prompt", default=None, help="prompt unico, reaproveitado em todos os chunks.")
    parser.add_argument("--prompt-file", default=None,
                         help="um prompt por linha; cada linha vira um chunk (ignora --chunks).")
    parser.add_argument("--beats-file", default=None,
                         help="lista JSON de beats, com fala real opcional por beat -- "
                              "ver docstring do modulo. Ignora --prompt/--prompt-file/--chunks.")
    parser.add_argument("--speech-language", default="en",
                         help="idioma default da fala (ISO, ex: en/pt/zh-cn) quando o beat nao "
                              "declara \"language\" -- ver script_pipeline/dialogue_tts.py.")
    parser.add_argument("--speech-emotion", default="16_neutra.wav",
                         help="arquivo default dentro da pasta de voz do falante "
                              "(EMOTIVE_VOICES_DIR/<speaker>/<emotion>) quando o beat "
                              "nao declara \"emotion\".")
    parser.add_argument("--tts-engine", default="auto", choices=["auto", "xtts", "qwen", "fish"],
                         help="fish precisa do servidor do fish-speech ja no ar "
                              "(fish-speech/START_API.ps1) -- ver MEMORIAL 3.53/3.55.")
    parser.add_argument("--no-ambient-audio", dest="ambient_audio", action="store_false", default=True,
                         help="desliga o som ambiente automatico nos chunks SEM fala "
                              "(volta ao silencio de antes no 2.3; nao afeta o 2.5, que "
                              "gera ambiente sozinho independente desta flag).")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--chunks", type=int, default=6, help="usado so com --prompt (nao --prompt-file).")
    parser.add_argument("--seconds-per-chunk", type=float, default=5.0)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-root", default=DEFAULT_GEMMA)
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234, help="chunk i usa seed + i.")
    parser.add_argument("--start-image", default=None,
                         help="imagem inicial opcional para o chunk 0 (ex: still de personagem/cenario).")
    parser.add_argument("--start-image-strength", type=float, default=0.8)
    parser.add_argument("--chain-strength", type=float, default=0.8,
                         help="forca do ultimo frame do chunk anterior condicionando o proximo.")
    parser.add_argument("--no-chain", action="store_true",
                         help="desliga o encadeamento (cada chunk gerado isolado -- baseline pra comparar drift).")
    parser.add_argument("--motion-history-frames", type=int, default=0,
                         help="EXPERIMENTAL: em vez de --chain-strength/imagem unica no ultimo frame, "
                              "recorta os ultimos N frames (pixel, nao latente) do chunk anterior e passa "
                              "como --video-head -- o VAE encoda o trecho junto, entao o latente carrega "
                              "direcao/velocidade real em vez de uma pose estatica ambigua. 0 desliga (default, "
                              "comportamento antigo). Sugestao inicial: 9 (~1 frame latente na grade 8k+1).")
    parser.add_argument("--motion-history-strength", type=float, default=0.8,
                         help="strength do --video-head, mesma escala de --chain-strength.")
    parser.add_argument("--end-keyframe", default=None,
                         help="imagem-alvo fixada no ultimo frame de TODO chunk (destino, nao ancora -- "
                              "ver comentario em render_scenes.py: da ao clipe pra onde ir em vez de deixar "
                              "derivar). Opcional, deixe vazio se nao tiver still de cenario/personagem.")
    parser.add_argument("--end-keyframe-strength", type=float, default=0.5)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    clips_dir = run_dir / "clips"
    frames_dir = run_dir / "frames"
    final_dir = run_dir / "final"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[continuous_chain] {msg}", flush=True)

    beats = load_beats(args)
    n_speech = sum(1 for b in beats if b.get("speech_text"))
    log(f"{len(beats)} chunk(s), {n_speech} com fala, {args.width}x{args.height}@{args.fps}fps.")
    if n_speech:
        synthesize_beat_speech(
            beats, language=args.speech_language, emotion=args.speech_emotion,
            tts_engine=args.tts_engine, work_dir=run_dir, log=log,
        )

    manifest = {"chunks": [], "width": args.width, "height": args.height, "fps": args.fps}
    chunk_paths: list[str] = []
    frame_offsets: list[int] = []  # fronteiras cumulativas, pra --cuts do video_doctor
    chain_image = args.start_image
    chain_strength = args.start_image_strength if args.start_image else args.chain_strength
    video_head = None  # so existe a partir do chunk 1 -- chunk 0 nao tem clipe anterior pra recortar
    cumulative = 0
    use_motion_history = args.motion_history_frames > 0

    for i, beat in enumerate(beats):
        prompt = beat["prompt"]
        audio_path = beat.get("audio_path")
        # Fala de verdade REGE a duracao do chunk (a fala nao encaixa em
        # --seconds-per-chunk arbitrario); sem fala, cai no valor uniforme de
        # sempre. Mesma dependencia do CLAUDE.md: emocao->duracao da
        # fala->duracao do plano.
        seconds = beat.get("duration_sec") or args.seconds_per_chunk
        num_frames = normalize_ltx_frames(seconds * args.fps)
        clip_path = clips_dir / f"chunk_{i:03d}.mp4"
        frame_path = frames_dir / f"chunk_{i:03d}_last.png"
        tail_path = clips_dir / f"chunk_{i:03d}_tail.mp4"
        seed = args.seed + i

        if clip_path.exists():
            log(f"chunk {i}: {clip_path.name} ja existe, pulando geracao.")
        else:
            audio_kind = "fala" if audio_path else ("ambiente" if args.ambient_audio else "mudo")
            cond_kind = "video-head" if (video_head and not args.no_chain) else \
                ("cadeia" if (chain_image and not args.no_chain) else "nenhuma")
            log(f"chunk {i}/{len(beats) - 1}: \"{prompt[:70]}\" ({num_frames} frames, audio={audio_kind}, "
                f"imagem={cond_kind})")
            ok = render_chunk(
                prompt=prompt, output_path=clip_path,
                checkpoint=args.checkpoint, gemma_root=args.gemma_root, upsampler=args.upsampler,
                width=args.width, height=args.height, fps=args.fps,
                num_frames=num_frames, steps=args.steps, seed=seed,
                chain_image=None if args.no_chain else chain_image,
                chain_strength=chain_strength,
                end_image=args.end_keyframe, end_image_strength=args.end_keyframe_strength,
                negative_prompt=args.negative_prompt, log=log,
                audio_input_path=audio_path, ambient_audio=args.ambient_audio,
                video_head=None if args.no_chain else video_head,
                video_head_frames=args.motion_history_frames,
                video_head_strength=args.motion_history_strength,
            )
            if not ok:
                log(f"chunk {i}: falhou, interrompendo (chunks anteriores ficam em {clips_dir}).")
                return 1
            log(f"chunk {i}: ok -> {clip_path}")

        if not args.no_chain:
            if use_motion_history:
                if not extract_tail_clip(str(clip_path), args.motion_history_frames, tail_path):
                    log(f"chunk {i}: nao consegui recortar o trecho final -- proximo chunk perde a memoria de movimento.")
                    video_head = None
                else:
                    video_head = str(tail_path)
            else:
                if not extract_last_frame(str(clip_path), frame_path):
                    log(f"chunk {i}: nao consegui extrair o ultimo frame -- proximo chunk perde a cadeia.")
                    chain_image = None
                else:
                    chain_image = str(frame_path)
                chain_strength = args.chain_strength

        cumulative += num_frames - 1 if i > 0 else num_frames  # 1o frame do chunk N+1 == ultimo do N
        frame_offsets.append(cumulative)
        chunk_paths.append(str(clip_path))
        manifest["chunks"].append({
            "index": i, "prompt": prompt, "seed": seed, "path": str(clip_path),
            "speech_text": beat.get("speech_text"), "audio_path": audio_path,
            "num_frames": num_frames, "cumulative_end_frame": cumulative,
        })

    manifest_path = run_dir / "chunks.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"manifesto -> {manifest_path}")

    from script_pipeline.assemble_final import concat_videos

    output_path = final_dir / "continuous.mp4"
    work_dir = run_dir / "intermediate"
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"concatenando {len(chunk_paths)} chunk(s) -> {output_path}")
    ok = concat_videos(chunk_paths, output_path, work_dir=work_dir, log=log)
    if not ok:
        log("concat falhou -- veja o log acima.")
        return 1

    # Fronteiras internas (exclui a ultima, que e o fim do video -- video_doctor
    # so quer onde COMECA cada plano, e o comeco do chunk 0 e implicito).
    cuts = ",".join(str(f) for f in frame_offsets[:-1])
    log(f"pronto -> {output_path}")
    if cuts:
        log("As fronteiras entre chunks vao aparecer como pico de mudanca -- "
            "isso NAO e defeito (MEMORIAL.md, corte nao e defeito). Rode:")
        log(f'  .venv\\Scripts\\python.exe video_doctor.py analyze "{output_path}" '
            f'--cuts {cuts} --previews {run_dir}\\tiras --plan {run_dir}\\plan.json')
        log("Revise as tiras (movimento rapido legitimo tambem acende o detector) antes de:")
        log(f'  .venv\\Scripts\\python.exe video_doctor.py repair --plan {run_dir}\\plan.json '
            f'-o "{final_dir}\\continuous_repaired.mp4"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
