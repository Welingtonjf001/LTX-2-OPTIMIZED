"""Stage [8]: concatenate every clip, in script order, into the final movie.

Same ffmpeg concat-demuxer pattern already established in music_maker_ui_v2/v3.py:
try a fast stream-copy concat first, fall back to a full re-encode if the clips'
codecs/parameters don't match closely enough for stream copy to work.

CLI: python -m script_pipeline.assemble_final --run-dir DIR [--output NAME]
Reads <run>/intermediate/mixed_clips.json (produced by mix_audio.py -- run that stage
even with no reverb/ambient requested, since it's what normalizes "final per-clip
video" into one field regardless of whether audio processing actually ran).
Writes <run>/final/<output>.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_mixed_clips(run_dir: Path) -> list[dict]:
    path = run_dir / "intermediate" / "mixed_clips.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run mix_audio first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_audio_stream(video_path: str, *, ffmpeg: str) -> bool:
    ffprobe = os.environ.get("LTX_FFPROBE", str(Path(ffmpeg).with_name("ffprobe.exe")))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool(result.stdout.strip())


def _video_duration(video_path: str, *, ffmpeg: str) -> float | None:
    """Duracao da faixa de VIDEO (nao do container, que vale a maior das faixas)."""
    ffprobe = os.environ.get("LTX_FFPROBE", str(Path(ffmpeg).with_name("ffprobe.exe")))
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=duration", "-of", "csv=p=0", video_path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


# Every clip fed to the concat demuxer must share ONE audio format.
CONCAT_AUDIO_RATE = "48000"
CONCAT_AUDIO_CHANNELS = "2"
CONCAT_FPS = "24"            # o fps do LTX; o lip-sync devolve 25
CONCAT_TIMESCALE = "12288"   # a timescale que o ffmpeg ja da aos clipes de 24 fps


def _normalize_audio_for_concat(video_path: str, work_dir: Path, *, ffmpeg: str, log,
                                 indice: int = 0) -> str:
    """Return a copy of this clip whose audio is AAC/48kHz/stereo, inventing a silent
    track when the clip has none. Video is stream-copied (only audio is re-encoded).

    MEASURED (2026-08-10), in two rounds -- the second because the first fix looked
    right and wasn't:
      1. The concat demuxer takes its output stream layout from the FIRST input. A
         film opening on a wordless action shot (video-only, no TTS audio to mux) came
         out with no audio track at all.
      2. Adding a silent track to only those clips was NOT enough: matching the
         *presence* of a stream doesn't matter, matching its *parameters* does. LTX
         writes dialogue audio as PCM at 16 kHz; the silence generated here was AAC at
         48 kHz. With -c copy the demuxer kept the first file's parameters and dropped
         the rest, so the film had a proper-looking AAC track measuring -91 dB --
         digital silence end to end. Verified with ffmpeg volumedetect, which is the
         check that catches this and a stream listing is not.
    Hence: normalise EVERY clip to identical audio parameters, not just the silent ones.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # BUGFIX auditoria 2026-09-16 (A20): nomear so pelo stem colide quando dois
    # clipes de PASTAS diferentes compartilham o basename (ex.: a/clip.mp4 e
    # b/clip.mp4, ou o padrao "shotNNN.mp4" repetido entre cenas com paths
    # relativos distintos) -- a segunda normalizacao sobrescreve a primeira, e
    # a lista final do concat aponta duas vezes para o MESMO arquivo. O indice
    # de posicao na lista de entrada garante nome unico independente do path.
    out_path = work_dir / f"{indice:04d}_{Path(video_path).stem}_norm.mp4"
    # MEDIDO 2026-09-13 ("O Primeiro Tour", 43 planos): depois do mix o video de cada
    # plano de fala saia 0,1-0,16 s mais curto que o audio, e o concat demuxer
    # empilha as duas faixas SEPARADAMENTE -- o filme terminou com 146,7 s de video
    # contra 141,0 s de audio, a fala escorregando a cada corte. O audio de cada
    # clipe agora tem EXATAMENTE a duracao do video (completa com silencio ou corta)
    # antes de entrar no concat.
    #
    # E a causa MAIOR, medida em seguida: com o audio ja igual ao video em cada clipe
    # (soma 138,53 s nos dois), o filme ainda saia com 144,3 s de video. Os planos de
    # fala voltam do lip-sync a 25 fps (timebase 1/12800) e os de acao ficam a 24 fps
    # (1/12288); o concat com -c copy mistura as bases de tempo e estica o video
    # ~5,75 s. Todo clipe e reencodado a 24 fps com a MESMA timescale antes do concat.
    dur_video = _video_duration(video_path, ffmpeg=ffmpeg)
    ajuste = ["-af", f"apad=whole_dur={dur_video:.6f},atrim=0:{dur_video:.6f}"] if dur_video else []
    video_uniforme = ["-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
                      "-r", CONCAT_FPS, "-video_track_timescale", CONCAT_TIMESCALE]
    if _has_audio_stream(video_path, ffmpeg=ffmpeg):
        command = [
            ffmpeg, "-y", "-v", "error", "-i", video_path,
            "-map", "0:v:0", "-map", "0:a:0", *video_uniforme, *ajuste,
            "-c:a", "aac", "-b:a", "192k", "-ar", CONCAT_AUDIO_RATE, "-ac", CONCAT_AUDIO_CHANNELS,
            str(out_path),
        ]
    else:
        command = [
            ffmpeg, "-y", "-v", "error", "-i", video_path,
            "-f", "lavfi", "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={CONCAT_AUDIO_RATE}",
            "-map", "0:v:0", "-map", "1:a:0", *video_uniforme,
            "-c:a", "aac", "-b:a", "192k", "-ar", CONCAT_AUDIO_RATE, "-ac", CONCAT_AUDIO_CHANNELS,
            "-shortest", str(out_path),
        ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not out_path.exists():
        log(f"{Path(video_path).name}: falha ao normalizar audio ({result.stderr[-300:]}); usando original.")
        return video_path
    return str(out_path)


def concat_videos(video_paths: list[str], output_path: Path, *, work_dir: Path, log) -> bool:
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    work_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = work_dir / "_audio_normalized"
    n_silent = sum(1 for p in video_paths if not _has_audio_stream(p, ffmpeg=ffmpeg))
    video_paths = [_normalize_audio_for_concat(p, norm_dir, ffmpeg=ffmpeg, log=log, indice=idx)
                  for idx, p in enumerate(video_paths)]
    log(f"assemble_final: audio normalizado para AAC {CONCAT_AUDIO_RATE}Hz estereo em "
        f"{len(video_paths)} clipe(s) ({n_silent} sem audio receberam silencio). "
        "Sem isso o concat descarta o audio dos clipes que divergem do primeiro.")

    # BUGFIX auditoria 2026-09-16 (A09): `encoding="ascii"` derruba com
    # UnicodeEncodeError em qualquer pasta/nome com acento (ex.: "João",
    # "ação") -- MEDIDO: reproduzido antes mesmo de chamar o ffmpeg. E os
    # apostrofos (nomes com "'") nao eram escapados, o que quebra a sintaxe do
    # ffconcat (cada `'` fecha a string do caminho no meio do nome). UTF-8 sem
    # BOM e o escape padrao do formato (`'` -> `'\''`) resolvem os dois.
    concat_list = work_dir / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8", newline="\n") as handle:
        for path in video_paths:
            safe_path = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(output_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not output_path.exists():
        log("Concat direto (stream copy) falhou; reencodificando H.264/AAC.")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "256k", str(output_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    if result.returncode != 0 or not output_path.exists():
        log(f"Concat falhou: {result.stderr[-2000:]}")
        return False
    return True


def _pick_music_track(music_path: str | None, music_dir: str | None, *, log) -> str | None:
    """Resolve qual arquivo de musica usar. `music_path` explicito sempre
    ganha; `music_dir` sorteia um arquivo de audio da pasta -- pedido do
    usuario 2026-09-12: os motores de video (LTX/MiniMax) so geram trilha
    para ALGUMAS cenas (as de dialogo, via audio_conditioning), entao o
    filme monta com trechos sem musica nenhuma. Uma trilha externa continua,
    escolhida aqui, cobre o filme inteiro independente do que cada motor
    gerou."""
    if music_path:
        return music_path
    if music_dir:
        import random

        extensoes = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
        candidatos = sorted(p for p in Path(music_dir).glob("*") if p.suffix.lower() in extensoes)
        if not candidatos:
            log(f"assemble_final: nenhum arquivo de audio em {music_dir}; seguindo sem musica.")
            return None
        escolha = random.choice(candidatos)
        log(f"assemble_final: musica sorteada de {music_dir}: {escolha.name}")
        return str(escolha)
    return None


def mix_music_bed(
    movie_path: Path, music_path: str, out_path: Path, *,
    music_volume: float, duck_ratio: float, duck_threshold: float, log,
) -> str | None:
    """Mistura uma trilha de musica CONTINUA sob o FILME JA MONTADO -- uma
    passada so, depois da concatenacao, nao por clipe. Por-clipe (o
    `--ambient-track` de mix_audio.py) reinicia o loop da musica a cada
    corte de plano, o que soa como a musica "parando e recomecando" bem no
    meio do filme -- exatamente o que o usuario reportou. Aqui a mesma
    trilha toca sem interrupcao do inicio ao fim.

    Ducking por SIDECHAIN contra a faixa de audio do PROPRIO filme (fala +
    o que cada motor gerou) -- mesmo mecanismo de `mix_audio.restore_bed`:
    a musica abaixa sozinha quando ha fala/som e volta no silencio, sem
    precisar marcar onde estao as falas."""
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    filtro = (
        "[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[voz][chave];"
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={music_volume}[musica];"
        f"[musica][chave]sidechaincompress=threshold={duck_threshold}:ratio={duck_ratio}:"
        "attack=5:release=400:makeup=1[abaixada];"
        "[voz][abaixada]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    r = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(movie_path), "-stream_loop", "-1", "-i", str(music_path),
         "-filter_complex", filtro, "-map", "0:v:0", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0 or not out_path.exists():
        log(f"assemble_final: falha ao misturar musica ({(r.stderr or '')[-500:]}); seguindo sem musica.")
        return None
    return str(out_path)


def _ffmpeg_bin() -> str:
    return os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")


def measure_loudness(path, *, target_lufs: float = -16.0, true_peak: float = -1.5,
                     lra: float = 7.0) -> dict | None:
    """Primeira passada do loudnorm: LUFS integrado, true peak, LRA e limiar.

    Devolve o dict do ffmpeg (chaves `input_i`, `input_tp`, `input_lra`,
    `input_thresh`, `target_offset`) ou None se nao conseguiu medir."""
    import json as _json
    import re as _re
    r = subprocess.run(
        [_ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path), "-vn",
         "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = _re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, flags=_re.S)
    if not match:
        return None
    try:
        return _json.loads(match.group(0))
    except ValueError:
        return None


def master_audio(movie_path: Path, out_path: Path, *, target_lufs: float = -16.0,
                 true_peak: float = -1.5, lra: float = 7.0, log) -> dict | None:
    """Masteriza a faixa do FILME montado: loudnorm em DUAS passadas (linear,
    sem bombeamento) + limiter de segurança, video copiado.

    Avaliacao do Voo 702 (2026-09-18): master a -12,8 LUFS com pico de
    +3,9 dBFS -- distorce no transcode. Alvo web: -16 LUFS / -1,5 dBTP.
    Devolve {"antes": ..., "depois": ...} (medidas) ou None em falha."""
    antes = measure_loudness(movie_path, target_lufs=target_lufs, true_peak=true_peak, lra=lra)
    if not antes:
        log("assemble_final: nao consegui medir a sonoridade; master de audio ignorado.")
        return None
    # Margem sobre o alvo de TP: o limiter de amostra nao ve o pico entre amostras.
    limite = 10 ** ((true_peak - 0.6) / 20.0)
    filtro = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:"
        f"measured_I={antes['input_i']}:measured_TP={antes['input_tp']}:"
        f"measured_LRA={antes['input_lra']}:measured_thresh={antes['input_thresh']}:"
        f"offset={antes['target_offset']}:linear=true,"
        f"alimiter=limit={limite:.4f}:attack=5:release=60:level=disabled"
    )
    r = subprocess.run(
        [_ffmpeg_bin(), "-y", "-v", "error", "-i", str(movie_path), "-map", "0:v:0", "-map", "0:a:0",
         "-c:v", "copy", "-af", filtro, "-ar", "48000", "-c:a", "aac", "-b:a", "256k", str(out_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out_path.exists():
        log(f"assemble_final: falha no master de audio ({(r.stderr or '')[-400:]}); mantendo o audio original.")
        return None
    depois = measure_loudness(out_path, target_lufs=target_lufs, true_peak=true_peak, lra=lra) or {}
    log(f"assemble_final: master de audio {antes['input_i']} LUFS / {antes['input_tp']} dBTP "
        f"-> {depois.get('input_i', '?')} LUFS / {depois.get('input_tp', '?')} dBTP "
        f"(alvo {target_lufs} / {true_peak})")
    try:
        if float(depois.get("input_tp", -99)) > true_peak + 0.5:
            log("assemble_final: AVISO -- true peak final ainda acima do alvo.")
    except (TypeError, ValueError):
        pass
    return {"antes": antes, "depois": depois}


def add_room_tone(movie_path: Path, out_path: Path, *, level_db: float, log) -> str | None:
    """Ambiencia continua (ruido marrom filtrado) sob o filme inteiro, para as
    emendas de audio entre planos nao aparecerem como degraus de silencio.

    Nao e J/L cut: um J/L de verdade precisa de audio ALEM das bordas do plano
    (margem/handle), que o pipeline ainda nao gera -- ver MEMORIAL 3.94."""
    filtro = (f"[1:a]lowpass=f=500,volume={level_db}dB[rt];"
              "[0:a][rt]amix=inputs=2:duration=first:normalize=0[aout]")
    r = subprocess.run(
        [_ffmpeg_bin(), "-y", "-v", "error", "-i", str(movie_path), "-f", "lavfi", "-i",
         "anoisesrc=color=brown:sample_rate=48000:amplitude=0.5",
         "-filter_complex", filtro, "-map", "0:v:0", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out_path.exists():
        log(f"assemble_final: falha ao adicionar room tone ({(r.stderr or '')[-300:]}); seguindo sem.")
        return None
    return str(out_path)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="movie.mp4")
    parser.add_argument("--music-path", default=None,
                        help="arquivo de musica especifico para tocar sob o filme inteiro")
    parser.add_argument("--music-dir", default=None,
                        help="pasta de musicas -- sorteia um arquivo se --music-path nao for dado")
    parser.add_argument("--music-volume", type=float, default=0.18,
                        help="nivel da musica antes do ducking (0-1)")
    parser.add_argument("--music-duck-ratio", type=float, default=8.0)
    parser.add_argument("--music-duck-threshold", type=float, default=0.02)
    parser.add_argument("--no-master", action="store_true",
                        help="nao normaliza o audio final (padrao: -16 LUFS / -1,5 dBTP)")
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--true-peak", type=float, default=-1.5)
    parser.add_argument("--room-tone-db", type=float, default=None,
                        help="ambiencia continua sob o filme inteiro (ex.: -46 dB); desligado por padrao")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    clips = _load_mixed_clips(run_dir)
    final_dir = run_folder.subdir(run_dir, "final")
    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731

    video_paths = [c["mixed_video_path"] for c in clips if c.get("mixed_video_path")]
    missing = [c["id"] for c in clips if not c.get("mixed_video_path")]
    if missing:
        log(f"assemble_final: {len(missing)} clipe(s) sem video final, nao incluidos: {missing}")
    if not video_paths:
        log("assemble_final: nenhum clipe disponivel para montar.")
        return 1

    output_path = final_dir / args.output
    log(f"assemble_final: concatenando {len(video_paths)} clipe(s) em ordem de roteiro -> {output_path}")
    ok = concat_videos(video_paths, output_path, work_dir=run_folder.subdir(run_dir, "intermediate"), log=log)

    if not ok:
        log("assemble_final: FALHOU.")
        return 1

    musica = _pick_music_track(args.music_path, args.music_dir, log=log)
    if musica:
        com_musica = final_dir / f"_com_musica_{output_path.name}"
        resultado = mix_music_bed(
            output_path, musica, com_musica, music_volume=args.music_volume,
            duck_ratio=args.music_duck_ratio, duck_threshold=args.music_duck_threshold, log=log,
        )
        if resultado:
            com_musica.replace(output_path)
            log(f"assemble_final: musica ({Path(musica).name}) misturada sob o filme inteiro.")

    if args.room_tone_db is not None:
        com_tone = final_dir / f"_com_roomtone_{output_path.name}"
        if add_room_tone(output_path, com_tone, level_db=args.room_tone_db, log=log):
            com_tone.replace(output_path)
            log(f"assemble_final: room tone continuo a {args.room_tone_db} dB sob o filme inteiro.")
    if not args.no_master:
        mastered = final_dir / f"_master_{output_path.name}"
        medidas = master_audio(output_path, mastered, target_lufs=args.target_lufs,
                               true_peak=args.true_peak, log=log)
        if medidas:
            mastered.replace(output_path)
            (final_dir / "master_audio.json").write_text(
                json.dumps(medidas, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"assemble_final: filme final pronto -> {output_path}")
    run_folder.mark_stage_complete(run_dir, "assemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
