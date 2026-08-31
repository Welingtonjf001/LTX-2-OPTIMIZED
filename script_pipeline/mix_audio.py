"""Stage [7]: ambient bed + reverb/spatialization, always AFTER lip-sync -- same
ordering discipline audio_fx.py already documents (reverb would smear the onsets
lip-sync keyed off of if applied first).

Unlike the music-video pipeline, each clip's dialogue audio is already embedded in
its video track by this point (render_scenes.py baked it in via --audio-input-path;
lipsync_scenes.py preserved/replaced it). So "mixing" here means: extract each clip's
audio track, optionally reverb it (audio_fx.apply_room_reverb, reused as-is), optionally
duck a shared ambient bed underneath, remux back. With neither option requested this
stage is a clean passthrough -- no re-encoding, no quality loss.

CLI: python -m script_pipeline.mix_audio --run-dir DIR
     [--room-preset none] [--room-distance 0.0] [--ambient-track PATH] [--ambient-volume 0.25]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_synced_clips(run_dir: Path) -> list[dict]:
    path = run_dir / "lipsync" / "synced_clips.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run lipsync_scenes first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _ffmpeg() -> str:
    return os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")


def _has_audio_stream(video_path: str) -> bool:
    ffprobe = os.environ.get("LTX_FFPROBE", str(Path(_ffmpeg()).with_name("ffprobe.exe")))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool(result.stdout.strip())


def _channels(path: str) -> int:
    """Numero de canais da primeira faixa de audio; 0 se nao houver."""
    ffprobe = os.environ.get("LTX_FFPROBE", str(Path(_ffmpeg()).with_name("ffprobe.exe")))
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return int((r.stdout or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def _para_estereo(path: str) -> str:
    """Trecho de filtro que leva a faixa a estereo 48k SEM mudar o nivel.

    `aformat=channel_layouts=stereo` num sinal MONO tira 3,01 dB: a matriz de
    rematrixagem preserva potencia total, entao espalhar um canal em dois da
    1/raiz(2) em cada. Correto para upmix de ambiencia, ERRADO para dialogo --
    a fala sairia mais baixa do que entrou. O `pan` copia com ganho 1."""
    if _channels(path) == 1:
        return "aresample=48000,pan=stereo|c0=c0|c1=c0"
    return "aresample=48000,aformat=channel_layouts=stereo"


def restore_bed(
    dialogue_video: str, bed_video: str, out_wav: Path, *,
    bed_volume: float, duck_ratio: float, duck_threshold: float, log,
) -> str | None:
    """Devolve a trilha do clipe CRU por baixo da fala, abaixando-a sob ela.

    `dialogue_video` e a saida do lip-sync (so a voz); `bed_video` e o clipe
    como o LTX o gerou (a trilha). O resultado e um wav com os dois.

    O ducking e por SIDECHAIN, com a propria fala como chave: a trilha cai
    quando alguem fala e sobe de volta no silencio, sozinha, sem ninguem marcar
    onde estao as falas. Um `amix` puro nao serve -- somaria as duas em nivel
    fixo e a fala ficaria disputando com a trilha o tempo todo.

    Falha devolvendo None: sem a cama o clipe fica como estava, que e o
    comportamento de antes, nunca um clipe quebrado."""
    ffmpeg = _ffmpeg()
    if not _has_audio_stream(bed_video):
        return None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # `apad` na cama e `duration=first`: os dois arquivos sao o MESMO plano e
    # diferem por milissegundos (5,330 contra 5,333 medidos), mas se a cama
    # acabar um quadro antes a trilha morre no fim de cada plano -- que e o
    # mesmo defeito, so que menor.
    filtro = (
        f"[0:a]{_para_estereo(dialogue_video)},asplit=2[fala][chave];"
        f"[1:a]{_para_estereo(bed_video)},volume={bed_volume},"
        "apad[cama];"
        f"[cama][chave]sidechaincompress=threshold={duck_threshold}:ratio={duck_ratio}:"
        "attack=5:release=250:makeup=1[abaixada];"
        "[fala][abaixada]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    r = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", dialogue_video, "-i", bed_video,
         "-filter_complex", filtro, "-map", "[aout]", str(out_wav)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0 or not out_wav.exists():
        log(f"{Path(dialogue_video).name}: falha ao restaurar a trilha "
            f"({(r.stderr or '')[-300:]}); seguindo so com a fala.")
        return None
    return str(out_wav)


def process_clip(
    video_path: str, out_path: Path, *, work_dir: Path, room_preset: str, room_distance: float,
    ambient_track: str | None, ambient_volume: float, log,
    bed_source: str | None = None, bed_volume: float = 0.8,
    duck_ratio: float = 6.0, duck_threshold: float = 0.03,
) -> str:
    """Return the path to use downstream (either the original, untouched, or a new
    remuxed file with reverb/ambient applied).

    `bed_source`: clipe CRU de onde recuperar a trilha que o lip-sync descartou.
    Ver restore_bed."""
    if room_preset == "none" and not ambient_track and not bed_source:
        return video_path
    if not _has_audio_stream(video_path):
        log(f"{Path(video_path).name}: sem faixa de audio; nada a processar.")
        return video_path

    ffmpeg = _ffmpeg()
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    extracted = work_dir / f"{stem}_extracted.wav"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", video_path, "-vn", "-ac", "2", "-ar", "48000", str(extracted)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if not extracted.exists():
        log(f"{Path(video_path).name}: falha ao extrair audio; mantendo clipe original.")
        return video_path

    audio_source = str(extracted)

    # PRIMEIRO a cama, depois o resto. O reverb e o ambiente devem valer para a
    # mistura inteira -- aplicar so na voz e depois somar a trilha seca poria os
    # dois em espacos acusticos diferentes no mesmo plano.
    if bed_source:
        com_cama = restore_bed(
            video_path, bed_source, work_dir / f"{stem}_com_cama.wav",
            bed_volume=bed_volume, duck_ratio=duck_ratio,
            duck_threshold=duck_threshold, log=log,
        )
        if com_cama:
            audio_source = com_cama
            log(f"{Path(video_path).name}: trilha do clipe restaurada sob a fala")

    if room_preset != "none":
        from audio_fx import apply_room_reverb
        reverbed = work_dir / f"{stem}_reverb.wav"
        result = apply_room_reverb(
            audio_source, str(reverbed), preset=room_preset, distance=room_distance, log=log,
        )
        if result and Path(result).exists():
            audio_source = str(result)

    if ambient_track:
        mixed = work_dir / f"{stem}_ambient_mix.wav"
        filter_complex = f"[1:a]volume={ambient_volume}[amb];[0:a][amb]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        mix_result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", audio_source, "-stream_loop", "-1", "-i", ambient_track,
             "-filter_complex", filter_complex, "-map", "[aout]", str(mixed)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if mix_result.returncode == 0 and mixed.exists():
            audio_source = str(mixed)
        else:
            log(f"{Path(video_path).name}: falha ao misturar ambiente ({mix_result.stderr[-500:]}); seguindo sem ambiente.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    remux = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", video_path, "-i", audio_source,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
         "-shortest", str(out_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if remux.returncode != 0 or not out_path.exists():
        log(f"{Path(video_path).name}: falha no remux final ({remux.stderr[-500:]}); mantendo clipe original.")
        return video_path
    log(f"{Path(video_path).name}: audio processado -> {out_path}")
    return str(out_path)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--room-preset", default="none")
    parser.add_argument("--room-distance", type=float, default=0.0)
    parser.add_argument("--ambient-track", default=None)
    parser.add_argument("--ambient-volume", type=float, default=0.25)
    parser.add_argument("--no-restore-bed", action="store_true",
                        help="nao devolver a trilha do clipe cru sob a fala. Sem "
                             "isto o lip-sync deixa o plano de dialogo SO com a "
                             "voz, e a trilha corta em cada fala.")
    parser.add_argument("--bed-volume", type=float, default=0.8,
                        help="trim estatico da trilha antes do ducking (0,8 = -2 dB)")
    parser.add_argument("--duck-ratio", type=float, default=6.0,
                        help="quanto a trilha cede sob a fala; maior = mais funda")
    parser.add_argument("--duck-threshold", type=float, default=0.03,
                        help="a partir de que nivel de fala o ducking age (0-1)")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    clips = _load_synced_clips(run_dir)
    work_dir = run_folder.subdir(run_dir, "intermediate")

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731
    log(f"mix_audio: room_preset={args.room_preset}, "
        f"ambient={'sim' if args.ambient_track else 'nao'}, "
        f"trilha_sob_a_fala={'nao' if args.no_restore_bed else 'sim'}")

    mixed_manifest = []
    for clip in clips:
        video_path = clip.get("final_video_path")
        if not video_path:
            mixed_manifest.append({**clip, "mixed_video_path": None})
            continue
        out_path = work_dir / f"{clip['id']}_mixed.mp4"
        # So faz sentido recuperar a cama de quem PERDEU a cama: o clipe que nao
        # passou pelo lip-sync ainda tem a trilha original na propria faixa, e
        # somar a ela mesma so criaria eco.
        cama = None
        if not args.no_restore_bed and clip.get("lipsync_applied"):
            bruto = clip.get("video_path")
            if bruto and bruto != video_path and Path(bruto).exists():
                cama = bruto
        final_path = process_clip(
            video_path, out_path, work_dir=work_dir / "_audio_scratch",
            room_preset=args.room_preset, room_distance=args.room_distance,
            ambient_track=args.ambient_track, ambient_volume=args.ambient_volume, log=log,
            bed_source=cama, bed_volume=args.bed_volume,
            duck_ratio=args.duck_ratio, duck_threshold=args.duck_threshold,
        )
        mixed_manifest.append({**clip, "mixed_video_path": final_path})

    (work_dir / "mixed_clips.json").write_text(
        json.dumps(mixed_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    ok_count = sum(1 for c in mixed_manifest if c.get("mixed_video_path"))
    log(f"mix_audio: {ok_count}/{len(clips)} clipe(s) prontos para montagem.")

    if ok_count == len(clips):
        run_folder.mark_stage_complete(run_dir, "mix")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
