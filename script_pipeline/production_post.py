"""Editorial conform and genuinely separate audio stems from explicit source assets.

Generated movie audio is never mislabelled as isolated Foley or sound effects.
Unassigned sound events remain visible in the delivery report.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from script_pipeline.production_project import read, write, project_for, sync_media, export_otio


def _pair_by_id(plan, edit):
    """(shot, entry) casados por `id`, nao por posicao.

    `plan['shots']` e `edit['shots']` sao arquivos separados; se o plano for reconformado
    (planos adicionados/removidos/reordenados) depois de a timeline existir, `zip()` casa
    o plano N com a timeline N por POSICAO e cruza planos errados em silencio -- ninguem
    audita essa correspondencia (`validate_timeline` so confere a timeline consigo mesma)."""
    por_id = {s['id']: s for s in plan['shots']}
    pares = []
    for entry in edit['shots']:
        shot = por_id.get(entry['id'])
        if shot is None:
            raise ValueError(f"Plano da timeline sem correspondente no shot_plan: {entry['id']}")
        pares.append((shot, entry))
    return pares


def ffmpeg(args):
    result = subprocess.run([os.environ.get('LTX_FFMPEG', 'C:/ffmpeg/bin/ffmpeg.exe'),
                             '-y', '-v', 'error', *map(str, args)], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-2000:])


def set_take(root, shot_id, media, approve=False, trim_in=0):
    """Import external motion/VFX take or select existing media, preserving earlier takes."""
    from script_pipeline.production_project import digest
    import shutil
    root, media = Path(root), Path(media).resolve()
    if not media.is_file() or media.suffix.lower() not in ('.mp4', '.mov', '.mkv'):
        raise ValueError('Informe um vídeo de tomada existente.')
    edit = read(root / 'editorial/timeline.json')
    from script_pipeline.production_project import validate_timeline
    validate_timeline(edit)
    shot = next(s for s in edit['shots'] if s['id'] == shot_id)
    take_id = digest([str(media), media.stat().st_size, media.stat().st_mtime_ns])[:16]
    dest = root / 'renders' / shot_id / (take_id + media.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != media:
        shutil.copy2(media, dest)
    take = {'id': take_id, 'media': str(dest), 'fingerprint': shot['fingerprint']}
    if not any(t['id'] == take_id for t in shot['takes']):
        shot['takes'].append(take)
    if trim_in < 0:
        raise ValueError('Entrada da tomada não pode ser negativa.')
    shot.update(selected_take=take_id, approval='approved' if approve else 'pending', trim_in=int(trim_in), stale=False)
    write(root / 'editorial/timeline.json', edit)
    export_otio(root)


def audio_stems(run):
    import numpy as np
    import wave
    root = project_for(run)
    edit = read(root / 'editorial/timeline.json')
    plan = read(Path(run) / 'parse/shot_plan.json')
    fps, rate = edit['fps'], 48000
    total = round(edit['duration_frames'] / fps * rate)
    cues = read(root / 'audio/cues.json', {})
    cues = {name: list(cues.get(name, [])) for name in ('dialogos', 'ambientes', 'foley', 'efeitos', 'musica')}
    lines = {(d['scene_index'], d['line_index']): d for d in read(Path(run) / 'dialogue/lines.json', [])}
    for shot, entry in _pair_by_id(plan, edit):
        line = lines.get((shot['scene'], shot.get('line_index')))
        if line:
            media, spoken = line.get('audio_path'), None
            window = shot.get('audio_window')
            if window and media and Path(media).is_file():
                # Fala longa dividida em planos: cada plano leva SO o seu recorte.
                from script_pipeline.speech_split import slice_wav
                media, spoken = slice_wav(media, window), float(window[1]) - float(window[0])
            elif media and Path(media).is_file():
                # SEM janela (o caso comum -- so fala longa ganha audio_window): mux_audio
                # (render_shots.py) centraliza pela duracao REAL do wav, nao pela duracao do
                # plano. Sem medir aqui tambem, o stem colava a fala no inicio do clipe (lead=0,
                # duration=span) enquanto o video final a centralizava -- os dois dessincronizavam.
                from script_pipeline.render_shots import _audio_seconds
                spoken = _audio_seconds(media)
            span = entry['duration_frames'] / fps
            # mux_audio centraliza a fala dentro do plano; o stem segue o mesmo ponto de entrada.
            lead = max(0.0, (span - spoken) / 2.0) if spoken else 0.0
            cues['dialogos'].append({'media': media, 'start': entry['start_frame'] / fps + lead,
                'duration': spoken or span, 'text': line['text'], 'gain': 1.0})
    report = {}
    out = root / 'audio/stems'
    out.mkdir(parents=True, exist_ok=True)
    for name, events in cues.items():
        mix = np.zeros((total, 2), dtype=np.float32)
        missing, resolved = [], 0
        for cue in events:
            media = cue.get('media')
            if not media or not Path(media).is_file():
                missing.append(cue)
                continue
            result = subprocess.run([os.environ.get('LTX_FFMPEG', 'C:/ffmpeg/bin/ffmpeg.exe'),
                '-v', 'error', '-ss', str(cue.get('source_in', 0)), '-i', media,
                '-t', str(cue['duration']), '-ar', str(rate), '-ac', '2', '-f', 'f32le', '-'], capture_output=True)
            if result.returncode:
                raise RuntimeError(f'Falha ao decodificar {media}')
            audio = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2)
            start = round(float(cue['start']) * rate)
            if start < 0:
                audio, start = audio[-start:], 0
            count = min(len(audio), total - start)
            if count > 0:
                audio = audio[:count].copy() * float(cue.get('gain', 1.0))
                fade = min(count, round(float(cue.get('fade_in', 0)) * rate))
                if fade:
                    audio[:fade] *= np.linspace(0, 1, fade)[:, None]
                fade = min(count, round(float(cue.get('fade_out', 0)) * rate))
                if fade:
                    audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
                mix[start:start + count] += audio
                resolved += 1
        peak = float(np.max(np.abs(mix))) if total else 0
        with wave.open(str(out / f'{name}.wav'), 'wb') as wav:
            wav.setparams((2, 2, rate, total, 'NONE', 'not compressed'))
            wav.writeframes((np.clip(mix, -1, 1) * 32767).astype('<i2').tobytes())
        report[name] = {'assigned_events': resolved, 'missing': missing, 'silent': resolved == 0,
                        'peak_before_clipping': peak, 'frames': total, 'sample_rate': rate}
    write(out / 'report.json', report)
    return report


def conform_movie(run, approved_only=False, master=True):
    """Frame accurate preview; approved delivery fails closed for missing/unapproved takes."""
    root = project_for(run)
    sync_media(run)
    edit = read(root / 'editorial/timeline.json')
    from script_pipeline.production_project import validate_timeline
    validate_timeline(edit)
    from script_pipeline.assemble_final import _load_mixed_clips, _video_duration, concat_videos
    mixed = {c['id']: c for c in _load_mixed_clips(Path(run))}
    plan = read(Path(run) / 'parse/shot_plan.json')
    work = root / 'editorial/conform'
    work.mkdir(parents=True, exist_ok=True)
    clips = []
    for shot, entry in _pair_by_id(plan, edit):
        if approved_only and entry['approval'] != 'approved':
            raise ValueError(f"Plano {entry['id']} ainda não aprovado")
        take = next((t for t in entry['takes'] if t['id'] == entry['selected_take']), None)
        if not take:
            raise ValueError(f"Plano obrigatório sem tomada: {entry['id']}")
        key = f"scene{shot['scene']:02d}_shot{shot['index']:03d}"
        # Pending engine takes use the postprocessed output; explicitly selected approved takes win.
        media = take['media'] if entry['approval'] == 'approved' else mixed.get(key, {}).get('mixed_video_path') or take['media']
        length = entry['duration_frames'] / edit['fps']
        start = entry['trim_in'] / edit['fps']
        duration = _video_duration(media, ffmpeg=os.environ.get('LTX_FFMPEG', 'C:/ffmpeg/bin/ffmpeg.exe'))
        if duration is None or duration + 1 / edit['fps'] < start + length:
            raise ValueError(f"Tomada curta demais para {entry['id']}: {duration}s, exige {start + length}s")
        dest = work / f"{entry['index']:04d}.mp4"
        ffmpeg(['-ss', start, '-i', media, '-t', length, '-r', edit['fps'],
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-c:a', 'aac', dest])
        clips.append(str(dest))
    dest = root / 'entregas' / ('filme_aprovado.mp4' if approved_only else 'previa_editorial.mp4')
    if not concat_videos(clips, dest, work_dir=work, log=print):
        raise RuntimeError('Falha na conformação editorial')
    if master:
        # A entrega tambem sai a -16 LUFS / -1,5 dBTP (Voo 702: master a -12,8 LUFS, pico +3,9 dBFS).
        from script_pipeline.assemble_final import master_audio
        mastered = dest.with_name('_master_' + dest.name)
        if master_audio(dest, mastered, log=print):
            mastered.replace(dest)
    audio_stems(run)
    return str(dest)


def make_rgba_asset(project, name, prompt, *, width=1024, height=576, seed=42, register_location=None):
    """Recorte com ALFA nativo (Qwen-Image-2.1) em `<projeto>/assets/rgba/<nome>.png`.

    Uso tipico: a AERONAVE mestre do Voo 702 (mesmo avião em todos os exteriores) -- registrada como
    referencia da locacao (`biblia/locacoes.json[<loc>].references.front`) para `conform_plan` levar a
    todos os planos daquela locacao. Tambem serve para props/figurantes de composicao (Blender/Natron)."""
    import qwen_image21_backend as q
    project = Path(project)
    out = project / "assets" / "rgba" / f"{name}.png"
    if not q.generate_rgba(prompt, out, width=width, height=height, seed=seed):
        raise RuntimeError(f"Qwen-Image-2.1 nao produziu um recorte RGBA valido para {name!r}")
    if register_location:
        path = project / "biblia" / "locacoes.json"
        locations = read(path, {})
        if register_location not in locations:
            raise ValueError(f"Locacao desconhecida: {register_location}")
        locations[register_location].setdefault("references", {})["front"] = str(out.resolve())
        write(path, locations)
    return str(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Pos-producao do projeto mestre")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rgba = sub.add_parser("rgba-asset", help="gera um recorte RGBA (Qwen-Image-2.1)")
    rgba.add_argument("--project", required=True)
    rgba.add_argument("--name", required=True)
    rgba.add_argument("--prompt", required=True)
    rgba.add_argument("--location", default=None, help="ex.: LOC_SKY: registra como referencia da locacao")
    rgba.add_argument("--width", type=int, default=1024)
    rgba.add_argument("--height", type=int, default=576)
    args = ap.parse_args(argv)
    if args.cmd == "rgba-asset":
        print(make_rgba_asset(args.project, args.name, args.prompt, width=args.width, height=args.height,
                              register_location=args.location))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
