"""Stage [4]: synthesize one dry (unprocessed) WAV per dialogue line via dialogue_tts.py.

Builds one TTS job per (scene, line), carrying the character's cast.json voice
assignment and a delivery hint (the line's parenthetical if present, else the LLM-
inferred "emotion" from parse_screenplay's enrichment -- used only by the qwen engine's
free-text "instruct"). Writes <run>/dialogue/scene_XX_line_YY.wav plus a
<run>/dialogue/lines.json manifest (line -> {audio_path, duration_sec, engine_used})
that render_scenes.py and lipsync_scenes.py both consume.

CLI: python -m script_pipeline.synthesize_dialogue --run-dir DIR [--engine auto]
     [--language pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from script_pipeline.voice_library import archetype_take, pick_take, style_clip_fits

LANGUAGE_MAP = {
    "pt": {"xtts": "pt", "qwen": "Portuguese"},
    "en": {"xtts": "en", "qwen": "English"},
    "es": {"xtts": "es", "qwen": "Spanish"},
    "fr": {"xtts": "fr", "qwen": "French"},
    "de": {"xtts": "de", "qwen": "German"},
    "ja": {"xtts": "ja", "qwen": "Japanese"},
    "ko": {"xtts": "ko", "qwen": "Korean"},
}


def _load_scenes(run_dir: Path) -> list[dict]:
    parse_dir = run_dir / "parse"
    enriched = parse_dir / "scenes_enriched.json"
    structural = parse_dir / "scenes.json"
    path = enriched if enriched.exists() else structural
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cast(run_dir: Path) -> dict:
    cast_path = run_dir / "characters" / "cast.json"
    if not cast_path.exists():
        raise FileNotFoundError(f"{cast_path} not found -- run cast_characters first.")
    return json.loads(cast_path.read_text(encoding="utf-8"))


def build_jobs(scenes: list[dict], cast: dict, *, language: str, dialogue_dir: Path) -> list[dict]:
    lang = LANGUAGE_MAP.get(language, LANGUAGE_MAP["pt"])
    jobs = []
    for scene in scenes:
        for line_index, line in enumerate(scene.get("dialogue", [])):
            character = line["character"]
            voice = cast.get(character, {}).get("voice", {})
            instruct = line.get("parenthetical") or line.get("emotion")
            job_id = f"scene{scene['index']:02d}_line{line_index:02d}"

            # XTTS clones whatever reference clip it is given and has NO textual emotion
            # control -- "instruct" reaches only the qwen engine. So the reference clip IS
            # the emotion. Three sources of acted takes, in order of precedence:
            #   1. archetype_voice  -- a deliberate human choice, so it outranks anything
            #                          auto-assigned; base clip + one acted take
            #   2. emotive_voice    -- 17 takes, matched to the line's emotion by keyword
            #   3. arquivo_estilo   -- one acted take per character, from mapa_vozes.csv
            # With none of them, every line comes out in the same neutral delivery.
            speaker_wav = voice.get("xtts_speaker_wav")
            archetype = voice.get("archetype_voice")
            emotive_voice = voice.get("emotive_voice")
            if archetype:
                take = archetype_take(archetype, instruct)
                if take:
                    speaker_wav = take
            elif emotive_voice:
                take = pick_take(emotive_voice, instruct)
                if take:
                    speaker_wav = take
            elif style_clip_fits(voice.get("xtts_speaker_wav_style"), instruct):
                speaker_wav = voice["xtts_speaker_wav_style"]
            jobs.append({
                "id": job_id,
                "scene_index": scene["index"],
                "line_index": line_index,
                "character": character,
                "text": line["text"],
                "instruct": instruct,
                # `instruct` e texto livre, que so o qwen usa; `emotion` e o
                # SLUG, que e o que a tabela de prosodia do xtts indexa.
                "emotion": line.get("emotion") or instruct,
                "language_code": lang["xtts"],
                "language_name": lang["qwen"],
                "xtts_speaker_wav": speaker_wav,
                "qwen_speaker": voice.get("qwen_speaker"),
                "gender": voice.get("gender"),
                "output_path": str(dialogue_dir / f"{job_id}.wav"),
            })
    return jobs


def _wav_duration_sec(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    # "fish" ficou disponivel em 2026-09-03 (MEMORIAL 3.53), mas o DEFAULT
    # voltou a ser "auto" no mesmo dia (MEMORIAL 3.55): fish precisa do
    # servidor externo ja no ar (START_API.ps1) e ainda e recente/nao
    # amadurecido aqui (bug de encoding CJK so descoberto e corrigido hoje,
    # slot unico de audio de referencia, sem fallback automatico). Pedido
    # explicito do usuario: nao descartar o XTTS, manter as duas opcoes
    # escolhiveis ate o fish provar mais horas de uso. Peca --engine fish
    # (ou o dropdown da UI) quando quiser usa-lo de proposito.
    parser.add_argument("--engine", default="auto", choices=["auto", "xtts", "qwen", "fish"])
    parser.add_argument("--language", default="pt", choices=list(LANGUAGE_MAP))
    args = parser.parse_args(argv)

    from script_pipeline import run_folder
    from script_pipeline.dialogue_tts import synthesize_batch

    run_dir = Path(args.run_dir).resolve()
    scenes = _load_scenes(run_dir)
    cast = _load_cast(run_dir)
    dialogue_dir = run_folder.subdir(run_dir, "dialogue")

    jobs = build_jobs(scenes, cast, language=args.language, dialogue_dir=dialogue_dir)
    if not jobs:
        run_folder.append_log(run_dir, "synthesize_dialogue: nenhuma fala encontrada (roteiro sem dialogo?).")
        run_folder.mark_stage_complete(run_dir, "dialogue")
        (dialogue_dir / "lines.json").write_text("[]", encoding="utf-8")
        return 0

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731
    log(f"synthesize_dialogue: sintetizando {len(jobs)} fala(s), engine={args.engine}, idioma={args.language}")
    results = synthesize_batch(jobs, engine=args.engine, log=log)

    by_id = {r["id"]: r for r in results}
    lines_manifest = []
    failures = 0
    for job in jobs:
        result = by_id.get(job["id"], {})
        ok = bool(result.get("ok"))
        entry = {
            "id": job["id"], "scene_index": job["scene_index"], "line_index": job["line_index"],
            "character": job["character"], "text": job["text"], "ok": ok,
            "audio_path": result.get("output_path") if ok else None,
            "engine_used": result.get("engine_used"),
            "duration_sec": _wav_duration_sec(result["output_path"]) if ok else None,
        }
        lines_manifest.append(entry)
        if not ok:
            failures += 1
            log(f"FALHA: {job['id']} ({job['character']}): {result.get('error')}")

    (dialogue_dir / "lines.json").write_text(
        json.dumps(lines_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log(f"synthesize_dialogue: {len(jobs) - failures}/{len(jobs)} fala(s) sintetizada(s) com sucesso.")

    if failures == 0:
        run_folder.mark_stage_complete(run_dir, "dialogue")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
