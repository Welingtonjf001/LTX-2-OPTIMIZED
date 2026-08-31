import argparse
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel
import librosa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--output", default="lyrics_timing.json")
    p.add_argument("--model", default="base")
    p.add_argument("--source-audio", default=None, help="Original mix used for metadata when transcribing a vocal stem")
    args = p.parse_args()
    model = WhisperModel(args.model, device="cpu", compute_type="int8", download_root="./models/whisper")
    segments, info = model.transcribe(args.audio, word_timestamps=True, vad_filter=True)
    rows = []
    for seg in segments:
        rows.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
                for w in (seg.words or [])
            ],
        })
    try:
        duration_sec = float(librosa.get_duration(path=args.audio))
    except Exception:
        duration_sec = None
    result = {
        "audio": os.path.abspath(args.source_audio or args.audio),
        "audio_basename": os.path.basename(args.source_audio or args.audio),
        "duration_sec": duration_sec,
        "language": info.language,
        "segments": rows,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"language={info.language} segments={len(rows)} words={sum(len(s['words']) for s in rows)}")


if __name__ == "__main__":
    main()
