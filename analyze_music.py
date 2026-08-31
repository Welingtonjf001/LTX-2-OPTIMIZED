import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--output", default="music_timing.json")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop, units="frames", trim=False
    )
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset, sr=sr, hop_length=hop, units="frames", backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    rms_norm = rms / max(float(np.max(rms)), 1e-9)

    beats = [float(t) for t in beat_times if 0 <= t < duration]
    downbeats = beats[::4]
    onsets = [float(t) for t in onset_times if 0 <= t < duration]
    energy = [
        {"time": float(t), "rms": float(v)}
        for t, v in zip(rms_times[::8], rms_norm[::8], strict=False)
        if t < duration
    ]
    result = {
        "audio": str(Path(args.audio).resolve()),
        "duration_sec": duration,
        "sample_rate": sr,
        "bpm": tempo_value,
        "beats_sec": beats,
        "downbeats_sec": downbeats,
        "onsets_sec": onsets,
        "energy": energy,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"duration_sec": duration, "bpm": tempo_value, "beats": len(beats), "downbeats": len(downbeats), "onsets": len(onsets)}))


if __name__ == "__main__":
    main()
