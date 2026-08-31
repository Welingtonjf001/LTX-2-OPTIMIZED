"""Shared vocal spatialization (room/environment reverb) for the talking-head
pipeline, used by music_maker_ui_v2.py / v3.py / GGUF / TensorRT variants.

Design constraint that decides WHERE this must be called from: Wav2Lip drives
mouth movement off a mel-spectrogram of the vocal audio, so it needs the DRY
(unprocessed) vocal stem. Reverb tails and pre-delay smear the onsets Wav2Lip
keys off and desync the mouth. So this module is applied AFTER Wav2Lip has
already produced the lip-synced clip, on the isolated vocal stem, which is
then remixed with the (untouched) instrumental stem to build the final master
audio track used at remux time -- never before Wav2Lip.

Two backends:
  - pedalboard (Spotify, MIT) if installed: algorithmic Freeverb-style room
    model with per-preset room_size/damping/wet/dry, plus a distance-driven
    lowpass + gain stage. Preferred: precise, no extra process spawn.
  - ffmpeg fallback (always available in this project -- every pipeline
    already shells out to LTX_FFMPEG): a multi-tap aecho approximation of
    early reflections plus a matching lowpass/volume distance stage.

Both backends preserve the exact input duration (no reverb tail appended
past the buffer), which matters because Wav2Lip/concat downstream expects
frame-accurate audio elsewhere in this project.
"""
import os
import subprocess

try:
    from pedalboard import Pedalboard, Reverb, LowpassFilter, Gain
    from pedalboard.io import AudioFile
    _HAS_PEDALBOARD = True
except Exception:
    _HAS_PEDALBOARD = False

# room_size/damping/width follow pedalboard.Reverb's Freeverb-style model
# (0..1 each). wet_level/dry_level are also 0..1 mix levels, not dB.
ROOM_PRESETS = {
    "none":          None,
    "estudio_seco":  dict(room_size=0.05, damping=0.5, wet_level=0.03, dry_level=0.97, width=0.5),
    "quarto_intimo": dict(room_size=0.25, damping=0.5, wet_level=0.12, dry_level=0.85, width=0.6),
    "sala_estar":    dict(room_size=0.40, damping=0.4, wet_level=0.20, dry_level=0.80, width=0.7),
    "salao_grande":  dict(room_size=0.85, damping=0.2, wet_level=0.35, dry_level=0.70, width=1.0),
    "catedral":      dict(room_size=1.00, damping=0.1, wet_level=0.45, dry_level=0.60, width=1.0),
    "ar_livre":      dict(room_size=0.15, damping=0.9, wet_level=0.05, dry_level=0.95, width=0.3),
}

ROOM_PRESET_LABELS = {
    "none": "Sem processamento (voz seca)",
    "estudio_seco": "Estudio seco (quase sem reflexao)",
    "quarto_intimo": "Quarto intimo / close-up",
    "sala_estar": "Sala de estar",
    "salao_grande": "Salao grande",
    "catedral": "Catedral / hall reverberante",
    "ar_livre": "Ar livre / externa",
}

ROOM_PRESET_CHOICES = [(ROOM_PRESET_LABELS[key], key) for key in ROOM_PRESETS.keys()]


def _ffmpeg_bin():
    return os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")


def has_pedalboard():
    return _HAS_PEDALBOARD


def apply_room_reverb(in_path, out_path, preset="quarto_intimo", distance=0.0, log=None):
    """Give a dry vocal WAV spatial/environmental character.

    preset: key into ROOM_PRESETS ("none" or missing -> passthrough, returns
      in_path unchanged, no file written).
    distance: 0.0 (perto da camera) .. 1.0 (longe) -- adds extra wet mix,
      attenuates highs and overall level to fake air absorption + falloff.
    Returns out_path on success, or in_path unchanged if skipped/failed.
    """
    def _log(msg):
        if log:
            log(msg)

    if not preset or preset == "none" or preset not in ROOM_PRESETS or ROOM_PRESETS[preset] is None:
        return in_path
    if not in_path or not os.path.exists(in_path):
        return in_path

    distance = max(0.0, min(1.0, float(distance)))
    params = dict(ROOM_PRESETS[preset])
    params["wet_level"] = min(1.0, params["wet_level"] + distance * 0.15)
    params["dry_level"] = max(0.0, params["dry_level"] - distance * 0.15)

    if _HAS_PEDALBOARD:
        try:
            with AudioFile(in_path) as f:
                audio = f.read(f.frames)
                sr = f.samplerate
            board = Pedalboard([
                Reverb(**params),
                LowpassFilter(cutoff_frequency_hz=20000 - distance * 9000),
                Gain(gain_db=-distance * 4.0),
            ])
            processed = board(audio, sr)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with AudioFile(out_path, 'w', sr, processed.shape[0]) as f:
                f.write(processed)
            _log(f"reverb '{preset}' aplicado (pedalboard, distancia={distance:.2f})")
            return out_path
        except Exception as exc:
            _log(f"pedalboard falhou ({exc}); tentando fallback ffmpeg")

    return _ffmpeg_reverb_fallback(in_path, out_path, params, distance, _log)


def _ffmpeg_reverb_fallback(in_path, out_path, params, distance, log):
    ffmpeg = _ffmpeg_bin()
    wet = params["wet_level"]
    decay = min(0.9, 0.3 + wet)
    lowpass_hz = int(20000 - distance * 9000)
    gain_db = -distance * 4.0
    af = (
        f"aecho=0.8:{decay:.2f}:40|80|120:{wet:.3f}|{wet * 0.6:.3f}|{wet * 0.3:.3f},"
        f"lowpass=f={lowpass_hz},"
        f"volume={10 ** (gain_db / 20):.3f}"
    )
    try:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", in_path, "-af", af, out_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            log(f"reverb aplicado (ffmpeg fallback, decay={decay:.2f})")
            return out_path
        log(f"ffmpeg reverb fallback falhou: {(result.stderr or '')[-500:]}")
    except Exception as exc:
        log(f"ffmpeg reverb fallback erro: {exc}")
    return in_path


def mix_vocal_with_instrumental(vocal_path, instrumental_path, out_path, ffmpeg=None, log=None):
    """Sum a (possibly reverberant) vocal stem back with the instrumental
    stem into a single master WAV. Uses amix with normalize=0 so the two
    Demucs-complementary stems recombine at their original relative levels
    instead of being attenuated -6dB each by ffmpeg's default normalization.
    """
    def _log(msg):
        if log:
            log(msg)

    ffmpeg = ffmpeg or _ffmpeg_bin()
    if not vocal_path or not os.path.exists(vocal_path):
        return instrumental_path
    if not instrumental_path or not os.path.exists(instrumental_path):
        return vocal_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", vocal_path, "-i", instrumental_path,
         "-filter_complex",
         "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
         "alimiter=limit=0.97:attack=5:release=50:level=false[aout]",
         "-map", "[aout]", out_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.exists(out_path):
        return out_path
    _log(f"mix vocal+instrumental falhou: {(result.stderr or '')[-500:]}")
    return instrumental_path


def build_spatialized_master(vocal_path, instrumental_path, work_dir, preset="quarto_intimo",
                              distance=0.0, stamp="", ffmpeg=None, log=None):
    """One-shot helper: reverb the vocal, remix with instrumental, return the
    path to a new master WAV -- or None if preset is 'none'/nothing to do,
    signalling the caller should keep using its original master audio.
    """
    if not preset or preset == "none":
        return None
    if not vocal_path or not os.path.exists(vocal_path):
        return None
    reverb_path = os.path.join(work_dir, f"vocal_reverb_{preset}_{stamp}.wav")
    reverb_vocal = apply_room_reverb(vocal_path, reverb_path, preset=preset, distance=distance, log=log)
    if instrumental_path and os.path.exists(instrumental_path):
        master_path = os.path.join(work_dir, f"master_spatialized_{stamp}.wav")
        return mix_vocal_with_instrumental(reverb_vocal, instrumental_path, master_path, ffmpeg=ffmpeg, log=log)
    return reverb_vocal
