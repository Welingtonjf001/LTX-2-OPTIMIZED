"""Remove LTX-2.3's burned-in subtitle band by cropping it off.

WHY THIS EXISTS -- and why it is a crop rather than a fix at the source.

LTX-2.3 paints garbled subtitles over any clip containing speech. The text is never
readable ("Yau oughe wras sherios.", "Thon, as Paho pascrs atrer."), it is drawn into
the pixels, and it appears in most clips with dialogue. Four remedies were tested on the
same prompt, seed and resolution, each attacking a different part of the process:

  1. "No subtitles or on-screen text" stated affirmatively in the prompt   -> subtitles
  2. Real negative prompt at CFG 3.0 via the 43 GB base checkpoint         -> subtitles
  3. res2s second-order sampler on both stages (the fix named in
     Lightricks/LTX-2.3 discussion #5)                                     -> subtitles
  4. NAG at the community's reported-working nag_scale=11.0, nag_alpha=0.25,
     nag_tau=2.5, verified present in the executed graph                   -> subtitles

Text, guidance, sampling and attention all failed, which points at the model's training
rather than at any knob. So this module does not argue with the model: it measures where
the text lands and cuts that band away.

MEASURED on a 1536x896 clip carrying a subtitle, near-white pixel density per 10% band:

    0-40%: 0.00%   40-50%: 0.15%   50-60%: 0.93%   60-70%: 0.95%
    70-80%: 1.20%  80-90%: 6.72%   90-100%: 0.10%

The text sits in the 80-90% band. Cropping to the top 78.6% (896 -> 704) drops the peak
to 1.19% -- what remains is scene lighting, not lettering -- and yields a 2.18:1 frame.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_KEEP = 0.786  # fraction of height to keep; see the measurement above
BRIGHT = 235          # luma above which a pixel counts as "possible lettering"


def _ffmpeg() -> str:
    return "ffmpeg"


def probe_size(path: str) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    width, height = (int(v) for v in result.stdout.strip().split(",")[:2])
    return width, height


def bright_profile(path: str, at_second: float = 2.5, bands: int = 10) -> list[float]:
    """Percentage of near-white pixels per horizontal band of one frame.

    A DIAGNOSTIC, NOT A SUBTITLE DETECTOR. It measures brightness, and only reads as
    "subtitle here" when the rest of the frame is dark. MEASURED counter-example: on the
    western clip at golden hour the peak band is 30-40% -- that is the sky, and there is
    no subtitle in that frame at all. Use it to confirm where text sits in a frame you
    have already seen carrying text; do not use it to decide whether text is present.
    """
    from PIL import Image  # imported lazily: callers that only crop don't need PIL

    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "f.png"
        subprocess.run([_ffmpeg(), "-y", "-v", "error", "-ss", str(at_second),
                        "-i", str(path), "-frames:v", "1", str(frame)], check=True)
        with Image.open(frame) as image:
            grey = image.convert("L")
            width, height = grey.size
            pixels = grey.load()
            profile = []
            for index in range(bands):
                y0, y1 = height * index // bands, height * (index + 1) // bands
                hits = sum(1 for y in range(y0, y1) for x in range(0, width, 2)
                           if pixels[x, y] > BRIGHT)
                total = (y1 - y0) * len(range(0, width, 2)) or 1
                profile.append(100.0 * hits / total)
    return profile


def crop(src: str, dest: str, *, keep: float = DEFAULT_KEEP, crf: int = 18) -> bool:
    """Crop the bottom off `src`. Audio is stream-copied, so speech is untouched."""
    width, height = probe_size(src)
    new_height = int(height * keep) // 2 * 2  # h264 needs even dimensions
    if new_height >= height:
        return False
    result = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-i", str(src),
         "-vf", f"crop={width}:{new_height}:0:0",
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "copy", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0 and Path(dest).exists()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="video file(s) to process")
    parser.add_argument("--keep", type=float, default=DEFAULT_KEEP,
                        help=f"fraction of the height to keep (default {DEFAULT_KEEP})")
    parser.add_argument("--suffix", default="_nosub",
                        help="output suffix; the source is never overwritten")
    parser.add_argument("--measure", action="store_true",
                        help="report the brightness profile only, crop nothing. Diagnostic "
                             "aid: it finds bright pixels, not text -- a bright sky peaks too.")
    parser.add_argument("--at", type=float, default=2.5,
                        help="timestamp (s) to sample when measuring")
    args = parser.parse_args(argv)

    failures = 0
    for video in args.videos:
        path = Path(video)
        if not path.exists():
            print(f"strip_subtitles: nao encontrado: {path}")
            failures += 1
            continue
        if args.measure:
            profile = bright_profile(str(path), at_second=args.at)
            peak = max(range(len(profile)), key=lambda i: profile[i])
            print(f"{path.name}:")
            for index, value in enumerate(profile):
                mark = "  <- pico" if index == peak else ""
                print(f"  banda {index * 10:3d}-{(index + 1) * 10:3d}%  {value:5.2f}%{mark}")
            continue
        dest = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        if crop(str(path), str(dest), keep=args.keep):
            width, height = probe_size(str(dest))
            print(f"strip_subtitles: {path.name} -> {dest.name} ({width}x{height})")
        else:
            print(f"strip_subtitles: falhou em {path.name}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
