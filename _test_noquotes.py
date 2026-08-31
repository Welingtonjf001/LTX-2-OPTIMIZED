# -*- coding: utf-8 -*-
"""Decisive test for the burned-in subtitles: same scene, dialogue described
INDIRECTLY (no quoted lines) instead of verbatim between quotes.

Everything else identical (seed, size, duration, negative prompt, distilled
variant). If the subtitles disappear, the cause is the quoted dialogue in the
POSITIVE prompt -- not the negative prompt, and not CFG (dev with real CFG 3/7
still produced them; see MEMORIAL.md 3.9).
"""
import sys
sys.path.insert(0, ".")
import ltx25_backend as b
from _test_nag_subtitles import NEGATIVE

NO_QUOTES = (
    "A cinematic live-action western confrontation on an empty frontier street at golden hour. "
    "Keep exactly two characters with fixed appearances and positions. Clara stands on the left, "
    "a sharp-eyed female gunslinger in her early thirties, wearing a dusty beige hat, red neckerchief, "
    "long brown coat and a rifle held low beside her leg. Cole stands on the right, a rugged male outlaw "
    "with dark stubble, a black hat, weathered leather coat and one hand hovering near his holstered "
    "revolver. The camera holds a tense medium two-shot and performs a very slow dolly forward. Dust "
    "crosses the street, a wooden sign creaks, distant horses snort and a lonely harmonica plays softly. "
    "All speech is spoken aloud in Brazilian Portuguese and is never shown as text. "
    "Clara narrows her eyes and speaks firmly, warning Cole to lower his revolver. Clara closes her mouth. "
    "Cole tilts his hat and answers in a low rough voice, refusing her and mentioning the gold on the train. "
    "Cole becomes silent. Clara tightens her grip on the rifle and answers with cold confidence about her aim. "
    "Clara closes her mouth. Cole gives a restrained smile, his spurs shifting in the dust, and answers that "
    "they will settle it at noon. A distant clock bell rings once. No gunfire, subtitles or written text."
)

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "distilled"
    out = b.generate(
        NO_QUOTES, f"outputs_25/noquotes_{variant}.mp4",
        negative=NEGATIVE, width=768, height=512,
        num_frames=361, frame_rate=24.0, seed=42,
        variant=variant, log_cb=lambda m: print(m, flush=True), timeout=7200,
    )
    print("OK ->", out)
