"""Discovery and emotion-matching over the xtts speaker library.

The library grew three tiers, and only the first was ever visible to the pipeline:

  1. FLAT legacy clips at the root: male.wav, female.wav, calm_female.wav, sandra.mp3
     plus 24 per-character takes (01_lyra_base.wav, ...) indexed by mapa_vozes.csv.
  2. 01_vozes_emotivas/<VOICE>/<NN>_<emotion>.wav -- 12 voices (F01-F06, M01-M06),
     each recorded in the SAME 17 emotional states. Uniformity verified.
  3. 02_vozes_de_personagens/<CNN_archetype>/ -- 28 archetypes (hero, witch, ghost,
     android, narrator, ...), each with base_identidade.wav plus one acted take.

Tiers 2 and 3 were invisible because discovery used a NON-recursive glob("*.wav").
That mattered: XTTS has no textual emotion control -- "instruct" reaches only the qwen
engine -- so a per-emotion reference recording is the ONLY way to make XTTS act a line.
A 17-emotion library is exactly the missing capability, sitting unused in subfolders.

Emotion names are Portuguese; the parser emits English tags ("angry, raised voice",
"whispering, afraid"). match_emotion() bridges the two by keyword.

The _APROX suffix on some takes is the library author's own honesty marker: those
emotions are approximated by the source recording rather than cleanly acted. It is
kept in the filename and ignored for matching.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

XTTS_SPEAKER_DIR = Path(r"E:\Users\home\Documents\xtts\webui\speakers")
EMOTIVE_DIR = "01_vozes_emotivas"
ARCHETYPE_DIR = "02_vozes_de_personagens"

# Portuguese emotion slug -> English/Portuguese cues the parser (or a screenplay
# parenthetical) might use. First match wins, so order is by specificity: "grito"
# before "raiva" because "screaming in anger" should get the shout take.
EMOTION_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("grito", ("shout", "scream", "yell", "roar", "grito", "gritando", "berr")),
    ("raiva", ("angry", "anger", "furious", "rage", "irritat", "raiva", "furioso", "furiosa")),
    ("com_medo", ("afraid", "fear", "scared", "terrified", "medo", "assustad")),
    ("angustia", ("anguish", "desperate", "distress", "agoniz", "angustia", "aflit",
                  "worried", "worry", "anxious", "nervous", "preocupad", "ansios")),
    ("tristeza", ("sad", "sorrow", "grief", "mourn", "tearful", "triste", "chorando")),
    ("desanimo", ("discouraged", "defeated", "weary", "resigned", "desanim", "abatid")),
    ("desapontamento", ("disappointed", "letdown", "desapont", "decepcion")),
    ("surpresa", ("surprised", "surprise", "astonish", "shocked", "startled", "surpres", "espant")),
    ("confusa", ("confused", "puzzled", "bewildered", "questioning", "confus", "duvid")),
    ("excitada", ("excited", "thrilled", "eager", "urgent", "breathless", "excitad", "urgente", "ofegante")),
    ("espontanea_entusiasmada", ("enthusiastic", "spontaneous", "lively", "entusiasm", "espontane")),
    ("alegre", ("happy", "cheerful", "joy", "bright", "smiling", "alegre", "feliz", "animad")),
    ("apaixonada", ("in love", "loving", "tender", "affectionate", "apaixonad", "carinhos")),
    ("sensual", ("sensual", "seductive", "sultry", "flirt", "sedutor")),
    ("desdem", ("disdain", "contempt", "scornful", "mocking", "sneer", "desdem", "despriz", "zomb")),
    ("calma", ("calm", "steady", "quiet", "gentle", "reassuring", "soft", "calmo", "calma", "tranquil", "sereno")),
    ("neutra", ("neutral", "flat", "matter-of-fact", "neutro", "neutra")),
]


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return folded.lower()


def _emotion_slug(filename: str) -> str:
    """'06_desapontamento_APROX.wav' -> 'desapontamento'."""
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)
    return re.sub(r"_APROX$", "", stem, flags=re.IGNORECASE)


def list_flat_speakers() -> list[str]:
    """Legacy root-level clips, by stem -- the historical behaviour, unchanged."""
    if not XTTS_SPEAKER_DIR.is_dir():
        return []
    return sorted(p.stem for p in XTTS_SPEAKER_DIR.glob("*.wav"))


def list_emotive_voices() -> dict[str, dict[str, Path]]:
    """{voice_id: {emotion_slug: absolute path}} for tier 2."""
    root = XTTS_SPEAKER_DIR / EMOTIVE_DIR
    if not root.is_dir():
        return {}
    voices: dict[str, dict[str, Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        takes = {_emotion_slug(f.name): f for f in sorted(folder.glob("*.wav"))}
        if takes:
            voices[folder.name] = takes
    return voices


def list_archetype_voices() -> dict[str, dict[str, Path]]:
    """{archetype_id: {'base': path, 'acted': path|None}} for tier 3."""
    root = XTTS_SPEAKER_DIR / ARCHETYPE_DIR
    if not root.is_dir():
        return {}
    voices: dict[str, dict[str, Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            continue
        base = next((w for w in wavs if "base" in w.stem), wavs[0])
        acted = next((w for w in wavs if "atuacao" in w.stem), None)
        voices[folder.name] = {"base": base, "acted": acted}
    return voices


def voice_gender(voice_id: str) -> Optional[str]:
    """Tier-2 ids encode gender in the prefix: F01_... / M01_..."""
    if re.match(r"^F\d", voice_id):
        return "female"
    if re.match(r"^M\d", voice_id):
        return "male"
    return None


def match_emotion(text: Optional[str], available: set[str]) -> Optional[str]:
    """Map a free-text emotion cue onto one of the library's emotion slugs.

    Returns None when nothing matches, which callers must treat as "use the neutral
    take" rather than as an error -- a wrong emotion is worse than a neutral one.
    """
    if not text:
        return None
    haystack = _fold(text)
    for slug, cues in EMOTION_CUES:
        if slug not in available:
            continue
        if any(cue in haystack for cue in cues):
            return slug
    return None


def pick_take(voice_id: str, emotion_text: Optional[str],
              voices: Optional[dict] = None) -> Optional[str]:
    """Absolute path (str) of the take for this voice best fitting `emotion_text`.

    Falls back to 'neutra', then 'calma', then any take. Returned as an absolute
    path because xtts's get_speaker_path() only accepts a bare name (root level) or
    a path ending in .wav -- a nested clip must be addressed by path.
    """
    voices = voices if voices is not None else list_emotive_voices()
    takes = voices.get(voice_id)
    if not takes:
        return None
    slug = match_emotion(emotion_text, set(takes))
    for candidate in (slug, "neutra", "calma"):
        if candidate and candidate in takes:
            return str(takes[candidate])
    return str(next(iter(takes.values())))


def archetype_take(archetype_id: str, emotion_text: Optional[str] = None,
                   archetypes: Optional[dict] = None) -> Optional[str]:
    """Absolute path of an archetype's clip: the acted take when the line carries an
    emotion the take actually covers, otherwise the base identity clip.

    Archetypes ship ONE acted take each, and which one varies by archetype
    (atuacao_shout for the hero, atuacao_anger for the witch, atuacao_calm for the
    mage). So the emotion is only honoured when it matches that specific take -- an
    "angry" line does not get the calm mage's only alternative recording just because
    an alternative exists.
    """
    archetypes = archetypes if archetypes is not None else list_archetype_voices()
    entry = archetypes.get(archetype_id)
    if not entry:
        return None
    acted = entry.get("acted")
    if acted and emotion_text:
        acted_slug = re.sub(r"^atuacao_", "", acted.stem, flags=re.IGNORECASE)
        if match_emotion(emotion_text, {_ACTED_ALIASES.get(acted_slug, acted_slug)}):
            return str(acted)
    return str(entry["base"])


# Archetype and mapa_vozes takes are named in English (atuacao_shout, ..._estilo_calm);
# EMOTION_CUES is keyed by the Portuguese slugs of the emotive library. Bridge the two.
_ACTED_ALIASES = {"shout": "grito", "anger": "raiva", "calm": "calma",
                  "whisper": "calma", "sad": "tristeza", "fear": "com_medo",
                  "worried": "angustia", "surprised": "surpresa", "excited": "excitada"}


def style_clip_fits(style_name: str, emotion_text: Optional[str]) -> bool:
    """Does a single acted take (e.g. '02_thoren_estilo_calm') suit this line?

    MEASURED regression this prevents: the first version used the style clip for ANY
    line carrying an emotion, so a "shouting, forceful" line was delivered with
    thoren's CALM take -- the acted clip actively fought the intended delivery. One
    acted take covers one emotion; when the line is not that emotion, the neutral base
    is the honest choice.
    """
    if not style_name or not emotion_text:
        return False
    slug = re.split(r"_estilo_|_atuacao_|^atuacao_", style_name)[-1].lower()
    slug = _ACTED_ALIASES.get(slug, slug)
    return match_emotion(emotion_text, {slug}) is not None


def describe_library() -> str:
    flat = list_flat_speakers()
    emotive = list_emotive_voices()
    archetypes = list_archetype_voices()
    emotions = sorted({e for takes in emotive.values() for e in takes})
    lines = [
        f"raiz (legado):        {len(flat)} clipe(s)",
        f"{EMOTIVE_DIR}: {len(emotive)} voz(es) x {len(emotions)} emocao(oes)",
        f"{ARCHETYPE_DIR}: {len(archetypes)} arquetipo(s)",
    ]
    if emotive:
        males = [v for v in emotive if voice_gender(v) == "male"]
        females = [v for v in emotive if voice_gender(v) == "female"]
        lines.append(f"  masculinas: {len(males)} | femininas: {len(females)}")
        lines.append(f"  emocoes: {', '.join(emotions)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_library())
