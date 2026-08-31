"""Screenplay parsing: deterministic structure first, LLM enrichment second.

Structure (scene headings, character cues, dialogue, parentheticals) is extracted by
regex/heuristics -- screenplay formatting is regular enough that this is reliable and,
critically, deterministic: it never hallucinates a scene or a line of dialogue that
isn't in the source text. The LLM (Gemma 3 12B IT, already on disk at
``models/gemma3`` -- see module docstring in ``verify_gemma_text.py`` for why this is
safe to reuse as a general text model) is used only to *enrich* what the structural
pass already found: a cinematic visual prompt per scene, and an emotion/delivery hint
for dialogue lines that have no explicit parenthetical. It never decides structure.

CLI: ``python -m script_pipeline.parse_screenplay --script FILE --run-dir DIR [--no-llm]``
Writes ``<run-dir>/parse/scenes.json`` (structural only) and, unless ``--no-llm``,
``<run-dir>/parse/scenes_enriched.json`` (structural + LLM fields).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

SCENE_HEADING_RE = re.compile(
    # Alternância do MAIS LONGO para o mais curto: com "INT|EXT|INT/EXT",
    # o cabeçalho "INT/EXT. CARRO" casava `INT` e o `[./]` engolia a barra,
    # deixando "EXT. CARRO" dentro do NOME DO LOCAL. Defeito silencioso.
    # As QUATRO combinações precisam estar listadas: na primeira correção eu pus
    # só INT/EXT e I/E, e "EXT/INT. PALACE..." caiu no mesmo buraco no dia
    # seguinte, com um roteiro real. Corrigido em 2026-08-26.
    r"^\s*(?:\d+[.\)]\s*)?(INT/EXT|EXT/INT|I/E|E/I|INT|EXT)[./]\s*(.+?)\s*(?:-\s*(.+))?$",
    re.IGNORECASE,
)
# Alternate scene-heading style seen in translated/adapted scripts (e.g. K-drama
# treatments): "Cena 1 -- Local, condicao" instead of "INT./EXT. LOCAL - TIME".
# No INT/EXT distinction available here, so heading_raw carries the whole line and
# location holds the free-text description; time_of_day is left for the LLM
# enrichment pass to infer from context (never guessed by regex).
SCENE_HEADING_ALT_RE = re.compile(r"^\s*Cena\s+\d+\s*[—–\-]\s*(.+)$", re.IGNORECASE)

# Períodos do dia reconhecidos no fim de um cabeçalho. Existe porque o
# cabeçalho de TRÊS partes -- "INT. TORRE DO RELÓGIO - ESCADA - NOITE", formato
# padrão em roteiro -- quebrava o parse: o regex divide no PRIMEIRO hífen, então
# `location` ficava "TORRE DO RELÓGIO" e `time_of_day` ficava "ESCADA - NOITE".
# MEDIDO 2026-08-26. Não era cosmético: `time_of_day` vai para os prompts de
# storyboard e render, então o modelo recebia "escada" como período do dia; e o
# sub-local sumia de `location`, fundindo cenas espacialmente distintas num
# lugar só. Inclui pt e en porque o pipeline aceita roteiro nas duas línguas.
_TIME_OF_DAY_WORDS = {
    "dia", "noite", "tarde", "manha", "manhã", "amanhecer", "anoitecer", "entardecer",
    "madrugada", "meia-noite", "meio-dia", "crepusculo", "crepúsculo", "alvorada",
    "continuo", "contínuo", "mais tarde", "momentos depois", "mesmo tempo",
    "day", "night", "morning", "evening", "afternoon", "dawn", "dusk", "midnight",
    "noon", "continuous", "later", "moments later", "same time", "sunset", "sunrise",
}


_HEADING_PREFIX_RE = re.compile(r"^\s*(?:\d+[.\)]\s*)?(?:INT/EXT|EXT/INT|I/E|E/I|INT|EXT)[./]\s*", re.IGNORECASE)


def _split_location_time(heading: str) -> tuple[str, str]:
    """Devolve (local, período) a partir do cabeçalho CRU.

    Trabalha no texto original, não nos grupos do regex, porque remontar a
    partir deles reformata o nome: `EXT. PONTE RIO-NITEROI - DIA` tem hífen
    SEM espaços dentro do próprio nome, o regex quebra ali, e qualquer
    remontagem devolve "PONTE RIO - NITEROI". Cortando só o sufixo do texto
    original, o nome fica intacto.

    A regra: só a ÚLTIMA parte separada por ` - ` conta como período, e só se
    parecer um. Qualquer coisa antes dela é sub-local e permanece no local.
    Quando nada parece período, tudo é local e o período fica vazio -- melhor
    vazio que errado, porque campo vazio o prompt omite e campo errado afirma."""
    corpo = _HEADING_PREFIX_RE.sub("", heading or "").strip()
    if not corpo:
        return "", ""
    # Só separadores COM espaço em volta: preserva hífen interno de nome próprio.
    partes = re.split(r"\s+[-–—]\s+", corpo)
    if len(partes) > 1 and partes[-1].strip().lower() in _TIME_OF_DAY_WORDS:
        periodo = partes[-1].strip()
        # Recorta o local do texto original, em vez de juntar as partes.
        corte = corpo.rfind(partes[-1])
        local = corpo[:corte].rstrip().rstrip("-–—").rstrip()
        return local, periodo
    return corpo, ""


PARENTHETICAL_RE = re.compile(r"^\s*\((.+)\)\s*$")
# A character cue: short, all-uppercase (accented letters allowed for PT names),
# optional trailing "(V.O.)"/"(O.S.)"/"(CONT'D)" style tag. Checked only after the
# scene-heading pattern has already been ruled out for this line.
CUE_RE = re.compile(
    r"^\s*([A-ZÀ-Ý][A-ZÀ-Ý0-9 .'\-]{0,38}?)(\s*\([^)]*\))?\s*$"
)
# Freeform-paragraph fallback (2026-08-10): a screenplay pasted as one continuous
# run-on paragraph, no line breaks, dialogue marked inline by a speech verb + colon
# + dash rather than any line-based convention (e.g. "ela exclama: - No horario! -").
# MEASURED against a real example: this is genuinely ambiguous for regex (dashes are
# both the dialogue delimiter AND ordinary punctuation inside compound words like
# "guarda-chuva"; some lines have no speech-verb marker at all, e.g. a bare
# "- Muito obrigado!" after an action sentence, with the speaker only inferable from
# who the PRECEDING sentence's subject was). Deliberately conservative: only extracts
# a line when a speech-verb marker is found, and only match delimiter dashes that
# have whitespace on both sides (so "guarda-chuva" is never split) -- under-extracting
# (missing a line, which the user can add by hand to cast.json/manually) is the safe
# failure mode here, not inventing or misattributing one.
_FREEFORM_SPEECH_VERB = r"\w*(?:exclam|diz|avis|pergunt|respond|grit|sussurr|fal)\w*(?:-\w+)?"
FREEFORM_SPEECH_MARKER_RE = re.compile(_FREEFORM_SPEECH_VERB + r"[:,]\s*(?:-\s*)?")
FREEFORM_DIALOGUE_CLOSE_RE = re.compile(r"(.+?)(?:(?=\s-\s)|([.!?])\s+(?=[A-ZÀ-Ú])|(?=,\s+\w*ndo\b)|\Z)")
# Character names in this style are introduced as a dash- or comma-flanked aside
# right after first mention, e.g. "a jovem - Park Min - personagem feminina" or
# "Um rapaz - Kim -" -- the same convention the user's own example used for both
# characters, so this is a real signal specific to this writing style, not a guess.
FREEFORM_NAME_INTRO_RE = re.compile(r"[-–]\s*([A-ZÀ-Ú][\wÀ-ú]*(?:\s[A-ZÀ-Ú][\wÀ-ú]*){0,2})\s*[-–,]")


def _extract_freeform_scene(text: str) -> Optional["Scene"]:
    """Fallback structural pass for a screenplay with no detectable scene heading
    anywhere (see module note above) -- treats the WHOLE text as one implicit scene
    and pulls out whatever dialogue the conservative markers above can find. Returns
    None (not an empty Scene) when there's nothing at all to work with, so the
    caller's "no scenes detected" error still fires for genuinely empty/garbage input
    rather than silently producing a scene with zero content."""
    stripped = text.strip()
    if not stripped:
        return None

    names: list[str] = []
    for m in FREEFORM_NAME_INTRO_RE.finditer(stripped):
        candidate = m.group(1).strip()
        if candidate and candidate not in names:
            names.append(candidate)
    name_pattern = re.compile("|".join(re.escape(n) for n in names)) if names else None

    dialogue_lines: list[DialogueLine] = []
    characters: list[str] = []
    for marker in FREEFORM_SPEECH_MARKER_RE.finditer(stripped):
        rest = stripped[marker.end():]
        close = FREEFORM_DIALOGUE_CLOSE_RE.match(rest)
        if not close:
            continue
        dialogue_text = (close.group(1) + (close.group(2) or "")).strip()
        if not dialogue_text:
            continue
        speaker = None
        if name_pattern:
            for nm in name_pattern.finditer(stripped[:marker.start()]):
                speaker = nm.group(0)  # keep the LAST (nearest-preceding) match
        speaker = speaker or "PERSONAGEM"
        dialogue_lines.append(DialogueLine(character=speaker, text=dialogue_text))
        if speaker not in characters:
            characters.append(speaker)

    if not dialogue_lines and not names:
        return None  # nothing structural found at all -- let the caller report failure

    # MEASURED (2026-08-11): heading_raw used to be stripped[:80] + "...", i.e. the raw
    # paragraph chopped mid-word. render_scenes pasted that fragment at the FRONT of
    # every LTX prompt, so the model read a truncated Portuguese sentence before any
    # real instruction -- a direct cause of nonsensical shots. A freeform script has no
    # real slugline, so say so plainly instead of inventing one out of debris.
    return Scene(
        index=1, heading_raw="", location="", time_of_day="", action_text=stripped,
        characters=characters, dialogue=dialogue_lines,
    )


# Alternate dialogue style: "N. Character: line text" all on one line, character
# name in normal case (not ALL-CAPS) -- also seen in the same adapted-script
# convention as SCENE_HEADING_ALT_RE. Character name is kept short/name-shaped
# (letters, spaces, hyphens, apostrophes, dots) to avoid matching an ordinary
# numbered action sentence like "1. She walks to the door." as dialogue.
INLINE_DIALOGUE_RE = re.compile(r"^\s*\d+[.\)]\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-. ]{0,40}?):\s*(.+)$")


@dataclasses.dataclass
class DialogueLine:
    character: str
    text: str
    parenthetical: Optional[str] = None
    emotion: Optional[str] = None  # filled by LLM enrichment when parenthetical is None
    beat_visual: Optional[str] = None  # filled by LLM enrichment: what's visually happening AT this line
    # (2026-08-10) render_scenes.py used the scene's WHOLE action_text as every
    # clip's base prompt -- every line in a scene got the identical description,
    # so nothing told the model what changed between one line and the next, which
    # is why generated clips didn't chain into coherent action. beat_visual is a
    # per-line "what's happening right now, continuing from the previous beat"
    # snippet the enrichment pass writes for every line (both engines), used
    # instead of the flat action_text when present.

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Scene:
    index: int
    heading_raw: str
    location: str
    time_of_day: str
    action_text: str = ""
    characters: list = dataclasses.field(default_factory=list)
    dialogue: list = dataclasses.field(default_factory=list)  # list[DialogueLine]
    visual_prompt: Optional[str] = None  # filled by LLM enrichment
    # O MEIO da obra (cel animation, 3D storybook, live action...), preenchido
    # pelo enriquecimento. Separado de visual_prompt porque vale para o FILME
    # inteiro e nao para o inicio de uma cena -- e porque quem consome e o
    # `look` do prompt de imagem, nao a descricao da acao. Ver _apply_setting.
    art_direction: Optional[str] = None
    # (2026-08-10) The scene's shots IN PLAYBACK ORDER -- both wordless action beats
    # (the bus arriving, doors opening) and the dialogue lines, interleaved.
    # Each entry: {"type": "action", "visual": str} or {"type": "dialogue", "line_index": int}.
    #
    # MEASURED, two rounds. Round 1: rendering ONLY dialogue lines turned a script
    # describing ~25s of story into a 9.9s film -- half the action had no line
    # attached, so it produced no shot at all. Round 2 asked the LLM for action beats
    # tagged with an "after_line" index, and THAT reproducibly scrambled the story:
    # the model dumped 8 of 13 beats onto after_line=-1, so the film opened with its
    # own ending (the bow, the sigh, the final questioning look) and only then played
    # the first line. Asking a model to assign each beat a relative index, out of
    # order, is the wrong shape of question. This field instead asks for ONE ordered
    # sequence -- the order IS the answer, not something inferred from indices.
    # Empty list keeps the old dialogue-only behaviour.
    shot_list: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["dialogue"] = [line.to_dict() if isinstance(line, DialogueLine) else line for line in self.dialogue]
        return data


def _looks_like_cue(line: str, next_nonblank: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
    """Return (character_name, trailing_tag) if `line` is plausibly a character cue."""
    if SCENE_HEADING_RE.match(line):
        return None
    if not line.strip():
        return None
    if line.strip() != line.strip().upper():
        return None  # must be all-caps (case-sensitive check against accented upper too)
    match = CUE_RE.match(line.strip())
    if not match:
        return None
    name = match.group(1).strip()
    if len(name) < 2:
        return None
    # A cue is only meaningful if something (dialogue or a parenthetical) follows it.
    if next_nonblank is None:
        return None
    return name, (match.group(2) or "").strip("() ").strip() or None


def _strip_markdown(text: str) -> str:
    """Screenplays pasted from a rendered Markdown source (chat, doc export) commonly
    carry **bold** around scene headings/character cues and a leading #/##/### on
    title lines -- none of that is screenplay structure, and left in place it breaks
    every structural regex below (e.g. "**Cena 1 -- ..." no longer starts with "Cena"
    at column 0). Stripped unconditionally: a real screenplay has no legitimate use
    for literal "**" or a leading "#" run at line-start, so this never costs
    anything on a plain-text script and unblocks a Markdown-formatted one."""
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)


def parse_structure(text: str) -> list[Scene]:
    """Deterministic regex/heuristic screenplay parse. Never invents content."""
    text = _strip_markdown(text)
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in raw_lines]

    scenes: list[Scene] = []
    current: Optional[Scene] = None
    pending_speaker: Optional[str] = None
    pending_parenthetical: Optional[str] = None
    dialogue_buffer: list[str] = []

    def flush_dialogue():
        nonlocal pending_speaker, pending_parenthetical, dialogue_buffer
        if pending_speaker and dialogue_buffer and current is not None:
            text_joined = " ".join(part.strip() for part in dialogue_buffer if part.strip())
            if text_joined:
                current.dialogue.append(DialogueLine(
                    character=pending_speaker,
                    text=text_joined,
                    parenthetical=pending_parenthetical,
                ))
                if pending_speaker not in current.characters:
                    current.characters.append(pending_speaker)
        pending_speaker = None
        pending_parenthetical = None
        dialogue_buffer = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        next_nonblank = next((later.strip() for later in lines[i + 1:] if later.strip()), None)

        heading = SCENE_HEADING_RE.match(stripped) if stripped else None
        if heading:
            flush_dialogue()
            location, time_of_day = _split_location_time(stripped)
            current = Scene(
                index=len(scenes) + 1,
                heading_raw=stripped,
                location=location,
                time_of_day=time_of_day,
            )
            scenes.append(current)
            continue

        heading_alt = SCENE_HEADING_ALT_RE.match(stripped) if stripped else None
        if heading_alt:
            flush_dialogue()
            current = Scene(
                index=len(scenes) + 1,
                heading_raw=stripped,
                location=heading_alt.group(1).strip(),
                time_of_day="",
            )
            scenes.append(current)
            continue

        if current is None:
            # Content before any scene heading -- ignore (title page / cold open text
            # without a slugline). A real screenplay always starts with a heading;
            # if this one doesn't, that's a formatting problem for the user to fix,
            # not something this parser should guess at.
            continue

        if not stripped:
            flush_dialogue()
            continue

        paren = PARENTHETICAL_RE.match(stripped)
        if paren and (pending_speaker is not None):
            # Parenthetical inside a dialogue block: attaches to the line, doesn't
            # start a new one.
            if pending_parenthetical:
                pending_parenthetical += "; " + paren.group(1).strip()
            else:
                pending_parenthetical = paren.group(1).strip()
            continue

        inline = INLINE_DIALOGUE_RE.match(stripped)
        if inline is not None:
            flush_dialogue()
            name = inline.group(1).strip()
            text = inline.group(2).strip()
            if name and text:
                current.dialogue.append(DialogueLine(character=name, text=text))
                if name not in current.characters:
                    current.characters.append(name)
            continue

        cue = _looks_like_cue(stripped, next_nonblank)
        if cue is not None:
            flush_dialogue()
            pending_speaker, tag = cue
            pending_parenthetical = tag
            continue

        if pending_speaker is not None:
            dialogue_buffer.append(stripped)
        else:
            current.action_text = (current.action_text + " " + stripped).strip()

    flush_dialogue()
    if not scenes:
        # Neither line-based convention (INT./EXT. nor "Cena N -- ...") found a
        # single heading anywhere -- try the freeform-paragraph fallback before
        # giving up entirely. Still returns [] (not a guess) if that finds nothing
        # structural either; main() reports "no scenes detected" exactly as before.
        freeform = _extract_freeform_scene(text)
        if freeform is not None:
            scenes = [freeform]
    return scenes


def scenes_to_json(scenes: list[Scene]) -> list[dict]:
    return [scene.to_dict() for scene in scenes]


def scenes_from_json(data: list[dict]) -> list[Scene]:
    scenes = []
    for item in data:
        dialogue = [DialogueLine(**line) for line in item.get("dialogue", [])]
        item = dict(item)
        item["dialogue"] = dialogue
        scenes.append(Scene(**item))
    return scenes


# --- LLM enrichment -----------------------------------------------------------

LANGUAGE_NAMES = {
    "pt": "portugues do Brasil", "en": "ingles", "es": "espanhol", "fr": "frances",
    "de": "alemao", "ja": "japones", "ko": "coreano",
}


def build_enrich_system_prompt(*, translate_scenes_to_english: bool, language: str) -> str:
    """The visual_prompt (fed to the image/video models) reads better in English --
    FLUX/LTX are trained overwhelmingly on English captions -- so that's the default.
    But this must stay a CHOICE, not a hardcoded side effect: dialogue TEXT itself
    (line["text"], what the character actually says) is NEVER touched by this prompt
    or by any other code path in the pipeline -- it flows unmodified from the parsed
    screenplay straight through to TTS and to the "saying: ..." clause embedded in the
    render prompt. Only the CAMERA/SCENE description language is a knob here.
    line_emotions stays in English regardless (short internal tag consumed only by the
    Qwen TTS 'instruct' field, never shown/spoken)."""
    if translate_scenes_to_english:
        visual_lang_instruction = "1-2 frases em ingles"
    else:
        lang_name = LANGUAGE_NAMES.get(language, language)
        visual_lang_instruction = f"1-2 frases em {lang_name} (NAO traduza para ingles)"
    line_visual_lang = "em ingles" if translate_scenes_to_english else f"em {LANGUAGE_NAMES.get(language, language)}"
    return (
        "Voce e um assistente de pre-producao cinematografica. Para a cena de roteiro "
        "fornecida, responda APENAS com um objeto JSON valido, sem markdown, no formato: "
        '{"visual_prompt": "...", "location": "...", "time_of_day": "...", '
        '"art_direction": "...", '
        '"line_visuals": {"0": "...", "1": "..."}, '
        '"shot_list": [{"action": "..."}, {"line": 0}, {"action": "..."}, {"line": 1}], '
        '"line_emotions": {"0": "...", "2": "..."}}. '
        f"visual_prompt: {visual_lang_instruction}, estilo cinematografico, descrevendo o que a "
        "camera ve no INICIO desta cena (local, iluminacao, atmosfera, personagens presentes). "
        "NUNCA traduza ou parafraseie as falas dos personagens -- este campo descreve so "
        "o que a camera ve, nao repete dialogo. "
        "location: o LUGAR da cena, em ingles, curto e nomeavel (ex: 'city bus stop', "
        "'imperial bedchamber'). E a IDENTIDADE do local -- serve para reconhecer que "
        "duas cenas se passam no mesmo lugar -- entao nao descreva acao nem personagens "
        "aqui. time_of_day: periodo e clima numa expressao curta em ingles (ex: "
        "'day, raining', 'night'). Preencha os dois SEMPRE, inclusive quando o roteiro "
        "nao traz cabecalho de cena. "
        "art_direction: o MEIO e o acabamento visual da obra, em ingles, como uma "
        "lista curta de termos de prompt de imagem (ex: 'polished hand-drawn cel "
        "animation, sharp ink lines, dramatic shadows'; '3D storybook animation, "
        "pastel colors, soft rim light'; 'photorealistic live action, 35mm film "
        "grain'). Se o texto disser em que MEIO a obra e feita -- anime, animacao "
        "3D, live action, aquarela, stop motion --, repita esse meio aqui; se nao "
        "disser nada sobre isso, devolva string vazia. NUNCA descreva a cena nem "
        "os personagens neste campo. "
        "line_visuals: OBRIGATORIO um item para CADA indice de fala listado em "
        "'Todas as falas EM ORDEM' (0-based, sem pular nenhum) -- uma frase curta "
        f"{line_visual_lang}, PRESENTE, descrevendo apenas o que muda ou acontece "
        "visualmente NESTE momento especifico (pose, gesto, movimento, expressao, o que o "
        "personagem esta fazendo enquanto fala). Cada item deve continuar naturalmente do "
        "anterior -- e uma sequencia de acao continua, nao repita a descricao da cena "
        "inteira em cada item, e nao invente eventos que nao estao no texto de acao "
        "fornecido nem mude a ordem dos eventos. "
        "shot_list: a DECUPAGEM da cena -- a lista dos planos NA ORDEM EXATA EM QUE "
        "APARECEM NA TELA, do primeiro ao ultimo, misturando acao e dialogo. "
        "Use {\"line\": N} para o plano em que a fala de indice N e dita (N e o indice "
        "0-based da lista 'Todas as falas EM ORDEM'), e "
        f"{{\"action\": \"...\"}} para um momento SEM ninguem falando, com uma frase curta "
        f"{line_visual_lang} no presente descrevendo o que a camera ve (ex: a chuva na rua, "
        "o onibus chegando, a porta abrindo, alguem entrando, o veiculo partindo). "
        "REGRAS OBRIGATORIAS: (a) siga a ordem cronologica do texto de acao -- o que "
        "acontece antes no texto vem antes na lista; (b) TODAS as falas devem aparecer, "
        "exatamente uma vez cada, e na ordem crescente de indice (0, depois 1, depois 2...); "
        "(c) os momentos de acao entram entre as falas conforme a ordem do texto; "
        "(d) extraia os momentos de acao APENAS do texto fornecido, nao invente. "
        "line_emotions: para cada indice de fala LISTADO em 'Falas sem rubrica explicita' "
        "(0-based, apenas essas), uma palavra ou frase curta em ingles descrevendo o tom "
        "emocional da entrega (ex: 'angry, raised voice', 'whispering, afraid'). "
        "Nao invente falas novas nem personagens novos."
    )


def _load_gemma():
    """Lazy import + load Gemma 3 12B IT as a general causal LM.

    MEASURED, not assumed: device_map="auto" across both GPUs was tried first and
    was catastrophically slow (~5-8 minutes per scene, ~300 output tokens) because
    this host's two GPUs have no P2P peer access (confirmed via tensorxx_ge.diagnose
    -- "peer_access": false) -- every one of Gemma3's 48 layers alternates GPU0/GPU1,
    so pipeline-parallel decoding relays through host memory on EVERY layer boundary
    for EVERY generated token. Loading 8-bit-quantized onto a single GPU (~12-13GiB
    for a 12B model, fits on either card) avoids that relay entirely. This stage runs
    before any LTX/TensorRT model is resident (sequential-stage concurrency model),
    so a single GPU is free to dedicate to it.
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    # GPU 0 (torch numbering) is the RTX 4070 (12GiB) -- too tight for an 8-bit
    # ~12-13GiB model plus quantization scratch memory (confirmed by OOM). GPU 1 is
    # the RTX 3090 (24GiB). device_map alone was NOT enough to keep bitsandbytes'
    # int8 quantization scratch buffers off GPU 0 (confirmed by OOM naming GPU 0's
    # 11.99GiB capacity even with device_map={"": 1}) -- torch.cuda.set_device must
    # also be set so torch.cuda.current_device() (which some bnb ops fall back to)
    # agrees.
    torch.cuda.set_device(1)
    gemma_root = str(ROOT / "models" / "gemma3")
    processor = AutoProcessor.from_pretrained(gemma_root)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_root, quantization_config=quant_config, device_map={"": 1},
    )
    model.eval()
    return model, processor


def _ask_gemma(model, processor, user_prompt: str, *, system_prompt: str, max_new_tokens: int = 300, log=print) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        # MEASURED (2026-08-09), two rounds: greedy decoding (do_sample=False) against
        # this model's own generation_config.json (which recommends do_sample=true,
        # top_p=0.95, top_k=64) reproducibly degenerated into multilingual token-salad
        # on a dense scene (10 dialogue lines). Switching to that recommended sampling
        # config fixes the salad but then reproducibly hits "CUDA error: device-side
        # assert triggered" inside torch.multinomial on the SAME scene (2/2) -- and
        # once that fires, the CUDA context is poisoned for the rest of the process:
        # even a same-call greedy fallback, and later torch.cuda.empty_cache(), raise
        # the identical stale assertion. There is no in-process recovery from that, so
        # a same-call fallback is pointless -- greedy is kept as the sole strategy here
        # because it never crashes the context, only sometimes produces bad JSON, which
        # _extract_json()/the caller's try/except already degrade gracefully from.
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction: the model is asked for raw JSON but may still
    wrap it in a code fence or add stray prose.

    MEASURED (2026-08-09): naive first-"{"-to-last-"}" extraction breaks when greedy
    decoding runs past the JSON's natural end and degenerates into rambling (this
    8-bit-quantized Gemma3 setup doesn't reliably emit EOS right after closing the
    object) -- any stray "}" in that trailing garbage got included, producing an
    unparseable blob even though a perfectly valid JSON object was generated first.
    Track brace depth from the first "{" (string-aware, so a literal brace inside a
    quoted value doesn't confuse the count) and stop at the FIRST balanced close --
    ignores everything the model generates afterward, garbage or not."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _build_scene_user_prompt(scene: Scene) -> tuple[str, list[tuple[int, str, str]]]:
    """Shared by both enrichment engines (in-process Gemma3, subprocess Gemma4) so
    the two never quietly drift into asking the model a differently-shaped question.

    ALL dialogue lines are listed IN ORDER (not just the ones missing a
    parenthetical, unlike the older emotion-only list) -- line_visuals needs the
    full in-order sequence to write a per-beat description that actually continues
    from one line to the next, which is the whole point of asking for it (see
    DialogueLine.beat_visual's docstring)."""
    lines_needing_emotion = [
        (i, line.character, line.text)
        for i, line in enumerate(scene.dialogue)
        if not line.parenthetical
    ]
    all_lines = [(i, line.character, line.text) for i, line in enumerate(scene.dialogue)]
    emotion_block = "\n".join(f"{i}: {name}: {text}" for i, name, text in lines_needing_emotion) or "(nenhuma)"
    all_lines_block = "\n".join(f"{i}: {name}: {text}" for i, name, text in all_lines) or "(nenhuma)"
    user_prompt = (
        f"Cabecalho: {scene.heading_raw}\n"
        f"Local: {scene.location}  Periodo: {scene.time_of_day}\n"
        f"Acao: {scene.action_text or '(nenhuma)'}\n"
        f"Personagens presentes: {', '.join(scene.characters) or '(nenhum)'}\n"
        f"Todas as falas EM ORDEM (indice: personagem: texto):\n{all_lines_block}\n"
        f"Falas sem rubrica explicita, para line_emotions (indice: personagem: texto):\n{emotion_block}"
    )
    return user_prompt, lines_needing_emotion


ESTILO_VISUAL_RE = re.compile(r"^[ \t]*ESTILO VISUAL[ \t]*:[ \t]*(.+?)[ \t]*$",
                              re.IGNORECASE | re.MULTILINE)


def extract_art_direction(text: str) -> tuple[str, str]:
    """Tira a linha "ESTILO VISUAL: ..." do texto e devolve (texto_limpo, meio).

    Quem escreve essa linha e o `prose_to_screenplay`, quando o texto de origem
    diz em que MEIO a obra e feita (anime, animacao 3D, live action). O meio nao
    cabe em nenhum campo do formato de roteiro -- nao e cabecalho, nao e acao,
    nao e fala -- e mesmo assim vale para o filme inteiro, entao vira um campo
    proprio em vez de sujeira dentro do texto de acao.

    A linha SAI do texto antes do parse: deixada la, ela viraria linha de acao e
    a descricao do acabamento entraria no prompt de VIDEO de algum plano, onde
    nao tem o que fazer.

    Deterministico de proposito. O modelo tambem devolve `art_direction` no
    enriquecimento, mas isto aqui e leitura de um dado que ja existe -- e por
    isso ganha dele, pela mesma regra que faz um cabecalho de cena real ganhar
    do palpite do modelo em _apply_setting."""
    m = ESTILO_VISUAL_RE.search(text or "")
    if not m:
        return text, ""
    return (text[:m.start()] + text[m.end():]).lstrip(), m.group(1).strip()[:200]


def _apply_setting(scene: Scene, payload: dict) -> None:
    """Preenche location/time_of_day vindos do enriquecimento -- SO quando o
    parse estrutural nao os achou.

    MEASURED (2026-08-27): a screenplay written as running prose has no scene
    heading, so _extract_freeform_scene stores location="" and time_of_day=""
    -- there is nowhere to read them from -- and nothing filled them in later.
    Two consumers broke on that, both silently:

      - shot_plan builds every storyboard_prompt without a setting. The
        ESTABLISHING shot of the bus-stop scene came out as "extreme wide
        establishing shot, the figure small within a vast frame. natural
        cinematic lighting." -- no bus stop, no rain, no daylight. Since the
        still becomes frame 0 of the clip (see MEMORIAL 3.24), the whole film
        was being born in the wrong place.
      - story_structure reported the location as "local_sem_nome", and without a
        location identity there is no spatial continuity between scenes to
        reason about at all.

    A real slugline WINS over the model guess: whoever wrote "INT. TORRE -
    NOITE" said what they meant, and overwriting that would trade data for
    inference."""
    for campo in ("location", "time_of_day"):
        if getattr(scene, campo):
            continue
        valor = payload.get(campo)
        if isinstance(valor, str) and valor.strip():
            setattr(scene, campo, valor.strip()[:120])
    # Mesma regra do cabecalho de cena: DADO LIDO GANHA DE PALPITE. Se a linha
    # "ESTILO VISUAL:" existia no texto, extract_art_direction ja preencheu isto
    # antes do enriquecimento, e o modelo nao sobrescreve.
    #
    # MEDIDO 2026-08-27, sem esta guarda: o texto dizia "polished hand-drawn cel
    # animation, sharp ink lines" e o enriquecimento devolveu "3D cinematic
    # animation, high contrast lighting" -- o palpite contradizia a fonte e ia
    # para o prompt. Desenho a mao e render 3D nao sao a mesma obra.
    if not scene.art_direction:
        arte = payload.get("art_direction")
        if isinstance(arte, str) and arte.strip():
            scene.art_direction = arte.strip()[:200]


def _apply_shot_list(scene: Scene, payload: dict, *, log=print) -> None:
    """Validate and store payload["shot_list"] as scene.shot_list. Shared by both
    enrichment engines.

    Defensive on purpose: this is free-form LLM output feeding a render loop. The
    invariant that MUST hold is that every dialogue line is rendered exactly once and
    in script order -- a dropped line loses spoken story, a duplicated one renders the
    same speech twice, and a reordered one is what produced the scrambled cut this
    field was introduced to fix (see Scene.shot_list). So: unknown/duplicate/
    out-of-range line references are discarded, and any line the model forgot is
    appended at the end rather than silently lost."""
    raw = payload.get("shot_list") or []
    if not isinstance(raw, list) or not raw:
        log(f"[parse_screenplay] cena {scene.index}: sem shot_list utilizavel; usando apenas as falas, em ordem.")
        scene.shot_list = [{"type": "dialogue", "line_index": i} for i in range(len(scene.dialogue))]
        return

    shots: list[dict] = []
    seen_lines: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "line" in item:
            try:
                line_index = int(item["line"])
            except (TypeError, ValueError):
                continue
            if not (0 <= line_index < len(scene.dialogue)) or line_index in seen_lines:
                continue  # out of range, or the model repeated a line
            seen_lines.add(line_index)
            shots.append({"type": "dialogue", "line_index": line_index})
        else:
            visual = str(item.get("action") or "").strip()
            if visual:
                shots.append({"type": "action", "visual": visual})

    missing = [i for i in range(len(scene.dialogue)) if i not in seen_lines]
    for line_index in missing:
        shots.append({"type": "dialogue", "line_index": line_index})
    if missing:
        log(f"[parse_screenplay] cena {scene.index}: {len(missing)} fala(s) ausente(s) da shot_list; anexada(s) ao final.")

    # The model is asked for ascending line order; if it disobeyed, that's exactly the
    # scrambling bug -- report it rather than shipping a jumbled cut silently.
    spoken = [s["line_index"] for s in shots if s["type"] == "dialogue"]
    if spoken != sorted(spoken):
        log(f"[parse_screenplay] cena {scene.index}: AVISO -- falas fora de ordem na shot_list ({spoken}); reordenando.")
        actions_before: dict[int, list[str]] = {}
        pending: list[str] = []
        for shot in shots:
            if shot["type"] == "action":
                pending.append(shot["visual"])
            else:
                actions_before.setdefault(shot["line_index"], []).extend(pending)
                pending = []
        rebuilt: list[dict] = []
        for line_index in sorted(spoken):
            for visual in actions_before.get(line_index, []):
                rebuilt.append({"type": "action", "visual": visual})
            rebuilt.append({"type": "dialogue", "line_index": line_index})
        rebuilt.extend({"type": "action", "visual": v} for v in pending)
        shots = rebuilt

    scene.shot_list = shots
    n_action = sum(1 for s in shots if s["type"] == "action")
    log(f"[parse_screenplay] cena {scene.index}: decupagem com {len(shots)} plano(s) "
        f"({n_action} de acao + {len(shots) - n_action} de fala).")


def enrich_scene(model, processor, scene: Scene, *, system_prompt: str, log=print) -> Scene:
    user_prompt, lines_needing_emotion = _build_scene_user_prompt(scene)
    # MEASURED (2026-08-09): the fixed 300-token default silently truncated the JSON
    # response before its closing brace on a dense scene (10 dialogue lines all
    # needing an emotion tag). Scaling this up too aggressively backfired the other
    # way: at 620 tokens this 8-bit-quantized Gemma3 setup degraded into multilingual
    # token-salad well past where the JSON itself would have ended (greedy decoding
    # with no reliable EOS). Kept moderate -- enough headroom for a 10-line scene's
    # JSON, not so much that overshoot has room to spiral. _extract_json() now stops
    # at the first balanced "}" regardless, so any trailing rambling within budget is
    # harmless either way -- this cap is about wasted generation time, not correctness.
    # line_visuals now asks for one item per dialogue line (not just the
    # emotion-only subset), so the budget must scale with len(scene.dialogue), not
    # just len(lines_needing_emotion).
    max_new_tokens = min(1000, 300 + 55 * len(scene.dialogue))
    raw = _ask_gemma(model, processor, user_prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens, log=log)
    payload = _extract_json(raw)
    if payload is None:
        log(f"[parse_screenplay] cena {scene.index}: resposta do LLM nao era JSON valido "
            f"(sem visual_prompt/beats/emocoes -- fallback para prompt estrutural); resposta bruta: {raw[:200]!r}")
        scene.visual_prompt = None
        return scene
    scene.visual_prompt = payload.get("visual_prompt")
    _apply_setting(scene, payload)
    line_visuals = payload.get("line_visuals") or {}
    for i, line in enumerate(scene.dialogue):
        beat = line_visuals.get(str(i))
        if beat:
            line.beat_visual = str(beat)
    _apply_shot_list(scene, payload, log=log)
    line_emotions = payload.get("line_emotions") or {}
    for i, _name, _text in lines_needing_emotion:
        emotion = line_emotions.get(str(i))
        if emotion:
            scene.dialogue[i].emotion = str(emotion)
    return scene


def enrich_with_llm(
    scenes: list[Scene], *, log=print, translate_scenes_to_english: bool = True, language: str = "pt",
) -> list[Scene]:
    system_prompt = build_enrich_system_prompt(
        translate_scenes_to_english=translate_scenes_to_english, language=language,
    )
    model, processor = _load_gemma()
    for scene in scenes:
        log(f"[parse_screenplay] enriquecendo cena {scene.index}/{len(scenes)}...")
        try:
            enrich_scene(model, processor, scene, system_prompt=system_prompt, log=log)
        except Exception as exc:  # noqa: BLE001
            # Enrichment is best-effort per scene (structural parse already has
            # everything downstream needs) -- never let one scene's LLM failure
            # (including a poisoned CUDA context after a device-side assert) abort
            # the whole parse stage and take the rest of the screenplay down with it.
            log(f"[parse_screenplay] cena {scene.index}: enriquecimento LLM falhou ({exc.__class__.__name__}: {exc}); seguindo sem visual_prompt/emocoes para esta cena.")
            scene.visual_prompt = None
    del model
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        # A poisoned CUDA context (see the try/except above) makes even cleanup calls
        # raise the same stale assertion -- harmless to skip, the process is exiting
        # right after this function returns anyway.
        log(f"[parse_screenplay] torch.cuda.empty_cache() falhou ({exc.__class__.__name__}), ignorando.")
    return scenes


GEMMA4_ENV_PYTHON = str(ROOT / "gemma4_env" / "Scripts" / "python.exe")
GEMMA4_WORKER = str(ROOT / "script_pipeline" / "llm_workers" / "gemma4_worker.py")


def enrich_with_llm_gemma4(
    scenes: list[Scene], *, log=print, translate_scenes_to_english: bool = True, language: str = "pt",
) -> list[Scene]:
    """Same enrichment contract/output as enrich_with_llm() (in-process Gemma3), but
    dispatches generation to gemma4_worker.py in the isolated gemma4_env venv -- see
    that module's docstring for why Gemma 4 can't share the main venv's transformers.
    One subprocess call for the WHOLE screenplay (not one per scene) -- the worker
    loads the model once and loops jobs itself, same batching discipline as
    tts_workers/qwen_worker.py."""
    import subprocess
    import tempfile

    system_prompt = build_enrich_system_prompt(
        translate_scenes_to_english=translate_scenes_to_english, language=language,
    )
    jobs = []
    scene_by_id: dict[str, tuple[Scene, list]] = {}
    for scene in scenes:
        user_prompt, lines_needing_emotion = _build_scene_user_prompt(scene)
        job_id = str(scene.index)
        jobs.append({
            "id": job_id, "system_prompt": system_prompt, "user_prompt": user_prompt,
            # line_visuals covers every dialogue line, not just the emotion-only
            # subset -- budget must scale with the full line count (see enrich_scene's
            # matching comment on the Gemma3 path for why this stays capped, not open-ended).
            "max_new_tokens": min(1000, 300 + 55 * len(scene.dialogue)),
        })
        scene_by_id[job_id] = (scene, lines_needing_emotion)

    with tempfile.TemporaryDirectory() as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        results_path = Path(tmp) / "results.json"
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [GEMMA4_ENV_PYTHON, "-u", GEMMA4_WORKER, "--jobs", str(jobs_path), "--results", str(results_path)]
        log(f"[parse_screenplay] enriquecendo {len(scenes)} cena(s) via Gemma4 (venv isolado)...")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            log(line.rstrip("\n"))
        process.wait()
        if not results_path.exists():
            log("[parse_screenplay] gemma4_worker nao produziu results.json (subprocesso pode ter crashado); seguindo sem enriquecimento em nenhuma cena.")
            for scene in scenes:
                scene.visual_prompt = None
            return scenes
        results = json.loads(results_path.read_text(encoding="utf-8"))

    for result in results:
        scene, lines_needing_emotion = scene_by_id.get(result["id"], (None, None))
        if scene is None:
            continue
        if not result["ok"]:
            log(f"[parse_screenplay] cena {scene.index}: gemma4 falhou ({result.get('error')}); seguindo sem visual_prompt/emocoes.")
            scene.visual_prompt = None
            continue
        payload = _extract_json(result["raw_text"] or "")
        if payload is None:
            log(f"[parse_screenplay] cena {scene.index}: resposta do Gemma4 nao era JSON valido; resposta bruta: {(result['raw_text'] or '')[:200]!r}")
            scene.visual_prompt = None
            continue
        scene.visual_prompt = payload.get("visual_prompt")
        _apply_setting(scene, payload)
        line_visuals = payload.get("line_visuals") or {}
        for i, line in enumerate(scene.dialogue):
            beat = line_visuals.get(str(i))
            if beat:
                line.beat_visual = str(beat)
        _apply_shot_list(scene, payload, log=log)
        line_emotions = payload.get("line_emotions") or {}
        for i, _name, _text in lines_needing_emotion:
            emotion = line_emotions.get(str(i))
            if emotion:
                scene.dialogue[i].emotion = str(emotion)
    return scenes


def enrich_with_llm_ollama(
    scenes: list[Scene], *, model: str, log=print,
    translate_scenes_to_english: bool = True, language: str = "pt",
) -> list[Scene]:
    """Same enrichment contract as the Gemma paths, served by Ollama.

    Added because gemma4-e2b is a ~2B effective-parameter model and MEASURED
    worse on the neighbouring extraction task: 2/4 dialogue lines preserved in
    91s, against 4/4 in 6s for qwen3.6-35b through Ollama. Enrichment feeds the
    storyboard prompts directly (visual_prompt / beat_visual), and gemma4 left
    beat_visual empty on half the lines, so the same gap costs picture quality
    here.

    `think: False` matters: reasoning models put chain-of-thought in a separate
    field and can spend the whole token budget there, returning EMPTY content --
    a silent failure that reads as a refusal (measured with qwen3.6).
    """
    import urllib.error
    import urllib.request

    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    system_prompt = build_enrich_system_prompt(
        translate_scenes_to_english=translate_scenes_to_english, language=language,
    )
    scene_by_id: dict[str, tuple[Scene, list]] = {}
    results = []
    log(f"[parse_screenplay] enriquecendo {len(scenes)} cena(s) via Ollama ({model})...")
    for scene in scenes:
        user_prompt, lines_needing_emotion = _build_scene_user_prompt(scene)
        job_id = str(scene.index)
        scene_by_id[job_id] = (scene, lines_needing_emotion)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "think": False,
            "options": {"num_predict": min(1500, 400 + 60 * len(scene.dialogue)),
                        "temperature": 0},
        }
        req = urllib.request.Request(
            f"{ollama_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.load(r)
            raw = (data.get("message") or {}).get("content", "")
            results.append({"id": job_id, "ok": True, "raw_text": raw})
        except Exception as e:
            log(f"[parse_screenplay] cena {scene.index}: Ollama falhou ({e}).")
            results.append({"id": job_id, "ok": False, "error": str(e), "raw_text": ""})

    for result in results:
        scene, lines_needing_emotion = scene_by_id.get(result["id"], (None, None))
        if scene is None:
            continue
        payload = _extract_json(result["raw_text"] or "") if result["ok"] else None
        if payload is None:
            log(f"[parse_screenplay] cena {scene.index}: sem JSON valido do Ollama; "
                "seguindo sem visual_prompt/emocoes.")
            scene.visual_prompt = None
            continue
        scene.visual_prompt = payload.get("visual_prompt")
        _apply_setting(scene, payload)
        for i, line in enumerate(scene.dialogue):
            beat = (payload.get("line_visuals") or {}).get(str(i))
            if beat:
                line.beat_visual = str(beat)
        _apply_shot_list(scene, payload, log=log)
        for i, _name, _text in lines_needing_emotion:
            emotion = (payload.get("line_emotions") or {}).get(str(i))
            if emotion:
                scene.dialogue[i].emotion = str(emotion)
    return scenes


# --- CLI ------------------------------------------------------------------------

def main(argv=None) -> int:
    # torch/transformers progress bars and generated PT-BR text can include
    # characters the default Windows cp1252 console can't encode; same fix already
    # used across this project's other entry points (e.g. start_tensorRT_*.bat set
    # PYTHONIOENCODING=utf-8 for the same reason).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="Path to the screenplay text file.")
    parser.add_argument("--run-dir", required=True, help="Run folder created by run_folder.create_run().")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment; write structure only.")
    parser.add_argument("--no-auto-structure", action="store_true",
                        help="Do not try to reformat prose/LTX-prompt input into screenplay format when no "
                             "INT./EXT. heading is found. Structure then stays 100%% deterministic, and "
                             "unformatted input simply fails as before.")
    parser.add_argument("--language", default="pt", choices=list(LANGUAGE_NAMES),
                         help="Screenplay/voice language (default pt = Brazilian Portuguese). Dialogue text is NEVER translated regardless of this or --translate-scenes-en.")
    parser.add_argument("--translate-scenes-en", dest="translate_scenes_en", action="store_true", default=True,
                         help="Write visual_prompt (scene/camera description fed to the image/video models) in English -- default on, FLUX/LTX read English prompts better.")
    parser.add_argument("--no-translate-scenes-en", dest="translate_scenes_en", action="store_false",
                         help="Keep visual_prompt in --language instead of translating to English.")
    parser.add_argument("--enrich-engine", default="gemma3",
                         help="gemma3: in-process, main venv (transformers 4.x). gemma4: subprocess in the "
                              "isolated gemma4_env venv (transformers 5.x, native bf16 E2B -- no quantization, "
                              "avoids the CUDA device-side-assert-under-sampling issue measured with Gemma3 8-bit).")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    text = Path(args.script).read_text(encoding="utf-8")
    # Um roteiro ja formatado tambem pode trazer a linha; tirada aqui, ela nao
    # vira linha de acao em cena nenhuma.
    text, arte_do_texto = extract_art_direction(text)
    scenes = parse_structure(text)
    parse_dir = run_folder.subdir(run_dir, "parse")
    if not scenes and not args.no_auto_structure and not args.no_llm:
        # The deterministic pass found no scene headings at all -- typically the
        # input is prose, a treatment, or an LTX audiovisual prompt rather than a
        # formatted screenplay. Convert it once, WRITE IT DOWN so the user can read
        # and edit what the model produced, then re-run the same deterministic
        # parser on the result. Structure still comes from the regex pass; the LLM
        # only supplies formatting. See prose_to_screenplay's docstring.
        from script_pipeline import prose_to_screenplay

        print("[parse_screenplay] nenhum cabecalho INT./EXT. encontrado; "
              "tentando reestruturar o texto para formato de roteiro...", file=sys.stderr)
        converted, missing = prose_to_screenplay.convert(
            text, engine=args.enrich_engine, log=lambda m: print(m, file=sys.stderr),
        )
        if converted:
            converted, arte_convertida = extract_art_direction(converted)
            if arte_convertida:
                arte_do_texto = arte_convertida
                print(f"[parse_screenplay] direcao de arte: {arte_convertida}", file=sys.stderr)
            auto_path = parse_dir / "screenplay_auto.txt"
            auto_path.write_text(converted, encoding="utf-8")
            print(f"[parse_screenplay] roteiro reestruturado salvo em {auto_path}", file=sys.stderr)
            run_folder.append_log(
                run_dir,
                f"parse_screenplay: texto de entrada reestruturado automaticamente para formato de "
                f"roteiro ({auto_path.name})."
                + (f" ATENCAO: {len(missing)} fala(s) nao sobreviveram literalmente." if missing else ""),
            )
            scenes = parse_structure(converted)
            if scenes:
                print(f"[parse_screenplay] reestruturacao OK: {len(scenes)} cena(s) detectada(s).",
                      file=sys.stderr)

    if not scenes:
        print("Nenhuma cena detectada -- confira se o roteiro tem cabecalhos INT./EXT. "
              "(a reestruturacao automatica tambem nao produziu cenas; use --no-auto-structure "
              "para desligar essa tentativa).", file=sys.stderr)
        return 1

    # O meio vale para o filme inteiro, entao entra em TODA cena. Fica antes do
    # enriquecimento de proposito: _apply_setting so preenche art_direction que
    # ainda esteja vazio, entao o dado lido ganha do palpite do modelo.
    if arte_do_texto:
        for cena in scenes:
            cena.art_direction = arte_do_texto

    structural_path = parse_dir / "scenes.json"
    structural_path.write_text(
        json.dumps(scenes_to_json(scenes), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    run_folder.append_log(
        run_dir,
        f"parse_screenplay: {len(scenes)} cena(s), "
        f"{sum(len(s.dialogue) for s in scenes)} fala(s) detectada(s).",
    )

    if not args.no_llm:
        # Any engine name that is not gemma3/gemma4 is taken as an Ollama model tag.
        if args.enrich_engine == "gemma4":
            enrich_fn = enrich_with_llm_gemma4
        elif args.enrich_engine == "gemma3":
            enrich_fn = enrich_with_llm
        else:
            enrich_fn = functools.partial(enrich_with_llm_ollama, model=args.enrich_engine)
        scenes = enrich_fn(
            scenes, log=lambda msg: run_folder.append_log(run_dir, msg),
            translate_scenes_to_english=args.translate_scenes_en, language=args.language,
        )
        enriched_path = parse_dir / "scenes_enriched.json"
        enriched_path.write_text(
            json.dumps(scenes_to_json(scenes), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        run_folder.append_log(run_dir, f"parse_screenplay: enriquecimento LLM salvo em {enriched_path}")

    run_folder.mark_stage_complete(run_dir, "parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
