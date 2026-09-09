"""Character registry: one entry per character, with a visual descriptor (for
storyboard/scene prompts) and a default voice assignment (for TTS) -- written as an
editable ``cast.json`` the user can hand-tune before rendering.

Deterministic by default (no model load): descriptor is a truncated snippet of the
first action line mentioning the character; voice is a round-robin assignment over
whatever reference speaker clips exist in the xtts install. ``--llm`` upgrades the
descriptor via a *single* batched Gemma 3 call covering all characters at once (not
one load per character -- loading Gemma 3 is the expensive part, not the generation).

CLI: ``python -m script_pipeline.cast_characters --run-dir DIR [--llm]``
Reads ``<run-dir>/parse/scenes_enriched.json`` (falls back to ``scenes.json``),
writes ``<run-dir>/characters/cast.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

XTTS_SPEAKER_DIR = Path(r"E:\Users\home\Documents\xtts\webui\speakers")

# MEDIDO 2026-08-27: o formato anunciado era {"descriptors": {"NOME": "..."}} e o
# qwen3.6 devolveu literalmente a chave "NOME", com os DOIS personagens espremidos
# numa string so. Nada falhava -- `descriptors.get("AKEMI")` so nao achava nada, e
# o casting caia no descritor deterministico em silencio. Placeholder dentro do
# exemplo de formato e ambiguo: o modelo nao tem como saber que "NOME" era para
# ser substituido. As chaves exigidas agora vao explicitas na mensagem do usuario
# (ver _enrich_descriptors_ollama), e aqui a regra e dita em vez de insinuada.
CAST_SYSTEM_PROMPT = (
    "Voce e um assistente de casting cinematografico. Responda APENAS com um objeto "
    'JSON valido no formato {"descriptors": {...}}. '
    "As CHAVES de 'descriptors' devem ser EXATAMENTE os nomes de personagem listados "
    "na mensagem do usuario, copiados letra por letra, um item por personagem -- nunca "
    "a palavra NOME, nunca um nome traduzido ou abreviado, nunca dois personagens no "
    "mesmo item. O VALOR de cada item e um descritor visual CONCRETO (2-3 frases "
    "curtas, em ingles, estilo prompt de imagem) baseado nos trechos de acao "
    "fornecidos, cobrindo SEMPRE estas quatro categorias, cada uma com um detalhe "
    "ESPECIFICO e nunca vago: (1) cabelo -- cor exata, comprimento, estilo/corte; "
    "(2) rosto/porte -- idade aproximada, formato do rosto ou tracos marcantes, "
    "compleicao fisica; (3) roupa -- cor exata, material/textura, cada peca visivel "
    "(nao so 'roupas escuras', diga 'dark-blue wool cloak with a frayed hem'); "
    "(4) um item ou marca distintiva unica (cicatriz, joia, arma, acessorio) que "
    "nenhum outro personagem da cena tenha. Frases como 'a young woman' ou "
    "'simple clothes' sozinhas sao inaceitaveis -- sao desejo, nao ancora visual: "
    "cada categoria PRECISA de um adjetivo ou substantivo concreto que sobreviva "
    "sozinho fora de contexto. Descreva SO como a pessoa e; nao conte o que ela faz. "
    "A ROUPA (categoria 3) tem que ser coerente com o PAPEL e o CENARIO que os "
    "trechos de acao descrevem -- se o personagem e chamado de aluno/estudante e a "
    "cena se passa numa escola/colegio, vista uniforme escolar (ou traje "
    "claramente de estudante daquele lugar), nao roupa de rua generica; se e "
    "medico, vista jaleco; e assim por diante. So fuja da roupa \"esperada\" se o "
    "texto disser explicitamente o contrario. "
    "Se nao houver informacao suficiente, invente algo plausivel e ESPECIFICO "
    "(nunca generico), mas nunca deixe vazio."
)


# AUDITORIA DE COMPLETUDE DO DESCRITOR -- pergunta feita pelo usuario 2026-09-07
# ("o script faz auditoria do que ira produzir?"): ate aqui, nao -- o
# `_default_descriptor` promete cobrir 4 categorias (cabelo, rosto/porte,
# roupa, item unico) mas nada verificava se o LLM realmente cobriu, e o
# Min-jae da corrida 20260907_ltx_distilled saiu sem NENHUMA palavra de roupa
# de uniforme/estudante -- so "moletom cinza largo, calca cargo preta,
# cadarco vermelho", plausivel para um adolescente generico, mas contradiz o
# proprio roteiro ("novo aluno"). Heuristica de palavra-chave, nao prova --
# mas pega exatamente esse caso antes do descritor virar prompt de imagem.
# Bilingue: o CAST_SYSTEM_PROMPT pede descritor em ingles, mas MEDIDO
# 2026-09-07 (cast.json de 20260907_ltx_distilled) o Ollama devolveu em
# portugues mesmo assim -- so checar palavra em ingles deixaria passar
# descritores em pt-BR completos como se estivessem vazios.
_DESCRIPTOR_CATEGORY_HINTS = {
    "cabelo": ("hair", "cabelo", "bald", "careca", "braid", "trança", "cabelos"),
    "rosto_porte": ("face", "build", "complexion", "skin", "eyes", "cheek", "jaw",
                     "shoulders", "frame", "-year-old", "years old", "age", "tall",
                     "short", "slim", "stocky", "rosto", "olhos", "compleição",
                     "anos", "magro", "alto", "baixa", "porte"),
    "roupa": ("wearing", "shirt", "jacket", "dress", "uniform", "pants", "trousers",
              "skirt", "coat", "sweater", "hoodie", "sleeve", "collar", "fabric",
              "cloth", "blazer", "vest", "tie", "veste", "vestindo", "uniforme",
              "camisa", "jaqueta", "calça", "saia", "casaco", "suéter", "moletom",
              "blusa", "gravata"),
    "item_unico": ("necklace", "scar", "ring", "bracelet", "glasses", "tattoo",
                   "badge", "pin", "earring", "watch", "bag", "backpack", "pendant",
                   "locket", "cane", "staff", "colar", "cicatriz", "anel", "pulseira",
                   "óculos", "tatuagem", "broche", "brinco", "relógio", "mochila",
                   "pingente", "bolsa"),
}


def _check_descriptor_completeness(descriptor: str) -> list[str]:
    """Categorias que o `CAST_SYSTEM_PROMPT` exige e que nao tem nenhuma
    palavra-chave reconhecivel no descritor final."""
    text = (descriptor or "").lower()
    return [cat for cat, kws in _DESCRIPTOR_CATEGORY_HINTS.items()
            if not any(kw in text for kw in kws)]


def _audit_and_fix_descriptors(characters: dict, *, model: str | None, log=print) -> dict:
    """Roda a checagem em todo personagem; para quem tem lacuna E tem um motor
    Ollama disponivel, faz UMA segunda chamada pedindo so as categorias que
    faltam (mais barato e mais preciso que regerar o descritor inteiro).
    Sempre grava o resultado em `info["descriptor_gaps"]` -- vazio quando
    completo -- para o cast.json carregar a auditoria consigo, nao so o
    resultado."""
    from script_pipeline.story_structure import _call_ollama

    gaps_found = 0
    for name, info in characters.items():
        gaps = _check_descriptor_completeness(info.get("descriptor", ""))
        if gaps and model:
            faltando_pt = ", ".join(gaps)
            fix_prompt = (
                f"Personagem: {name}\nDescritor atual: {info['descriptor']}\n"
                f"Categorias FALTANDO neste descritor: {faltando_pt}. "
                "Responda APENAS com um objeto JSON {\"descriptors\": {\"" + name +
                "\": \"...\"}} contendo o descritor COMPLETO reescrito (nao so o "
                "trecho novo), agora cobrindo TODAS as 4 categorias exigidas "
                "com detalhe concreto e especifico."
            )
            payload = _call_ollama(CAST_SYSTEM_PROMPT, fix_prompt, model, log=log) or {}
            fixed = _match_descriptor(payload.get("descriptors") or {}, name)
            if fixed:
                info["descriptor"] = fixed
                gaps = _check_descriptor_completeness(fixed)
        if gaps:
            gaps_found += 1
            log(f"[cast_characters] AUDITORIA: {name} ainda sem {', '.join(gaps)} "
                "no descritor -- confira cast.json antes de gerar stills.")
        info["descriptor_gaps"] = gaps
    if gaps_found:
        log(f"[cast_characters] auditoria de descritor: {gaps_found}/{len(characters)} "
            "personagem(ns) com categoria(s) faltando (ver acima).")
    else:
        log(f"[cast_characters] auditoria de descritor: {len(characters)}/{len(characters)} "
            "completos (cabelo, rosto/porte, roupa, item unico).")
    return characters


def _load_scenes(run_dir: Path) -> list[dict]:
    parse_dir = run_dir / "parse"
    enriched = parse_dir / "scenes_enriched.json"
    structural = parse_dir / "scenes.json"
    path = enriched if enriched.exists() else structural
    if not path.exists():
        raise FileNotFoundError(f"No parsed scenes found in {parse_dir} -- run parse_screenplay first.")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_characters(scenes: list[dict]) -> dict:
    """Return {name: {"snippets": [...], "line_count": int}} in first-appearance order."""
    characters: dict = {}
    for scene in scenes:
        for name in scene.get("characters", []):
            characters.setdefault(name, {"snippets": [], "line_count": 0})
        action = scene.get("action_text", "") or ""
        for name in scene.get("characters", []):
            entry = characters[name]
            for sentence in re.split(r"(?<=[.!?])\s+", action):
                if name.title() in sentence or name in sentence:
                    if sentence.strip() and sentence.strip() not in entry["snippets"]:
                        entry["snippets"].append(sentence.strip())
        for line in scene.get("dialogue", []):
            char_entry = characters.setdefault(line["character"], {"snippets": [], "line_count": 0})
            char_entry["line_count"] += 1
            # O PARENTESE DA FALA TAMBEM E MATERIAL DE APARENCIA -- as vezes.
            #
            # Ele e a rubrica de interpretacao ("urgente", "sussurrando"), e na
            # maioria das vezes nao diz nada sobre como a pessoa e. Mas o
            # conversor de prosa erra para ca: MEDIDO 2026-08-27, a descricao da
            # Lyra -- cabelo prateado, capa azul-escura, cajado de cristal --
            # saiu neste parentese em vez da linha de acao, e o casting, sem ver
            # nada, INVENTOU "jaqueta de couro e camiseta de banda".
            #
            # Corrigir isso so no prompt do conversor nao basta: ja tentado, e
            # empurrar a aparencia para a linha de acao fez o modelo omitir a
            # deixa do personagem e a cena perdeu TODAS as falas. Ler dos dois
            # lugares e recuperavel em codigo; depender de obediencia nao e.
            #
            # Uma rubrica curta de emocao entra como ruido pequeno: e um entre
            # ate tres trechos, e o prompt do casting manda descrever so como a
            # pessoa E. Uma aparencia inteira, ao contrario, se perderia.
            par = (line.get("parenthetical") or "").strip()
            if par and len(par) > 25 and par not in char_entry["snippets"]:
                char_entry["snippets"].insert(0, par)
    return characters


# A descriptor is pasted into every storyboard/render prompt, so it must contain only
# what the character LOOKS LIKE. MEASURED (2026-08-10): with a run-on freeform
# screenplay, action_text is one huge block, so the "first sentence mentioning the
# character" came back carrying the character's dialogue verbatim
# ("... ela exclama: - No horario!"). FLUX read that quoted speech as an instruction
# and drew SPEECH BALLOONS with the text into every storyboard -- which then rode into
# every video clip conditioned on it. Cut the descriptor at the first speech marker.
_SPEECH_CUT_RE = re.compile(
    r"\s*[-–]?\s*\b\w*(?:exclam|diz|avis|pergunt|respond|grit|sussurr|fal)\w*(?:-\w+)?\b\s*[:,]?.*$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_descriptor(text: str) -> str:
    """Strip dialogue and quoted speech out of a visual descriptor."""
    text = _SPEECH_CUT_RE.sub("", text)
    text = re.sub(r"[\"“”][^\"“”]*[\"“”]", "", text)  # any leftover quoted speech
    return re.sub(r"\s{2,}", " ", text).strip(" -–,;:")


def _default_descriptor(name: str, snippets: list[str]) -> str:
    if snippets:
        cleaned = _clean_descriptor(snippets[0])[:200]
        if cleaned:
            return cleaned
    return f"{name.title()}, a character in the story; no visual description given yet -- please edit."


def _list_xtts_speakers() -> list[str]:
    if not XTTS_SPEAKER_DIR.is_dir():
        return []
    return sorted(p.stem for p in XTTS_SPEAKER_DIR.glob("*.wav"))


# --- hand-curated voice map -------------------------------------------------------
# The gender heuristic below can only pick from files whose NAME contains "male" or
# "female", so a folder of purpose-recorded character voices (01_lyra_base.wav,
# 02_thoren_base.wav, ...) was MEASURED as 24 of 27 files invisible to it -- the
# pipeline kept round-robinning the three generic clips. mapa_vozes.csv, sitting in
# that same folder, already maps character -> base clip -> style clip -> direction.
# Reading it makes curated voices win over the guess, and is the only place the
# pipeline gets a per-emotion reference clip: XTTS has no textual emotion control
# (that is a Qwen-only feature), so a separate "shouting"/"worried" recording is its
# native way to act a line.
VOICE_MAP_CSV = XTTS_SPEAKER_DIR / "mapa_vozes.csv"


def _normalize_name(name: str) -> str:
    """Fold case, accents and punctuation so 'Princesa Celeste' matches 'PRINCESA CELESTE'."""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def load_voice_map(path: Path = VOICE_MAP_CSV) -> dict:
    """Parse mapa_vozes.csv into {normalized_character: {base, style, direction, source}}.

    Returns {} when the file is absent -- the map is an optional enhancement, never a
    requirement, so a machine without it keeps the old behaviour exactly.
    """
    if not path.exists():
        return {}
    mapping: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                character = (row.get("personagem") or "").strip()
                base = (row.get("arquivo_base") or "").strip()
                if not character or not base:
                    continue
                style = (row.get("arquivo_estilo") or "").strip()
                mapping[_normalize_name(character)] = {
                    "base": Path(base).stem,
                    "style": Path(style).stem if style else None,
                    "direction": (row.get("direcao") or "").strip() or None,
                    "source": (row.get("voz_origem") or "").strip() or None,
                }
    except (OSError, csv.Error):
        return {}
    return mapping


def _emotive_do_interprete(voz_origem: Optional[str], emotive: dict) -> Optional[str]:
    """Pasta de tomadas emotivas do MESMO interprete que o voice_map indica.

    O CSV nomeia quem gravou (`voz_origem` = "Kristin Hughes") e a biblioteca
    emotiva nomeia a pasta pelo primeiro nome (`F02_jovem_expressiva_kristin`).
    Casar os dois devolve ao personagem curado as 17 tomadas que ele estava
    perdendo, SEM trocar o interprete -- e a mesma pessoa, em outra sessao.

    O casamento e EXATO no ultimo componente da pasta, nunca aproximado, e o
    motivo esta no proprio CSV: a Clara tem `voz_origem` "Andi", que e mulher, e
    existe um `M01_jovem_leve_andy`, que e homem. Um casamento por prefixo ou
    por distancia trocaria o genero dela em silencio. Dos 12 do mapa, 9 casam
    exato; os 3 que nao (Andi, voicebynatalie, Peter) seguem como antes."""
    if not voz_origem or not emotive:
        return None
    primeiro = voz_origem.strip().split()[0].lower() if voz_origem.strip() else ""
    if not primeiro:
        return None
    for vid in emotive:
        if vid.rsplit("_", 1)[-1].lower() == primeiro:
            return vid
    return None


def _match_voice(name: str, voice_map: dict) -> Optional[dict]:
    """Exact normalized match first, then a first-token match.

    The screenplay may say "KIM" where the map says "Kim Woo-jin"; matching on the
    first token catches that without matching two different characters who merely
    share a surname.
    """
    key = _normalize_name(name)
    if key in voice_map:
        return voice_map[key]
    first = key.split(" ")[0] if key else ""
    if not first:
        return None
    hits = [v for k, v in voice_map.items() if k.split(" ")[0] == first]
    return hits[0] if len(hits) == 1 else None


# Crude PT/EN gendered-word cues, just to avoid the obviously-wrong default (e.g.
# assigning a female reference voice to a character described as "he"/"o guarda"/
# "homem"). Not linguistically rigorous -- it only has to beat blind round-robin,
# and the whole point of cast.json is that the user can fix any miss by hand.
_MALE_CUES = re.compile(
    r"\b(ele|homem|senhor|garoto|menino|rapaz|mo[cç]o|guarda|pai|filho|irm[aã]o|marido|"
    r"rei|pr[ií]ncipe|he|him|his|man|boy|father|son|brother|husband|king|prince|mr\.?)\b"
    r"|\b(o|os|um|uns) jove(m|ns)\b|\b(o|um) adolescente\b",
    re.IGNORECASE)
# "jovem" and "adolescente" are gender-neutral words in Portuguese -- the article carries
# the gender ("a jovem" / "o jovem"), so the article must be part of the pattern.
# MEASURED: without this, "uma jovem coreana" scored unknown and the alternating
# fallback handed a female character a male voice (flagged as a guess, but still wrong).
_FEMALE_CUES = re.compile(
    r"\b(ela|mulher|senhora|garota|menina|mo[cç]a|dama|m[aã]e|filha|irm[aã]|esposa|"
    r"rainha|princesa|she|her|hers|woman|girl|mother|daughter|sister|wife|queen|princess|"
    r"mrs\.?|ms\.?)\b"
    r"|\b(a|as|uma|umas) jove(m|ns)\b|\b(a|uma) adolescente\b",
    re.IGNORECASE)


def _guess_gender(name: str, descriptor: str) -> Optional[str]:
    text = f"{name} {descriptor}"
    male_hit = bool(_MALE_CUES.search(text))
    female_hit = bool(_FEMALE_CUES.search(text))
    if male_hit and not female_hit:
        return "male"
    if female_hit and not male_hit:
        return "female"
    return None


def assign_voices(characters: dict) -> dict:
    """Default voice assignment: pick an xtts reference clip whose filename matches
    a guessed gender when possible, round-robin within that pool.

    MEASURED (2026-08-09): when gender is unknown (no textual cue -- e.g. a
    romanized non-PT/EN name like "Ji-hoon" with no "ele"/"he" nearby), the old
    fallback round-robinned over the FULL unfiltered speaker list *by raw index*
    (alphabetical filename order) -- for the first unknown-gender character that
    silently meant "whatever file sorts first", which happened to be a
    female-tagged clip for a male character, with no signal anywhere that a guess
    had even been made. Fixed: alternate between the male/female pools themselves
    (never the raw list) for unknown-gender characters, so the pick is always at
    least a *coherent* voice, and mark it explicitly with gender_guessed=True so
    downstream (cast.json review, the UI) can flag it for the user to confirm --
    a guess presented as unquestioned fact is worse than a guess that visibly asks
    to be checked."""
    from script_pipeline.voice_library import list_emotive_voices, voice_gender

    # Prefer the 17-emotion library over the flat clips: those voices carry a gender in
    # their id (F01_/M01_) instead of needing it guessed from a filename substring, and
    # each one can act a line. Falling back to the flat clips keeps machines without the
    # library working exactly as before.
    emotive = list_emotive_voices()
    if emotive:
        male_pool = [v for v in emotive if voice_gender(v) == "male"]
        female_pool = [v for v in emotive if voice_gender(v) == "female"]
    else:
        male_pool = female_pool = []
    if not male_pool or not female_pool:
        speakers = _list_xtts_speakers() or ["male"]
        male_pool = male_pool or [s for s in speakers if "female" not in s and "male" in s] or speakers
        female_pool = female_pool or [s for s in speakers if "female" in s] or speakers
    male_i = female_i = 0
    unknown_toggle = 0
    voice_map = load_voice_map()

    assignment = {}
    for name, info in characters.items():
        # A curated entry beats every heuristic below: someone chose that clip for this
        # character on purpose, which no filename-substring guess can improve on.
        mapped = _match_voice(name, voice_map)
        if mapped:
            assignment[name] = {
                "engine": "auto",
                "gender": _guess_gender(name, info.get("descriptor", "")) or "unknown",
                "gender_guessed": False,  # irrelevant here: the voice was chosen by hand
                "xtts_speaker_wav": mapped["base"],
                "xtts_speaker_wav_style": mapped["style"],  # used for lines with emotion
                # NAO e None. O mapa curado da DOIS clipes (base + um estilo);
                # a biblioteca emotiva do mesmo interprete da 17. Zerar o campo
                # aqui fazia toda fala do personagem sair no mesmo clipe neutro,
                # com a emocao ja calculada e descartada -- ver o docstring de
                # _emotive_do_interprete. Sem interprete correspondente volta a
                # None, e o comportamento e o de antes.
                "emotive_voice": _emotive_do_interprete(mapped["source"], emotive),
                "archetype_voice": None,  # see the note on the heuristic branch below
                "voice_map_source": mapped["source"],
                "voice_direction": mapped["direction"],
                "qwen_speaker": None,
                "qwen_instruct_default": mapped["direction"],  # doubles as Qwen's baseline
            }
            continue

        gender = _guess_gender(name, info.get("descriptor", ""))
        gender_guessed = gender is None
        if gender is None:
            # Alternate so consecutive unknown-gender characters don't all land on
            # the same voice -- still an honest guess, never a silent miss.
            gender = "male" if unknown_toggle % 2 == 0 else "female"
            unknown_toggle += 1
        if gender == "male":
            speaker = male_pool[male_i % len(male_pool)]
            male_i += 1
        else:
            speaker = female_pool[female_i % len(female_pool)]
            female_i += 1
        assignment[name] = {
            "engine": "auto",
            "gender": gender,  # always "male"/"female" now -- see gender_guessed for confidence
            "gender_guessed": gender_guessed,  # True: no textual cue found, alternated as a placeholder -- verify by hand
            "xtts_speaker_wav": speaker,
            # Set only for tier-2 voices: names the folder whose 17 takes synthesize_dialogue
            # picks from per line. Absent for flat clips, which have no emotional range.
            "emotive_voice": speaker if speaker in emotive else None,
            # Deliberately never auto-filled. Matching "um mago idoso" to C05_mago is a
            # semantic judgement, and a wrong archetype is a worse failure than a plain
            # voice -- it gives a character the wrong PERSONA, not just the wrong timbre.
            # Left as an explicit choice in cast.json / the UI dropdown.
            "archetype_voice": None,
            "qwen_speaker": None,  # left for dialogue_tts.py's auto-pick, or hand-edit
            "qwen_instruct_default": None,  # e.g. "calm, warm tone" -- optional per-character baseline
        }
    return assignment


def _match_descriptor(descritores: dict, name: str) -> Optional[str]:
    """Acha o descritor de `name` sem depender da CAIXA da chave.

    MEDIDO 2026-08-27: pedindo descritores para "Park Min" e "Kim", o
    qwen3.6-35b devolveu as chaves "PARK MIN" e "KIM" -- convencao de roteiro,
    onde nome de personagem e sempre em caixa alta. Com `descriptors.get(name)`
    o resultado era 0/2 encontrados e os dois caiam no descritor deterministico,
    sem erro nenhum: o motor tinha escrito descritores bons e eles eram jogados
    fora em silencio. Vale para os dois motores -- o Gemma tem a mesma tendencia."""
    if not isinstance(descritores, dict):
        return None
    alvo = " ".join(name.split()).casefold()
    for chave, valor in descritores.items():
        if not isinstance(valor, str) or not valor.strip():
            continue
        if " ".join(str(chave).split()).casefold() == alvo:
            return valor.strip()
    return None


def _enrich_descriptors_llm(characters: dict) -> dict:
    from script_pipeline.parse_screenplay import _ask_gemma, _extract_json, _load_gemma  # local import: heavy

    lines = []
    for name, info in characters.items():
        snippet_text = " ".join(info["snippets"][:3]) or "(sem trechos de acao)"
        lines.append(f"{name}: {snippet_text}")
    user_prompt = "Personagens e trechos de acao:\n" + "\n".join(lines)

    model, processor = _load_gemma()
    # BUGFIX (2026-08-11): system_prompt became a required keyword when _ask_gemma was
    # refactored for the language/translation options, but this call site was never
    # updated -- so --llm raised TypeError immediately and this path had never actually
    # run. CAST_SYSTEM_PROMPT is the prompt it was always meant to send.
    raw = _ask_gemma(model, processor, user_prompt, system_prompt=CAST_SYSTEM_PROMPT, max_new_tokens=400)
    del model
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- a poisoned CUDA context must not lose the cast
        pass

    payload = _extract_json(raw) or {}
    descriptors = payload.get("descriptors", {})
    for name, info in characters.items():
        llm_descriptor = _match_descriptor(descriptors, name)
        info["descriptor"] = llm_descriptor or _default_descriptor(name, info["snippets"])
    return characters


def _enrich_descriptors_ollama(characters: dict, model: str, log=print) -> dict:
    """Descritores visuais via Ollama -- o mesmo motor que o resto da cadeia usa.

    POR QUE ESTE CAMINHO EXISTE, ALEM DO --llm

    O descritor deterministico e um recorte do texto de ACAO, e isso so devolve
    aparencia quando o roteiro segue a convencao de apresentar personagem entre
    parenteses ("XIAO-LAN (early 20s, mint-green silk robes), hurries..."). Num
    roteiro em PROSA CORRIDA nao ha convencao nenhuma para minerar, e o recorte
    devolve enredo puro. MEDIDO 2026-08-27, cena do ponto de onibus:

        "cena - no ponto de onibus a jovem - Park Min - personagem feminina -
         aguarda o onibus - dia chuvoso - o onibus para no ponto - ela"

    Isso ia colado em TODO storyboard_prompt e em TODO video_prompt. E
    `shot_plan.appearance_only()` nao salva: sem parentese e sem "wearing", ele
    devolve o descritor inteiro de proposito -- a troca que ele assume ("perder
    aparencia e pior que carregar um pouco de acao") vale para uma linha de
    apresentacao, e se inverte aqui, onde o texto e 100% acao e 0% aparencia.

    O --llm ja existia para isso, mas carrega o Gemma 3 em processo. A cadeia
    inteira ja migrou para o Ollama (MEMORIAL 3.13: 6s contra 91s), o Ollama ja
    esta no ar quando este estagio roda, e o `--engine` ja atravessa todos os
    outros estagios. Reusar e mais barato que ligar um segundo motor.

    Falhar aqui NAO derruba o casting: cai no descritor deterministico, que e o
    comportamento de antes."""
    from script_pipeline.story_structure import _call_ollama

    lines = []
    for name, info in characters.items():
        trechos = " ".join(info["snippets"][:3]) or "(sem trechos de acao)"
        lines.append(f"{name}: {trechos}")
    # As chaves exigidas vao repetidas no fim: dizer o formato no system prompt nao
    # bastou (ver CAST_SYSTEM_PROMPT), e repetir a lista e o que ancora o modelo.
    user = ("Personagens e trechos de acao:" + chr(10) + chr(10).join(lines) + chr(10) +
            chr(10) + "Chaves obrigatorias de 'descriptors', exatamente estas e so estas: "
            + ", ".join(characters))

    payload = _call_ollama(CAST_SYSTEM_PROMPT, user, model, log=log) or {}
    descritores = payload.get("descriptors") or {}
    achados = 0
    for name, info in characters.items():
        vindo = _match_descriptor(descritores, name)
        if vindo:
            info["descriptor"] = vindo
            achados += 1
        else:
            info["descriptor"] = _default_descriptor(name, info["snippets"])
    log(f"[cast_characters] {achados}/{len(characters)} descritor(es) visual(is) "
        f"escrito(s) via Ollama ({model}).")
    if achados < len(characters):
        log("[cast_characters] os demais cairam no recorte do texto de acao -- "
            "confira cast.json antes de gerar stills.")
    return characters


def build_cast(scenes: list[dict], *, use_llm: bool = False, reference_images: dict | None = None,
               engine: str | None = None, log=print) -> dict:
    """reference_images: {character_name: image_path}. A character with a reference
    photo gets it recorded as `reference_image`; generate_storyboards.py then seeds
    that character's shots from the photo (FLUX.2 keeps facial identity across a
    reference -- verified with a controlled test earlier in this project) instead of
    relying on a text descriptor alone, which is what makes the same character look
    like a different person from shot to shot."""
    characters = collect_characters(scenes)
    if use_llm:
        characters = _enrich_descriptors_llm(characters)
    elif engine:
        characters = _enrich_descriptors_ollama(characters, engine, log=log)
    else:
        for name, info in characters.items():
            info["descriptor"] = _default_descriptor(name, info["snippets"])

    characters = _audit_and_fix_descriptors(characters, model=engine, log=log)

    voices = assign_voices(characters)

    refs = reference_images or {}
    cast = {}
    for name, info in characters.items():
        cast[name] = {
            "descriptor": info["descriptor"],
            "line_count": info["line_count"],
            "voice": voices[name],
            # None => identity comes from the text descriptor only (the "automatic"
            # mode); a path => that photo anchors this character's look.
            "reference_image": refs.get(name),
            # Auditoria de completude (ver _check_descriptor_completeness) -- vazio
            # quando as 4 categorias exigidas estao cobertas.
            "descriptor_gaps": info.get("descriptor_gaps", []),
        }
    return cast


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--llm", action="store_true", help="Use Gemma3 to write richer visual descriptors.")
    parser.add_argument("--engine", default=None,
                        help="Tag de modelo do Ollama para escrever os descritores visuais "
                             "(ex: qwen3.6-35b-a3b:latest). Sem isto, o descritor e um recorte "
                             "do texto de acao -- que so vira aparencia se o roteiro apresentar "
                             "o personagem entre parenteses.")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    scenes = _load_scenes(run_dir)
    cast = build_cast(scenes, use_llm=args.llm, engine=args.engine)

    characters_dir = run_folder.subdir(run_dir, "characters")
    cast_path = characters_dir / "cast.json"
    cast_path.write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")

    run_folder.append_log(
        run_dir,
        f"cast_characters: {len(cast)} personagem(ns) -> {cast_path} "
        f"(edite este arquivo antes de continuar, se quiser ajustar descritores/vozes).",
    )
    run_folder.mark_stage_complete(run_dir, "cast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
