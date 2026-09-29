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


# FIDELIDADE AO ROTEIRO-FONTE -- achado da avaliacao visual 2026-09-17 (Palacio
# Esmeralda): a auditoria de COMPLETUDE acima (_check_descriptor_completeness)
# so confere se as 4 CATEGORIAS estao presentes, nunca se o CONTEUDO bate com
# o que o roteiro realmente diz. MEDIDO: o roteiro descreve Mei-Li vestindo
# "a luxurious, flowing Hanfu of celestial blue silk... classical high-bun...
# buyao", e o descritor gerado (qwen3.6, via Ollama) saiu "chin-length
# chestnut-brown bob... fitted crimson velvet tunic... dark grey linen
# trousers" -- 4 categorias cobertas, zero relacao com o texto-fonte. Isso
# passa a auditoria de completude porque ela nunca olha PARA o roteiro, so
# para a FORMA do descritor.
_FIDELITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "of", "in", "on", "her", "his",
    "she", "he", "is", "was", "were", "to", "at", "by", "for", "that", "this",
    "into", "onto", "their", "them", "she's", "his's", "its", "as", "while",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
            if w not in _FIDELITY_STOPWORDS}


# ACHADO 2026-09-29 (REENTRY WINDOW): a sobreposicao de palavras pega invencao TOTAL
# (a Lyra virando jaqueta de couro), mas nao pega TROCA PONTUAL de um atributo decisivo
# quando o resto do vocabulario e parecido o bastante (os dois falam de "suit"/"flight"/
# "chest"/"collar" porque os dois sao trajes de voo). MEDIDO: "long dark hair, charcoal
# suit" -> "short, dark-brown hair... dark-blue suit" teve 30% de sobreposicao (acima do
# limiar de 20%) apesar de trocar comprimento de cabelo E cor da roupa -- duas
# contradicoes diretas que a taxa agregada nao enxerga. Estas funcoes comparam
# CATEGORIA POR CATEGORIA (comprimento de cabelo, cor perto de "hair", cor perto de uma
# peca de roupa) em vez de só contar palavras em comum.
_HAIR_LENGTH_SHORT_WORDS = ("short", "cropped", "buzz", "pixie", "crew-cut", "shaved", "bald")
_HAIR_LENGTH_LONG_WORDS = ("long", "flowing", "waist-length", "floor-length")
_COLOR_WORDS = (
    "black", "brown", "auburn", "blonde", "blond", "red", "gray", "grey", "white",
    "silver", "charcoal", "blue", "green", "navy", "olive", "tan", "orange",
    "purple", "pink", "gold", "chestnut",
)
_GARMENT_WORDS = ("jacket", "coat", "suit", "uniform", "jumpsuit", "dress", "robe", "cloak", "tunic")


def _hair_length_bucket(text: str) -> str | None:
    low = (text or "").lower()
    for m in re.finditer(r"\bhair\b", low):
        window = low[max(0, m.start() - 40):m.start()]
        if any(w in window for w in _HAIR_LENGTH_SHORT_WORDS):
            return "short"
        if any(w in window for w in _HAIR_LENGTH_LONG_WORDS):
            return "long"
    return None


def _color_near(text: str, keyword: str) -> str | None:
    """Cor mais PROXIMA de `keyword` (ex.: 'hair', 'jacket') numa janela curta antes
    dela -- nao a primeira da lista fixa `_COLOR_WORDS`. BUGFIX (2026-09-29, achado
    pelo proprio teste de regressao): com janela larga e busca por ordem de lista, uma
    frase como "dark-blue jacket" antecedida de "black hair, " pegava "black" (que so
    aparece antes na string, nao mais perto do substantivo) em vez de "blue" -- a cor
    do CABELO vazava pra dentro da checagem da cor da ROUPA. Agora pega a ocorrencia
    de cor com maior indice na janela (mais perto de `keyword`). Substring-safe:
    "dark-brown" contem "brown", entao um lado dizendo "brown" e o outro "dark-brown"
    NAO conta como contradicao (mesma cor, detalhe a mais); só cores REALMENTE
    diferentes (auburn vs brown) contam."""
    low = (text or "").lower()
    for m in re.finditer(rf"\b{keyword}\b", low):
        window = low[max(0, m.start() - 40):m.start()]
        best_color, best_pos = None, -1
        for color in _COLOR_WORDS:
            pos = window.rfind(color)
            if pos > best_pos:
                best_pos, best_color = pos, color
        if best_color:
            return best_color
    return None


def _garment_color(text: str) -> str | None:
    for garment in _GARMENT_WORDS:
        color = _color_near(text, garment)
        if color:
            return color
    return None


def _attribute_contradiction(anchor: str, descriptor: str) -> str | None:
    """Devolve uma explicacao curta se anchor/descriptor contradizem um atributo
    especifico (comprimento de cabelo, cor do cabelo, cor da roupa) -- None se nao
    houver base de comparacao ou os dois baterem. Duas cores so contam como
    contraditorias se nenhuma for substring da outra."""
    a_len, d_len = _hair_length_bucket(anchor), _hair_length_bucket(descriptor)
    if a_len and d_len and a_len != d_len:
        return f"comprimento de cabelo ({a_len} -> {d_len})"
    a_hair_color, d_hair_color = _color_near(anchor, "hair"), _color_near(descriptor, "hair")
    if a_hair_color and d_hair_color and a_hair_color != d_hair_color \
            and a_hair_color not in d_hair_color and d_hair_color not in a_hair_color:
        return f"cor do cabelo ({a_hair_color} -> {d_hair_color})"
    a_garment_color, d_garment_color = _garment_color(anchor), _garment_color(descriptor)
    if a_garment_color and d_garment_color and a_garment_color != d_garment_color \
            and a_garment_color not in d_garment_color and d_garment_color not in a_garment_color:
        return f"cor da roupa ({a_garment_color} -> {d_garment_color})"
    return None


def _ground_truth_snippet(snippets: list[str]) -> Optional[str]:
    """A apresentacao visual do PROPRIO roteiro, se ele deu uma: o primeiro
    trecho de acao que descreve como o personagem esta VESTIDO (mesmo sinal
    de 'wearing' que shot_plan.appearance_only() ja usa como ancora de
    apresentacao). Trechos vem em ordem de aparicao no roteiro -- o primeiro
    que bate e a apresentacao, nao uma mencao de acao posterior."""
    for s in snippets:
        low = s.lower()
        if len(s) > 30 and any(k in low for k in
                               ("wearing", "dressed in", "veste ", "vestindo")):
            return s
    return None


def _enforce_descriptor_fidelity(name: str, descriptor: str, anchor: str, log=print) -> str:
    """Compara o descritor final contra a apresentacao original por
    sobreposicao lexical de palavras de conteudo. Pouca sobreposicao e sinal
    forte de invencao (o LLM tem plena liberdade para escolher OUTRAS
    palavras para a MESMA roupa, mas nao para descrever uma roupa totalmente
    diferente) -- quando isso acontece, o roteiro vence: descarta o
    descritor gerado e usa o texto original (limpo de dialogo) no lugar."""
    anchor_words = _content_words(anchor)
    if not anchor_words:
        return descriptor
    overlap = anchor_words & _content_words(descriptor)
    ratio = len(overlap) / len(anchor_words)
    contradiction = _attribute_contradiction(anchor, descriptor)
    if ratio < 0.2 or contradiction:
        motivo = (f"{ratio:.0%} de sobreposicao" if ratio < 0.2 else f"contradiz {contradiction}")
        log(f"[cast_characters] AUDITORIA: descritor de {name} contradiz a apresentacao "
            f"do roteiro (\"{anchor[:90]}...\") -- {motivo}, "
            "substituindo pelo texto original do roteiro.")
        cleaned = _clean_descriptor(anchor)
        if len(cleaned) > 300:
            corte = cleaned.rfind(" ", 0, 300)
            cleaned = cleaned[:corte if corte > 0 else 300].rstrip(",;: ") + "..."
        return cleaned
    return descriptor


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


# --- figurantes recorrentes sem nome proprio -------------------------------------
# MEDIDO 2026-09-27 (CERCO EM SEUL): o Presidente, o motociclista/terrorista e o
# atirador agem em metade dos planos e nunca entravam no elenco -- o elenco so
# conhecia quem FALA ou tem nome proprio. Sem identidade, o shot_plan jogava a
# acao deles em insert e cada still inventava outra pessoa. O mesmo individuo
# aparece com varios nomes ("motorcycle rider" = "terrorist" = "bomber"), e so um
# modelo de linguagem junta isso; o codigo valida que cada apelido EXISTE no texto.
EXTRAS_SYSTEM_PROMPT = (
    "Voce e um assistente de casting. Responda APENAS com JSON valido no formato "
    '{"extras": [{"name": "...", "aliases": ["..."], "descriptor": "...", '
    '"gender": "male|female|unknown", "age": 0}]}. '
    "Liste as PESSOAS SEM NOME PROPRIO que agem ou sao alvo de acao em mais de um "
    "momento do texto e que sao sempre o MESMO individuo (ex.: 'the President', 'the "
    "motorcycle rider' que depois e chamado de 'the terrorist'). NAO inclua multidoes, "
    "grupos, plurais, figurantes de fundo que aparecem uma vez so, nem os personagens "
    "ja nomeados que a mensagem lista. 'name': identificador curto em INGLES, CAIXA "
    "ALTA, sem espaco (ex.: PRESIDENT, TERRORIST). 'aliases': TODAS as formas como o "
    "texto se refere a essa pessoa, copiadas EXATAMENTE como aparecem (sem artigo). O "
    "texto MISTURA idiomas (roteiro em portugues, acoes em ingles): liste as formas dos "
    "DOIS idiomas (ex.: 'Presidente' E 'President'; 'motociclista' E 'motorcycle rider' "
    "E 'terrorist'). 'descriptor': 2 frases em ingles, estilo prompt "
    "de imagem, com cabelo, idade/porte, roupa com cores exatas e um item distintivo, "
    "coerentes com o papel e o cenario. 'age': idade aproximada em anos (0 se "
    "desconhecida). Se nao houver ninguem assim, responda {\"extras\": []}."
)

# Sem LLM: papeis comuns com artigo definido (um individuo especifico). Nao junta
# sinonimos -- cada papel vira um figurante -- e por isso e so o plano B.
_PAPEIS = (
    "president", "prime minister", "king", "queen", "driver", "bodyguard", "guard", "soldier",
    "officer", "police officer", "policeman", "cop", "detective", "motorcycle rider", "rider",
    "motorcyclist", "biker", "terrorist", "bomber", "gunman", "sniper", "shooter", "attacker",
    "assassin", "kidnapper", "hostage", "thief", "robber", "suspect", "fugitive", "victim",
    "waiter", "waitress", "bartender", "doctor", "nurse", "patient", "pilot", "captain",
    "flight attendant", "stranger", "reporter", "journalist", "photographer", "vendor",
    "clerk", "receptionist", "priest", "teacher", "boss", "janitor", "paramedic",
    "presidente", "motorista", "guarda", "soldado", "policial", "motociclista", "terrorista",
    "atirador", "sequestrador", "ladr[aã]o", "suspeito", "fugitivo", "v[ií]tima", "gar[cç]om",
    "m[eé]dico", "enfermeira", "piloto", "comiss[aá]ria", "rep[oó]rter", "vendedor", "padre",
    "professor", "chefe",
)
_PAPEL_PT_EN = {
    "presidente": "president", "motorista": "driver", "guarda": "guard", "soldado": "soldier",
    "policial": "police officer", "motociclista": "motorcyclist", "terrorista": "terrorist",
    "atirador": "shooter", "sequestrador": "kidnapper", "ladrao": "thief", "ladrão": "thief",
    "suspeito": "suspect", "fugitivo": "fugitive", "vitima": "victim", "vítima": "victim",
    "garcom": "waiter", "garçom": "waiter", "medico": "doctor", "médico": "doctor",
    "enfermeira": "nurse", "piloto": "pilot", "comissaria": "flight attendant",
    "comissária": "flight attendant", "reporter": "reporter", "repórter": "reporter",
    "vendedor": "vendor", "padre": "priest", "professor": "teacher", "chefe": "boss",
}
# Formas equivalentes do MESMO papel (traducao PT <-> EN e sinonimo direto). NUNCA
# junta papeis diferentes: que "o motociclista" e "o terrorista" sao a mesma pessoa e
# decisao de ENREDO, tomada pelo LLM lendo a historia -- em outro roteiro podem ser
# duas pessoas. So entram como apelido se aparecem no texto (validado em detect_extras).
_FAMILIAS = (
    ("president", "presidente"),
    ("terrorist", "terrorista"),
    ("motorcycle rider", "motociclista", "motorcyclist", "biker"),
    ("bomber", "homem-bomba"),
    ("fugitive", "fugitivo"),
    ("sniper", "franco-atirador"),
    ("shooter", "atirador", "gunman"),
    ("driver", "motorista"), ("guard", "guarda", "bodyguard", "segurança"),
    ("police officer", "policial", "policeman", "cop"), ("soldier", "soldado"),
    ("kidnapper", "sequestrador"), ("hostage", "refém"), ("thief", "ladrão", "robber"),
    ("suspect", "suspeito"), ("victim", "vítima"), ("waiter", "garçom"),
    ("waitress", "garçonete"), ("doctor", "médico"), ("nurse", "enfermeira"),
    ("pilot", "piloto"), ("flight attendant", "comissária"), ("reporter", "repórter"),
    ("vendor", "vendedor"), ("priest", "padre"), ("teacher", "professor"), ("boss", "chefe"),
)
_FAMILIAS_DE_PAPEL = {forma: [f for f in familia if f != forma]
                      for familia in _FAMILIAS for forma in familia}

_PAPEL_RE = re.compile(r"\b(?:the|o|a)\s+(" + "|".join(_PAPEIS) + r")\b", re.IGNORECASE)


def _texto_das_cenas(scenes: list[dict]) -> str:
    partes = []
    for sc in scenes:
        partes.append(sc.get("action_text", "") or "")
        for item in sc.get("shot_list", []) or []:
            partes += [str(item.get("visual") or ""), str(item.get("actor") or "")]
        for line in sc.get("dialogue", []) or []:
            partes.append(str(line.get("beat_visual") or ""))
    return "\n".join(p for p in partes if p)


def _padrao_forma(forma: str) -> Optional[re.Pattern]:
    """"HA-EUN" casa "Ha-eun", "Ha eun" e "Haeun", sempre com fronteira de palavra
    ("PRESIDENT" nao casa dentro de "presidential")."""
    partes = [re.escape(p) for p in re.split(r"[-\s]+", (forma or "").strip().lower()) if p]
    if not partes:
        return None
    return re.compile(r"(?<![a-z0-9])" + r"[-\s]?".join(partes) + r"(?![a-z0-9])")


def _cita_forma(texto: str, forma: str) -> bool:
    padrao = _padrao_forma(forma)
    return bool(padrao and padrao.search((texto or "").lower()))


def personagens_citados(texto: str, cast: dict) -> list[str]:
    """Personagens do cast.json citados no texto, na ordem em que aparecem: nome
    proprio ou apelido de figurante (`aliases`). Fonte de audio fora de quadro
    (`on_screen: false`) fica de fora -- nao tem corpo para descrever."""
    baixo = (texto or "").lower()
    achados = []
    for nome, info in (cast or {}).items():
        if (info or {}).get("on_screen") is False:
            continue
        posicoes = []
        for forma in (nome, *((info or {}).get("aliases") or [])):
            padrao = _padrao_forma(forma)
            m = padrao.search(baixo) if padrao else None
            if m:
                posicoes.append(m.start())
        if posicoes:
            achados.append((min(posicoes), nome))
    return [nome for _, nome in sorted(achados)]


def _snippets_de(texto: str, formas: list[str]) -> list[str]:
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n", texto) if f.strip()]
    achadas = []
    for frase in frases:
        if any(_cita_forma(frase, f) for f in formas) and frase not in achadas:
            achadas.append(frase)
    return achadas


def detect_extras(scenes: list[dict], known: list[str], *, engine: str | None,
                  log=print) -> dict:
    """{NOME: {"aliases", "descriptor", "gender", "age", "snippets"}} dos figurantes
    recorrentes. Um figurante so entra se algum apelido aparece de fato no texto e
    ele e citado em pelo menos DUAS frases (uma aparicao so e fundo, nao elenco)."""
    texto = _texto_das_cenas(scenes)
    conhecidos = {_normalize_name(n) for n in known}
    candidatos: list[dict] = []
    if engine:
        from script_pipeline.story_structure import _call_ollama
        user = (f"Personagens ja nomeados (NAO liste): {', '.join(known) or '(nenhum)'}\n\n"
                f"Texto:\n{texto[:12000]}")
        candidatos = list(((_call_ollama(EXTRAS_SYSTEM_PROMPT, user, engine, log=log) or {})
                           .get("extras")) or [])
    if not candidatos:
        contagem: dict[str, int] = {}
        for m in _PAPEL_RE.finditer(texto):
            papel = m.group(1).lower()
            contagem[papel] = contagem.get(papel, 0) + 1
        # Mesmo papel em dois idiomas ("the President" / "o presidente") e UMA pessoa.
        por_papel: dict[str, dict] = {}
        for papel, n in contagem.items():
            canon = _PAPEL_PT_EN.get(papel, papel)
            item = por_papel.setdefault(canon, {"n": 0, "aliases": []})
            item["n"] += n
            item["aliases"].append(papel)
        candidatos = [{"name": re.sub(r"[^A-Z]", "_", canon.upper()), "aliases": v["aliases"],
                       "descriptor": "", "gender": "unknown", "age": 0}
                      for canon, v in por_papel.items() if v["n"] >= 2]
    extras: dict = {}
    for c in candidatos:
        nome = re.sub(r"[^A-Z0-9_-]", "", str(c.get("name") or "").upper().replace(" ", "_"))
        brutos = [str(x).strip() for x in c.get("aliases") or []]
        # O modelo tende a devolver so as formas de UM idioma (MEDIDO 2026-09-27: so
        # "Presidente"/"motociclista", enquanto as acoes enriquecidas dizem "the President"/
        # "the motorcycle rider" -- o shot_plan nao reconheceria ninguem). Completa com as
        # formas equivalentes do outro idioma que o texto de fato usa.
        for forma in list(brutos) + [nome.replace("_", " ")]:
            brutos += _FAMILIAS_DE_PAPEL.get(forma.strip().lower(), [])
        aliases = []
        for a in brutos:
            if a and a.lower() not in {x.lower() for x in aliases} and _cita_forma(texto, a):
                aliases.append(a)
        if not nome or not aliases or _normalize_name(nome) in conhecidos or nome in extras:
            continue
        snippets = _snippets_de(texto, aliases)
        if len(snippets) < 2:
            continue
        idade = c.get("age")
        extras[nome] = {"aliases": aliases, "descriptor": str(c.get("descriptor") or "").strip(),
                        "gender": c.get("gender") if c.get("gender") in ("male", "female") else None,
                        "age": int(idade) if isinstance(idade, (int, float)) and idade > 0 else None,
                        "snippets": snippets[:3]}
    if extras:
        log(f"[cast_characters] {len(extras)} figurante(s) recorrente(s) sem nome: "
            + ", ".join(f"{n} ({'/'.join(v['aliases'])})" for n, v in extras.items()))
    return extras


# --- idade e genero para a VOZ ---------------------------------------------------
# MEDIDO 2026-09-27 (CERCO EM SEUL): a agente de 35 anos recebeu F01_infantil (a
# primeira voz feminina da lista) e a agente de 20 e poucos recebeu voz MASCULINA
# (descritor sem pronome -> genero "adivinhado" por alternancia). A foto de
# referencia, quando existe, e a fonte mais confiavel das duas coisas.
_IDADE_DECADA = re.compile(r"\b(early|mid|late)?[- ]?(\d)0s\b", re.IGNORECASE)
_IDADE_ANOS = re.compile(r"\b(\d{1,2})[- ](?:year[- ]old|years? old|anos)\b", re.IGNORECASE)
_IDADE_PALAVRA = (
    (re.compile(r"\b(child|kid|little (?:boy|girl)|crian[cç]a|menin[oa])\b", re.I), 9),
    (re.compile(r"\b(teen\w*|adolescente)\b", re.I), 16),
    (re.compile(r"\b(elderly|old (?:man|woman)|idos[oa]|velh[oa]|senior)\b", re.I), 72),
)


def _parse_age(text: str) -> Optional[int]:
    m = _IDADE_ANOS.search(text or "")
    if m:
        return int(m.group(1))
    m = _IDADE_DECADA.search(text or "")
    if m:
        base = int(m.group(2)) * 10
        return base + {"early": 2, "mid": 5, "late": 8}.get((m.group(1) or "mid").lower(), 5)
    for padrao, idade in _IDADE_PALAVRA:
        if padrao.search(text or ""):
            return idade
    return None


def _faixa(idade: Optional[int]) -> Optional[str]:
    if idade is None:
        return None
    return "infantil" if idade < 14 else "jovem" if idade < 30 else "adulto" if idade < 55 else "maduro"


def _faixa_da_voz(voice_id: str) -> Optional[str]:
    v = voice_id.lower()
    for chave, faixa in (("infantil", "infantil"), ("jovem", "jovem"), ("adult", "adulto"),
                         ("madur", "maduro")):
        if chave in v:
            return faixa
    return None


_ORDEM_FAIXAS = ["infantil", "jovem", "adulto", "maduro"]


def _escolhe_voz(pool: list[str], faixa: Optional[str], usadas: set) -> str:
    """Voz da faixa etaria certa, preferindo uma ainda nao usada no elenco. Sem
    idade conhecida, nunca devolve voz infantil para quem nao e crianca."""
    def ordem(v: str) -> tuple:
        fv = _faixa_da_voz(v)
        if faixa and fv:
            dist = abs(_ORDEM_FAIXAS.index(faixa) - _ORDEM_FAIXAS.index(fv))
        else:
            dist = 0 if fv not in ("infantil",) else 9
        return (dist, v in usadas)
    return sorted(pool, key=ordem)[0]


def _photo_gender_age(path: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Genero/idade do maior rosto da foto (insightface genderage, o mesmo modelo
    do consistency_audit). Qualquer falha devolve (None, None)."""
    if not path or not Path(path).exists():
        return None, None
    try:
        import cv2
        import numpy as np
        from script_pipeline.consistency_audit import _get_app
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        faces = _get_app().get(img) if img is not None else []
        if not faces:
            return None, None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        sexo = {"M": "male", "F": "female"}.get(str(getattr(f, "sex", "")).upper())
        idade = getattr(f, "age", None)
        return sexo, int(idade) if idade is not None else None
    except Exception:
        return None, None


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
    usadas: list[str] = []
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
            usadas.append(mapped["base"])
            continue

        # Ordem de confianca: foto de referencia > inferencia do LLM sobre o texto
        # inteiro > pista textual no descritor > alternancia (marcada como palpite).
        foto_genero, foto_idade = _photo_gender_age(info.get("reference_image"))
        gender = (foto_genero or info.get("gender_hint")
                  or _guess_gender(name, info.get("descriptor", "")))
        gender_source = ("photo" if foto_genero else "llm" if info.get("gender_hint")
                         else "text" if gender else None)
        gender_guessed = gender is None
        if gender is None:
            # Alternate so consecutive unknown-gender characters don't all land on
            # the same voice -- still an honest guess, never a silent miss.
            gender = "male" if unknown_toggle % 2 == 0 else "female"
            unknown_toggle += 1
        # Idade: o ROTEIRO define o personagem; a foto so estima a do ator (MEDIDO:
        # Min-jun "mid-30s" no roteiro, 23 pela foto). Foto so quando o texto cala.
        idade = info.get("age_hint") or _parse_age(info.get("descriptor", "")) or foto_idade
        pool = male_pool if gender == "male" else female_pool
        if emotive and all(v in emotive for v in pool):
            speaker = _escolhe_voz(pool, _faixa(idade), set(usadas))
        elif gender == "male":
            speaker = male_pool[male_i % len(male_pool)]
            male_i += 1
        else:
            speaker = female_pool[female_i % len(female_pool)]
            female_i += 1
        usadas.append(speaker)
        assignment[name] = {
            "engine": "auto",
            "gender": gender,  # always "male"/"female" now -- see gender_guessed for confidence
            "gender_guessed": gender_guessed,  # True: no textual cue found, alternated as a placeholder -- verify by hand
            "gender_source": gender_source,
            "age": idade,
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


GENDER_AGE_SYSTEM_PROMPT = (
    "Responda APENAS com JSON valido no formato "
    '{"people": {"<NOME>": {"gender": "male|female|unknown", "age": 0}}}. '
    "Para cada nome listado, deduza genero e idade aproximada A PARTIR DO TEXTO: "
    "pronomes e concordancias que se referem a essa pessoa (cuidado: 'ela'/'ele' "
    "podem se referir a outra pessoa da mesma frase), descricoes e papel. Use "
    "'unknown' e age 0 quando o texto nao permitir concluir -- nunca chute pelo som "
    "do nome. As chaves sao exatamente os nomes listados."
)


def _infer_gender_age_ollama(characters: dict, texto: str, engine: str, log=print) -> None:
    """Preenche gender_hint/age_hint lendo o roteiro inteiro. O descritor visual
    sozinho quase nunca tem pronome ("Short, dark hair tied back...") -- foi assim
    que a Ha-eun virou voz masculina. Falha aqui so deixa os campos vazios."""
    from script_pipeline.story_structure import _call_ollama
    user = (f"Nomes: {', '.join(characters)}\n\nTexto:\n{texto[:12000]}")
    pessoas = ((_call_ollama(GENDER_AGE_SYSTEM_PROMPT, user, engine, log=log) or {})
               .get("people")) or {}
    for name, info in characters.items():
        dado = pessoas.get(name) or next(
            (v for k, v in pessoas.items() if _normalize_name(k) == _normalize_name(name)), None)
        if not isinstance(dado, dict):
            continue
        if dado.get("gender") in ("male", "female") and not info.get("gender_hint"):
            info["gender_hint"] = dado["gender"]
        idade = dado.get("age")
        if isinstance(idade, (int, float)) and idade > 0 and not info.get("age_hint"):
            info["age_hint"] = int(idade)


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

    for name, info in characters.items():
        anchor = _ground_truth_snippet(info["snippets"])
        if anchor:
            info["descriptor"] = _enforce_descriptor_fidelity(
                name, info["descriptor"], anchor, log=log)

    # Figurantes ANTES da auditoria de completude e da inferencia de genero/idade: MEDIDO
    # 2026-09-27, o descritor do figurante vindo do LLM saia generico ("a middle-aged man
    # in a formal dark suit" -- sem cabelo nem cores) e escapava da auditoria, que ja
    # tinha rodado.
    extras = detect_extras(scenes, list(characters), engine=engine, log=log)
    for name, ex in extras.items():
        characters[name] = {
            "snippets": ex["snippets"], "line_count": 0,
            "descriptor": ex["descriptor"] or _default_descriptor(name, ex["snippets"]),
            "gender_hint": ex["gender"], "age_hint": ex["age"],
        }

    characters = _audit_and_fix_descriptors(characters, model=engine, log=log)

    refs = reference_images or {}
    texto = _texto_das_cenas(scenes)
    if engine:
        _infer_gender_age_ollama(characters, texto, engine, log=log)
    for name, info in characters.items():
        info["reference_image"] = refs.get(name)

    voices = assign_voices(characters)

    cast = {}
    for name, info in characters.items():
        on_screen = not _is_offscreen_voice(name)
        cast[name] = {
            # Fonte fora de quadro (ver _is_offscreen_voice): sem descritor
            # visual nenhum -- gerar um so daria ao shot_plan/render um corpo
            # pra inventar. Achado por auditoria externa 2026-09-15: "VOZ"
            # (roteiro 2, voz misteriosa pelo radio) virava silhueta
            # holografica com cabelo e tunica, contradizendo o proprio
            # enquadramento pedido ("no face in frame").
            "descriptor": info["descriptor"] if on_screen else "",
            "line_count": info["line_count"],
            "voice": voices[name],
            # None => identity comes from the text descriptor only (the "automatic"
            # mode); a path => that photo anchors this character's look.
            "reference_image": refs.get(name),
            # Auditoria de completude (ver _check_descriptor_completeness) -- vazio
            # quando as 4 categorias estao cobertas.
            "descriptor_gaps": info.get("descriptor_gaps", []),
            # on_screen=False: personagem nunca deve virar sujeito de um still
            # nem receber close (shot_plan.py filtra isso ao montar o plano).
            "on_screen": on_screen,
        }
        if name in extras:
            # Lidos pelo shot_plan: "extra" entra na lista de personagens da cena
            # e "aliases" reconhece o figurante pelo papel ("the motorcycle rider").
            cast[name]["extra"] = True
            cast[name]["aliases"] = extras[name]["aliases"]
    return cast


def revoice(cast: dict, log=print) -> list[str]:
    """Reavalia as vozes de um cast.json existente (foto, idade, genero), sem mexer
    em voz escolhida a mao (`voice_locked`) nem no mapa curado. Devolve os nomes
    alterados. Pensado para depois do import_reference: a foto chega DEPOIS do
    casting, e era ela que teria evitado a voz masculina da Ha-eun."""
    alvo = {}
    for name, info in cast.items():
        voz = info.get("voice") or {}
        if voz.get("voice_locked") or voz.get("voice_map_source"):
            continue
        alvo[name] = {"descriptor": info.get("descriptor", ""),
                      "reference_image": info.get("reference_image"),
                      "gender_hint": None if voz.get("gender_guessed") else voz.get("gender"),
                      "age_hint": None}
    novas = assign_voices(alvo) if alvo else {}
    mudou = []
    for name, voz in novas.items():
        antiga = cast[name].get("voice") or {}
        if (antiga.get("xtts_speaker_wav"), antiga.get("gender")) != (voz["xtts_speaker_wav"], voz["gender"]):
            log(f"[cast_characters] voz de {name}: {antiga.get('xtts_speaker_wav')} "
                f"({antiga.get('gender')}) -> {voz['xtts_speaker_wav']} ({voz['gender']}, "
                f"fonte {voz.get('gender_source') or 'palpite'}, idade {voz.get('age')})")
            cast[name]["voice"] = {**antiga, **voz}
            mudou.append(name)
    return mudou


# Nomes convencionais de fonte de audio SEM presenca fisica em cena -- a
# convencao de roteiro pra isso e um cue generico como "VOZ" (radio,
# narracao, interfone), nunca um nome proprio de personagem. Comparado sem
# acento/caixa; lista pequena e explicita de proposito -- e melhor deixar
# passar um narrador nao-convencional (fica com descritor normal) do que
# apagar por engano um personagem real que se chame parecido.
_OFFSCREEN_VOICE_NAMES = {"voz", "voice", "narrador", "narrator", "narracao",
                           "radio", "interfone", "vo", "off"}


def _is_offscreen_voice(name: str) -> bool:
    norm = "".join(c for c in (name or "").lower() if c.isalpha())
    return norm in _OFFSCREEN_VOICE_NAMES


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
    parser.add_argument("--revoice", action="store_true",
                        help="nao refaz o elenco: so reavalia as vozes do cast.json existente "
                             "(foto de referencia, idade, genero). Voz com voice_locked=true fica.")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    if args.revoice:
        cast_path = run_dir / "characters" / "cast.json"
        cast = json.loads(cast_path.read_text(encoding="utf-8"))
        mudou = revoice(cast)
        cast_path.write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cast_characters] --revoice: {len(mudou)} voz(es) alterada(s)"
              + (f": {', '.join(mudou)}" if mudou else ""))
        return 0
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
