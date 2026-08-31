"""Turn free-form prose (a treatment, a synopsis, or an LTX audiovisual prompt)
into screenplay formatting that parse_screenplay's deterministic pass can read.

WHY THIS IS A SEPARATE, EXPLICIT STEP
parse_screenplay's core promise is that structure is deterministic: regex finds
scenes and dialogue, so it "never hallucinates a scene or a line of dialogue
that isn't in the source text". An LLM that writes the screenplay would break
exactly that promise. So this module is kept OUT of that pass: it runs only when
the deterministic pass found nothing at all, it writes its output to a file the
user can read and edit, and it VERIFIES that every quoted line of dialogue in
the source survived verbatim into the result (see check_dialogue_preserved).
The deterministic parser still does the actual structure extraction afterwards.

The verbatim check matters because dialogue is not decoration downstream: it is
sent to TTS and then lip-synced. A paraphrased line would be silently spoken
wrong, which is much worse than a formatting failure.

CLI:
    python -m script_pipeline.prose_to_screenplay --script IN.txt --out OUT.txt
                                                  [--engine gemma4|gemma3] [--language pt]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMMA4_ENV_PYTHON = str(ROOT / "gemma4_env" / "Scripts" / "python.exe")
GEMMA4_WORKER = str(ROOT / "script_pipeline" / "llm_workers" / "gemma4_worker.py")

# Straight and typographic quote pairs. LTX prompts written by the prompt guide
# use the curly ones; hand-typed text usually uses straight ones.
_QUOTE_PAIRS = [("“", "”"), ('"', '"'), ("«", "»")]


def extract_quoted_dialogue(text: str) -> list[str]:
    """Every quoted span in the source, in order. Used to verify the rewrite kept
    the spoken lines verbatim -- not to build structure."""
    found: list[str] = []
    for open_q, close_q in _QUOTE_PAIRS:
        if open_q == close_q:
            # Straight quotes: pair them up sequentially.
            parts = text.split(open_q)
            for i in range(1, len(parts), 2):
                s = parts[i].strip()
                if s:
                    found.append(s)
        else:
            for m in re.finditer(re.escape(open_q) + r"(.+?)" + re.escape(close_q), text, re.DOTALL):
                s = m.group(1).strip()
                if s:
                    found.append(s)
    return found


def _normalize(s: str) -> str:
    """Loose comparison for the verbatim check.

    Quote style, whitespace runs, and MISSING space after sentence punctuation
    must not count as "the line was changed". That last one is not hypothetical:
    the first real input had a typo ("presença.Desde"), the model emitted the
    normal "presença. Desde", and a stricter check reported the whole speech as
    lost. A verbatim check that cries wolf on cosmetic fixes gets ignored, which
    defeats its purpose -- it must fire only on words actually differing."""
    s = re.sub(r"[“”«»\"]", "", s)
    s = re.sub(r"([.,;:!?])(?=\S)", r"\1 ", s)  # "a.B" -> "a. B"
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def enforce_screenplay_format(text: str) -> str:
    """Deterministically repair the two format invariants an LLM reliably gets
    wrong, instead of hoping a better prompt fixes them.

    MEASURED on the first real conversion:
      - A character cue written as "Lyra" instead of "LYRA" is invisible to
        parse_screenplay's CUE_RE (which requires uppercase), so that speech was
        silently DROPPED -- 3 lines parsed out of 4.
      - Dialogue kept its surrounding quotes, which are prompt punctuation, not
        screenplay syntax, and would be carried into TTS.

    Cue casing is repaired conservatively: only names that already appear as a
    proper uppercase cue somewhere in the document get uppercased elsewhere, so
    an ordinary sentence starting with a word is never mistaken for a cue.
    """
    lines = text.splitlines()

    # Pass 1: collect names that are already valid uppercase cues.
    known: set[str] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"^([A-ZÀ-Ý][A-ZÀ-Ý0-9 .'\-]{0,38}?)(\s*\([^)]*\))?$", stripped)
        if not m:
            continue
        name = m.group(1).strip()
        nxt = next((lines[j].strip() for j in range(i + 1, len(lines)) if lines[j].strip()), "")
        if len(name) >= 2 and nxt:
            known.add(name.upper())

    # Also treat every capitalised word that leads a "Name ... , «quote»" prose
    # line as a candidate name, so pass 2 can promote those to real cues even
    # when the model never wrote that name as a proper cue anywhere.
    for line in lines:
        m = re.match(r"^\s*([A-ZÀ-Ý][\wÀ-ÿ'\-]{1,30})\b.*?[“\"«]", line)
        if m:
            known.add(m.group(1).upper())

    # Pass 1.5: a stage direction written under a cue would be SPOKEN by TTS.
    lines = _demote_stage_direction_blocks(lines, known)

    # Pass 2: uppercase lone lines matching a known cue; promote prose lines that
    # bury dialogue in a quote; strip leftover dialogue quotes.
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.upper() in known and stripped != stripped.upper():
            out.append(stripped.upper())
            continue

        promoted = _promote_embedded_dialogue(stripped, known)
        if promoted is not None:
            out.extend(promoted)
            continue

        cleaned = stripped
        if len(cleaned) > 1 and cleaned[0] in "“\"«":
            cleaned = cleaned[1:]
        if len(cleaned) > 1 and cleaned[-1] in "”\"»":
            cleaned = cleaned[:-1]
        out.append(cleaned if cleaned != stripped else line)
    return "\n".join(out)


_SPEECH_VERB_RE = re.compile(
    r"\b(diz|dizendo|declara|responde|exclama|pergunta|afirma|sussurra|grita|avisa|conta|"
    r"says?|answers?|declares?|replies|whispers?|shouts?)\b",
    re.IGNORECASE,
)


def _demote_stage_direction_blocks(lines: list[str], known: set[str]) -> list[str]:
    """Undo "stage direction written as dialogue".

    MEASURED (qwen3.6 via Ollama): it emitted

        LYRA
        Toca uma runa brilhante, vira-se para Thoren e diz com urgência.

        LYRA
        Thoren, as runas despertaram...

    The first block is a rubric, not speech, but a cue + following line is
    exactly what parse_screenplay reads as dialogue -- so TTS would *speak the
    stage direction*. The tell is unambiguous and cheap to detect: the SAME cue
    repeats immediately, and the first block's line contains a speech verb. Real
    screenplays never repeat a cue back-to-back like that, so the rule does not
    fire on genuine dialogue.
    """
    cue_re = re.compile(r"^([A-ZÀ-Ý][A-ZÀ-Ý0-9 .'\-]{1,38})$")
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = cue_re.match(stripped)
        if m and stripped.upper() in known:
            # gather this block: cue, then its first non-blank line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            body = lines[j].strip() if j < len(lines) else ""
            # find the next cue after the body
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            nxt = lines[k].strip() if k < len(lines) else ""
            if (body and _SPEECH_VERB_RE.search(body) and cue_re.match(nxt)
                    and nxt.upper() == stripped.upper()):
                out.append(body)   # keep it, as an ACTION line (no cue above)
                out.append("")
                i = k
                continue
        out.append(lines[i])
        i += 1
    return out


def _promote_embedded_dialogue(line: str, known: set[str]) -> list[str] | None:
    """Split a prose line that buries a spoken line in quotes into proper
    screenplay blocks, or return None if it isn't that shape.

    MEASURED: the model intermittently emits
        Lyra olha para cima ... e diz com a respiração curta, "A lua está...
    instead of a LYRA cue + dialogue. The words survive (so a text-only verbatim
    check passes) but parse_screenplay reads the whole thing as an ACTION line,
    so the speech never reaches TTS. Recovering it deterministically is more
    reliable than re-prompting the model and hoping for better formatting.
    """
    m = re.match(r"^\s*([A-ZÀ-Ý][\wÀ-ÿ'\-]{1,30})\b(.*?)[“\"«](.+?)[”\"»]?\s*$", line)
    if not m:
        return None
    name, action_rest, speech = m.group(1), m.group(2).strip(), m.group(3).strip()
    if name.upper() not in known or not speech:
        return None
    blocks: list[str] = []
    # Keep the stage business as an action line; drop the trailing speech verb
    # clause ("e diz com a respiração curta,") so it doesn't read as narration.
    action = re.sub(r"[,:]?\s*(e\s+)?(diz|responde|declara|exclama|pergunta|afirma|"
                    r"sussurra|grita|conta|avisa)\b.*$", "", f"{name}{action_rest}",
                    flags=re.IGNORECASE).strip(" ,;:")
    if action and action.lower() != name.lower():
        blocks.append(action + ("" if action.endswith(".") else "."))
        blocks.append("")
    blocks.append(name.upper())
    blocks.append(speech)
    return blocks


def check_dialogue_preserved(source: str, screenplay: str) -> tuple[list[str], list[str]]:
    """Returns (kept, missing) quoted lines from *source* that survived as actual
    PARSED DIALOGUE in *screenplay*.

    Deliberately stricter than "the words appear somewhere in the text". A line
    the model left buried inside a prose action sentence passes a text search but
    is read as narration by parse_screenplay -- so it never reaches TTS and is
    never spoken. That failure is invisible to a textual check, which is exactly
    the kind of silent wrongness this module exists to prevent. So the check runs
    the real parser and looks only at what came out as dialogue, falling back to
    a text search only if the parser cannot be imported.
    """
    try:
        from script_pipeline.parse_screenplay import parse_structure

        scenes = parse_structure(screenplay)
        haystacks = [_normalize(d.text or "") for s in scenes for d in s.dialogue]
    except Exception:
        haystacks = [_normalize(screenplay)]

    kept, missing = [], []
    for line in extract_quoted_dialogue(source):
        needle = _normalize(line)
        found = any(needle in h or h in needle for h in haystacks)
        (kept if found else missing).append(line)
    return kept, missing


SYSTEM_PROMPT = """Você converte texto livre em FORMATO DE ROTEIRO. Você NÃO inventa história.

REGRAS ABSOLUTAS:
1. Toda fala entre aspas no texto original deve aparecer no roteiro EXATAMENTE
   igual, palavra por palavra, no idioma original. Nunca traduza, resuma,
   corrija ou reescreva uma fala.
2. Não crie personagens, falas ou acontecimentos que não estejam no texto.
3. Se o texto não disser quem fala, use o nome que o texto usa para a pessoa.
4. APARÊNCIA É CONTEÚDO, NÃO ENFEITE. Quando o texto descrever como um
   personagem é (idade, cabelo, roupa, armadura, cor, objeto que carrega),
   essa descrição tem de aparecer na PRIMEIRA linha de ação em que ele
   entra, entre parênteses, logo depois do nome em maiúsculas:

       AKEMI (jovem samurai, rabo de cavalo preto, armadura vermelha
       laqueada, cachecol branco, katana curta brilhante) bloqueia o golpe.

   É assim que o resto da produção descobre com que aparência desenhar cada
   personagem. Descrição de aparência perdida aqui não volta em nenhum
   estágio seguinte. Não invente aparência que o texto não deu.

   Isso NÃO muda o formato do bloco de fala: toda fala continua com o nome
   sozinho numa linha, parêntese curto de interpretação (opcional) e a fala
   embaixo -- mesmo que o nome já tenha aparecido na linha de ação logo acima.
   Repetir o nome ali parece redundante e não é: sem ele a fala não é
   reconhecida como fala de ninguém.
5. IDADE É PARTE DA APARÊNCIA. Se o texto diz "menina", "criança", "garoto",
   "idoso", use ESSA palavra no parêntese. "Jovem" não substitui "menina": um
   filme infantil com uma adulta no lugar da criança está errado, e nenhum
   estágio depois deste consegue perceber a troca.
6. Se o texto disser em que MEIO a obra é feita -- animação anime, animação
   3D, live action, aquarela, stop motion --, escreva isso na PRIMEIRA linha
   do arquivo, antes do cabeçalho de cena, exatamente neste formato:

       ESTILO VISUAL: polished hand-drawn cel animation, sharp ink lines

   Uma linha só, em inglês, em termos de prompt de imagem. Sem ela o meio se
   perde e cada plano acaba desenhado num acabamento diferente.

FORMATO DE SAÍDA (texto puro, sem comentários seus, sem cercas de código):

INT. NOME DO LUGAR - PERÍODO DO DIA
(ou EXT. ... se a cena for externa)

O cabeçalho leva o lugar e o período REAIS da cena, tirados do texto:
"INT. CORREDOR DE PORTAS - DIA", "EXT. TEMPLO EM RUÍNAS - NOITE". Se o texto
não disser o período, escolha o que o texto sugere (chuva forte e portais
brilhando = NOITE) -- mas NUNCA escreva as palavras LOCAL, MOMENTO, PERÍODO
ou NOME DO LUGAR: elas são a descrição do formato, não texto para copiar.

Linha de ação descrevendo o que se vê, em terceira pessoa, presente.

NOME DO PERSONAGEM
(rubrica curta de interpretação, opcional)
A fala exatamente como no original.

NOME DO PERSONAGEM
Outra fala exatamente como no original.

Use uma linha em branco entre blocos. O nome do personagem vai sozinho em uma
linha, em MAIÚSCULAS. Descrições de câmera, luz e som viram linhas de ação."""


def _run_gemma4(user_prompt: str, *, max_new_tokens: int, log=print) -> str | None:
    """One-shot call to the isolated gemma4_env worker (same pattern as
    parse_screenplay.enrich_with_llm_gemma4)."""
    if not Path(GEMMA4_ENV_PYTHON).exists():
        log(f"[prose_to_screenplay] gemma4_env nao encontrado em {GEMMA4_ENV_PYTHON}")
        return None
    jobs = [{"id": "convert", "system_prompt": SYSTEM_PROMPT,
             "user_prompt": user_prompt, "max_new_tokens": max_new_tokens}]
    with tempfile.TemporaryDirectory() as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        results_path = Path(tmp) / "results.json"
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [GEMMA4_ENV_PYTHON, "-u", GEMMA4_WORKER,
                   "--jobs", str(jobs_path), "--results", str(results_path)]
        log("[prose_to_screenplay] convertendo prosa -> roteiro via Gemma4 (venv isolado)...")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, universal_newlines=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log(line.rstrip("\n"))
        proc.wait()
        if not results_path.exists():
            log("[prose_to_screenplay] worker nao produziu results.json.")
            return None
        results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results or not results[0].get("ok"):
        log(f"[prose_to_screenplay] conversao falhou: {results[0].get('error') if results else 'sem resultado'}")
        return None
    return (results[0].get("raw_text") or "").strip()


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def _run_ollama(user_prompt: str, *, model: str, max_new_tokens: int, log=print) -> str | None:
    """Convert via a model served by Ollama.

    Added as an option because gemma4-e2b (the transformers default) is a ~2B
    effective-parameter model, and its formatting slips are what forced the
    deterministic repairs in this module. A larger served model is worth
    benchmarking against it -- see _bench in the CLI.

    NOTE (CLAUDE.md): the Ollama desktop app serves a DIFFERENT model directory
    than `ollama serve` with OLLAMA_MODELS=G:\\ollama\\models. If a model that
    exists on disk 404s here, that mismatch is the usual cause, so the error
    message lists what this instance actually serves.
    """
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        # Reasoning models (qwen3.x, qwq, deepseek-r1...) write chain-of-thought
        # into a separate `thinking` field and only then fill `content`. MEASURED:
        # with thinking on, qwen3.6-35b spent the whole token budget reasoning and
        # returned done_reason="length" with content EMPTY -- a silent failure that
        # looks like the model refusing. Reformatting needs no chain-of-thought, so
        # it is turned off; models without the feature ignore the flag.
        "think": False,
        "options": {"num_predict": max_new_tokens, "temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    log(f"[prose_to_screenplay] convertendo prosa -> roteiro via Ollama ({model})...")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        log(f"[prose_to_screenplay] Ollama HTTP {e.code}: {body}")
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as r:
                served = [m["name"] for m in json.load(r).get("models", [])]
            log(f"[prose_to_screenplay] esta instancia serve: {served}")
        except Exception:
            pass
        return None
    except Exception as e:
        log(f"[prose_to_screenplay] Ollama indisponivel: {e}")
        return None
    return (data.get("message") or {}).get("content", "").strip()


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def convert(text: str, *, engine: str = "gemma4", log=print) -> tuple[str | None, list[str]]:
    """Prose -> screenplay text. Returns (screenplay_or_None, missing_dialogue_lines)."""
    quoted = extract_quoted_dialogue(text)
    log(f"[prose_to_screenplay] {len(quoted)} fala(s) entre aspas detectada(s) no texto de origem.")

    # Budget: the whole screenplay must fit, and dialogue is reproduced verbatim,
    # so scale with the source's own size rather than using a flat cap.
    budget = min(3000, 600 + int(len(text) / 2))
    if engine == "gemma4":
        raw = _run_gemma4(text, max_new_tokens=budget, log=log)
    elif engine == "gemma3":
        raw = _run_gemma3(text, max_new_tokens=budget, log=log)
    else:
        # anything else is taken as an Ollama model tag, e.g. "qwen3.6-35b-a3b"
        raw = _run_ollama(text, model=engine, max_new_tokens=budget, log=log)
    if not raw:
        return None, quoted

    screenplay = enforce_screenplay_format(_strip_code_fence(raw))
    kept, missing = check_dialogue_preserved(text, screenplay)
    if missing:
        log(f"[prose_to_screenplay] ATENCAO: {len(missing)} de {len(quoted)} fala(s) NAO sobreviveram "
            "literalmente a conversao. Elas serao faladas errado pelo TTS se o roteiro for usado assim:")
        for m in missing:
            log(f"    - {m[:120]}")
    else:
        log(f"[prose_to_screenplay] todas as {len(kept)} fala(s) preservadas literalmente.")
    return screenplay, missing


def _run_gemma3(user_prompt: str, *, max_new_tokens: int, log=print) -> str | None:
    """In-process Gemma3 path, for parity with parse_screenplay's --enrich-engine."""
    try:
        from script_pipeline.parse_screenplay import _ask_gemma, _load_gemma
    except Exception as e:  # pragma: no cover - import-time environment issue
        log(f"[prose_to_screenplay] nao foi possivel carregar o caminho gemma3: {e}")
        return None
    log("[prose_to_screenplay] convertendo prosa -> roteiro via Gemma3 (venv principal)...")
    model, processor = _load_gemma()
    try:
        return _ask_gemma(model, processor, user_prompt,
                          system_prompt=SYSTEM_PROMPT, max_new_tokens=max_new_tokens, log=log)
    finally:
        del model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prosa/prompt LTX -> formato de roteiro")
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="gemma4",
                    help="gemma4 | gemma3 | qualquer tag de modelo servida pelo Ollama")
    args = ap.parse_args(argv)

    text = Path(args.script).read_text(encoding="utf-8")
    screenplay, missing = convert(text, engine=args.engine)
    if screenplay is None:
        print("Conversao falhou.", file=sys.stderr)
        return 1
    Path(args.out).write_text(screenplay, encoding="utf-8")
    print(f"[prose_to_screenplay] escrito: {args.out}")
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
