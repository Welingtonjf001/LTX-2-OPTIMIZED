"""XTTS-v2 batch worker. Runs INSIDE xtts's own isolated venv (webui/venv), not the
main project venv -- invoked via subprocess by dialogue_tts.py, mirroring how
tensorxx_ge/lipsync.py shells out to LatentSync's own conda env.

Loads the model exactly once, then generates every job in the batch, so per-call
overhead (~20s model load, confirmed empirically) is paid once per pipeline run, not
once per dialogue line.

Usage: <xtts venv python> xtts_worker.py --jobs jobs.json --results results.json
jobs.json: list of {"id": str, "text": str, "language": str (ISO code, e.g. "pt"/"en"),
                     "speaker_wav": str (name in webui/speakers, no extension, or an
                     absolute .wav path), "output_path": str}
results.json (written on exit): list of {"id": str, "ok": bool, "output_path": str|null,
                                          "error": str|null, "seconds": float}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

XTTS_ROOT = Path(r"E:\Users\home\Documents\xtts\webui")
sys.path.insert(0, str(XTTS_ROOT))

DEFAULT_OPTIONS = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
}

# xtts_funcs.TTSWrapper.local_generation() hardcodes enable_text_splitting=True,
# which -- MEASURED, not assumed -- produces an unnatural ~20s silence gap between
# sentences for any multi-sentence line (confirmed: a 2-sentence, ~15-word line
# rendered as 27s of audio, ~20s of it near-silence, vs ~5s for the same text split
# and generated sentence-by-sentence). Splitting ourselves and concatenating with a
# short, fixed pause avoids that library-internal quirk entirely.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PAUSE_BETWEEN_SENTENCES_SEC = 0.35

# PAUSA POR SINAL, nao pausa unica.
#
# A pausa fixa de 0,35 s era audivel COMO PONTUACAO: o mesmo intervalo, sempre,
# em todo limite de frase. Fala de gente nao faz isso -- reticencia suspende,
# exclamacao emenda, ponto final fecha -- e o ouvido pega a regularidade na
# hora. Valores em segundos, antes do fator de emocao.
PAUSA_POR_SINAL = {
    "...": 0.55,   # suspensao: a mais longa, porque e ela que o sinal PEDE
    "\u2026": 0.55,
    "!": 0.18,     # emenda: exclamacao empurra para a proxima frase
    "?": 0.30,
    ".": 0.32,
}
PAUSA_PADRAO = 0.32

# EMOCAO -> (fator de velocidade, fator de pausa).
#
# Slugs do `01_vozes_emotivas` (17). O clipe de referencia ja da o TIMBRE da
# emocao; o que faltava era o ANDAMENTO. Urgencia e correr e nao respirar --
# fator de pausa < 1 encurta os silencios, que e o que cria pressa. Tristeza e o
# contrario. Fora da tabela, nada muda (1,0 / 1,0).
PROSODIA = {
    "com_medo":                 (1.10, 0.65),
    "grito":                    (1.12, 0.55),
    "raiva":                    (1.10, 0.60),
    "angustia":                 (1.06, 0.75),
    "excitada":                 (1.12, 0.60),
    "espontanea_entusiasmada":  (1.12, 0.60),
    "surpresa":                 (1.05, 0.70),
    "alegre":                   (1.05, 0.80),
    "confusa":                  (0.97, 1.15),
    "desdem":                   (0.96, 1.10),
    "apaixonada":               (0.94, 1.20),
    "sensual":                  (0.90, 1.35),
    "calma":                    (0.97, 1.10),
    "tristeza":                 (0.90, 1.30),
    "desanimo":                 (0.92, 1.25),
    "desapontamento":           (0.93, 1.20),
    "neutra":                   (1.00, 1.00),
}


# Palavra-chave (PT/EN, o que qualquer motor de enriquecimento tende a
# escrever) -> slug mais proximo dos 17. MEDIDO 2026-08-29: o parse via
# Gemma4 (screenplay_to_video, motor diferente do Ollama/qwen3.6 usado na
# decupagem) escreve "urgente", "com respiracao curta", "determined,
# resolute" -- nenhum bate com um slug por substring, e a prosodia caia
# sempre no neutro em silencio. Isto e um classificador de PALAVRA-CHAVE, nao
# de frase: cada termo aqui e escolhido para nao colidir com os outros.
SINONIMOS_DE_EMOCAO = {
    "com_medo": ("urgente", "urgent", "urgency", "pressa", "hurried", "rushed",
                "respiracao curta", "breathless", "assustad", "afraid", "scared",
                "medo", "receoso", "apavorad"),
    "excitada": ("forca crescente", "growing strength", "crescendo",
                "building intensity", "energico", "energetic"),
    "raiva": ("determined", "resolute", "determinad", "resolut", "convicto",
             "firme", "irritad", "furios", "angry", "anger"),
    "calma": ("calm", "tranquil", "sereno", "steady", "composed"),
    "tristeza": ("sad", "triste", "melanc"),
    "alegre": ("happy", "feliz", "content"),
    "surpresa": ("surprised", "surpres", "espantad", "shocked", "chocad"),
    "espontanea_entusiasmada": ("excited", "animad", "empolgad", "entusiasmad"),
    "confusa": ("confused", "confus", "perdid"),
    "desdem": ("disdain", "desdem", "desprez"),
    "apaixonada": ("loving", "apaixonad", "amoros"),
    "sensual": ("sensual", "seductive", "seduto"),
    "desanimo": ("discouraged", "desanimad", "desalentad"),
    "desapontamento": ("disappointed", "decepcionad", "desapontad"),
    "grito": ("shouting", "screaming", "gritand", "berrand"),
    "angustia": ("anguish", "angustia", "aflit", "agoniad"),
}


def prosodia_de(emocao) -> tuple:
    """(velocidade, fator_de_pausa) para o slug de emocao da fala.

    Aceita tambem texto livre (o `instruct`, que pode vir de um parentese do
    roteiro): procura um slug conhecido dentro dele, depois um sinonimo
    conhecido (ver SINONIMOS_DE_EMOCAO). Sem correspondencia nenhuma devolve
    o neutro, entao um roteiro sem emocao reconhecivel soa exatamente como
    antes -- nunca pior, so as vezes sem melhora."""
    t = (emocao or "").strip().lower()
    if not t:
        return (1.0, 1.0)
    if t in PROSODIA:
        return PROSODIA[t]
    for slug, valores in PROSODIA.items():
        if slug in t:
            return valores
    for slug, palavras in SINONIMOS_DE_EMOCAO.items():
        if any(p in t for p in palavras):
            return PROSODIA[slug]
    return (1.0, 1.0)


def _pausa_apos(frase: str, fator: float) -> float:
    """Pausa depois desta frase, pelo sinal que a fecha."""
    f = frase.rstrip()
    for sinal, seg in PAUSA_POR_SINAL.items():
        if f.endswith(sinal):
            return seg * fator
    return PAUSA_PADRAO * fator


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


# Caracteres que o texto MANTEM para `_pausa_apos` ler o sinal certo, mas que
# NAO vao para o modelo -- a pausa ja e inserida por codigo (silencio real
# entre os segmentos), entao pedir para o XTTS "ler" a reticencia tambem e
# pedir a mesma coisa duas vezes, e a reticencia especificamente (tres
# caracteres seguidos, sem equivalente comum em fala) e o tipo de sinal que
# tokenizers de TTS mais erram -- reportado como "vocaliza a pontuacao".
_LIMPA_PARA_MODELO = [("...", ""), ("…", "")]


def _texto_para_modelo(sentence: str) -> str:
    """O texto que de fato vai para `process_tts_to_file` -- sem os sinais que
    ja viram pausa por codigo. `_pausa_apos` continua lendo a frase ORIGINAL
    (com a reticencia), so o audio nao tenta pronuncia-la."""
    t = sentence
    for velho, novo in _LIMPA_PARA_MODELO:
        t = t.replace(velho, novo)
    return t.strip() or sentence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    results = []

    from scripts.tts_funcs import TTSWrapper  # noqa: E402  (needs sys.path tweak above)

    t_load = time.time()
    tts = TTSWrapper(
        output_folder=str(XTTS_ROOT / "output"),
        speaker_folder=str(XTTS_ROOT / "speakers"),
        lowvram=False,
        model_source="local",
        model_version="v2.0.2",
        device="cuda",
    )
    tts.load_model(XTTS_ROOT)
    print(f"[xtts_worker] model loaded in {time.time() - t_load:.1f}s", flush=True)

    import numpy as np
    import soundfile as sf

    for job in jobs:
        job_id = job.get("id", "?")
        t0 = time.time()
        try:
            output_path = Path(job["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sentences = _split_sentences(job["text"])
            velocidade, fator_pausa = prosodia_de(
                job.get("emotion") or job.get("instruct"))
            # copia: DEFAULT_OPTIONS e de modulo, e alterar no lugar vazaria a
            # velocidade de uma fala para todas as seguintes do lote.
            options = dict(DEFAULT_OPTIONS, speed=round(
                DEFAULT_OPTIONS["speed"] * velocidade, 3))

            segments = []
            sample_rate = None
            for i, sentence in enumerate(sentences):
                part_path = output_path.with_name(f"{output_path.stem}.part{i}.wav")
                tts.process_tts_to_file(
                    XTTS_ROOT,
                    text=_texto_para_modelo(sentence),
                    language=job.get("language", "pt"),
                    ref_speaker_wav=job["speaker_wav"],
                    options=options,
                    file_name_or_path=str(part_path),
                )
                data, sr = sf.read(str(part_path))
                sample_rate = sr
                segments.append(data)
                part_path.unlink(missing_ok=True)

            if len(segments) > 1:
                combined = segments[0]
                for i, seg in enumerate(segments[1:]):
                    # A pausa e a que o SINAL DA FRASE ANTERIOR pede, escalada
                    # pela emocao -- por isso `sentences[i]`, nao `sentences[i+1]`.
                    dur = _pausa_apos(sentences[i], fator_pausa)
                    pause = np.zeros(int(dur * sample_rate), dtype=segments[0].dtype)
                    combined = np.concatenate([combined, pause, seg])
            else:
                combined = segments[0]
            sf.write(str(output_path), combined, sample_rate)

            results.append({
                "id": job_id, "ok": True, "output_path": str(output_path),
                "error": None, "seconds": time.time() - t0,
            })
            print(f"[xtts_worker] {job_id} ok in {time.time() - t0:.1f}s "
                  f"({len(sentences)} sentence(s), speed={options['speed']}, "
                  f"pausa x{fator_pausa}) -> {output_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "id": job_id, "ok": False, "output_path": None,
                "error": str(exc), "seconds": time.time() - t0,
            })
            print(f"[xtts_worker] {job_id} FAILED: {exc}", flush=True)

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
