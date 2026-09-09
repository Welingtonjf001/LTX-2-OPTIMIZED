"""Unified dialogue-TTS interface: engine="auto"|"xtts"|"qwen"|"fish", mirroring
the engine-selection + fallback pattern already proven in tensorxx_ge/lipsync.py
(auto/latentsync/wav2lip). xtts/qwen run in their OWN isolated venv via subprocess
(script_pipeline/tts_workers/{xtts,qwen}_worker.py) -- same isolation model as
LatentSync's external conda env. fish talks to a PERSISTENT HTTP server instead
(fish-speech/tools/api_server.py, must already be running -- see START_API.ps1),
so its worker only makes requests, it never loads a model itself.

v1 default preference for "auto": xtts (model already downloaded, zero setup, proven
in the initial smoke test). Qwen3-TTS is the richer alternative (per-line emotion via
free-text "instruct", closer mapping to screenplay parentheticals) but only its
CustomVoice checkpoint is wired up here -- see tts_workers/qwen_worker.py. fish-speech
(added 2026-09-03, MEMORIAL 3.53) is NOT in "auto"'s fallback chain on purpose --
it needs a manually-started server, unlike xtts/qwen which self-contain a model
load per call; "auto" would otherwise silently try to reach a server nobody started.
Ask for it explicitly with engine="fish" (decupagem's synthesize_dialogue.py
defaults to it since 2026-09-03 -- see EMOTION_TO_FISH_TAG for how emotion_
director's automatic per-line emotion reaches its Rich Emotion Library tags).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
WORKERS_DIR = Path(__file__).resolve().parent / "tts_workers"

XTTS_PYTHON = Path(r"E:\Users\home\Documents\xtts\webui\venv\Scripts\python.exe")
XTTS_WORKER = WORKERS_DIR / "xtts_worker.py"
XTTS_SPEAKER_DIR = Path(r"E:\Users\home\Documents\xtts\webui\speakers")

QWEN_PYTHON = Path(r"E:\Users\home\Documents\Qwen3-TTS\.venv\Scripts\python.exe")
QWEN_WORKER = WORKERS_DIR / "qwen_worker.py"
QWEN_MODEL_DIR = Path(r"E:\Users\home\Documents\Qwen3-TTS\models\0.6B-CustomVoice")

# Fish Speech (2026-09-03): instalado em E:\...\fish-speech, ja falando com a
# MESMA biblioteca de speakers do XTTS (README_LOCAL.md do proprio fish-speech).
# Ao contrario do XTTS/Qwen (carregam o modelo por chamada, dentro do worker),
# o fish-speech roda como SERVIDOR HTTP persistente -- o worker so faz
# requisicoes; subir/derrubar o servidor fica por conta de quem for usar
# (START_API.ps1), nao deste modulo. Ver MEMORIAL 3.53 pro teste completo
# (5 idiomas + tags de emocao, todos medidos funcionando) e pro bug de
# encoding CJK que precisou de PYTHONUTF8=1 no lancamento do servidor.
FISH_PYTHON = Path(r"E:\Users\home\Documents\fish-speech\.venv\Scripts\python.exe")
FISH_WORKER = WORKERS_DIR / "fish_worker.py"
FISH_API_URL = os.environ.get("FISH_API_URL", "http://127.0.0.1:8080/v1/tts")
FISH_REF_TEXT_CACHE = XTTS_SPEAKER_DIR / "_fish_ref_transcripts.json"

# emocao (slug de PROSODIA em xtts_worker.py, os mesmos 17 de emotion_director)
# -> tag da Rich Emotion Library do fish-speech. Aproximacao semantica, nao
# mapeamento 1:1 -- nem toda emocao tem tag equivalente; None = fish sintetiza
# so pela clonagem do timbre/take de referencia, sem marcador extra no texto.
EMOTION_TO_FISH_TAG = {
    "com_medo": "[panting]",
    "grito": "[screaming]",
    "raiva": "[angry]",
    "angustia": "[sigh]",
    "excitada": "[excited]",
    "espontanea_entusiasmada": "[excited tone]",
    "surpresa": "[surprised]",
    "alegre": "[delight]",
    "confusa": None,
    "desdem": None,
    "apaixonada": None,
    "sensual": "[low voice]",
    "calma": None,
    "tristeza": "[sad]",
    "desanimo": "[sigh]",
    "desapontamento": "[sigh]",
    "neutra": None,
}

# CustomVoice preset speakers with a rough gender tag, used only as a fallback when
# cast.json's per-character "qwen_speaker" is left null. Lowercase to match
# Qwen3TTSModel.get_supported_speakers()'s actual return values -- NOT the
# capitalized names in the README's speaker table (confirmed by a live call:
# ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']).
QWEN_PRESET_SPEAKERS = {
    "male": ["ryan", "aiden", "uncle_fu", "dylan", "eric"],
    "female": ["vivian", "serena", "ono_anna", "sohee"],
}

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class TTSStatus:
    xtts_available: bool
    qwen_available: bool
    fish_available: bool
    missing: tuple

    @property
    def available(self) -> bool:
        return self.xtts_available or self.qwen_available or self.fish_available


def tts_status() -> TTSStatus:
    missing = []
    xtts_ok = XTTS_PYTHON.is_file()
    if not xtts_ok:
        missing.append(str(XTTS_PYTHON))
    qwen_ok = QWEN_PYTHON.is_file() and QWEN_MODEL_DIR.is_dir()
    if not qwen_ok:
        missing.append(str(QWEN_MODEL_DIR))
    # So confere que o venv existe, NAO se o servidor esta no ar (isso o worker
    # ja checa por HTTP e falha com mensagem clara) -- mesma convencao dos
    # outros dois motores, que tambem nao verificam se o modelo carrega.
    fish_ok = FISH_PYTHON.is_file()
    if not fish_ok:
        missing.append(str(FISH_PYTHON))
    return TTSStatus(xtts_available=xtts_ok, qwen_available=qwen_ok, fish_available=fish_ok,
                     missing=tuple(missing))


def list_xtts_speakers() -> list[str]:
    if not XTTS_SPEAKER_DIR.is_dir():
        return []
    return sorted(p.stem for p in XTTS_SPEAKER_DIR.glob("*.wav"))


def default_qwen_speaker(gender: Optional[str], index: int = 0) -> str:
    pool = QWEN_PRESET_SPEAKERS.get(gender or "male", QWEN_PRESET_SPEAKERS["male"])
    return pool[index % len(pool)]


def _run_worker(python_exe: Path, worker_script: Path, jobs: list[dict], *, log: LogFn) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="dialogue_tts_") as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        results_path = Path(tmp) / "results.json"
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [str(python_exe), "-u", str(worker_script), "--jobs", str(jobs_path), "--results", str(results_path)]
        log(f"Comando: {' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
            encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            log(line.rstrip("\n"))
        process.wait()

        if not results_path.exists():
            log(f"Worker terminou (codigo {process.returncode}) sem gravar results.json.")
            return [{"id": job["id"], "ok": False, "output_path": None, "error": "worker produced no results"} for job in jobs]
        return json.loads(results_path.read_text(encoding="utf-8"))


def _xtts_job(job: dict) -> dict:
    # xtts's process_tts_to_file() treats a RELATIVE output path as relative to its
    # OWN webui/output folder, not the caller's cwd -- always resolve to absolute
    # here (in the caller's process) so that ambiguity never reaches the worker.
    return {
        "id": job["id"],
        "text": job["text"],
        "language": job.get("language_code", "pt"),
        "speaker_wav": job.get("xtts_speaker_wav") or "male",
        # A emocao chega ao XTTS por DOIS canais, nao um. O clipe de referencia
        # da o TIMBRE; este campo da o ANDAMENTO (velocidade e pausa) -- ver
        # xtts_worker.prosodia_de. Antes so o primeiro existia, e por isso uma
        # fala de panico saia no mesmo ritmo de uma fala calma.
        "emotion": job.get("emotion") or job.get("instruct"),
        "output_path": str(Path(job["output_path"]).resolve()),
    }


def _fish_ref_text(speaker_wav: str) -> str:
    """Transcricao (cacheada em disco) do que e falado no clipe de referencia,
    via faster-whisper -- o fish-speech clona melhor com o texto real do que
    sem nada (o README_LOCAL.md do proprio fish-speech deixa o campo editavel
    de proposito porque a biblioteca XTTS nao vem com transcricao nenhuma).
    MEDIDO 2026-09-03: sem isso ainda funciona (zero-shot), mas o texto real
    foi o que validou os 5 testes de idioma no MEMORIAL 3.53."""
    cache: dict = {}
    if FISH_REF_TEXT_CACHE.exists():
        try:
            cache = json.loads(FISH_REF_TEXT_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    if speaker_wav in cache:
        return cache[speaker_wav]

    from faster_whisper import WhisperModel

    modelo = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _info = modelo.transcribe(speaker_wav, language="en")
    texto = " ".join(s.text for s in segs).strip()
    cache[speaker_wav] = texto
    FISH_REF_TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FISH_REF_TEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return texto


def _fish_job(job: dict) -> dict:
    speaker_wav = job.get("xtts_speaker_wav") or str(XTTS_SPEAKER_DIR / "male.wav")
    # Emocao automatizada: o slug que emotion_director.py ja escreveu em
    # scenes_enriched.json chega aqui como job["emotion"], sem passo manual
    # nenhum -- so traduz pro vocabulario de tags do fish-speech.
    tag = EMOTION_TO_FISH_TAG.get(job.get("emotion") or "")
    texto = f"{tag} {job['text']}" if tag else job["text"]
    ref_text = ""
    if os.path.isfile(speaker_wav):
        try:
            ref_text = _fish_ref_text(speaker_wav)
        except Exception:  # noqa: BLE001 -- sem transcricao ainda sintetiza (zero-shot)
            ref_text = ""
    return {
        "id": job["id"],
        "text": texto,
        "reference_audio": speaker_wav,
        "reference_text": ref_text,
        "output_path": str(Path(job["output_path"]).resolve()),
    }


def _qwen_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "text": job["text"],
        "language": job.get("language_name", "Portuguese"),
        "speaker": job.get("qwen_speaker") or default_qwen_speaker(job.get("gender")),
        "instruct": job.get("instruct"),
        "output_path": str(Path(job["output_path"]).resolve()),
    }


def synthesize_batch(jobs: list[dict], *, engine: str = "auto", log: LogFn = _noop_log) -> list[dict]:
    """Synthesize a batch of dialogue lines. Each job needs at least: id, text,
    output_path, xtts_speaker_wav (for xtts), qwen_speaker/gender (for qwen),
    language_code ("pt"/"en"/...), language_name ("Portuguese"/"English"/...), and
    optionally "instruct" (emotion/delivery hint, used only by qwen).

    Returns one result dict per job, in the SAME order as the input, each with at
    least {"id", "ok", "output_path", "engine_used"}.
    """
    if not jobs:
        return []

    status = tts_status()
    if engine == "auto":
        primary = "xtts" if status.xtts_available else ("qwen" if status.qwen_available else None)
    else:
        primary = engine

    disponivel = {"xtts": status.xtts_available, "qwen": status.qwen_available,
                  "fish": status.fish_available}
    if primary is None or not disponivel.get(primary, False):
        log(f"Motor de TTS '{primary}' indisponivel: {status.missing}")
        return [{"id": job["id"], "ok": False, "output_path": None, "engine_used": None, "error": "engine unavailable"} for job in jobs]

    log(f"Sintetizando {len(jobs)} fala(s) via {primary}...")
    if primary == "xtts":
        raw_results = _run_worker(XTTS_PYTHON, XTTS_WORKER, [_xtts_job(j) for j in jobs], log=log)
    elif primary == "fish":
        # Servidor persistente, nao carrega modelo por chamada -- se nao
        # estiver de pe (START_API.ps1), o worker falha job a job com
        # mensagem clara em vez de travar tentando subir sozinho.
        raw_results = _run_worker(FISH_PYTHON, FISH_WORKER, [_fish_job(j) for j in jobs], log=log)
    else:
        raw_results = _run_worker(QWEN_PYTHON, QWEN_WORKER, [_qwen_job(j) for j in jobs], log=log)

    by_id = {r["id"]: r for r in raw_results}
    results = []
    failed_jobs = []
    for job in jobs:
        result = by_id.get(job["id"], {"id": job["id"], "ok": False, "output_path": None, "error": "missing from worker output"})
        result["engine_used"] = primary
        results.append(result)
        if not result.get("ok"):
            failed_jobs.append(job)

    # engine="auto" gets one fallback pass through the other engine for anything that
    # failed -- same spirit as lipsync.py falling back from LatentSync to Wav2Lip.
    if engine == "auto" and failed_jobs:
        fallback = "qwen" if primary == "xtts" else "xtts"
        fallback_available = status.qwen_available if fallback == "qwen" else status.xtts_available
        if fallback_available:
            log(f"{len(failed_jobs)} fala(s) falharam em {primary}; tentando {fallback}...")
            if fallback == "xtts":
                fb_raw = _run_worker(XTTS_PYTHON, XTTS_WORKER, [_xtts_job(j) for j in failed_jobs], log=log)
            else:
                fb_raw = _run_worker(QWEN_PYTHON, QWEN_WORKER, [_qwen_job(j) for j in failed_jobs], log=log)
            fb_by_id = {r["id"]: r for r in fb_raw}
            for i, result in enumerate(results):
                if not result.get("ok") and result["id"] in fb_by_id:
                    fb_result = fb_by_id[result["id"]]
                    fb_result["engine_used"] = fallback
                    results[i] = fb_result

    return results
