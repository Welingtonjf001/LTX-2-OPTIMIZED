"""Fish Speech batch worker. Roda DENTRO do venv proprio do fish-speech (nao
o principal do LTX) -- mesmo motivo do xtts_worker.py: ormsgpack/requests/
fish_speech.utils.schema so existem la, e chamar via subprocesso evita ter
que casar as dependencias dos dois projetos.

Ao contrario do xtts_worker (carrega o modelo em processo), o fish-speech
roda como SERVIDOR HTTP persistente (`tools/api_server.py`, ver README_LOCAL.md
do fish-speech) -- este worker so faz requisicoes, nao carrega peso nenhum.
Precisa do servidor já no ar (`START_API.ps1`); se nao estiver, cada job falha
com uma mensagem clara em vez de tentar subir o servidor sozinho (carga do
modelo leva ~1 min e usa ~22 GB de VRAM -- decisao de subir fica com quem
chama, nao com o worker).

jobs.json: list of {"id": str, "text": str, "reference_audio": str (caminho
absoluto do wav), "reference_text": str (transcricao do reference_audio --
ver dialogue_tts._fish_ref_text, feito via faster-whisper do lado de fora),
"output_path": str}.
results.json (escrito ao sair): list of {"id", "ok", "output_path", "error",
"seconds"}.

BUGFIX 2026-09-03: o fish-speech tem um print() de debug (content_sequence.py
::print_in_green) que quebra com UnicodeEncodeError em texto CJK quando o
processo nao esta em UTF-8 -- MEDIDO contra coreano e chines, ambos falhando
com 'charmap' codec antes deste fix. `PYTHONUTF8`/`PYTHONIOENCODING` tem que
estar setados no AMBIENTE do servidor (nao deste worker -- o bug e no lado do
servidor); ver dialogue_tts.py / MEMORIAL 3.53 pra como o servidor deve subir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FISH_API_URL = os.environ.get("FISH_API_URL", "http://127.0.0.1:8080/v1/tts")


def _synthesize_one(job: dict) -> bytes:
    import ormsgpack
    import requests

    ref_audio = job.get("reference_audio")
    ref_text = job.get("reference_text") or ""
    references = []
    if ref_audio and os.path.isfile(ref_audio):
        with open(ref_audio, "rb") as f:
            references.append({"audio": f.read(), "text": ref_text})

    payload = {
        "text": job["text"],
        "references": references,
        "reference_id": None,
        "format": "wav",
        "latency": "normal",
        "max_new_tokens": 1024,
        "chunk_length": 300,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "streaming": False,
        "use_memory_cache": "off",
        "seed": None,
    }
    resp = requests.post(
        FISH_API_URL,
        params={"format": "msgpack"},
        data=ormsgpack.packb(payload),
        headers={"content-type": "application/msgpack"},
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    results = []

    try:
        import requests

        requests.get(FISH_API_URL.rsplit("/v1/", 1)[0] + "/v1/health", timeout=5)
    except Exception as exc:  # noqa: BLE001
        print(f"[fish_worker] servidor do fish-speech nao respondeu em {FISH_API_URL} "
              f"({exc}) -- suba com START_API.ps1 antes de usar este motor.", flush=True)
        for job in jobs:
            results.append({"id": job.get("id", "?"), "ok": False, "output_path": None,
                            "error": "servidor fish-speech indisponivel", "seconds": 0.0})
        Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    for job in jobs:
        job_id = job.get("id", "?")
        t0 = time.time()
        try:
            audio_bytes = _synthesize_one(job)
            output_path = Path(job["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            results.append({"id": job_id, "ok": True, "output_path": str(output_path),
                            "error": None, "seconds": time.time() - t0})
            print(f"[fish_worker] {job_id} ok in {time.time() - t0:.1f}s -> {output_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"id": job_id, "ok": False, "output_path": None,
                            "error": str(exc), "seconds": time.time() - t0})
            print(f"[fish_worker] {job_id} FAILED: {exc}", flush=True)

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
