"""Auditoria de RITMO: planos longos ou curtos demais para o que a acao pede.

Faltava uma camada que julgasse a duracao de planos de ACAO (sem fala) -- ate aqui so existia
controle de duracao para FALA (`speech_split.py`, corta acima de `--max-speech-seconds`) e um piso
para orbita de camera (`ORBIT_MIN_SECONDS` em `shot_plan.py`). Isso deixava passar planos de acao
esticados ou cortados sem ninguem avisar (ex.: 2,1s fixos no estilo `classico` para um wide de acao
com perseguicao, ou 6s+ parado num close sem fala).

So RELATA -- nunca bloqueia nem encurta/estica um plano sozinho (duracao errada as vezes e escolha
deliberada de direcao). Roda depois do `shot_plan` (nao depende de still/video prontos).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Faixas plausiveis por enquadramento, em segundos -- vocabulario de cinema comum, nao ciencia
# exata: um close costuma ser mais curto que um wide (menos para o olho explorar), um insert e
# quase sempre rapido (chama atencao pra 1 coisa so). Accao dentro do PROPRIO texto do plano (fuga,
# perseguicao, tiro, queda, explosao) pede o minimo mais alto do intervalo -- comprimir isso corta
# o movimento pela metade.
FRAMING_RANGES = {
    "extreme_close": (0.6, 3.0),
    "close": (0.8, 5.0),
    "medium": (1.0, 6.0),
    "ots": (1.0, 6.0),
    "wide": (1.5, 8.0),
    "full": (1.5, 8.0),
    "insert": (0.4, 2.5),
    "establishing": (2.0, 10.0),
}
DEFAULT_RANGE = (0.8, 8.0)

ACTION_KEYWORDS = (
    "runs", "running", "chases", "chasing", "sprint", "sprints", "jumps", "jumping", "leaps",
    "dives", "explosion", "explodes", "gunfire", "gunshot", "shoots", "shooting", "fires",
    "falls", "falling", "crashes", "collides", "struggles", "fights", "fighting", "tackles",
    "corre", "correndo", "persegue", "perseguicao", "salta", "pula", "mergulha", "explode",
    "explosao", "tiro", "disparo", "atira", "cai", "queda", "colide", "luta", "briga",
)


def _range_for(shot: dict) -> tuple[float, float]:
    lo, hi = FRAMING_RANGES.get(str(shot.get("framing", "")).casefold(), DEFAULT_RANGE)
    blob = " ".join(str(shot.get(k, "")) for k in ("video_prompt", "storyboard_prompt", "action")).casefold()
    if any(kw in blob for kw in ACTION_KEYWORDS):
        # Acao textual pede pelo menos o dobro do minimo do enquadramento: um wide de
        # perseguicao em 1,5s corta o movimento antes de comecar a se ler.
        lo = max(lo, min(hi, lo * 2.0))
    return lo, hi


def audit(plan: dict) -> dict:
    findings = []
    for shot in plan.get("shots", []):
        seconds = float(shot.get("seconds") or 0)
        if seconds <= 0:
            continue
        lo, hi = _range_for(shot)
        if shot.get("line_index") is not None:
            continue  # fala tem controle proprio (speech_split); nao duplicar aviso aqui
        if seconds < lo:
            findings.append({"shot": shot.get("position", shot.get("index")), "framing": shot.get("framing"),
                             "seconds": seconds, "expected_min": lo, "expected_max": hi,
                             "type": "too_short",
                             "reason": f"{seconds:.2f}s abaixo do minimo plausivel ({lo:.2f}s) "
                                       f"para {shot.get('framing')}"})
        elif seconds > hi:
            findings.append({"shot": shot.get("position", shot.get("index")), "framing": shot.get("framing"),
                             "seconds": seconds, "expected_min": lo, "expected_max": hi,
                             "type": "too_long",
                             "reason": f"{seconds:.2f}s acima do maximo plausivel ({hi:.2f}s) "
                                       f"para {shot.get('framing')}"})
    total = sum(float(s.get("seconds") or 0) for s in plan.get("shots", []))
    return {"status": "ok" if not findings else "warnings", "findings": findings,
            "total_seconds": round(total, 2), "shots_audited": len(plan.get("shots", []))}


def summary(report: dict) -> str:
    if report["status"] == "ok":
        return f"[pacing] ok -- {report['shots_audited']} plano(s), {report['total_seconds']:.1f}s total"
    curtos = sum(1 for f in report["findings"] if f["type"] == "too_short")
    longos = sum(1 for f in report["findings"] if f["type"] == "too_long")
    linhas = "\n".join(f"  plano {f['shot']} ({f['framing']}): {f['reason']}" for f in report["findings"])
    return (f"[pacing] {len(report['findings'])} aviso(s) de ritmo ({curtos} curto(s), {longos} "
            f"longo(s) demais) de {report['shots_audited']} plano(s):\n{linhas}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Auditoria de ritmo (planos longos/curtos demais)")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    report = audit(plan)
    out = run / "shots"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pacing_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
