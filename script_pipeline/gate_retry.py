"""Laco de regeneracao dos planos reprovados pelo gate visual.

Antes (2026-09-18) o gate so DETECTAVA: um plano reprovado travava o filme
inteiro depois de todos os clipes terem sido gerados (3-40 min de GPU cada).
Agora, quando o gate reprova, so os planos reprovados sao refeitos com outra
seed (o cache por conteudo invalida sozinho, porque a seed entra na chave do
still e do clipe) e reauditados -- ate `max_retries` rodadas.

Os modelos NAO ficam trocando de lugar a cada plano: cada rodada e uma passada
de geracao so dos reprovados + uma passada do gate so deles. O relatorio do
gate mescla por plano (ver visual_continuity_audit.merge_results), entao a
reauditoria parcial nunca apaga a completa.
"""
from __future__ import annotations

import json
from pathlib import Path

SEED_STEP = 7919  # primo grande: rodadas nunca colidem com seed + indice do plano


def format_shot_spec(shots: list[int]) -> str:
    """[0,2,3,4,7] -> "0,2-4,7" (o formato de --only-shots)."""
    shots = sorted(set(int(s) for s in shots))
    if not shots:
        return ""
    partes, ini, prev = [], shots[0], shots[0]
    for s in shots[1:]:
        if s == prev + 1:
            prev = s
            continue
        partes.append(str(ini) if ini == prev else f"{ini}-{prev}")
        ini = prev = s
    partes.append(str(ini) if ini == prev else f"{ini}-{prev}")
    return ",".join(partes)


def failed_checks(entry: dict) -> list[str]:
    """Nomes das checagens que reprovaram, do texto "failed checks: a, b" do gate."""
    for reason in reversed(entry.get("reasons") or []):
        if str(reason).startswith("failed checks:"):
            return [x.strip() for x in str(reason).split(":", 1)[1].split(",") if x.strip()]
    return []


def repair_note(entry: dict, shot: dict) -> str:
    """Instrucao curta de CORRECAO a partir das checagens reprovadas (vazia = sem reparo por edicao).

    Regenerar com outra semente joga fora tudo o que estava certo no still; um motor com edicao
    nativa (Qwen-Image-2.1) recebe o still e so a lista do que corrigir."""
    if entry.get("auditor_error"):
        return ""
    checks = set(failed_checks(entry))
    contract = shot.get("continuity_contract") or {}
    parts = []
    if "forbidden_text_detected" in checks:
        parts.append("Remove any caption, subtitle, watermark or invented lettering.")
    if "location_match" in checks:
        place = shot.get("location") or shot.get("location_id") or "the scene location"
        parts.append(f"The location must clearly be {place}, with its fixed layout.")
    if "aircraft_state_match" in checks:
        parts.append("The aircraft must be airborne above clouds; no ground, runway, tarmac or landing gear.")
    if checks & {"identity_match", "subjects_match"} and shot.get("subject"):
        parts.append(f"{shot['subject']}'s face must clearly match the reference portrait and be fully visible.")
    if "speaker_remains_visible" in checks:
        parts.append("The speaking person stays in the foreground with the face fully visible.")
    if "persistent_objects_match" in checks and contract.get("persistent_objects"):
        parts.append("Show at least one of: " + ", ".join(str(o) for o in contract["persistent_objects"][:4]) + ".")
    return " ".join(parts)


def write_repair_notes(run: Path, stage: str, shots: list[int]) -> dict:
    """Grava shots/gate_repair_notes.json para os planos reprovados (so stills; o clipe nao se edita)."""
    if stage != "stills":
        return {}
    run = Path(run)
    try:
        report = json.loads((run / "shots" / "visual_stills_audit.json").read_text(encoding="utf-8"))
        plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    por_plano = {int(r["shot"]): r for r in report.get("results", [])}
    notas = {}
    for i in shots:
        entry = por_plano.get(int(i))
        if entry and not entry.get("pass") and 0 <= int(i) < len(plan["shots"]):
            nota = repair_note(entry, plan["shots"][int(i)])
            if nota:
                notas[str(int(i))] = nota
    path = run / "shots" / "gate_repair_notes.json"
    if notas:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()
    return notas


def _write_seed_overrides(run: Path, shots: list[int], seed: int) -> None:
    """Persiste a semente de cada plano refeito (lida por render_shots.render)."""
    path = Path(run) / "shots" / "seed_overrides.json"
    try:
        atual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        atual = {}
    for s in shots:
        atual[str(int(s))] = int(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(atual, indent=2), encoding="utf-8")


def gate_with_retries(run: Path, *, stage: str, audit_cmd: list, regen_cmd, run_step,
                      blocked_fn, base_seed: int, max_retries: int, expand_fn=None, notes_fn=None,
                      total_shots: int = 0, max_regen_fraction: float = 0.4, log=print) -> bool:
    """Roda o gate; se reprovar, refaz SO os reprovados e reaudita.

    audit_cmd: comando do gate completo (list). Para reauditar so os refeitos,
        acrescenta ["--only-shots", spec].
    regen_cmd(spec, seed) -> list: comando que regenera os planos de `spec`.
    run_step(nome, cmd, obrigatorio=True) -> bool (tipicamente `passo`).
    blocked_fn() -> list[int]: planos reprovados no ultimo relatorio.
    expand_fn(shots) -> list[int]: opcional; acrescenta dependentes (ex.: planos que usaram
        como referencia de locacao um still reprovado -- ver location_master.py).
    Devolve True se o gate acabou aprovado."""
    ok = run_step(f"gate {stage}", audit_cmd)
    rounds = []
    attempt = 0
    while not ok and attempt < max_retries:
        shots = blocked_fn()
        if not shots:
            # Reprovou sem plano identificavel (runtime fora do ar, relatorio
            # parcial): regenerar nao ajuda, e isso nao e defeito de imagem.
            log(f"[gate {stage}] reprovado sem planos identificaveis; sem regeneracao.")
            break
        if expand_fn is not None:
            shots = expand_fn(shots)
        if total_shots and len(shots) > max_regen_fraction * total_shots:
            # Reprovar mais que ~40% do filme numa rodada e sintoma de GATE calibrado
            # errado, nao de 27 imagens ruins: regenerar em massa so queima horas de GPU
            # (visto 2026-09-19: 2 rodadas x 68 stills e 60% seguiam reprovados).
            log(f"[gate {stage}] {len(shots)}/{total_shots} planos reprovados "
                f"(> {max_regen_fraction:.0%}): gate suspeito, SEM regeneracao em massa. Revise o relatorio.")
            break
        attempt += 1
        seed = base_seed + SEED_STEP * attempt
        spec = format_shot_spec(shots)
        log(f"[gate {stage}] rodada {attempt}/{max_retries}: refazendo planos {spec} (seed {seed})")
        rounds.append({"round": attempt, "shots": shots, "seed": seed})
        _write_seed_overrides(run, shots, seed)
        if notes_fn is not None:
            notes_fn(shots)
        # obrigatorio=True aqui NAO aborta a corrida (este run_step e local a esta
        # funcao) -- so faz o retorno dizer a verdade sobre o subprocesso, para
        # distinguir "regeneracao crashou" (reauditar de novo o still velho e
        # desperdicio) de "gerou e ainda reprovou".
        regen_ok = run_step(f"regenerar {stage} ({spec})", regen_cmd(spec, seed), obrigatorio=True)
        if not regen_ok:
            log(f"[gate {stage}] rodada {attempt}: a regeneracao falhou (still/clipe antigo mantido); "
                f"reauditar contra ele so gastaria uma rodada -- pulando a reauditoria desta rodada.")
            rounds[-1]["regen_failed"] = True
            ok = False
            continue
        ok = run_step(f"gate {stage} (reauditoria {spec})", audit_cmd + ["--only-shots", spec],
                      obrigatorio=True)
    if ok:
        (Path(run) / "shots" / "gate_repair_notes.json").unlink(missing_ok=True)
    restante = [] if ok else blocked_fn()
    out = Path(run) / "shots" / f"visual_gate_{stage}_retries.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage": stage, "max_retries": max_retries, "rounds": rounds,
                               "approved": bool(ok), "unresolved": restante},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok:
        log(f"[gate {stage}] {len(restante)} plano(s) seguem reprovados apos "
            f"{attempt} rodada(s): {format_shot_spec(restante) or '-'}")
    return bool(ok)
