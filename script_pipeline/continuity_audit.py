"""Contrato de continuidade por plano.

Esta auditoria existe para impedir que a decupagem perca o espaço, o avião,
os objetos já apresentados ou os personagens secundários entre planos. Ela
audita o contrato textual que alimenta o still/I2V e também registra quais
arquivos foram usados; não finge que um classificador visual consegue provar
sozinho que um objeto está no quadro. Falhas estruturais bloqueiam a passada
de vídeo e ficam em ``shots/continuity_audit.json``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# Deliberately opt-in. These terms are evaluated only when a plan carries an
# explicit `aircraft_contract`; ordinary scripts never inherit aviation rules.
AIRCRAFT_TERMS = ("same twin-engine passenger jet", "airborne", "in flight", "flying")

OBJECT_ALIASES = {
    "cockpit_panel": ("instrument panel", "control panel", "painel de instrumentos"),
    "windshield": ("windshield", "para-brisa"),
    "weather_radar": ("weather radar", "radar de tempo", "radar"),
    "aisle": ("single central aisle", "single aisle", "corredor central", "corredor"),
    "seat_3_3": ("3-3 seating", "3-3 seats", "assentos 3-3"),
    "overhead_bins": ("overhead bin", "overhead bins", "compartimento superior", "bagageiro"),
    "meal_carts": ("meal cart", "meal carts", "service cart", "carrinho"),
    "jumpseats": ("jumpseat", "jumpseats", "assento de salto"),
    "harnesses": ("four-point harness", "harnesses", "cinto de quatro pontos"),
    "seatbelt_signs": ("seatbelt signs", "sinal de cinto"),
    "emergency_strips": ("emergency strips", "faixas de emergência"),
    "row_19_bin": ("row 19", "bin 19", "fileira 19"),
    "backpack": ("backpack", "mochila"),
    "galleys": ("forward galley", "aft galley", "galley", "galley dianteira", "galley traseira"),
    "storm_clouds": ("storm clouds", "dense storm clouds", "nuvens de tempestade"),
    "passengers": ("passengers", "passageiros", "passengers seated"),
}


def _text(value) -> str:
    return str(value or "").casefold()


def _contains(text: str, aliases) -> bool:
    return any(alias.casefold() in text for alias in aliases)


def _objects(text: str) -> set[str]:
    return {key for key, aliases in OBJECT_ALIASES.items() if _contains(text, aliases)}


def _roster(plan: dict, scene_number: int) -> set[str]:
    """Elenco esperado da cena, vindo do contrato e não do próprio resultado.

    Derivar a lista somente de ``subject``/``co_subject`` tornava a auditoria
    circular: um secundário omitido do plano também sumia da lista esperada e
    ``secondary_character_missing`` nunca podia disparar.
    """
    contracted: set[str] = set()
    appeared: set[str] = set()
    for shot in plan.get("shots", []):
        if shot.get("scene") != scene_number:
            continue
        for value in (shot.get("continuity_contract") or {}).get("scene_characters", []):
            if value:
                contracted.add(str(value))
        for field in ("subject", "co_subject"):
            value = shot.get(field)
            if value:
                appeared.add(str(value))
    # Projetos antigos não carregam scene_characters; neles preservamos o
    # comportamento compatível, sem inventar um elenco ausente do contrato.
    return contracted or appeared


def build_report(run_dir: str | Path, *, block_on_missing: bool = True) -> dict:
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "parse" / "shot_plan.json"
    if not plan_path.exists():
        return {"status": "missing_plan", "blocking": ["parse/shot_plan.json"]}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # Deliberately opt-in, mesmo padrao do AIRCRAFT_TERMS acima: so exige location_id/continuity
    # quando o projeto REALMENTE os preenche (Voo 702, com location_master/biblia). Um roteiro
    # avulso sem projeto mestre nunca ganha esses campos do shot_plan -- bloquear nesse caso e
    # falso positivo em TODO plano, nao continuidade quebrada (achado rodando de verdade, 2026-09-21).
    plan_uses_location_contracts = any(s.get("location_id") for s in plan.get("shots", []))
    groups = defaultdict(list)
    for shot in plan.get("shots", []):
        groups[int(shot.get("scene", 0))].append(shot)
    scenes = []
    blocking = []
    warnings = []
    for scene_no, shots in sorted(groups.items()):
        roster = _roster(plan, scene_no)
        locations = {s.get("location_id") for s in shots if s.get("location_id")}
        if plan_uses_location_contracts and len(locations) != 1:
            blocking.append({"scene": scene_no, "type": "location_drift",
                             "locations": sorted(x for x in locations if x)})
        scene_objects: set[str] = set()
        for shot in shots:
            scene_objects |= _objects(" ".join(str(shot.get(k, "")) for k in
                                               ("storyboard_prompt", "video_prompt", "continuity")))
        presented: set[str] = set()
        shot_reports = []
        for shot in sorted(shots, key=lambda s: int(s.get("index", 0))):
            idx = int(shot.get("index", -1))
            blob = " ".join(str(shot.get(k, "")) for k in
                            ("storyboard_prompt", "video_prompt", "continuity", "location"))
            lower = _text(blob)
            missing_location = plan_uses_location_contracts and (
                not shot.get("location_id") or not shot.get("continuity"))
            if missing_location:
                blocking.append({"scene": scene_no, "shot": idx, "type": "missing_location_contract"})
            aircraft_required = bool(
                (shot.get("continuity_contract") or {}).get("aircraft") or
                (shot.get("continuity") or {}).get("aircraft_contract")
            )
            if aircraft_required:
                identity_ok = "twin-engine passenger jet" in lower and "same" in lower
                airborne_ok = any(term in lower for term in ("airborne", "in flight", "flying"))
                missing_aircraft = ([] if identity_ok else ["same twin-engine passenger jet"])
                if not airborne_ok:
                    missing_aircraft.append("airborne/in flight")
            else:
                missing_aircraft = []
            # Só planos com contrato explícito recebem esta regra. No Voo 702
            # o perfil coloca o contrato em todos os planos; outros roteiros e
            # cenas do mesmo projeto não herdam aviação por proximidade.
            if missing_aircraft:
                blocking.append({"scene": scene_no, "shot": idx, "type": "aircraft_state_drift",
                                 "missing": missing_aircraft})
            expected = _objects(lower) | presented
            # A close-up may legitimately crop a persistent object. Wide,
            # medium, OTS and insert shots must keep the previously presented
            # inventory in their conditioning text.
            out_of_frame = str(shot.get("framing", "")).casefold() == "close"
            missing_objects = sorted(expected - _objects(lower)) if not out_of_frame else []
            declared_objects = (shot.get("continuity_contract") or {}).get("persistent_objects", [])
            missing_declared = []
            if not out_of_frame:
                for obj in declared_objects:
                    words = [w for w in re.findall(r"[\w-]+", _text(obj)) if len(w) > 2]
                    if words and not all(w in lower for w in words[:2]):
                        missing_declared.append(obj)
            missing_objects = sorted(set(missing_objects + missing_declared))
            if missing_objects:
                warnings.append({"scene": scene_no, "shot": idx, "type": "object_not_carried",
                                 "objects": missing_objects})
            presented |= _objects(lower)
            shot_reports.append({"shot": idx, "location_id": shot.get("location_id"),
                                 "framing": shot.get("framing"), "characters": sorted(
                                     x for x in (shot.get("subject"), shot.get("co_subject")) if x),
                                 "objects": sorted(_objects(lower)),
                                 "aircraft_in_flight": not missing_aircraft,
                                 "out_of_frame_allowed": out_of_frame,
                                 "missing_objects": missing_objects})
        # Secondary characters are scene-level continuity: every named person
        # must have an explicit planned appearance and be listed in the scene
        # contract, even when a given close-up crops them out.
        appeared = {name for report in shot_reports for name in report["characters"]}
        missing_secondary = sorted(roster - appeared)
        if missing_secondary:
            blocking.append({"scene": scene_no, "type": "secondary_character_missing",
                             "characters": missing_secondary})
        scenes.append({"scene": scene_no, "location_ids": sorted(x for x in locations if x),
                       "characters": sorted(roster), "secondary_characters": missing_secondary,
                       "objects_presented": sorted(scene_objects), "shots": shot_reports})

    report = {"status": "blocked" if blocking and block_on_missing else "ok",
              "blocking": blocking, "warnings": warnings, "scenes": scenes,
              "rules": {"location": "one location_id per scene", "aircraft": list(AIRCRAFT_TERMS),
                        "secondary_characters": "scene roster must be represented in planned shots",
                        "objects": "persistent objects are carried in prompt; close-ups may crop them"}}
    out = run_dir / "shots" / "continuity_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_summary(report: dict) -> str:
    if report.get("status") == "missing_plan":
        return "[continuity_audit] shot_plan ausente."
    return (f"[continuity_audit] status={report.get('status')}; "
            f"{sum(len(s.get('shots', [])) for s in report.get('scenes', []))} plano(s), "
            f"{len(report.get('blocking', []))} bloqueio(s), "
            f"{len(report.get('warnings', []))} aviso(s).")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    report = build_report(args.run_dir, block_on_missing=not args.report_only)
    print(format_summary(report))
    return 1 if report.get("status") == "blocked" and not args.report_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
