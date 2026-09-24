"""Referencia mestra de locacao: nunca adotar (nem manter) um still reprovado.

ACHADO 2026-09-19 (Voo 702): `render_shots` registra o PRIMEIRO still de cada
locacao em `shots/location_refs.json` e o reaproveita como referencia dos
planos seguintes. Em `LOC_SKY` esse primeiro still foi o plano 27 -- o aviao
na pista, exatamente o que o gate reprovou depois. A referencia errada
contaminou os planos 28-32 (mesmo aviao, mesma pista) e o gate, que compara
contra essa mesma referencia, passou a medir "consistencia com o erro".

Aqui: quando o gate reprova o plano que e a referencia de uma locacao, a
referencia e removida (o proximo still da locacao a re-registra) e os planos
que dependiam dela entram na regeneracao.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _refs_path(run: Path) -> Path:
    return Path(run) / "shots" / "location_refs.json"


def ref_shot_index(path: str) -> int | None:
    m = re.search(r"shot(\d+)_", Path(str(path)).name)
    return int(m.group(1)) if m else None


def invalidate_bad_refs(run: Path, blocked: list[int]) -> dict[str, int]:
    """Remove de location_refs.json as referencias que SAO um plano reprovado.

    Devolve {chave_da_locacao: indice_do_plano_que_era_a_referencia}."""
    path = _refs_path(run)
    if not path.exists():
        return {}
    refs = json.loads(path.read_text(encoding="utf-8"))
    bloqueados = set(int(b) for b in blocked)
    removidas = {}
    for key, raw in list(refs.items()):
        idx = ref_shot_index(raw)
        if idx is not None and idx in bloqueados:
            removidas[key] = idx
            del refs[key]
    if removidas:
        path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    return removidas


def dependents(plan: dict, removidas: dict[str, int]) -> list[int]:
    """Planos POSTERIORES da mesma locacao: foram gerados com a referencia ruim."""
    out = []
    for idx, shot in enumerate(plan.get("shots", [])):  # os arquivos shotNNN seguem a POSICAO
        chave = str(shot.get("location_id") or shot.get("scene"))
        if chave in removidas and idx > removidas[chave]:
            out.append(idx)
    return sorted(out)


def expand_blocked(run: Path, blocked: list[int]) -> list[int]:
    """blocked + dependentes das referencias invalidadas (usado pelo laco de regeneracao)."""
    removidas = invalidate_bad_refs(run, blocked)
    if not removidas:
        return sorted(set(blocked))
    plan = json.loads((Path(run) / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    return sorted(set(blocked) | set(dependents(plan, removidas)))
