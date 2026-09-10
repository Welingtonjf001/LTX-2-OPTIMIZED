"""Relatorio agregado dos stills de uma corrida -- ponto de revisao ANTES do
estagio de video, pedido do usuario 2026-09-10 depois de dois defeitos reais
passarem direto pro vídeo sem aviso nenhum: um still com dois personagens
onde o segundo saiu com a cara clonada do primeiro, e um drift de estilo
fotorrealista<->3D entre planos da MESMA corrida.

Junta o que ja e' medido por still (`consistency_score` de
`consistency_audit.check_consistency`, `duplicate_flag` de
`consistency_audit.detect_duplicate_faces`, ambos gravados em
`shots/stills/stills.json` por `render_shots.py`) num resumo unico, legivel,
que aponta EXATAMENTE quais planos merecem revisao humana antes de gastar
GPU gerando video -- nunca bloqueia a corrida sozinho (mesma filosofia de
toda auditoria deste pipeline: reporta, nao decide por quem revisa).

CLI standalone, para auditar uma corrida depois do fato:
    python -m script_pipeline.storyboard_audit --run-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Mesmo limiar que consistency_audit usa para "e a mesma pessoa" -- um score
# abaixo disso e' um still que PODE ter perdido a identidade do personagem
# (ou, como medido no MEMORIAL, um plano wide/perfil onde a metrica so mede
# mal mesmo com a identidade certa -- por isso "suspeito", nao "errado").
SCORE_SUSPEITO = 0.35


def build_report(run_dir: Path) -> dict:
    """Le `shots/stills/stills.json` e monta o resumo. Devolve {} se o
    manifesto ainda nao existir (corrida nao chegou nos stills)."""
    manifest_path = Path(run_dir) / "shots" / "stills" / "stills.json"
    if not manifest_path.exists():
        return {}
    manifesto = json.loads(manifest_path.read_text(encoding="utf-8"))

    planos_com_score = []
    scores_suspeitos = []
    planos_duplicados = []
    for idx, info in manifesto.items():
        score = info.get("consistency_score")
        if score is not None:
            planos_com_score.append(score)
            if score < SCORE_SUSPEITO:
                scores_suspeitos.append({"plano": int(idx), "score": score, "file": info.get("file")})
        if info.get("duplicate_flag"):
            planos_duplicados.append({
                "plano": int(idx), "file": info.get("file"),
                "rostos_detectados": info.get("duplicate_faces_detected"),
            })

    return {
        "total_stills": len(manifesto),
        "medidos_para_consistencia": len(planos_com_score),
        "score_medio": (sum(planos_com_score) / len(planos_com_score)) if planos_com_score else None,
        "planos_score_suspeito": sorted(scores_suspeitos, key=lambda p: p["score"]),
        "planos_duplicidade_suspeita": planos_duplicados,
    }


def format_summary(report: dict) -> str:
    """Texto curto pronto pro log -- o que a pessoa le antes de decidir se
    revisa a galeria ou segue direto pro video."""
    if not report:
        return "[storyboard_audit] sem stills.json ainda -- nada para auditar."
    linhas = [
        f"[storyboard_audit] {report['total_stills']} still(s), "
        f"{report['medidos_para_consistencia']} medido(s) para consistencia"
        + (f", score medio {report['score_medio']:.3f}" if report["score_medio"] is not None else "") + "."
    ]
    if report["planos_duplicidade_suspeita"]:
        nomes = ", ".join(f"plano {p['plano']} ({p['file']})" for p in report["planos_duplicidade_suspeita"])
        linhas.append(f"[storyboard_audit] ALERTA -- DUPLICIDADE DE ROSTO suspeita em: {nomes} -- "
                       f"confira antes de gerar video (pode ser 2 personagens com a mesma cara).")
    if report["planos_score_suspeito"]:
        nomes = ", ".join(f"plano {p['plano']} ({p['score']:.3f})" for p in report["planos_score_suspeito"])
        linhas.append(f"[storyboard_audit] ALERTA -- {len(report['planos_score_suspeito'])} plano(s) com "
                       f"consistencia facial abaixo de {SCORE_SUSPEITO}: {nomes}.")
    if not report["planos_duplicidade_suspeita"] and not report["planos_score_suspeito"]:
        linhas.append("[storyboard_audit] nenhum problema detectado -- ok para seguir pro video.")
    return "\n".join(linhas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    report = build_report(run_dir)
    summary = format_summary(report)
    print(summary)
    if report:
        out_path = run_dir / "shots" / "storyboard_audit.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[storyboard_audit] relatorio completo em {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
