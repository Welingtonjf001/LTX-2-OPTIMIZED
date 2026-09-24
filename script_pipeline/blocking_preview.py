"""Pre-visualizacao BARATA de blocking/movimento, antes de gastar still ou video.

Pedido do usuario (2026-09-23, depois do CERCO EM SEUL sair ruim com still/video reais): uma etapa
prevalidacao dos planos de MOVIMENTO e POSICIONAMENTO que nao dependa de FLUX/Qwen nem de LTX --
so os numeros que `motion_conditioner.py --apply` ja calcula (`parse/motion_plan.json`:
`formation` com `anchor_m`/`screen_side` por ator, e `commands` com `primitive`/`target` por
plano). Nao e um substituto da auditoria visual (essa audita PIXEL contra o still/clipe reais);
e um diagrama esquematico top-down por plano, gerado em segundos, para pegar erro de POSICIONAMENTO
e de DIRECAO antes de qualquer GPU de difusao rodar.

Roda depois do estagio `motion` (precisa de motion_plan.json) e antes de `sheet`/`stills` -- opt-in,
so quando `--motion-conditioning` estiver ligado (sem motion_plan.json, nao ha o que desenhar)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 320, 220
MARGIN = 12
METERS_TO_PX = 40  # 1 m de anchor_m -> 40 px no diagrama
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]

# Primitivas que envolvem DOIS personagens no mesmo eixo (perseguicao, queda, resgate, protecao) --
# o diagrama desenha uma seta actor->partner com rotulo, para o olho notar rapido se o alvo faz
# sentido (personagens no mesmo lado da tela nunca deveriam "perseguir" um ao outro de frente).
PARTNERED = {"pursue", "contact", "takedown", "protect", "reach"}


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _actor_color(names: list[str], name: str | None) -> str:
    if not name:
        return "#888888"
    return COLORS[names.index(name) % len(COLORS)] if name in names else "#888888"


def _panel(command: dict, formation: dict, names: list[str], shot_meta: dict | None) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, PANEL_H), "#101014")
    draw = ImageDraw.Draw(img)
    cx, cy = PANEL_W // 2, PANEL_H // 2 + 20
    font = _font(12)
    font_small = _font(10)

    # Camera (triangulo apontando pra cima da cena) e o cone de quadro, referencia visual de
    # "quem esta dentro do enquadramento" -- so geometrico, nao sabe o que o still de verdade mostrou.
    draw.polygon([(cx, PANEL_H - 10), (cx - 18, PANEL_H - 30), (cx + 18, PANEL_H - 30)], fill="#444444")
    draw.line([(cx, PANEL_H - 20), (cx - 90, 10)], fill="#2a2a2a")
    draw.line([(cx, PANEL_H - 20), (cx + 90, 10)], fill="#2a2a2a")

    for name, info in formation.items():
        x_m, y_m = info["anchor_m"]
        x = cx + int(x_m * METERS_TO_PX)
        y = cy - int(y_m * METERS_TO_PX)
        color = _actor_color(names, name)
        r = 9
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white")
        draw.text((x - r, y + r + 2), name, fill=color, font=font_small)

    actor, partner = command.get("actor"), command.get("partner")
    primitive = command.get("primitive", "")
    if actor and actor in formation:
        ax_m, ay_m = formation[actor]["anchor_m"]
        ax, ay = cx + int(ax_m * METERS_TO_PX), cy - int(ay_m * METERS_TO_PX)
        if primitive in PARTNERED and partner and partner in formation:
            px_m, py_m = formation[partner]["anchor_m"]
            px, py = cx + int(px_m * METERS_TO_PX), cy - int(py_m * METERS_TO_PX)
            draw.line([(ax, ay), (px, py)], fill="#ffdd55", width=2)
            mx, my = (ax + px) // 2, (ay + py) // 2
            draw.text((mx - 20, my - 14), primitive, fill="#ffdd55", font=font_small)
        elif primitive in {"locomote", "turn"}:
            draw.ellipse([ax - 14, ay - 14, ax + 14, ay + 14], outline="#88ff88", width=2)

    idx = command.get("shot_index")
    header = f"#{idx}  {shot_meta.get('framing', '?') if shot_meta else '?'}  {command.get('duration_s', 0):.2f}s"
    draw.text((MARGIN, 4), header, fill="white", font=font)
    label = f"{actor or '-'} -> {partner or '-'}  [{primitive}]"
    draw.text((MARGIN, PANEL_H - 60), label, fill="#cccccc", font=font_small)
    instr = str(command.get("instruction", ""))[:60]
    draw.text((MARGIN, PANEL_H - 46), instr, fill="#888888", font=font_small)
    return img


def _checks(motion_plan: dict) -> list[str]:
    """Avisos baratos de posicionamento -- nao substitui a auditoria visual, so pega erro
    geometrico obvio antes de qualquer still existir."""
    avisos = []
    for scene in motion_plan.get("scenes", []):
        formation = scene.get("formation", {})
        occupied: dict[tuple, str] = {}
        for name, info in formation.items():
            key = tuple(info.get("anchor_m", []))
            if key in occupied:
                avisos.append(f"cena {scene['scene']}: {name} e {occupied[key]} ocupam a mesma "
                              f"ancora {key} -- vao se sobrepor no quadro")
            occupied[key] = name
        for cmd in scene.get("commands", []):
            actor, partner, primitive = cmd.get("actor"), cmd.get("partner"), cmd.get("primitive")
            if primitive in PARTNERED and not partner:
                avisos.append(f"plano {cmd['shot_index']}: primitiva '{primitive}' exige parceiro, "
                              "mas nenhum foi resolvido -- vai cair em movimento generico")
            if actor and partner and actor == partner:
                avisos.append(f"plano {cmd['shot_index']}: ator e parceiro sao a mesma pessoa ({actor})")
            if primitive == "pursue" and actor in formation and partner in formation:
                a_side = formation[actor].get("screen_side")
                p_side = formation[partner].get("screen_side")
                if a_side == p_side:
                    avisos.append(f"plano {cmd['shot_index']}: perseguicao {actor}->{partner} com os "
                                  f"dois do MESMO lado de tela ({a_side}) -- eixo de acao pode confundir")
    return avisos


def build_contact_sheet(run_dir: str | Path, *, columns: int = 4) -> tuple[Path, list[str]]:
    run_dir = Path(run_dir)
    motion_plan = json.loads((run_dir / "parse" / "motion_plan.json").read_text(encoding="utf-8"))
    try:
        shots_by_index = {s.get("position", s.get("index")): s
                          for s in json.loads((run_dir / "parse" / "shot_plan.json")
                                              .read_text(encoding="utf-8")).get("shots", [])}
    except (OSError, ValueError):
        shots_by_index = {}

    panels = []
    for scene in motion_plan.get("scenes", []):
        formation = scene.get("formation", {})
        names = sorted(formation)
        for cmd in scene.get("commands", []):
            panels.append(_panel(cmd, formation, names, shots_by_index.get(cmd.get("shot_index"))))

    if not panels:
        raise ValueError("motion_plan.json sem commands -- rode --motion-conditioning --apply antes")

    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * PANEL_W, rows * PANEL_H), "#000000")
    for i, panel in enumerate(panels):
        x, y = (i % columns) * PANEL_W, (i // columns) * PANEL_H
        sheet.paste(panel, (x, y))

    out = run_dir / "shots" / "blocking_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    avisos = _checks(motion_plan)
    (run_dir / "shots" / "blocking_preview_avisos.json").write_text(
        json.dumps({"avisos": avisos}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, avisos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--columns", type=int, default=4)
    args = ap.parse_args(argv)
    out, avisos = build_contact_sheet(args.run_dir, columns=args.columns)
    print(f"[blocking-preview] contact sheet -> {out}", flush=True)
    if avisos:
        print(f"[blocking-preview] {len(avisos)} aviso(s) de posicionamento:", flush=True)
        for a in avisos:
            print(f"  - {a}", flush=True)
    else:
        print("[blocking-preview] nenhum aviso geometrico.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
