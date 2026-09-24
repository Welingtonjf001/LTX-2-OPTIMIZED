"""Converte a decupagem em instrucoes de movimento por personagem.

MotionBricks nao recebe texto cinematografico e nao renderiza video: o release
publico planeja e decodifica movimento esqueletico de um agente humanoide a
partir de primitivas, alvos espaciais e clips de estilo.  Este modulo e a ponte
deliberadamente explicita entre esses dois mundos:

    shot_plan.json -> score temporal por ator ->
      (a) motion_prompt curto para LTX/MiniMax;
      (b) motionbricks_request.json para um adaptador 3D/retarget.

O arquivo de request NAO finge ser a API nativa do MotionBricks. Ele e um
contrato estavel do projeto: um adaptador deve converter ``primitive``,
``target`` e ``style_hint`` para os smart primitives e para o esqueleto que
estiver em uso. Assim o plano continua util mesmo sem o runtime 3D instalado.

REVISAO 2026-09-17 (auditoria do plano real dos piratas, 13 tomadas): a
primeira versao classificava por substring solta e errou 4 das 9 tomadas com
personagem -- "gestures towards the ship" virou idle (token era "points",
nao "gesture"), "holds a spyglass up" virou idle ("hold hands" so), "looking
out at the horizon" virou idle ("looks" so), e "hulls almost TOUCHING" virou
reach porque "touch" casou dentro de "touching" descrevendo os NAVIOS, nao o
ator. Agora: regex com fronteira de palavra e lemas cobertos; tomada sem
sujeito e sempre `environment` (antes, o fallback com o texto da cena inteira
inventava "The subject must walk" num plano de estabelecimento vazio); e
close de FALA vira a primitiva `speak`, que nao pede gesto de corpo -- e a
regra medida em MEMORIAL 3.81/3.85: gesto de corpo num close de fala tira a
boca do quadro e derruba o lip-sync.

Uso:
  python -m script_pipeline.motion_conditioner --plan RUN/parse/shot_plan.json --apply
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "motion-score/v2"
MIN_SEPARATION_M = 1.0

# Ordem importa: contato/dupla tem prioridade sobre caminhada, e caminhar tem
# prioridade sobre um gesto incidental escrito na mesma frase. Cada padrao e
# uma alternancia com \b -- "turn" nao casa em "returns", "touch" nao casa em
# "touching" (que aqui descreve cascos de navio, nao um ator).
_W = r"(?<![A-Za-z])(?:{})(?![A-Za-z])"
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("takedown", re.compile(_W.format(
        r"tackles?|tackling|knocks? (?:him|her|them) down|takes? (?:him|her|them) down|"
        r"derruba|imobiliza|joga ao ch[aã]o")),
     "sprint into one controlled tackle, bring the partner safely to the ground, then pin and hold position"),
    ("fire", re.compile(_W.format(
        r"fires?|firing|shoots?|shooting|aims?|aiming|atir(?:a|ar|ando)|dispara|mirando")),
     "raise the weapon, aim toward the stated target, fire controlled shots, then hold aim; never turn toward camera"),
    ("protect", re.compile(_W.format(
        r"shields?|shielding|protects?|protecting|covers? (?:him|her|them)|"
        r"protege|protegendo|cobre o presidente")),
     "move between the partner and the threat, shield the partner, and hold a protective position"),
    ("pursue", re.compile(_W.format(
        r"chases?|chasing|pursues?|pursuing|runs? after|running after|persegue|perseguindo")),
     "sprint after the fleeing target along the same street axis, weaving past pedestrians without stopping"),
    ("contact", re.compile(_W.format(
        r"hugs?|hugging|embraces?|embracing|kiss(?:es|ing)?|handshakes?|shakes? (?:her|his|their) hand|"
        r"fights?|fighting|grabs?|grabbing|holds? hands|hold hands|"
        r"abra[cç]a|beija|aperta a m[aã]o|luta")),
     "approach the partner, stop at a safe conversational distance, then perform one controlled contact action"),
    ("reach", re.compile(_W.format(
        r"picks? up|picks?|takes?|taking|hands? (?:over|him|her|them)|gives?|giving|opens?|opening|"
        r"pulls?|pulling|drags?|dragging|puxa|puxando|arrasta|arrastando|"
        r"touch(?:es)?|places?|placing|holds?|holding|raises?|raising|lifts?|lifting|grips?|"
        r"pega|entrega|abre|toca|coloca|segura|ergue")),
     "make one deliberate reach toward the named prop or partner, then return to a stable pose"),
    ("locomote", re.compile(_W.format(
        r"walks?|walking|runs?|running|sprints?|sprinting|"
        r"hurr(?:y|ies|ying)|enters?|entering|crosses|crossing|"
        r"approach(?:es|ing)?|steps?|stepping|dash(?:es)?|glides?|strides?|paces?|"
        r"caminha|corre|dispara em corrida|entra|atravessa|aproxima|passos")),
     "sprint along the established path, accelerate decisively, keep the partner in the same direction of travel, then continue moving"),
    ("turn", re.compile(_W.format(
        r"turns?|turning|looks?|looking|gaz(?:es|ing)|glances?|glancing|faces|facing|stares?|staring|"
        r"vira|olha|olhando|encara")),
     "turn head and upper body toward the focus, keeping feet planted"),
    ("gesture", re.compile(_W.format(
        r"gestures?|gesturing|nods?|nodding|shakes? (?:her|his|their) head|points?|pointing|"
        r"waves?|waving|shrugs?|smiles?|smiling|grins?|cries|leans?|leaning|"
        r"crosses (?:her|his|their) arms|arms crossed|"
        r"gesticula|assente|balan[cç]a|aponta|acena|sorri|chora|inclina")),
     "use one restrained, readable gesture, then settle"),
)

SPEAKING_FRAMINGS = {"close", "extreme_close"}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def infer_primitive(beat: str, subject: str, partner: str, *,
                    framing: str = "", speaking: bool = False) -> tuple[str, str]:
    """Retorna uma primitiva pequena, nunca uma descricao coreografica vaga.

    Sem sujeito nao ha ator: e `environment`, antes de olhar qualquer verbo
    (o texto de fallback pode ser a cena inteira e descrever outra pessoa).
    Close de fala vira `speak`: a boca precisa ficar no quadro, sem gesto de
    corpo nem deslocamento -- o movimento ali e de cabeca e olhar, e o
    lip-sync manda (MEMORIAL 3.81/3.85)."""
    if not subject:
        return "environment", "no character motion; keep the environment stable"
    if speaking and framing in SPEAKING_FRAMINGS:
        return "speak", ("speak naturally with subtle head and eye movement, shoulders steady, "
                         "mouth clearly visible, no hand gestures in frame")
    lower = beat.casefold()
    for primitive, pattern, instruction in RULES:
        if pattern.search(lower):
            # "contact" sem parceiro conhecido nao e seguro para um controlador
            # multiagente; preserva a intencao como gesto em vez de inventar alvo.
            if primitive == "contact" and not partner:
                return "gesture", "make a restrained expressive gesture and remain in place"
            return primitive, instruction
    return "idle", "maintain a natural idle with subtle breathing and weight shift"


def _meeting_point(own_anchor: list[float], partner_anchor: list[float],
                   min_separation_m: float) -> list[float]:
    """Ponto de encontro pra um `approach_partner`: o MEIO do caminho entre as
    duas ancoras, mas parado a `min_separation_m/2` do centro, do lado de
    quem vai andar.

    ACHADO (avaliacao com 2 agentes reais no MotionBricks, 2026-09-17): sem
    isto, `target["destination"]` era a ancora ORIGINAL do parceiro -- e
    quando os DOIS lados da cena tem `approach_partner` um pro outro ao mesmo
    tempo (a conversa comum, dois personagens se aproximando), cada um anda
    ate a marca de ONDE O OUTRO COMECOU. MEDIDO: os dois se cruzavam e
    TROCAVAM de lado de tela inteiro, terminando mais LONGE um do outro
    (2,75 m) do que a distancia inicial (2,4 m) -- o oposto de "se
    encontram para conversar". Convergindo pro meio do caminho, cada um fica
    do seu proprio lado (screen_side preservado) e a 1x min_separation_m de
    distancia no final."""
    mx = (own_anchor[0] + partner_anchor[0]) / 2.0
    my = (own_anchor[1] + partner_anchor[1]) / 2.0
    dx, dy = own_anchor[0] - partner_anchor[0], own_anchor[1] - partner_anchor[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist < 1e-6:
        return [mx, my]
    half = min_separation_m / 2.0
    return [mx + dx / dist * half, my + dy / dist * half]


def _formation(characters: set[str], sides: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Reserva uma ancora por ator para conservar eixo e evitar sobreposicao."""
    result: dict[str, dict[str, Any]] = {}
    used: defaultdict[str, int] = defaultdict(int)
    for name in sorted(characters):
        side = sides.get(name, "left")
        ordinal = used[side]
        used[side] += 1
        # Um terceiro personagem no mesmo lado sobe no eixo Z, em vez de ocupar
        # a mesma coordenada do primeiro. E uma restricao de blocking, nao uma
        # garantia fisica do video generativo.
        x = -1.2 if side == "left" else 1.2
        result[name] = {"screen_side": side, "anchor_m": [x, round(ordinal * 1.35, 2)],
                        "facing": "toward_center"}
    return result


def _video_clause(subject: str, primitive: str, instruction: str, partner: str) -> str:
    if primitive == "environment":
        return "Motion constraint: no character enters; the environment remains stable"
    if primitive == "speak":
        return f"Character motion constraint: {subject} must {instruction}"
    partner_clause = f" with {partner}" if partner else ""
    return (f"Character motion constraint: {subject}{partner_clause} must {instruction}; "
            "preserve identity, body count and screen position")


def build_motion_score(plan: dict[str, Any]) -> dict[str, Any]:
    """Compila o score sem chamar LLM: auditavel e reproduzivel por tomada."""
    sides = plan.get("screen_sides") or {}
    grouped: defaultdict[Any, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, shot in enumerate(plan.get("shots") or []):
        grouped[shot.get("scene")].append((index, shot))

    scenes: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    warnings: list[str] = []
    for scene_id, indexed_shots in grouped.items():
        actors = {_norm(s.get("subject")) for _, s in indexed_shots if _norm(s.get("subject"))}
        actors.update(_norm(s.get("co_subject")) for _, s in indexed_shots if _norm(s.get("co_subject")))
        formation = _formation(actors, sides)
        cursor = 0.0
        commands: list[dict[str, Any]] = []
        for global_index, shot in indexed_shots:
            subject, partner = _norm(shot.get("subject")), _norm(shot.get("co_subject"))
            beat = _norm(shot.get("beat")) or _norm(shot.get("fallback"))
            primitive, instruction = infer_primitive(
                beat, subject, partner, framing=_norm(shot.get("framing")),
                speaking=shot.get("line_index") is not None)
            seconds = max(0.1, float(shot.get("seconds") or 0.1))
            target: dict[str, Any] = {"mode": "hold_anchor", "min_separation_m": MIN_SEPARATION_M}
            if primitive == "pursue":
                # Co-subjects in a chase run in the same direction; converging
                # toward one another reverses the screen geography and cancels
                # the pursuit. The adapter continues the established heading.
                target["mode"] = "short_path"
                target["destination"] = None
            elif primitive in {"locomote", "contact", "takedown", "protect"} or (
                    primitive == "reach" and partner):
                if partner:
                    target["mode"] = "approach_partner"
                    own_anchor = (formation.get(subject) or {}).get("anchor_m")
                    partner_anchor = (formation.get(partner) or {}).get("anchor_m")
                    if own_anchor and partner_anchor:
                        target["destination"] = _meeting_point(own_anchor, partner_anchor, MIN_SEPARATION_M)
                    else:
                        target["destination"] = partner_anchor
                else:
                    # Sem parceiro nao ha destino conhecido: o adaptador escolhe um
                    # caminho curto a partir da ancora, em vez de "andar" ate a
                    # propria posicao (que a v1 mandava e equivale a ficar parado).
                    target["mode"] = "short_path"
                    target["destination"] = None
            elif primitive == "turn":
                target["mode"] = "face_partner" if partner else "turn_in_place"
            elif primitive == "speak":
                target["mode"] = "face_camera_or_partner" if not partner else "face_partner"
            command = {
                "shot_index": global_index, "scene": scene_id, "start_s": round(cursor, 3),
                "duration_s": seconds, "actor": subject or None, "partner": partner or None,
                "primitive": primitive, "instruction": instruction, "beat_source": beat,
                "target": target, "style_hint": shot.get("style") or plan.get("style") or "classico",
                "motion_prompt": _video_clause(subject, primitive, instruction, partner),
            }
            commands.append(command)
            flat.append(command)
            cursor += seconds
            if primitive == "contact" and len(actors) > 2:
                warnings.append(f"cena {scene_id}, tomada {global_index}: contato com {len(actors)} atores; "
                                "exige coreografia/retarget conjunto, nao duas inferencias independentes.")
        scenes.append({"scene": scene_id, "duration_s": round(cursor, 3), "formation": formation,
                       "commands": commands})
    return {"schema_version": SCHEMA_VERSION, "source": "shot_plan.json", "fps": plan.get("fps", 24),
            "min_separation_m": MIN_SEPARATION_M, "scenes": scenes, "commands": flat, "warnings": warnings,
            "adapter_contract": {"runtime": "MotionBricks", "requires": ["per-actor current pose/context",
                "retarget map from MotionBricks G1 skeleton to character rig", "style reference clip per actor"],
                "multi_actor": "Run one agent per actor; a scene director resolves anchors, partners and contacts before inference."}}


def apply_motion_score(plan: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    """Anexa a condicao ao prompt sem destruir o prompt da decupagem.

    Idempotente: uma clausula ja presente (rodar --apply duas vezes, ou um
    shot_plan que ja veio condicionado) nao e anexada de novo."""
    result = copy.deepcopy(plan)
    by_index = {command["shot_index"]: command for command in score["commands"]}
    for index, shot in enumerate(result.get("shots") or []):
        command = by_index[index]
        shot["motion_conditioning"] = {key: command[key] for key in (
            "actor", "partner", "primitive", "instruction", "target", "style_hint")}
        clause = command["motion_prompt"]
        base = _norm(shot.get("video_prompt")).rstrip(".")
        previous = _norm(shot.get("motion_prompt"))
        if previous and previous in base:
            base = base.replace(previous, "").rstrip(". ").rstrip(".")
        shot["motion_prompt"] = clause
        if clause not in base:
            shot["video_prompt"] = f"{base}. {clause}."
    result["motion_conditioning"] = {"schema_version": SCHEMA_VERSION, "score_file": "motion_plan.json",
                                      "applied": True}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Condiciona movimento de personagens a partir da decupagem")
    ap.add_argument("--plan", required=True, help="shot_plan.json de entrada")
    ap.add_argument("--out", default=None, help="motion_plan.json de saida")
    ap.add_argument("--apply", action="store_true", help="atualiza o shot_plan com motion_prompt para LTX/MiniMax")
    ap.add_argument("--applied-out", default=None, help="destino do plano enriquecido; padrao sobrescreve --plan")
    args = ap.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    score = build_motion_score(plan)
    out = Path(args.out) if args.out else plan_path.with_name("motion_plan.json")
    out.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.apply:
        applied = Path(args.applied_out) if args.applied_out else plan_path
        applied.write_text(json.dumps(apply_motion_score(plan, score), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[motion_conditioner] plano enriquecido para video: {applied}")
    print(f"[motion_conditioner] {len(score['commands'])} comando(s), {len(score['warnings'])} alerta(s): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
