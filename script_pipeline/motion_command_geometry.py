"""Logica PURA (so numpy) do controlador de MotionBricks dirigido por
motion_plan.json -- separada de `motionbricks_command_controller.py` de
proposito, porque aquele arquivo so importa no venv do MotionBricks
(`motionbricks.motion_backbone.demo.controllers`, que arrasta keyboard/scipy/
mujoco). Esta aqui roda e testa no `.venv` do LTX normal
(`tests/test_motion_command_geometry.py`), sem precisar daquele runtime.

`motionbricks_command_controller.MotionPlanController` so chama estas
funcoes; nao duplica a logica.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

ARRIVAL_EPS_M = 0.15

# Primitiva do motion_plan.json -> clipe REAL do checkpoint G1 publico. So
# "locomote" e "contact" (que sempre tem parceiro -- ver motion_conditioner.
# infer_primitive) usam o clipe "walk" de verdade; o resto cai em "idle" por
# falta de clipe proprio no checkpoint (ver docstring de
# motionbricks_command_controller.py para o porque).
PRIMITIVE_TO_CLIP: dict[str, str] = {
    "environment": "idle", "idle": "idle", "speak": "idle", "gesture": "idle",
    "turn": "idle", "fire": "idle", "reach": "idle", "locomote": "walk",
    "pursue": "walk", "contact": "walk", "takedown": "walk", "protect": "walk",
}


def scene_time_offsets(motion_plan: dict[str, Any]) -> dict[Any, float]:
    """Deslocamento GLOBAL (soma das duracoes das cenas anteriores) de cada
    cena, na ordem em que aparecem em `motion_plan["scenes"]`.

    ACHADO 2026-09-17 (auditoria externa): o produtor (`motion_conditioner.
    build_motion_score`) reinicia `cursor = 0.0` a CADA cena -- `start_s` no
    JSON e relativo a propria cena, nao ao filme inteiro. Sem este
    deslocamento, `commands_for_actor` misturava comandos de cenas diferentes
    num unico relogio, e duas cenas com o mesmo ator podiam se sobrepor (as
    duas comecando em start_s=0) -- reproduzido com 2 cenas de 2s da ANA: a
    segunda nunca era selecionada porque a busca global parava na primeira
    janela [0,2) que batesse."""
    offsets: dict[Any, float] = {}
    cursor = 0.0
    for scene in motion_plan.get("scenes", []):
        offsets[scene["scene"]] = cursor
        cursor += float(scene.get("duration_s", 0.0))
    return offsets


def commands_for_actor(motion_plan: dict[str, Any], actor: str) -> list[dict[str, Any]]:
    """`start_s`/`duration_s` no retorno sao GLOBAIS (deslocados por
    `scene_time_offsets`) -- os comandos originais no JSON continuam
    relativos a propria cena; esta funcao nao muta `motion_plan`."""
    offsets = scene_time_offsets(motion_plan)
    cmds = []
    for c in motion_plan.get("commands", []):
        if c.get("actor") != actor:
            continue
        offset = offsets.get(c.get("scene"), 0.0)
        cmds.append({**c, "start_s": c["start_s"] + offset})
    if not cmds:
        atores = sorted({c.get("actor") for c in motion_plan.get("commands", []) if c.get("actor")})
        raise ValueError(f"nenhum comando para o ator {actor!r} em motion_plan.json "
                        f"-- atores disponiveis: {atores}")
    return sorted(cmds, key=lambda c: c["start_s"])


def current_command(commands: Sequence[dict[str, Any]], sim_time: float) -> dict[str, Any] | None:
    """`commands` ja ordenados por start_s, GLOBAL (ver commands_for_actor).
    Devolve None quando nao ha comando ativo em `sim_time` -- lacuna antes do
    primeiro comando, entre dois comandos, ou depois do ultimo. ACHADO
    2026-09-17 (auditoria externa): a versao antiga devolvia o ULTIMO
    comando da lista sempre que o sim_time caia numa lacuna e ja tivesse
    passado do primeiro comando -- um ator podia executar uma tomada futura
    (ex.: um `contact` marcado pra daqui a 5s) horas antes da hora. Ver
    `idle_command_for_gap` pro que fazer no lugar."""
    for cmd in commands:
        if cmd["start_s"] <= sim_time < cmd["start_s"] + cmd["duration_s"]:
            return cmd
    return None


def idle_command_for_gap(commands: Sequence[dict[str, Any]], sim_time: float) -> dict[str, Any]:
    """Comando sintetico de idle pra quando `current_command` devolve None.
    Usa a cena do ULTIMO comando que ja terminou (mantem o ator parado onde
    a ultima tomada o deixou) ou, se `sim_time` e ANTES do primeiro comando
    do ator, a cena do PRIMEIRO (espera no lugar de entrada em vez de
    aparecer na origem). Sem parceiro -- idle nunca anda, so evita herdar um
    `target`/`partner` de um comando de outra hora."""
    passados = [c for c in commands if c["start_s"] + c["duration_s"] <= sim_time]
    referencia = passados[-1] if passados else commands[0]
    return {"primitive": "idle", "scene": referencia.get("scene"), "partner": None, "target": None}


def resolve_command(commands: Sequence[dict[str, Any]], sim_time: float) -> dict[str, Any]:
    """`current_command` com o fallback de lacuna ja aplicado -- o que os
    controladores devem chamar (em vez de tratar None na mao toda vez)."""
    cmd = current_command(commands, sim_time)
    return cmd if cmd is not None else idle_command_for_gap(commands, sim_time)


def formation_by_scene(motion_plan: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    return {scene["scene"]: (scene.get("formation") or {}) for scene in motion_plan.get("scenes", [])}


def partner_anchor(command: dict[str, Any], formation: dict[str, Any],
                   live_positions: dict[str, Any] | None = None,
                   anchor_scale_m: float = 1.0) -> list[float] | None:
    """Onde mirar pro parceiro, em ordem de preferencia -- SEMPRE em metros
    de MUNDO no retorno (o chamador nao deve escalar de novo):

    1. A posicao AO VIVO dele (`live_positions[partner]`) -- ja em metros de
       mundo (e o que `MotionPlanController` publica), por isso NAO leva
       `anchor_scale_m` -- so existe quando um `SceneDirector` esta rodando
       os dois agentes juntos, passo a passo, e converge de verdade (o alvo
       se move com o parceiro de verdade) em vez de mirar um ponto
       pre-calculado da posicao INICIAL.
    2. O destino ja resolvido no JSON (locomote/contact, ver
       `motion_conditioner._meeting_point` -- meio do caminho entre as
       ancoras ORIGINAIS, calculado uma vez, sem coordenacao em tempo real) --
       em unidades de ANCORA, escalado aqui.
    3. A ancora estatica dele em `formation` (turn/speak apontando pro
       parceiro, que nao anda -- a ancora dele e a posicao real o tempo
       todo, so nao veio marcada como "destination") -- escalado aqui.

    ACHADO 2026-09-17 (auditoria externa): `facing_for`/`walk_target`
    escalavam de novo o retorno desta funcao, mesmo quando ja vinha de
    `live_positions` (ja em metros de mundo) -- com anchor_scale_m != 1 o
    alvo saia na direcao ERRADA (ex.: parceiro a esquerda virava alvo a
    direita). Agora quem escala e so esta funcao, uma vez, e so nos dois
    casos estaticos."""
    partner = command.get("partner")
    if partner and live_positions is not None and partner in live_positions:
        pos = live_positions[partner]
        return [float(pos[0]), float(pos[1])]
    target = command.get("target") or {}
    destino = target.get("destination")
    if destino:
        return [float(destino[0]) * anchor_scale_m, float(destino[1]) * anchor_scale_m]
    if partner:
        info = formation.get(partner)
        if info:
            anchor = info.get("anchor_m")
            if anchor:
                return [float(anchor[0]) * anchor_scale_m, float(anchor[1]) * anchor_scale_m]
    return None


def _unit_xy(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-3 else None


def maintain_heading(prev_root_xy: np.ndarray) -> np.ndarray:
    """`prev_root_xy`: historico [N, 2] de posicoes recentes, mais antiga
    primeiro. Continua a direcao/velocidade que ja tinha -- mesmo principio
    que `random_controller`/`WASD_controller` usam no estado idle."""
    if len(prev_root_xy) < 2:
        return np.array([1.0, 0.0, 0.0])
    qvel = (prev_root_xy[1:] - prev_root_xy[:-1]).mean(axis=0)
    unit = _unit_xy(qvel)
    if unit is None:
        return np.array([1.0, 0.0, 0.0])
    return np.array([unit[0], unit[1], 0.0])


def facing_for(command: dict[str, Any], root_xy: np.ndarray, formation: dict[str, Any],
               prev_root_xy: np.ndarray, anchor_scale_m: float = 1.0,
               live_positions: dict[str, Any] | None = None) -> np.ndarray:
    # partner_anchor ja devolve metros de MUNDO (estatico escalado, ao vivo
    # nao) -- nao escalar de novo aqui (ACHADO 2026-09-17, ver partner_anchor).
    anchor = partner_anchor(command, formation, live_positions, anchor_scale_m)
    if anchor is not None:
        delta = np.asarray(anchor[:2], dtype=np.float64) - root_xy
        unit = _unit_xy(delta)
        if unit is not None:
            return np.array([unit[0], unit[1], 0.0])
    # Sem parceiro (turn_in_place, fala/gesto solo): encara o centro do
    # quadro -- a mesma convencao 'facing': 'toward_center' que
    # motion_conditioner.build_motion_score reserva em `formation`.
    if abs(root_xy[0]) > 1e-6:
        return np.array([-np.sign(root_xy[0]), 0.0, 0.0])
    return maintain_heading(prev_root_xy)


def walk_target(command: dict[str, Any], root_xy: np.ndarray, formation: dict[str, Any],
                anchor_scale_m: float = 1.0, live_positions: dict[str, Any] | None = None,
                min_separation_m: float | None = None,
                brake_margin_m: float = 1.0, own_actor: str | None = None
                ) -> tuple[np.ndarray | None, bool]:
    """Devolve (direcao_unitaria_ou_None, chegou). `direcao=None` quando nao ha
    destino conhecido (locomote sem parceiro, de proposito -- ver
    motion_conditioner) OU quando ja chegou perto o bastante do destino.

    NAO aplica `target['min_separation_m']` como piso de chegada contra o
    destino PRE-CALCULADO -- ACHADO 2026-09-17: a separacao segura ja e
    resolvida NA ORIGEM (`_meeting_point` em motion_conditioner.py poe o
    destino de cada ator a METADE da separacao do ponto de encontro).
    Repetir o piso ali SOMAVA com aquele: dois agentes com destino a 0,5 m
    cada um do centro (ja 1,0 m entre si por construcao) liam
    min_separation_m=1,0 m como distancia de chegada e paravam sem sair do
    lugar. So a tolerancia de chegada (ARRIVAL_EPS_M) vale contra o destino.

    `min_separation_m` (com `live_positions`): freio de colisao AO VIVO,
    diferente do piso removido acima -- mede a distancia ATUAL contra a
    posicao ATUAL do parceiro (nao o destino pre-calculado), e para se ja
    estiver mais perto que isso.

    `brake_margin_m`: o freio PEDE pra parar assim que a distancia cai abaixo
    de `min_separation_m + brake_margin_m`, nao exatamente em
    `min_separation_m` -- MEDIDO 2026-09-17, dois agentes reais convergindo
    (ANA/BIA, `motionbricks_scene_director.py`), tres pontos:

        margem 0,0 m -> separacao minima real 0,45 m (contra 1,0 m pedido)
        margem 0,5 m -> 0,45 m -- SEM EFEITO NENHUM (identico bit a bit)
        margem 1,0 m -> 1,009 m -- <1% de erro
        margem 2,0 m (limiar > distancia inicial) -> nem chegam a andar

    O salto de "sem efeito" pra "quase exato" entre 0,5 e 1,0 m (em vez de
    melhorar gradualmente) sugere que `full_navigation_agent.generate_new_
    frames` planeja em BLOCOS (a chamada usa `get_controller_dt()*2 ~= 0,53s`
    de uma vez) -- pedir pra frear alguns quadros mais cedo dentro do MESMO
    bloco nao muda o bloco ja decidido; so um salto grande o bastante move o
    pedido pro bloco ANTERIOR. Isto e um limite de GRANULARIDADE deste
    checkpoint/harness de inferencia, nao um erro de calculo de distancia --
    o freio em si (a comparacao `live_dist <= min_separation_m +
    brake_margin_m`) esta correto, testado e comprovado no extremo (margem 2m
    trava o agente parado, como esperado). Calibrar de novo (testando alguns
    valores como acima) se o passo de tempo/checkpoint mudar. Sem
    `live_positions` (modo antigo, um agente so) nenhum dos dois parametros
    dispara -- nao ha parceiro rodando pra medir contra.

    O freio checa TODOS os outros atores em `live_positions`, nao so
    `command.get("partner")` -- ACHADO 2026-09-17 (auditoria externa): a
    versao antiga so via o parceiro DESIGNADO da tomada, entao um terceiro
    ator parado no caminho (ou uma locomocao SEM parceiro, que nem entrava
    neste bloco) nao freava nada. `own_actor` exclui a propria entrada (o
    controlador publica a posicao do proprio ator em `live_positions` antes
    do proximo passo -- ver `MotionPlanController.generate_control_signals`)."""
    if live_positions is not None and min_separation_m:
        for other_actor, other_pos in live_positions.items():
            if other_actor == own_actor:
                continue
            other_xy = np.asarray(other_pos[:2], dtype=np.float64)
            live_dist = float(np.linalg.norm(other_xy - root_xy))
            if live_dist <= min_separation_m + brake_margin_m:
                return None, True
    # partner_anchor ja devolve metros de MUNDO -- nao escalar de novo aqui.
    destino = partner_anchor(command, formation, live_positions, anchor_scale_m)
    if destino is None:
        return None, False
    dest_xy = np.asarray(destino[:2], dtype=np.float64)
    delta = dest_xy - root_xy
    dist = float(np.linalg.norm(delta))
    if dist <= ARRIVAL_EPS_M:
        return None, True
    return np.array([delta[0] / dist, delta[1] / dist, 0.0]), False
