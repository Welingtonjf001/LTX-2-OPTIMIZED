"""Controlador do MotionBricks dirigido por `motion_plan.json` (motion_conditioner.py),
em vez do teclado (`WASD_controller`) ou do aleatorio (`random_controller`) que o
repo traz. E o adaptador que MOTIONBRICKS_INTEGRATION.md apontava como faltante:
"um controlador proprio com a interface de generate_control_signals(...) que leia
commands[] do motion_plan.json".

A navegacao em si (qual comando esta "atual", pra onde andar, pra onde
encarar) vive em `motion_command_geometry.py` -- puro numpy, testado no
`.venv` do LTX. Esta classe so encaixa aquela logica na interface que
`navigation_demo`/`full_navigation_agent` esperam de um controlador
(`generate_control_signals`, `get_prev_qpos`, `get_controller_dt`), e por
isso PRECISA do venv do MotionBricks (importa `motionbricks.*`).

Roda com o PYTHON DO VENV DO MOTIONBRICKS, nao com o `.venv` do LTX -- ver
`motionbricks_headless.py`, que faz o `sys.path.insert` e usa esta classe.

LIMITE MEDIDO 2026-09-17, e por que ele existe: o checkpoint publico
`out/G1-clip.ckpt` so tem clipes de LOCOMOCAO -- `idle` e 12 variantes de
ESTILO DE CAMINHADA (zombie, boxing, stealth, injured, dance, crawling...),
ver `clip_holder_G1.CLIPS` em `motionbricks/motion_backbone/demo/clips.py`.
Nao ha clipe de gesto de mao, de fala, de alcancar objeto nem de contato. As
primitivas `speak`, `gesture`, `reach` e `contact` do `motion_plan.json` SAO
reais e uteis por si (delas vem o `motion_prompt` de texto que LTX/MiniMax ja
consomem) -- mas como ENTRADA PARA ESTE CONTROLADOR so podem ser aproximadas:

- `speak`, `gesture`, `reach` (e `turn`, `environment`) viram `idle` (o
  agente para e so gira para encarar o alvo).
- `contact` (como `locomote`) vira `walk` DE VERDADE -- CORRECAO 2026-09-18
  (auditoria externa apontou este paragrafo dizendo "todas viram idle",
  incluindo contact, o que contradizia `PRIMITIVE_TO_CLIP` logo abaixo):
  contact sempre tem parceiro (ver motion_conditioner.infer_primitive), e o
  checkpoint publico consegue pelo menos APROXIMAR contato caminhando ate
  perto do parceiro e parando -- nao ha gesto de aperto de mao/abraco de
  verdade (isso sim precisaria de clipe de upper-body que nao existe), mas
  ha locomocao real ate a proximidade, diferente de speak/gesture/reach que
  ficam parados o tempo todo.

Ver `PRIMITIVE_TO_CLIP` em motion_command_geometry.py pro mapa exato. Isto
nao e um defeito desta classe -- e o que o checkpoint publico oferece. Um
checkpoint com clipes de upper-body ligaria speak/gesture/reach a movimento
de verdade sem mudar uma linha aqui: so o mapa em motion_command_geometry.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as t

from motionbricks.motion_backbone.demo.controllers import base_controller

from motion_command_geometry import (
    PRIMITIVE_TO_CLIP,
    commands_for_actor,
    facing_for,
    formation_by_scene,
    maintain_heading,
    resolve_command,
    walk_target,
)


class MotionPlanController(base_controller):
    """Um agente por ATOR: `actor` filtra os comandos que dizem respeito a esse
    personagem -- os outros ficam de fora desta instancia.

    `live_positions` (opcional): dicionario COMPARTILHADO {ator: xy}, mantido
    por fora (ver `motionbricks_scene_director.py`) com a posicao ATUAL de
    cada agente que estiver rodando junto nesta cena. Sem isto (uso normal,
    um agente so), o comportamento e exatamente o de antes: destino
    pre-calculado em `motion_plan.json`, sem coordenacao em tempo real. Rode
    um controlador por ator; o director resolve contato/colisao entre eles."""

    def __init__(self, motion_plan: dict[str, Any], actor: str, *, clips: str = "G1",
                anchor_scale_m: float = 1.0, live_positions: dict[str, Any] | None = None,
                min_separation_m: float | None = None, brake_margin_m: float = 1.0, **kwargs):
        super().__init__(clips, **kwargs)
        self._actor = actor
        self._anchor_scale = anchor_scale_m
        self._commands = commands_for_actor(motion_plan, actor)
        self._formation_by_scene = formation_by_scene(motion_plan)
        self._live_positions = live_positions
        self._min_separation_m = min_separation_m if min_separation_m is not None \
            else motion_plan.get("min_separation_m")
        self._brake_margin_m = brake_margin_m
        self._prev_qpos: np.ndarray | None = None
        self._NUM_HISTORY_STEPS = 5
        self._sim_time = 0.0

    @property
    def commands(self) -> list[dict[str, Any]]:
        """Comandos deste ator, em ordem -- publico pra quem monta o loop
        (motionbricks_headless.py) saber quanto tempo/quantos passos rodar."""
        return self._commands

    def reset(self) -> None:
        super().reset()
        self._sim_time = 0.0
        self._prev_qpos = None

    def generate_control_signals(self, viewer, mj_model, mj_data, visualize: bool = True,
                                 control_info: dict | None = None) -> dict[str, Any]:
        if self._prev_qpos is None:
            self._prev_qpos = np.zeros((self._NUM_HISTORY_STEPS, mj_model.nq))
            self._prev_qpos[:] = mj_data.qpos.copy().reshape(1, -1)

        command = resolve_command(self._commands, self._sim_time)
        formation = self._formation_by_scene.get(command.get("scene"), {})
        clip_name = PRIMITIVE_TO_CLIP.get(command["primitive"], "idle")
        # MotionBricks' public checkpoint has locomotion clips only. A reach
        # toward a named partner (e.g. pulling the president clear) can at least
        # move to the partner; prop reaches without a partner remain idle.
        if command["primitive"] == "reach" and command.get("partner"):
            clip_name = "walk"
        # ACHADO 2026-09-17 (2 agentes reais no diretor de cena): a simulacao
        # do MotionBricks sempre NASCE cada agente na origem local (0,0) --
        # nunca na ancora que `formation` atribuiu a ele. Com um agente so
        # isso passa despercebido (nao ha ninguem pra comparar posicao); com
        # DOIS, os dois nasciam sobrepostos e o freio de colisao ao vivo
        # disparava no primeiro passo, achando que ja tinham chegado sem sair
        # do lugar. Corrige deslocando a posicao LOCAL do agente pela sua
        # PROPRIA ancora -- so translacao, entao a DIRECAO de qualquer vetor
        # (movimento, encarar) e identica com ou sem o deslocamento; so a
        # posicao usada pra medir distancia/chegada/colisao muda.
        own_anchor = (formation.get(self._actor) or {}).get("anchor_m")
        offset = np.asarray(own_anchor[:2], dtype=np.float64) * self._anchor_scale if own_anchor else np.zeros(2)
        root_xy = mj_data.qpos[:2].copy() + offset
        prev_root_xy = self._prev_qpos[:, :2] + offset

        if clip_name == "walk":
            direction, chegou = walk_target(
                command, root_xy, formation, self._anchor_scale,
                live_positions=self._live_positions, min_separation_m=self._min_separation_m,
                brake_margin_m=self._brake_margin_m, own_actor=self._actor)
            if direction is None:
                if chegou:
                    # Chegou perto do parceiro (destino, OU freio de colisao
                    # ao vivo se `live_positions` estiver ligado): para de
                    # andar mesmo que o plano ainda peca "walk" pelo resto da
                    # tomada -- evita passar por cima dele.
                    clip_name = "idle"
                    movement_direction = np.zeros(3)
                    facing_direction = facing_for(command, root_xy, formation, prev_root_xy,
                                                  self._anchor_scale, self._live_positions)
                else:
                    # "locomote" sem parceiro (destino None de proposito, ver
                    # motion_conditioner) -- anda na direcao que ja levava.
                    movement_direction = maintain_heading(prev_root_xy)
                    facing_direction = movement_direction
            else:
                movement_direction = direction
                facing_direction = direction
        else:
            movement_direction = np.zeros(3)
            facing_direction = facing_for(command, root_xy, formation, prev_root_xy,
                                          self._anchor_scale, self._live_positions)

        if self._live_positions is not None:
            self._live_positions[self._actor] = root_xy
        self._prev_qpos = np.concatenate((self._prev_qpos[1:], mj_data.qpos.copy().reshape(1, -1)), axis=0)
        self._sim_time += mj_model.opt.timestep

        mode_idx = list(self._clip_holder_class.CLIPS.keys()).index(clip_name)
        mode = t.tensor([mode_idx])
        control_signals = {
            "movement_direction": t.from_numpy(movement_direction).view([1, -1]).float(),
            "facing_direction": t.from_numpy(facing_direction).view([1, -1]).float(),
            "mode": mode.view([1, -1]),
        }
        control_signals["allowed_pred_num_tokens"] = self.get_default_allowed_pred_num_tokens(mode.item())
        return control_signals
