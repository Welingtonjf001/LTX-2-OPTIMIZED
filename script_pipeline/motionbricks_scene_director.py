"""Diretor de cena: roda 2+ agentes do MotionBricks NO MESMO RELOGIO de
simulacao, cada um numa `navigation_demo` PROPRIA (nao ha mundo MuJoCo
compartilhado -- so o RELOGIO e um dicionario de posicoes e), e realimenta a
posicao ATUAL de cada ator nos outros a cada passo, via `live_positions` do
`MotionPlanController`.

O QUE ISTO RESOLVE (achado 2026-09-17, dois agentes reais se aproximando):
sem coordenacao em tempo real, `approach_partner` so pode mirar um ponto
PRE-CALCULADO da posicao INICIAL do parceiro (`_meeting_point`, em
motion_conditioner.py) -- funciona, mas nao se adapta se o parceiro andar
diferente do esperado, e o overshoot natural do blend de locomocao deste
checkpoint (MEDIDO: ate 0,36 m alem do 1,0 m pretendido) fica sem freio. Com
o diretor, cada agente sabe a posicao REAL do outro a cada passo, mira nela
direto (nao no ponto pre-calculado) e freia assim que a distancia ao vivo cai
abaixo de `min_separation_m` -- convergencia de verdade, nao um alvo fixo.

O QUE ISTO NAO RESOLVE: contato fisico (aperto de mao, abraco, luta). O
checkpoint publico (`out/G1-clip.ckpt`) so tem clipes de LOCOMOCAO -- ver
`PRIMITIVE_TO_CLIP` em motion_command_geometry.py. O diretor traz os dois
agentes pra perto um do outro, parados, encarando-se; nao ha colisao real
nem sincronizacao de gesto porque nao existe clipe de gesto pra sincronizar.
Isso precisa de um checkpoint com clipes de upper-body -- nao e um limite
deste script.

DEFASAGEM DE UM PASSO: os agentes rodam em SEQUENCIA dentro do mesmo laco
(nao em paralelo de verdade), entao quem roda por ultimo numa rodada ve a
posicao dos outros ATUALIZADA nesta MESMA rodada, e quem roda primeiro ve a
posicao deles da rodada ANTERIOR. A dt=1/30s isso e ~3cm de erro no pior
caso -- irrelevante perto de min_separation_m (tipicamente 1 m).

Roda com o PYTHON DO VENV DO MOTIONBRICKS:

    ...\\motionbricks\\.venv\\Scripts\\python.exe script_pipeline/motionbricks_scene_director.py \\
        --motion-plan RUN/parse/motion_plan.json --actors "ANA,BIA" --out-dir saida/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

MOTIONBRICKS_ROOT = r"E:\Users\home\Documents\GR00T-WholeBodyControl\motionbricks"


def demo_args(steps: int, seed: int) -> SimpleNamespace:
    """Mesmo espelho de motionbricks_headless.py -- ver docstring la."""
    return SimpleNamespace(
        humanoid_xml="assets/skeletons/g1/scene_29dof.xml", result_dir="./out",
        data_root="./datasets", explicit_dataset_folder=None, reprocess_clips=0,
        controller="random", lookat_movement_direction=0, has_viewer=0,
        pre_filter_qpos=1, source_root_realignment=1, target_root_realignment=1,
        force_canonicalization=1, skip_ending_target_cond=0, random_speed_scale=0,
        speed_scale=[0.8, 1.2], generate_dt=2.0, max_steps=steps, random_seed=seed,
        num_runs=1, use_qpos=1, planner="default", allowed_mode=None, clips="G1",
        return_model_configs=True, return_dataloader=True, recording_dir=None,
        EXP="default",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion-plan", required=True,
                    help="parse/motion_plan.json da decupagem (motion_conditioner.py --apply)")
    ap.add_argument("--actors", required=True,
                    help="nomes dos personagens a dirigir juntos, separados por virgula")
    ap.add_argument("--out-dir", default=None, help="pasta pra um <ator>.npz por personagem")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--anchor-scale", type=float, default=1.0,
                    help="escala (m) das ancoras de blocking do motion_plan")
    ap.add_argument("--min-separation", type=float, default=None,
                    help="sobrepoe o min_separation_m do motion_plan.json (freio de colisao ao vivo)")
    ap.add_argument("--brake-margin", type=float, default=1.0,
                    help="quanto ANTES de min_separation_m o freio comeca a pedir parada, pra "
                         "absorver o planejamento em blocos deste checkpoint (padrao 1.0m, "
                         "calibrado 2026-09-17: <1%% de erro na separacao final medida -- "
                         "ver motion_command_geometry.walk_target para a tabela completa)")
    args = ap.parse_args(argv)

    actors = [a.strip() for a in args.actors.split(",") if a.strip()]
    if len(actors) < 2:
        print("--actors precisa de 2+ nomes -- com 1 so, use motionbricks_headless.py", file=sys.stderr)
        return 1
    motion_plan_path = Path(args.motion_plan).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    os.chdir(MOTIONBRICKS_ROOT)
    sys.path.insert(0, MOTIONBRICKS_ROOT)
    import numpy as np
    import torch as t
    import mujoco
    from motionbricks.motion_backbone.demo.utils import navigation_demo
    from motionbricks_command_controller import MotionPlanController
    from motion_command_geometry import formation_by_scene

    motion_plan = json.loads(motion_plan_path.read_text(encoding="utf-8"))
    min_sep = args.min_separation if args.min_separation is not None else motion_plan.get("min_separation_m", 1.0)

    live_positions: dict[str, np.ndarray] = {}
    agents = {}
    total_duration = 0.0
    for actor in actors:
        t0 = time.time()
        demo = navigation_demo(demo_args(1, args.seed))
        controller = MotionPlanController(
            motion_plan, actor, anchor_scale_m=args.anchor_scale,
            live_positions=live_positions, min_separation_m=min_sep,
            brake_margin_m=args.brake_margin)
        demo.controller = controller
        agents[actor] = demo
        # Publica a posicao inicial ANTES de qualquer passo -- sem isto, o
        # primeiro agente a rodar nao acha os outros em live_positions. Ja no
        # referencial de MUNDO (qpos local + ancora), como o controlador
        # publica a partir do 1o passo -- sem isto os dois nasceriam
        # sobrepostos em (0,0) e o freio de colisao dispararia sem ninguem
        # sair do lugar (ACHADO 2026-09-17, ver o comentario em
        # MotionPlanController.generate_control_signals).
        primeiro_comando = controller.commands[0]
        formation0 = formation_by_scene(motion_plan).get(primeiro_comando.get("scene"), {})
        ancora0 = (formation0.get(actor) or {}).get("anchor_m")
        offset0 = np.asarray(ancora0[:2], dtype=np.float64) * args.anchor_scale if ancora0 else np.zeros(2)
        live_positions[actor] = demo.mj_data.qpos[:2].copy() + offset0
        duration = max(c["start_s"] + c["duration_s"] for c in controller.commands)
        total_duration = max(total_duration, duration)
        print(f"[mb-director] {actor}: modelos carregados em {time.time() - t0:.1f}s, "
              f"{len(controller.commands)} comando(s), {duration:.2f}s", flush=True)

    dt = next(iter(agents.values())).mj_model.opt.timestep
    steps = max(1, int(round(total_duration / dt)))
    print(f"[mb-director] {len(actors)} ator(es), {total_duration:.2f}s -> {steps} passo(s) "
          f"(dt={dt:.4f}s), min_separation_m={min_sep:.2f}", flush=True)

    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    for demo in agents.values():
        demo.full_agent.reset()
        demo.controller.reset()
    frames = {actor: [] for actor in actors}
    freio_disparado: dict[str, int] = {}

    t1 = time.time()
    for step in range(1, steps + 1):
        for actor in actors:
            demo = agents[actor]
            antes = live_positions[actor].copy()
            qpos = demo.full_agent.get_next_frame()
            frames[actor].append(np.array(qpos, dtype=np.float32).reshape(-1))
            demo.mj_data.qpos[:] = qpos
            control = demo.controller.generate_control_signals(
                None, demo.mj_model, demo.mj_data, visualize=False,
                control_info={"force_idle": False, "allowed_mode": None})
            control["context_mujoco_qpos"] = demo.full_agent.get_context_mujoco_qpos()
            with t.no_grad():
                demo.full_agent.generate_new_frames(control, demo.controller.get_controller_dt() * 2.0)
            mujoco.mj_forward(demo.mj_model, demo.mj_data)
            # Freio de colisao: so pra log -- generate_control_signals ja usou
            # o freio internamente (walk_target); isto aqui e so visibilidade.
            depois = live_positions[actor]
            if np.linalg.norm(depois - antes) < 1e-6 and step > 1:
                freio_disparado[actor] = freio_disparado.get(actor, 0) + 1

    elapsed = time.time() - t1
    print(f"[mb-director] {steps} passo(s) x {len(actors)} ator(es) em {elapsed:.2f}s "
          f"({steps * len(actors) / max(elapsed, 1e-6):.0f} agente-fps)", flush=True)

    def _own_anchor(actor):
        for form in formation_by_scene(motion_plan).values():
            info = form.get(actor)
            if info:
                return np.asarray(info["anchor_m"][:2], dtype=np.float64) * args.anchor_scale
        return np.zeros(2)

    for actor in actors:
        qpos_all = np.stack(frames[actor])
        anchor_mundo = _own_anchor(actor)
        mundo_final = qpos_all[-1, :2] + anchor_mundo
        print(f"[mb-director] {actor}: qpos {qpos_all.shape}, "
              f"raiz final (mundo)=({mundo_final[0]:.3f},{mundo_final[1]:.3f})", flush=True)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{actor}.npz"
            # ACHADO 2026-09-17 (auditoria externa, risco adicional): o NPZ so
            # guardava `qpos` (LOCAL -- cada agente nasce em (0,0), ver
            # MotionPlanController.generate_control_signals) e `dt`. Quem
            # reabrisse o arquivo depois precisava reconstruir a ancora de
            # mundo por fora (motion_plan.json + formation_by_scene) pra
            # saber onde o agente REALMENTE estava. Agora salva tambem
            # `anchor_m` (deslocamento usado) e `qpos_world_xy` (raiz ja em
            # metros de mundo, pronto pra usar sem reconstruir contexto
            # externo -- ex.: alimentar script_pipeline/pose_video.py).
            #
            # LIMITE CONHECIDO, nao resolvido aqui: `_own_anchor` usa a
            # PRIMEIRA cena (na ordem do dict) em que o ator aparece --
            # correto pra uma corrida de cena unica (o caso de teste deste
            # script ate agora), mas NAO acompanha troca de formacao entre
            # cenas dentro do mesmo NPZ. Ver [[project_pose_adapter_and_h3_controlnet]].
            qpos_world_xy = qpos_all[:, :2] + anchor_mundo
            np.savez(out_path, qpos=qpos_all, dt=dt, actor=actor,
                     anchor_m=anchor_mundo, qpos_world_xy=qpos_world_xy)
            print(f"[mb-director] {actor}: salvo em {out_path}")

    if len(actors) == 2:
        # qpos salvo e LOCAL a cada agente (nasce em (0,0) sempre -- ver
        # comentario em MotionPlanController.generate_control_signals);
        # desloca pela ancora de cada um pra medir no MESMO referencial que
        # `formation`/`live_positions` usam.
        qa = np.stack(frames[actors[0]])[:, :2] + _own_anchor(actors[0])
        qb = np.stack(frames[actors[1]])[:, :2] + _own_anchor(actors[1])
        n = min(len(qa), len(qb))
        dist_final = float(np.linalg.norm(qa[n - 1] - qb[n - 1]))
        dist_min = float(np.min(np.linalg.norm(qa[:n] - qb[:n], axis=1)))
        print(f"[mb-director] distancia entre {actors[0]} e {actors[1]} (referencial de mundo): "
              f"final={dist_final:.3f} m, minima_no_percurso={dist_min:.3f} m "
              f"(min_separation_m pedido={min_sep:.2f} m)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
