"""Runner HEADLESS do MotionBricks (NVIDIA, G1).

POR QUE EXISTE: `scripts/interactive_demo_g1.py` do repo importa
`mujoco.viewer` no topo do arquivo, e esse import exige `glfw` mesmo com
`--has_viewer 0` -- MEDIDO 2026-09-17: o smoke morre em
`ModuleNotFoundError: No module named 'glfw'` antes de carregar um unico
checkpoint. Este runner reproduz SO o ramo sem viewer daquele script
(`navigation_demo` + laco `get_next_frame`/`generate_new_frames`) sem tocar
no repo do MotionBricks, e grava o qpos de cada frame -- que e o que um
retarget para o rig do personagem consome depois.

Roda com o PYTHON DO VENV DO MOTIONBRICKS (torch 2.4+cu124, mujoco 3.13),
nao com o .venv do LTX:

    # Smoke aleatorio (so prova que o runtime carrega e gera):
    ...\\motionbricks\\.venv\\Scripts\\python.exe script_pipeline/motionbricks_headless.py --steps 120 --out out.npz

    # Dirigido pela decupagem, um ator por vez:
    ...\\motionbricks\\.venv\\Scripts\\python.exe script_pipeline/motionbricks_headless.py \\
        --motion-plan RUN/parse/motion_plan.json --actor "SEO-YEON" --out seo_yeon.npz

Com `--motion-plan`, o controlador e `MotionPlanController` (motionbricks_
command_controller.py), que le `commands[]` por ator em vez do `random` do
proprio repo -- ver o limite do checkpoint publico (so locomocao) documentado
la. `--steps` sem `--motion-plan` continua sendo o smoke aleatorio de sempre.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

MOTIONBRICKS_ROOT = r"E:\Users\home\Documents\GR00T-WholeBodyControl\motionbricks"


def demo_args(steps: int, seed: int) -> SimpleNamespace:
    """Espelho dos defaults do argparse de interactive_demo_g1.py, mais o que
    o main() dele anexa depois do parse. Sem viewer, controlador aleatorio."""
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
    ap.add_argument("--steps", type=int, default=120,
                    help="ignorado com --motion-plan: a duracao dos comandos do ator manda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="npz com qpos[T, 36] e dt")
    ap.add_argument("--motion-plan", default=None,
                    help="parse/motion_plan.json da decupagem (motion_conditioner.py --apply)")
    ap.add_argument("--actor", default=None,
                    help="nome do personagem a dirigir -- obrigatorio com --motion-plan")
    ap.add_argument("--anchor-scale", type=float, default=1.0,
                    help="escala (m) das ancoras de blocking do motion_plan")
    args = ap.parse_args(argv)
    if args.motion_plan and not args.actor:
        print("--motion-plan exige --actor", file=sys.stderr)
        return 1
    # Resolvidos ANTES do chdir: --motion-plan/--out normalmente vem do
    # run-dir da decupagem (outro repo), caminho relativo ao cwd de quem
    # chamou -- depois do chdir para MOTIONBRICKS_ROOT um caminho relativo
    # apontaria pro lugar errado.
    motion_plan_path = Path(args.motion_plan).resolve() if args.motion_plan else None
    out_path = Path(args.out).resolve() if args.out else None

    os.chdir(MOTIONBRICKS_ROOT)
    sys.path.insert(0, MOTIONBRICKS_ROOT)
    import json
    import numpy as np
    import torch as t
    import mujoco
    from motionbricks.motion_backbone.demo.utils import navigation_demo
    # Modulo IRMAO (mesma pasta script_pipeline/) -- import simples, sem
    # prefixo de pacote: este arquivo roda como script solto, nao via `-m`,
    # e o interpretador ja poe a pasta dele em sys.path[0].
    from motionbricks_command_controller import MotionPlanController

    t0 = time.time()
    demo = navigation_demo(demo_args(args.steps, args.seed))
    print(f"[mb-headless] modelos carregados em {time.time() - t0:.1f}s "
          f"(cuda={t.cuda.is_available()}, fps={demo.inferencer.motion_rep.fps})", flush=True)

    steps = args.steps
    if motion_plan_path:
        motion_plan = json.loads(motion_plan_path.read_text(encoding="utf-8"))
        demo.controller = MotionPlanController(motion_plan, args.actor, anchor_scale_m=args.anchor_scale)
        total_duration = max(c["start_s"] + c["duration_s"] for c in demo.controller.commands)
        dt = demo.mj_model.opt.timestep
        steps = max(1, int(round(total_duration / dt)))
        print(f"[mb-headless] motion_plan: ator={args.actor!r}, {len(demo.controller.commands)} "
              f"comando(s), {total_duration:.2f}s -> {steps} passo(s) (dt={dt:.4f}s)", flush=True)

    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    demo.full_agent.reset()
    demo.controller.reset()
    frames = []
    t1 = time.time()
    for step in range(1, steps + 1):
        # force_idle so faz sentido no smoke aleatorio (acalma o root antes de
        # parar); dirigido pelo plano, quem decide parar e o proprio comando.
        force_idle = (motion_plan_path is None) and (step + 100 > steps)
        qpos = demo.full_agent.get_next_frame()
        frames.append(np.array(qpos, dtype=np.float32).reshape(-1))
        demo.mj_data.qpos[:] = qpos
        control = demo.controller.generate_control_signals(
            None, demo.mj_model, demo.mj_data, visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": None})
        control["context_mujoco_qpos"] = demo.full_agent.get_context_mujoco_qpos()
        with t.no_grad():
            demo.full_agent.generate_new_frames(
                control, demo.controller.get_controller_dt() * 2.0)
        mujoco.mj_forward(demo.mj_model, demo.mj_data)
    elapsed = time.time() - t1
    qpos_all = np.stack(frames)
    print(f"[mb-headless] {len(frames)} frames em {elapsed:.2f}s "
          f"({len(frames) / max(elapsed, 1e-6):.0f} fps), qpos {qpos_all.shape}, "
          f"raiz min/max x={qpos_all[:, 0].min():.2f}/{qpos_all[:, 0].max():.2f}", flush=True)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, qpos=qpos_all, dt=demo.mj_model.opt.timestep)
        print(f"[mb-headless] salvo em {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
