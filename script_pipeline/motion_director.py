"""Diretor de acao/movimento: liga a decupagem (shot_plan.json) ao InterGen
(texto -> esqueleto de duas pessoas) e ao adaptador de pose (pose_video.py),
produzindo um video de pose por PLANO com 2 personagens nomeados e uma pista
de interacao fisica no texto -- e o elo que faltava entre "temos InterGen/
Inter-X funcionando" e "um plano da decupagem usa isso de verdade".

Fluxo por plano candidato:
  1. `find_interaction_shots()` -- casa `beat`/`fallback`/`video_prompt` do
     shot contra um dicionario de pistas PT/EN, so em planos com `subject` E
     `co_subject` (2 personagens nomeados).
  2. `run_intergen_batch()` -- roda o InterGen UMA VEZ pra todos os prompts
     unicos encontrados (carregar o checkpoint custa ~15s; rodar por plano
     separado desperdicaria isso a cada plano).
  3. `pose_video.render_pose_video()` -- esqueleto (T,2,J,3) do InterGen vira
     video OpenPose no tamanho/duracao REAIS do plano (largura/altura que a
     corrida usou, `shot["frames"]`).
  4. `render_conditioned_shot()` -- chama `ltx25_backend.generate()`, O MESMO
     caminho que `render_shots.py` usa em producao: still do plano como I2V
     (identidade) + o video de pose como IC-LoRA Union-Control (movimento).

NAO tenta ligar o personagem generico do InterGen (sempre "pessoa 1"/"pessoa
2") a qual ator especifico fica em qual lado de tela -- a guia de pose e um
sinal de MOVIMENTO, a identidade continua vindo do still (I2V) e do
video_prompt (que ja descreve cada personagem). Ver MEMORIAL/memoria de
projeto pra essa limitacao.

CLI:
    .venv/Scripts/python.exe -m script_pipeline.motion_director \\
        --run-dir outputs/decupagem_01_encontro_corredor --width 768 --height 512

    --dry-run lista os planos casados sem gerar nada (rapido, sem GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERGEN_ROOT = Path(os.environ.get("INTERGEN_ROOT", r"E:\Users\home\Documents\InterGen"))
INTERGEN_PYTHON = INTERGEN_ROOT / ".venv" / "Scripts" / "python.exe"
INTERGEN_SOURCE_FPS = 20.0  # utils/preprocess.py::FPS -- ver achado da auditoria externa

# Pistas PT/EN -> prompt InterGen (frases do proprio prompts.txt de exemplo do
# InterGen, que sao as mais testadas nesta sessao). Ordem importa: a PRIMEIRA
# pista que bater decide -- mais especifico primeiro.
INTERACTION_CUES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(luta|briga|duelo|combate|golpe|soca|espada|fight|duel|combat|punch|sword)", re.I),
     "In an intense boxing match, one is continuously punching while the other is defending and counterattacking."),
    (re.compile(r"\b(dan[çc]a|dance)", re.I),
     "With fiery passion two dancers entwine in Latin dance sublime."),
    (re.compile(r"\b(abra[çc]|embrace|hug)", re.I),
     "Two people embrace each other."),
    (re.compile(r"\b(beij|kiss)", re.I),
     "Two people embrace each other."),  # proxy discreto, evita gerar beijo explicito
    (re.compile(r"\b(revere|bow)", re.I),
     "Two people bow to each other."),
    (re.compile(r"\b(comemor|celebra|celebrate)", re.I),
     "Two good friends jump in the same rhythm to celebrate."),
    (re.compile(r"\b(esbarr|colid|bump|collide)", re.I),
     "Two people bump into each other."),
    # ACHADO 2026-09-18 (rodando nos dois filmes-alvo): "push" sozinho colide
    # com "the camera pushes in slowly" -- boilerplate de MOVIMENTO DE CAMERA
    # que aparece em quase todo shot["video_prompt"] com movement="push", sem
    # nenhuma relacao com empurrar uma pessoa. Removido o "push" isolado em
    # ingles; "empurra" (PT) nao colide porque o boilerplate e em ingles.
    (re.compile(r"\b(empurr|agarr|segur.*bra[çc]o|grab(?:s|bed|bing)? (?:him|her|them|each other))", re.I),
     "Two people bump into each other."),
    (re.compile(r"\b(encara|confront|desafi|glare|standoff|face.?off|tense stare)", re.I),
     "The two are blaming each other and having an intense argument."),
]


def _shot_text(shot: dict[str, Any]) -> str:
    return " ".join(str(shot.get(k, "")) for k in ("beat", "fallback", "video_prompt", "storyboard_prompt"))


def find_interaction_shots(shot_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """[{shot, cue_pattern, intergen_prompt}] -- so planos com subject E
    co_subject (2 personagens nomeados) cujo texto bate alguma pista.

    `shot["_global_index"]` (0-indexado na lista `shots` inteira, NAO
    reiniciado por cena) fica anexado -- ACHADO 2026-09-18 (rodando de
    verdade no filme dos piratas): `position` e 0-indexado POR CENA (a
    primeira tomada de cada cena e position=0), mas os arquivos de still
    extraidos em `intermediate/_verify_frames/sceneNN_shotMMM.png` numeram
    MMM pela posicao GLOBAL na lista inteira -- so bate com `position`
    direto na PRIMEIRA cena do roteiro (onde os dois indices coincidem por
    acaso); em qualquer cena depois da primeira, `position` sozinho aponta
    pro arquivo errado."""
    found = []
    for global_index, shot in enumerate(shot_plan.get("shots", [])):
        if not (shot.get("subject") and shot.get("co_subject")):
            continue
        shot = {**shot, "_global_index": global_index}
        texto = _shot_text(shot)
        for pattern, prompt in INTERACTION_CUES:
            if pattern.search(texto):
                found.append({"shot": shot, "cue": pattern.pattern, "intergen_prompt": prompt})
                break
    return found


def run_intergen_batch(prompts: list[str], *, timeout: int = 600, log=print) -> dict[str, Path]:
    """Roda o InterGen UMA VEZ pra todos os `prompts` unicos (preserva ordem).
    Devolve {prompt: caminho do primeiro _0_joints.npy}. Levanta RuntimeError
    se o InterGen nao estiver instalado (ver project_intergen_setup.md) ou se
    a geracao falhar."""
    if not INTERGEN_PYTHON.exists():
        raise RuntimeError(f"InterGen nao encontrado em {INTERGEN_PYTHON} -- ver project_intergen_setup.md")
    unicos = list(dict.fromkeys(prompts))  # dedup preservando ordem
    if not unicos:
        return {}

    prompts_path = INTERGEN_ROOT / "prompts.txt"
    backup = INTERGEN_ROOT / "prompts.txt.director_bak"
    results_dir = INTERGEN_ROOT / "results"
    had_backup = prompts_path.exists()
    if had_backup:
        shutil.copy2(prompts_path, backup)
    try:
        prompts_path.write_text("\n".join(unicos) + "\n", encoding="utf-8")
        for old in results_dir.glob("*_joints.npy"):
            old.unlink()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "1"
        log(f"[motion_director] rodando InterGen para {len(unicos)} prompt(s) unico(s)...")
        proc = subprocess.run(
            [str(INTERGEN_PYTHON), "tools/infer.py"], cwd=str(INTERGEN_ROOT),
            env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"InterGen falhou (codigo {proc.returncode}):\n{proc.stderr[-3000:]}")

        out = {}
        for prompt in unicos:
            nome = prompt[:48]
            candidato = results_dir / f"{nome}_0_joints.npy"
            if candidato.exists():
                out[prompt] = candidato
            else:
                log(f"[motion_director] AVISO -- nao achei joints pra {prompt!r} (esperado {candidato.name})")
        return out
    finally:
        if had_backup:
            shutil.move(str(backup), str(prompts_path))
        elif prompts_path.exists():
            prompts_path.unlink()


def _locate_still(run_dir: Path, scene: int, global_index: int) -> Path | None:
    """Tenta achar UMA imagem que sirva de still (I2V) pro plano -- corridas
    antigas nao guardam o still original avulso (cache por conteudo, ver
    project_auditoria_scripts_20260916.md), so o QUADRO FINAL do clipe ja
    renderizado (`intermediate/_verify_frames/`). Nao e o still que
    condicionou a geracao original, mas serve pra um teste/re-render real.

    `global_index` e o indice 0-based na lista `shots` INTEIRA (ver
    `find_interaction_shots`) -- os arquivos numeram MMM por essa posicao
    global, NAO por `shot["position"]` (que reinicia em 0 a cada cena;
    ACHADO 2026-09-18, ver docstring de `find_interaction_shots`)."""
    candidatos = [
        run_dir / "intermediate" / "_verify_frames" / f"scene{scene:02d}_shot{global_index:03d}.png",
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None


def render_conditioned_shot(shot: dict[str, Any], pose_video: Path, still: Path, out_path: Path,
                            *, timeout: int = 2400, log=print) -> str:
    """Chama ltx25_backend.generate() -- O MESMO caminho de producao que
    render_shots.py usa -- com o still do plano (I2V) + o video de pose
    (IC-LoRA Union-Control, mesmo LoRA 2.3 que carrega no 2.5).

    ACHADO 2026-09-18 (primeiro teste real deste modulo): union-control tem
    reference_downscale_factor=2 (ver CLAUDE.md) -- o guia de video e
    reamostrado pra height//2 x width//2 e ENTAO passa pela VAE (compressao
    espacial 32x). Se width/height nao forem multiplos de 64, o resultado de
    `//2` nao e multiplo de 32 e a VAE do guia quebra (`einops.EinopsError:
    Shape mismatch`) depois de MINUTOS de amostragem (o erro so aparece no
    ultimo no do grafo). Resolucoes 768x512/1280x704 (ja validadas em
    producao) sao multiplas de 64; 864x480 (o que usei no primeiro teste,
    copiado de um frame extraido de um render MiniMax antigo) NAO E."""
    w, h = shot.get("_width", 768), shot.get("_height", 512)
    if w % 64 or h % 64:
        raise ValueError(
            f"width={w}/height={h} precisam ser multiplos de 64 "
            "(reference_downscale_factor=2 do union-control + compressao espacial "
            "32x da VAE) -- use 768x512 ou 1280x704.")
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("LTX_COMFY_EXTRA_ARGS", "--disable-dynamic-vram")
    os.environ["LTX_COMFY_CACHE_NONE"] = "1"
    import ltx25_backend
    import ltx_loras

    # ic_lora["lora"] precisa do NOME LOCAL do arquivo -- ltx25_backend.
    # _require_lora()/installed_path() fazem lookup literal de arquivo em
    # models/loras|2.5/loras, NAO pela chave do catalogo (achado rodando o
    # teste real deste modulo pela primeira vez: passar a chave direto
    # ("union-control-2.3") falha com FileNotFoundError).
    _, lora_local = ltx_loras.resolve("union-control-2.3")

    return ltx25_backend.generate(
        shot["video_prompt"], str(out_path),
        width=shot.get("_width", 768), height=shot.get("_height", 512),
        num_frames=shot["frames"], frame_rate=24.0, seed=1234,
        image_path=str(still), image_strength=1.0,
        ic_lora={"lora": lora_local, "video": str(pose_video), "strength": 1.0},
        log_cb=log, timeout=timeout,
    )


def render_conditioned_shot_minimax(shot: dict[str, Any], joints, still: Path, out_path: Path,
                                    *, width: int = 768, height: int = 480,
                                    source_fps: float = INTERGEN_SOURCE_FPS,
                                    timeout: int = 3600, log=print) -> str:
    """Equivalente a `render_conditioned_shot()`, mas pro MiniMax H3 em vez do
    LTX 2.5 -- mesma ideia (esqueleto do InterGen vira guia de movimento,
    still do plano da identidade), caminho de conditioning diferente:

    - O H3 nao tem IC-LoRA de pose; usa `MiniMaxH3FunControlNetApply` (peso
      `minimax_h3_fun_controlnet_union_pruned_*.safetensors`, ja plugado em
      `minimax_h3_backend.generate(control_video=...)`, ver CLAUDE.md secao
      MiniMax H3). O video de controle e gerado por
      `motion_to_h3_controlnet.render_h3_control_video()`, NAO por
      `pose_video.render_pose_video()` direto -- a grade do H3 e diferente
      (quadros 17n+5, dimensoes multiplas de 32, nao 64) e `render_h3_control_
      video()` ja aplica os dois ajustes sozinho a partir dos MESMOS `joints`.
    - O H3 nao tem I2V/still como parametro proprio (`image_path`, como no
      LTX) -- a identidade entra por `ref_images` (0-2 imagens de
      referencia). Passa o still do plano como unica `ref_image`.
    - `duration_seconds` (nao `num_frames`) -- o grafo arredonda pro 17n+5
      mais proximo sozinho; aqui pede-se a duracao equivalente a
      `shot["frames"]` a 24fps pra ficar alinhado ao plano do roteiro,
      mas a contagem REAL de saida pode nao bater exatamente com o LTX no
      mesmo shot (round-trip por `h3_frame_count`, ver motion_to_h3_
      controlnet.py) -- os dois motores nunca vao produzir o MESMO numero de
      quadros pro mesmo plano.

    ⚠️ NAO VALIDADO com GPU real (2026-09-18): o ControlNet do H3 estava
    medido em ~500s/passo com `MINIMAX_H3_VARIANT=fp8int8` (padrao) --
    ~15-25x mais lento que o baseline sem controle, hipotese nao testada de
    mismatch de familia de quantizacao (peso do controle e int8_convrot, unet
    padrao e fp8_scaled; testar `MINIMAX_H3_VARIANT=int8convrot` antes de
    rodar isto em producao). Ver memoria de projeto
    project_pose_adapter_and_h3_controlnet.md. Esta funcao so fecha o
    caminho de CODIGO que faltava (motion_director so tinha o lado LTX) --
    nao desbloqueia a performance."""
    sys.path.insert(0, str(REPO_ROOT))
    from script_pipeline.motion_to_h3_controlnet import render_h3_control_video
    import minimax_h3_backend

    control_path = out_path.with_name(out_path.stem + "_h3control.mp4")
    n_frames, h3_w, h3_h = render_h3_control_video(
        joints, control_path, width=width, height=height,
        num_frames=shot["frames"], source_fps=source_fps)
    log(f"[motion_director] H3 control video: {n_frames} quadros @ {h3_w}x{h3_h} -> {control_path}")

    duration_seconds = shot["frames"] / 24.0
    return minimax_h3_backend.generate(
        shot["video_prompt"], str(out_path),
        # aspect_ratio: NAO passar "16:9" -- o node ResolutionSelector exige o
        # rotulo completo do combo ("16:9 (Widescreen)"); "16:9" sozinho
        # derruba com HTTP 400 (ACHADO 2026-09-18, validacao com GPU real).
        # Deixa no default do backend (minimax_h3_backend.DEFAULT_ASPECT), que
        # ja tem o rotulo certo.
        ref_images=[str(still)],
        duration_seconds=duration_seconds, seed=1234,
        control_video=str(control_path), control_strength=1.0,
        log_cb=log, timeout=timeout,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", choices=["ltx", "minimax"], default="ltx",
                    help="minimax usa MiniMaxH3FunControlNetApply em vez de IC-LoRA -- "
                         "NAO VALIDADO com GPU real, ver render_conditioned_shot_minimax")
    ap.add_argument("--width", type=int, default=768,
                    help="multiplo de 64 no engine ltx (union-control exige); "
                         "arredondado pra multiplo de 32 no engine minimax")
    ap.add_argument("--height", type=int, default=512,
                    help="multiplo de 64 no engine ltx (union-control exige); "
                         "arredondado pra multiplo de 32 no engine minimax")
    ap.add_argument("--out-dir", default=None, help="padrao: <run-dir>/motion_director")
    ap.add_argument("--limit", type=int, default=None, help="renderiza no maximo N planos casados")
    ap.add_argument("--dry-run", action="store_true", help="so lista os planos casados, sem GPU")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    shot_plan = json.loads((run_dir / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    matches = find_interaction_shots(shot_plan)
    print(f"[motion_director] {len(matches)} plano(s) com 2 personagens + pista de interacao "
          f"de {shot_plan.get('total_shots')} plano(s) totais.")
    for m in matches:
        s = m["shot"]
        print(f"  cena {s['scene']} pos {s['position']}: {s['subject']}+{s['co_subject']} "
              f"[{m['cue']}] -> InterGen: {m['intergen_prompt']!r}")
    if args.dry_run or not matches:
        return 0

    if args.limit:
        matches = matches[: args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "motion_director"
    out_dir.mkdir(parents=True, exist_ok=True)

    from script_pipeline.pose_video import render_pose_video

    joints_by_prompt = run_intergen_batch([m["intergen_prompt"] for m in matches])

    resultados = []
    for m in matches:
        shot, prompt = m["shot"], m["intergen_prompt"]
        tag = f"scene{shot['scene']:02d}_pos{shot['position']:02d}"
        joints_path = joints_by_prompt.get(prompt)
        if joints_path is None:
            print(f"[motion_director] {tag}: sem esqueleto do InterGen -- pulando.")
            continue
        import numpy as np
        joints = np.load(joints_path)

        still = _locate_still(run_dir, shot["scene"], shot["_global_index"])
        if still is None:
            print(f"[motion_director] {tag}: nenhum still encontrado -- pulando render.")
            continue
        out_path = out_dir / f"{tag}_conditioned.mp4"

        try:
            if args.engine == "minimax":
                render_conditioned_shot_minimax(shot, joints, still, out_path,
                                                width=args.width, height=args.height)
                pose_registrada = str(out_path.with_name(out_path.stem + "_h3control.mp4"))
            else:
                pose_path = out_dir / f"{tag}_pose.mp4"
                n = render_pose_video(joints, pose_path, width=args.width, height=args.height,
                                      fps=24.0, source_fps=INTERGEN_SOURCE_FPS, num_frames=shot["frames"])
                print(f"[motion_director] {tag}: video de pose {n} quadros -> {pose_path}")
                shot["_width"], shot["_height"] = args.width, args.height
                render_conditioned_shot(shot, pose_path, still, out_path)
                pose_registrada = str(pose_path)
            print(f"[motion_director] {tag}: renderizado ({args.engine}) -> {out_path}")
            resultados.append({"tag": tag, "scene": shot["scene"], "position": shot["position"],
                               "engine": args.engine, "pose_video": pose_registrada,
                               "still": str(still), "output": str(out_path)})
        except Exception as exc:
            print(f"[motion_director] {tag}: FALHOU ({type(exc).__name__}: {exc})")

    (out_dir / "director_report.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[motion_director] {len(resultados)}/{len(matches)} plano(s) renderizado(s). "
          f"Relatorio: {out_dir / 'director_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
