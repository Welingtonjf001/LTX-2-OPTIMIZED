"""Estágio [5-D]: variante DECUPADA do estágio 5, plugável no pipeline validado.

O QUE ISTO É, E O QUE NÃO É

`render_scenes.py` (estágio 5) já renderiza um clipe por fala, com duração vinda
do TTS e o storyboard da cena como condicionamento. Ele está validado e NÃO é
tocado por este arquivo.

Esta é uma variante que troca a unidade de renderização: em vez de um clipe por
FALA, um clipe por PLANO de uma decupagem (`shot_plan.json`), com enquadramento,
ângulo, movimento e lado de tela decididos antes. As diferenças que justificam
existir:

  - o still é POR PLANO e carrega o enquadramento; em `render_scenes` o
    storyboard é por CENA, o mesmo para todos os clipes dela. MEDIDO em §3.21:
    o LTX trata câmera como destino e deriva ao longo do clipe, então um plano
    curto nunca chega ao enquadramento pedido pelo prompt -- ele precisa vir
    pronto no frame 0.
  - beats de AÇÃO viram planos próprios; em `render_scenes` a cena sem diálogo
    vira um clipe só e as ações intercaladas não existem como planos.
  - o texto da fala NÃO entra no prompt (§3.23: é o gatilho da legenda queimada).

COMO ELE NÃO ANULA O QUE JÁ FUNCIONA

Escreve `scenes/clips.json` no MESMO formato que `render_scenes` escreve. Com
isso os estágios [6] lipsync_scenes, [7] mix_audio, [8] assemble_final e
[9] verify_output rodam depois deste sem nenhuma modificação -- é o contrato de
manifesto que os liga, não o nome do módulo que os produziu.

Por isso este arquivo NÃO faz lip-sync nem concat: `render_shots.py` faz, para
uso avulso, mas dentro do pipeline quem faz é o estágio validado. Duplicar seria
manter duas implementações da mesma coisa divergindo.

    parse -> cast -> [emotion_director] -> TTS -> story_structure -> shot_plan
          -> [5-D] ESTE  ->  [6] lipsync -> [7] mix -> [8] assemble -> [9] verify

CLI:
    python -m script_pipeline.render_shots_stage --run-dir DIR [--width 960 --height 544]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_clips_manifest(feitos: list, plan: dict, dialogue: dict) -> list:
    """Converte a saída de render_shots para o formato de `scenes/clips.json`.

    Os campos são os que `lipsync_scenes` e `mix_audio` leem: `id`,
    `scene_index`, `line_index`, `character`, `video_path`, `audio_path`, `ok`.
    Plano de ação entra com `audio_path: None` -- é assim que o estágio 6
    reconhece "nada a sincronizar" e passa o clipe adiante intacto."""
    out = []
    for f in feitos:
        i = f["shot"]
        shot = plan["shots"][i]
        chave = (shot.get("scene"), shot.get("line_index"))
        wav = dialogue.get(chave, (None, None))[0]
        out.append({
            "id": f"scene{shot['scene']:02d}_shot{i:03d}",
            "scene_index": shot["scene"],
            "line_index": shot.get("line_index"),
            "character": shot.get("subject") or None,
            "video_path": f.get("clip"),
            "audio_path": wav,
            "ok": bool(f.get("clip")),
            # Campos extras da decupagem: os estágios seguintes ignoram o que
            # não conhecem, e eles ficam disponíveis para inspeção humana.
            "framing": shot.get("framing"),
            "angle": shot.get("angle"),
            "movement": shot.get("movement"),
            "screen_side": shot.get("screen_side"),
            "style": shot.get("style"),
            "planned_frames": shot.get("frames"),
        })
    return out


def main() -> int:
    import script_pipeline.render_shots as rs
    from script_pipeline import run_folder

    ap = argparse.ArgumentParser(description="Estagio 5-D: render por decupagem")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image-engine", default="flux", choices=["flux", "sd35", "sdxl"],
                    help='motor de imagem dos stills. flux = melhor adesao a enquadramento e lado de tela e aceita imagem de referencia, mas carrega ~4 min e encosta no teto de VRAM da 3090. sd35 = carrega em ~1 min, cabe em ~12 GB e amostra mais rapido, mas obedece menos o enquadramento e NAO tem referencia. --checkpoint/--clip/--vae explicitos continuam ganhando disto.')
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="passos de amostragem; sem isto usa o padrao do motor")
    ap.add_argument("--cfg", type=float, default=None,
                    help="CFG; sem isto usa o padrao do motor (o FLUX Klein ignora e usa guidance)")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--videos-only", action="store_true")
    ap.add_argument("--only-shots", default=None,
                    help="renderiza SO estes planos, por indice: \"0,2-5\"")
    ap.add_argument("--no-reference", action="store_true")
    # Ver render_shots.py: sem condicionamento o LTX inventa voz propria.
    ap.add_argument("--no-audio-conditioning", action="store_true")
    args = ap.parse_args()

    # O motor decide checkpoint, encoders e amostragem de uma vez. Passar
    # --checkpoint sozinho continua valendo: quem foi explicito manda.
    from script_pipeline.generate_storyboards import engine_defaults
    _motor = engine_defaults(args.image_engine)
    if not args.checkpoint:
        args.checkpoint = _motor["checkpoint"]
    if args.clip is None:
        args.clip = _motor["clip"]
    if args.vae is None:
        args.vae = _motor["vae"]

    run_dir = Path(args.run_dir).resolve()
    plan_path = run_dir / "parse" / "shot_plan.json"
    if not plan_path.exists():
        print(f"[5-D] {plan_path} nao existe -- rode shot_plan antes.", file=sys.stderr)
        return 1
    plan = json.load(open(plan_path, encoding="utf-8"))

    dlg_path = run_dir / "dialogue" / "lines.json"
    dialogo = rs.load_dialogue(str(dlg_path)) if dlg_path.exists() else {}
    if not dialogo:
        print("[5-D] sem dialogue/lines.json: os planos de fala sairao sem audio "
              "e o estagio 6 nao tera o que sincronizar.")

    shots_dir = run_folder.subdir(run_dir, "shots")
    log = lambda m: print(m, flush=True)  # noqa: E731
    feitos = rs.render(plan, shots_dir, width=args.width, height=args.height,
                       fps=args.fps, checkpoint=args.checkpoint, clip=args.clip,
                       vae=args.vae, seed=args.seed, limit=None,
                       stills_only=args.stills_only, videos_only=args.videos_only,
                       steps=args.steps, cfg=args.cfg, only_shots=args.only_shots,
                       use_reference=not args.no_reference, dialogue=dialogo,
                       audio_conditioning=not args.no_audio_conditioning, log=log)
    if args.stills_only:
        print(f"[5-D] {sum(1 for f in feitos if f.get('still'))} still(s); "
              "rode de novo com --videos-only para os clipes.")
        return 0

    # O handoff: mesmo caminho e mesmo formato que render_scenes produz.
    scenes_dir = run_folder.subdir(run_dir, "scenes")
    manifesto = build_clips_manifest(feitos, plan, dialogo)
    (scenes_dir / "clips.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for c in manifesto if c["ok"])
    print(f"\n[5-D] {ok}/{len(manifesto)} clipe(s) -> {scenes_dir / 'clips.json'}")
    print("[5-D] siga com os estagios validados:")
    print(f"    python -m script_pipeline.lipsync_scenes --run-dir {run_dir}")
    print(f"    python -m script_pipeline.mix_audio      --run-dir {run_dir}")
    print(f"    python -m script_pipeline.assemble_final --run-dir {run_dir}")
    if ok == len(manifesto):
        run_folder.mark_stage_complete(run_dir, "render")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
