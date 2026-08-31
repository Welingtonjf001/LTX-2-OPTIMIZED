"""Union-Control com DOIS guias encadeados: pose + profundidade.

Experimento. A IC-LoRA Union-Control foi treinada com Canny + Depth + Pose, mas
o workflow tem UM no `LTXAddVideoICLoRAGuide` -- ou seja, ela aceita um tipo de
controle por vez, nao os tres em paralelo. O que este patch testa e se ela
aceita os tres ENCADEADOS: o no de guia tem entradas e saidas do mesmo tipo
(positive/negative/latent), entao um segundo guia pode consumir a saida do
primeiro.

    python comfy_union_depth_patch.py --pose bundle/pose.mp4 --depth bundle/depth.mp4 \
        --prompt "..." --out _api_union_depth.json
    python comfy_run.py --workflow _api_union_depth.json

A pergunta que ele responde: alimentar profundidade ALEM da pose melhora a
aderencia (PCK) ou atrapalha? Nao ha teoria que decida -- a curva de forca da
propria Union-Control estava invertida no MEMORIAL, e so a medicao mostrou.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

COMFY_INPUT = Path("ComfyUI/input")
GUIDE_NODE = "5012"
GUIDE2_NODE = "5112"          # o segundo guia, criado aqui
LOAD_VIDEO2_NODE = "5101"     # LoadVideo da profundidade
COMPONENTS2_NODE = "5100"     # GetVideoComponents
RESIZE2_NODE = "5128"         # ResizeImageMaskNode
ICLORA_NODE = "5011"
# quem consumia a saida do guia 1 passa a consumir a do guia 2
CONSUMIDORES = {"5013": ("positive", "negative"), "4828": ("positive", "negative"),
                "4528": ("video_latent",)}


def copiar_para_input(src: Path, prefixo: str) -> str:
    """Nome unico por execucao -- o ComfyUI cacheia LoadVideo pelo nome."""
    if not src.exists():
        raise SystemExit(f"nao existe: {src}")
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    dig = hashlib.sha1(f"{src.resolve()}|{src.stat().st_mtime_ns}".encode()).hexdigest()[:8]
    nome = f"choreo_{prefixo}_{dig}{src.suffix}"
    shutil.copy2(src, COMFY_INPUT / nome)
    return nome


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iclora-strength", type=float, default=1.0)
    ap.add_argument("--distilled-strength", type=float, default=0.5)
    ap.add_argument("--pose-strength", type=float, default=1.0)
    ap.add_argument("--depth-strength", type=float, default=1.0)
    ap.add_argument("--prefix", default="uniondepth")
    ap.add_argument("--out", default="_api_union_depth.json")
    args = ap.parse_args()

    # 1. parte do patch de pose que ja e validado -- nao reimplementar
    base_out = Path(f"_api_union_base_{args.prefix}.json")
    cmd = [
        sys.executable, "comfy_union_patch.py",
        "--pose", args.pose, "--prompt", args.prompt,
        "--strength", str(args.pose_strength),
        "--iclora-strength", str(args.iclora_strength),
        "--distilled-strength", str(args.distilled_strength),
        "--seed", str(args.seed),
        "--width", str(args.width), "--height", str(args.height),
        "--prefix", args.prefix, "--out", str(base_out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"patch de pose falhou:\n{(r.stdout or '') + (r.stderr or '')}")

    api = json.loads(base_out.read_text(encoding="utf-8"))
    if GUIDE_NODE not in api:
        raise SystemExit(f"no de guia {GUIDE_NODE} sumiu do workflow")

    depth_nome = copiar_para_input(Path(args.depth), args.prefix + "_depth")
    g1 = api[GUIDE_NODE]

    # 2. carregar a profundidade
    api[LOAD_VIDEO2_NODE] = {"class_type": "LoadVideo", "inputs": {"file": depth_nome}}
    api[COMPONENTS2_NODE] = {"class_type": "GetVideoComponents",
                             "inputs": {"video": [LOAD_VIDEO2_NODE, 0]}}
    api[RESIZE2_NODE] = {
        "class_type": "ResizeImageMaskNode",
        "inputs": {
            "input": [COMPONENTS2_NODE, 0],
            "scale_method": "lanczos",
            "resize_type": "scale dimensions",
            "resize_type.width": args.width,
            "resize_type.height": args.height,
            "resize_type.crop": "center",
        },
    }

    # 3. segundo guia, consumindo a saida do primeiro
    api[GUIDE2_NODE] = {
        "class_type": "LTXAddVideoICLoRAGuide",
        "inputs": {
            **{k: v for k, v in g1["inputs"].items()
               if k in ("frame_idx", "crop", "use_tiled_encode", "tile_size",
                        "tile_overlap", "vae", "latent_downscale_factor")},
            "strength": args.depth_strength,
            "positive": [GUIDE_NODE, 0],
            "negative": [GUIDE_NODE, 1],
            "latent": [GUIDE_NODE, 2],
            "image": [RESIZE2_NODE, 0],
        },
    }

    # 4. religar quem lia o guia 1 para ler o guia 2
    religados = 0
    for no, campos in CONSUMIDORES.items():
        if no not in api:
            continue
        for campo in campos:
            v = api[no]["inputs"].get(campo)
            if isinstance(v, list) and v and v[0] == GUIDE_NODE:
                api[no]["inputs"][campo] = [GUIDE2_NODE, v[1]]
                religados += 1
    if religados == 0:
        raise SystemExit("ninguem consumia o guia 1 -- workflow inesperado")

    Path(args.out).write_text(json.dumps(api, indent=2), encoding="utf-8")
    base_out.unlink(missing_ok=True)
    print(f"dois guias: pose(strength={args.pose_strength}) -> "
          f"depth(strength={args.depth_strength}); {religados} consumidores religados")
    print(f"saida: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
