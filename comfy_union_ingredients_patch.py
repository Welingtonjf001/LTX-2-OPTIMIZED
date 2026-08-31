"""Union-Control (pose) + Ingredients (personagem) no MESMO workflow.

    python comfy_union_ingredients_patch.py --pose bundle/pose.mp4 \
        --reference personagem.png --prompt "..." --out _api_combo.json

Por que um patch novo em vez de reusar um dos dois: os workflows existentes
disputam o MESMO slot. O `LTXAddVideoICLoRAGuide` (no 5012) tem um unico campo
`image`, e o union poe ali o video de pose enquanto o ingredients poe a imagem
de referencia repetida por `RepeatImageBatch`. Medido: com a referencia entrando
como frame inicial no workflow do union, a aderencia a pose caiu de PCK@0.2 0,76
para 0,17 -- os dois sinais competem.

A saida aqui e' um ENCADEAMENTO, que a estrutura dos nos permite:

    3940 -> 4922 (distilled) -> 5011 (union) -> 6011 (ingredients) -> CFGGuider.model

    1241 -> 5012 (guia POSE, downscale de 5011)
              |-- positive/negative --> 6012 (guia PERSONAGEM, downscale de 6011)
              +-- latent ------------->        |
                                               +--> CFGGuider / CropGuides / ConcatAVLatent

RESSALVA HONESTA: encadear guia ja falhou uma vez neste projeto -- o MEMORIAL
registra em 2026-08-15 que profundidade como segundo guia encadeado ficou
"indistinguivel de ruido". Aquilo era outro conteudo no segundo guia, mas o
risco e' o mesmo. O resultado disto PRECISA ser medido com `choreo/verify.py`,
nao suposto.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_INPUT = ROOT / "ComfyUI" / "input"

# nos que ja existem no union base
CKPT = "3940"
ICLORA_UNION = "5011"
GUIDE_POSE = "5012"
CFG_GUIDER = "4828"
CROP_GUIDES = "5013"
CONCAT_LATENT = "4528"

# nos novos (ids fora da faixa usada pelo workflow original)
ICLORA_ING = "6011"
GUIDE_CHAR = "6012"
LOAD_REF = "6004"
RESIZE_REF = "6069"
REPEAT_REF = "6093"
LEN_REF = "6072"

# Nesta instalacao os nomes de LoRA sao PLANOS (sem prefixo de pasta);
# o valor com "ltxv/ltx2/" e' rejeitado na validacao do ComfyUI.
ING_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="_api_union_base.json")
    ap.add_argument("--pose", required=True, help="pose.mp4 do bundle")
    ap.add_argument("--reference", required=True, help="imagem do personagem")
    ap.add_argument("--prompt", default="a person dancing, full body, plain background")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--ingredients-strength", type=float, default=1.0,
                    help="forca da IC-LoRA de personagem")
    ap.add_argument("--char-guide-strength", type=float, default=1.0,
                    help="forca do GUIA de personagem -- e' aqui que se equilibra "
                         "identidade contra aderencia a pose")
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--prefix", default="combo")
    a = ap.parse_args()

    api = json.load(open(a.base, encoding="utf-8"))
    for n in (ICLORA_UNION, GUIDE_POSE, CFG_GUIDER):
        if n not in api:
            sys.exit(f"no {n} ausente em {a.base} -- base inesperada")

    # --- referencia para dentro do input/ do ComfyUI (LoadImage le por nome) --
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    ref = Path(a.reference)
    ref_nome = f"{a.prefix}_{ref.name}"
    shutil.copy2(ref, COMFY_INPUT / ref_nome)

    # --- 2a IC-LoRA, encadeada DEPOIS da union -------------------------------
    api[ICLORA_ING] = {
        "class_type": "LTXICLoRALoaderModelOnly",
        "inputs": {"lora_name": ING_LORA,
                   "strength_model": a.ingredients_strength,
                   "model": [ICLORA_UNION, 0]},
    }

    # --- caminho da imagem de referencia, repetida por frame -----------------
    api[LOAD_REF] = {"class_type": "LoadImage", "inputs": {"image": ref_nome}}
    # O conversor generico mapeia o DynamicCombo errado: `scale_method` so
    # aceita nearest-exact/bilinear/area/bicubic/lanczos, e o MODO de resize
    # vai em `resize_type.*`. Mesmo conserto que os dois patches existentes ja
    # fazem nos seus proprios resizes.
    api[RESIZE_REF] = {"class_type": "ResizeImageMaskNode",
                       "inputs": {"input": [LOAD_REF, 0],
                                  "resize_type": "scale dimensions",
                                  "resize_type.width": a.width,
                                  "resize_type.height": a.height,
                                  "resize_type.crop": "center",
                                  "scale_method": "lanczos"}}
    api[LEN_REF] = {"class_type": "PrimitiveInt", "inputs": {"value": a.frames}}
    api[REPEAT_REF] = {"class_type": "RepeatImageBatch",
                       "inputs": {"amount": [LEN_REF, 0], "image": [RESIZE_REF, 0]}}

    # --- 2o guia: personagem, consumindo o condicionamento do guia de pose ---
    g = api[GUIDE_POSE]["inputs"]
    api[GUIDE_CHAR] = {
        "class_type": "LTXAddVideoICLoRAGuide",
        "inputs": {
            "frame_idx": 0,
            "strength": a.char_guide_strength,
            "latent_downscale_factor": [ICLORA_ING, 1],
            "crop": "disabled",
            "use_tiled_encode": g.get("use_tiled_encode", False),
            "tile_size": g.get("tile_size", 256),
            "tile_overlap": g.get("tile_overlap", 64),
            # encadeamento: o que sai do guia de POSE entra aqui
            "positive": [GUIDE_POSE, 0],
            "negative": [GUIDE_POSE, 1],
            "latent": [GUIDE_POSE, 2],
            "vae": g.get("vae", [CKPT, 2]),
            "image": [REPEAT_REF, 0],
        },
    }

    # --- religa quem consumia o guia de pose para o guia de personagem -------
    religados = 0
    for nid, n in api.items():
        if nid in (GUIDE_CHAR, GUIDE_POSE):
            continue
        for campo, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and v and str(v[0]) == GUIDE_POSE:
                n["inputs"][campo] = [GUIDE_CHAR, v[1]]
                religados += 1
    # o modelo tambem passa a vir da 2a IC-LoRA
    for campo, v in api[CFG_GUIDER]["inputs"].items():
        if isinstance(v, list) and v and str(v[0]) == ICLORA_UNION and v[1] == 0:
            api[CFG_GUIDER]["inputs"][campo] = [ICLORA_ING, 0]
            religados += 1

    json.dump(api, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"[combo] union+ingredients: {len(api)} nos, {religados} religacoes")
    print(f"[combo] referencia={ref_nome} forca_guia={a.char_guide_strength} -> {a.out}")
    print("[combo] LEMBRE de medir com choreo/verify.py -- encadear guia ja deu ruido antes")


if __name__ == "__main__":
    main()
