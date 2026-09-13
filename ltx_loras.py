"""Catalogo dos LoRAs de VIDEO do LTX (2.3/2.5): o que existe, onde fica, como se usa.

Por que um catalogo em codigo e nao so arquivos numa pasta: um LoRA de video nao
e so um peso. Ele tem TIPO (LoRA comum, IC-LoRA guiado por imagem, por video de
controle, por trilhas), forca recomendada, gatilho de prompt e, no IC-LoRA, fator
de reducao da referencia -- e errar qualquer um desses nao da erro, da video pior
em silencio. O backend (ltx25_backend) e a decupagem leem daqui em vez de cada um
repetir numeros tirados de model card.

Compatibilidade 2.3 -> 2.5 (VERIFICADO 2026-09-12, tres fontes independentes):
  - cabecalho dos safetensors: os transformers 2.3 e 2.5 tem os MESMOS 1772 pesos
    lineares, com os mesmos shapes, em 48 blocos. So muda o
    `text_embedding_projection`, que no 2.5 foi para dentro do encoder Gemma4.
    LoRA treinado no 2.3 carrega no 2.5 sem chave faltando;
  - os workflows OFICIAIS 2.5 vendorizados (ComfyUI-LTXVideo/example_workflows/2.5)
    carregam IC-LoRAs 2.3: union-control, motion-track, ingredients, deblur,
    in-outpainting;
  - model card do Lightricks/LTX-2.5: a grande maioria dos LoRAs/IC-LoRAs do 2.3
    roda no 2.5 sem mudanca, "com excecoes -- valide antes de producao".
Carregar sem erro nao prova que o efeito se mantem: `validado_25` diz, entrada por
entrada, se isso ja foi VISTO numa geracao nesta maquina.

So baixamos .safetensors e workflows .json (dados). Nada de codigo de terceiros:
os LoRAs que exigem custom node proprio ficam marcados em `requires` e nao sao
ligados pelo backend ate alguem decidir instalar e revisar esse node.

Uso:
  python ltx_loras.py list
  python ltx_loras.py status [--compat]
  python ltx_loras.py download --set core
  python ltx_loras.py download better-human-motion vbvr-i2v
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
# ComfyUI le as duas pastas como categoria "loras" (extra_model_paths.yaml, secoes
# ltx_optimized e ltx_25) e lista os arquivos pelo NOME -- por isso os nomes locais
# sao unicos entre as duas, e as pastas ficam planas (subpasta vira "sub\\arq" no
# Windows, a mesma armadilha registrada em ltx25_backend.GGUF_VARIANTS).
LORA_DIRS = {"2.3": MODELS / "loras", "2.5": MODELS / "2.5" / "loras"}
STAGING = MODELS / "_hf_staging"
TRANSFORMER_25 = MODELS / "2.5" / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"

# Tipos. O backend decide o grafo pelo tipo, entao ele e contrato, nao rotulo.
KINDS = {
    "lora": "LoRA comum (LoraLoaderModelOnly): muda o modelo inteiro, sem guia",
    "ic_image": "IC-LoRA guiado por UMA imagem repetida em todos os quadros (folha de referencia)",
    "ic_refs": "IC-LoRA guiado por sequencia curta de referencias (MSR: sujeitos + cenario)",
    "ic_video": "IC-LoRA guiado por video de controle (canny/depth/pose/movimento de camera)",
    "ic_tracks": "IC-LoRA guiado por trilhas de pontos desenhadas (LTXVDrawTracks)",
    "v2v": "IC-LoRA de pos-producao: o proprio video vira a guia (upscale, deblur, relight, dublagem)",
    "plugin": "exige custom node que NAO esta instalado -- o backend recusa",
    "audio_ref": "exige tokens de audio de referencia (ID-LoRA) -- nao ligado na decupagem",
}


@dataclass(frozen=True)
class LoraSpec:
    key: str
    repo: str                    # repositorio HF ("" = veio de fora do HF)
    remote: str                  # arquivo dentro do repositorio
    local: str                   # nome local; e por ele que o ComfyUI lista
    base: str                    # base de TREINO: "2.3" | "2.5" (define a pasta)
    kind: str
    strength: float
    sets: tuple = ()
    gated: bool = False          # licenca precisa ser aceita no site, pela conta
    trigger: str = ""
    trigger_pos: str = ""        # "start" | "end"
    extras: tuple = ()           # workflows/configs do mesmo repo (so .json)
    requires: str = ""           # o que falta para usar de verdade
    validado_25: bool = False
    nota: str = ""


SPECS = (
    # --- ja instalados antes deste catalogo ------------------------------------
    LoraSpec("ingredients-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
             "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
             "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", "2.3", "ic_image", 1.0,
             sets=("instalado",), gated=True,
             nota="Folha de referencia (personagens, objetos, locacao em paineis sobre fundo preto, "
                  "sem texto) -> video. Prompt em duas partes: 'Reference sheet: ...' e "
                  "'Generated video: ...'. Treino 768x448, 121 quadros. E o que o workflow "
                  "oficial 2.5 de Ingredients carrega."),
    LoraSpec("union-control-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
             "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
             "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "2.3", "ic_video", 1.0,
             sets=("instalado",),
             nota="Canny + depth + pose de um video de referencia; referencia a meia resolucao "
                  "(fator 2 no metadata). Workflow oficial 2.5 usa este arquivo."),
    LoraSpec("multiref-storyboard-v2", "", "",
             "ltx_2.3_Multi-Ref Character Storyboard_V2.safetensors", "2.3", "lora", 1.0,
             sets=("instalado",),
             nota="Veio de um workflow RunningHub (ai-toolkit). Sem documentacao de uso; "
                  "MEMORIAL 3.51 -- carrega e gera, efeito em consistencia nao medido."),

    # --- nucleo pedido (sem licenca a aceitar) ----------------------------------
    LoraSpec("msr-2.5", "LiconStudio/LTX-2.5-Multiple-Subject-Reference",
             "LTX-2.5-Licon-MSR-V1.safetensors", "ltx-2.5-licon-msr-v1.safetensors",
             "2.5", "plugin", 1.0, sets=("core",),
             extras=("LTX2.5-MSR-sample-workflow.json",),
             requires="custom node ComfyUI-LTX2.5-MSR (github liconstudio) -- traz "
                      "reference_slot_embedding que o LoraLoader nativo ignora",
             nota="Ate 5 referencias (personagens, roupa, objeto, fundo) viram tokens no mesmo "
                  "espaco do video, cada uma com slot proprio. Nomeie 'Image 1', 'Image 2' no prompt."),
    LoraSpec("msr-2.3-v2", "LiconStudio/LTX-2.3-Multiple-Subject-Reference",
             "LTX-2.3-Licon-MSR-V2.safetensors", "ltx-2.3-licon-msr-v2.safetensors",
             "2.3", "ic_refs", 1.0, sets=("core",),
             extras=("LTX-2.3_MSR_sample_workflow_V2.json",),
             nota="V2 = menos deriva de identidade que a V1. 2 a 5 referencias; 50 fps para acao. "
                  "O node do autor (LiconMSR) so monta a sequencia de imagens -- reescrita em "
                  "script_pipeline/ic_references.py, sem instalar o plugin. Referencia fator 1."),
    LoraSpec("id-lora-talkvid", "AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K",
             "lora_weights.safetensors", "ltx-2.3-id-lora-talkvid-3k.safetensors",
             "2.3", "audio_ref", 1.0, sets=("core",), extras=("config.json",),
             requires="audio de referencia como tokens (LTXVSetAudioRefTokens) e o modelo "
                      "GERANDO a fala -- na decupagem a fala e do TTS e fica congelada",
             nota="So mexe no ramo de AUDIO (audio_attn, audio_ff, audio_to_video_attn): "
                  "identidade de VOZ, nao de rosto. Prompt '[VISUAL]: ... [SPEECH]: ... [SOUNDS]: ...'."),
    LoraSpec("id-lora-celebvhq", "AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K",
             "lora_weights.safetensors", "ltx-2.3-id-lora-celebvhq-3k.safetensors",
             "2.3", "audio_ref", 1.0, sets=("core",), extras=("config.json",),
             requires="mesmo caso do id-lora-talkvid",
             nota="Mesma arquitetura do TalkVid, treinado em CelebV-HQ."),
    LoraSpec("talking-head-av", "elix3r/LTX-2.3-22b-AV-LoRA-talking-head",
             "LTX-2.3-22b-AV-LoRA-talking-head-v1.safetensors",
             "ltx-2.3-22b-av-lora-talking-head-v1.safetensors", "2.3", "lora", 1.0,
             sets=("core",), trigger="OHWXPERSON", trigger_pos="start",
             extras=("Workflows/LTX-2-3-I2V-Custom-Audio.json", "CAPTIONS.json"),
             nota="ATENCAO: e LoRA de UM personagem especifico (voz e rosto do dataset do autor). "
                  "Nao generaliza para o nosso elenco; o valor e a receita de treino (Fish S2 Pro + "
                  "ltx-trainer), que pede ~77 GB de VRAM."),
    LoraSpec("better-human-motion", "vpakarinen/better-human-motion-ltx-lora",
             "motion_stabilizer_ltx_lora_v2.safetensors", "ltx-2.3-better-human-motion-v2.safetensors",
             "2.3", "lora", 0.6, sets=("core",),
             nota="Movimento corporal humano mais suave. Card: forca 0.4-0.8, 720x1280."),
    LoraSpec("motion-enhancer-n4w", "rzgar/LTX-2.3-Motion-Enhancer-n4w",
             "LTX-2.3-Motion-Enhancer-n4w.safetensors", "ltx-2.3-motion-enhancer-n4w.safetensors",
             "2.3", "lora", 0.65, sets=("core",),
             nota="ATENCAO: 'n4w'/'N54W' no vocabulario do autor e NSFW -- e um intensificador de "
                  "movimento geral afinado para esse tipo de conteudo. Forca 0.65 empilhado, 0.75-1.0 "
                  "sozinho. Nunca ligado por padrao."),
    LoraSpec("motion-track", "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
             "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
             "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
             "2.3", "ic_tracks", 1.0, sets=("core",),
             nota="Trilhas de pontos esparsas desenhadas sobre o primeiro quadro guiam o movimento. "
                  "Referencia a meia resolucao (fator 2): NAO combina com audio_conditioning. "
                  "Workflow oficial 2.5 usa este arquivo, com I2V a 0.7."),
    LoraSpec("cameraman-v2", "Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2",
             "LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors", "ltx-2.3-ic-lora-cameraman-v2.safetensors",
             "2.3", "ic_video", 1.0, sets=("core",),
             nota="Replica o movimento de camera de um video de referencia. Sem gatilho. Card: nao "
                  "descer de 960x512; forca da imagem inicial 0.5-0.7 da mais movimento."),
    LoraSpec("cameraman-v1", "Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1",
             "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors", "ltx-2.3-ic-lora-cameraman-v1.safetensors",
             "2.3", "ic_video", 1.0, sets=("core",),
             nota="Versao pedida; a v2 do mesmo autor tem dataset maior (343 pares)."),
    LoraSpec("cdrama-canny", "SyFeee/ltx2.3-chinese-drama-iclora-canny",
             "lora_weights_step_06000.safetensors", "ltx-2.3-cdrama-iclora-canny-6000.safetensors",
             "2.3", "ic_video", 1.0, sets=("core",),
             nota="Canny de um video de referencia (OpenCV 100/200, blur 1) na resolucao de saida. "
                  "Composicao e silhueta, SEM identidade. Guia 1.0 fiel; 0.5-0.7 mais livre. Treinado "
                  "com CFG 4 e 20 passos -- no destilado (CFG 1) esta fora da distribuicao."),
    LoraSpec("cdrama-char", "SyFeee/ltx2.3-chinese-drama-charlora",
             "lora_weights_step_03000.safetensors", "ltx-2.3-cdrama-charlora-3000.safetensors",
             "2.3", "lora", 0.9, sets=("core",),
             nota="Estilo de drama historico chines (78 episodios). Gatilhos char_0_person, "
                  "char_1_person... sao os ATORES da serie -- usar gatilho puxa aquele rosto e briga "
                  "com a folha de personagem. Pilha validada pelo autor: char 0.9 + canny 1.0."),
    LoraSpec("transition", "joyfox/LTX-2.3-Transition-LORA",
             "ltx2.3-transition.safetensors", "ltx-2.3-transition-joyfox.safetensors",
             "2.3", "lora", 1.0, sets=("core",), trigger="zhuanchang", trigger_pos="end",
             extras=("LTX-2.3.json",),
             nota="Transicao/transformacao primeiro->ultimo quadro. Card recomenda CFG 4 (dev)."),
    LoraSpec("vbvr-i2v", "LiconStudio/Ltx2.3-VBVR-lora-I2V",
             "Ltx2.3-Licon-VBVR-I2V-390K-R32.safetensors", "ltx-2.3-vbvr-i2v-390k.safetensors",
             "2.3", "lora", 0.8, sets=("core",),
             nota="Raciocinio de video (VBVR, 390K clipes): prompt complexo, fisica, estabilidade de "
                  "enquadramento. Forca nao documentada no card; 0.8 e ponto de partida, nao medida."),

    # --- oficiais com licenca a aceitar (gated=auto) ------------------------------
    LoraSpec("ingredients-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
             "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
             "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors", "2.5", "ic_image", 1.0,
             sets=("gated",), gated=True,
             nota="Mesmo uso do ingredients-2.3, retreinado no 2.5 (publicado 2026-09-10)."),
    LoraSpec("cinemagraph-2.5", "Lightricks/LTX-2.5-22b-LoRA-Cinemagraph",
             "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors",
             "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors", "2.5", "lora", 1.1,
             sets=("gated",), gated=True, trigger="CINEMAGRAPH_MOTION", trigger_pos="start",
             nota="Movimento localizado com o resto congelado; diga no prompt o que se move e o que "
                  "fica parado. Fora de escopo: camera em movimento e corpo humano em movimento -- "
                  "serve a insert/estabelecimento (vela, chuva, lanterna), nao a fala."),
    LoraSpec("pixel-upscaler-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
             "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
             "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", "2.5", "v2v", 1.0,
             sets=("gated",), gated=True,
             nota="Upscale generativo x2 do clipe aprovado (referencia a meia resolucao). Mantem "
                  "composicao, movimento e identidade; nao mudar corte/aspecto entre fonte e saida."),
    LoraSpec("deblur-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Deblur",
             "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors",
             "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors", "2.5", "v2v", 1.0,
             sets=("gated",), gated=True, nota="Recupera nitidez de take bom com desfoque."),
    LoraSpec("clean-plate-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate",
             "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors",
             "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors", "2.5", "v2v", 1.0,
             sets=("gated",), gated=True, nota="Reconstroi o fundo sem o elemento (clean plate)."),
    LoraSpec("relight-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-Relight",
             "ltx-2.3-22b-ic-lora-relight-1.0.safetensors",
             "ltx-2.3-22b-ic-lora-relight-1.0.safetensors", "2.3", "v2v", 1.0,
             sets=("gated",), gated=True, extras=("LTX-2.3_Relight_ICLoRA_SingleStage_Distilled.json",),
             requires="esfera de luz renderizada no canto do video (node Sphere-Light-Render, nao instalado)",
             nota="So EXTERIOR. Direcao do sol constante no clipe. Treino 1280x704, 121 quadros."),
    LoraSpec("dubit-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
             "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
             "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors", "2.3", "v2v", 1.0,
             sets=("gated",), gated=True,
             nota="Dublagem/lip-sync: video + audio novo -> labios ressincronizados. Candidato a "
                  "substituir o Wav2Lip/LatentSync da etapa [6]. Workflow oficial 2.3 vendorizado "
                  "(LTX-2.3_ICLoRA_DubIt_Two_Stage_Distilled.json)."),

    # --- oficiais com licenca, prioridade baixa para drama -------------------------
    LoraSpec("day-to-night-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night",
             "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors",
             "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors", "2.5", "v2v", 1.0,
             sets=("gated-extra",), gated=True),
    LoraSpec("decompression-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Decompression",
             "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors",
             "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors", "2.5", "v2v", 1.0,
             sets=("gated-extra",), gated=True),
    LoraSpec("colorization-2.5", "Lightricks/LTX-2.5-22b-IC-LoRA-Colorization",
             "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors",
             "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors", "2.5", "v2v", 1.0,
             sets=("gated-extra",), gated=True),
    LoraSpec("in-outpainting-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
             "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
             "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors", "2.3", "v2v", 1.0,
             sets=("gated-extra",), gated=True),
    LoraSpec("hdr-2.3", "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
             "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
             "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors", "2.3", "v2v", 1.0,
             sets=("gated-extra",), gated=True, extras=("ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",)),
)
BY_KEY = {s.key: s for s in SPECS}
BY_LOCAL = {s.local: s for s in SPECS}


def installed_path(local_name: str) -> Path | None:
    for d in LORA_DIRS.values():
        p = d / local_name
        if p.exists():
            return p
    return None


def resolve(name: str) -> tuple[LoraSpec | None, str]:
    """Aceita a CHAVE do catalogo ('better-human-motion') ou um nome/caminho de
    arquivo. Devolve (spec ou None, nome como o ComfyUI lista)."""
    if name in BY_KEY:
        return BY_KEY[name], BY_KEY[name].local
    base = os.path.basename(name)
    return BY_LOCAL.get(base), base


def parse_lora_arg(value: str, default_strength: float | None = None) -> tuple[str, float]:
    """'chave[:forca]' ou 'arquivo[:forca]'. So corta no ULTIMO ':' quando o que vem
    depois e numero -- 'E:\\x.safetensors' nao pode virar ('E', ...)."""
    nome, forca = value, None
    if ":" in value:
        cabeca, cauda = value.rsplit(":", 1)
        try:
            forca = float(cauda)
            nome = cabeca
        except ValueError:
            pass
    spec, local = resolve(nome)
    if forca is None:
        forca = default_strength if default_strength is not None else (spec.strength if spec else 1.0)
    return local, float(forca)


def apply_triggers(prompt: str, local_names) -> str:
    """Poe os gatilhos que os LoRAs escolhidos exigem, uma vez so. Explicito de
    proposito: um LoRA com gatilho e sem ele no prompt e o caso 'carregou e nao fez
    nada' que nao aparece em log nenhum."""
    specs = [BY_LOCAL[n] for n in local_names if n in BY_LOCAL]
    inicio = [s.trigger for s in specs if s.trigger and s.trigger_pos == "start" and s.trigger not in prompt]
    fim = [s.trigger for s in specs if s.trigger and s.trigger_pos == "end" and s.trigger not in prompt]
    out = prompt
    if inicio:
        out = ", ".join(inicio) + ", " + out
    if fim:
        out = out.rstrip().rstrip(".") + ", " + ", ".join(fim)
    return out


# --- compatibilidade contra o transformer 2.5 (so cabecalho, sem carregar peso) ---

def _header(path: Path) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return (h.pop("__metadata__", None) or {}), h


def reference_downscale(local_name: str) -> float:
    """`reference_downscale_factor` do metadata -- o mesmo valor que o
    LTXICLoRALoaderModelOnly le na execucao, lido AQUI antes de montar o grafo para
    recusar combinacoes que falhariam em silencio (ver ltx25_backend.build_workflow).
    Sem a chave, 1.0 -- o mesmo padrao do no."""
    p = installed_path(os.path.basename(local_name))
    if p is None:
        return 1.0
    try:
        return float(_header(p)[0].get("reference_downscale_factor", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


def _norm(k: str) -> str:
    for p in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if k.startswith(p):
            return k[len(p):]
    return k


_BASE_CACHE: dict = {}


def _base_linear_shapes() -> dict:
    if "shapes" not in _BASE_CACHE:
        _, h = _header(TRANSFORMER_25)
        _BASE_CACHE["shapes"] = {_norm(k): v["shape"] for k, v in h.items()
                                 if k.endswith(".weight") and len(v["shape"]) == 2}
    return _BASE_CACHE["shapes"]


def compat_25(path: Path) -> dict:
    """Cada par lora_A/lora_B tem de cair num peso linear que EXISTE no 2.5 com o
    shape certo. Chaves que nao sao LoRA (ex.: reference_slot_embedding do MSR 2.5)
    vao para `extra` -- e isso que denuncia um LoRA que precisa de node proprio."""
    meta, h = _header(path)
    base = _base_linear_shapes()
    ok = faltando = shape_ruim = 0
    extra = set()
    for k, v in h.items():
        m = re.match(r"(.*)\.(lora_A|lora_down)\.weight$", k)
        if not m:
            if not re.search(r"\.(lora_B|lora_up)\.weight$|\.alpha$", k):
                extra.add(re.sub(r"\.\d+(\.|$)", ".#\\1", k))
            continue
        alvo = _norm(m.group(1)) + ".weight"
        if alvo not in base:
            faltando += 1
            continue
        up = h.get(k.replace("lora_A", "lora_B").replace("lora_down", "lora_up"))
        out_f, in_f = base[alvo]
        if v["shape"][1] != in_f or (up is not None and up["shape"][0] != out_f):
            shape_ruim += 1
        else:
            ok += 1
    return {"alvos_ok": ok, "faltando_no_25": faltando, "shape_errado": shape_ruim,
            "extra": sorted(extra)[:8],
            "reference_downscale_factor": meta.get("reference_downscale_factor")}


# --- download -----------------------------------------------------------------------

def _remote_size(api, spec: LoraSpec) -> int | None:
    try:
        info = api.get_paths_info(spec.repo, [spec.remote])
        return info[0].size if info else None
    except Exception:
        return None


def download(keys, *, log=print) -> dict:
    """Baixa para uma pasta de preparo no MESMO volume e so entao move para o nome
    final: um download interrompido nunca deixa um .safetensors truncado com o nome
    que o ComfyUI lista (ele listaria e quebraria na carga, longe daqui)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    api = HfApi()
    rel = {}
    for key in keys:
        spec = BY_KEY[key]
        if not spec.repo:
            rel[key] = {"status": "sem_repositorio"}
            continue
        dest = LORA_DIRS[spec.base] / spec.local
        dest.parent.mkdir(parents=True, exist_ok=True)
        tamanho = _remote_size(api, spec)
        if dest.exists() and tamanho and dest.stat().st_size == tamanho:
            log(f"[loras] {key}: ja instalado ({tamanho/1e9:.2f} GB) -> {dest.name}")
            rel[key] = {"status": "ja_instalado", "arquivo": str(dest)}
            continue
        preparo = STAGING / spec.repo.replace("/", "__")
        log(f"[loras] {key}: baixando {spec.repo}/{spec.remote}"
            f"{f' ({tamanho/1e9:.2f} GB)' if tamanho else ''}...")
        try:
            p = hf_hub_download(spec.repo, spec.remote, local_dir=str(preparo))
        except GatedRepoError:
            log(f"[loras] {key}: LICENCA PENDENTE -- aceite em https://huggingface.co/{spec.repo}")
            rel[key] = {"status": "licenca_pendente", "url": f"https://huggingface.co/{spec.repo}"}
            continue
        except Exception as e:
            log(f"[loras] {key}: FALHOU {type(e).__name__}: {str(e)[:200]}")
            rel[key] = {"status": "falhou", "erro": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        os.replace(p, dest)
        for extra in spec.extras:
            try:
                pe = hf_hub_download(spec.repo, extra, local_dir=str(preparo))
                alvo = dest.parent / "_workflows" / key / os.path.basename(extra)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pe, alvo)
            except Exception as e:
                log(f"[loras] {key}: extra {extra} falhou ({type(e).__name__}); seguindo")
        shutil.rmtree(preparo, ignore_errors=True)
        c = compat_25(dest)
        log(f"[loras] {key}: OK -> {dest}  | compat 2.5: {c['alvos_ok']} alvos ok, "
            f"{c['faltando_no_25']} faltando, {c['shape_errado']} shape errado"
            f"{', extra=' + str(c['extra']) if c['extra'] else ''}")
        rel[key] = {"status": "baixado", "arquivo": str(dest), "compat_25": c}
    return rel


def keys_for_set(nome: str) -> list[str]:
    if nome == "all":
        return [s.key for s in SPECS if s.repo]
    return [s.key for s in SPECS if nome in s.sets]


def video_loras_available(kinds=("lora",)) -> list[LoraSpec]:
    """Entradas instaladas de um tipo -- o que uma UI pode oferecer sem mentir."""
    return [s for s in SPECS if s.kind in kinds and installed_path(s.local)]


def status(compat: bool = False) -> list[dict]:
    linhas = []
    for s in SPECS:
        p = installed_path(s.local)
        linha = {"key": s.key, "kind": s.kind, "base": s.base, "gated": s.gated,
                 "instalado": bool(p), "gb": round(p.stat().st_size / 1e9, 2) if p else None,
                 "validado_25": s.validado_25, "requires": s.requires}
        if compat and p:
            linha["compat_25"] = compat_25(p)
        linhas.append(linha)
    return linhas


def main() -> int:
    ap = argparse.ArgumentParser(description="Catalogo de LoRAs de video do LTX")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    st = sub.add_parser("status")
    st.add_argument("--compat", action="store_true", help="confere as chaves contra o transformer 2.5")
    dl = sub.add_parser("download")
    dl.add_argument("keys", nargs="*")
    dl.add_argument("--set", dest="conjunto", choices=["core", "gated", "gated-extra", "all"])
    dl.add_argument("--report", default=str(ROOT / "logs" / "lora_download_report.json"))
    args = ap.parse_args()

    if args.cmd == "list":
        for s in SPECS:
            print(f"{s.key:<24} {s.kind:<10} base {s.base}  forca {s.strength:<4} "
                  f"{'[licenca] ' if s.gated else ''}{s.trigger and 'gatilho=' + s.trigger}")
            if s.requires:
                print(f"{'':<24} requer: {s.requires}")
        return 0
    if args.cmd == "status":
        print(json.dumps(status(args.compat), ensure_ascii=False, indent=1))
        return 0
    keys = list(args.keys) + (keys_for_set(args.conjunto) if args.conjunto else [])
    desconhecidas = [k for k in keys if k not in BY_KEY]
    if desconhecidas or not keys:
        print(f"chaves desconhecidas: {desconhecidas}; conhecidas: {sorted(BY_KEY)}")
        return 2
    rel = download(dict.fromkeys(keys))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[loras] relatorio: {args.report}")
    return 0 if all(r["status"] in ("baixado", "ja_instalado") for r in rel.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
