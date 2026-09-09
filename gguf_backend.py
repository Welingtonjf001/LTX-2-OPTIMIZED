"""GGUF/ComfyUI generation backend for the music-video UIs.

Keeps a single ComfyUI server process alive for the whole session (started on
first use, reused across every scene) so the 22B transformer is loaded ONCE
instead of once per scene like the fp8 subprocess path. Builds the LTX-2.3
"Full" (dev) T2V workflow from the upstream example, patched to run off a
local GGUF checkpoint + the distilled LoRA (few-step generation), and submits
it via the ComfyUI HTTP API.

When a scene provides a conditioning image (the UI's "Start Image" for that
scene, or an auto-chained last-frame from the previous scene), generation
routes through the IC-LoRA "Ingredients" workflow instead of plain T2V: the
image becomes an identity/appearance reference (via LTXAddVideoICLoRAGuide),
not a literal first-frame video condition like the fp8 pipeline's --image.
Validated end-to-end: at ingredients_lora_strength=1.0 with a neutral prompt,
the reference's identity comes through very strongly (near-literal
reproduction of the reference photo) -- lower strengths are exposed as a
tunable but not yet empirically validated by us.

Known gaps in this first version (kept honest rather than silently wrong):
  - Scene-to-scene continuity via image conditioning is repurposed as identity
    reference (see above) rather than true first-frame video conditioning --
    a materially different effect than the fp8 pipeline's --image. Multiple
    conditioning frames for one scene are not supported; only the first is
    used, and a warning is logged if more than one was provided.
  - The input song audio only drives scene duration/frame count (same as the
    fp8 path already does for timing) -- it is not fed into the transformer as
    audio-latent conditioning (LTXVAudioVAEEncode + LTXVSetAudioRefTokens),
    unlike the fp8 pipeline's --audio-input-path. The generated audio track is
    T2A (from scratch), independent of the source song.
"""
import atexit
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import comfy_gguf_patch as gguf_patch
import comfy_ingredients_patch as ing_patch
import comfy_workflow_tool as wf_tool

COMFY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI")
COMFY_SERVER = "http://127.0.0.1:8188"
COMFY_PORT = 8188
FULL_WORKFLOW_PATH = os.path.join(
    COMFY_ROOT, "custom_nodes", "ComfyUI-LTXVideo", "example_workflows", "2.3",
    "LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json",
)
FULL_SAVE_NODE_ID = "4823"  # SaveVideo node feeding the "output_F" (dev/Full) branch
INGREDIENTS_WORKFLOW_PATH = os.path.join(
    COMFY_ROOT, "custom_nodes", "ComfyUI-LTXVideo", "example_workflows", "2.3",
    "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json",
)
INGREDIENTS_SAVE_NODE_ID = "4852"  # SaveVideo node in the Ingredients workflow

# Q6_K over UD-Q5_K_S: same dev weights, less aggressive quantization
# (16.55 GB vs 15.20 GB).  Both fit the 3090's 24 GB, so the extra 1.35 GB buys
# quality at no practical cost.
DEFAULT_GGUF = "ltx-2.3-22b-dev-Q6_K.gguf"
DEFAULT_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DEFAULT_STEPS = 8
DEFAULT_INGREDIENTS_LORA_STRENGTH = ing_patch.DEFAULT_INGREDIENTS_LORA_STRENGTH
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_server_proc = None
_server_log_handle = None
_object_info_cache = None
_base_api_cache = None
_ingredients_base_cache = None


def list_available_gguf() -> list[str]:
    """List .gguf files under models/, for the UI's model dropdown. Falls back
    to just DEFAULT_GGUF if the directory can't be scanned (e.g. UI running
    from a different cwd), so the dropdown never ends up empty."""
    try:
        files = sorted(f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".gguf"))
        return files or [DEFAULT_GGUF]
    except OSError:
        return [DEFAULT_GGUF]


def _http_get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _http_post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def server_is_up() -> bool:
    try:
        urllib.request.urlopen(f"{COMFY_SERVER}/", timeout=3)
        return True
    except Exception:
        return False


def ensure_server(log_cb=None, boot_timeout: int = 120) -> None:
    """Start the ComfyUI server if it isn't already running. Idempotent."""
    global _server_proc, _server_log_handle
    if server_is_up():
        return

    def log(msg):
        if log_cb:
            log_cb(msg)
        print(msg, flush=True)

    log("[gguf] ComfyUI não está no ar; iniciando servidor (isso carrega o modelo, pode levar ~1min)...")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    _server_log_handle = open(os.path.join(logs_dir, "comfyui_gguf.log"), "a", encoding="utf-8", buffering=1)
    try:
        _server_proc = subprocess.Popen(
            [python_exe, "-u", "main.py", "--windows-standalone-build",
             "--extra-model-paths-config", "extra_model_paths.yaml",
             "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--disable-auto-launch",
             # Stale intermediate-node caching across unrelated /prompt submissions
             # (different scene = different prompt/duration/seed every time) has
             # produced a reproducible shape-mismatch crash in the sampler when
             # reusing this long-lived server across scenes (see MEMORIAL.md,
             # LTX-2.5 section). Correctness > the iteration-speed benefit here.
             "--cache-none",
             # MEDIDO 2026-09-04/05 (LTX-2-OPTIMIZED MEMORIAL 3.33/3.58): dynamic
             # VRAM ligado (o padrao do ComfyUI) trava ao encenar um modelo
             # grande com outro ja ocupando a placa -- mesma assinatura em 3
             # backends diferentes deste projeto ate agora. Sem motivo medido
             # pra deixar ligado aqui.
             "--disable-dynamic-vram"],
            cwd=COMFY_ROOT, env=env,
            stdout=_server_log_handle, stderr=subprocess.STDOUT,
        )
    except Exception:
        _server_log_handle.close()
        _server_log_handle = None
        raise
    # Release the port (and the model VRAM) if the parent process exits without
    # calling shutdown_server() explicitly -- e.g. the UI window is closed or
    # the process is Ctrl+C'd. atexit runs on normal interpreter shutdown and
    # on most signal-driven exits; it will not run on SIGKILL/taskkill /F, but
    # covers everything short of that.
    atexit.register(shutdown_server)

    t0 = time.time()
    while time.time() - t0 < boot_timeout:
        if server_is_up():
            log(f"[gguf] ComfyUI pronto em {time.time()-t0:.1f}s.")
            return
        if _server_proc.poll() is not None:
            return_code = _server_proc.returncode
            shutdown_server()
            raise RuntimeError(f"ComfyUI encerrou durante o boot (exit code {return_code}).")
        time.sleep(2)
    shutdown_server()
    raise RuntimeError(f"ComfyUI não respondeu em {boot_timeout}s.")


def shutdown_server() -> None:
    global _server_proc, _server_log_handle
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None
    if _server_log_handle is not None:
        _server_log_handle.close()
        _server_log_handle = None


def _get_object_info() -> dict:
    global _object_info_cache
    if _object_info_cache is None:
        _object_info_cache = wf_tool.fetch_object_info(COMFY_SERVER)
    return _object_info_cache


def _get_base_api_workflow() -> dict:
    """UI workflow -> API format, pruned to the dev/Full T2V branch. Cached
    because the conversion only depends on the (static) example file."""
    global _base_api_cache
    if _base_api_cache is None:
        ui_wf = json.load(open(FULL_WORKFLOW_PATH, encoding="utf-8"))
        api = wf_tool.convert(ui_wf, _get_object_info())
        _base_api_cache = wf_tool.prune_to_output(api, FULL_SAVE_NODE_ID)
    return copy.deepcopy(_base_api_cache)


def _get_ingredients_base_workflow() -> dict:
    """UI workflow -> API format, pruned to the Ingredients branch. Cached
    for the same reason as _get_base_api_workflow()."""
    global _ingredients_base_cache
    if _ingredients_base_cache is None:
        ui_wf = json.load(open(INGREDIENTS_WORKFLOW_PATH, encoding="utf-8"))
        api = wf_tool.convert(ui_wf, _get_object_info())
        _ingredients_base_cache = wf_tool.prune_to_output(api, INGREDIENTS_SAVE_NODE_ID)
    return copy.deepcopy(_ingredients_base_cache)


def build_workflow(
    prompt: str, width: int, height: int, num_frames: int, seed: int,
    steps: int = DEFAULT_STEPS, gguf: str = DEFAULT_GGUF, lora: str = DEFAULT_LORA,
    prefix: str = "scene",
) -> dict:
    api = _get_base_api_workflow()
    return _patch_in_memory(api, prompt, width, height, num_frames, seed, steps, gguf, lora, prefix)


def build_ingredients_workflow(
    prompt: str, reference_image_path: str, width: int, height: int, num_frames: int, seed: int,
    steps: int = DEFAULT_STEPS, gguf: str = DEFAULT_GGUF,
    ingredients_lora_strength: float = DEFAULT_INGREDIENTS_LORA_STRENGTH,
    distilled_lora_strength: float = ing_patch.DEFAULT_DISTILLED_LORA_STRENGTH,
    prefix: str = "scene",
) -> dict:
    """Build the reference-image-conditioned (IC-LoRA "Ingredients") workflow."""
    api = _get_ingredients_base_workflow()
    return _patch_ingredients_in_memory(
        api, prompt, reference_image_path, width, height, num_frames, seed, steps,
        gguf, ingredients_lora_strength, distilled_lora_strength, prefix,
    )


def _patch_in_memory(api, prompt, width, height, num_frames, seed, steps, gguf, lora, prefix):
    P = gguf_patch
    api[P.GGUF_LOADER_ID] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf}}
    api[P.VAE_LOADER_ID] = {"class_type": "VAELoader", "inputs": {"vae_name": r"vae\ltx-2.3-22b-dev_video_vae.safetensors"}}
    api[P.SAMPLER_ID] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}

    if P.LORA_NODE in api:
        api[P.LORA_NODE]["inputs"]["model"] = [P.GGUF_LOADER_ID, 0]
        api[P.LORA_NODE]["inputs"]["lora_name"] = lora
        api[P.LORA_NODE]["inputs"]["strength_model"] = 1.0
    P.rewire(api, P.CKPT_NODE, 0, [P.LORA_NODE, 0])
    P.rewire(api, P.CKPT_NODE, 2, [P.VAE_LOADER_ID, 0])
    P.rewire(api, P.CLOWN_NODE, None, [P.SAMPLER_ID, 0])
    api.pop(P.CKPT_NODE, None)
    api.pop(P.CLOWN_NODE, None)

    # Text-to-video: drop the image-conditioning branch, feed the empty latent
    # straight into the AV concat (see module docstring: i2v not wired yet).
    P.rewire(api, P.IMG_COND_NODE, 0, [P.LATENT_NODE, 0])
    for nid in (P.IMG_COND_NODE, P.PREPROCESS_NODE, P.RESIZE_NODE, P.LOADIMAGE_NODE):
        api.pop(nid, None)
    api = P.prune_orphans(api, FULL_SAVE_NODE_ID)

    if P.AUDIO_VAE_NODE in api:
        api[P.AUDIO_VAE_NODE]["inputs"]["ckpt_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
    if P.TEXT_ENC_NODE in api:
        api[P.TEXT_ENC_NODE]["inputs"]["text_encoder"] = "gemma_3_12B_it.safetensors"
        api[P.TEXT_ENC_NODE]["inputs"]["ckpt_name"] = r"text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors"

    if P.SCHEDULER_NODE in api:
        api[P.SCHEDULER_NODE]["inputs"]["steps"] = steps
    if P.LATENT_NODE in api:
        api[P.LATENT_NODE]["inputs"]["width"] = width
        api[P.LATENT_NODE]["inputs"]["height"] = height
    if P.FRAMES_NODE in api:
        api[P.FRAMES_NODE]["inputs"]["value"] = num_frames
    if P.SAVE_NODE in api:
        api[P.SAVE_NODE]["inputs"]["filename_prefix"] = prefix

    # Prompt text: only the POSITIVE CLIPTextEncode (the one wired into
    # LTXVConditioning.positive) is overwritten. The negative prompt node is
    # left untouched -- replacing both would silently break negative guidance.
    positive_id = None
    for nid, node in api.items():
        if node["class_type"] == "LTXVConditioning":
            pos_ref = node["inputs"].get("positive")
            if isinstance(pos_ref, list):
                positive_id = pos_ref[0]
            break
    if positive_id and positive_id in api and "text" in api[positive_id]["inputs"]:
        api[positive_id]["inputs"]["text"] = prompt
    else:
        raise RuntimeError("não foi possível localizar o nó CLIPTextEncode positivo via LTXVConditioning")

    for node in api.values():
        if node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = int(seed)

    return api


def _patch_ingredients_in_memory(
    api, prompt, reference_image_path, width, height, num_frames, seed, steps,
    gguf, ingredients_lora_strength, distilled_lora_strength, prefix,
):
    P = ing_patch

    # LoadImage reads by filename from ComfyUI/input/, not by absolute path.
    # Prefix with a short random tag so concurrent/successive scenes using
    # different (or same-named) reference photos never collide.
    comfy_input_dir = os.path.join(COMFY_ROOT, "input")
    os.makedirs(comfy_input_dir, exist_ok=True)
    ref_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(reference_image_path)}"
    shutil.copy2(reference_image_path, os.path.join(comfy_input_dir, ref_filename))

    api[P.GGUF_LOADER_ID] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf}}
    api[P.VAE_LOADER_ID] = {"class_type": "VAELoader", "inputs": {"vae_name": P.DEFAULT_VAE}}

    n_model = P.rewire(api, P.CKPT_NODE, 0, [P.GGUF_LOADER_ID, 0])
    if P.DISTILLED_LORA_NODE in api:
        api[P.DISTILLED_LORA_NODE]["inputs"]["lora_name"] = P.DEFAULT_DISTILLED_LORA
        api[P.DISTILLED_LORA_NODE]["inputs"]["strength_model"] = distilled_lora_strength
    if P.ICLORA_NODE in api:
        api[P.ICLORA_NODE]["inputs"]["lora_name"] = P.DEFAULT_INGREDIENTS_LORA
        api[P.ICLORA_NODE]["inputs"]["strength_model"] = ingredients_lora_strength

    n_vae = P.rewire(api, P.CKPT_NODE, 2, [P.VAE_LOADER_ID, 0])
    api.pop(P.CKPT_NODE, None)

    # Fix ResizeImageMaskNode's DynamicCombo widget (see comfy_ingredients_patch.py
    # module docstring: the generic converter mismaps it).
    if P.RESIZE_NODE in api:
        link = api[P.RESIZE_NODE]["inputs"].get("input")
        api[P.RESIZE_NODE]["inputs"] = P.resize_inputs(link, width, height)

    if P.TEXT_ENC_NODE in api:
        api[P.TEXT_ENC_NODE]["inputs"]["text_encoder"] = P.DEFAULT_TEXT_ENCODER
        api[P.TEXT_ENC_NODE]["inputs"]["ckpt_name"] = P.DEFAULT_CONNECTORS
    if P.AUDIO_VAE_NODE in api:
        api[P.AUDIO_VAE_NODE]["inputs"]["ckpt_name"] = P.DEFAULT_AUDIO_CKPT
    if P.LOAD_IMAGE_NODE in api:
        api[P.LOAD_IMAGE_NODE]["inputs"]["image"] = ref_filename
    if num_frames and P.VIDEO_LENGTH_NODE in api:
        api[P.VIDEO_LENGTH_NODE]["inputs"]["value"] = num_frames
    if P.SAVE_NODE in api:
        api[P.SAVE_NODE]["inputs"]["filename_prefix"] = prefix
    if P.POSITIVE_CLIP_NODE in api and "text" in api[P.POSITIVE_CLIP_NODE]["inputs"]:
        api[P.POSITIVE_CLIP_NODE]["inputs"]["text"] = prompt

    for node in api.values():
        if node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = int(seed)

    return api


def _submit_and_wait(workflow: dict, output_path: str, log, stop_flag_getter, timeout: int) -> bool:
    """Submit a built workflow to ComfyUI, poll until done, copy the result to
    *output_path*. Shared by both the plain T2V and the Ingredients paths --
    once submitted, waiting for/retrieving a result is identical either way.
    """
    client_id = str(uuid.uuid4())
    try:
        res = _http_post(f"{COMFY_SERVER}/prompt", {"prompt": workflow, "client_id": client_id})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        log(f"ComfyUI rejeitou o workflow (HTTP {e.code}): {body[:1500]}")
        return False

    prompt_id = res.get("prompt_id")
    log(f"cena enfileirada no ComfyUI (prompt_id={prompt_id})")

    t0 = time.time()
    last_heartbeat = t0
    heartbeat_every = 30  # seconds; even short scenes can take several minutes on the first model load
    # with NO intermediate ComfyUI progress event we log -- without this, a
    # long-running scene looks identical to a stuck/frozen one to the user.
    while time.time() - t0 < timeout:
        if stop_flag_getter is not None and stop_flag_getter():
            try:
                _http_post(f"{COMFY_SERVER}/interrupt", {})
            except Exception:
                pass
            log("geração interrompida pelo usuário.")
            return False

        if time.time() - last_heartbeat >= heartbeat_every:
            last_heartbeat = time.time()
            state = "processando"
            try:
                q = _http_get(f"{COMFY_SERVER}/queue", timeout=10)
                running_ids = {item[1] for item in q.get("queue_running", [])}
                pending_ids = {item[1] for item in q.get("queue_pending", [])}
                if prompt_id in running_ids:
                    state = "gerando (modelo ativo na GPU)"
                elif prompt_id in pending_ids:
                    state = "na fila, aguardando vez"
            except Exception:
                pass
            log(f"ainda {state}... {time.time()-t0:.0f}s decorridos (a primeira cena pode levar vários minutos)")

        hist = _http_get(f"{COMFY_SERVER}/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                for _nid, out in (entry.get("outputs") or {}).items():
                    for _key, items in (out.items() if isinstance(out, dict) else []):
                        for it in items if isinstance(items, list) else []:
                            if isinstance(it, dict) and it.get("filename"):
                                src = os.path.join(COMFY_ROOT, "output", it.get("subfolder", ""), it["filename"])
                                if os.path.exists(src):
                                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                    shutil.copy2(src, output_path)
                                    log(f"cena salva em {output_path} ({time.time()-t0:.1f}s)")
                                    return True
                log("ComfyUI reportou sucesso mas nenhum arquivo de saída foi encontrado.")
                return False
            if status.get("status_str") == "error":
                log(f"ComfyUI reportou erro: {json.dumps(status.get('messages', []), ensure_ascii=False)[:1500]}")
                return False
        time.sleep(3)

    log(f"timeout após {timeout}s aguardando a cena.")
    return False


def generate_scene(
    prompt: str, output_path: str, width: int, height: int, num_frames: int,
    fps: int, steps: int, seed: int, audio_path: str | None = None,
    conditioning_frames: list | None = None, gguf: str = DEFAULT_GGUF,
    ingredients_lora_strength: float = DEFAULT_INGREDIENTS_LORA_STRENGTH,
    log_cb=None, stop_flag_getter=None, timeout: int = 1800,
) -> bool:
    """Generate one scene via the GGUF/ComfyUI backend, writing the result to
    *output_path* (matching the same convention the fp8 subprocess path uses,
    so the rest of the pipeline -- concat, upscale, lipsync -- is untouched).

    If *conditioning_frames* has an entry, routes through the IC-LoRA
    "Ingredients" workflow using its image as an identity/appearance
    reference (see module docstring) instead of plain T2V. Only the first
    entry is used; a warning is logged if more than one was supplied.
    Returns True on success.
    """
    def log(msg):
        if log_cb:
            log_cb(msg)
        print(f"[GGUF] {msg}", flush=True)

    if audio_path:
        log("Nota: o áudio de entrada define apenas a duração da cena (como já ocorre no fp8); "
            "o backend GGUF ainda não usa o áudio real como condicionamento do vídeo.")

    ensure_server(log_cb=log)
    prefix = f"scene_{uuid.uuid4().hex[:8]}"

    if conditioning_frames:
        if len(conditioning_frames) > 1:
            log(f"AVISO: {len(conditioning_frames)} imagens de referência fornecidas; "
                "o backend GGUF só usa a primeira (encadeamento multi-imagem não é suportado).")
        ref_path = conditioning_frames[0][0]
        log(f"usando imagem de referência para identidade via IC-LoRA Ingredients: {ref_path} "
            f"(strength={ingredients_lora_strength}; isto substitui o condicionamento de 1º frame "
            "que o caminho fp8 faz com esta mesma imagem -- efeito diferente, não comparável 1:1).")
        try:
            workflow = build_ingredients_workflow(
                prompt, ref_path, width, height, num_frames, seed, steps=steps, gguf=gguf,
                ingredients_lora_strength=ingredients_lora_strength, prefix=prefix,
            )
        except FileNotFoundError:
            log(f"ERRO: imagem de referência não encontrada em disco: {ref_path}")
            return False
    else:
        workflow = build_workflow(prompt, width, height, num_frames, seed, steps=steps, gguf=gguf, prefix=prefix)

    return _submit_and_wait(workflow, output_path, log, stop_flag_getter, timeout)
