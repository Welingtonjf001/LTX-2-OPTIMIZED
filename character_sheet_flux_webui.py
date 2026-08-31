import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import requests


ROOT = Path(r"E:\Users\home\Documents\LTX-2-OPTIMIZED")
COMFY = ROOT / "ComfyUI"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = ROOT / "models" / "flux-2-klein-9b-fp8.safetensors"
BACKEND_URL = "http://127.0.0.1:8192"
WEBUI_PORT = 7866
INPUT_DIR = COMFY / "input"
OUTPUT_DIR = COMFY / "output"
TEMPLATE = Path(r"E:\Users\home\Documents\MiniMax-H3\venv\Lib\site-packages\comfyui_workflow_templates_json\templates\image_flux2_klein_image_edit_9b_base.json")
BACKEND = None


def start_backend():
    global BACKEND
    try:
        requests.get(BACKEND_URL, timeout=1)
        return
    except Exception:
        pass
    args = [
        str(PYTHON), str(COMFY / "main.py"),
        "--listen", "127.0.0.1", "--port", "8192", "--cuda-device", "1",
        # A RTX 3090 tem memoria suficiente para manter o FLUX e o VAE na GPU.
        # Evitar lowvram/cpu-vae reduz bastante o tempo de cada geração.
        "--disable-auto-launch",
        "--extra-model-paths-config", str(COMFY / "extra_model_paths.yaml"),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    BACKEND = subprocess.Popen(args, cwd=COMFY, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            requests.get(BACKEND_URL, timeout=1)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("O backend ComfyUI não respondeu na porta 8192.")


def stop_backend():
    global BACKEND
    if BACKEND and BACKEND.poll() is None:
        BACKEND.terminate()
        try:
            BACKEND.wait(timeout=8)
        except subprocess.TimeoutExpired:
            BACKEND.kill()
    BACKEND = None


atexit.register(stop_backend)


def upload_image(path: str) -> str:
    source = Path(path)
    target = INPUT_DIR / f"character_sheet_{uuid.uuid4().hex}{source.suffix.lower()}"
    shutil.copy2(source, target)
    with target.open("rb") as handle:
        response = requests.post(f"{BACKEND_URL}/upload/image", files={"image": (target.name, handle, "application/octet-stream")}, data={"overwrite": "true"}, timeout=120)
    response.raise_for_status()
    return response.json().get("name", target.name)


def add_node(prompt, node_id, class_type, inputs):
    prompt[str(node_id)] = {"class_type": class_type, "inputs": inputs}


def build_prompt(refs, prompt_text, negative, width, height, steps, seed, filename_prefix):
    prompt = {}
    add_node(prompt, 1, "UNETLoader", {"unet_name": MODEL.name, "weight_dtype": "default"})
    add_node(prompt, 2, "KSamplerSelect", {"sampler_name": "euler"})
    add_node(prompt, 3, "Flux2Scheduler", {"steps": int(steps), "width": int(width), "height": int(height)})
    add_node(prompt, 4, "RandomNoise", {"noise_seed": int(seed)})
    add_node(prompt, 7, "CLIPLoader", {"clip_name": "Qwen3-8B-FP8-native-bf16.safetensors", "type": "flux2", "device": "default"})
    add_node(prompt, 8, "VAELoader", {"vae_name": "flux2-vae.safetensors"})
    add_node(prompt, 9, "EmptyFlux2LatentImage", {"width": int(width), "height": int(height), "batch_size": 1})
    add_node(prompt, 12, "CLIPTextEncode", {"clip": [7, 0], "text": prompt_text})
    add_node(prompt, 13, "CLIPTextEncode", {"clip": [7, 0], "text": negative})
    add_node(prompt, 5, "CFGGuider", {"model": [1, 0], "positive": [12, 0], "negative": [13, 0], "cfg": 5.0})
    add_node(prompt, 6, "SamplerCustomAdvanced", {"noise": [4, 0], "guider": [5, 0], "sampler": [2, 0], "sigmas": [3, 0], "latent_image": [9, 0]})

    last_conditioning = [12, 0]
    for index, ref in enumerate(refs[:3]):
        image_node = 20 + index * 4
        scale_node = image_node + 1
        encode_node = image_node + 2
        reference_node = image_node + 3
        name = upload_image(ref)
        add_node(prompt, image_node, "LoadImage", {"image": name})
        add_node(prompt, scale_node, "ImageScaleToTotalPixels", {"image": [image_node, 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1})
        add_node(prompt, encode_node, "VAEEncode", {"pixels": [scale_node, 0], "vae": [8, 0]})
        add_node(prompt, reference_node, "ReferenceLatent", {"conditioning": last_conditioning, "latent": [encode_node, 0]})
        last_conditioning = [reference_node, 0]

    add_node(prompt, 10, "VAEDecode", {"samples": [6, 0], "vae": [8, 0]})
    add_node(prompt, 11, "SaveImage", {"images": [10, 0], "filename_prefix": filename_prefix})
    prompt["5"]["inputs"]["positive"] = last_conditioning
    for node in prompt.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], int):
                node["inputs"][key] = [str(value[0]), value[1]]
    return prompt


def wait_for_result(prompt_id):
    for _ in range(1800):
        response = requests.get(f"{BACKEND_URL}/history/{prompt_id}", timeout=10)
        if response.ok:
            data = response.json().get(prompt_id)
            if data:
                if data.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(json.dumps(data, ensure_ascii=False))
                outputs = data.get("outputs", {})
                for node_output in outputs.values():
                    for item in node_output.get("images", []):
                        filename = item["filename"]
                        subfolder = item.get("subfolder", "")
                        output_type = item.get("type", "output")
                        url = f"{BACKEND_URL}/view?filename={requests.utils.quote(filename)}&subfolder={requests.utils.quote(subfolder)}&type={output_type}"
                        image_path = OUTPUT_DIR / "character_sheet" / Path(filename).name
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(requests.get(url, timeout=120).content)
                        return str(image_path)
        time.sleep(1)
    raise TimeoutError("A geração ultrapassou o limite de espera de 30 minutos.")


def generate(ref1, ref2, ref3, prompt_text, negative, width, height, steps, seed):
    refs = [x for x in (ref1, ref2, ref3) if x]
    if not MODEL.exists():
        raise FileNotFoundError(f"Modelo FLUX não encontrado em {MODEL}")
    if not refs:
        raise gr.Error("Adicione pelo menos uma imagem de referência.")
    start_backend()
    workflow = build_prompt(refs, prompt_text, negative, width, height, steps, seed, "character_sheet/character_sheet")
    response = requests.post(f"{BACKEND_URL}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=120)
    if not response.ok:
        raise RuntimeError(response.text)
    prompt_id = response.json()["prompt_id"]
    return wait_for_result(prompt_id)


with gr.Blocks(title="Character Sheet FLUX") as demo:
    gr.Markdown("# Character Sheet FLUX\nGere somente pranchas de personagem para referência 3D. Use de 1 a 3 imagens para preservar identidade, roupa e proporções.")
    with gr.Row():
        ref1 = gr.Image(type="filepath", label="Referência 1", sources=["upload"])
        ref2 = gr.Image(type="filepath", label="Referência 2", sources=["upload"])
        ref3 = gr.Image(type="filepath", label="Referência 3", sources=["upload"])
    prompt = gr.Textbox(label="Descrição da personagem", value="adult Korean female fashion model, consistent identity, full-body character sheet, front view, left profile, right profile, back view, neutral A-pose, same outfit and hairstyle in every view, clean studio background, orthographic reference sheet, realistic 3D character design, complete hands and feet")
    negative = gr.Textbox(label="Negative prompt", value="inconsistent face, different outfit, extra limbs, missing fingers, cropped feet, duplicate person, text, logo, watermark, perspective distortion")
    with gr.Row():
        width = gr.Slider(512, 1536, value=768, step=64, label="Largura")
        height = gr.Slider(512, 1536, value=1024, step=64, label="Altura")
        steps = gr.Slider(4, 30, value=8, step=1, label="Passos")
        seed = gr.Number(value=42, precision=0, label="Seed")
    button = gr.Button("Gerar character sheet", variant="primary")
    result = gr.Image(label="Resultado", type="filepath")
    button.click(generate, [ref1, ref2, ref3, prompt, negative, width, height, steps, seed], result)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=WEBUI_PORT, inbrowser=True, show_error=True)
