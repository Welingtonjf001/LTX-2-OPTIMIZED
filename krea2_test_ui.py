import gradio as gr
import torch
import os
import requests
import json
import time
from PIL import Image
import io

# Configurações de Caminhos
COMFYUI_DIR = os.path.abspath("./ComfyUI")
MODELS_DIR = os.path.abspath("./models")
COMFYUI_API_URL = "http://127.0.0.1:8188"

# Verificação de Modelos
REQUIRED_MODELS = {
    "diffusion": "models/flux1-krea-dev_fp8_scaled.safetensors",
    "vae": "models/vae/ae.safetensors",
    "clip": "models/text_encoders/clip_l.safetensors",
    "t5": "models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
}

def check_models():
    missing = []
    for name, path in REQUIRED_MODELS.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    return missing

def check_comfyui_connection():
    try:
        response = requests.get(f"{COMFYUI_API_URL}/system_stats", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_comfyui_workflow():
    # Flux/Krea Workflow - Final Adjusted for API Validation
    return {
        "3": {
            "inputs": {
                "seed": 0,
                "steps": 20,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["10", 0],
                "positive": ["12", 0],
                "negative": ["17", 0],
                "latent_image": ["16", 0]
            },
            "class_type": "KSampler"
        },
        "10": {
            "inputs": {
                "unet_name": "flux1-krea-dev_fp8_scaled.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            },
            "class_type": "UNETLoader"
        },
        "11": {
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux"
            },
            "class_type": "DualCLIPLoader"
        },
        "12": {
            "inputs": {
                "text": "",
                "clip": ["11", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "17": {
            "inputs": {
                "conditioning": ["12", 0]
            },
            "class_type": "ConditioningZeroOut"
        },
        "16": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage"
        },
        "13": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["14", 0]
            },
            "class_type": "VAEDecode"
        },
        "14": {
            "inputs": {
                "vae_name": "vae\\ae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "15": {
            "inputs": {
                "filename_prefix": "Krea2_WebUI",
                "images": ["13", 0]
            },
            "class_type": "SaveImage"
        }
    }

def queue_prompt(prompt_text, width, height, seed):
    workflow = get_comfyui_workflow()
    workflow["12"]["inputs"]["text"] = prompt_text
    workflow["3"]["inputs"]["seed"] = seed
    workflow["16"]["inputs"]["width"] = width
    workflow["16"]["inputs"]["height"] = height

    p = {"prompt": workflow}
    try:
        response = requests.post(f"{COMFYUI_API_URL}/prompt", json=p, timeout=5)
        response.raise_for_status()
        return response.json().get("prompt_id"), None
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        return None, detail


def wait_for_image(prompt_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{COMFYUI_API_URL}/history/{prompt_id}", timeout=10)
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                detail = messages[-1] if messages else "erro desconhecido"
                raise RuntimeError(f"ComfyUI falhou: {detail}")

            for output in history.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    image_info = images[0]
                    image_response = requests.get(
                        f"{COMFYUI_API_URL}/view",
                        params=image_info,
                        timeout=30,
                    )
                    image_response.raise_for_status()
                    image = Image.open(io.BytesIO(image_response.content)).copy()
                    return image, image_info["filename"]
        time.sleep(2)
    raise TimeoutError("Tempo limite excedido aguardando a imagem do ComfyUI.")

def generate_images(prompt, width, height, seed, num_images):
    # 1. Verificação de Modelos
    missing = check_models()
    if missing:
        yield None, f"❌ Erro: Modelos ausentes: {', '.join(missing)}.", 0
        return

    # 2. Verificação de Backend (ComfyUI)
    yield None, "🔍 Verificando conexão com Backend (ComfyUI)...", 5
    time.sleep(1)
    
    if not check_comfyui_connection():
        yield None, "⚠️ Backend ComfyUI não detectado em http://127.0.0.1:8188. \nPor favor, inicie o 'start_comfyui_ltx.bat' primeiro.", 10
        return

    yield None, "✅ Backend ComfyUI conectado!", 15
    time.sleep(1)

    # 3. Simulação de Carregamento (para UX)
    models_to_load = list(REQUIRED_MODELS.keys())
    for i, model in enumerate(models_to_load):
        prog = 15 + int((i / len(models_to_load)) * 30)
        yield None, f"📦 Carregando {model} no VRAM... [{prog}%]", prog
        time.sleep(0.5)

    # 4. Geração Real via API
    last_image = None
    for img_idx in range(1, num_images + 1):
        yield None, f"🎨 Enviando Imagem {img_idx}/{num_images} para o ComfyUI...", 50
        
        prompt_id, error = queue_prompt(prompt, width, height, int(seed) + img_idx - 1)
        
        if prompt_id:
            yield None, f"✨ Gerando imagem {img_idx}/{num_images} no ComfyUI (ID: {prompt_id})...", 60
            try:
                last_image, filename = wait_for_image(prompt_id)
            except (requests.RequestException, RuntimeError, TimeoutError) as exc:
                yield None, f"❌ Falha durante a geração: {exc}", 60
                return
            yield last_image, f"✅ Imagem {img_idx} concluída e salva como {filename}.", 95
        else:
            yield None, f"❌ Falha ao enviar imagem {img_idx} para a API: {error}", 50
            return

    yield last_image, f"🚀 Todas as {num_images} imagens foram concluídas com sucesso!", 100

with gr.Blocks(title="Krea2 Production WebUI") as demo:
    gr.Markdown("# 🎨 Krea2 Production Interface")
    gr.Markdown("Interface conectada ao backend ComfyUI para geração real com Flux.1 Krea Dev.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="A futuristic city in cyberpunk style...", lines=3)
            with gr.Row():
                width = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Largura")
                height = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Altura")
            
            with gr.Row():
                seed = gr.Number(value=42, label="Seed", precision=0)
                num_images = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Quantidade de Imagens")
            
            with gr.Row():
                btn = gr.Button("🚀 Gerar Agora", variant="primary")
                stop_btn = gr.Button("🛑 Parar", variant="stop")
            
        with gr.Column():
            output_img = gr.Image(label="Resultado (Preview)", type="pil")
            status = gr.Textbox(label="Logs do Servidor", interactive=False, lines=8)
            progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progresso de Geração", interactive=False)
            
    btn.click(
        fn=generate_images,
        inputs=[prompt, width, height, seed, num_images],
        outputs=[output_img, status, progress_bar]
    )

if __name__ == "__main__":
    print("Iniciando Krea2 Production UI...")
    demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True)
