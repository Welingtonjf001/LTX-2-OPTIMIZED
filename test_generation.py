import requests
import time
import os

COMFYUI_API_URL = "http://127.0.0.1:8188"

def test_gen():
    # Corrected Workflow matching the FINAL adjusted lauch
    workflow = {
        "3": {
            "inputs": {
                "seed": 42,
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
                "clip_name": "clip_l.safetensors",
                "type": "flux"
            },
            "class_type": "CLIPLoader"
        },
        "12": {
            "inputs": {
                "text": "A professional studio photo of a futuristic cyberpunk city, 8k resolution, highly detailed",
                "clip": ["11", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "17": {
            "inputs": {
                "text": "text, watermark, logo, bad quality",
                "clip": ["11", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "16": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
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
                "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "15": {
            "inputs": {
                "filename_prefix": "Krea2_Final_Test",
                "images": ["13", 0]
            },
            "class_type": "SaveImage"
        }
    }

    p = {"prompt": workflow}
    try:
        print(f"Sending request to {COMFYUI_API_URL}...")
        response = requests.post(f"{COMFYUI_API_URL}/prompt", json=p, timeout=10)
        if response.status_code == 200:
            prompt_id = response.json().get("prompt_id")
            print(f"✅ Success! Prompt ID: {prompt_id}")
            print("The image is being generated in the backend.")
            return prompt_id
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_gen()
