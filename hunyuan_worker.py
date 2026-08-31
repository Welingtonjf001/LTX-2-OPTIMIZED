import sys
import os
import gc
from pathlib import Path
from PIL import Image

repo = Path(__file__).resolve().parent / "Hunyuan3D-2"
# Nesta máquina o PyTorch enumera a RTX 3090 (24 GB) como CUDA 1.
# Restrinja o processo a ela antes de qualquer import que carregue torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", r"G:\models\huggingface-cache")
os.environ.setdefault("HUNYUAN_CACHE_DIR", r"G:\models\hunyuan3d")
sys.path.insert(0, str(repo))
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def main(source, output):
    image = Image.open(source).convert("RGBA")
    image.thumbnail((1536, 1536))
    image = BackgroundRemover()(image)
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini"
    )
    shape = shape_pipeline(image=image)[0]
    # A etapa de textura e a de geometria não precisam coexistir na VRAM.
    del shape_pipeline
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    mesh = texture_pipeline(shape, image=image)
    mesh.export(output)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
