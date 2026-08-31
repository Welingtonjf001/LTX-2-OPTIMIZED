@echo off
setlocal
title TensorXX-GE Studio - TensorRT + LTX (fila, imagens, LoRA, lip-sync)
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED" || exit /b 1

rem Keep both physical GPUs visible: CUDA 0 = RTX 4070 (TensorRT), CUDA 1 = RTX 3090 (LTX).
set "CUDA_VISIBLE_DEVICES=0,1"
set "PYTHONUNBUFFERED=1"
rem torch.onnx/TensorRT print non-ASCII progress markers; cp1252 consoles abort on them.
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "LTX_TRANSFORMER_GPU_MEMORY=14GiB"
set "LTX_TRANSFORMER_CPU_MEMORY=32GiB"
set "LTX_TEXT_ENCODER_GPU_MEMORY=2GiB"
set "LTX_UPSAMPLER_GPU_MEMORY=1GiB"

echo Verificando TensorRT e as duas GPUs...
".venv\Scripts\python.exe" -m tensorxx_ge.diagnose --require-tensorrt || (
  pause
  exit /b 1
)
echo TensorXX-GE Studio: http://127.0.0.1:7808
".venv\Scripts\python.exe" -u -m tensorxx_ge.webui_studio --port 7808
pause
