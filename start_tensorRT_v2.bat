@echo off
setlocal
title LTX-2 Music Video Maker V2 (TensorRT) - Photos + CLIP
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED" || exit /b 1

rem CUDA 0 = RTX 4070 for the TensorRT Gemma engine; CUDA 1 = RTX 3090 for LTX
rem (and for the same-process ASR/Demucs/CLIP/Wav2Lip/upscale stages the full
rem V2 app also runs -- see tensorxx_ge/webui_v2.py).
set "CUDA_VISIBLE_DEVICES=0,1"
set "CLIP_DEVICE=cuda"
set "PYTHONUNBUFFERED=1"
rem torch.onnx/TensorRT print non-ASCII progress markers; cp1252 consoles abort on them.
set "PYTHONIOENCODING=utf-8"
rem ASR (faster-whisper) and CLIP Interrogator (BLIP/CLIP) download their
rem models on first use, same as start_music_video_v2.bat.
set "HF_HUB_OFFLINE=0"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
rem This process also loads ctranslate2 (faster-whisper, used for lyrics), which
rem bundles its own libiomp5md.dll alongside torch's own copy -- see
rem start_music_video_v2.bat for the full explanation of this workaround.
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"
rem Stage 2 (tensorxx_ge.run_ltx_with_tensorrt) forwards these to the native
rem pipeline subprocess; raised from the generic TensorRT defaults (14/32/2/1
rem GiB) to match start_music_video_v2.bat's measured fix -- 1GiB for the
rem upsampler forced near-total CPU offload of that stage (431s/step vs
rem ~13s/step for the transformer stage on this same hardware).
set "LTX_TRANSFORMER_GPU_MEMORY=18GiB"
set "LTX_TRANSFORMER_CPU_MEMORY=32GiB"
set "LTX_TEXT_ENCODER_GPU_MEMORY=4GiB"
set "LTX_UPSAMPLER_GPU_MEMORY=18GiB"
rem Vulkan GPU 1 is the otherwise-idle RTX 4070 in this installation.
set "LTX_UPSCALE_GPU=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

echo Verificando TensorRT e as duas GPUs...
".venv\Scripts\python.exe" -m tensorxx_ge.diagnose --require-tensorrt || (
  pause
  exit /b 1
)
echo Checking port 7806...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7806" ^| findstr "LISTENING"') do (
  echo Stopping old TensorRT V2 process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo Music Video V2 (TensorRT): http://127.0.0.1:7806
".venv\Scripts\python.exe" -u -m tensorxx_ge.webui_v2 --port 7806
pause
