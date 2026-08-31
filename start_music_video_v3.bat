@echo off
setlocal
title LTX-2 Music Video Maker V3 - LatentSync 1.6
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter LTX-2 directory.
  pause
  exit /b 1
)

rem CUDA index 1 is the RTX 3090 24GB in this installation.
set "CUDA_VISIBLE_DEVICES=1"
set "CLIP_DEVICE=cuda"
set "LTX_TORCH_COMPILE=0"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
rem This process also loads ctranslate2 (faster-whisper, used for lyrics), which
rem bundles its own libiomp5md.dll alongside torch's own copy. Two competing
rem OpenMP runtimes in the same process is a known cause of sporadic access
rem violations (0xC0000005) deep in torch_cpu.dll under heavy multi-threaded
rem CPU compute (confirmed via Windows Event Viewer during a real crash).
rem KMP_DUPLICATE_LIB_OK works around the duplicate-runtime corruption; pinning
rem thread counts to the 6 physical cores (this machine: i5-10400F, 6C/12T)
rem avoids oversubscription across hyperthreads for the same CPU-bound kernels.
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"
rem Each stage releases its model (del + cleanup_memory()) before the next
rem dispatches -- ltx_pipelines/distilled.py lines 242-357 -- so these never
rem overlap. 1GiB for the upsampler forced near-total CPU offload of that
rem stage: measured 431s for its first step on this same hardware (33x the
rem transformer stage's ~13s/step) while the 3090 sat at 11% utilization.
set "LTX_TRANSFORMER_GPU_MEMORY=18GiB"
set "LTX_TRANSFORMER_CPU_MEMORY=32GiB"
set "LTX_TEXT_ENCODER_GPU_MEMORY=4GiB"
set "LTX_UPSAMPLER_GPU_MEMORY=18GiB"
set "LTX_UPSCALE_GPU=1"
rem LatentSync overrides visibility in its subprocess and uses logical device 1 (RTX 3090).
set "LATENTSYNC_CUDA_DEVICE=1"
set "LATENTSYNC_ROOT=E:\Users\home\Documents\LatentSync"
set "LATENTSYNC_PYTHON=E:\Users\home\Documents\LatentSync\.conda_env\python.exe"
set "HF_HUB_OFFLINE=0"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

echo Checking port 7804...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7804" ^| findstr "LISTENING"') do (
  echo Stopping old Music Video V3 process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Music Video V3: http://127.0.0.1:7804
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7804"
"E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv\Scripts\python.exe" -u music_maker_ui_v3.py

rem Release the port in case Gradio (or a subprocess it spawned) left
rem something bound after this window closes.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7804" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
