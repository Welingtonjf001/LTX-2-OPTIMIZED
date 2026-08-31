@echo off
setlocal
title LTX-2 WebUI - Triton/PyTorch optimized
REM Launcher with explicit GPU, memory and compiler controls.
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERROR: could not enter the LTX-2 directory.
  pause
  exit /b 1
)

REM CUDA index 1 is the RTX 3090 24GB; CUDA index 0 is the RTX 4070 12GB.
set "LTX_GPU=1"
set "CUDA_VISIBLE_DEVICES=%LTX_GPU%"
set "CLIP_DEVICE=cuda"

REM Runtime controls. Set LTX_TORCH_COMPILE=1 to opt in for this WebUI.
set "LTX_TORCH_COMPILE=0"
REM expandable_segments is unsupported on this platform -- the log warns about
REM it ("not supported on this platform, Triggered internally...") on every
REM single allocation, and silently falls back to the default allocator
REM anyway. max_split_size_mb:128 is what v2/v3 already use successfully here.
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"
REM Each stage (text encoder / transformer / upsampler) fully releases its
REM model (del + cleanup_memory()) before the next one dispatches -- see
REM ltx_pipelines/distilled.py lines 242-357 -- so these budgets never
REM overlap and each can safely approach the 3090's 24GiB on its own.
REM Previous upsampler budget of 1GiB forced near-total CPU offload of that
REM stage: measured 431s for its first denoising step (33x slower than the
REM 14GiB transformer stage's ~13s/step) while the 3090 sat at 11%
REM utilization -- PCIe weight-swapping, not compute. Raised accordingly.
set "LTX_TRANSFORMER_GPU_MEMORY=18GiB"
set "LTX_TRANSFORMER_CPU_MEMORY=32GiB"
set "LTX_TEXT_ENCODER_GPU_MEMORY=4GiB"
set "LTX_UPSAMPLER_GPU_MEMORY=18GiB"

REM Short cache paths reduce Windows path-length and stale-compile issues.
set "TORCHINDUCTOR_CACHE_DIR=E:\ltx_torchinductor_cache"
set "TRITON_CACHE_DIR=E:\ltx_triton_cache"
if not exist "%TORCHINDUCTOR_CACHE_DIR%" mkdir "%TORCHINDUCTOR_CACHE_DIR%"
if not exist "%TRITON_CACHE_DIR%" mkdir "%TRITON_CACHE_DIR%"

REM Release only the process listening on the WebUI port.
echo Checking port 7860...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
  echo Stopping old WebUI process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Verifying PyTorch/Triton on GPU %LTX_GPU%...
".venv\Scripts\python.exe" -c "import torch,triton; print('torch='+torch.__version__+' triton='+triton.__version__+' cuda='+str(torch.cuda.is_available()))"
if errorlevel 1 (
  echo ERROR: PyTorch/Triton verification failed.
  pause
  exit /b 1
)

echo Iniciando LTX-2 WebUI na GPU lógica %LTX_GPU%...
echo Torch.compile: %LTX_TORCH_COMPILE%
echo O navegador sera aberto automaticamente em http://127.0.0.1:7860
start "" cmd /c "timeout /t 6 /nobreak >nul && start "" http://127.0.0.1:7860"
".venv\Scripts\python.exe" -u web_ui_v4.py

rem Release the port in case Gradio left something bound after this window
rem closes.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
