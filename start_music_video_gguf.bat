@echo off
setlocal
title LTX-2 Music Video Maker GGUF - ComfyUI Backend
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter LTX-2 directory.
  pause
  exit /b 1
)
rem CUDA index 1 is the RTX 3090 24GB in this installation. gguf_backend.py
rem inherits this to launch ComfyUI on the same GPU.
set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "LTX_UPSCALE_GPU=1"
rem Same defensive threading fix validated on the fp8 path (2026-07-24): avoids
rem sporadic torch_cpu.dll access violations from competing OpenMP runtimes
rem under heavy CPU compute. Cheap to keep here too since the ComfyUI backend
rem subprocess inherits this environment. i5-10400F: 6 physical cores.
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"

rem Release only the process listening on the GGUF UI port, if an old instance exists.
echo Checking port 7805...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7805" ^| findstr "LISTENING"') do (
  echo Stopping old Music Video GGUF process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Music Video GGUF: http://127.0.0.1:7805
echo (First scene of the session starts ComfyUI + loads the model: ~1min. Every scene after that reuses it.)
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7805"
"E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv\Scripts\python.exe" -u music_maker_ui_gguf.py

rem Release the UI port. gguf_backend.py registers its own atexit cleanup for
rem the ComfyUI subprocess it spawns (port 8188); this only covers this UI's
rem own port.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7805" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
