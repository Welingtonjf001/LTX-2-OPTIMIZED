@echo off
setlocal
title LTX-2 Screenplay to Video
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter LTX-2 directory.
  pause
  exit /b 1
)
set "HF_HUB_OFFLINE=1"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

REM OBRIGATORIO: esta rota sobe o ComfyUI para o estagio de storyboard (FLUX).
REM Sem isto ele liga dynamic VRAM por padrao e engasga encenando o encoder --
REM MEDIDO 2026-08-29: 4 tentativas de 600s presas em "Model Initializing...",
REM 40 min perdidos, GPU a 24 GB/100% o tempo todo. Ver MEMORIAL.md 3.44 e 3.33.
set "LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram"
echo Checking port 7810...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7810" ^| findstr "LISTENING"') do (
  echo Stopping old Screenplay UI process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo Screenplay to Video: http://127.0.0.1:7810
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7810"
"E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv\Scripts\python.exe" -u screenplay_ui.py

rem Release the port in case Gradio left something bound after this window
rem closes.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7810" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
