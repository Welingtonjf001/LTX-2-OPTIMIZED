@echo off
setlocal
title LTX-2.5 Music Video Maker V2
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERROR: could not enter the LTX-2 directory.
  pause
  exit /b 1
)

REM CUDA index 1 is the RTX 3090 24GB; CUDA index 0 is the RTX 4070 12GB.
set "CUDA_VISIBLE_DEVICES=1"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

REM ---------------------------------------------------------------------
REM ESCOLHA DA VERSAO DO MODELO LTX-2.5 (edite a linha abaixo):
REM   distilled = 8 passos, CFG 1. Rapido (~6 min/clipe de 15s).
REM               ATENCAO: nesta variante o negative prompt NAO tem efeito
REM               (em CFG 1 o ramo negativo se cancela na formula do CFG),
REM               entao legendas queimadas nao sao removiveis por prompt.
REM   dev       = transformer nao destilado, CFG real (video 3 / audio 7),
REM               15 passos. O negative prompt FUNCIONA. Varias vezes mais
REM               lento (~30 avaliacoes do modelo contra 8).
set "LTX25_VARIANT=distilled"

REM OBRIGATORIO para o estagio de video: sem isto o ComfyUI liga dynamic VRAM
REM por padrao e engasga encenando o encoder de ~25 GB. MEDIDO: 13 min presos
REM em "prepared for dynamic VRAM loading" sem a flag, contra 151s com ela
REM (mesmo clipe). Ver CLAUDE.md, secao LTX-2.5, e MEMORIAL.md 3.33.
set "LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram"
REM ---------------------------------------------------------------------


REM NOTE: the LTX_TRANSFORMER_*_MEMORY / LTX_UPSAMPLER_* budgets used by the 2.3
REM launchers are knobs of the NATIVE ltx_pipelines offload path (accelerate
REM max_memory). The 2.5 route generates through ComfyUI, which does its own
REM memory management, so those variables are deliberately NOT set here --
REM setting them would suggest a control this route does not have.

echo Checking port 7903...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7903" ^| findstr "LISTENING"') do (
  echo Stopping old LTX-2.5 Music Video V2 process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo LTX-2.5 Music Video V2: http://127.0.0.1:7903
echo (A primeira geracao sobe o ComfyUI em 8188 e carrega o modelo 2.5: ~1min.)
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7903"
".venv\Scripts\python.exe" -u music_maker_ui_v2_25.py

REM Release this UI's port. The ComfyUI server on 8188 is shut down by
REM ltx25_backend's atexit hook when this process exits normally; it is left
REM alone here on purpose, since it may have been started separately for other work.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7903" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
