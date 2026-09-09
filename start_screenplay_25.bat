@echo off
setlocal
title LTX-2.5 Screenplay to Video
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter LTX-2 directory.
  pause
  exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "PYTHONUNBUFFERED=1"
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
REM
REM   gguf-q6k  = GGUF Q6_K via ComfyUI-GGUF, 8 passos/CFG 1 igual distilled.
REM               MEDIDO 2026-09-06: 37 porcento mais rapido que distilled na
REM               mesma cena (265s contra 797s numa cena de dialogo pesada) --
REM               `audio_conditioning` CONFIRMADO compativel (testado de
REM               proposito, e o que sustenta o lip-sync da decupagem). Ainda
REM               NAO testado: variante dev, upscale de 2 estagios, keyframes
REM               -- nao trocar se o fluxo usar algum desses.

REM OBRIGATORIO para o estagio de video: sem isto o ComfyUI liga dynamic VRAM
REM por padrao e engasga encenando o encoder de ~25 GB. MEDIDO: 13 min presos
REM em "prepared for dynamic VRAM loading" sem a flag, contra 151s com ela
REM (mesmo clipe). Ver CLAUDE.md, secao LTX-2.5, e MEMORIAL.md 3.33.
set "LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram"
REM ---------------------------------------------------------------------


REM This is the whole 2.5 switch for the screenplay chain. The chain is
REM screenplay_ui.py -> screenplay_to_video.py -> script_pipeline/render_scenes.py,
REM and only the last one actually spawns the renderer. render_scenes.py reads
REM this variable to pick the module (default: the 2.3 native pipeline), so the
REM 2.5 route reuses the entire chain instead of duplicating ~1900 lines of
REM screenplay/casting/lipsync logic that has nothing to do with the model version.
set "LTX_PIPELINE_MODULE=ltx_pipelines_25"

REM Only the RENDER stage runs on 2.5. The other stages of this pipeline
REM (storyboard via Flux, TTS, lip-sync, mixing) are unchanged and still use
REM their own models.

echo Checking port 7910...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7910" ^| findstr "LISTENING"') do (
  echo Stopping old LTX-2.5 Screenplay process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo LTX-2.5 Screenplay to Video: http://127.0.0.1:7910
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7910"
"E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv\Scripts\python.exe" -u screenplay_ui.py --port 7910

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7910" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
