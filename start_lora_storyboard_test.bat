@echo off
setlocal
title Teste de LoRA - LTX 2.3 via ComfyUI
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

REM Sobe (ou reaproveita) o ComfyUI compartilhado na 3090 -- mesma convencao
REM do ltx25_backend/storyboard. Nao defina CUDA_VISIBLE_DEVICES aqui: quem
REM decide isso e o processo que efetivamente sobe o servidor.
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"

echo Checking port 7915...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7915" ^| findstr "LISTENING"') do (
  echo Stopping old process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Teste de LoRA (LTX 2.3 / ComfyUI): http://127.0.0.1:7915
".venv\Scripts\python.exe" -u lora_storyboard_test_ui.py --port 7915

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7915" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
