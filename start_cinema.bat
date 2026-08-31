@echo off
REM Atalho para subir a CinemaMaker UI (roteiro multi-cena via Gemini + geracao LTX-2) na RTX 3090
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
set CUDA_VISIBLE_DEVICES=1
set HF_HUB_OFFLINE=1
rem Each stage releases its model (del + cleanup_memory()) before the next
rem dispatches -- ltx_pipelines/distilled.py lines 242-357 -- so these never
rem overlap. 1GiB for the upsampler forced near-total CPU offload of that
rem stage: measured 431s for its first step on this same hardware (33x the
rem transformer stage's ~13s/step) while the 3090 sat at 11% utilization.
set LTX_TRANSFORMER_GPU_MEMORY=18GiB
set LTX_TRANSFORMER_CPU_MEMORY=32GiB
set LTX_TEXT_ENCODER_GPU_MEMORY=4GiB
set LTX_UPSAMPLER_GPU_MEMORY=18GiB
rem Gradio tries 7860 first, falls back to 7861 if busy -- release both so a
rem stale instance from a previous session doesn't force the fallback.
for %%P_PORT in (7860 7861) do (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%%P_PORT" ^| findstr "LISTENING"') do (
    echo Stopping old process on port %%P_PORT, PID %%P...
    taskkill /PID %%P /T /F >nul 2>&1
  )
)
timeout /t 2 /nobreak >nul

echo Iniciando LTX-2 CinemaMaker na RTX 3090...
echo Cole sua chave de API do Google Gemini no campo da interface.
echo Abra no navegador o endereco mostrado na linha "Running on local URL" (normalmente http://127.0.0.1:7860; se estiver ocupada, sera 7861).
".venv\Scripts\python.exe" -u film_maker_ui_v4.py

rem Release whichever of the two ports Gradio ended up using.
for %%P_PORT in (7860 7861) do (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%%P_PORT" ^| findstr "LISTENING"') do (
    taskkill /PID %%P /T /F >nul 2>&1
  )
)
pause
