@echo off
setlocal
title LTX-2 Music Video Maker

cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
    echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"
rem RTX 3090 profile: keep complete transformer blocks on GPU while preserving
rem headroom for high-resolution activations, the VAE and spatial upsampler.
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
rem This process also loads ctranslate2 (faster-whisper, used for
rem lyrics), which bundles its own libiomp5md.dll alongside torch's own copy.
rem Two competing OpenMP runtimes in the same process is a known cause of
rem sporadic access violations (0xC0000005) deep in torch_cpu.dll under heavy
rem multi-threaded CPU compute (confirmed via Windows Event Viewer during a
rem real crash on v2). KMP_DUPLICATE_LIB_OK works around the duplicate-runtime
rem corruption; pinning thread counts to the 6 physical cores (this machine:
rem i5-10400F, 6C/12T) avoids oversubscription across hyperthreads.
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"

echo Iniciando LTX-2 Music Video Maker na RTX 3090...
echo A interface sera aberta em http://127.0.0.1:7801

start "" cmd /c "timeout /t 6 /nobreak >nul && start "" http://127.0.0.1:7801"
".venv\Scripts\python.exe" -u music_maker_ui.py

rem Release the port in case Gradio (or a subprocess it spawned, e.g. the
rem ComfyUI GGUF backend) left something bound after this window closes.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7801" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)

echo.
echo Interface encerrada. Pressione qualquer tecla para fechar.
pause >nul
endlocal
