@echo off
setlocal
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED\ComfyUI"
if errorlevel 1 exit /b 1
set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "PYTHONUNBUFFERED=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

rem Release only the process listening on the ComfyUI port, if a stale
rem instance from a previous session is still bound to it.
echo Checking port 8188...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8188" ^| findstr "LISTENING"') do (
  echo Stopping old ComfyUI process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

rem --cache-none: stale intermediate-node caching across unrelated /prompt
rem submissions (different prompt/duration/seed each time) produced a
rem reproducible shape-mismatch crash in the sampler on this server -- see
rem MEMORIAL.md, LTX-2.5 section. Costs some iteration speed on interactive
rem re-runs; worth it for correctness when driving this via the API/scripts.
"..\.venv\Scripts\python.exe" -u main.py --windows-standalone-build --extra-model-paths-config extra_model_paths.yaml --listen 127.0.0.1 --port 8188 --cache-none

rem Belt-and-suspenders: if the server left an orphaned child (or was force-
rem closed) still bound to the port, free it now so the next launch doesn't
rem have to fight a zombie listener.
echo Releasing port 8188...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8188" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
