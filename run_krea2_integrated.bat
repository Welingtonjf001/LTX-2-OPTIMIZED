@echo off
title Krea2 Integrated Launcher
setlocal
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"

echo [1/3] Starting Backend (ComfyUI)...
:: Start ComfyUI in a new window with a specific title for later termination
start "COMFYUI_KREA_BACKEND" cmd /c "start_comfyui_ltx.bat"

echo [2/3] Waiting for Backend to initialize...
:wait_loop
timeout /t 5 > nul
curl -fsS http://127.0.0.1:8188/system_stats > nul
if %errorlevel% equ 0 (
    echo ✅ Backend is UP and responding!
    goto start_ui
)
echo ⏳ Backend still loading... please wait...
goto wait_loop

:start_ui
echo [3/3] Starting Krea2 WebUI...
echo --------------------------------------------------
echo The WebUI is now running. 
echo When you close this window, the ComfyUI backend will be terminated.
echo --------------------------------------------------
".venv\Scripts\python.exe" krea2_test_ui.py

echo.
echo Closing Backend...
:: Kill the ComfyUI process by window title
taskkill /F /T /FI "WINDOWTITLE eq COMFYUI_KREA_BACKEND*"
echo.
echo System shut down successfully.
pause
