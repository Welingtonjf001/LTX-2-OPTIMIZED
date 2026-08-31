@echo off
setlocal
title Character Sheet FLUX WebUI
set "ROOT=E:\Users\home\Documents\LTX-2-OPTIMIZED"
set "PORT=7866"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do taskkill /PID %%P /T /F >nul 2>&1
rem 8192 e a porta dedicada do backend ComfyUI desta WebUI.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8192" ^| findstr "LISTENING"') do taskkill /PID %%P /T /F >nul 2>&1
cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\python.exe" -u "%ROOT%\character_sheet_flux_webui.py"
endlocal
