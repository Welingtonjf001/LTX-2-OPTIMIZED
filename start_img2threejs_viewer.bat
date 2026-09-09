@echo off
setlocal
title img2threejs - visualizador e exportador de modelo 3D
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

REM UI puramente local: le imagem + arquivo de modelo/factory, nao toca GPU.
set "PYTHONUNBUFFERED=1"

echo Checking port 7914...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7914" ^| findstr "LISTENING"') do (
  echo Stopping old img2threejs viewer process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo img2threejs viewer: http://127.0.0.1:7914
start "" cmd /c "timeout /t 6 /nobreak >nul && start "" http://127.0.0.1:7914"
".venv\Scripts\python.exe" -u img2threejs_viewer_ui.py --port 7914

REM Libera a porta ao fechar.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7914" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
