@echo off
setlocal
title Video Doctor - diagnostico e correcao temporal (LTX 2.3 e 2.5)
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

REM Esta UI e pos-producao pura: le um mp4 pronto e escreve outro. So encosta
REM na GPU se voce pedir regeneracao parcial (que chama o LTX) ou escolher o
REM fluxo RAFT. Por isso NAO define CUDA_VISIBLE_DEVICES -- quem precisa da
REM 3090 e o ltx25_backend, que ja a seleciona no subprocesso dele.
set "PYTHONUNBUFFERED=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

REM Modelo do RIFE. rife-v4.6 aceita timestep arbitrario (-s), que e o que o
REM reparo usa para preencher vaos de qualquer tamanho. Modelos antigos
REM (rife-v2.x, rife-HD) so fazem o ponto medio.
set "LTX_RIFE_MODEL=rife-v4.6"

REM NOTA: a mesma aba existe dentro do music_maker v2/v3 (2.3 e 2.5). Este
REM launcher serve para examinar um video qualquer sem abrir uma UI de geracao.

echo Checking port 7912...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7912" ^| findstr "LISTENING"') do (
  echo Stopping old Video Doctor process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Video Doctor: http://127.0.0.1:7912
echo (Fluxo: 1. Analisar  ^>  2. REVISAR as tiras  ^>  3. Reparar so o que for defeito)
start "" cmd /c "timeout /t 6 /nobreak >nul && start "" http://127.0.0.1:7912"
".venv\Scripts\python.exe" -u video_doctor_ui.py --port 7912

REM Libera a porta ao fechar.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7912" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
