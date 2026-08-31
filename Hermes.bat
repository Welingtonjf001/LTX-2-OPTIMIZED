@echo off
setlocal

title Hermes Local Agent Launcher

set "OLLAMA_MODELS=G:\ollama\models"
set "HERMES_EXE=C:\Users\user\AppData\Local\hermes\hermes-agent\bin\hermes.exe"
set "OLLAMA_EXE=C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe"
set "SYSTEM_DOCTOR_MONITOR=G:\hermes\system-doctor\start_monitor.bat"
set "DASHBOARD_URL=http://127.0.0.1:9119"
set "CLEANUP_PS=%~dp0HermesCleanup.ps1"
set "HERMES_CONFIG=C:\Users\user\AppData\Local\hermes\config.yaml"

echo.
echo ==========================================
echo   Hermes Local Agent
echo ==========================================
echo.
echo Ollama models: %OLLAMA_MODELS%
echo Hermes:        %HERMES_EXE%
echo Dashboard:     %DASHBOARD_URL%
echo.

if not exist "%HERMES_EXE%" (
  echo [ERRO] Hermes nao encontrado:
  echo %HERMES_EXE%
  pause
  exit /b 1
)

if not exist "%OLLAMA_EXE%" (
  echo [ERRO] Ollama nao encontrado:
  echo %OLLAMA_EXE%
  pause
  exit /b 1
)

rem Watchdog handles clicking X on this console, where cmd cannot run its exit label.
start "Hermes Cleanup Watchdog" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%CLEANUP_PS%" -ParentPid 0

rem Keep one canonical Ollama: the WSL instance backed by G:\ollama\models.
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1

echo [1/5] Verificando Ollama...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; exit 0 } catch { exit 1 }"

if errorlevel 1 (
  echo [2/5] Iniciando Ollama no WSL...
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc "systemctl start ollama"
  timeout /t 8 /nobreak >nul
) else (
  echo [2/5] Ollama ja esta rodando.
)

for /f "tokens=1" %%I in ('wsl.exe -d Ubuntu-22.04 -e bash -lc "hostname -I"') do set "WSL_OLLAMA_IP=%%I"
if defined WSL_OLLAMA_IP (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$p='%HERMES_CONFIG%'; $ip='%WSL_OLLAMA_IP%'; $s=(Get-Content -LiteralPath $p -Raw) -replace 'http://[^/]+:11434/v1', ('http://' + $ip + ':11434/v1'); [IO.File]::WriteAllText($p,$s,(New-Object Text.UTF8Encoding($false)))"
  echo Hermes Ollama endpoint: http://%WSL_OLLAMA_IP%:11434/v1
)

echo [3/5] Verificando modelos locais...
"%OLLAMA_EXE%" list

echo.
echo [4/5] Iniciando SystemDoctor Monitor...
if exist "%SYSTEM_DOCTOR_MONITOR%" (
  start "Hermes SystemDoctor Monitor" "%SYSTEM_DOCTOR_MONITOR%"
) else (
  echo [AVISO] Monitor nao encontrado: %SYSTEM_DOCTOR_MONITOR%
)

echo [5/5] Iniciando Hermes Dashboard...
start "Hermes Dashboard" "%HERMES_EXE%" dashboard --host 127.0.0.1 --port 9119 --no-open

timeout /t 4 /nobreak >nul

echo Abrindo navegador...
start "" "%DASHBOARD_URL%"

echo.
echo Pronto. Esta janela controla a sessao Hermes.
echo Ao fechar esta janela, os modelos, Ollama e o dashboard serao encerrados.
echo.
echo Se quiser usar o chat pelo terminal, rode:
echo "%HERMES_EXE%" --skills persistent-agent-delivery
echo.
echo Para usar MoA no chat do Hermes:
echo /moa sua tarefa aqui
echo.

echo.
echo Pressione uma tecla para encerrar e liberar a memoria...
pause >nul
echo Encerrando modelos e servidores...
powershell -NoProfile -ExecutionPolicy Bypass -File "%CLEANUP_PS%" -ParentPid -1
endlocal
