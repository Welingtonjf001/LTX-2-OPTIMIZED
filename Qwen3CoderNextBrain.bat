@echo off
setlocal

set "OLLAMA_MODELS=G:\ollama\models"
set "HERMES_EXE=C:\Users\user\AppData\Local\hermes\hermes-agent\bin\hermes.exe"
set "MODEL=qwen3-coder-next"

echo.
echo ==========================================
echo   Qwen3-Coder-Next Brain Installer
echo ==========================================
echo.
echo Store: %OLLAMA_MODELS%
echo Model: %MODEL%
echo.

echo [1/3] Baixando/retomando modelo...
ollama pull %MODEL%
if errorlevel 1 (
  echo [ERRO] Falha no download do %MODEL%.
  pause
  exit /b 1
)

echo.
echo [2/3] Configurando Hermes para usar %MODEL%...
"%HERMES_EXE%" config set model.default %MODEL%

echo.
echo [3/3] Teste rapido do modelo no Ollama...
ollama run %MODEL% "Responda em portugues, uma frase curta: pronto."

echo.
echo Pronto. Reinicie o dashboard/sessao do Hermes para carregar o novo cerebro.
echo Dashboard: http://127.0.0.1:9119
echo.
pause
endlocal
