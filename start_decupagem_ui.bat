@echo off
setlocal
title Decupagem UI - roteiro para filme (porta 7913)
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

REM NAO defina CUDA_VISIBLE_DEVICES aqui, nem as variaveis do estagio de video:
REM a UI monta o ambiente do subprocesso ela mesma (ver ENV_VIDEO em
REM decupagem_ui.py). Definir aqui so criaria um segundo lugar para divergir.

echo.
echo  Decupagem UI em http://127.0.0.1:7913
echo  Feche esta janela para encerrar.
echo.

".venv\Scripts\python.exe" -u decupagem_ui.py

echo.
pause
