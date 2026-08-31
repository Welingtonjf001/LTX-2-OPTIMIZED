@echo off
setlocal
title StoryPlay 2.5 - storyboard com quadros intermediarios
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

REM NAO defina CUDA_VISIBLE_DEVICES aqui. Este fluxo chama gemma4_worker.py
REM (parse/enriquecimento), que faz torch.cuda.set_device(1) e device_map={"":1}
REM -- ou seja, espera as DUAS placas visiveis, com a 3090 no indice 1. Fixar
REM CUDA_VISIBLE_DEVICES=1 deixa a 3090 como unica visivel, ela vira indice 0, e
REM o worker morre com "CUDA error: invalid device ordinal" (medido 2026-08-23).
REM Por isso start_screenplay_ui.bat e start_screenplay_25.bat tambem nao setam.
REM Quem precisa da 3090 ja se resolve sozinho: ltx25_backend.ensure_server() e
REM generate_storyboards.ensure_comfyui_running() definem CUDA_VISIBLE_DEVICES=1
REM no ambiente do subprocesso do ComfyUI que eles mesmos iniciam.
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

REM ---------------------------------------------------------------------
REM ESCOLHA DA VERSAO DO MODELO LTX-2.5 (a UI tambem tem esse seletor;
REM esta variavel define o valor inicial):
REM   distilled = 8 passos, CFG 1. Rapido (~13 min para 30s).
REM   dev       = transformer nao destilado, CFG real (video 3 / audio 7),
REM               15 passos. Negative prompt funciona. Varias vezes mais lento.
set "LTX25_VARIANT=distilled"

REM OBRIGATORIO para o estagio de video: sem isto o ComfyUI liga dynamic VRAM
REM por padrao e engasga encenando o encoder de ~25 GB. MEDIDO: 13 min presos
REM em "prepared for dynamic VRAM loading" sem a flag, contra 151s com ela
REM (mesmo clipe). Ver CLAUDE.md, secao LTX-2.5, e MEMORIAL.md 3.33.
set "LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram"
REM ---------------------------------------------------------------------

REM NOTA sobre duracao: legendas queimadas so aparecem em clipes CURTOS --
REM medido: 17s = 11/12 frames com legenda; 30s e 45s = zero. Gere 30s ou mais
REM se o clipe tiver falas. Ver MEMORIAL.md secao 3.11.
REM
REM NOTA sobre quadros: cada quadro intermediario ALONGA o clipe em 8 frames
REM (LTXVAddGuide acrescenta 1 frame latente por guia). 30s com 4 quadros sai
REM ~31,4s. Ver MEMORIAL.md secao 3.12.
REM
REM As demais etapas (storyboard via FLUX, cast, enriquecimento via Gemma4)
REM sobem seus proprios modelos quando necessario. O ComfyUI e iniciado pelo
REM ltx25_backend na primeira geracao (porta 8188) e encerrado junto com esta
REM janela pelo atexit dele.

echo Checking port 7911...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7911" ^| findstr "LISTENING"') do (
  echo Stopping old StoryPlay 2.5 process PID %%P...
  taskkill /PID %%P /T /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo StoryPlay 2.5: http://127.0.0.1:7911
echo (Fluxo: 1. Analisar texto  ^>  2. Gerar quadros  ^>  3. Gerar video)
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7911"
".venv\Scripts\python.exe" -u storyplay25.py --port 7911

REM Libera a porta desta UI. O ComfyUI em 8188 e desligado pelo atexit do
REM ltx25_backend em saida normal; nao e morto aqui de proposito, pois pode ter
REM sido iniciado separadamente para outro trabalho.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7911" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /T /F >nul 2>&1
)
pause
