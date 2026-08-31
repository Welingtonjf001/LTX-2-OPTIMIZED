@echo off
setlocal enabledelayedexpansion
title LTX-2.5 Decupagem - roteiro para cena decupada
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo ERRO: nao foi possivel acessar o diretorio do LTX-2.
  pause
  exit /b 1
)

REM Diferente dos outros start_*.bat: este NAO sobe uma UI numa porta. O
REM run_decupagem e uma CLI que roda a cadeia inteira e termina. Arraste o
REM arquivo do roteiro sobre este .bat, ou rode e informe o caminho.

set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"

REM ---------------------------------------------------------------------
REM ESTAGIO DE VIDEO: as duas linhas abaixo NAO sao ajuste fino, sao o que
REM faz o clipe fechar nesta placa. Sem elas o estagio de video encalha a
REM ~24 GB de 24,5 e nao progride -- medido em quatro configuracoes
REM diferentes (MEMORIAL.md secao 3.33).
REM
REM   --disable-dynamic-vram : o ComfyUI liga "dynamic VRAM" por padrao e ele
REM       engasga ao encenar o encoder de 25 GB. O mesmo plano de 81 frames
REM       nao fechava em 13 min e passa a fechar em 151 s. Com ele, um plano
REM       de 337 frames fecha em 394 s.
REM   distilled (bf16) : NAO troque por distilled-int8 aqui. MEDIDO 2026-08-28:
REM       o int8 tem 20 GB e CABE na placa, entao o ComfyUI o carrega inteiro
REM       ("full load: True") e nao sobra VRAM para o latente -- um plano de 129
REM       frames travou duas vezes. O bf16 tem 39 GB, nao cabe, e o gerenciador
REM       e obrigado a descarregar 19 GB ("loaded partially") -- o mesmo plano
REM       fecha em 1087 s. Caber inteiro e o problema. Ver MEMORIAL 3.35.
REM
REM Ate o estagio `animatic` nada disto e usado (os stills sao FLUX), entao
REM deixar ligado nao custa nada.
set "LTX25_VARIANT=distilled"
set "LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram"
REM ---------------------------------------------------------------------

REM NAO defina CUDA_VISIBLE_DEVICES aqui. Esta cadeia chama gemma4_worker.py em
REM alguns caminhos, que faz torch.cuda.set_device(1) esperando as DUAS placas
REM visiveis. Quem precisa da 3090 se resolve sozinho: ltx25_backend e
REM generate_storyboards definem a variavel no subprocesso que eles iniciam.

REM ---------------------------------------------------------------------
REM ESTILO DA DECUPAGEM:
REM   classico       cobertura convencional, camera discreta
REM   tenso          quadro travado, corte seco, planos curtos
REM   nervoso        handheld, planos muito curtos
REM   intimista      close e olhar dominam, luz quente, camera quase parada
REM   contemplativo  planos longos e abertos, movimento lento
set "ESTILO=classico"

REM TROCA DE ESTILO (opcional). A marca vigora dali PARA A FRENTE, ate a
REM proxima marca -- como troca de bobina.
REM   por CENA   : 1:nervoso,3:contemplativo,5:tenso
REM   por TOMADA : 1.10:intimista   (da tomada 10 da cena 1 em diante, e segue
REM                atravessando o fim da cena)
REM O numero da tomada e o que aparece na tabela do plano de decupagem.
set "TROCAS="

REM ATE ONDE IR:
REM   animatic = para no RASCUNHO. Stills na duracao planejada com as vozes,
REM              montado em segundos e SEM GPU de difusao. E o ponto de revisao
REM              barato: tudo ja foi decidido, nada foi renderizado em video.
REM   final    = vai ate o filme montado (leva ~1h para uma cena de 1 min).
set "ATE=animatic"

REM RESOLUCAO. 960x544 e o padrao do workflow oficial 2.5. A 768x512 com plano
REM aberto um rosto ocupa 1,6 celula latente e derrete; ver MEMORIAL secao 3.17.
set "LARGURA=960"
set "ALTURA=544"

REM Reescolher a VOZ de cada personagem pelo descritor (alem da emocao por
REM fala, que e sempre feita). Deixe vazio para manter as vozes ja atribuidas.
set "RECAST="
REM ---------------------------------------------------------------------

REM O motor de LLM da estrutura e da emocao. Precisa do Ollama NO AR com a
REM pasta certa: `ollama serve` com OLLAMA_MODELS=G:\ollama\models, ou o app
REM desktop encerrado antes. Sem ele a cadeia SEGUE, mas a estrutura sai so com
REM a camada deterministica e a emocao cai no neutro. Ver CLAUDE.md.
set "MOTOR=qwen3.6-35b-a3b:latest"

set "ROTEIRO=%~1"
if "%ROTEIRO%"=="" (
  echo.
  echo Arraste o arquivo do roteiro sobre este .bat, ou informe o caminho:
  set /p ROTEIRO="Roteiro (.txt): "
)
if not exist "%ROTEIRO%" (
  echo ERRO: roteiro nao encontrado: %ROTEIRO%
  pause
  exit /b 1
)

for %%F in ("%ROTEIRO%") do set "NOME=%%~nF"
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do set "STAMP=%%c%%b%%a_%%d%%e"
set "STAMP=%STAMP: =0%"
set "RUN=outputs\decupagem\%STAMP%_%NOME%"

echo.
echo ============================================================
echo  Roteiro : %ROTEIRO%
echo  Run     : %RUN%
echo  Estilo  : %ESTILO%   Trocas: %TROCAS%
echo  Ate     : %ATE%      Resolucao: %LARGURA%x%ALTURA%
echo ============================================================
echo.

set "EXTRA="
if not "%TROCAS%"=="" set "EXTRA=%EXTRA% --style-changes "%TROCAS%""
if not "%RECAST%"=="" set "EXTRA=%EXTRA% --recast"

".venv\Scripts\python.exe" -u -m script_pipeline.run_decupagem ^
  --run-dir "%RUN%" --script "%ROTEIRO%" ^
  --style "%ESTILO%" --ate "%ATE%" ^
  --width %LARGURA% --height %ALTURA% --engine "%MOTOR%" %EXTRA%

echo.
if "%ATE%"=="animatic" (
  echo Rascunho em: %RUN%\shots\animatic.mp4
  echo Revise e rode de novo com ATE=final para gerar o video.
) else (
  echo Filme em: %RUN%\final\
)
echo.
pause
