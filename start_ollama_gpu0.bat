@echo off
REM Fase 1 do plano de otimizacao de GPU (MEMORIAL.md SS3.143, 2026-09-30):
REM pina o servidor Ollama na GPU 0 (a segunda RTX 3090), deixando a GPU 1
REM livre e exclusiva para FLUX/LTX/MiniMax H3/LongCat -- elimina a disputa
REM de VRAM que hoje obriga run_decupagem a descarregar o Ollama
REM (ollama_runtime.unload_all()) antes de subir o motor de imagem/video.
REM
REM Nao mexe em nenhuma variavel de ambiente GLOBAL do Windows -- so as
REM define para ESTE processo (ollama app.exe) e seus filhos (ollama.exe
REM serve). Os subprocessos que este repositorio ja sobe (ltx25_backend.py,
REM minimax_h3_backend.py, etc.) continuam definindo CUDA_VISIBLE_DEVICES=1
REM explicitamente no proprio ambiente que passam pro subprocess.Popen, entao
REM nao herdam nada daqui -- os dois convivem sem conflito.
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_MODELS=W:\ollama\models
start "" "C:\Users\user\AppData\Local\Programs\Ollama\ollama app.exe"
