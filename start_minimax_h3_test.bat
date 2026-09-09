@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

REM ---------------------------------------------------------------------
REM CHECKPOINT DO MINIMAX H3 (edite as duas linhas abaixo pra trocar):
REM   FP8 (unet) + INT8 (encoder) = padrao desde 2026-09-06. MEDIDO: mesma
REM     cena/seed, turbo 4 passos -- SEMPRE mais rapido que o w4a8 antigo
REM     (397-666s contra 554-704s), com checkpoints menos comprimidos
REM     (qualidade nominal maior). Arquivos: models/diffusion_models/
REM     minimax_h3_ref2va_pruned_fp8_scaled.safetensors (~21GB) e
REM     models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
REM     (~27GB), baixados de Comfy-Org/MiniMax-H3 no HuggingFace.
REM   Pra voltar ao w4a8 original (13GB unet + 15GB encoder, mais leve em
REM   disco mas mais lento e de qualidade nominal menor), descomente as
REM   duas linhas MINIMAX_H3_* abaixo.
REM
REM set "MINIMAX_H3_UNET=minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors"
REM set "MINIMAX_H3_CLIP=qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors"
REM
REM GGUF Q4_K_M foi medido MAIS LENTO que os dois de cima (707-986s) --
REM nao vale como alternativa aqui, so documentado pra nao redescobrir:
REM set "MINIMAX_H3_UNET=minimax_h3_ref2va-Q4_K_M.gguf"
REM ---------------------------------------------------------------------

echo MiniMax H3 Gradio: http://127.0.0.1:7916
echo Backend ComfyUI esperado: http://127.0.0.1:8189
"%~dp0.venv\Scripts\python.exe" -u "%~dp0minimax_h3_test_ui.py"
pause
