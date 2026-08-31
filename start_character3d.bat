@echo off
set CHAR3D_MODEL_ROOT=G:\models
set CHAR3D_PORT=7865
set COMFYUI_ROOT=E:\Users\home\Documents\LTX-2-OPTIMIZED\ComfyUI
rem Para o PyTorch local, CUDA 1 corresponde à RTX 3090 de 24 GB.
set CUDA_VISIBLE_DEVICES=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start "Character 3D Lab" cmd /k python character3d_webui.py
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; 1..60 | %% { try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7865/config -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 2 } }; if($ok){ Start-Process http://127.0.0.1:7865 } else { Write-Host 'WebUI nao respondeu.' }"
