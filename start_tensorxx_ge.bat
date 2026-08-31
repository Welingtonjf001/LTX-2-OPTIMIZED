@echo off
setlocal
title TensorXX-GE diagnostics for LTX-2.3
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter the LTX-2 directory.
  pause
  exit /b 1
)

rem Keep both physical CUDA devices visible: cuda:0 = RTX 4070, cuda:1 = RTX 3090.
set CUDA_VISIBLE_DEVICES=0,1
set PYTHONUNBUFFERED=1
rem torch.onnx/TensorRT print non-ASCII progress markers; cp1252 consoles abort on them.
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1

".venv\Scripts\python.exe" -m tensorxx_ge.diagnose
echo.
echo TensorRT is optional. See tensorxx_ge\README.md for install, capture and engine commands.
pause

