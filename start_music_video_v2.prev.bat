@echo off
setlocal
title LTX-2 Music Video Maker V2 - Photos + CLIP
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
if errorlevel 1 (
  echo Could not enter LTX-2 directory.
  pause
  exit /b 1
)
rem Use RTX 4070 for CLIP analysis; keep RTX 3090 available for LTX.
set "CUDA_VISIBLE_DEVICES=0"
set "CLIP_DEVICE=cuda"
set "HF_HUB_OFFLINE=0"
set "LTX_FFMPEG=C:/ffmpeg/bin/ffmpeg.exe"
echo Music Video V2: http://127.0.0.1:7803
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:7803"
"E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv\Scripts\python.exe" -u music_maker_ui_v2.py
pause
