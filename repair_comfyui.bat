@echo off
echo [1/3] Cleaning zombie processes...
taskkill /F /IM python.exe > nul 2>&1

echo [2/3] Updating ComfyUI dependencies...
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r ComfyUI/requirements.txt

echo [3/3] Checking LTX-Video requirements...
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade

echo.
echo ✅ System cleaned and updated. Please try running run_krea2_integrated.bat again.
pause
