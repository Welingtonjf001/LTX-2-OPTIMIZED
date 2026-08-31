@echo off
title Krea2 Test UI Launcher
cd /d "%~dp0"

echo Starting Krea2 Test UI...
echo Using environment: .venv
echo.

".venv\Scripts\python.exe" krea2_test_ui.py

echo.
echo Application closed.
pause