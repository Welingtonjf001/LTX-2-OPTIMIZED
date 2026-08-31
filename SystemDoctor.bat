@echo off
setlocal

set "SYSTEM_DOCTOR_HOME=G:\hermes\system-doctor"
set "SYSTEM_DOCTOR_DB=G:\hermes\system-doctor\system_events.sqlite3"
set "OLLAMA_MODELS=G:\ollama\models"
set "PY=C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"

echo.
echo ==========================================
echo   Hermes SystemDoctor
echo ==========================================
echo.
echo MCP server:
echo   G:\hermes\system-doctor\server.py
echo.
echo SQLite history:
echo   %SYSTEM_DOCTOR_DB%
echo.

echo Running one monitor snapshot...
"%PY%" "G:\hermes\system-doctor\monitor.py" --once

echo.
echo Testing direct system summary...
"%PY%" -c "import sys,json; sys.path.insert(0, r'G:\hermes\system-doctor'); import server; print(json.dumps(server.get_system_summary(False), indent=2, ensure_ascii=False))"

echo.
echo To test MCP inside Hermes:
echo   C:\Users\user\AppData\Local\hermes\hermes-agent\bin\hermes.exe mcp test system-doctor
echo.
pause
endlocal
