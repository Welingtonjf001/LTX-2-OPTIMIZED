@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\qwen_claude_agent\qwen_claude_web.py"
) else (
  python "tools\qwen_claude_agent\qwen_claude_web.py"
)
