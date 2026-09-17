@echo off
setlocal enabledelayedexpansion
rem BUGFIX auditoria 2026-09-16 (A14): a versao anterior tinha tres problemas:
rem   1. `taskkill /F /IM python.exe` matava TODO Python do usuario, incluindo
rem      qualquer geracao/servidor rodando em outro lugar (MiniMax H3, LongCat,
rem      Fish Speech, etc) -- agora filtra por linha de comando, so processos
rem      cuja CommandLine mencione ComfyUI.
rem   2. Nenhum passo verificava errorlevel -- pip podia falhar silenciosamente
rem      e a mensagem final dizia "concluido" mesmo assim.
rem   3. Forcava torch/torchvision/torchaudio pelo indice cu124, divergindo do
rem      cu128 que o CLAUDE.md documenta para o .venv desta maquina (medido
rem      2026-09-14: torch 2.8.0+cu128) -- rodar isto substituiria a combinacao
rem      de bibliotecas VALIDADA por outra nunca testada aqui.
echo [1/3] Encerrando processos ComfyUI travados (NAO mexe em outros Python)...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'ComfyUI' } | ForEach-Object { Write-Host ('  encerrando PID ' + $_.ProcessId + ': ' + $_.CommandLine.Substring(0, [Math]::Min(120,$_.CommandLine.Length))); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo [2/3] Atualizando dependencias do ComfyUI...
cd /d "E:\Users\home\Documents\LTX-2-OPTIMIZED"
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo FALHOU ao atualizar o pip -- abortando antes de mexer em mais nada.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install -r ComfyUI/requirements.txt
if errorlevel 1 (
    echo FALHOU ao instalar ComfyUI/requirements.txt -- abortando.
    pause
    exit /b 1
)

echo [3/3] Conferindo torch/torchvision/torchaudio (mantendo cu128, ver CLAUDE.md)...
rem Sem --upgrade incondicional: so reinstala se a importacao falhar, para nao
rem trocar uma combinacao ja validada por outra so porque um pacote atrasou.
.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, 'cuda ok' if torch.cuda.is_available() else 'SEM CUDA')"
if errorlevel 1 (
    echo torch nao importa -- reinstalando pelo indice cu128 documentado no CLAUDE.md.
    .venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo FALHOU ao reinstalar torch/torchvision/torchaudio -- abortando.
        pause
        exit /b 1
    )
)

echo.
echo Reparo concluido. Reinicie o processo/servidor que estava com problema.
echo Se o problema persistir, o defeito NAO e de dependencia -- nao rode este script de novo sem investigar antes.
pause
