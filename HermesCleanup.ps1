param(
    [int]$ParentPid = 0,
    [string]$OllamaExe = 'C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe',
    [string]$Distro = 'Ubuntu-22.04'
)

$ErrorActionPreference = 'SilentlyContinue'

if ($ParentPid -gt 0) {
    while (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 2
    }
} else {
    # Watchdog mode: locate the batch command and wait until its console is closed.
    do {
        $launcher = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'Hermes\.bat'
        })
        if ($launcher.Count -gt 0) { Start-Sleep -Seconds 2 }
    } while ($launcher.Count -gt 0)
}

# Unload every model exposed by the local Ollama API before stopping the server.
try {
    $ps = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
    foreach ($model in @($ps.models)) {
        $body = @{ model = $model.name; keep_alive = 0; prompt = '' } | ConvertTo-Json
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec 10 | Out-Null
    }
} catch { }

# Stop both possible Ollama hosts used by this workstation.
try { & $OllamaExe stop qwen3:30b 2>$null | Out-Null } catch { }
try { & $OllamaExe stop qwen3:8b 2>$null | Out-Null } catch { }
Get-Process -Name 'ollama app' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name 'ollama' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
try { wsl.exe -d $Distro -u root -e bash -lc 'systemctl stop ollama' | Out-Null } catch { }

Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'hermes-agent.*dashboard'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
