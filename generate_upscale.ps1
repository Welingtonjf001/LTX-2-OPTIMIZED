param(
  [Parameter(Mandatory=$true)][string]$Prompt,
  [string]$Image = "",                          # opcional: caminho de imagem p/ image->video
  [int]$Width = 768, [int]$Height = 448,        # resolucao BAIXA de geracao (rapido, VRAM folgada)
  [int]$NumFrames = 49, [int]$Fps = 24, [int]$Steps = 8, [int]$Seed = 42,
  [string]$UpscaleModel = "realesr-animevideov3-x2", [int]$UpscaleScale = 2,
  [string]$Output = "E:\Users\home\Documents\LTX-2-OPTIMIZED\out_final.mp4",
  [switch]$KeepAudio                             # por padrao SEM audio (mais rapido); -KeepAudio p/ manter
)
$ErrorActionPreference = "Stop"
$root = "E:\Users\home\Documents\LTX-2-OPTIMIZED"
$py = "$root\.venv\Scripts\python.exe"
$env:CUDA_VISIBLE_DEVICES = '1'   # RTX 3090
$env:HF_HUB_OFFLINE = '1'
# BUGFIX auditoria 2026-09-16 (A13): nome fixo (`_gen_lowres.mp4`) -- duas
# corridas simultaneas se pisam (a segunda sobrescreve/apaga o low-res que a
# primeira ainda esta upscalando). Sufixo de PID isola cada execucao.
$lowres = "$root\_gen_lowres_$PID.mp4"

$genArgs = @(
  "-u","-m","ltx_pipelines.distilled",
  "--distilled-checkpoint-path","$root\models\ltx-2.3-22b-distilled-fp8.safetensors",
  "--gemma-root","$root\models\gemma3",
  "--spatial-upsampler-path","$root\models\ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
  "--prompt",$Prompt, "--output-path",$lowres,
  "--width","$Width","--height","$Height","--num-frames","$NumFrames",
  "--frame-rate","$Fps","--num-inference-steps","$Steps","--seed","$Seed",
  "--quantization","fp8-cast"
)
if ($Image) { $genArgs += @("--image",$Image,"0","1.0") }
if (-not $KeepAudio) { $genArgs += "--disable-audio" }

Write-Host "=== [GERACAO] $Width x $Height, $NumFrames frames ==="
$t0 = Get-Date
& $py @genArgs
if ($LASTEXITCODE -ne 0) { throw "geracao falhou (exit $LASTEXITCODE)" }
$genSec = ((Get-Date)-$t0).TotalSeconds
Write-Host ("[GERACAO OK] {0:N0}s" -f $genSec)

Write-Host "=== [UPSCALE] x$UpscaleScale ($UpscaleModel) ==="
$t1 = Get-Date
& powershell -ExecutionPolicy Bypass -File "$root\upscale_video.ps1" -InFile $lowres -Model $UpscaleModel -Scale $UpscaleScale -Output $Output
# BUGFIX auditoria 2026-09-16 (A12): o exit code do upscale (processo FILHO)
# nao era conferido -- se ele falhasse, o script seguia em frente, apagava o
# low-res (unica copia do resultado da geracao) e imprimia "PRONTO" mesmo sem
# $Output existir.
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Output)) {
    Write-Host "FALHOU: upscale nao produziu $Output (exit $LASTEXITCODE). Low-res preservado em $lowres."
    exit 1
}
$upSec = ((Get-Date)-$t1).TotalSeconds
Remove-Item $lowres -Force -ErrorAction SilentlyContinue
Write-Host ("=== PRONTO: {0} | geracao {1:N0}s + upscale {2:N0}s = {3:N0}s total ===" -f $Output, $genSec, $upSec, ($genSec+$upSec))
