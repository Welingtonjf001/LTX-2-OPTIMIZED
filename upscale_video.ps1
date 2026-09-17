param(
  [Parameter(Mandatory=$true)][string]$InFile,
  # (param renomeado de Input para evitar a variavel automatica $input)
  [string]$Output = "",
  [string]$Model = "realesr-animevideov3-x2",   # video 2x; use "realesrgan-x4plus" p/ foto 4x
  [int]$Scale = 2,
  [int]$Gpu = 0
)
$ErrorActionPreference = "Stop"
$root  = "E:\Users\home\Documents\LTX-2-OPTIMIZED"
$rex   = "$root\tools\realesrgan\realesrgan-ncnn-vulkan.exe"
$ffmpeg = & "$root\.venv\Scripts\python.exe" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if (-not $Output) { $Output = [IO.Path]::ChangeExtension($InFile, $null).TrimEnd('.') + "_x$Scale.mp4" }

# fps de origem
$fps = & "$root\.venv\Scripts\python.exe" -c "import imageio.v3 as iio,sys; print(iio.immeta(r'$InFile',plugin='pyav').get('fps',24))"

# BUGFIX auditoria 2026-09-16 (A13): pasta temporaria fixa (tools\_upscale_tmp)
# apagada no INICIO e no FIM -- duas execucoes simultaneas (ex.: upscale de
# duas cenas em paralelo) se pisam: a segunda corrida apaga os frames que a
# primeira ainda esta lendo/escrevendo. Sufixo de PID+timestamp torna cada
# corrida isolada.
$tmp = "$root\tools\_upscale_tmp_{0}_{1}" -f $PID, (Get-Date -Format "yyyyMMddHHmmssfff")
$fin = "$tmp\in"; $fout = "$tmp\out"
New-Item -ItemType Directory -Force -Path $fin,$fout | Out-Null

# BUGFIX auditoria 2026-09-16 (A12): $ErrorActionPreference="Stop" so cobre
# erros do PROPRIO PowerShell -- um executavel nativo (ffmpeg, realesrgan) que
# falha e devolve exit code != 0 NAO lanca excecao, so imprime no stderr e
# segue em frente. Sem checar $LASTEXITCODE, o script chegava em "PRONTO" com
# um video vazio/corrompido ou nenhum frame upscalado. Cada chamada nativa
# agora e conferida antes do proximo passo, e os intermediarios NAO sao
# apagados em falha (uteis para diagnostico).
function Assert-LastExit($etapa) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALHOU ($etapa): exit code $LASTEXITCODE. Intermediarios preservados em $tmp"
        exit 1
    }
}

Write-Host "[1/3] extraindo frames ($fps fps)..."
& $ffmpeg -loglevel error -i $InFile "$fin\f_%06d.png"
Assert-LastExit "extracao de frames"

$n = (Get-ChildItem $fin -Filter *.png).Count
if ($n -eq 0) {
    Write-Host "FALHOU: nenhum frame extraido de $InFile. Intermediarios preservados em $tmp"
    exit 1
}
Write-Host "[2/3] upscale de $n frames com $Model (x$Scale) na GPU $Gpu..."
& $rex -i $fin -o $fout -n $Model -s $Scale -f png -g $Gpu
Assert-LastExit "upscale (realesrgan-ncnn-vulkan)"

$nOut = (Get-ChildItem $fout -Filter *.png -ErrorAction SilentlyContinue).Count
if ($nOut -ne $n) {
    Write-Host "FALHOU: $n frame(s) de entrada, $nOut de saida apos o upscale. Intermediarios preservados em $tmp"
    exit 1
}

Write-Host "[3/3] remontando video (preservando audio de origem, se houver)..."
& $ffmpeg -loglevel error -y -framerate $fps -i "$fout\f_%06d.png" -i $InFile `
  -map 0:v:0 -map "1:a:0?" -c:v libx264 -pix_fmt yuv420p -crf 16 -c:a aac -shortest $Output
Assert-LastExit "remontagem final"

if (-not (Test-Path $Output)) {
    Write-Host "FALHOU: $Output nao foi criado apesar do exit code 0. Intermediarios preservados em $tmp"
    exit 1
}

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "PRONTO -> $Output"
