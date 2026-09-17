param(
  [string]$Folder = "E:\Users\home\Documents\LTX-2-OPTIMIZED",  # onde estao as cenas scene_*.mp4
  [string]$Pattern = "scene_*.mp4",
  [string]$UpscaleModel = "realesr-animevideov3-x2", [int]$UpscaleScale = 2,
  [string]$Output = "E:\Users\home\Documents\LTX-2-OPTIMIZED\filme_final_upscaled.mp4",
  [switch]$NoConcat   # se setado, so upscala cada cena (nao junta)
)
$ErrorActionPreference = "Stop"
$root = "E:\Users\home\Documents\LTX-2-OPTIMIZED"
$ffmpeg = & "$root\.venv\Scripts\python.exe" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"

$scenes = Get-ChildItem $Folder -Filter $Pattern | Sort-Object Name
if ($scenes.Count -eq 0) { throw "nenhuma cena '$Pattern' em $Folder" }
Write-Host "=== $($scenes.Count) cenas encontradas ==="

# BUGFIX auditoria 2026-09-16 (A13): pasta fixa (`tools\_film_up`) -- dois
# filmes upscalados em paralelo se pisam. Sufixo de PID isola cada corrida.
$upDir = "$root\tools\_film_up_$PID"
New-Item -ItemType Directory -Force -Path $upDir | Out-Null
$upFiles = @()
$i = 0
foreach ($s in $scenes) {
  $i++
  $o = "$upDir\up_{0:D3}.mp4" -f $i
  Write-Host "--- upscale cena $i/$($scenes.Count): $($s.Name) ---"
  & powershell -ExecutionPolicy Bypass -File "$root\upscale_video.ps1" -InFile $s.FullName -Model $UpscaleModel -Scale $UpscaleScale -Output $o
  # BUGFIX auditoria 2026-09-16 (A12): o exit code do upscale por cena nao era
  # conferido -- uma cena que falhasse entrava mesmo assim em $upFiles, e o
  # concat mais abaixo ou falhava com um erro generico do ffmpeg (sem dizer
  # QUAL cena) ou, pior, produzia um filme mais curto sem avisar.
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $o)) {
    Write-Host "FALHOU: cena $i ($($s.Name)) nao upscalou (exit $LASTEXITCODE). Abortando -- cenas ja feitas ficam em $upDir."
    exit 1
  }
  $upFiles += $o
}

if ($NoConcat) {
  Write-Host "=== cenas upscaladas em $upDir (sem concatenar) ==="
  return
}

# concatena as cenas upscaladas num filme unico
$listFile = "$upDir\list.txt"
# BUGFIX auditoria 2026-09-16 (A09): Encoding ASCII quebra em nome de cena com
# acento -- mesmo defeito e mesmo conserto do assemble_final.py (UTF-8 sem BOM
# + apostrofos escapados pela sintaxe do ffconcat).
$upFiles | ForEach-Object { "file '$($_.Replace('\','/').Replace("'","'\''"))'" } |
  Set-Content -Encoding UTF8 $listFile
Write-Host "=== concatenando $($upFiles.Count) cenas -> filme final ==="
& $ffmpeg -loglevel error -y -f concat -safe 0 -i $listFile -c copy $Output
# BUGFIX auditoria 2026-09-16 (A12): exit code do ffmpeg de concat tambem nao
# era conferido -- o script dizia "FILME PRONTO" mesmo sem $Output existir, e
# ainda apagava as cenas upscaladas (unica copia intermediaria) em seguida.
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Output)) {
  Write-Host "FALHOU: concat nao produziu $Output (exit $LASTEXITCODE). Cenas upscaladas preservadas em $upDir."
  exit 1
}
Remove-Item $upDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "=== FILME PRONTO: $Output ==="
