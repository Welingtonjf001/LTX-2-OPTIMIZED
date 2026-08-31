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

$upDir = "$root\tools\_film_up"
Remove-Item $upDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $upDir | Out-Null
$upFiles = @()
$i = 0
foreach ($s in $scenes) {
  $i++
  $o = "$upDir\up_{0:D3}.mp4" -f $i
  Write-Host "--- upscale cena $i/$($scenes.Count): $($s.Name) ---"
  & powershell -ExecutionPolicy Bypass -File "$root\upscale_video.ps1" -InFile $s.FullName -Model $UpscaleModel -Scale $UpscaleScale -Output $o
  $upFiles += $o
}

if ($NoConcat) {
  Write-Host "=== cenas upscaladas em $upDir (sem concatenar) ==="
  return
}

# concatena as cenas upscaladas num filme unico
$listFile = "$upDir\list.txt"
$upFiles | ForEach-Object { "file '$($_.Replace('\','/'))'" } | Set-Content -Encoding ASCII $listFile
Write-Host "=== concatenando $($upFiles.Count) cenas -> filme final ==="
& $ffmpeg -loglevel error -y -f concat -safe 0 -i $listFile -c copy $Output
Remove-Item $upDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "=== FILME PRONTO: $Output ==="
