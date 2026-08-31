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

$tmp = "$root\tools\_upscale_tmp"
$fin = "$tmp\in"; $fout = "$tmp\out"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $fin,$fout | Out-Null

Write-Host "[1/3] extraindo frames ($fps fps)..."
& $ffmpeg -loglevel error -i $InFile "$fin\f_%06d.png"

$n = (Get-ChildItem $fin -Filter *.png).Count
Write-Host "[2/3] upscale de $n frames com $Model (x$Scale) na GPU $Gpu..."
& $rex -i $fin -o $fout -n $Model -s $Scale -f png -g $Gpu

Write-Host "[3/3] remontando video (preservando audio de origem, se houver)..."
& $ffmpeg -loglevel error -y -framerate $fps -i "$fout\f_%06d.png" -i $InFile `
  -map 0:v:0 -map "1:a:0?" -c:v libx264 -pix_fmt yuv420p -crf 16 -c:a aac -shortest $Output

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "PRONTO -> $Output"
