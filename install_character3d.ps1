param([string]$ModelRoot = "G:\models", [switch]$SkipModels)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ModelRoot, "$ModelRoot\sam-3d-body", "$ModelRoot\hunyuan3d" | Out-Null
Write-Host "Raiz de modelos: $ModelRoot"
if ($SkipModels) { Write-Host "Modo estrutura: downloads pulados."; exit 0 }
if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
  Write-Host "Instalando huggingface_hub..."
  python -m pip install -U huggingface_hub
  # BUGFIX auditoria 2026-09-16 (A12): pip podia falhar (rede, permissao) e o
  # script seguia direto pro download com o CLI ainda ausente.
  if ($LASTEXITCODE -ne 0) { Write-Host "FALHOU ao instalar huggingface_hub -- abortando."; exit 1 }
}
Write-Host "Baixando Hunyuan3D-2mini (shape) para G:\models\hunyuan3d..."
huggingface-cli download tencent/Hunyuan3D-2mini --local-dir "$ModelRoot\hunyuan3d"
# BUGFIX auditoria 2026-09-16 (A12): download nao verificado -- rede caindo no
# meio terminava em "Instalacao concluida" com o modelo pela metade.
if ($LASTEXITCODE -ne 0) { Write-Host "FALHOU ao baixar Hunyuan3D-2mini (exit $LASTEXITCODE)."; exit 1 }
Write-Host "SAM 3D Body exige aceite/licença no Hugging Face. Depois de autenticar, execute:"
Write-Host "huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir $ModelRoot\sam-3d-body"
Write-Host "Instalação concluída (Hunyuan3D-2mini). O Flux existente não foi copiado."
