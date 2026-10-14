$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到项目虚拟环境：$Python"
}

$RequiredModels = @(
    "config\models\yolov8s.pt",
    "config\models\buffalo_l\det_10g.onnx",
    "config\models\buffalo_l\w600k_r50.onnx"
)
foreach ($RelativePath in $RequiredModels) {
    $ModelPath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        throw "缺少发布模型：$ModelPath"
    }
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --clean --noconfirm "packaging\AIOTVideo.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 构建失败，退出码 $LASTEXITCODE"
    }
    Write-Host "构建完成：$ProjectRoot\dist\AIOTVideo\AIOTVideo.exe"
}
finally {
    Pop-Location
}
