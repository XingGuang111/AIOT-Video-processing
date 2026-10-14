$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Candidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)
$Compiler = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $Compiler) {
    throw "未找到 Inno Setup 6。安装后重新运行本脚本。"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "dist\AIOTVideo\AIOTVideo.exe"))) {
    throw "未找到发布目录，请先运行 packaging\build_windows.ps1。"
}

Push-Location $ProjectRoot
try {
    & $Compiler "packaging\AIOTVideo.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "安装包构建失败，退出码 $LASTEXITCODE"
    }
    Write-Host "安装包已生成到：$ProjectRoot\release"
}
finally {
    Pop-Location
}
