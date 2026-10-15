# 사용: .\scripts\run_windows.ps1 <사진폴더|영상> -MarkerSize 0.25 -Out <출력폴더> [-Quality high] [-Gcp gcp.csv] [-Check check.csv]
param(
    [Parameter(Mandatory = $true, Position = 0)][string[]]$Inputs,
    [Parameter(Mandatory = $true)][double]$MarkerSize,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Quality = "high",
    [string]$Gcp = "",
    [string]$Check = ""
)
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "env.ps1")
$args2 = @("-m", "photo3d", "run") + $Inputs + @("--out", $Out, "--marker-size", $MarkerSize,
          "--quality", $Quality, "--gpu", "--colmap-bin", $env:COLMAP_BIN)
if ($Gcp) { $args2 += @("--gcp", $Gcp) }
if ($Check) { $args2 += @("--check", $Check) }
Push-Location $Root
& "$Root\.venv\Scripts\python.exe" @args2
Pop-Location
