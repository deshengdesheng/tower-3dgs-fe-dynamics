param(
    [string]$ChainDirectory = '',
    [int]$FrameCount = 40,
    [int]$FrameStride = 10,
    [string]$Resolution = '1024x1536',
    [double]$DisplacementScale = 50,
    [double]$Azimuth = 90,
    [double]$Elevation = 15,
    [double]$DistanceScale = 3.0,
    [string]$Distribution = $(if ($env:GSPLAT_WSL_DISTRIBUTION) { $env:GSPLAT_WSL_DISTRIBUTION } else { 'Ubuntu' }),
    [string]$ProcessingPython = $env:GSPLAT_PROCESSING_PYTHON,
    [string]$RenderingPython = $env:GSPLAT_RENDERING_PYTHON,
    [string]$GsplatExtension = $env:GSPLAT_PREBUILT
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProcessingPython)) {
    throw 'Set GSPLAT_PROCESSING_PYTHON or pass -ProcessingPython with the WSL Python used by the pipeline.'
}
if ([string]::IsNullOrWhiteSpace($RenderingPython)) {
    throw 'Set GSPLAT_RENDERING_PYTHON or pass -RenderingPython with a Python that has the gsplat CUDA runtime.'
}
$towerRepository = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ChainDirectory)) {
    $ChainDirectory = Join-Path $towerRepository 'outputs/real_tower_roi_continuous_corrected_20261017_v4'
}
$towerChain = (Resolve-Path -LiteralPath $ChainDirectory).Path
$towerRuntime = Join-Path $towerChain 'step_iii/runtime'
$towerManifest = Join-Path $towerRuntime 'runtime_manifest.json'
$towerTransform = Join-Path $towerRuntime 'gs_to_fe.txt'
foreach ($towerRequired in @($towerManifest, $towerTransform)) {
    if (-not (Test-Path -LiteralPath $towerRequired -PathType Leaf)) {
        throw "Missing runtime input: $towerRequired"
    }
}
$towerPayload = Get-Content -LiteralPath $towerManifest -Raw | ConvertFrom-Json
if ($FrameCount -lt 2 -or $FrameStride -lt 1 -or (($FrameCount - 1) * $FrameStride) -ge $towerPayload.vibration_shape[0]) {
    throw 'FrameCount/FrameStride must select at least two frames within the existing displacement stream.'
}
$towerOutput = Join-Path $towerChain ('step_iv/dynamic_playback_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$towerLinuxRepository = (& wsl -d $Distribution -- wslpath -u $towerRepository.Replace('\', '/')).Trim()
$towerLinuxRuntime = (& wsl -d $Distribution -- wslpath -u $towerRuntime.Replace('\', '/')).Trim()
$towerLinuxTransform = (& wsl -d $Distribution -- wslpath -u $towerTransform.Replace('\', '/')).Trim()
$towerLinuxOutput = (& wsl -d $Distribution -- wslpath -u $towerOutput.Replace('\', '/')).Trim()
$towerRenderScript = $towerLinuxRepository + '/scripts/run_dynamic_gs.py'
$towerRenderArguments = @(
    '-u'
    $towerRenderScript
    '--runtime-dir'
    $towerLinuxRuntime
    '--output-dir'
    $towerLinuxOutput
    '--gs-to-fe-transform'
    $towerLinuxTransform
    '--frames'
    "$FrameCount"
    '--frame-stride'
    "$FrameStride"
    '--resolution'
    $Resolution
    '--scale-disp'
    "$DisplacementScale"
    '--radius-clip'
    '0'
    '--device'
    'cuda'
    '--azimuth-deg'
    "$Azimuth"
    '--elevation-deg'
    "$Elevation"
    '--distance-scale'
    "$DistanceScale"
    '--save-moving-frames'
    '--save-undeformed'
)
if (-not [string]::IsNullOrWhiteSpace($GsplatExtension)) {
    $towerRenderArguments += @('--gsplat-prebuilt', $GsplatExtension)
}
$towerRenderArguments += @('--renderer-python', $RenderingPython)
& wsl -d $Distribution -- $ProcessingPython @towerRenderArguments
if ($LASTEXITCODE -ne 0) { throw "gsplat rendering failed with exit code $LASTEXITCODE" }
Write-Output $towerOutput
Write-Output 'Completed real gsplat rendering of the same expanded Gaussian model. Translation replay only; FE solving was not rerun.'
