param(
    [ValidateSet('Test', 'Smoke', 'Viewer', 'PaperTower')]
    [string]$Task = 'Smoke',
    [string]$Distro = $(if ($env:GSPLAT_WSL_DISTRIBUTION) { $env:GSPLAT_WSL_DISTRIBUTION } else { 'Ubuntu' }),
    [string]$Python = '',
    [string]$CudaHome = '/usr/local/cuda-11.8',
    [string]$SourceRoot = '',
    [string]$OutputDir = '',
    [string]$Resolution = '768x1152',
    [int]$Frames = 4,
    [int]$FrameStart = 0,
    [int]$FrameStride = 1,
    [double]$ScaleDisp = 50.0,
    [double]$AzimuthDeg = -35.0,
    [double]$ElevationDeg = 10.0,
    [double]$DistanceScale = 0.32,
    [int]$DeformationFrame = 0
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = $env:GSPLAT_PYTHON
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    throw 'A tested WSL Python interpreter is required; pass -Python or set GSPLAT_PYTHON.'
}
if ($Python.LastIndexOf('/') -lt 1) {
    throw 'Python must be an absolute WSL path; pass -Python or set GSPLAT_PYTHON.'
}
if ($Task -eq 'PaperTower') {
    if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
        $SourceRoot = $env:GSPLAT_TOWER_SOURCE_ROOT
    }
    if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
        throw 'PaperTower requires -SourceRoot or GSPLAT_TOWER_SOURCE_ROOT pointing to the private tower assets.'
    }
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $OutputDir = "outputs/real_tower_demo_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    }
    $OutputDir = $OutputDir -replace '\\', '/'
    if ([System.IO.Path]::IsPathRooted($OutputDir)) {
        throw 'OutputDir must be a new repository-relative path.'
    }
    $OutputPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
    $RepoPrefix = [System.IO.Path]::GetFullPath($RepoRoot) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $OutputPath.StartsWith($RepoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputDir must stay inside the repository.'
    }
    if (Test-Path -LiteralPath $OutputPath) {
        throw "OutputDir already exists; refusing to overwrite: $OutputPath"
    }
}
$WslRoot = & wsl.exe -d $Distro --exec /usr/bin/wslpath -a -u $RepoRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the repository path in WSL.' }
$WslRoot = $WslRoot.Trim()
$PythonBin = $Python.Substring(0, $Python.LastIndexOf('/'))
$WslArguments = @(
    '-d', $Distro, '--cd', $WslRoot, '--exec', '/usr/bin/env',
    'PYTHONDONTWRITEBYTECODE=1', 'MAX_JOBS=2', 'GSPLAT_3DGS_ONLY=1',
    "CUDA_HOME=$CudaHome", "PYTHONPATH=${WslRoot}:${WslRoot}/examples",
    "PATH=${PythonBin}:${CudaHome}/bin:/usr/local/bin:/usr/bin:/bin",
    $Python, '-B'
)

switch ($Task) {
    'Test' { $WslArguments += @('-m', 'unittest', 'discover', '-s', 'tests', '-v') }
    'Smoke' { $WslArguments += @('scripts/run_synthetic_smoke.py') }
    'Viewer' {
        $Synthetic = "$WslRoot/outputs/synthetic_smoke"
        $WslArguments += @(
            'auto_play_highres.py', '--static_ckpt', "$Synthetic/inputs/static.pt",
            '--moving_ckpt', "$Synthetic/mapped.pt",
            '--vib_path', "$Synthetic/inputs/vibration.npy",
            '--scale_disp', '1', '--host', '127.0.0.1'
        )
    }
    'PaperTower' {
        $WslArguments += @(
            'scripts/run_real_tower_pipeline.py',
            '--source-root', $SourceRoot,
            '--output-dir', $OutputDir,
            '--resolution', $Resolution,
            '--frames', "$Frames",
            '--frame-start', "$FrameStart",
            '--frame-stride', "$FrameStride",
            '--scale-disp', "$ScaleDisp",
            '--azimuth-deg', "$AzimuthDeg",
            '--elevation-deg', "$ElevationDeg",
            '--distance-scale', "$DistanceScale",
            '--deformation-frame', "$DeformationFrame"
        )
    }
}

& wsl.exe @WslArguments
exit $LASTEXITCODE
