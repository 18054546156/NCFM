param(
    [string]$ProjectRoot = "C:\xxyProject\NCFMproject_0528",
    [string]$CodeName = "NCFM_code_unified_20260530",
    [string]$ExpName = "blood_seed0_0530",
    [int]$Niter = 20000,
    [int]$EvalEpochs = 2000,
    [int]$CamSamples = 100,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$CodeRoot = Join-Path $ProjectRoot $CodeName
$ExpRoot = Join-Path (Join-Path $ProjectRoot "experiments") $ExpName
$LauncherLogs = Join-Path $ExpRoot "launcher_logs"
$OldRoot = Join-Path $ProjectRoot "ncfm_t512_main_20260528"
$CondaActivate = "C:\Users\Administrator\anaconda3\Scripts\activate.bat"

New-Item -ItemType Directory -Force -Path $ExpRoot, $LauncherLogs | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ExpRoot "data\medmnist") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ExpRoot "checkpoints\pretrain") | Out-Null

$SourceData = Join-Path $OldRoot "data\medmnist\bloodmnist.npz"
$TargetData = Join-Path $ExpRoot "data\medmnist\bloodmnist.npz"
if (!(Test-Path $TargetData)) {
    Copy-Item -Force $SourceData $TargetData
}

$SourcePretrain = Join-Path $OldRoot "checkpoints\pretrain\bloodmnist"
$TargetPretrain = Join-Path $ExpRoot "checkpoints\pretrain\bloodmnist"
if (!(Test-Path (Join-Path $TargetPretrain "premodel19_trained.pth.tar"))) {
    if (Test-Path $TargetPretrain) {
        Remove-Item -Recurse -Force $TargetPretrain
    }
    Copy-Item -Recurse -Force $SourcePretrain $TargetPretrain
}

$PatchNotes = Join-Path $ExpRoot "PATCH_NOTES.md"
$PatchNotesContent = @(
    "# Patch Notes",
    "",
    "Date: 2026-05-30",
    "",
    "## Code package",
    "",
    "- Run code: $CodeRoot",
    "- Experiment root: $ExpRoot",
    "- Dataset: BloodMNIST only",
    "- Seed policy: single run, seed0",
    "- Parallelism: two worker processes, one per L20 GPU",
    "",
    "## Changes made for this run",
    "",
    "- Created unified code package from the L20-verified T512 code base.",
    "- Merged SSIM regularization into the unified package.",
    "- Added BloodMNIST seed0 hyperparameter runner: scripts/run_bloodmnist_seed0_sweep.py",
    "- Added report collector: scripts/collect_bloodmnist_seed0_reports.py",
    "- Preserved Windows/L20 compatibility: gloo backend, USE_LIBUV=0, direct script launch on Windows.",
    "- Added per-run artifact_manifest.json with config, final pt, eval checkpoint, commands, and metrics.",
    "",
    "## Planned groups",
    "",
    "- Baseline: T = 128 / 256 / 512 / 1024",
    "- Local Patch NCFD: lambda = 0.3 / 0.5 / 0.8, grid = 2 / 4 / 7, localT = 512",
    "- DAM Attention: lambda = 10 / 50 / 100, layers = [0,1] / [1,2] / [0,1,2]",
    "- SSIM: six compact settings over ssim_weight and ssim_grids",
    "",
    "## Important outputs",
    "",
    "- Per-run metrics: runs/bloodmnist/ipc10/<GROUP>/metrics.json",
    "- Per-run manifest: runs/bloodmnist/ipc10/<GROUP>/artifact_manifest.json",
    "- Final synthetic data: results/condense/**/distilled_data/data_20000.pt",
    "- Eval checkpoint: checkpoints/synthetic_train/bloodmnist/ipc10_<GROUP>_best.pth.tar",
    "- Merged reports: merged_reports/*.csv and merged_reports/*.md"
) -join [Environment]::NewLine
$PatchNotesContent | Set-Content -Encoding UTF8 $PatchNotes

$CommonArgs = "--exp_root `"$ExpRoot`" --workers $Workers --niter $Niter --eval_epochs $EvalEpochs --epoch_eval_interval 100 --cam_samples $CamSamples --summary_prefix bloodmnist_seed0_worker"

foreach ($gpu in @(0, 1)) {
    $CmdPath = Join-Path $LauncherLogs "worker_gpu$gpu.cmd"
    $OutPath = Join-Path $LauncherLogs "worker_gpu$gpu.out"
    $ErrPath = Join-Path $LauncherLogs "worker_gpu$gpu.err"
    $Chunk = $gpu
    $WorkerCmd = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
python scripts\run_bloodmnist_seed0_sweep.py $CommonArgs --gpu $gpu --chunk_id $Chunk --num_chunks 2 > "$OutPath" 2> "$ErrPath"
"@
    $WorkerCmd | Set-Content -Encoding ASCII $CmdPath

    $Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
    $Startup.ShowWindow = 0
    $CommandLine = "cmd.exe /d /c `"$CmdPath`""
    $Proc = ([wmiclass]"Win32_Process").Create($CommandLine, $CodeRoot, $Startup)
    $Proc.ProcessId | Set-Content -Encoding ASCII (Join-Path $LauncherLogs "worker_gpu$gpu.pid")
}

$CollectorCmd = Join-Path $LauncherLogs "collector.cmd"
$CollectorOut = Join-Path $LauncherLogs "collector.out"
$CollectorErr = Join-Path $LauncherLogs "collector.err"
$CollectorContent = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
:loop
python scripts\collect_bloodmnist_seed0_reports.py --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
timeout /t 600 /nobreak > nul
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'worker_gpu0.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'worker_gpu1.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
python scripts\collect_bloodmnist_seed0_reports.py --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
"@
$CollectorContent | Set-Content -Encoding ASCII $CollectorCmd

$Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$Startup.ShowWindow = 0
$CollectorCommandLine = "cmd.exe /d /c `"$CollectorCmd`""
$CollectorProc = ([wmiclass]"Win32_Process").Create($CollectorCommandLine, $CodeRoot, $Startup)
$CollectorProc.ProcessId | Set-Content -Encoding ASCII (Join-Path $LauncherLogs "collector.pid")

$Runner = Join-Path $CodeRoot "scripts\run_bloodmnist_seed0_sweep.py"
& python $Runner --exp_root $ExpRoot --list_only | Set-Content -Encoding UTF8 (Join-Path $ExpRoot "RUN_PLAN.txt")

Write-Host "Launched BloodMNIST seed0 sweep."
Write-Host "CodeRoot: $CodeRoot"
Write-Host "ExpRoot: $ExpRoot"
Write-Host "Launcher logs: $LauncherLogs"
