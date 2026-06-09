param(
    [string]$ProjectRoot = "C:\xxyProject\NCFMproject_0528",
    [string]$CodeName = "NCFM_code_unified_20260530",
    [string]$BloodExpName = "blood_seed0_0530",
    [string]$ExpName = "pneumonia_seed0_0530",
    [int]$Niter = 20000,
    [int]$EvalEpochs = 2000,
    [int]$CamSamples = 100,
    [int]$Workers = 4,
    [int]$PollSeconds = 600
)

$ErrorActionPreference = "Stop"

$Dataset = "pneumoniamnist"
$CodeRoot = Join-Path $ProjectRoot $CodeName
$BloodRoot = Join-Path (Join-Path $ProjectRoot "experiments") $BloodExpName
$ExpRoot = Join-Path (Join-Path $ProjectRoot "experiments") $ExpName
$LauncherLogs = Join-Path $ExpRoot "launcher_logs"
$OldRoot = Join-Path $ProjectRoot "ncfm_t512_main_20260528"
$CondaActivate = "C:\Users\Administrator\anaconda3\Scripts\activate.bat"

function Write-Log($Message) {
    New-Item -ItemType Directory -Force -Path $LauncherLogs | Out-Null
    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    $Line | Tee-Object -FilePath (Join-Path $LauncherLogs "after_blood_launcher.log") -Append
}

function Count-BloodMetrics {
    $RunRoot = Join-Path $BloodRoot "runs\bloodmnist\ipc10"
    if (!(Test-Path $RunRoot)) { return 0 }
    return @(Get-ChildItem -Path $RunRoot -Filter metrics.json -Recurse -ErrorAction SilentlyContinue).Count
}

function Find-ExistingPath($Candidates, $Description) {
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }
    throw "Could not find $Description. Tried: $($Candidates -join '; ')"
}

New-Item -ItemType Directory -Force -Path $ExpRoot, $LauncherLogs | Out-Null
Write-Log "Waiting for BloodMNIST sweep to reach 28/28 before launching PneumoniaMNIST."
while ((Count-BloodMetrics) -lt 28) {
    $Count = Count-BloodMetrics
    Write-Log "BloodMNIST completed $Count/28; sleeping $PollSeconds seconds."
    Start-Sleep -Seconds $PollSeconds
}
Write-Log "BloodMNIST completed 28/28. Preparing PneumoniaMNIST."

New-Item -ItemType Directory -Force -Path (Join-Path $ExpRoot "data\medmnist") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ExpRoot "checkpoints\pretrain") | Out-Null

$SourceData = Find-ExistingPath @(
    (Join-Path $OldRoot "data\medmnist\pneumoniamnist.npz"),
    (Join-Path $ProjectRoot "shared_assets\data\medmnist\pneumoniamnist.npz"),
    (Join-Path $ProjectRoot "data\medmnist\pneumoniamnist.npz")
) "PneumoniaMNIST data"
$TargetData = Join-Path $ExpRoot "data\medmnist\pneumoniamnist.npz"
if (!(Test-Path $TargetData)) {
    Copy-Item -Force $SourceData $TargetData
}

$SourcePretrain = Find-ExistingPath @(
    (Join-Path $OldRoot "checkpoints\pretrain\pneumoniamnist"),
    (Join-Path $ProjectRoot "shared_assets\checkpoints\pretrain\pneumoniamnist"),
    (Join-Path $ProjectRoot "checkpoints\pretrain\pneumoniamnist")
) "PneumoniaMNIST pretrain directory"
$TargetPretrain = Join-Path $ExpRoot "checkpoints\pretrain\pneumoniamnist"
if (!(Test-Path (Join-Path $TargetPretrain "premodel19_trained.pth.tar"))) {
    if (Test-Path $TargetPretrain) {
        Remove-Item -Recurse -Force $TargetPretrain
    }
    Copy-Item -Recurse -Force $SourcePretrain $TargetPretrain
}

$PatchNotes = Join-Path $ExpRoot "PATCH_NOTES.md"
$PatchNotesContent = @(
    "# PneumoniaMNIST Sweep Patch Notes",
    "",
    "Date: 2026-05-30",
    "",
    "- Code root: $CodeRoot",
    "- Experiment root: $ExpRoot",
    "- Dataset: PneumoniaMNIST",
    "- IPC: 10",
    "- Seed policy: seed0 single run",
    "- Matrix: same 28 groups as BloodMNIST",
    "- Queue workers: 4 total, 2 per L20 GPU",
    "- Starts only after BloodMNIST reaches 28/28 metrics.",
    "",
    "## Outputs",
    "",
    "- Per-group metrics: runs/pneumoniamnist/ipc10/<GROUP>/metrics.json",
    "- Per-group manifest: runs/pneumoniamnist/ipc10/<GROUP>/artifact_manifest.json",
    "- Final synthetic data: results/condense/**/distilled_data/data_20000.pt",
    "- Eval checkpoint: checkpoints/synthetic_train/pneumoniamnist/ipc10_<GROUP>_best.pth.tar",
    "- Merged reports: merged_reports/pneumoniamnist_seed0_*.csv and *.md"
) -join [Environment]::NewLine
$PatchNotesContent | Set-Content -Encoding UTF8 $PatchNotes

$Runner = Join-Path $CodeRoot "scripts\run_medmnist_seed0_sweep.py"
& python $Runner --dataset $Dataset --exp_root $ExpRoot --list_only | Set-Content -Encoding UTF8 (Join-Path $ExpRoot "RUN_PLAN.txt")

$CommonArgs = "--dataset $Dataset --exp_root `"$ExpRoot`" --workers $Workers --niter $Niter --eval_epochs $EvalEpochs --epoch_eval_interval 100 --cam_samples $CamSamples --summary_prefix pneumoniamnist_seed0_queue"

$Jobs = @(
    @{ Worker = "queue_gpu0a"; Gpu = "0"; WorkerId = "0a" },
    @{ Worker = "queue_gpu0b"; Gpu = "0"; WorkerId = "0b" },
    @{ Worker = "queue_gpu1a"; Gpu = "1"; WorkerId = "1a" },
    @{ Worker = "queue_gpu1b"; Gpu = "1"; WorkerId = "1b" }
)

foreach ($Job in $Jobs) {
    $CmdPath = Join-Path $LauncherLogs ($Job.Worker + ".cmd")
    $OutPath = Join-Path $LauncherLogs ($Job.Worker + ".out")
    $ErrPath = Join-Path $LauncherLogs ($Job.Worker + ".err")
    $PidPath = Join-Path $LauncherLogs ($Job.Worker + ".pid")

    $Content = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
python scripts\run_medmnist_seed0_queue.py $CommonArgs --gpu $($Job.Gpu) --worker_id $($Job.WorkerId) > "$OutPath" 2> "$ErrPath"
"@
    $Content | Set-Content -Encoding ASCII $CmdPath

    $Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
    $Startup.ShowWindow = 0
    $Proc = ([wmiclass]"Win32_Process").Create("cmd.exe /d /c `"$CmdPath`"", $CodeRoot, $Startup)
    $Proc.ProcessId | Set-Content -Encoding ASCII $PidPath
    Write-Log "Launched $($Job.Worker) PID=$($Proc.ProcessId) GPU=$($Job.Gpu)"
}

$CollectorCmd = Join-Path $LauncherLogs "collector.cmd"
$CollectorOut = Join-Path $LauncherLogs "collector.out"
$CollectorErr = Join-Path $LauncherLogs "collector.err"
$CollectorContent = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
:loop
python scripts\collect_medmnist_seed0_reports.py --dataset $Dataset --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
timeout /t 600 /nobreak > nul
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'queue_gpu0a.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'queue_gpu0b.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'queue_gpu1a.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
tasklist /FI "PID eq $(Get-Content (Join-Path $LauncherLogs 'queue_gpu1b.pid'))" | findstr /I "cmd.exe" > nul
if %ERRORLEVEL%==0 goto loop
python scripts\collect_medmnist_seed0_reports.py --dataset $Dataset --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
"@
$CollectorContent | Set-Content -Encoding ASCII $CollectorCmd

$Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$Startup.ShowWindow = 0
$CollectorProc = ([wmiclass]"Win32_Process").Create("cmd.exe /d /c `"$CollectorCmd`"", $CodeRoot, $Startup)
$CollectorProc.ProcessId | Set-Content -Encoding ASCII (Join-Path $LauncherLogs "collector.pid")
Write-Log "Launched collector PID=$($CollectorProc.ProcessId)"

Write-Host "PneumoniaMNIST seed0 sweep launched."
Write-Host "CodeRoot: $CodeRoot"
Write-Host "ExpRoot: $ExpRoot"
Write-Host "Launcher logs: $LauncherLogs"
