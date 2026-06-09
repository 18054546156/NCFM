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

$TargetData = Join-Path $ExpRoot "data\medmnist\bloodmnist.npz"
if (!(Test-Path $TargetData)) {
    Copy-Item -Force (Join-Path $OldRoot "data\medmnist\bloodmnist.npz") $TargetData
}

$TargetPretrain = Join-Path $ExpRoot "checkpoints\pretrain\bloodmnist"
if (!(Test-Path (Join-Path $TargetPretrain "premodel19_trained.pth.tar"))) {
    if (Test-Path $TargetPretrain) {
        Remove-Item -Recurse -Force $TargetPretrain
    }
    Copy-Item -Recurse -Force (Join-Path $OldRoot "checkpoints\pretrain\bloodmnist") $TargetPretrain
}

$CommonArgs = "--exp_root `"$ExpRoot`" --workers $Workers --niter $Niter --eval_epochs $EvalEpochs --epoch_eval_interval 100 --cam_samples $CamSamples --summary_prefix bloodmnist_seed0_queue"
$GpuMap = @(0, 1, 0, 1)

for ($worker = 0; $worker -lt 4; $worker++) {
    $gpu = $GpuMap[$worker]
    $CmdPath = Join-Path $LauncherLogs "queue_worker${worker}_gpu${gpu}.cmd"
    $OutPath = Join-Path $LauncherLogs "queue_worker${worker}_gpu${gpu}.out"
    $ErrPath = Join-Path $LauncherLogs "queue_worker${worker}_gpu${gpu}.err"
    $WorkerCmd = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
python scripts\run_bloodmnist_seed0_queue.py $CommonArgs --gpu $gpu --worker_id $worker > "$OutPath" 2> "$ErrPath"
"@
    $WorkerCmd | Set-Content -Encoding ASCII $CmdPath
    $Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
    $Startup.ShowWindow = 0
    $CommandLine = "cmd.exe /d /c `"$CmdPath`""
    $Proc = ([wmiclass]"Win32_Process").Create($CommandLine, $CodeRoot, $Startup)
    $Proc.ProcessId | Set-Content -Encoding ASCII (Join-Path $LauncherLogs "queue_worker${worker}_gpu${gpu}.pid")
}

$CollectorCmd = Join-Path $LauncherLogs "queue_collector.cmd"
$CollectorOut = Join-Path $LauncherLogs "queue_collector.out"
$CollectorErr = Join-Path $LauncherLogs "queue_collector.err"
$CollectorContent = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
:loop
python scripts\collect_bloodmnist_seed0_reports.py --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
timeout /t 600 /nobreak > nul
tasklist /FI "IMAGENAME eq python.exe" /V | findstr /I "run_bloodmnist_seed0_queue.py" > nul
if %ERRORLEVEL%==0 goto loop
python scripts\collect_bloodmnist_seed0_reports.py --exp_root "$ExpRoot" > "$CollectorOut" 2> "$CollectorErr"
"@
$CollectorContent | Set-Content -Encoding ASCII $CollectorCmd
$Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$Startup.ShowWindow = 0
$CollectorProc = ([wmiclass]"Win32_Process").Create("cmd.exe /d /c `"$CollectorCmd`"", $CodeRoot, $Startup)
$CollectorProc.ProcessId | Set-Content -Encoding ASCII (Join-Path $LauncherLogs "queue_collector.pid")

"Launched 4 queue workers: two per L20 GPU."
"ExpRoot: $ExpRoot"
"LauncherLogs: $LauncherLogs"
