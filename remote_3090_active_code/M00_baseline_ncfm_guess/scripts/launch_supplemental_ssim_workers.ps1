param(
    [string]$ProjectRoot = "C:\xxyProject\NCFMproject_0528",
    [string]$CodeName = "NCFM_code_unified_20260530",
    [string]$ExpName = "blood_seed0_0530"
)

$ErrorActionPreference = "Stop"

$CodeRoot = Join-Path $ProjectRoot $CodeName
$ExpRoot = Join-Path (Join-Path $ProjectRoot "experiments") $ExpName
$LogRoot = Join-Path $ExpRoot "launcher_logs"
$CondaActivate = "C:\Users\Administrator\anaconda3\Scripts\activate.bat"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Jobs = @(
    @{
        Worker = "supp_ssim_gpu0"
        Gpu = "0"
        Groups = "SSIM_w005_G124,SSIM_w01_G12,SSIM_w05_G124"
    },
    @{
        Worker = "supp_ssim_gpu1"
        Gpu = "1"
        Groups = "SSIM_w01_G1,SSIM_w01_G124,SSIM_w10_G124"
    }
)

foreach ($Job in $Jobs) {
    $CmdPath = Join-Path $LogRoot ($Job.Worker + ".cmd")
    $OutPath = Join-Path $LogRoot ($Job.Worker + ".out")
    $ErrPath = Join-Path $LogRoot ($Job.Worker + ".err")
    $PidPath = Join-Path $LogRoot ($Job.Worker + ".pid")

    $Content = @"
@echo off
call "$CondaActivate" pyprc
cd /d "$CodeRoot"
python scripts\run_bloodmnist_seed0_sweep.py --exp_root "$ExpRoot" --workers 4 --niter 20000 --eval_epochs 2000 --epoch_eval_interval 100 --cam_samples 100 --summary_prefix bloodmnist_seed0_supp --gpu $($Job.Gpu) --groups $($Job.Groups) > "$OutPath" 2> "$ErrPath"
"@
    $Content | Set-Content -Encoding ASCII $CmdPath

    $Startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
    $Startup.ShowWindow = 0
    $Proc = ([wmiclass]"Win32_Process").Create("cmd.exe /d /c `"$CmdPath`"", $CodeRoot, $Startup)
    $Proc.ProcessId | Set-Content -Encoding ASCII $PidPath
    Write-Host ("Launched " + $Job.Worker + " PID=" + $Proc.ProcessId + " groups=" + $Job.Groups)
}
