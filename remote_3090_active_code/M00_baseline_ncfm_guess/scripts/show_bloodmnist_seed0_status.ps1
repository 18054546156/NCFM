param(
    [string]$ProjectRoot = "C:\xxyProject\NCFMproject_0528",
    [string]$ExpName = "blood_seed0_0530"
)

$ExpRoot = Join-Path (Join-Path $ProjectRoot "experiments") $ExpName
$LauncherLogs = Join-Path $ExpRoot "launcher_logs"

Write-Host "ExpRoot: $ExpRoot"
Write-Host ""

foreach ($gpu in 0, 1) {
    $PidPath = Join-Path $LauncherLogs "worker_gpu$gpu.pid"
    if (Test-Path $PidPath) {
        $pidValue = Get-Content $PidPath
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "worker_gpu$gpu PID=$pidValue RUNNING"
        } else {
            Write-Host "worker_gpu$gpu PID=$pidValue EXITED"
        }
    } else {
        Write-Host "worker_gpu$gpu PID missing"
    }
}

Write-Host ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

Write-Host ""
$Metrics = Get-ChildItem (Join-Path $ExpRoot "runs\bloodmnist\ipc10") -Recurse -Filter "metrics.json" -ErrorAction SilentlyContinue
Write-Host "Completed metrics.json files: $($Metrics.Count) / 28"

$Reports = Join-Path $ExpRoot "merged_reports"
if (Test-Path $Reports) {
    Write-Host ""
    Write-Host "Merged reports:"
    Get-ChildItem $Reports -File | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
}
