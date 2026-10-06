<#
.SYNOPSIS
    Registers (or removes) the optional Windows scheduled task "volrisk-live-daily" (docs/LIVE_SPEC.md section 8).

.DESCRIPTION
    The task runs `uv run python -m volrisk_live daily` every day at 07:00 UTC (change with -AtUtc) in the project
    folder, as the current user, only while that user is logged on (no password is stored). The trigger is set in
    UTC, so it does not move with the local summer time. Output is appended to data\logs\live_daily.log. Nothing in
    the project runs this script automatically: run it yourself once if you want the daily schedule, and run it with
    -Unregister to remove the task again.

    daily = update -> score -> stamp/upgrade timestamps -> forecast (+ stamp) -> forward-test report -> summary.
    A missed run (computer off or user logged off) starts as soon as possible afterwards (StartWhenAvailable). When
    that is late in the UTC day, an asset whose next session has already closed when the payload is written gets no
    forecast (its outcome exists; the payload lists it under checks.closed_assets), and `daily` logs a warning.

    Why 07:00 UTC. A forecast proves something only if it is recorded before its outcome exists. The live data end
    is the last complete UTC day, so a run on UTC day D forecasts the sessions of day D:
      - the previous day's files must be published: Dukascopy (EUR/USD, S&P 500) about 00:05 UTC, Binance daily zips
        (BTC, ETH) about 01:30-02:30 UTC (Last-Modified of the files of 2026-09-28 to 2026-10-01);
      - the run must be written before the S&P 500 opens (13:30 UTC, 14:30 UTC in the northern winter), so that
        forecast is recorded before its session opens;
      - BTC/ETH (UTC-day sessions) and EUR/USD (17:00 New York to 17:00 New York, i.e. 21:00/22:00 UTC) are already
        trading when the previous day's data are complete; their forecasts are recorded during the session, from data
        that end at the previous session, and must be written before the session closes - the earliest close is
        EUR/USD at 21:00 UTC, the S&P 500 closes at 20:00/21:00 UTC;
      - the OpenTimestamps proof submitted at the run has to reach a Bitcoin block before that first close (it usually
        takes a few hours), so earlier is better.
    A run late in the UTC evening (e.g. 23:30 local time in Europe) would record the EUR/USD and S&P 500 forecasts
    after their sessions closed. -AtUtc therefore accepts 03:00-12:00 UTC only, unless -AnyTime is given.
    Every payload records its start time (run_utc) and its write time (checks.computed_utc); the CLI, the dashboard
    and the reports judge "recorded before the session opened / closed" from the later of the two.

.PARAMETER Unregister
    Remove the task instead of registering it.

.PARAMETER AtUtc
    UTC time of day for the run, HH:mm (default 07:00; accepted 03:00-12:00 unless -AnyTime).

.PARAMETER AnyTime
    Accept an -AtUtc outside 03:00-12:00 UTC (the forecasts of some assets would then not precede their outcome).

.PARAMETER TaskName
    Name of the scheduled task (default volrisk-live-daily).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1 -AtUtc 06:30

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [switch]$Unregister,
    [string]$AtUtc = "07:00",
    [switch]$AnyTime,
    [string]$TaskName = "volrisk-live-daily"
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EarliestUtc = [timespan]"03:00"  # previous UTC day's Binance zips published (about 01:30-02:30 UTC)
$LatestUtc = [timespan]"12:00"    # written before the S&P 500 opens at 13:30 UTC, long before the first close

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed the scheduled task '$TaskName'."
    }
    else {
        Write-Host "There is no scheduled task named '$TaskName'."
    }
    return
}

$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) {
    throw "uv was not found on PATH. Install uv (https://docs.astral.sh/uv/) or add it to PATH, then run this again."
}
try {
    $TimeUtc = [datetime]::ParseExact($AtUtc, "HH:mm", [Globalization.CultureInfo]::InvariantCulture).TimeOfDay
}
catch {
    throw "-AtUtc must be a UTC time of day as HH:mm (24-hour clock), got '$AtUtc'."
}
if (-not $AnyTime -and ($TimeUtc -lt $EarliestUtc -or $TimeUtc -gt $LatestUtc)) {
    throw ("-AtUtc $AtUtc is outside 03:00-12:00 UTC: before 03:00 the previous day's Binance files may not be " +
        "published yet; after 12:00 the S&P 500 forecast is no longer recorded before its session opens, and late " +
        "in the day the next EUR/USD and S&P 500 sessions have already closed (their forecasts would not precede " +
        "their outcome). Pass -AnyTime to register it anyway.")
}

$LogDir = Join-Path $ProjectDir "data\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "live_daily.log"

# First run: the next occurrence of AtUtc. The trigger's start boundary is written in UTC ("...Z"), which makes the
# Task Scheduler fire at that UTC time every day ("synchronize across time zones"), whatever the local summer time.
$NowUtc = [datetime]::UtcNow
$FirstUtc = $NowUtc.Date.Add($TimeUtc)
if ($FirstUtc -le $NowUtc) {
    $FirstUtc = $FirstUtc.AddDays(1)
}
$FirstLocal = $FirstUtc.ToLocalTime()

# cmd.exe runs the command so that stdout and stderr are appended to the log; PYTHONUTF8 keeps the log UTF-8.
$Command = "set `"PYTHONUTF8=1`" && `"$Uv`" run python -m volrisk_live daily >> `"$Log`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" -Argument "/d /c $Command" `
    -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At $FirstLocal
$Trigger.StartBoundary = $FirstUtc.ToString("yyyy-MM-ddTHH:mm:ss", [Globalization.CultureInfo]::InvariantCulture) + "Z"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$Principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$Description = "volrisk live forecasts: uv run python -m volrisk_live daily at $AtUtc UTC in $ProjectDir " +
    "(log: $Log). Remove with scripts\register_daily_task.ps1 -Unregister."

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Principal $Principal -Description $Description -Force | Out-Null

$Registered = (Get-ScheduledTask -TaskName $TaskName).Triggers[0].StartBoundary
$Next = (Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime
Write-Host "Registered the scheduled task '$TaskName': daily at $AtUtc UTC (trigger start $Registered)."
Write-Host "  command : uv run python -m volrisk_live daily"
Write-Host "  folder  : $ProjectDir"
Write-Host "  log     : $Log"
Write-Host "  next run: $Next local time"
Write-Host "Run it now with: Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove it with : powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1 -Unregister"
