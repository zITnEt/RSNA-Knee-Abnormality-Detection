param([int]$RetrySeconds = 300)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $RepoRoot '../..')).Path
$Report = Join-Path $ProjectRoot 'results/resnet50_cloud'
$GcloudCommand = 'D:\tools\google-cloud-sdk\bin\gcloud.cmd'
$Instance = 'YOUR_GPU_VM'
$Project = 'YOUR_GCP_PROJECT'
$Zone = 'us-central1-a'
$Archive = Join-Path $Report 'resnet50_cloud_10epochs.tar.gz'
$Bootstrap = Join-Path $PSScriptRoot 'bootstrap_resnet50_cloud.sh'
$Credential = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.kaggle/kaggle.json'
$Log = Join-Path $Report 'retry_worker.log'
$State = Join-Path $Report 'retry_state.json'
$Sha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
$Common = @('--zone', $Zone, '--project', $Project, '--quiet')
New-Item -ItemType Directory -Force -Path $Report | Out-Null
try {
    $WorkerLock = [IO.File]::Open((Join-Path $Report 'retry_worker.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    Write-Output 'A retry worker is already active'
    exit 0
}
[IO.File]::WriteAllText((Join-Path $Report 'retry_worker_pid.txt'), [string]$PID)

function Save-State($Phase, $Attempt, $Message) {
    $value = @{ phase=$Phase; attempt=$Attempt; message=$Message; pid=$PID;
        instance=$Instance; zone=$Zone; updated_utc=[DateTime]::UtcNow.ToString('o') }
    $json = $value | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText("$State.tmp", $json)
    Move-Item -LiteralPath "$State.tmp" -Destination $State -Force
}

function Invoke-Cloud([string[]]$Arguments) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $lines = & $GcloudCommand @Arguments 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    $message = ($lines | ForEach-Object { $_.ToString() }) -join "`n"
    Add-Content -LiteralPath $Log -Value ("[{0}] exit={1}`n{2}" -f [DateTime]::UtcNow.ToString('o'), $code, $message)
    return @{ code=$code; message=$message }
}

function Get-RetryReason([string]$Message) {
    if ($Message -match 'ZONE_RESOURCE_POOL_EXHAUSTED|resource_availability|gpu_availability') {
        return 'capacity'
    }
    if ($Message -match 'ConnectTimeout|ReadTimeout|ConnectionError|Connection reset|Max retries exceeded|timed out|temporarily unavailable|Service Unavailable|Internal Server Error|\b(429|500|502|503|504)\b') {
        return 'network'
    }
    return $null
}

function Require-Cloud([string[]]$Arguments) {
    for ($retry=0; $retry -lt 3; $retry++) {
        $result = Invoke-Cloud $Arguments
        if ($result.code -eq 0) { return $result }
        if ((Get-RetryReason $result.message) -ne 'network') { break }
        Start-Sleep -Seconds 15
    }
    throw ('Cloud operation failed; see retry_worker.log')
}

$attempt = 0
try {
    if (-not (Test-Path -LiteralPath $Credential)) { throw 'Local Kaggle credential file is missing' }
    while ($true) {
        $attempt++
        Save-State 'requesting_capacity' $attempt 'Starting the specified VM'
        $result = Invoke-Cloud (@('compute','instances','start',$Instance) + $Common)
        if ($result.code -eq 0 -or $result.message -match 'already running|already in the RUNNING') { break }
        $reason = Get-RetryReason $result.message
        if (-not $reason) {
            throw 'VM start failed with a non-transient error; see retry_worker.log'
        }
        if ($reason -eq 'capacity') {
            Save-State 'waiting_for_capacity' $attempt "L4 unavailable; retry in $RetrySeconds seconds"
        } else {
            Save-State 'waiting_for_cloud' $attempt "Temporary Google API network error; retry in $RetrySeconds seconds"
        }
        Start-Sleep -Seconds $RetrySeconds
    }
    Save-State 'connecting' $attempt 'VM started; waiting for SSH'
    $connected = $false
    for ($i=0; $i -lt 12; $i++) {
        $result = Invoke-Cloud (@('compute','ssh',$Instance,'--command',
            'mkdir -p ~/rsna-resnet50-20261013/resnet50_cloud/.auth && chmod 700 ~/rsna-resnet50-20261013/resnet50_cloud/.auth') + $Common)
        if ($result.code -eq 0) { $connected=$true; break }
        Start-Sleep -Seconds 15
    }
    if (-not $connected) { throw 'VM is running but SSH did not become available' }
    Save-State 'deploying' $attempt 'Copying verified package and isolated Kaggle configuration over SSH'
    $null = Require-Cloud (@('compute','scp',$Archive,$Bootstrap,($Instance+':~/')) + $Common)
    # Transport only over SSH; never include credentials in Cloud Storage or logs.
    $null = Require-Cloud (@('compute','scp',$Credential,
        ($Instance+':~/rsna-resnet50-20261013/resnet50_cloud/.auth/kaggle.json')) + $Common)
    $command = 'chmod 600 ~/rsna-resnet50-20261013/resnet50_cloud/.auth/kaggle.json; nohup bash ~/bootstrap_resnet50_cloud.sh ' + $Sha + ' > ~/rsna-resnet50-20261013/bootstrap.log 2>&1 < /dev/null &'
    $null = Require-Cloud (@('compute','ssh',$Instance,'--command',$command) + $Common)
    Save-State 'remote_setup_started' $attempt 'Detached VM setup will verify data and run 10 epochs; inspect bootstrap.log and run/progress.json'
    while ($true) {
        Start-Sleep -Seconds $RetrySeconds
        $result = Invoke-Cloud (@('compute','ssh',$Instance,'--command',
            'cat ~/rsna-resnet50-20261013/deployment_status.json') + $Common)
        if ($result.code -ne 0) { continue }
        if ($result.message -match '"state"\s*:\s*"training_complete"') {
            Save-State 'training_complete' $attempt 'All 10 epochs completed on the specified VM'
            break
        }
        if ($result.message -match '"state"\s*:\s*"setup_failed"') {
            throw 'VM setup or training failed; inspect remote bootstrap.log and the last saved checkpoint'
        }
        Save-State 'remote_job_active' $attempt 'Data staging or 10-epoch training is active on the specified VM'
    }
} catch {
    Save-State 'needs_attention' $attempt $_.Exception.Message
    Add-Content -LiteralPath $Log -Value $_.Exception.Message
    exit 1
} finally {
    $WorkerLock.Dispose()
}
