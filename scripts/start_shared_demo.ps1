param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Personal', 'Lan', 'Ngrok', 'LangServe', 'Prepare', 'LanFirewall')]
    [string]$Mode,
    [switch]$NoBrowser,
    [string]$NgrokConfigPath
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$ngrokAuthToken = if ($Mode -in @('Ngrok', 'LangServe')) { $env:NGROK_AUTHTOKEN }
Remove-Item Env:\NGROK_AUTHTOKEN -ErrorAction SilentlyContinue
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$demo = Join-Path $root 'demo_materials'
$logs = Join-Path $root 'logs'
$personalMode = $Mode -eq 'Personal'
$personalStateFile = Join-Path $logs 'personal-streamlit.json'

if ($Mode -eq 'LanFirewall') {
    $command = 'advfirewall firewall add rule name="RAG assignment LAN 8501" dir=in action=allow protocol=TCP localport=8501 remoteip=LocalSubnet profile=any'
    $admin = Start-Process -FilePath netsh.exe -ArgumentList $command -Verb RunAs -Wait -PassThru
    if ($admin.ExitCode -ne 0) { throw 'Could not add the LAN firewall rule.' }
    Write-Host 'TCP 8501 allowed for the local subnet.'
    return
}

. (Join-Path $PSScriptRoot 'server_gpu_check.ps1')
. (Join-Path $PSScriptRoot 'ngrok_helpers.ps1')

function Get-Response($Url, [int]$Timeout = 3) {
    try { Invoke-RestMethod -UseBasicParsing -Uri $Url -TimeoutSec $Timeout -ErrorAction Stop }
    catch { $null }
}

function Start-LoggedProcess($Executable, [string[]]$Arguments, $Name) {
    Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $logs "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $logs "$Name.stderr.log") -WindowStyle Hidden -PassThru
}

function Wait-ForService($Process, $Url, $Name) {
    for ($i = 0; $i -lt 60; $i++) {
        if ($Process.HasExited) { break }
        $response = Get-Response $Url
        if ($null -ne $response) { return $response }
        Start-Sleep -Seconds 1
    }
    throw "$Name did not start. See $logs\$Name.stderr.log."
}

function Get-LanAddress {
    $interfaces = [Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()
    foreach ($interface in $interfaces | Where-Object OperationalStatus -eq Up) {
        $properties = $interface.GetIPProperties()
        $gateway = $properties.GatewayAddresses | Where-Object { $_.Address.AddressFamily -eq 'InterNetwork' }
        if ($gateway) {
            $ipv4 = $properties.UnicastAddresses | Where-Object { $_.Address.AddressFamily -eq 'InterNetwork' } | Select-Object -First 1
            if ($ipv4) { return $ipv4.Address.IPAddressToString }
        }
    }
    throw 'Connect this PC to its Ethernet or Wi-Fi network and retry.'
}

function Assert-FreePort([int]$Port) {
    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if ($listeners | Where-Object Port -eq $Port) { throw "Port $Port is already in use. Close the previous server window first." }
}

$env:PYTHONUTF8 = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_LLM_LIBRARY = 'vulkan'
$env:OLLAMA_VULKAN = '1'
$env:OLLAMA_MODELS = Join-Path $root 'cache\ollama_models'
$env:HF_HOME = Join-Path $root 'cache\huggingface'
$env:RAG_OLLAMA_URL = 'http://127.0.0.1:11434'
$env:RAG_REQUIRE_GGUF_FILE = 'false'
if ($personalMode) {
    $env:RAG_PUBLIC_SERVER_MODE = 'false'
    $env:RAG_INDEX_DIR = Join-Path $root 'faiss_index'
    $env:RAG_PROFILE_PATH = Join-Path $root 'data\private_profile.md'
    $env:RAG_PERSONAL_ACADEMIC_RECORDS = Join-Path $root 'data\academic_records.json'
    $env:HF_HUB_OFFLINE = '1'
    foreach ($required in @((Join-Path $env:RAG_INDEX_DIR 'index.faiss'),
            (Join-Path $env:RAG_INDEX_DIR 'index.pkl'), $env:RAG_PROFILE_PATH,
            $env:RAG_PERSONAL_ACADEMIC_RECORDS)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Original personal data file is missing: $required"
        }
    }
}
else {
    Remove-Item Env:\RAG_PERSONAL_ACADEMIC_RECORDS -ErrorAction SilentlyContinue
    $env:RAG_PUBLIC_SERVER_MODE = 'true'
    $env:RAG_INDEX_DIR = Join-Path $demo 'faiss_index'
    $env:RAG_PROFILE_PATH = Join-Path $demo 'documents\demo-profile.md'
}
$config = Get-Content -Raw -LiteralPath (Join-Path $root 'deployment_config.json') | ConvertFrom-Json
$env:RAG_OLLAMA_MODEL = $config.model

New-Item -ItemType Directory -Path $logs -Force | Out-Null
if ($Mode -in @('Lan', 'Ngrok', 'Prepare') -and
    -not (Test-Path -LiteralPath (Join-Path $env:RAG_INDEX_DIR 'index.pkl'))) {
    & $python 'scripts/build_index.py' --data-dir demo_materials/documents --index-dir demo_materials/faiss_index
    if ($LASTEXITCODE -ne 0) { throw 'The demonstration index could not be built.' }
}

$tunnelMode = $Mode -in @('Ngrok', 'LangServe')
$ngrokConfigArgs = @()
if ($NgrokConfigPath) { $ngrokConfigArgs = @('--config', $NgrokConfigPath) }
if ($tunnelMode) {
    Assert-FreePort 4040
    $ngrok = Find-Ngrok -ProjectRoot $root
    try {
        & $ngrok config check @ngrokConfigArgs *> $null
        $configReady = $LASTEXITCODE -eq 0
    }
    catch { $configReady = $false }
    if (-not $configReady -and -not $ngrokAuthToken) {
        throw 'Sign in at https://dashboard.ngrok.com/get-started/your-authtoken and run SETUP_NGROK.cmd from this project folder. Then run this launcher again.'
    }
}

$port = switch ($Mode) {
    Personal { 8503 }
    Lan { 8501 }
    Ngrok { 8080 }
    LangServe { [int]$config.backend_port }
}
if ($personalMode -and (Test-Path -LiteralPath $personalStateFile -PathType Leaf)) {
    $personalUrl = "http://127.0.0.1:$port"
    $record = Get-Content -Raw -LiteralPath $personalStateFile | ConvertFrom-Json
    $running = Get-Process -Id $record.processId -ErrorAction SilentlyContinue
    if ($running -and $running.StartTime.ToFileTimeUtc().ToString() -eq $record.started -and
        (Get-Response "$personalUrl/_stcore/health")) {
        Write-Host "Personal chat URL: $personalUrl" -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $personalUrl }
        return
    }
}
if ($port) { Assert-FreePort $port }
$ollama = Join-Path $root 'tools\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $ollama)) {
    $installed = Get-Command 'ollama.exe' -ErrorAction SilentlyContinue
    $ollama = if ($installed) { $installed.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
}
if (-not (Test-Path -LiteralPath $ollama)) { throw 'Ollama was not found in the project tools folder or the Windows installation.' }

if (-not (Get-Response 'http://127.0.0.1:11434/api/version')) {
    Write-Host 'Starting the local Ollama service...'
    Assert-FreePort 11434
    $ollamaProcess = Start-LoggedProcess $ollama @('serve') 'demo-ollama'
    $null = Wait-ForService $ollamaProcess 'http://127.0.0.1:11434/api/version' 'demo-ollama'
}

$model = $config.model
$models = Get-Response 'http://127.0.0.1:11434/api/tags'
if ($model -notin @($models.models.name) -and "${model}:latest" -notin @($models.models.name)) {
    Write-Host 'Importing the coursework GGUF into Ollama. The first run takes a few minutes.'
    $modelfile = Join-Path $root 'model\professor_ljh_gguf\Modelfile'
    & $python 'scripts/import_course_model.py' $model $modelfile
    if ($LASTEXITCODE -ne 0) { throw 'Ollama model import failed.' }
}

if ($Mode -in @('Prepare', 'LangServe')) {
    Write-Host 'Warming up the coursework model on this PC...'
    $gpu = Assert-LocalOllamaGpu -ModelName $model
    Write-Host "Ollama ready: $($gpu.Processor)."
    if ($Mode -eq 'Prepare') { return }
}

$server = $null
$tunnel = $null
try {
    if ($Mode -eq 'LangServe') {
        & $python 'scripts/configure_cloud_backend.py'
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the LangServe connection key.' }
        $localUrl = "http://127.0.0.1:$port"
        $server = Start-LoggedProcess $python @('langserve_server.py') 'demo-langserve'
        $ready = Wait-ForService $server "$localUrl/health" 'demo-langserve'
        if ($ready.service -ne 'local-gpu-langserve') { throw 'The expected LangServe service did not respond.' }
    }
    else {
        $listenAddress = if ($Mode -eq 'Lan') { Get-LanAddress } else { '127.0.0.1' }
        $localUrl = "http://${listenAddress}:$port"
        $arguments = @('-m', 'streamlit', 'run', 'app.py', '--server.address', $listenAddress,
            '--server.port', "$port", '--server.headless', 'true')
        $logName = if ($personalMode) { 'personal-streamlit' } else { "demo-$Mode" }
        $server = Start-LoggedProcess $python $arguments $logName
        $null = Wait-ForService $server "$localUrl/_stcore/health" $logName
        if ($personalMode) {
            [pscustomobject]@{
                processId = $server.Id
                started = $server.StartTime.ToFileTimeUtc().ToString()
            } | ConvertTo-Json | Set-Content -LiteralPath $personalStateFile -Encoding UTF8
        }
    }

    $openUrl = $localUrl
    if ($tunnelMode) {
        $ngrokArguments = @('http', $localUrl, '--url', $config.backend_url,
            '--log=stdout', '--log-format=json', '--inspect=false') + $ngrokConfigArgs
        try {
            if ($ngrokAuthToken) { $env:NGROK_AUTHTOKEN = $ngrokAuthToken }
            $tunnel = Start-LoggedProcess $ngrok $ngrokArguments "demo-ngrok-$Mode"
        }
        finally {
            Remove-Item Env:\NGROK_AUTHTOKEN -ErrorAction SilentlyContinue
            $ngrokAuthToken = $null
        }
        $publicUrl = $null
        for ($i = 0; $i -lt 45; $i++) {
            if ($tunnel.HasExited) { break }
            $state = Get-Response 'http://127.0.0.1:4040/api/tunnels'
            $entry = @($state.tunnels) | Where-Object {
                $_.public_url -like 'https://*' -and $_.config.addr -in @($localUrl, "127.0.0.1:$port")
            } | Select-Object -First 1
            if ($entry) { $publicUrl = $entry.public_url.TrimEnd('/'); break }
            Start-Sleep -Seconds 1
        }
        if (-not $publicUrl) { throw "ngrok could not connect. See $logs\demo-ngrok-$Mode.stdout.log and stderr.log." }
        $openUrl = $publicUrl
    }

    if ($Mode -eq 'LangServe') {
        & $python 'scripts/configure_cloud_backend.py' --backend-url $publicUrl
        if ($LASTEXITCODE -ne 0) { throw 'Could not save the active domain for Streamlit Cloud.' }
        Write-Host "LangServe API: $publicUrl/llm/invoke" -ForegroundColor Green
        Write-Host 'Streamlit Cloud settings: .server_secrets\streamlit_secrets.toml'
    }
    else {
        Write-Host "Open this chat URL: $openUrl" -ForegroundColor Green
        if ($Mode -eq 'Lan') { Write-Host 'For other devices on the local subnet. Windows Firewall may need the companion rule.' }
        if (-not $NoBrowser) { Start-Process $openUrl }
    }

    Write-Host 'Keep this window open. Press Q to stop this server.'
    while (-not $server.HasExited -and ($null -eq $tunnel -or -not $tunnel.HasExited)) {
        if (-not [Console]::IsInputRedirected -and [Console]::KeyAvailable -and [Console]::ReadKey($true).Key -eq 'Q') { return }
        Start-Sleep -Seconds 1
    }
    throw "A server process stopped. See $logs for details."
}
finally {
    foreach ($child in @($tunnel, $server)) {
        if ($null -ne $child -and -not $child.HasExited) {
            # Windows venv launchers have a child Python process of their own.
            & taskkill.exe /PID $child.Id /T /F 2>$null | Out-Null
        }
    }
}
