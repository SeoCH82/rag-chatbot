# Shared by setup_windows.ps1. Loading this file does not install anything.
$Python311Probe = "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)"

function Test-Python311 {
    param([string]$Executable)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    try {
        & $Executable -c $Python311Probe *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-PythonModules {
    param([string]$Executable, [string[]]$Modules)
    try {
        # In Windows PowerShell 5.1, native stderr can terminate a script when
        # ErrorActionPreference=Stop. Treat import errors as a failed probe.
        & $Executable -c 'import importlib,sys; [importlib.import_module(name) for name in sys.argv[1:]]' @Modules *> $null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
}

function Find-Python311 {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")
    )
    if (Test-Python311 $candidates[0]) { return $candidates[0] }
    foreach ($key in @(
        'HKCU:\Software\Python\PythonCore\3.11\InstallPath',
        'HKLM:\Software\Python\PythonCore\3.11\InstallPath'
    )) {
        try {
            if (Test-Path -LiteralPath $key) {
                $installKey = Get-Item -LiteralPath $key -ErrorAction Stop
                $registered = $installKey.GetValue('ExecutablePath')
                if ($registered) { $candidates += $registered }
                $installDirectory = $installKey.GetValue('')
                if ($installDirectory) {
                    $candidates += Join-Path $installDirectory 'python.exe'
                }
            }
        }
        catch { } # Registry access is optional; continue with other interpreters.
    }

    $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyCommand) {
        try {
            $resolved = & $pyCommand.Source -3.11 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                $candidates += ([string]$resolved).Trim()
            }
        }
        catch { }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
        $candidates += $pythonCommand.Source
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python311 $candidate) { return $candidate }
    }
    return $null
}

function Install-Python311 {
    param([Parameter(Mandatory=$true)][string]$ProjectRoot)

    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'Python 3.11 64-bit requires 64-bit Windows.'
    }
    $installDirectory = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311'
    $installerDirectory = Join-Path $ProjectRoot 'cache\installers'
    New-Item -ItemType Directory -Path $installerDirectory -Force | Out-Null
    $installerPath = Join-Path $installerDirectory 'python-3.11.9-amd64.exe'
    $downloadPath = Join-Path $installerDirectory 'python-3.11.9-amd64.pending.exe'
    $installerUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'

    Write-Host 'Python 3.11 (64-bit) is missing. Downloading the official installer...'
    $previousTls = [Net.ServicePointManager]::SecurityProtocol
    try {
        [Net.ServicePointManager]::SecurityProtocol = $previousTls -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $downloadPath -TimeoutSec 600
    }
    catch {
        throw "Python download failed. Check your internet connection. $($_.Exception.Message)"
    }
    finally {
        [Net.ServicePointManager]::SecurityProtocol = $previousTls
    }

    $signature = Get-AuthenticodeSignature -LiteralPath $downloadPath
    if ($signature.Status -ne 'Valid' -or
        -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch 'O=Python Software Foundation(?:,|$)') {
        throw 'Python installer signature verification failed. The installer was not executed.'
    }
    Move-Item -LiteralPath $downloadPath -Destination $installerPath -Force

    Write-Host 'Installing Python 3.11 for this user. Existing Python versions will be kept.'
    $arguments = @(
        '/quiet', 'InstallAllUsers=0',
        ('TargetDir="{0}"' -f $installDirectory),
        'Include_pip=1', 'Include_launcher=0', 'InstallLauncherAllUsers=0',
        'PrependPath=0', 'AssociateFiles=0', 'Shortcuts=0',
        'Include_test=0', 'Include_doc=0', 'Include_tcltk=0'
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) {
        throw "Python installation failed (exit code $($process.ExitCode))."
    }
    $executable = Join-Path $installDirectory 'python.exe'
    if (-not (Test-Python311 $executable)) {
        throw 'Python installation finished but Python 3.11 64-bit could not be verified.'
    }
    return $executable
}
