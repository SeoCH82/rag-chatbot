$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ProjectRoot 'scripts\ngrok_helpers.ps1')
$exe = Find-Ngrok -ProjectRoot $ProjectRoot
$secure = Read-Host 'Paste YOUR AUTHTOKEN from ngrok dashboard (input is hidden)' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($token) -or $token.StartsWith('rd_')) {
        throw 'Use YOUR AUTHTOKEN, not a reserved-domain ID.'
    }
    # ngrok stores it in its user config, outside the files uploaded to GitHub.
    & $exe config add-authtoken $token *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Could not save the ngrok authtoken.' }
    Write-Host 'ngrok credentials were saved for this Windows user.'
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $token = $null
    $secure.Dispose()
}
