function Find-Ngrok {
    param([string]$ProjectRoot)
    $local = Join-Path $ProjectRoot 'tools\ngrok.exe'
    if (Test-Path -LiteralPath $local -PathType Leaf) { return $local }
    $command = Get-Command ngrok.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw 'Install ngrok on the desktop, or put ngrok.exe in this project tools folder. https://ngrok.com/download/windows'
}

function Test-NgrokTunnel {
    param($Tunnels, [string]$PublicUrl, [int]$Port)
    foreach ($tunnel in @($Tunnels)) {
        if (-not $tunnel -or -not $tunnel.public_url -or -not $tunnel.config) { continue }
        if ($tunnel.public_url.TrimEnd('/') -eq $PublicUrl.TrimEnd('/') -and
            $tunnel.config.addr -in @("http://127.0.0.1:$Port", "127.0.0.1:$Port")) {
            return $true
        }
    }
    return $false
}
