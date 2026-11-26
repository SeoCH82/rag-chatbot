# This file defines functions only; importing it never starts a model.
function Get-ModelGpuStatus {
    param($Models, [Parameter(Mandatory=$true)][string]$ModelName)
    $expected = if ($ModelName.Contains(':')) { $ModelName } else { "$($ModelName):latest" }
    $loaded = @($Models) | Where-Object {
        $_.name -eq $ModelName -or $_.name -eq $expected -or
        $_.model -eq $ModelName -or $_.model -eq $expected
    } | Select-Object -First 1
    if (-not $loaded) { throw "Model is not loaded: $ModelName" }
    if (-not $loaded.PSObject.Properties['size_vram']) {
        throw 'Ollama did not report GPU memory. Update Ollama and try again.'
    }
    $vram = [long]$loaded.size_vram
    if ($vram -le 0) {
        throw '100% CPU detected. GPU server startup stopped. Check the NVIDIA driver, Ollama GPU support and available VRAM.'
    }
    $processor = if ([long]$loaded.size -gt 0 -and $vram -ge [long]$loaded.size) {
        'GPU'
    } else {
        'CPU/GPU (partial GPU offload)'
    }
    return [pscustomobject]@{
        Model = $loaded.name
        Processor = $processor
        GpuMemoryBytes = $vram
    }
}

function Assert-LocalOllamaGpu {
    param([Parameter(Mandatory=$true)][string]$ModelName)
    # These requests always go to the server PC itself, never a cloud endpoint.
    $baseUrl = 'http://127.0.0.1:11434'
    $payload = @{
        model = $ModelName
        prompt = 'Hello'
        stream = $false
        keep_alive = '5m'
        options = @{ num_ctx = 2048; num_predict = 1 }
    } | ConvertTo-Json -Depth 4
    $null = Invoke-RestMethod -Uri "$baseUrl/api/generate" -Method Post `
        -ContentType 'application/json' -Body $payload -TimeoutSec 600
    $response = Invoke-RestMethod -Uri "$baseUrl/api/ps" -Method Get -TimeoutSec 10
    return Get-ModelGpuStatus -Models $response.models -ModelName $ModelName
}

function Get-LanClientUrls {
    param([int]$Port = 8501)
    try {
        [Net.Dns]::GetHostAddresses([Net.Dns]::GetHostName()) |
            Where-Object {
                $_.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork -and
                -not [Net.IPAddress]::IsLoopback($_) -and
                $_.ToString() -notlike '169.254.*'
            } |
            ForEach-Object { "http://$($_.ToString()):$Port" } |
            Sort-Object -Unique
    }
    catch { return @() }
}
