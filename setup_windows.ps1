$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
. (Join-Path $ProjectRoot 'scripts\python_bootstrap.ps1')
$VenvIsUsable = Test-Python311 $VenvPython

if (-not $VenvIsUsable) {
    $PythonExecutable = Find-Python311
    if (-not $PythonExecutable) {
        $PythonExecutable = Install-Python311 -ProjectRoot $ProjectRoot
    }
    if (Test-Path -LiteralPath $VenvRoot) {
        $backupName = ".venv_incompatible_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
        Rename-Item -LiteralPath $VenvRoot -NewName $backupName
        Write-Output "기존 가상환경을 $backupName 폴더에 보관했습니다."
    }
    & $PythonExecutable -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        throw "Python 가상환경 생성에 실패했습니다."
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip 업그레이드에 실패했습니다. 인터넷 연결을 확인하세요."
}
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Python 패키지 설치에 실패했습니다. 인터넷 연결과 저장공간을 확인하세요."
}
& $VenvPython -c "import streamlit, faiss, sentence_transformers, langchain_ollama"
if ($LASTEXITCODE -ne 0) {
    throw "필수 Python 패키지 확인에 실패했습니다."
}

Write-Output "설치 완료: $VenvPython"
