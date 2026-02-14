$ErrorActionPreference = "Stop"

$python = "D:\Tools\Python3.11.9\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Setup completed."
Write-Host "Run: .\.venv\Scripts\python.exe main.py"
