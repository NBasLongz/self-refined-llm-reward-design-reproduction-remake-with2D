$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonExe = "C:\Users\longn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $pythonExe "app.py"
