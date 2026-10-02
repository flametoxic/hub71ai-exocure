# Start the CURE backend on Windows PowerShell.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not $env:CURE_LANGUAGE_MODE) { $env:CURE_LANGUAGE_MODE = "local" }
if (-not $env:CURE_REAL_CITY) { $env:CURE_REAL_CITY = "1" }
if (-not $env:ARRIVAL_TEMPLATE_KEY) { $env:ARRIVAL_TEMPLATE_KEY = "demo-template-key-change-me" }
if (-not $env:ARRIVAL_VAULT_KEY) { $env:ARRIVAL_VAULT_KEY = "demo-vault-key-change-me-0001" }
if (-not $env:ARRIVAL_CORE_KEY) { $env:ARRIVAL_CORE_KEY = "demo-core-key-change-me-00001" }
if (-not $env:ARRIVAL_CORE_DIR) { $env:ARRIVAL_CORE_DIR = ".\core_store" }
if (-not $env:HOST) { $env:HOST = "0.0.0.0" }
if (-not $env:PORT) { $env:PORT = "8000" }

if (-not (Test-Path -LiteralPath "arrival\demo\policy.yaml")) {
    python -m tools.gen_demo_data
}
python -m arrival.sign_templates arrival/demo/agents.yaml
New-Item -ItemType Directory -Force -Path $env:ARRIVAL_CORE_DIR | Out-Null
python -m uvicorn demo_app:app --host $env:HOST --port $env:PORT
