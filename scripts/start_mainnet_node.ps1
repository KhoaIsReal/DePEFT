<#
.SYNOPSIS
    DePEFT All-In-One Mainnet Genesis & Production Node Launcher (PowerShell).

.DESCRIPTION
    Compiles, initializes security configuration, and launches a production
    node on the DePEFT Mainnet with strict anti-fraud enforcement.

.PARAMETER RpcPort
    HTTP JSON-RPC Port (Default: 8545).

.PARAMETER P2pPort
    P2P TCP Gossip Port (Default: 9000).

.PARAMETER HostAddress
    Host IP address to bind to (Default: 0.0.0.0).

.PARAMETER Bootnodes
    Comma-separated bootnode addresses (leave empty for Genesis node).

.PARAMETER DataDir
    Path to persistent blockchain state directory (Default: $HOME\.depeft\mainnet).

.PARAMETER Daemon
    Run node as a background process.

.EXAMPLE
    .\scripts\start_mainnet_node.ps1 -RpcPort 8545 -P2pPort 9000
#>

[CmdletBinding()]
param (
    [int]$RpcPort = 8545,
    [int]$P2pPort = 9000,
    [string]$HostAddress = "0.0.0.0",
    [string]$Bootnodes = "",
    [string]$DataDir = "$HOME\.depeft\mainnet",
    [switch]$Daemon
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "================================================================================" -ForegroundColor Blue
Write-Host "          🌐 DePEFT Mainnet Production Node All-In-One Launcher                 " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Blue
Write-Host "Network:            MAINNET (depeft-mainnet-1)" -ForegroundColor Green
Write-Host "Host Address:       $HostAddress"
Write-Host "RPC Port:           $RpcPort"
Write-Host "P2P Port:           $P2pPort"
Write-Host "Data Directory:     $DataDir"
if ([string]::IsNullOrWhiteSpace($Bootnodes)) {
    Write-Host "Node Role:          GENESIS BOOTSTRAP NODE #1" -ForegroundColor Yellow
} else {
    Write-Host "Bootnodes:          $Bootnodes"
}
Write-Host "================================================================================" -ForegroundColor Blue

# 1. Check Toolchain
Write-Host "[1/5] Checking Rust toolchain..." -ForegroundColor Cyan
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Cargo/Rust is not installed or not in PATH. Please install Rust from https://rustup.rs"
    exit 1
}
$rustVersion = rustc --version
Write-Host "✓ Rust toolchain verified: $rustVersion" -ForegroundColor Green

# 2. Setup Data Directory
Write-Host "[2/5] Initializing persistent data directory..." -ForegroundColor Cyan
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}
Write-Host "✓ Data directory ready: $DataDir" -ForegroundColor Green

# 3. Compile in Release Mode
Write-Host "[3/5] Compiling DePEFT node binary in release mode..." -ForegroundColor Cyan
cargo build --release --bin depeft
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile DePEFT node binary."
    exit 1
}

$binPath = Join-Path $RootDir "target\release\depeft.exe"
if (-not (Test-Path $binPath)) {
    $binPath = Join-Path $RootDir "target/release/depeft"
}
Write-Host "✓ Production binary: $binPath" -ForegroundColor Green

# 4. Construct Command Arguments
$cmdArgs = @(
    "node", "start",
    "--mainnet",
    "--port", $RpcPort.ToString(),
    "--p2p-port", $P2pPort.ToString(),
    "--host", $HostAddress,
    "--data-dir", $DataDir
)

if (-not [string]::IsNullOrWhiteSpace($Bootnodes)) {
    $cmdArgs += @("--bootnodes", $Bootnodes)
}

# 5. Launch Node
Write-Host "[4/5] Launching Mainnet Node..." -ForegroundColor Cyan
if ($Daemon) {
    $logFile = Join-Path $DataDir "node.log"
    $proc = Start-Process -FilePath $binPath -ArgumentList $cmdArgs -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru
    Write-Host "✓ Node started in background (PID: $($proc.Id))" -ForegroundColor Green
    Write-Host "  Logs: $logFile"

    # Health Check
    Write-Host "[5/5] Checking node health via HTTP JSON-RPC..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
    $healthy = $false
    for ($i = 0; $i -lt 15; $i++) {
        try {
            $status = Invoke-RestMethod -Uri "http://127.0.0.1:$RpcPort/api/v1/status" -TimeoutSec 2
            if ($status.block_height -ge 1) {
                $healthy = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if ($healthy) {
        Write-Host "================================================================================" -ForegroundColor Blue
        Write-Host "🎉 DePEFT Mainnet Node is LIVE & HEALTHY!" -ForegroundColor Green
        Write-Host "================================================================================" -ForegroundColor Blue
        Write-Host "  - RPC API:        http://127.0.0.1:$RpcPort/api/v1/status"
        Write-Host "  - P2P Gossip:     $HostAddress:$P2pPort"
        Write-Host "  - Process ID:     $($proc.Id)"
        Write-Host "  - Stop Node:      Stop-Process -Id $($proc.Id)"
        Write-Host "================================================================================" -ForegroundColor Blue
    } else {
        Write-Warning "Node started but did not respond to status endpoint. Check $logFile for details."
    }
} else {
    Write-Host "[*] Running interactively in foreground (Press Ctrl+C to stop)..." -ForegroundColor Yellow
    & $binPath $cmdArgs
}
