#!/usr/bin/env bash
# ==============================================================================
# DePEFT All-In-One Mainnet Genesis & Production Node Launcher
# ==============================================================================
# This script initializes, compiles, verifies hardware security, and launches
# a high-performance production Node on the DePEFT Mainnet.
#
# Usage:
#   bash scripts/start_mainnet_node.sh [options]
#
# Options:
#   --rpc-port <port>       HTTP JSON-RPC Port (Default: 8545)
#   --p2p-port <port>       P2P TCP Gossip Port (Default: 9000)
#   --host <address>        Bind address (Default: 0.0.0.0)
#   --bootnodes <peers>     Comma-separated bootnodes (Empty = Genesis Node)
#   --data-dir <path>       Mainnet state directory (Default: $HOME/.depeft/mainnet)
#   --daemon                Run as background system daemon
#   --help                  Display this help message
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Default parameters
RPC_PORT=8545
P2P_PORT=9000
HOST_ADDR="0.0.0.0"
BOOTNODES=""
DATA_DIR="${HOME}/.depeft/mainnet"
DAEMON_MODE=false

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rpc-port)
            RPC_PORT="$2"
            shift 2
            ;;
        --p2p-port)
            P2P_PORT="$2"
            shift 2
            ;;
        --host)
            HOST_ADDR="$2"
            shift 2
            ;;
        --bootnodes)
            BOOTNODES="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --help|-h)
            echo "DePEFT Mainnet Node Launcher"
            echo "Usage: bash scripts/start_mainnet_node.sh [--rpc-port 8545] [--p2p-port 9000] [--daemon]"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1"
            echo "Use --help for usage instructions."
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "          🌐 DePEFT Mainnet Production Node All-In-One Launcher                 "
echo "================================================================================"
echo "Network:            MAINNET (depeft-mainnet-1)"
echo "Host Address:       ${HOST_ADDR}"
echo "RPC Port:           ${RPC_PORT}"
echo "P2P Port:           ${P2P_PORT}"
echo "Data Directory:     ${DATA_DIR}"
if [[ -z "${BOOTNODES}" ]]; then
    echo "Node Role:          GENESIS BOOTSTRAP NODE #1"
else
    echo "Bootnodes:          ${BOOTNODES}"
fi
echo "================================================================================"

# 1. Dependency & Toolchain Verification
echo "[1/6] Checking system dependencies..."
if ! command -v curl &> /dev/null; then
    echo "[!] 'curl' is required. Please install curl."
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "[!] 'git' is required. Please install git."
    exit 1
fi
if ! command -v cargo &> /dev/null; then
    echo "[*] Rust toolchain not detected. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "✓ Rust toolchain verified: $(rustc --version)"

# 2. Hardware Security & TEE Probing
echo "[2/6] Probing host hardware security & TEE drivers..."
TEE_FOUND=false
TEE_TYPE="None"

if [[ -e "/dev/tdx_guest" || -e "/dev/tdx-guest" || -e "/dev/tdx_attest" || -d "/sys/firmware/tdx" ]]; then
    TEE_FOUND=true
    TEE_TYPE="Intel TDX (Trust Domain Extensions)"
elif [[ -e "/dev/sgx_enclave" || -e "/dev/sgx/enclave" || -e "/dev/isgx" ]]; then
    TEE_FOUND=true
    TEE_TYPE="Intel SGX (Software Guard Extensions)"
elif [[ -e "/dev/sev-guest" || -e "/dev/sev" ]]; then
    TEE_FOUND=true
    TEE_TYPE="AMD SEV-SNP (Secure Encrypted Virtualization)"
elif [[ -e "/dev/nitro_enclaves" ]]; then
    TEE_FOUND=true
    TEE_TYPE="AWS Nitro Enclaves"
fi

if [[ "$TEE_FOUND" = true ]]; then
    echo "✓ Genuine TEE Hardware detected: ${TEE_TYPE}"
    echo "  -> Node is fully equipped to validate evaluations inside hardware enclave!"
else
    echo "ℹ Host TEE: Software Host Environment (No native TEE character device found)"
    echo "  -> Mainnet node will enforce strict verification on all connected validator quotes."
    echo "  -> Simulation quotes will be strictly REJECTED and fraudulent validators auto-slashed."
fi

# 3. Prepare Secure Storage Directory
echo "[3/6] Setting up secure storage directory..."
mkdir -p "${DATA_DIR}"
chmod 700 "${DATA_DIR}"
echo "✓ Storage directory initialized with restricted permissions (0700)."

# 4. Configure Firewall (if UFW active)
if command -v ufw &> /dev/null && sudo -n ufw status 2>/dev/null | grep -q "Status: active"; then
    echo "[*] Configuring UFW firewall rules..."
    sudo ufw allow "${RPC_PORT}/tcp" comment 'DePEFT Mainnet RPC' || true
    sudo ufw allow "${P2P_PORT}/tcp" comment 'DePEFT Mainnet P2P' || true
    echo "✓ Ports ${RPC_PORT} and ${P2P_PORT} allowed in UFW."
fi

# 5. Compile Production Binary in Release Mode
echo "[4/6] Compiling DePEFT node in release mode..."
cargo build --release --bin depeft
BIN="${ROOT_DIR}/target/release/depeft"
echo "✓ Production binary compiled: ${BIN}"

# 6. Launch Node
echo "[5/6] Starting DePEFT Mainnet Node..."
CMD_ARGS=(
    node start
    --mainnet
    --port "${RPC_PORT}"
    --p2p-port "${P2P_PORT}"
    --host "${HOST_ADDR}"
    --data-dir "${DATA_DIR}"
)

if [[ -n "${BOOTNODES}" ]]; then
    CMD_ARGS+=(--bootnodes "${BOOTNODES}")
fi

if [[ "${DAEMON_MODE}" = true ]]; then
    LOG_FILE="${DATA_DIR}/node.log"
    PID_FILE="${DATA_DIR}/node.pid"
    
    nohup "${BIN}" "${CMD_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    NODE_PID=$!
    echo "${NODE_PID}" > "${PID_FILE}"
    
    echo "✓ Node launched in background (PID: ${NODE_PID})"
    echo "  Log output: ${LOG_FILE}"
else
    echo "[*] Launching Node interactively in foreground..."
    echo "    (Press Ctrl+C to terminate)"
    exec "${BIN}" "${CMD_ARGS[@]}"
fi

# Health Check if launched in daemon mode
if [[ "${DAEMON_MODE}" = true ]]; then
    echo "[6/6] Verifying node health..."
    sleep 3
    MAX_TRIES=15
    COUNT=0
    HEALTHY=false
    
    while [[ $COUNT -lt $MAX_TRIES ]]; do
        if curl -s "http://127.0.0.1:${RPC_PORT}/api/v1/status" 2>/dev/null | grep -q "block_height"; then
            HEALTHY=true
            break
        fi
        sleep 1
        COUNT=$((COUNT+1))
    done

    if [[ "$HEALTHY" = true ]]; then
        STATUS_JSON=$(curl -s "http://127.0.0.1:${RPC_PORT}/api/v1/status")
        echo "================================================================================"
        echo "🎉 DePEFT Mainnet Node is LIVE & HEALTHY!"
        echo "================================================================================"
        echo "  - Status:         ${STATUS_JSON}"
        echo "  - RPC API:        http://127.0.0.1:${RPC_PORT}/api/v1/status"
        echo "  - P2P Gossip:     ${HOST_ADDR}:${P2P_PORT}"
        echo "  - Logs:           tail -f ${LOG_FILE}"
        echo "  - Stop Node:      kill \$(cat ${PID_FILE})"
        echo "================================================================================"
    else
        echo "[!] Node started but did not respond to health checks within 15 seconds."
        echo "    Check logs for details: cat ${LOG_FILE}"
        exit 1
    fi
fi
