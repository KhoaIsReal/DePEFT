#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "         DePEFT Local Testnet Deployment Script (Multi-Node Swarm)              "
echo "================================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="/tmp/depeft_testnet"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR/node1" "$DATA_DIR/node2" "$DATA_DIR/node3"

echo "[1/3] Compiling DePEFT node binary in release mode..."
cargo build --release

BIN="$ROOT_DIR/target/release/depeft"

echo "[2/3] Starting Bootstrap Validator Node (Node 1 on :8545 / P2P :9000)..."
$BIN node start --port 8545 --p2p-port 9000 --data-dir "$DATA_DIR/node1" --testnet-tee-sim --enable-operator-endpoints --enable-faucet > "$DATA_DIR/node1.log" 2>&1 &
PID1=$!
echo "      Node 1 PID: $PID1 | Dashboard: http://127.0.0.1:8545"

sleep 2

echo "[3/3] Starting Peer Validator Node (Node 2 on :8546 / P2P :9001)..."
$BIN node start --port 8546 --p2p-port 9001 --bootnodes "127.0.0.1:9000" --data-dir "$DATA_DIR/node2" --testnet-tee-sim --enable-operator-endpoints --enable-faucet > "$DATA_DIR/node2.log" 2>&1 &
PID2=$!
echo "      Node 2 PID: $PID2 | Dashboard: http://127.0.0.1:8546"

echo "================================================================================"
echo "✓ DePEFT Multi-Node Testnet is RUNNING!"
echo "  - Web Dashboard: http://127.0.0.1:8545"
echo "  - Logs: $DATA_DIR/node1.log, $DATA_DIR/node2.log"
echo "  - To Stop: kill $PID1 $PID2"
echo "================================================================================"
