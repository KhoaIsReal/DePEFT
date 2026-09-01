#!/usr/bin/env bash
set -e

# ==============================================================================
# DePEFT VPS 1-Click Testnet Deployment & Bootstrap Script
# ==============================================================================

echo "================================================================================"
echo "           🚀 DePEFT VPS Testnet Automated Setup & Deployment                  "
echo "================================================================================"

# 1. Check Docker & Docker Compose
if ! command -v docker &> /dev/null; then
    echo "[!] Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
fi

if ! docker compose version &> /dev/null; then
    echo "[!] Docker Compose plugin missing. Please install docker-compose-plugin."
fi

# 2. Get Public IP
PUBLIC_IP=$(curl -s ifconfig.me || echo "127.0.0.1")
echo "[*] Public IP detected: $PUBLIC_IP"

# 3. Open Firewall Ports if UFW is active
if command -v ufw &> /dev/null && sudo ufw status | grep -q "Status: active"; then
    echo "[*] Configuring UFW firewall rules..."
    sudo ufw allow 8545/tcp comment 'DePEFT RPC'
    sudo ufw allow 9000/tcp comment 'DePEFT P2P Swarm'
    sudo ufw allow 4001/tcp comment 'IPFS Swarm'
    echo "✓ Ports 8545, 9000, 4001 opened."
fi

# 4. Launch Containers
echo "[*] Starting DePEFT Node & IPFS Kubo Storage daemon..."
docker compose down -v 2>/dev/null || true
docker compose up -d --build

# 5. Wait for Node health check
echo "[*] Waiting for Node RPC to become healthy..."
sleep 5

MAX_RETRIES=10
COUNT=0
HEALTHY=false

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://127.0.0.1:8545/api/v1/status" | grep -q "block_height"; then
        HEALTHY=true
        break
    fi
    echo "    Waiting for RPC response ($((COUNT+1))/$MAX_RETRIES)..."
    sleep 3
    COUNT=$((COUNT+1))
done

if [ "$HEALTHY" = true ]; then
    echo "================================================================================"
    echo "🎉 DePEFT Public Testnet is LIVE & RUNNING!"
    echo "================================================================================"
    echo "  - RPC API: http://$PUBLIC_IP:8545/api/v1/status"
    echo "  - P2P Gossip: $PUBLIC_IP:9000"
    echo "  - IPFS Gateway: http://$PUBLIC_IP:8080"
    echo ""
    echo "💡 How to connect an external Miner:"
    echo "  cargo run -- miner run --node-url http://$PUBLIC_IP:8545 --task-id 1"
    echo "================================================================================"
else
    echo "[!] Node failed to respond in time. Check logs with: docker compose logs -f"
fi
