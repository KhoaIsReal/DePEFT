import requests
import json
from typing import Dict, Any, List, Optional

class DePeftClient:
    """Python Client for communicating with any DePEFT App-Chain Node."""

    def __init__(self, node_url: str = "http://127.0.0.1:8545"):
        self.node_url = node_url.rstrip("/")

    def get_status(self) -> Dict[str, Any]:
        """Fetch node health, block height, and active tournament rounds."""
        resp = requests.get(f"{self.node_url}/api/v1/status", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Retrieve all active fine-tuning tasks on the network."""
        resp = requests.get(f"{self.node_url}/api/v1/tasks", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_task(self, task_id: int) -> Dict[str, Any]:
        """Retrieve specific task configuration and parameters by ID."""
        resp = requests.get(f"{self.node_url}/api/v1/tasks/{task_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_balance(self, account_id: str) -> int:
        """Query account balance of $DEPEFT tokens."""
        resp = requests.get(f"{self.node_url}/api/v1/accounts/{account_id}/balance", timeout=5)
        resp.raise_for_status()
        return resp.json().get("balance", 0)

    def request_faucet(self, account_id: str, amount: int = 10000) -> Dict[str, Any]:
        """Request testnet $DEPEFT tokens from on-chain node faucet."""
        resp = requests.post(
            f"{self.node_url}/api/v1/faucet",
            json={"account": account_id, "amount": amount},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def upload_dataset(self, data: bytes) -> str:
        """Upload dataset or model artifact to Content-Addressable Storage (CAS)."""
        hex_data = data.hex()
        resp = requests.post(
            f"{self.node_url}/api/v1/storage",
            json={"data_hex": hex_data},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("cid")

    def download_artifact(self, cid: str) -> bytes:
        """Download raw SafeTensors adapter or dataset by CID."""
        resp = requests.get(f"{self.node_url}/api/v1/storage/{cid}", timeout=30)
        resp.raise_for_status()
        return resp.content

    def get_peers(self) -> List[str]:
        """List all connected P2P overlay swarm peers."""
        resp = requests.get(f"{self.node_url}/api/v1/p2p/peers", timeout=5)
        resp.raise_for_status()
        return resp.json().get("connected_peers", [])
