# 🤖 AGENTS.md : Autonomous Agent & Node Operator Guide

This document specifies the protocols, APIs, and execution lifecycles for autonomous AI agents, automated miners, TEE validators, and node operators interacting with the **DePEFT Decentralized Parameter-Efficient Fine-Tuning Network**.

---

## 🏛️ Autonomous Agent Roles in DePEFT

DePEFT operates as an autonomous multi-agent decentralized economy with five primary agent personas:

```mermaid
graph TD
    ClientAgent["🧑‍💻 Client Agent<br/>(Task Creator & Bounty Funder)"] -->|"CreateTask Transaction"| AppChainNode["🌐 App-Chain Node<br/>(Axum RPC & CometBFT Consensus)"]
    
    AppChainNode -->|"Gossip TaskSpec"| MinerAgent["⚡ Miner Agent<br/>(Candle QLoRA Trainer)"]
    MinerAgent -->|"CommitAdapter (Hash)"| AppChainNode
    MinerAgent -->|"RevealAdapter (SafeTensors + CID)"| IPFS["📦 IPFS Swarm / CAS"]
    
    AppChainNode -->|"Gossip Candidate Adapters"| ValidatorAgent["🔒 Validator Agent<br/>(TEE Sandbox + Hardware Quote)"]
    IPFS -->|"Fetch .safetensors"| ValidatorAgent
    ValidatorAgent -->|"SubmitEvaluation + AttestationQuote"| AppChainNode
    
    AppChainNode -->|"Relative Consensus (Borda Count)"| ReLoRAEngine["🧬 ReLoRA Fusion<br/>(W_N+1 = W_N + ΔW*)"]
    ReLoRAEngine -->|"Bounty Payout & Weight Checkpoint"| ClientAgent
```

| Agent Persona | Responsibility | Compute Profile | Cryptographic Role |
|---|---|---|---|
| **Client Agent** | Creates tasks, specifies base model & target modules, deposits bounty escrow | CPU / Lightweight | Signs `CreateTask` via Ed25519 |
| **Miner Agent** | Downloads base model, runs QLoRA autograd, commits/reveals adapters | GPU (CUDA/ROCm) / High-perf CPU | Signs `CommitAdapter` & `RevealAdapter` |
| **Validator Agent** | Runs evaluation in hardware TEE sandbox, benchmarks candidate models, ranks miners | Intel SGX / AMD SEV-SNP / AWS Nitro | Generates `AttestationQuote` & signs `SubmitEvaluation` |
| **Consensus Validator** | Proposes blocks, votes in 2-phase BFT commit, verifies TEE quotes | High-availability Node | Signs `PREVOTE` & `PRECOMMIT` |
| **Storage Gateway** | Pins datasets and `.safetensors` model weights to global IPFS network | IPFS Kubo Daemon / Filecoin | Resolves Content Identifiers (CIDs) |

---

## 🔑 Keypair & Identity Management for Autonomous Agents

Autonomous agents generate and maintain Ed25519 cryptographic keypairs for transaction authentication and balance management.

### Agent Keypair Generation
```bash
cargo run -- key generate
```
Example Output:
```json
{
  "account_id": "0x404bb34320987b91769156755149dc7d052ade207e198343169bbe7838dcc007",
  "public_key": "0x404bb34320987b91769156755149dc7d052ade207e198343169bbe7838dcc007",
  "secret_key": "0x1a8f9c2d..."
}
```

Agents store their `secret_key` in secure environment variables:
```bash
export DEPEFT_AGENT_KEY="0x1a8f9c2d..."
export DEPEFT_NODE_URL="http://127.0.0.1:8545"
```

---

## 📡 HTTP REST / JSON-RPC Agent API Endpoints

Autonomous agents communicate with any live App-Chain Node over HTTP/JSON-RPC (default: `http://127.0.0.1:8545`).

### 1. Node Health & Chain Status
- **`GET /api/v1/status`**
  ```json
  {
    "block_height": 142,
    "total_tasks": 12,
    "active_rounds": 3,
    "connected_p2p_peers": 8
  }
  ```

### 2. Query Account Balance
- **`GET /api/v1/balance/{account_id}`**
  ```json
  {
    "account_id": "0x404bb343...",
    "balance": 50000
  }
  ```

### 3. Task Management Endpoints
- **`GET /api/v1/tasks`**: Returns array of all active [`TaskSpec`](file:///home/khoa/DePEFT/src/blockchain/types.rs#L45-L65) records.
- **`GET /api/v1/tasks/{task_id}`**: Retrieves task configuration, dataset CID, base model ID, and current round state.

### 4. Signed Transaction Submission
- **`POST /api/v1/tx`**
  - Payload: [`SignedTransaction`](file:///home/khoa/DePEFT/src/crypto/keys.rs#L70-L80)
  ```json
  {
    "tx": {
      "CreateTask": {
        "client": "0x404bb343...",
        "base_model_id": [81, 119, 101, 110, 47, 81, 119, 101, 110, 50, 46, 53, 45, 55, 66],
        "base_model_hash": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "dataset_cid": [98, 97, 102, 121, 98, 101, 105...],
        "peft_method": "QLoRA_NF4",
        "max_rank": 16,
        "target_modules": [[113, 95, 112, 114, 111, 106], [118, 95, 112, 114, 111, 106]],
        "bounty_pool": 30000,
        "epoch_blocks": 50
      }
    },
    "sender_public_key": [64, 75, 179, 67...],
    "signature": [120, 40, 199, 145...]
  }
  ```

### 5. Content-Addressable Storage (CAS)
- **`POST /api/v1/storage`**: Uploads raw binary (multipart form or binary body), returns multihash CID.
- **`GET /api/v1/storage/{cid}`**: Downloads raw `.safetensors` model weights or dataset chunks.

---

## ⚡ Miner Agent Execution Loop

An automated Miner Agent performs the following steps during each ReLoRA tournament round:

```mermaid
sequenceDiagram
    autonumber
    actor M as Miner Agent
    participant N as App-Chain Node
    participant CAS as Hybrid Storage (IPFS)
    
    N->>M: Gossip TaskSpec (Base Model CID + Dataset CID)
    M->>CAS: Download Base Model W_N & Training Split
    M->>M: Execute Candle QLoRA Autograd Training (SGD/Adam)
    M->>M: Export Adapter to SafeTensors & Generate Salt
    M->>M: Compute commit_hash = SHA256(adapter_hash || salt)
    M->>N: Broadcast SignedTransaction::CommitAdapter
    Note over M,N: Wait for Reveal Phase
    M->>CAS: Upload adapter.safetensors -> Receive CID
    M->>N: Broadcast SignedTransaction::RevealAdapter(CID, salt)
```

### CLI Command for Miner Agent:
```bash
cargo run -- miner run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_miner_secret_key \
  --task-id 1 \
  --hardware "NVIDIA RTX 4090 / CUDA"
```

---

## 🔒 Validator Agent Execution Loop

An automated Validator Agent operates inside a protected Trusted Execution Environment (TEE) sandbox to evaluate candidate adapters against private test sets:

```mermaid
sequenceDiagram
    autonumber
    actor V as Validator Agent (TEE)
    participant N as App-Chain Node
    participant CAS as Hybrid Storage (IPFS)
    
    N->>V: Gossip Revealed Adapters list [(miner_1, cid_1), (miner_2, cid_2)]
    V->>CAS: Fetch .safetensors for each candidate
    V->>V: Load base model W_N + apply candidate adapter inside TEE Enclave
    V->>V: Evaluate cross-entropy loss & perplexity on Private Test Set
    V->>V: Compute ordinal ranking: [miner_alpha > miner_beta > miner_gamma]
    V->>V: Generate Hardware Attestation Quote (MRENCLAVE, report_data)
    V->>N: Broadcast SignedTransaction::SubmitEvaluation(ranking, quote)
```

### CLI Command for Validator Agent:
```bash
cargo run -- validator run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_validator_secret_key \
  --task-id 1 \
  --hardware "Intel Xeon / AVX-512"
```

---

## 🛡️ Byzantine Fault Tolerant (BFT) Consensus Validator Agent

Consensus Validators participate in block proposing and 2-phase commit voting:

1. **PROPOSE**: Designated proposer selects valid mempool transactions and broadcasts a candidate block.
2. **PREVOTE**: Validators verify block hash, state root, and TEE attestation quotes, signing `VoteType::Prevote`.
3. **PRECOMMIT**: Upon receiving $> 2/3$ Prevotes (Proof-of-Lock), validators sign `VoteType::Precommit`.
4. **COMMIT**: Once $> 2/3$ Precommits are gathered, the block is finalized and appended to the immutable blockchain.

### CLI Command for BFT Consensus Testing:
```bash
cargo run -- bft-demo --validators 4 --blocks 5
```

---

## 🤖 CI / CD & Automated Bot Workflows

For continuous benchmarking and headless node operation in headless Docker or Kubernetes clusters:

```yaml
# Example Kubernetes Pod definition for DePEFT Miner Node
apiVersion: v1
kind: Pod
metadata:
  name: depeft-miner-worker
spec:
  containers:
  - name: depeft-miner
    image: depeft-node:latest
    command: ["cargo", "run", "--release", "--", "miner", "run"]
    args:
      - "--node-url=http://depeft-validator-node:8545"
      - "--secret-key=$(DEPEFT_MINER_KEY)"
      - "--task-id=1"
      - "--hardware=NVIDIA A100 / CUDA 12.2"
    resources:
      limits:
        nvidia.com/gpu: 1
```
