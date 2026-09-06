# 🌐 DePEFT: Decentralized Parameter-Efficient Fine-Tuning
### ReLoRA Multi-Round Tournament Protocol on an Application-Specific Blockchain

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)]()
[![Clippy](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-blue.svg)](./LICENSE)

> [!WARNING]
> ### ⚠️ SAFETY WARNING / SCAM ALERT
> **CURRENTLY IN TESTNET PHASE. IF ANYONE CLAIMS DEPEFT HAS LAUNCHED ON MAINNET, IT IS A SCAM!**
>
> The **DePEFT project is currently in active Testnet development**. We have **NOT** launched any token or smart contract on Mainnet (Ethereum, Base, Solana, BNB Chain, or any DEX/CEX). Any token claiming to be DePEFT on mainnet is an unauthorized counterfeit scam.

**DePEFT** is an open-source decentralized network that allows anyone to fund, train, and evaluate Large Language Models (LLMs) collectively without trusting a centralized AI cloud provider. 

Instead of full model retraining (which costs millions of dollars), DePEFT uses **Parameter-Efficient Fine-Tuning (PEFT / QLoRA)**: miners only train tiny low-rank adapter matrices ($\Delta W$). Across successive tournament epochs, optimal adapters are verified inside **Hardware TEE Enclaves**, ranked via **Borda Count Consensus**, and permanently merged into the base model weights ($W_{N+1} = W_N + \Delta W^*$).

---

## 📖 Table of Contents
1. [How DePEFT Works in Plain English](#-how-depeft-works-in-plain-english)
2. [Network Roles](#-network-roles)
3. [Complete Step-by-Step Usage Guide](#-complete-step-by-step-usage-guide)
   - [Step 1: Installation & Building](#step-1-installation--building)
   - [Step 2: Generate Cryptographic Account Keypairs](#step-2-generate-cryptographic-account-keypairs)
   - [Step 3: Start an App-Chain Node Daemon](#step-3-start-an-app-chain-node-daemon)
   - [Step 4: Claim Free Testnet Tokens (Faucet)](#step-4-claim-free-testnet-tokens-faucet)
   - [Step 5: Create a Fine-Tuning Task (Client)](#step-5-create-a-fine-tuning-task-client)
   - [Step 6: Run a Decentralized Miner (Miner)](#step-6-run-a-decentralized-miner-miner)
   - [Step 7: Run a TEE Evaluator (Validator)](#step-7-run-a-tee-evaluator-validator)
   - [Step 8: Inspect Consensus & P2P Swarm](#step-8-inspect-consensus--p2p-swarm)
4. [Python SDK Guide](#-python-sdk-guide)
5. [HTTP REST / JSON-RPC API Reference](#-http-rest--json-rpc-api-reference)
6. [Interactive Demos & Simulations](#-interactive-demos--simulations)
7. [Architecture & 4 Core Layers](#-architecture--4-core-layers)
8. [Testing & Quality Verification](#-testing--quality-verification)
9. [⚡ Fast Q&A for Crypto Traders & Node Operators (FAQ)](./FAST_Q&A.md)
10. [License](#-license)

---

## 💡 How DePEFT Works in Plain English

```
  1. Client Agent deposits bounty & creates a Task on-chain
                         │
                         ▼
  2. Miners train QLoRA adapters locally (CUDA / ROCm / CPU)
                         │
                         ▼
  3. Miners submit Commit Hash: SHA256(adapter_hash || salt) [Anti-Plagiarism]
                         │
                         ▼
  4. Miners reveal adapter on CAS / IPFS (.safetensors)
                         │
                         ▼
  5. Validators evaluate adapters inside Hardware TEE Enclave on Private Test Set
                         │
                         ▼
  6. Validators submit Attestation Quote + Ranking -> Borda Count Relative Consensus
                         │
                         ▼
  7. Winning weights merged into Base Model: W_N+1 = W_N + ΔW* & Bounty Released
```

1. **Task Escrow**: A Client deposits tokens ($DEPEFT) into escrow and defines the task (e.g., base model `Qwen/Qwen2.5-7B`, target modules `q_proj, v_proj`, and training dataset CID).
2. **Commit-Reveal Tournament**:
   - **Commit Phase**: Miners download the base weights, train a QLoRA adapter locally, and post a cryptographic hash `SHA256(adapter || salt)`. No one can copy their work before deadline.
   - **Reveal Phase**: Miners upload their `.safetensors` adapter file to IPFS / Content-Addressable Storage (CAS) and reveal the salt.
3. **TEE Private Evaluation**: Validators load the candidate adapters into secure hardware enclaves (Intel SGX or AMD SEV-SNP) and score them on a **private test set** that no miner can see.
4. **Borda Count Consensus**: Validators rank miners from best to worst. Because different GPUs produce minor floating-point precision drifts (BF16 vs FP16), the App-Chain aggregates **ordinal rankings** (Borda Count) rather than raw loss floats.
5. **ReLoRA Fusion**: The winning adapter is permanently fused into the base model weights, the winner receives the bounty from escrow, and the next round begins on the newly evolved model ($W_1, W_2, ...$).

---

## 👥 Network Roles

| Role | What They Do | Requirements | Command |
|---|---|---|---|
| **Node Operator** | Runs the blockchain daemon, stores state in SQLite, relays P2P transactions, hosts Web Dashboard | 2 vCPU, 4GB RAM | `depeft node start` |
| **Client / Task Creator** | Funds tasks with bounties, provides datasets & target base model | Any machine | `depeft task create` |
| **Miner** | Downloads base models, trains low-rank adapters, earns bounties | GPU (CUDA/ROCm) or CPU | `depeft miner run` |
| **TEE Validator** | Evaluates candidate adapters inside hardware enclave, signs attestation quotes | Intel SGX / AMD SEV / AWS Nitro (or Sim mode) | `depeft validator run` |

---

## 🛠️ Complete Step-by-Step Usage Guide

### Step 1: Installation & Building

Make sure you have Rust 1.80+ installed:
```bash
git clone https://github.com/KhoaIsReal/DePEFT.git
cd DePEFT
cargo build --release
```
The compiled executable is located at `target/release/depeft`. You can also run commands directly with `cargo run --`.

---

### Step 2: Generate Cryptographic Account Keypairs

All transactions on DePEFT (creating tasks, submitting commits, voting) are signed with **Ed25519** cryptographic keypairs.

Generate a new keypair:
```bash
cargo run -- key generate
```
Output:
```text
=== Generated New DePEFT Ed25519 Account Keypair ===
Public Address: 0x404bb343c6827a58a983b6329e4720970b8c95a0
Public Key:     0x280e227092147743d548325ebef50beadca2e4cbe45bfefba3ca87884ecb510a
Secret Key:     0x8e833b5c33feff4cf19cb8d67ec1bb84c59d9f5cb68b5a8370f1a9d18ce58826
```

Inspect an existing secret key:
```bash
cargo run -- key inspect 0x8e833b5c33feff4cf19cb8d67ec1bb84c59d9f5cb68b5a8370f1a9d18ce58826
```

---

### Step 3: Start an App-Chain Node Daemon

#### Option A: Start a local node for development / testnet (with simulator TEE & faucet)
```bash
cargo run -- node start \
  --port 8545 \
  --p2p-port 9000 \
  --testnet-tee-sim \
  --enable-operator-endpoints \
  --enable-faucet
```
Key flags:
- `--testnet-tee-sim`: Automatically whitelists the official testnet simulator TEE measurement and hardware keys so validators can run without dedicated Intel SGX hardware.
- `--enable-operator-endpoints`: Enables `POST /api/v1/storage` so miners can upload `.safetensors` adapters to the node.
- `--enable-faucet`: Enables `POST /api/v1/faucet` for testnet token distribution.

#### Option B: 1-Click Multi-Node Local Swarm
To test a real multi-node network with 2 connected peers:
```bash
./scripts/start_local_testnet.sh
```
- **Node 1 (Bootstrap)**: `http://127.0.0.1:8545` (P2P: `9000`)
- **Node 2 (Peer)**: `http://127.0.0.1:8546` (P2P: `9001`)
- **Web Dashboard**: Open `http://127.0.0.1:8545` in your browser!

---

### Step 4: Claim Free Testnet Tokens (Faucet)

Before creating tasks or interacting on testnet, claim tokens:

Via cURL:
```bash
curl -X POST http://127.0.0.1:8545/api/v1/faucet \
  -H "Content-Type: application/json" \
  -d '{"account": "0x404bb343c6827a58a983b6329e4720970b8c95a0", "amount": 50000}'
```

Check balance:
```bash
curl http://127.0.0.1:8545/api/v1/accounts/0x404bb343c6827a58a983b6329e4720970b8c95a0/balance
```

---

### Step 5: Create a Fine-Tuning Task (Client)

Clients deposit escrow tokens and specify the model architecture:

```bash
cargo run -- task create \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_your_client_secret_key \
  --model-id "Qwen/Qwen2.5-7B" \
  --bounty 30000 \
  --top-k 3 \
  --merge single
```
Parameters:
- `--model-id`: Target Hugging Face base model identifier.
- `--bounty`: Escrow tokens locked for the tournament round.
- `--top-k`: Number of winning miners sharing rewards (e.g. `1` for Winner-Takes-All, or `3`/`5` for exponential decay).
- `--merge`: Weight fusion strategy (`single` for Top 1, or `ensemble` for weighted multi-miner merge).

List all active tasks:
```bash
cargo run -- task list --node-url http://127.0.0.1:8545
```

---

### Step 6: Run a Decentralized Miner (Miner)

A Miner listens for active tasks, executes autograd QLoRA training on available compute hardware, submits the commit hash, uploads the `.safetensors` adapter, and reveals:

```bash
cargo run -- miner run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_your_miner_secret_key \
  --task-id 1 \
  --device auto \
  --lr 0.03
```
Supported `--device` targets:
- `auto`: Automatically selects the fastest available backend (CUDA $\to$ ROCm $\to$ Metal $\to$ WGPU $\to$ CPU).
- `cuda`: NVIDIA GPUs with cuDNN acceleration.
- `rocm`: AMD Radeon GPUs with ROCm HIP.
- `cpu`: High-performance multithreaded CPU with AVX-512 vectorization.

---

### Step 7: Run a TEE Evaluator (Validator)

A Validator runs inside a Trusted Execution Environment (TEE). It fetches revealed candidate adapters, benchmarks them on a protected private dataset, generates a hardware attestation quote, and submits the ranking vote on-chain:

```bash
cargo run -- validator run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_your_validator_secret_key \
  --task-id 1
```

---

### Step 8: Inspect Consensus & P2P Swarm

Inspect connected P2P peers:
```bash
cargo run -- p2p peers --node-url http://127.0.0.1:8545
```

Connect node to a remote peer manually:
```bash
cargo run -- p2p connect --node-url http://127.0.0.1:8545 --addr "127.0.0.1:9001"
```

Check IPFS Kubo daemon status:
```bash
cargo run -- ipfs status --api-url http://127.0.0.1:5001
```

---

## 🐍 Python SDK Guide

The official Python client library is located in `sdk/python/`.

### Installation
```bash
cd sdk/python
pip install -e .
```

### Complete Python Workflow
```python
from depeft import DePeftClient, generate_keypair
import time

# 1. Connect to node
client = DePeftClient("http://127.0.0.1:8545")

# 2. Generate a keypair for your bot / script
account = generate_keypair()
print(f"Address: {account['account_id']}")
print(f"Secret:  {account['secret_key']}")

# 3. Request faucet funds
client.request_faucet(account["account_id"], 20000)

# 4. Check chain status and balance
status = client.get_status()
balance = client.get_balance(account["account_id"])
print(f"Block #{status['block_height']} | Tasks: {status['tasks_count']} | Balance: {balance} $DEPEFT")

# 5. Query active tasks
tasks = client.get_tasks()
for task in tasks:
    print(f"Task #{task['task_id']}: Model={task['base_model_id_str']} | Bounty={task['bounty_pool']}")
```

---

## 📡 HTTP REST / JSON-RPC API Reference

All DePEFT nodes expose a JSON API at `http://127.0.0.1:8545`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/status` | Current block height, active task count, CAS objects, peers count |
| `GET` | `/api/v1/tasks` | Array of all registered `TaskSpec` objects |
| `GET` | `/api/v1/tasks/:id` | Detailed task specification by ID |
| `GET` | `/api/v1/tasks/:id/rounds/:round` | Round status, current phase (Commit/Reveal/Evaluation/Merge), revealed miners |
| `POST` | `/api/v1/tx` | Submit signed transaction (`CreateTask`, `CommitAdapter`, `RevealAdapter`, `SubmitEvaluation`) |
| `GET` | `/api/v1/accounts/:account/balance` | Query token balance and anti-replay nonce |
| `POST` | `/api/v1/faucet` | Claim testnet tokens (disabled in production unless `--enable-faucet`) |
| `POST` | `/api/v1/storage` | Upload raw binary artifact (Hex encoded), returns CID |
| `GET` | `/api/v1/storage/:cid` | Download binary `.safetensors` weights or dataset |
| `GET` | `/api/v1/p2p/peers` | List all connected P2P gossip peers |
| `POST` | `/api/v1/p2p/connect` | Request node to dial a remote TCP peer |

---

## 🎮 Interactive Demos & Simulations

DePEFT includes rich built-in simulation tools to test every architecture layer without setting up multi-machine networks:

### 1. Full 4-Layer ReLoRA Tournament Simulation
Runs 3 complete tournament rounds showing all 4 layers with synthetic datasets, commit-reveal, Borda counting, and weight evolution:
```bash
cargo run -- demo --rounds 3 --peft qlora-nf4 --miners 3 --validators 3
```

### 2. Real Hugging Face Candle Transformer Training
Trains a real Decoder Transformer language model with autograd, backpropagation, and Hugging Face `.safetensors` weight merge:
```bash
cargo run -- llm-demo --rounds 3 --steps 15 --device auto
```

### 3. CometBFT Consensus & Slashing Demonstration
Simulates a 4-node BFT committee committing blocks and slashing an equivocation validator:
```bash
cargo run -- bft-demo --validators 4 --blocks 5
```

### 4. Hardware TEE Remote Attestation Quote Demo
Generates a real hardware attestation quote inside Intel SGX DCAP / AMD SEV-SNP and verifies it against the on-chain state machine:
```bash
cargo run -- tee-quote
```

### 5. PEFT Quantization Compression Benchmark
Benchmarks memory footprint and Mean Squared Error (MSE) across FP32, QLoRA NF4 (4-bit NormalFloat), and INT4:
```bash
cargo run -- benchmark
```

---

## 🏛️ Architecture & 4 Core Layers

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Consensus & State Layer (App-Chain State Machine & CometBFT)                           │
│    • Deterministic Ledger: Token Balances, Nonces (Anti-Replay), Escrows, TaskSpecs        │
│    • Tendermint/CometBFT 2-Phase Commit: Propose -> Prevote -> Precommit -> Commit        │
│    • Borda Count Rank Consensus: Immune to GPU floating-point precision drift             │
│    • On-Chain TEE Verifier: Enforces MRENCLAVE / MRSIGNER and Platform Key Whitelists     │
└────────────────────────────┬──────────────────────────────────────▲───────────────────────┘
                             │                                      │
                             ▼                                      │
┌───────────────────────────────────────────┐      ┌────────────────┴───────────────────────┐
│ 2. Miner Layer (Heterogeneous Compute)    │      │ 3. Validator Layer (TEE Sandbox)       │
│    • Hugging Face Candle Deep Learning    │      │    • Private Test Set Evaluation       │
│    • QLoRA (NF4/INT4) Autograd Training   │      │    • Hardware Attestation Quotes       │
│    • SafeTensors Zero-Copy Export         │      │    • Vector DB Plagiarism Detection    │
│    • SHA-256 Commit-Reveal Hashing        │      │    • Relative Consensus Borda Ranking  │
└────────────────────────────┬──────────────┘      └────────────────▲───────────────────────┘
                             │                                      │
                             ▼                                      │
┌───────────────────────────────────────────────────────────────────┴───────────────────────┐
│ 4. Storage & Networking Layer                                                             │
│    • Hybrid CAS: Local SSD Cache (~/.depeft/storage) + Global IPFS Kubo Daemon Swarm      │
│    • Async TCP P2P Swarm with Gossip Deduplication and Address Rate-Limiting               │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing & Quality Verification

DePEFT is built with production rigor. Run the full test suite and strict lint checks:

```bash
# Run all 45 unit and integration tests
cargo test

# Ensure 0 warnings with strict clippy
cargo clippy --all-targets -- -D warnings
```

---

## 📄 License

- **Blockchain Core & Node (`src/`, `Cargo.toml`)**: [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE).
- **Python SDK (`sdk/python/`)**: Dual-licensed under [Apache-2.0](./sdk/python/LICENSE-APACHE) OR [MIT](./sdk/python/LICENSE-MIT).
- **Smart Contracts (`contracts/`)**: Dual-licensed under [Apache-2.0](./contracts/LICENSE-APACHE) OR [MIT](./contracts/LICENSE-MIT).
