# 🌐 DePEFT: Decentralized Parameter-Efficient Fine-Tuning
### ReLoRA Multi-Round Tournament Protocol on an Application-Specific Blockchain

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](./LICENSE)

An end-to-end, production-grade Rust implementation of the **DePEFT Architecture Design** specification for decentralized AI fine-tuning using Hugging Face Candle QLoRA, Hardware TEE Remote Attestation, IPFS decentralized storage, and CometBFT 2-phase commit consensus.

---

## 📚 Documentation Index

- **[ARCHITECTURE.md](./ARCHITECTURE.md)**: Deep-dive whitepaper specification of all 4 layers, ReLoRA weight fusion, Borda count aggregation, and CometBFT 2-phase commit.
- **[AGENTS.md](./AGENTS.md)**: Autonomous agent guide, node operator manual, REST/JSON-RPC API schema, and miner/validator execution loops.
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**: Contribution guidelines, toolchain setup, architectural invariants, code standards, and PR workflows.
- **[SECURITY.md](./SECURITY.md)**: Threat modeling, anti-fraud guarantees, TEE attestation verification, and vulnerability disclosure policy.

---

## 🏛️ Architecture Overview (4 Core Layers)

```
                       ┌─────────────────────────────────────────────────────────┐
                       │          Client (Creates Task & Escrows Bounty)         │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Consensus & State Layer (App-Chain State Machine & CometBFT)                                         │
│    • Deterministic State Machine (Escrow, Balances, TaskSpec Registry, Faucet API)                      │
│    • Tendermint/CometBFT 2-Phase Commit Consensus (Propose -> Prevote -> Precommit -> Commit)          │
│    • Relative Consensus Engine (Borda Count Rank Aggregation & Floating-Point Drift Resistance)         │
│    • On-Chain Hardware TEE Attestation Verifier (MRENCLAVE / MRSIGNER Whitelist Checking)              │
└──────────────────────┬────────────────────────────────────────────────────▲─────────────────────────────┘
                       │ TaskSpec ($W_N$, CID)                               │ Consensus Ranking & Payout
                       ▼                                                    │
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ 2. Miner Network Layer (Heterogeneous Compute)  │   │ 3. Validator Layer (Off-Chain Workers & TEE)    │
│    • Hugging Face Candle Autograd PEFT Engine   │   │    • Isolated Hardware TEE Enclave (SGX / SEV)   │
│    • QLoRA Matrix Updates ($\Delta W = B \times A$)  │   │    • Floating-point Drift Tolerance (BF16/FP16) │
│    • SafeTensors Binary Packaging               │   │    • Hardware Attestation Quote Generation      │
│    • SHA-256 Commit-Reveal Hashing & Salt       │   │    • Vector DB Plagiarism & Collision Indexing  │
└──────────────────────┬──────────────────────────┘   └─────────────────────▲───────────────────────────┘
                       │                                                    │
                       │ Publish $\Delta W$ .safetensors                     │ Pull & Verify $\Delta W$
                       ▼                                                    │
┌───────────────────────────────────────────────────────────────────────────┴─────────────────────────────┐
│ 4. Storage & Database Layer                                                                             │
│    • Live IPFS Kubo RPC Integration (/api/v0/add, /api/v0/cat, /api/v0/pin)                             │
│    • HybridStorageManager (Local Disk CAS Cache ~/.depeft/storage + Global IPFS Swarm)                  │
│    • SafeTensors Standard Serialization / Deserialization                                               │
│    • Embedded Lightweight Vector Database (Adapter Signature Indexing & Cosine Similarity)              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Web Explorer & Real-Time Management Dashboard

DePEFT comes with a built-in single-file Web Explorer and management portal served directly by any node at `http://127.0.0.1:8545/`:
- **Real-Time Metrics**: Current Block Height, Active PEFT Tournaments, CAS Storage Artifacts, and Connected P2P Peers.
- **Task Browser**: Inspect registered Base Models (`Qwen2.5-7B`, `LLaMA-3`), LoRA configurations, and escrowed bounty balances.
- **🚰 On-Chain Testnet Faucet**: Instantly claim 10,000 $DEPEFT testnet tokens to fund or simulate tasks.
- **Interactive JSON Console**: Query REST/JSON-RPC node state directly from the browser.

---

## 📦 Project Structure

```
DePEFT/
├── contracts/                       # Layer 1: EVM & App-Chain Smart Contracts
│   ├── DePeftToken.sol              # Standard ERC-20 Network Token ($DEPEFT)
│   ├── DePeftEscrow.sol             # Task Bounty Escrow & Automated Winner Settlement
│   └── scripts/deploy.js            # Hardhat / Node Deployment Script
├── sdk/                             # Developer SDKs & AI Tooling
│   └── python/                      # Python Client Library (depeft) for PyTorch & HF Candle
│       ├── depeft/
│       │   ├── client.py            # DePeftClient (Task management, Storage, Faucet, RPC)
│       │   ├── crypto.py            # Ed25519 key generation and transaction signing
│       │   └── __init__.py
│       └── setup.py
├── scripts/                         # Automation & Network Orchestration
│   └── start_local_testnet.sh       # 1-Click Multi-Node Local P2P Testnet Launcher
├── src/                             # Core Rust App-Chain Protocol
│   ├── lib.rs                       # Crate root exporting all layers
│   ├── main.rs                      # Comprehensive CLI, Node Daemon & Tournament Simulator
│   ├── consensus/                   # CometBFT 2-Phase Commit Engine
│   ├── tee/                         # Hardware TEE Remote Attestation Engine
│   ├── candle_peft/                 # Real Deep Learning LLM Engine with Candle
│   ├── p2p/                         # P2P Overlay Network & Gossip Protocol
│   ├── crypto/                      # Ed25519 Cryptography & Signed Transactions
│   ├── node/                        # Live App-Chain HTTP REST/JSON-RPC Node & Web Dashboard
│   │   ├── dashboard.rs             # Embedded HTML/CSS/JS Web Explorer
│   │   └── server.rs                # Axum REST & Faucet API
│   ├── client/                      # Rust RPC Client
│   ├── blockchain/                  # App-Chain State Machine & Relative Consensus
│   ├── miner/                       # Miner Network Layer & Autograd Worker
│   ├── validator/                   # Validator Layer & TEE Evaluation Worker
│   ├── storage/                     # Storage Layer (Hybrid CAS, IPFS, Vector DB)
│   ├── ml/                          # Mathematical PEFT Primitives (NF4, INT4)
│   └── tournament/                  # ReLoRA Multi-Round Tournament Engine
├── tests/
│   └── integration_tests.rs         # 16 Comprehensive Integration & Protocol Tests
└── .github/workflows/ci.yml         # Automated GitHub Actions CI Testing & Build Pipeline
```

---

## 🚀 Quickstart & Real Network Operations

### 1. Launch Multi-Node Local Testnet (1-Click)
Deploy a full 2-node P2P testnet swarm with bootstrap consensus and live web dashboard:
```bash
./scripts/start_local_testnet.sh
```
Open **http://127.0.0.1:8545** in your browser to access the Web Explorer.

---

### 2. Use the Python SDK (`depeft`)
Interact with the live network from Python scripts or Jupyter Notebooks:
```python
from depeft import DePeftClient, generate_keypair

client = DePeftClient("http://127.0.0.1:8545")

# 1. Request testnet tokens from faucet
account = generate_keypair()
print(client.request_faucet(account["account_id"], amount=10000))

# 2. Check balance & chain status
print("Status:", client.get_status())
print("Balance:", client.get_balance(account["account_id"]))
```

---

### 3. Run Real Candle LLM Transformer ReLoRA Tournament
Fine-tune real LoRA adapters on a Decoder Transformer language model with autograd, HuggingFace `.safetensors` export, TEE evaluation, and ReLoRA weight fusion:
```bash
cargo run -- llm-demo --rounds 3 --steps 10
```

---

### 4. Run Byzantine Fault Tolerant (BFT) State Finality & Consensus
Execute Tendermint/CometBFT 2-phase commit with 2/3+ validator quorum and equivocation slashing:
```bash
cargo run -- bft-demo --validators 4 --blocks 3
```

---

### 5. Deploy EVM Smart Contracts ($DEPEFT & Escrow)
Deploy smart contracts to Sepolia or Base Sepolia testnet using Hardhat:
```bash
cd contracts
npm install
cp .env.example .env # Configure PRIVATE_KEY
npm run deploy:sepolia
# or deploy to Base Sepolia
npm run deploy:base-sepolia
```

---

### 6. Deploy 1-Click Public App-Chain Testnet to VPS / Cloud
Deploy a live public testnet node with IPFS Kubo storage on any cloud instance:
```bash
./scripts/deploy_vps_testnet.sh
```

---

## 🧪 Comprehensive Test Suite

Run the full integration test suite covering all cryptographic, ML, TEE, IPFS, BFT, and P2P layers:

```bash
cargo test
```

```text
running 16 tests
test test_embedded_vector_db_similarity_and_plagiarism ... ok
test test_commit_reveal_anti_collusion_verification ... ok
test test_relative_consensus_borda_aggregation ... ok
test test_disk_ipfs_storage_persistence ... ok
test test_task_spec_data_structure ... ok
test test_safetensors_serialization_roundtrip ... ok
test test_bft_consensus_2_phase_commit_and_equivocation_slashing ... ok
test test_nf4_quantization_and_dequantization ... ok
test test_candle_lora_linear_forward_and_merge ... ok
test test_ed25519_cryptographic_signing_and_verification ... ok
test test_hardware_tee_remote_attestation_and_on_chain_verification ... ok
test test_hybrid_storage_and_ipfs_cas_caching ... ok
test test_live_node_http_rpc_integration ... ok
test test_p2p_swarm_bidirectional_gossip_and_deduplication ... ok
test test_relora_tournament_multi_round_convergence ... ok
test test_candle_llm_transformer_relora_tournament ... ok

test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.70s
```

---

## 📄 Licensing & Open Source Permissions

DePEFT uses a modular multi-license architecture to protect core blockchain innovation while maximizing developer adoption:

- **Core Node & Consensus Engine (`src/`, `Cargo.toml`)**: Licensed under the **[GNU General Public License v3.0 (GPL-3.0)](./LICENSE)**. Any forks or network distributions of the core node must remain open-source.
- **Python Client SDK (`sdk/python/`)**: Dual-licensed under **[Apache-2.0](./sdk/python/LICENSE-APACHE)** OR **[MIT](./sdk/python/LICENSE-MIT)**, allowing external AI developers and agents to freely integrate DePEFT into proprietary or open workflows.
- **Smart Contracts (`contracts/`)**: Dual-licensed under **[Apache-2.0](./contracts/LICENSE-APACHE)** OR **[MIT](./contracts/LICENSE-MIT)** for permissive EVM deployment and DApp integration.


