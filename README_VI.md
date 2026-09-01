# 🌐 DePEFT: Decentralized Parameter-Efficient Fine-Tuning
### ReLoRA Multi-Round Tournament Protocol on an Application-Specific Blockchain

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](./LICENSE)

Bản cài đặt hoàn chỉnh viết bằng Rust cho đặc tả thiết kế **DePEFT Architecture Design** phục vụ decentralized AI fine-tuning sử dụng Hugging Face Candle QLoRA, Hardware TEE Remote Attestation, IPFS decentralized storage và CometBFT 2-phase commit consensus.

---

## 📚 Mục lục Tài liệu

- **[ARCHITECTURE_VI.md](./ARCHITECTURE_VI.md)** (Bản tiếng Anh: [ARCHITECTURE.md](./ARCHITECTURE.md)): Đặc tả chi tiết 4 layers, ReLoRA weight fusion, Borda count aggregation và CometBFT 2-phase commit.
- **[AGENTS.md](./AGENTS.md)**: Autonomous agent guide, node operator manual, REST/JSON-RPC API schema và execution loops của Miner / Validator.
- **[CONTRIBUTING_VI.md](./CONTRIBUTING_VI.md)** (Bản tiếng Anh: [CONTRIBUTING.md](./CONTRIBUTING.md)): Contribution guidelines, toolchain setup, architectural invariants, code standards và PR workflows.
- **[SECURITY_VI.md](./SECURITY_VI.md)** (Bản tiếng Anh: [SECURITY.md](./SECURITY.md)): Threat modeling, anti-fraud guarantees, TEE attestation verification và vulnerability disclosure policy.

---

## 🏛️ Tổng quan Kiến trúc (4 Core Layers)

```
                       ┌─────────────────────────────────────────────────────────┐
                       │          Client (Tạo Task & Escrow Bounty)              │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Consensus & State Layer (App-Chain State Machine & CometBFT)                                         │
│    • Máy trạng thái tất định (Escrow, Balances, Sổ đăng ký TaskSpec, Faucet API)                          │
│    • Tendermint/CometBFT 2-Phase Commit Consensus (Propose -> Prevote -> Precommit -> Commit)          │
│    • Relative Consensus Engine (Borda Count Rank Aggregation & Kháng floating-point drift)              │
│    • On-Chain Hardware TEE Attestation Verifier (Kiểm tra Whitelist MRENCLAVE / MRSIGNER)               │
└──────────────────────┬────────────────────────────────────────────────────▲─────────────────────────────┘
                       │ TaskSpec ($W_N$, CID)                               │ Consensus Ranking & Payout
                       ▼                                                    │
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ 2. Miner Network Layer (Heterogeneous Compute)  │   │ 3. Validator Layer (Off-Chain Workers & TEE)    │
│    • Hugging Face Candle Autograd PEFT Engine   │   │    • Isolated Hardware TEE Enclave (SGX / SEV)   │
│    • QLoRA Matrix Updates ($\Delta W = B \times A$)│   │    • Floating-point Drift Tolerance (BF16/FP16) │
│    • SafeTensors Binary Packaging               │   │    • Hardware Attestation Quote Generation      │
│    • SHA-256 Commit-Reveal Hashing & Salt       │   │    • Vector DB Plagiarism & Collision Indexing  │
└──────────────────────┬──────────────────────────┘   └─────────────────────▲───────────────────────────┘
                       │                                                    │
                       │ Publish $\Delta W$ .safetensors                     │ Pull & Verify $\Delta W$
                       ▼                                                    │
┌───────────────────────────────────────────────────────────────────────────┴─────────────────────────────┐
│ 4. Storage & Database Layer                                                                             │
│    • Live IPFS Kubo RPC Integration (/api/v0/add, /api/v0/cat, /api/v0/pin)                             │
│    • HybridStorageManager (Local Disk CAS Cache ~/.depeft/storage + IPFS Swarm toàn cầu)                │
│    • SafeTensors Standard Serialization / Deserialization                                               │
│    • Embedded Lightweight Vector Database (Adapter Signature Indexing & Cosine Similarity)              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Web Explorer & Real-Time Dashboard

DePEFT đi kèm built-in single-file Web Explorer được phục vụ trực tiếp bởi bất kỳ node nào tại `http://127.0.0.1:8545/`:
- **Real-Time Metrics**: Current Block Height, Active PEFT Tournaments, CAS Storage Artifacts và Connected P2P Peers.
- **Task Browser**: Kiểm tra các Base Model đã đăng ký (`Qwen2.5-7B`, `LLaMA-3`), LoRA configurations và escrowed bounty balances.
- **🚰 On-Chain Testnet Faucet**: Nhận ngay 10.000 testnet tokens $DEPEFT để tạo task hoặc giả lập mạng.
- **Interactive JSON Console**: Truy vấn REST/JSON-RPC node state trực tiếp trên browser.

---

## 📦 Cấu trúc Thư mục Dự án

```
DePEFT/
├── contracts/                       # Layer 1: EVM & App-Chain Smart Contracts
│   ├── DePeftToken.sol              # Standard ERC-20 Network Token ($DEPEFT)
│   ├── DePeftEscrow.sol             # Task Bounty Escrow & Automated Winner Settlement
│   └── scripts/deploy.js            # Hardhat / Node Deployment Script
├── sdk/                             # Developer SDKs & AI Tooling
│   └── python/                      # Python Client Library (depeft) cho PyTorch & HF Candle
│       ├── depeft/
│       │   ├── client.py            # DePeftClient (Task management, Storage, Faucet, RPC)
│       │   ├── crypto.py            # Tạo khóa Ed25519 và ký transaction
│       │   └── __init__.py
│       └── setup.py
├── scripts/                         # Automation & Network Orchestration
│   └── start_local_testnet.sh       # 1-Click Multi-Node Local P2P Testnet Launcher
├── src/                             # Core Rust App-Chain Protocol
│   ├── lib.rs                       # Crate root export tất cả layers
│   ├── main.rs                      # CLI, Node Daemon & Tournament Simulator
│   ├── consensus/                   # CometBFT 2-Phase Commit Engine
│   ├── tee/                         # Hardware TEE Remote Attestation Engine
│   ├── candle_peft/                 # Deep Learning LLM Engine với Candle
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

## 🚀 Quickstart & Vận hành Mạng

### 1. Khởi chạy Multi-Node Local Testnet (1-Click)
Triển khai 2-node P2P testnet swarm với bootstrap consensus và live web dashboard:
```bash
./scripts/start_local_testnet.sh
```
Mở **http://127.0.0.1:8545** trên trình duyệt để truy cập Web Explorer.

---

### 2. Sử dụng Python SDK (`depeft`)
Tương tác với live network từ Python scripts hoặc Jupyter Notebooks:
```python
from depeft import DePeftClient, generate_keypair

client = DePeftClient("http://127.0.0.1:8545")

# 1. Yêu cầu testnet tokens từ faucet
account = generate_keypair()
print(client.request_faucet(account["account_id"], amount=10000))

# 2. Check balance & chain status
print("Status:", client.get_status())
print("Balance:", client.get_balance(account["account_id"]))
```

---

### 3. Chạy Candle LLM Transformer ReLoRA Tournament Thực tế
Fine-tune LoRA adapters trên Decoder Transformer language model với autograd, HuggingFace `.safetensors` export, TEE evaluation và ReLoRA weight fusion:
```bash
cargo run -- llm-demo --rounds 3 --steps 10
```

---

### 4. Chạy Byzantine Fault Tolerant (BFT) State Finality & Consensus
Thực thi Tendermint/CometBFT 2-phase commit với $> 2/3$ validator quorum và equivocation slashing:
```bash
cargo run -- bft-demo --validators 4 --blocks 3
```

---

### 5. Deploy EVM Smart Contracts ($DEPEFT & Escrow)
Deploy smart contracts lên Sepolia hoặc Base Sepolia testnet sử dụng Hardhat:
```bash
cd contracts
npm install
cp .env.example .env # Configure PRIVATE_KEY
npm run deploy:sepolia
# hoặc deploy lên Base Sepolia
npm run deploy:base-sepolia
```

---

### 6. Deploy Public App-Chain Testnet lên VPS / Cloud (1-Click)
Deploy live public testnet node kèm IPFS Kubo storage trên bất kỳ VPS instance nào:
```bash
./scripts/deploy_vps_testnet.sh
```

---

## 🧪 Comprehensive Test Suite

Chạy toàn bộ integration test suite bao gồm cryptographic, ML, TEE, IPFS, BFT và P2P layers:

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

DePEFT sử dụng kiến trúc modular multi-license:

- **Core Node & Consensus Engine (`src/`, `Cargo.toml`)**: Được cấp phép theo **[GNU General Public License v3.0 (GPL-3.0)](./LICENSE)**. Mọi bản fork hoặc phân phối network của core node bắt buộc phải open-source.
- **Python Client SDK (`sdk/python/`)**: Dual-licensed dưới **[Apache-2.0](./sdk/python/LICENSE-APACHE)** HOẶC **[MIT](./sdk/python/LICENSE-MIT)**, cho phép external AI developers và agents tích hợp DePEFT vào cả proprietary lẫn open workflows.
- **Smart Contracts (`contracts/`)**: Dual-licensed dưới **[Apache-2.0](./contracts/LICENSE-APACHE)** HOẶC **[MIT](./contracts/LICENSE-MIT)** cho việc deploy EVM và DApp integration.
