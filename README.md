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
│    • Deterministic State Machine (Escrow, Balances, TaskSpec Registry)                                 │
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

## 🔄 5-Phase ReLoRA Tournament Lifecycle

For each tournament round $N \in \{1, 2, \dots, K\}$:

1. **Task Initialization ($W_N$)**:
   - Client creates a `TaskSpec`, specifying Base Model $W_N$, IPFS dataset CID, PEFT method (`QLoRA_NF4`), target modules (`q_proj`, `v_proj`, `out_proj`), and locks bounty funds into escrow.
2. **Commit Phase**:
   - Heterogeneous miners (NVIDIA CUDA, AMD ROCm, bare-metal CPU) fine-tune local LoRA matrices $\Delta W$ using Candle autograd.
   - Miners submit an on-chain commitment: `commit_hash = SHA256(adapter_hash || salt)` to prevent front-running and plagiarism.
3. **Reveal Phase**:
   - Miners upload `.safetensors` adapter files to IPFS and reveal `(adapter_cid, salt)`.
   - The App-Chain deterministically validates that `SHA256(SHA256(safetensors) || salt) == commit_hash`.
4. **Evaluation Phase (Private Test Set & Relative Consensus)**:
   - Validators download adapters from IPFS and evaluate them inside an isolated **Hardware TEE Sandbox** on a **Private Test Set** (preventing data leakage and overfitting).
   - Rather than relying on fragile floating-point equality across heterogeneous GPUs, validators submit **Relative Rankings** accompanied by cryptographic **Hardware Attestation Quotes**.
   - The App-Chain runs **Borda Count Relative Consensus** to elect the Top-1 winner.
5. **Merge Phase (Weight Evolution $W_{N+1}$)**:
   - The winning adapter weights are permanently fused into the base model:
     $$W_{N+1} = W_N + \frac{\alpha}{r} (B^* A^*)$$
   - The winning miner receives the escrowed bounty reward, and the network advances to epoch $N+1$.

---

## 📦 Project Structure

```
DePEFT/
├── src/
│   ├── lib.rs                       # Crate root exporting all layers
│   ├── main.rs                      # Comprehensive CLI, Node Daemon & Tournament Simulator
│   ├── consensus/                   # Layer 1: Tendermint/CometBFT 2-Phase Commit Engine
│   │   ├── types.rs                 # Block, BlockHeader, Vote, VoteType, BlockCommit
│   │   ├── state_machine.rs         # BftEngine (Propose -> Prevote -> Precommit -> Commit)
│   │   ├── validator_set.rs         # ValidatorSet (2/3+ Quorum, Proposer round-robin)
│   │   ├── slashing.rs              # SlashingEngine (Equivocation double-vote detection)
│   │   └── mod.rs
│   ├── tee/                         # Hardware TEE Remote Attestation Engine
│   │   ├── types.rs                 # AttestationQuote, EnclaveMeasurement (MRENCLAVE/MRSIGNER)
│   │   ├── enclave.rs               # HardwareTeeEnclave (Intel SGX DCAP / AMD SEV-SNP quotes)
│   │   ├── verifier.rs              # OnChainTeeVerifier (Cryptographic Quote & Registry checking)
│   │   └── mod.rs
│   ├── candle_peft/                 # Real Deep Learning LLM Engine with Candle
│   │   ├── transformer.rs           # Decoder Transformer LM (LLaMA/Qwen) with RMSNorm & SwiGLU
│   │   ├── lora.rs                  # CandleLoraLinear (Forward, Autograd, SafeTensors export)
│   │   ├── trainer.rs               # CandleMinerTrainer (SGD autograd optimization)
│   │   ├── evaluator.rs             # CandleValidatorEvaluator (Cross-entropy & Perplexity)
│   │   ├── merger.rs                # CandleWeightMerger (ReLoRA weight fusion)
│   │   ├── tokenizer.rs             # SimpleByteTokenizer for text processing
│   │   └── mod.rs
│   ├── p2p/                         # P2P Overlay Network & Gossip Protocol
│   │   ├── types.rs                 # PeerId, P2pMessage (Handshake, BroadcastTx, AnnounceCid)
│   │   ├── codec.rs                 # Async Length-Delimited TCP framing codec
│   │   ├── swarm.rs                 # P2pSwarm, Peer routing table, Gossip deduplication
│   │   └── mod.rs
│   ├── crypto/                      # Production Cryptography & Signing
│   │   ├── keys.rs                  # Ed25519 AccountKeypair & SignedTransaction envelope
│   │   └── mod.rs
│   ├── node/                        # Live App-Chain HTTP REST/JSON-RPC Node
│   │   ├── server.rs                # Axum server, API endpoints, P2P state synchronization
│   │   └── mod.rs
│   ├── client/                      # RPC Client
│   │   ├── rpc.rs                   # DePeftClient talking to live node over HTTP
│   │   └── mod.rs
│   ├── blockchain/                  # Layer 1: App-Chain State Machine
│   │   ├── types.rs                 # TaskSpec, AccountId, PeftType, RoundPhase, Summary
│   │   ├── state.rs                 # AppChainState, Escrow, Balances, RoundContext
│   │   ├── transactions.rs          # CreateTask, CommitAdapter, RevealAdapter, SubmitEvaluation
│   │   └── relative_consensus.rs    # Borda Count rank aggregation engine
│   ├── miner/                       # Layer 2: Miner Compute Network
│   │   ├── trainer.rs               # QLoRA fine-tuning engine, commit-reveal hashing
│   │   └── worker.rs                # MinerNode worker and automated lifecycle
│   ├── validator/                   # Layer 3: Validator Off-Chain Workers
│   │   ├── tee.rs                   # TEE Sandbox enclave for Private Test Set isolation
│   │   ├── evaluator.rs             # Multi-hardware evaluation with float drift tolerance
│   │   └── worker.rs                # ValidatorNode worker
│   ├── storage/                     # Layer 4: Decentralized Storage & CAS Database
│   │   ├── kubo_client.rs           # Live IPFS Kubo RPC client (/api/v0/add, /api/v0/cat, /api/v0/pin)
│   │   ├── hybrid_storage.rs        # HybridStorageManager (Disk CAS cache + Live IPFS network)
│   │   ├── disk_ipfs.rs             # Persistent Disk-backed CAS for ~/.depeft/storage
│   │   ├── ipfs.rs                  # Content Addressable Storage (CAS) simulator
│   │   ├── safetensors.rs           # Standard SafeTensors serializer / parser
│   │   └── vector_db.rs             # Embedded Lightweight Vector DB (Cosine Similarity)
│   ├── ml/                          # Mathematical PEFT Primitives
│   │   ├── tensor.rs                # Matrix ops, NF4 (NormalFloat4) and INT4 quantization
│   │   ├── lora.rs                  # QLoRALinear layer, forward, backward, adapter merge
│   │   ├── model.rs                 # Multi-module DePEFTModel with full backpropagation
│   │   └── dataset.rs               # Train/test split and synthetic domain benchmark generator
│   └── tournament/                  # ReLoRA Orchestration
│       └── engine.rs                # Multi-round ReLoRA Tournament Engine
└── tests/
    └── integration_tests.rs         # Comprehensive unit, crypto, CAS, Candle, TEE, IPFS, BFT, and P2P tests
```

---

## 🚀 Quickstart & CLI Usage

### 1. Run Real Candle LLM Transformer ReLoRA Tournament
Fine-tune real LoRA adapters on a Decoder Transformer language model with autograd, HuggingFace `.safetensors` export, TEE evaluation, and ReLoRA weight fusion:
```bash
cargo run -- llm-demo --rounds 3 --steps 10
```

### 2. Run Byzantine Fault Tolerant (BFT) State Finality & Consensus
Execute Tendermint/CometBFT 2-phase commit with 2/3+ validator quorum, aggregate signatures, and equivocation slashing:
```bash
cargo run -- bft-demo --validators 4 --blocks 3
```

### 3. Verify Hardware TEE Remote Attestation Quote & Anti-Fraud Security
Generate and verify cryptographic Intel SGX / AMD SEV hardware quotes on-chain:
```bash
cargo run -- tee-quote
```

### 4. Live IPFS Kubo & Decentralized Storage Network Integration
Inspect local/remote IPFS Kubo daemon status & connected peers:
```bash
cargo run -- ipfs status --api-url http://127.0.0.1:5001
```
Upload and pin `.safetensors` model weights or datasets to IPFS:
```bash
cargo run -- ipfs put ./model_weights.safetensors
```
Retrieve content by CID from the global IPFS swarm:
```bash
cargo run -- ipfs cat <CID> --output ./downloaded.safetensors
```

### 5. Keypair Generation & Wallet Management
Generate a cryptographically secure Ed25519 account keypair:
```bash
cargo run -- key generate
```
Inspect an existing private key:
```bash
cargo run -- key inspect <0x_secret_key>
```

### 6. Run Live App-Chain Node Daemon with P2P Overlay
Start Node 1 (Bootstrap node with P2P on port 9000, HTTP on 8545):
```bash
cargo run -- node start --port 8545 --p2p-port 9000
```
Start Node 2 (Connects to Node 1 via P2P bootnodes):
```bash
cargo run -- node start --port 8546 --p2p-port 9001 --bootnodes 127.0.0.1:9000
```

### 7. P2P Peer Discovery & Management
List connected P2P peers:
```bash
cargo run -- p2p peers --node-url http://127.0.0.1:8545
```
Connect to a remote P2P peer over TCP:
```bash
cargo run -- p2p connect --node-url http://127.0.0.1:8545 --addr 127.0.0.1:9001
```

### 8. Client Task Submission & Query
Submit a new fine-tuning task on the live network (automatically gossiped to all peers):
```bash
cargo run -- task create --node-url http://127.0.0.1:8545 --secret-key <0x_key> --model "Qwen/Qwen2.5-7B" --bounty 30000
```
List active network tasks:
```bash
cargo run -- task list --node-url http://127.0.0.1:8545
```

### 9. Run Miner & Validator Daemons Against Live Node
Run a Miner daemon (fine-tunes QLoRA, signs commit-reveal, uploads `.safetensors` to CAS):
```bash
cargo run -- miner run --node-url http://127.0.0.1:8545 --secret-key <0x_miner_key> --task-id 1 --hardware "NVIDIA RTX 4090 / CUDA"
```
Run a Validator daemon (evaluates adapters in secure TEE sandbox and submits consensus rank vote):
```bash
cargo run -- validator run --node-url http://127.0.0.1:8545 --secret-key <0x_val_key> --task-id 1 --hardware "Intel Xeon / AVX-512"
```

### 10. Run PEFT Quantization Benchmarks & Vector DB Search
```bash
# Benchmark NF4 vs INT4 vs FP32
cargo run -- benchmark

# Search Embedded Vector DB for adapter signatures
cargo run -- vector-query
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

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
