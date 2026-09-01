# 🤝 Contributing to DePEFT

Thank you for your interest in contributing to **DePEFT** (Decentralized Parameter-Efficient Fine-Tuning)!

DePEFT is an open-source decentralized AI app-chain bringing scalable, verifiable LoRA/QLoRA multi-round tournament fine-tuning to decentralized networks with TEE hardware attestation, Hugging Face Candle autograd, IPFS storage, and CometBFT consensus.

---

## 🧭 Architecture Principles & Invariants

When contributing code, you must strictly uphold the following core architectural invariants:

1. **Frozen Base Model Weights**: Base model weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ must remain completely frozen throughout miner local training. Only low-rank matrices $A$ and $B$ receive gradient updates.
2. **TEE Sandbox Isolation**: Private evaluation datasets must **never** be transmitted across the network or logged in plain text. They must remain strictly sealed within the TEE Sandbox Enclave.
3. **Deterministic Relative Consensus**: Validator loss scores can have slight micro-drift due to heterogeneous hardware (CUDA, ROCm, AVX-512). The App-Chain must **never** average floating-point loss values directly; it must evaluate relative ordinal rankings using Borda Count rank aggregation.
4. **Anti-Collusion Commit-Reveal**: Miners must submit `commit_hash = SHA256(adapter_hash || salt)` during the Commit Phase before uploading the `.safetensors` adapter file in the Reveal Phase.
5. **Cryptographic State Mutation**: No transaction may modify `AppChainState` without valid Ed25519 signature verification against the sender's public key.
6. **BFT Finality**: Blocks are only appended to the immutable ledger when $> 2/3$ supermajority `PRECOMMIT` cryptographic votes are gathered.

---

## 🛠️ Development Setup & Toolchain

### Prerequisites
- **Rust Toolchain**: 1.80+ (Stable)
- **Cargo**: Included with Rust (`rustup default stable`)
- **IPFS Kubo Daemon** (Optional for live IPFS tests, defaults to internal disk CAS)

### Clone & Build
```bash
git clone https://github.com/khoadepeft/DePEFT.git
cd DePEFT

# Build all modules and binaries
cargo build

# Run comprehensive test suite
cargo test
```

---

## 🧪 Testing Guidelines

All new features, bug fixes, and protocol improvements must include corresponding unit and integration tests.

### Running Tests
```bash
# Run all tests
cargo test

# Run tests with backtrace
RUST_BACKTRACE=1 cargo test

# Run a specific test
cargo test test_candle_llm_transformer_relora_tournament -- --nocapture
```

### Key Integration Test Targets:
- [`tests/integration_tests.rs`](file:///home/khoa/DePEFT/tests/integration_tests.rs):
  - `test_candle_lora_linear_forward_and_merge`: Verifies LoRA math and SafeTensors fusion.
  - `test_candle_llm_transformer_relora_tournament`: Full 3-round Candle LLM Transformer training loop.
  - `test_hardware_tee_remote_attestation_and_on_chain_verification`: Hardware quote verification & anti-fraud guards.
  - `test_hybrid_storage_and_ipfs_cas_caching`: Local disk CAS + live IPFS Kubo sync.
  - `test_bft_consensus_2_phase_commit_and_equivocation_slashing`: 2-phase BFT commit and double-voting slashing.
  - `test_p2p_swarm_bidirectional_gossip_and_deduplication`: Length-delimited TCP framing & gossip flooding.

---

## 📐 Rust Style & Code Standards

1. **Formatting**: Ensure your code is formatted with standard `rustfmt`:
   ```bash
   cargo fmt --check
   ```
2. **Clippy (Zero Warnings Enforcement)**: Run Clippy with warnings treated as errors and ensure clean compilation:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
3. **Error Handling**: Use `anyhow::Result` and descriptive errors via `ensure!` or `bail!` in domain logic. Avoid unhandled `.unwrap()` calls in production code paths.
4. **Documentation**: Add Rustdoc comments (`///`) to all public structs, enums, traits, and functions.

---

## 🚀 Pull Request Workflow

1. **Fork the repository** and create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-peft-optimizer
   ```
2. **Make your changes** following the code standards and architectural invariants.
3. **Verify tests pass**:
   ```bash
   cargo test
   ```
4. **Commit with descriptive conventional commits**:
   - `feat(candle): add FlashAttention support to Transformer LM`
   - `fix(p2p): handle socket reconnection on dropped TCP stream`
   - `docs(agents): update REST API spec for task creation`
5. **Open a Pull Request** describing your changes, motivation, and test coverage.

---

## 🔒 Reporting Security Vulnerabilities

Please do not disclose security vulnerabilities publicly on GitHub issues. If you discover a security issue relating to cryptographic signing, TEE remote attestation, or BFT consensus safety, please review [SECURITY.md](file:///home/khoa/DePEFT/SECURITY.md) and report it to the core security maintainers.
