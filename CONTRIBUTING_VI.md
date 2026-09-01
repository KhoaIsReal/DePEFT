# 🤝 Contributing to DePEFT

Cảm ơn bạn đã quan tâm và muốn đóng góp cho **DePEFT** (Decentralized Parameter-Efficient Fine-Tuning)!

DePEFT là open-source decentralized AI app-chain mang đến giải pháp scalable, verifiable LoRA/QLoRA multi-round tournament fine-tuning cho decentralized networks với TEE hardware attestation, Hugging Face Candle autograd, IPFS storage và CometBFT consensus.

---

## 🧭 Nguyên tắc Kiến trúc & Invariants Bắt buộc

Khi đóng góp mã nguồn, bạn phải tuân thủ nghiêm ngặt các core architectural invariants sau:

1. **Frozen Base Model Weights**: Base model weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ phải được freeze hoàn toàn trong suốt quá trình miner local training. Chỉ low-rank matrices $A$ và $B$ mới nhận gradient updates.
2. **TEE Sandbox Isolation**: Private test sets **tuyệt đối không** được truyền qua network hoặc ghi log dưới dạng plain text. Chúng phải được seal tuyệt đối bên trong TEE Sandbox Enclave.
3. **Deterministic Relative Consensus**: Validator loss scores có thể bị micro-drift do heterogeneous hardware (CUDA, ROCm, AVX-512). App-Chain **không bao giờ** được average trực tiếp floating-point loss values; blockchain bắt buộc phải đánh giá relative ordinal rankings bằng Borda Count rank aggregation.
4. **Anti-Collusion Commit-Reveal**: Miner bắt buộc phải submit `commit_hash = SHA256(adapter_hash || salt)` trong Commit Phase trước khi upload file adapter `.safetensors` trong Reveal Phase.
5. **Cryptographic State Mutation**: Không transaction nào được phép thay đổi `AppChainState` nếu không có Ed25519 signature hợp lệ khớp với sender public key.
6. **BFT Finality**: Blocks chỉ được commit vào ledger bất biến khi thu thập đủ $> 2/3$ supermajority `PRECOMMIT` cryptographic votes.

---

## 🛠️ Development Setup & Toolchain

### Prerequisites
- **Rust Toolchain**: 1.80+ (Stable)
- **Cargo**: Đi kèm sẵn với Rust (`rustup default stable`)
- **IPFS Kubo Daemon** (Tùy chọn cho live IPFS tests, mặc định sử dụng internal disk CAS)

### Clone & Build
```bash
git clone https://github.com/khoadepeft/DePEFT.git
cd DePEFT

# Build all modules and binaries
cargo build

# Run test suite
cargo test
```

---

## 🧪 Testing Guidelines

Tất cả tính năng mới, bug fixes và protocol improvements đều phải có unit tests và integration tests tương ứng.

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
  - `test_candle_lora_linear_forward_and_merge`: Kiểm tra LoRA math và SafeTensors fusion.
  - `test_candle_llm_transformer_relora_tournament`: Full 3-round Candle LLM Transformer training loop.
  - `test_hardware_tee_remote_attestation_and_on_chain_verification`: Hardware quote verification & anti-fraud guards.
  - `test_hybrid_storage_and_ipfs_cas_caching`: Đồng bộ local disk CAS + live IPFS Kubo.
  - `test_bft_consensus_2_phase_commit_and_equivocation_slashing`: 2-phase BFT commit và slashing validator double-voting.
  - `test_p2p_swarm_bidirectional_gossip_and_deduplication`: Length-delimited TCP framing & gossip flooding.

---

## 📐 Rust Style & Code Standards

1. **Formatting**: Đảm bảo code được format chuẩn bằng `rustfmt`:
   ```bash
   cargo fmt --check
   ```
2. **Clippy (Zero Warnings Enforcement)**: Chạy Clippy với warnings treated as errors và đảm bảo clean compilation:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
3. **Error Handling**: Sử dụng `anyhow::Result` cùng descriptive errors qua `ensure!` hoặc `bail!` trong domain logic. Tránh unhandled `.unwrap()` trong production code paths.
4. **Documentation**: Thêm Rustdoc comments (`///`) cho tất cả public structs, enums, traits và functions.

---

## 🚀 Pull Request Workflow

1. **Fork repository** và tạo feature branch từ `main`:
   ```bash
   git checkout -b feature/my-new-peft-optimizer
   ```
2. **Thực hiện thay đổi** tuân thủ theo code standards và architectural invariants.
3. **Verify tests pass**:
   ```bash
   cargo test
   ```
4. **Commit với conventional commits**:
   - `feat(candle): add FlashAttention support to Transformer LM`
   - `fix(p2p): handle socket reconnection on dropped TCP stream`
   - `docs(agents): update REST API spec for task creation`
5. **Mở Pull Request** mô tả chi tiết changes, motivation và test coverage.

---

## 🔒 Báo cáo Lỗ hổng Bảo mật

Vui lòng không public các lỗ hổng bảo mật lên GitHub Issues. Nếu bạn phát hiện vấn đề liên quan đến cryptographic signing, TEE remote attestation hoặc BFT consensus safety, vui lòng xem [SECURITY_VI.md](file:///home/khoa/DePEFT/SECURITY_VI.md) và liên hệ trực tiếp với core maintainers.
