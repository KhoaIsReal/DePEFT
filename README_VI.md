# 🌐 DePEFT: Decentralized Parameter-Efficient Fine-Tuning
### Giao thức Đấu trường Đa vòng ReLoRA trên Blockchain Chuyên dụng (App-Chain)

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](./LICENSE)

Bản cài đặt hoàn chỉnh viết bằng ngôn ngữ Rust cho đặc tả thiết kế **Kiến trúc DePEFT** phục vụ tinh chỉnh AI phi tập trung sử dụng Hugging Face Candle QLoRA, Hardware TEE Remote Attestation, mạng lưu trữ phi tập trung IPFS, và cơ chế đồng thuận CometBFT 2-phase commit.

---

## 📚 Mục lục Tài liệu

- **[ARCHITECTURE_VI.md](./ARCHITECTURE_VI.md)** (Bản tiếng Anh: [ARCHITECTURE.md](./ARCHITECTURE.md)): Đặc tả chi tiết 4 tầng kiến trúc, hợp nhất trọng số ReLoRA, tổng hợp Borda Count và cơ chế 2-phase commit CometBFT.
- **[AGENTS.md](./AGENTS.md)**: Hướng dẫn agent tự hành, sổ tay vận hành node, lược đồ API REST/JSON-RPC và vòng lặp thực thi của Miner / Validator.
- **[CONTRIBUTING_VI.md](./CONTRIBUTING_VI.md)** (Bản tiếng Anh: [CONTRIBUTING.md](./CONTRIBUTING.md)): Hướng dẫn đóng góp, thiết lập môi trường phát triển, chuẩn mã nguồn và quy trình PR.
- **[SECURITY_VI.md](./SECURITY_VI.md)** (Bản tiếng Anh: [SECURITY.md](./SECURITY.md)): Mô hình mối đe dọa, cơ chế chống gian lận, xác thực TEE attestation và chính sách công bố lỗ hổng.

---

## 🏛️ Tổng quan Kiến trúc (4 Tầng Cốt lõi)

```
                       ┌─────────────────────────────────────────────────────────┐
                       │          Client (Tạo Task & Ký quỹ Bounty)              │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Tầng Đồng thuận & Trạng thái (Máy trạng thái App-Chain & CometBFT)                                   │
│    • Máy trạng thái tất định (Escrow, Số dư, Sổ đăng ký TaskSpec, Faucet API)                          │
│    • Đồng thuận Tendermint/CometBFT 2-Phase Commit (Propose -> Prevote -> Precommit -> Commit)          │
│    • Động cơ Đồng thuận Tương đối (Tổng hợp hạng Borda Count & Kháng trôi dấu phẩy động)               │
│    • Bộ xác thực TEE Attestation On-Chain (Kiểm tra Danh sách trắng MRENCLAVE / MRSIGNER)              │
└──────────────────────┬────────────────────────────────────────────────────▲─────────────────────────────┘
                       │ TaskSpec ($W_N$, CID)                               │ Xếp hạng Đồng thuận & Trả thưởng
                       ▼                                                    │
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ 2. Tầng Mạng lưới Thợ đào (Tính toán Dị thể)    │   │ 3. Tầng Validator (Tác vụ Ngoại chuỗi & TEE)    │
│    • Động cơ Hugging Face Candle Autograd PEFT  │   │    • Sandbox TEE Phần cứng Cách ly (SGX / SEV)  │
│    • Cập nhật Ma trận QLoRA ($\Delta W = B \times A$) │   │    • Kháng sai số trôi dấu phẩy động (BF16/FP16)│
│    • Đóng gói Nhị phân SafeTensors              │   │    • Tạo Bằng chứng Remote Attestation Phần cứng│
│    • Cơ chế Băm Commit-Reveal SHA-256 & Muối    │   │    • Lập chỉ mục Vector DB Phát hiện Đạo văn    │
└──────────────────────┬──────────────────────────┘   └─────────────────────▲───────────────────────────┘
                       │                                                    │
                       │ Xuất bản $\Delta W$ .safetensors                   │ Kéo & Xác thực $\Delta W$
                       ▼                                                    │
┌───────────────────────────────────────────────────────────────────────────┴─────────────────────────────┐
│ 4. Tầng Lưu trữ & Cơ sở Dữ liệu                                                                         │
│    • Tích hợp trực tiếp IPFS Kubo RPC (/api/v0/add, /api/v0/cat, /api/v0/pin)                           │
│    • HybridStorageManager (Cache CAS đĩa cục bộ ~/.depeft/storage + IPFS Swarm toàn cầu)                │
│    • Tuần tự hóa / Giải tuần tự hóa chuẩn SafeTensors                                                   │
│    • Cơ sở dữ liệu Vector nhúng siêu nhẹ (Lập chỉ mục chữ ký Adapter & Độ tương đồng Cosine)            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Trình khám phá Web (Web Explorer) & Bảng điều khiển Thời gian thực

DePEFT tích hợp sẵn Web Explorer dạng single-file được phục vụ trực tiếp bởi bất kỳ node nào tại địa chỉ `http://127.0.0.1:8545/`:
- **Chỉ số Thời gian thực**: Chiều cao khối hiện tại, Các giải đấu PEFT đang diễn ra, Đối tượng lưu trữ CAS, và Số lượng Node P2P kết nối.
- **Duyệt Task**: Kiểm tra các Base Model đã đăng ký (`Qwen2.5-7B`, `LLaMA-3`), cấu hình LoRA và số dư tiền thưởng ký quỹ.
- **🚰 On-Chain Testnet Faucet**: Nhận ngay 10.000 token testnet $DEPEFT để tạo task hoặc giả lập mạng.
- **Interactive JSON Console**: Truy vấn trạng thái REST/JSON-RPC của node trực tiếp trên trình duyệt.

---

## 📦 Cấu trúc Thư mục Dự án

```
DePEFT/
├── contracts/                       # Tầng 1: Hợp đồng thông minh EVM & App-Chain
│   ├── DePeftToken.sol              # Token ERC-20 chuẩn của mạng lưới ($DEPEFT)
│   ├── DePeftEscrow.sol             # Ký quỹ tiền thưởng Task & Tự động quyết toán
│   └── scripts/deploy.js            # Script triển khai Hardhat / Node
├── sdk/                             # SDK & Công cụ AI cho lập trình viên
│   └── python/                      # Thư viện Python Client (depeft) cho PyTorch & HF Candle
│       ├── depeft/
│       │   ├── client.py            # DePeftClient (Quản lý Task, Storage, Faucet, RPC)
│       │   ├── crypto.py            # Tạo khóa Ed25519 và ký giao dịch
│       │   └── __init__.py
│       └── setup.py
├── scripts/                         # Tự động hóa & Khởi chạy mạng lưới
│   └── start_local_testnet.sh       # Script 1-click khởi chạy Testnet P2P Multi-Node cục bộ
├── src/                             # Mã nguồn Giao thức Cốt lõi Rust App-Chain
│   ├── lib.rs                       # Crate root xuất tất cả các tầng
│   ├── main.rs                      # CLI, Node Daemon & Trình giả lập Giải đấu
│   ├── consensus/                   # Động cơ CometBFT 2-Phase Commit
│   ├── tee/                         # Động cơ Hardware TEE Remote Attestation
│   ├── candle_peft/                 # Động cơ Deep Learning LLM thực thụ với Candle
│   ├── p2p/                         # Mạng phủ P2P & Giao thức Gossip
│   ├── crypto/                      # Mật mã học Ed25519 & Giao dịch đã ký
│   ├── node/                        # Node App-Chain HTTP REST/JSON-RPC & Web Dashboard
│   │   ├── dashboard.rs             # Web Explorer nhúng HTML/CSS/JS
│   │   └── server.rs                # Axum REST & Faucet API
│   ├── client/                      # Rust RPC Client
│   ├── blockchain/                  # Máy trạng thái App-Chain & Đồng thuận tương đối
│   ├── miner/                       # Tầng Mạng lưới Miner & Autograd Worker
│   ├── validator/                   # Tầng Validator & TEE Evaluation Worker
│   ├── storage/                     # Tầng Lưu trữ (Hybrid CAS, IPFS, Vector DB)
│   ├── ml/                          # Các nguyên thủy toán học PEFT (NF4, INT4)
│   └── tournament/                  # Động cơ Giải đấu Đa vòng ReLoRA
├── tests/
│   └── integration_tests.rs         # 16 Bài kiểm thử Tích hợp & Giao thức toàn diện
└── .github/workflows/ci.yml         # Pipeline CI Build & Test tự động trên GitHub Actions
```

---

## 🚀 Khởi động Nhanh & Vận hành Mạng lưới

### 1. Khởi chạy Local Testnet Đa Node (1-Click)
Triển khai mạng testnet P2P gồm 2 node với đồng thuận bootstrap và web dashboard trực tiếp:
```bash
./scripts/start_local_testnet.sh
```
Mở **http://127.0.0.1:8545** trên trình duyệt để truy cập Web Explorer.

---

### 2. Sử dụng Python SDK (`depeft`)
Tương tác với mạng lưới từ script Python hoặc Jupyter Notebook:
```python
from depeft import DePeftClient, generate_keypair

client = DePeftClient("http://127.0.0.1:8545")

# 1. Yêu cầu token testnet từ faucet
account = generate_keypair()
print(client.request_faucet(account["account_id"], amount=10000))

# 2. Kiểm tra số dư & trạng thái chuỗi
print("Trạng thái:", client.get_status())
print("Số dư:", client.get_balance(account["account_id"]))
```

---

### 3. Chạy Giải đấu ReLoRA Mô hình Transformer Candle LLM Thực tế
Tinh chỉnh adapter LoRA trên mô hình ngôn ngữ Decoder Transformer với autograd, xuất HuggingFace `.safetensors`, đánh giá TEE và hợp nhất trọng số ReLoRA:
```bash
cargo run -- llm-demo --rounds 3 --steps 10
```

---

### 4. Kiểm tra Tính tất định Trạng thái & Đồng thuận BFT
Thực thi cơ chế CometBFT 2-phase commit với quorum $> 2/3$ validator và cơ chế phạt (slashing) khi bỏ phiếu hai lần:
```bash
cargo run -- bft-demo --validators 4 --blocks 3
```

---

### 5. Triển khai Smart Contracts EVM ($DEPEFT & Escrow)
Triển khai hợp đồng lên testnet Sepolia hoặc Base Sepolia bằng Hardhat:
```bash
cd contracts
npm install
cp .env.example .env # Cấu hình PRIVATE_KEY
npm run deploy:sepolia
# hoặc triển khai lên Base Sepolia
npm run deploy:base-sepolia
```

---

### 6. Triển khai Testnet App-Chain Công khai lên VPS / Cloud (1-Click)
Triển khai một node testnet công khai kèm lưu trữ IPFS Kubo trên bất kỳ VPS nào:
```bash
./scripts/deploy_vps_testnet.sh
```

---

## 🧪 Bộ Kiểm thử Tích hợp Toàn diện

Chạy toàn bộ bộ kiểm thử tích hợp bao gồm tất cả các tầng mật mã, ML, TEE, IPFS, BFT và P2P:

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

## 📄 Giấy phép Bản quyền (Licensing)

DePEFT sử dụng kiến trúc đa giấy phép theo module nhằm bảo vệ sáng tạo công nghệ blockchain cốt lõi đồng thời thúc đẩy việc ứng dụng của cộng đồng lập trình viên:

- **Node cốt lõi & Động cơ Đồng thuận (`src/`, `Cargo.toml`)**: Cấp phép theo **[GNU General Public License v3.0 (GPL-3.0)](./LICENSE)**. Mọi bản fork hoặc phân phối mạng lưới của node cốt lõi đều phải giữ mã nguồn mở.
- **Python Client SDK (`sdk/python/`)**: Giấy phép kép **[Apache-2.0](./sdk/python/LICENSE-APACHE)** HOẶC **[MIT](./sdk/python/LICENSE-MIT)**, cho phép các nhà phát triển AI và agent tự do tích hợp DePEFT vào cả các dự án thương mại lẫn nguồn mở.
- **Hợp đồng thông minh (`contracts/`)**: Giấy phép kép **[Apache-2.0](./contracts/LICENSE-APACHE)** HOẶC **[MIT](./contracts/LICENSE-MIT)** để linh hoạt triển khai EVM và tích hợp DApp.
