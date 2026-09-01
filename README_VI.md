# 🌐 DePEFT: Nền tảng Fine-Tune AI Phi Tập Trung
### Giao thức Huấn luyện Mô hình AI Đa vòng (ReLoRA) trên Blockchain Chuyên Dụng

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-blue.svg)](./LICENSE)

DePEFT là dự án mã nguồn mở viết bằng Rust, cho phép nhiều người cùng tham gia huấn luyện và tinh chỉnh (fine-tune) các mô hình ngôn ngữ lớn (LLM) một cách minh bạch, an toàn và phi tập trung. Hệ thống sử dụng Hugging Face Candle QLoRA, môi trường bảo mật phần cứng TEE, mạng lưu trữ IPFS và cơ chế đồng thuận CometBFT.

---

## 📚 Danh mục Tài liệu

- **[README.md](./README.md)**: Tài liệu tiếng Anh chính của dự án.
- **[ARCHITECTURE_VI.md](./ARCHITECTURE_VI.md)**: Giải thích chi tiết 4 tầng kiến trúc hệ thống, cách ghép trọng số ReLoRA và cơ chế đồng thuận.
- **[AGENTS.md](./AGENTS.md)**: Sổ tay hướng dẫn cho bot tự hành, thợ đào (miner), người kiểm định (validator) và cách kết nối qua API.
- **[CONTRIBUTING_VI.md](./CONTRIBUTING_VI.md)**: Hướng dẫn cài đặt môi trường, quy tắc viết code và cách gửi đóng góp (Pull Request).
- **[SECURITY_VI.md](./SECURITY_VI.md)**: Chính sách bảo mật, các nguy cơ tấn công và cách báo cáo lỗi bảo mật.

---

## 🏛️ Tổng quan Kiến trúc (4 Tầng Cốt lõi)

```
                       ┌─────────────────────────────────────────────────────────┐
                       │           Người dùng (Tạo Task & Nạp Tiền thưởng)       │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Tầng 1: Blockchain & Cơ chế Đồng thuận (App-Chain & CometBFT)                                           │
│    • Quản lý số dư ví, tiền thưởng ký quỹ, danh sách task và Faucet nhận token thử nghiệm               │
│    • Thuật toán đồng thuận CometBFT 2-Phase Commit (Đề xuất -> Bỏ phiếu -> Xác nhận -> Lưu trữ)         │
│    • Xếp hạng thợ đào theo Borda Count để tránh sai lệch điểm giữa các dòng card màn hình               │
│    • Kiểm tra chứng thực an toàn từ phần cứng TEE ngay trên chuỗi                                       │
└──────────────────────┬────────────────────────────────────────────────────▲─────────────────────────────┘
                       │ Gửi thông tin Task và mô hình                       │ Trả thưởng cho thợ đào thắng
                       ▼                                                    │
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ Tầng 2: Mạng lưới Thợ đào (Miner)               │   │ Tầng 3: Kiểm định & Đánh giá (Validator & TEE)   │
│    • Huấn luyện LoRA bằng Hugging Face Candle   │   │    • Đánh giá mô hình trong vùng an toàn TEE    │
│    • Chỉ cập nhật trọng số nhỏ mà không đổi gốc │   │    • Chống gian lận bằng dữ liệu kiểm thử kín   │
│    • Xuất file SafeTensors                      │   │    • Tạo chữ ký phần cứng chứng thực kết quả    │
│    • Mã băm Commit-Reveal chống đạo bài         │   │    • Dùng Vector DB để phát hiện sao chép bài   │
└──────────────────────┬──────────────────────────┘   └─────────────────────▲───────────────────────────┘
                       │                                                    │
                       │ Tải file trọng số lên                              │ Lấy file về để chấm điểm
                       ▼                                                    │
┌───────────────────────────────────────────────────────────────────────────┴─────────────────────────────┐
│ Tầng 4: Hệ thống Lưu trữ Dữ liệu                                                                        │
│    • Tích hợp trực tiếp với mạng lưu trữ phi tập trung IPFS Kubo                                        │
│    • Kết hợp lưu nhanh trên ổ cứng máy tính và đồng bộ lên IPFS                                         │
│    • Hỗ trợ đọc ghi định dạng chuẩn SafeTensors                                                         │
│    • Cơ sở dữ liệu Vector gọn nhẹ giúp so sánh độ giống nhau giữa các mô hình                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Danh Sách Tính Năng Toàn Diện

| Phân nhóm | Tính năng | Mô tả chi tiết |
|---|---|---|
| **🧬 Giao thức ReLoRA Tournament** | **Tiến hóa Trọng số Liên tục** | Ghép các cập nhật trọng số tốt nhất vào mô hình gốc: $W_{N+1} = W_N + \Delta W^*$ qua từng epoch giải đấu. |
| | **Cơ chế Chia Thưởng Top-K Linh hoạt** | Hỗ trợ `WinnerTakesAll` (100% cho Top 1), `TopKDecay` (chia thưởng giảm dần 50%, 25%, 12.5%...) và `TopKBordaWeighted`. |
| | **Gộp Trọng số Ensemble Đa Miner** | Hỗ trợ `SingleWinner` và `EnsembleWeighted` ($W_{N+1} = W_N + \sum \alpha_i \Delta W_i$) gộp tri thức của Top-K miner. |
| **⚡ Mạng lưới Miner & Động cơ ML** | **Hugging Face Candle Autograd** | Engine Deep Learning thuần Rust cho phép huấn luyện Transformer với thuật toán lan truyền ngược (Backpropagation). |
| | **Lượng tử hóa Đa dạng** | Hỗ trợ trực tiếp **LoRA** (FP32/BF16/FP16), **QLoRA-NF4** (4-bit NormalFloat), và **QLoRA-INT4** (4-bit Integer). |
| | **Đóng gói Chuẩn SafeTensors** | Đọc/ghi nhị phân zero-copy các ma trận adapter tương thích hoàn toàn với hệ sinh thái Hugging Face Hub. |
| | **Giao thức Commit-Reveal Chống Đạo Bài** | Gửi giao dịch 2 pha dùng mã băm `SHA256(adapter_hash \|\| salt)` ngăn chặn hành vi sao chép trọng số trước hạn chót. |
| **🔒 Validator & Bảo mật Phần cứng TEE** | **Chứng thực Phần cứng TEE Từ xa** | Hỗ trợ Intel SGX (DCAP), AMD SEV-SNP, AWS Nitro với chữ ký số chứng thực phần cứng (Attestation Quote). |
| | **Đánh giá trên Private Test Set** | Đánh giá Cross-Entropy Loss & Perplexity trên tập dữ liệu kiểm thử kín được bảo vệ hoàn toàn bên trong enclave TEE. |
| | **Đồng thuận Thứ hạng Borda Count** | Cơ chế tổng hợp thứ hạng tương đối miễn nhiễm hoàn toàn với sai lệch số thực trôi (Floating-Point Drift) giữa các phần cứng GPU/CPU khác nhau. |
| | **Phát hiện Đạo văn bằng Vector DB** | Cơ sở dữ liệu vector nhúng tính toán độ tương đồng Cosine Similarity để cảnh báo và loại bỏ bài nộp sao chép. |
| **🏛️ App-Chain & Đồng thuận CometBFT** | **Đồng thuận CometBFT 2-Phase Commit** | Đồng thuận Byzantine Fault Tolerant đầy đủ 4 pha: `Propose`, `Prevote`, `Precommit`, và `Commit`. |
| | **Cơ chế Phạt Equivocation Slashing** | Tự động phát hiện, tịch thu tiền cọc và loại bỏ các validator ký 2 khối mâu thuẫn (Double Signing). |
| | **Ký quỹ Escrow & Faucet Tự động** | Quản trị số dư ví, TaskSpec, khóa/giải ngân tiền thưởng ký quỹ tự động và Faucet nhận token thử nghiệm. |
| **📦 Hệ thống Lưu trữ & Mạng P2P** | **Lưu trữ Kết hợp IPFS Kubo & Local CAS** | Cầu nối liền mạch giữa bộ nhớ đệm cục bộ (`~/.depeft/storage`) và mạng lưu trữ phân tán IPFS toàn cầu. |
| | **Mạng Gossip P2P Swarm** | Mạng P2P qua giao thức TCP bất đồng bộ với cơ chế lan truyền tin, lọc trùng lặp và giới hạn bộ nhớ chống tấn công DoS. |
| **🖥️ Công cụ Lập trình & Quản trị Node** | **Giao diện Web Explorer Dashboard** | Dashboard tích hợp chạy tại `http://127.0.0.1:8545/` theo dõi block, task, storage CAS, JSON console và Faucet. |
| | **Python SDK (`depeft`)** | Thư viện Python hỗ trợ ký giao dịch Ed25519, quản lý task và tích hợp với PyTorch / Hugging Face. |
| | **Smart Contracts EVM Mainnet** | Hợp đồng [`DePeftToken.sol`](./contracts/DePeftToken.sol) (ERC-20) và [`DePeftEscrow.sol`](./contracts/DePeftEscrow.sol) cho mạng EVM L1/L2. |
| | **Bộ Lệnh CLI Toàn diện** | CLI mạnh mẽ quản trị node, miner, validator, key, P2P, IPFS, benchmark lượng tử hóa và chạy demo. |

---

## 🖥️ Giao diện Web Explorer Trực quan

DePEFT tích hợp sẵn một trang Web Explorer gọn nhẹ, bạn có thể mở trực tiếp từ trình duyệt tại `http://127.0.0.1:8545/`:
- **Theo dõi thời gian thực**: Chiều cao khối, các giải đấu AI đang chạy, các file lưu trữ và số lượng máy đang kết nối.
- **Xem danh sách Task**: Xem mô hình gốc (`Qwen2.5-7B`, `LLaMA-3`), thông số LoRA và số tiền thưởng.
- **🚰 Faucet nhận Token miễn phí**: Bấm nhận ngay 10.000 token $DEPEFT thử nghiệm để tạo task hoặc thử nghiệm mạng.
- **Bảng điều khiển JSON**: Gửi lệnh trực tiếp đến node mà không cần cài đặt phần mềm phụ trợ.

---

## 📦 Cấu trúc Thư mục Dự án

```
DePEFT/
├── contracts/                       # Smart Contract (Token $DEPEFT và Hợp đồng giữ tiền thưởng)
├── sdk/                             # Thư viện cho lập trình viên
│   └── python/                      # Thư viện Python (depeft) dùng cho PyTorch và HF Candle
├── scripts/                         # Các đoạn script chạy nhanh hệ thống
│   └── start_local_testnet.sh       # Lệnh 1-click chạy mạng thử nghiệm trên máy
├── src/                             # Mã nguồn chính viết bằng Rust
│   ├── consensus/                   # Thuật toán đồng thuận CometBFT
│   ├── tee/                         # Cơ chế chứng thực an toàn phần cứng TEE
│   ├── candle_peft/                 # Động cơ học sâu AI dùng Candle
│   ├── p2p/                         # Mạng giao tiếp giữa các máy tính (P2P)
│   ├── node/                        # Máy chủ Web và API cho Node
│   ├── blockchain/                  # Quản lý số dư, giao dịch và xếp hạng
│   ├── miner/                       # Logic dành cho thợ đào huấn luyện AI
│   ├── validator/                   # Logic dành cho người chấm điểm và đánh giá
│   └── storage/                     # Bộ phận lưu trữ file và kết nối IPFS
└── tests/                           # Toàn bộ 16 bài kiểm tra tự động
```

---

## 🚀 Hướng dẫn Bắt đầu Nhanh

### 1. Khởi chạy mạng thử nghiệm trên máy (1 dòng lệnh)
Khởi động ngay 2 node mạng P2P kết nối với nhau và mở giao diện web:
```bash
./scripts/start_local_testnet.sh
```
Sau đó mở trình duyệt tại địa chỉ: **http://127.0.0.1:8545**.

---

### 2. Sử dụng thư viện Python (`depeft`)
Giao tiếp với hệ thống trực tiếp từ Python:
```python
from depeft import DePeftClient, generate_keypair

client = DePeftClient("http://127.0.0.1:8545")

# 1. Tạo ví và xin token thử nghiệm từ Faucet
account = generate_keypair()
print(client.request_faucet(account["account_id"], amount=10000))

# 2. Kiểm tra số dư và trạng thái mạng
print("Trạng thái:", client.get_status())
print("Số dư ví:", client.get_balance(account["account_id"]))
```

---

### 3. Chạy thử nghiệm Huấn luyện AI thực tế
Thử nghiệm toàn bộ quy trình: thợ đào huấn luyện mô hình Transformer, xuất file SafeTensors, chuyển vào vùng an toàn TEE để chấm điểm và ghép trọng số tốt nhất:
```bash
cargo run -- llm-demo --rounds 3 --steps 10
```

---

### 4. Kiểm tra Cơ chế Đồng thuận BFT
Chạy mô phỏng 4 node tham gia bỏ phiếu và chốt khối:
```bash
cargo run -- bft-demo --validators 4 --blocks 3
```

---

### 5. Triển khai Smart Contract
Triển khai hợp đồng lên mạng Sepolia hoặc Base Sepolia:
```bash
cd contracts
npm install
cp .env.example .env # Điền khóa bí mật PRIVATE_KEY vào đây
npm run deploy:sepolia
```

---

## 🧪 Chạy Kiểm thử Toàn bộ Hệ thống

Chạy toàn bộ 16 bài test kiểm tra tự động:

```bash
cargo test
```

Kết quả:
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

## 📄 Bản quyền Mã Nguồn

- **Mã nguồn Node & Blockchain (`src/`, `Cargo.toml`)**: Giấy phép **[AGPL-3.0](./LICENSE)**. Các phiên bản phân phối lại bắt buộc phải giữ mã nguồn mở.
- **Thư viện Python SDK (`sdk/python/`)**: Giấy phép kép **[Apache-2.0](./sdk/python/LICENSE-APACHE)** hoặc **[MIT](./sdk/python/LICENSE-MIT)**, thoải mái tích hợp vào các ứng dụng cá nhân hoặc thương mại.
- **Smart Contract (`contracts/`)**: Giấy phép kép **[Apache-2.0](./contracts/LICENSE-APACHE)** hoặc **[MIT](./contracts/LICENSE-MIT)**.
