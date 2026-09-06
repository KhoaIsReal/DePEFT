# 🌐 DePEFT: Nền tảng Fine-Tune AI Phi Tập Trung
### Giao thức Huấn luyện Mô hình AI Đa vòng (ReLoRA) trên Blockchain Chuyên Dụng

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)]()
[![Clippy](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![Engine](https://img.shields.io/badge/ML%20Engine-Hugging%20Face%20Candle-blue.svg)](https://github.com/huggingface/candle)
[![Consensus](https://img.shields.io/badge/consensus-CometBFT%20%2F%20Tendermint-blueviolet.svg)]()
[![TEE](https://img.shields.io/badge/TEE-Intel%20SGX%20%7C%20AMD%20SEV--SNP-informational.svg)]()
[![Storage](https://img.shields.io/badge/storage-IPFS%20Kubo%20%7C%20Filecoin-teal.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-blue.svg)](./LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Tham%20Gia%20Cộng%20Đồng-5865F2?logo=discord&logoColor=white)](https://discord.gg/23FNT8MP7R)

> [!WARNING]
> ### ⚠️ CẢNH BÁO AN TOÀN
> **!!! ĐANG TRONG GIAI ĐOẠN TESTNET, NẾU BẠN THẤY THẰNG NÀO BẢO NÓ LÊN MAINNET RỒI THÌ CÓ THỂ ĐÓ LÀ LỪA ĐẢO !!!**
>
> Dự án **DePEFT hiện tại chỉ đang chạy thử nghiệm (Testnet)** và hoàn toàn **CHƯA phát hành bất kỳ token nào trên Mainnet** (Ethereum, Base, Solana, BNB Chain hay bất kỳ sàn giao dịch DEX/CEX nào). Mọi lời mời mua bán token, airdrop nạp tiền, hoặc hợp đồng tự xưng là DePEFT Mainnet ở thời điểm hiện tại đều là **giả mạo và lừa đảo**.

**DePEFT** là mạng lưới phi tập trung mã nguồn mở cho phép bất kỳ ai cũng có thể tài trợ tiền thưởng, huấn luyện (miner) và đánh giá (validator) các Mô hình Ngôn ngữ Lớn (LLM) mà không cần phụ thuộc vào bất kỳ nhà cung cấp đám mây tập trung nào.

Thay vì huấn luyện lại toàn bộ mô hình (tốn hàng triệu USD), DePEFT áp dụng kỹ thuật **Parameter-Efficient Fine-Tuning (PEFT / QLoRA)**: các thợ đào chỉ huấn luyện các ma trận adapter kích thước nhỏ ($\Delta W$). Qua từng vòng giải đấu (epoch), các adapter tốt nhất sẽ được kiểm định trong **Môi trường An toàn Phần cứng TEE**, xếp hạng bằng **Đồng thuận Thứ hạng Borda Count**, và được gộp vĩnh viễn vào mô hình gốc ($W_{N+1} = W_N + \Delta W^*$).

---

## 📖 Mục Lục
1. [Giải Thích Dễ Hiểu Cách DePEFT Hoạt Động](#-giải-thích-dễ-hiểu-cách-depeft-hoạt-động)
2. [Các Vai Trò Trong Mạng Lưới](#-các-vai-trò-trong-mạng-lưới)
3. [Hướng Dẫn Sử Dụng Chi Tiết Từng Bước (A-Z)](#-hướng-dẫn-sử-dụng-chi-tiết-từng-bước-a-z)
   - [Bước 1: Cài đặt & Biên dịch](#bước-1-cài-đặt--biên-dịch)
   - [Bước 2: Tạo Ví & Cặp Khóa Ed25519](#bước-2-tạo-ví--cặp-khóa-ed25519)
   - [Bước 3: Khởi chạy Node Daemon](#bước-3-khởi-chạy-node-daemon)
   - [Bước 4: Nhận Token Thử Nghiệm Miễn Phí (Faucet)](#bước-4-nhận-token-thử-nghiệm-miễn-phí-faucet)
   - [Bước 5: Tạo Nhiệm vụ Huấn Luyện AI (Client)](#bước-5-tạo-nhiệm-vụ-huấn-luyện-ai-client)
   - [Bước 6: Chạy Thợ Đào Huấn Luyện (Miner)](#bước-6-chạy-thợ-đào-huấn-luyện-miner)
   - [Bước 7: Chạy Trình Kiểm Định TEE (Validator)](#bước-7-chạy-trình-kiểm-định-tee-validator)
   - [Bước 8: Kiểm tra Mạng P2P và IPFS](#bước-8-kiểm-tra-mạng-p2p-và-ipfs)
4. [Hướng Dẫn Dùng Python SDK](#-hướng-dẫn-dùng-python-sdk)
5. [Bảng Tra Cứu API REST / JSON-RPC](#-bảng-tra-cứu-api-rest--json-rpc)
6. [Các Lệnh Chạy Mô Phỏng Trực Quan](#-các-lệnh-chạy-mô-phỏng-trực-quan)
7. [Kiến Trúc 4 Tầng Cốt Lõi](#-kiến-trúc-4-tầng-cốt-lõi)
8. [Kiểm Thử & Đảm Bảo Chất Lượng](#-kiểm-thử--đảm-bảo-chất-lượng)
9. [⚡ Câu Hỏi Nhanh Dành Cho Nhà Đầu Tư & Thợ Đào (FAST Q&A)](./FAST_Q&A_VI.md)
10. [Bản Quyền Mã Nguồn](#-bản-quyền-mã-nguồn)

---

## 💡 Giải Thích Dễ Hiểu Cách DePEFT Hoạt Động

```
  1. Khách hàng nạp token vào Quỹ Escrow & Tạo Task trên Blockchain
                                │
                                ▼
  2. Các Miner huấn luyện adapter QLoRA trên máy của mình (GPU / CPU)
                                │
                                ▼
  3. Miner nộp mã băm Commit Hash: SHA256(adapter || salt) [Chống Sao Chép Bài]
                                │
                                ▼
  4. Hết hạn Commit, Miner tải file .safetensors lên bộ nhớ và gửi mã Reveal
                                │
                                ▼
  5. Validator tải adapter vào Vùng Bảo Mật TEE để chấm điểm trên Test Set Kín
                                │
                                ▼
  6. Validator gửi Chữ ký Phần cứng TEE + Bảng Xếp Hạng -> Đồng thuận Borda Count
                                │
                                ▼
  7. Trọng số thắng cuộc được gộp vào Mô hình Gốc: W_N+1 = W_N + ΔW* & Trả thưởng
```

1. **Ký quỹ Tiền thưởng**: Người tạo task nạp token $DEPEFT vào hợp đồng ký quỹ, chọn mô hình gốc (ví dụ `Qwen/Qwen2.5-7B`), các tầng cần can thiệp (ví dụ `q_proj, v_proj`) và bộ dữ liệu huấn luyện.
2. **Giải đấu 2 Pha Commit-Reveal**:
   - **Pha Commit**: Các miner tải mô hình gốc về, huấn luyện thêm trên GPU/CPU, sau đó chỉ nộp lên blockchain mã băm ẩn `SHA256(adapter || salt)`. Việc này đảm bảo các miner khác không thể nhìn trộm hay copy trọng số của nhau trước giờ chót.
   - **Pha Reveal**: Khi chuyển pha, miner tải file trọng số `.safetensors` lên kho lưu trữ và công khai chuỗi salt để chuỗi xác thực.
3. **Chấm điểm Kín trong Phần cứng TEE**: Validator đưa các adapter của miner vào môi trường phần cứng cách ly (Intel SGX hoặc AMD SEV-SNP) và chấm điểm trên **tập kiểm thử kín (Private Test Set)** mà miner không thể tiếp cận để học vẹt.
4. **Đồng thuận Thứ hạng Borda Count**: Các dòng card (NVIDIA, AMD, CPU) khi chạy số thực thường có sai lệch thập phân nhỏ (floating-point drift). DePEFT giải quyết việc này bằng cách cho validator xếp hạng thứ bậc (1, 2, 3...) thay vì so sánh trực tiếp số điểm thập phân.
5. **Tiến hóa Trọng số ReLoRA**: Adapter chiến thắng được hòa nhập vĩnh viễn vào mô hình gốc ($W_1 = W_0 + \Delta W$). Miner thắng cuộc nhận tiền thưởng, và vòng thi đấu tiếp theo sẽ diễn ra trên nền mô hình mới đã thông minh hơn.

---

## 👥 Các Vai Trò Trong Mạng Lưới

| Vai trò | Trách nhiệm chính | Yêu cầu phần cứng | Lệnh khởi chạy |
|---|---|---|---|
| **Node Operator** | Vận hành node blockchain, lưu trữ dữ liệu, chuyển tiếp giao dịch P2P, mở Web Explorer | 2 vCPU, 4GB RAM | `depeft node start` |
| **Client (Khách hàng)** | Nạp tiền thưởng, đăng ký mô hình và dữ liệu huấn luyện | Máy tính thông thường | `depeft task create` |
| **Miner (Thợ đào AI)** | Tải mô hình, huấn luyện ma trận adapter, nhận thưởng token | GPU (CUDA/ROCm) hoặc CPU | `depeft miner run` |
| **Validator (Kiểm định)** | Tải adapter, kiểm tra trong vùng bảo mật TEE, ký chứng thực phần cứng | Intel SGX / AMD SEV (hoặc cờ Sim) | `depeft validator run` |

---

## 🛠️ Hướng Dẫn Sử Dụng Chi Tiết Từng Bước (A-Z)

### Bước 1: Cài đặt & Biên dịch

Yêu cầu máy tính đã cài đặt Rust (bản 1.80 trở lên):
```bash
git clone https://github.com/KhoaIsReal/DePEFT.git
cd DePEFT
cargo build --release
```
File thực thi nằm tại `target/release/depeft`. Bạn cũng có thể dùng `cargo run --` để thực hiện các lệnh.

---

### Bước 2: Tạo Ví & Cặp Khóa Ed25519

Tất cả các hành động trên DePEFT (tạo task, commit bài, bỏ phiếu) đều cần chữ ký mật mã **Ed25519**.

Tạo một ví tài khoản mới:
```bash
cargo run -- key generate
```
Kết quả hiển thị:
```text
=== Generated New DePEFT Ed25519 Account Keypair ===
Public Address: 0x404bb343c6827a58a983b6329e4720970b8c95a0
Public Key:     0x280e227092147743d548325ebef50beadca2e4cbe45bfefba3ca87884ecb510a
Secret Key:     0x8e833b5c33feff4cf19cb8d67ec1bb84c59d9f5cb68b5a8370f1a9d18ce58826
```

Kiểm tra lại thông tin ví từ Secret Key:
```bash
cargo run -- key inspect 0x8e833b5c33feff4cf19cb8d67ec1bb84c59d9f5cb68b5a8370f1a9d18ce58826
```

---

### Bước 3: Khởi chạy Node Daemon

#### Cách 1: Khởi chạy 1 Node cục bộ (hỗ trợ testnet và nhận coin miễn phí)
```bash
cargo run -- node start \
  --port 8545 \
  --p2p-port 9000 \
  --testnet-tee-sim \
  --enable-operator-endpoints \
  --enable-faucet
```
Các cờ quan trọng:
- `--testnet-tee-sim`: Tự động kích hoạt khóa tin cậy mô phỏng TEE để bạn có thể chạy validator mà không cần phần cứng Intel SGX máy chủ đắt tiền.
- `--enable-operator-endpoints`: Mở cổng `POST /api/v1/storage` để thợ đào tải file `.safetensors` lên node.
- `--enable-faucet`: Mở cổng `POST /api/v1/faucet` để phát token thử nghiệm.

#### Cách 2: Chạy mạng thử nghiệm 2 Node P2P (1 dòng lệnh)
```bash
./scripts/start_local_testnet.sh
```
- **Node 1 (Gốc)**: `http://127.0.0.1:8545` (P2P: `9000`)
- **Node 2 (Ngang hàng)**: `http://127.0.0.1:8546` (P2P: `9001`)
- **Giao diện Web Explorer**: Mở ngay `http://127.0.0.1:8545` trên trình duyệt!

---

### Bước 4: Nhận Token Thử Nghiệm Miễn Phí (Faucet)

Dùng lệnh cURL để nhận 50.000 token $DEPEFT vào ví:
```bash
curl -X POST http://127.0.0.1:8545/api/v1/faucet \
  -H "Content-Type: application/json" \
  -d '{"account": "0x404bb343c6827a58a983b6329e4720970b8c95a0", "amount": 50000}'
```

Kiểm tra số dư ví:
```bash
curl http://127.0.0.1:8545/api/v1/accounts/0x404bb343c6827a58a983b6329e4720970b8c95a0/balance
```

---

### Bước 5: Tạo Nhiệm vụ Huấn Luyện AI (Client)

Dành cho người cần tinh chỉnh mô hình và treo thưởng:

```bash
cargo run -- task create \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_secret_key_cua_ban \
  --model-id "Qwen/Qwen2.5-7B" \
  --bounty 30000 \
  --top-k 3 \
  --merge single
```
Ý nghĩa tham số:
- `--model-id`: Tên mô hình gốc trên Hugging Face.
- `--bounty`: Số lượng token tiền thưởng đặt cọc ký quỹ.
- `--top-k`: Số lượng thợ đào được chia thưởng (ví dụ `1` là người về nhất nhận hết, hoặc `3`/`5` để chia thưởng giảm dần).
- `--merge`: Cách gộp trọng số (`single`: lấy mẫu tốt nhất, `ensemble`: gộp trọng số của top miner).

Xem danh sách tất cả các task đang có:
```bash
cargo run -- task list --node-url http://127.0.0.1:8545
```

---

### Bước 6: Chạy Thợ Đào Huấn Luyện (Miner)

Miner sẽ tự động kết nối vào Task, nhận mô hình, tính toán huấn luyện cục bộ bằng thuật toán Candle QLoRA, nộp commit hash, tải file trọng số lên và nộp bằng chứng reveal:

```bash
cargo run -- miner run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_secret_key_cua_miner \
  --task-id 1 \
  --device auto \
  --lr 0.03
```
Các chế độ `--device`:
- `auto`: Tự động phát hiện phần cứng tối ưu nhất (CUDA $\to$ ROCm $\to$ Metal $\to$ WGPU $\to$ CPU).
- `cuda`: Chạy trên card màn hình NVIDIA.
- `rocm`: Chạy trên card màn hình AMD.
- `cpu`: Chạy trên vi xử lý CPU đa luồng với AVX-512.

---

### Bước 7: Chạy Trình Kiểm Định TEE (Validator)

Validator chờ các miner nộp bài, tải các file trọng số về, nạp vào vùng an toàn TEE, chấm điểm trên tập kiểm thử kín, sinh chữ ký chứng thực phần cứng và bỏ phiếu xếp hạng:

```bash
cargo run -- validator run \
  --node-url http://127.0.0.1:8545 \
  --secret-key 0x_secret_key_cua_validator \
  --task-id 1
```

---

### Bước 8: Kiểm tra Mạng P2P và IPFS

Xem danh sách các node P2P đang liên kết:
```bash
cargo run -- p2p peers --node-url http://127.0.0.1:8545
```

Chủ động kết nối với một node P2P khác:
```bash
cargo run -- p2p connect --node-url http://127.0.0.1:8545 --addr "127.0.0.1:9001"
```

Kiểm tra trạng thái máy chủ lưu trữ IPFS Kubo:
```bash
cargo run -- ipfs status --api-url http://127.0.0.1:5001
```

---

## 🐍 Hướng Dẫn Dùng Python SDK

Thư viện Python SDK chính thức nằm tại thư mục `sdk/python/`.

### Cài đặt thư viện
```bash
cd sdk/python
pip install -e .
```

### Kịch bản Python mẫu hoàn chỉnh
```python
from depeft import DePeftClient, generate_keypair
import time

# 1. Kết nối tới node mạng
client = DePeftClient("http://127.0.0.1:8545")

# 2. Tạo khóa tài khoản cho bot tự hành
account = generate_keypair()
print(f"Địa chỉ ví:  {account['account_id']}")
print(f"Khóa bí mật: {account['secret_key']}")

# 3. Yêu cầu cấp token từ Faucet
client.request_faucet(account["account_id"], 20000)

# 4. Kiểm tra trạng thái mạng và số dư ví
status = client.get_status()
balance = client.get_balance(account["account_id"])
print(f"Block #{status['block_height']} | Số task: {status['tasks_count']} | Số dư: {balance} $DEPEFT")

# 5. Xem danh sách các task đang hoạt động
tasks = client.get_tasks()
for task in tasks:
    print(f"Task #{task['task_id']}: Mô hình={task['base_model_id_str']} | Tiền thưởng={task['bounty_pool']}")
```

---

## 📡 Bảng Tra Cứu API REST / JSON-RPC

Mỗi Node DePEFT mở một cổng API tại `http://127.0.0.1:8545`:

| Phương thức | Đường dẫn API | Mục đích sử dụng |
|---|---|---|
| `GET` | `/api/v1/status` | Xem chiều cao khối, số task, số file lưu trữ, số máy ngang hàng |
| `GET` | `/api/v1/tasks` | Lấy danh sách tất cả các task `TaskSpec` trên mạng |
| `GET` | `/api/v1/tasks/:id` | Xem chi tiết thông số của task theo ID |
| `GET` | `/api/v1/tasks/:id/rounds/:round` | Xem trạng thái vòng thi đấu (Pha, danh sách miner đã reveal) |
| `POST` | `/api/v1/tx` | Gửi giao dịch đã ký (`CreateTask`, `CommitAdapter`, `RevealAdapter`, `SubmitEvaluation`) |
| `GET` | `/api/v1/accounts/:account/balance` | Tra cứu số dư ví và mã chống gửi trùng lặp (nonce) |
| `POST` | `/api/v1/faucet` | Nhận token thử nghiệm (bật khi có cờ `--enable-faucet`) |
| `POST` | `/api/v1/storage` | Tải dữ liệu nhị phân lên kho lưu trữ CAS (trả về mã CID) |
| `GET` | `/api/v1/storage/:cid` | Tải file trọng số `.safetensors` hoặc dữ liệu qua mã CID |
| `GET` | `/api/v1/p2p/peers` | Danh sách địa chỉ IP các node P2P đang kết nối |
| `POST` | `/api/v1/p2p/connect` | Yêu cầu node chủ động kết nối tới một peer P2P khác |

---

## 🎮 Các Lệnh Chạy Mô Phỏng Trực Quan

DePEFT cung cấp sẵn các bộ lệnh mô phỏng giúp bạn thử nghiệm nhanh mọi tầng kiến trúc chỉ trên 1 máy duy nhất:

### 1. Mô phỏng Giải đấu ReLoRA 4 Tầng Đầy Đủ
Mô phỏng 3 round thi đấu với 3 thợ đào và 3 validator, kiểm tra toàn bộ quy trình commit-reveal, Borda count và tiến hóa trọng số:
```bash
cargo run -- demo --rounds 3 --peft qlora-nf4 --miners 3 --validators 3
```

### 2. Huấn luyện Mô hình Transformer Ngôn Ngữ Thực Tế
Huấn luyện mô hình Decoder Transformer bằng thư viện Hugging Face Candle với thuật toán lan truyền ngược autograd và xuất file SafeTensors:
```bash
cargo run -- llm-demo --rounds 3 --steps 15 --device auto
```

### 3. Mô phỏng Đồng thuận BFT & Phạt Gian Lận
Mô phỏng 4 validator tham gia bỏ phiếu CometBFT 2-phase commit và tịch thu cọc (slash) khi phát hiện validator gian lận:
```bash
cargo run -- bft-demo --validators 4 --blocks 5
```

### 4. Kiểm tra Chữ ký Chứng thực Phần cứng TEE
Tạo quote chứng thực bên trong Intel SGX / AMD SEV và kiểm tra việc xác thực tính hợp lệ trên chuỗi:
```bash
cargo run -- tee-quote
```

### 5. Đo lường Hiệu Quả Lượng Tử Hóa (Benchmark)
Đo tốc độ và độ lệch sai số giữa FP32, QLoRA NF4 (4-bit) và INT4 (4-bit):
```bash
cargo run -- benchmark
```

---

## 🏛️ Kiến Trúc 4 Tầng Cốt Lõi

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Tầng 1: Blockchain & Cơ chế Đồng thuận (App-Chain & CometBFT)                             │
│    • Sổ cái tất định: Quản lý số dư, Nonce (chống phát lại), Quỹ Escrow, TaskSpec        │
│    • Đồng thuận CometBFT 2-Phase Commit: Propose -> Prevote -> Precommit -> Commit        │
│    • Đồng thuận thứ hạng Borda Count: Miễn nhiễm với sai số thập phân giữa các GPU        │
│    • Trình xác thực TEE trên chuỗi: Quản lý whitelist MRENCLAVE / MRSIGNER                │
└────────────────────────────┬──────────────────────────────────────▲───────────────────────┘
                             │                                      │
                             ▼                                      │
┌───────────────────────────────────────────┐      ┌────────────────┴───────────────────────┐
│ Tầng 2: Mạng lưới Thợ đào (Miner)         │      │ Tầng 3: Kiểm định & TEE (Validator)    │
│    • Deep Learning bằng HF Candle         │      │    • Đánh giá trên tập kiểm thử kín    │
│    • Huấn luyện QLoRA (NF4 / INT4)        │      │    • Chữ ký chứng thực phần cứng TEE   │
│    • Đóng gói file chuẩn SafeTensors      │      │    • Phát hiện sao chép bằng Vector DB │
│    • Mã băm Commit-Reveal chống gian lận  │      │    • Đồng thuận thứ hạng Borda Count   │
└────────────────────────────┬──────────────┘      └────────────────▲───────────────────────┘
                             │                                      │
                             ▼                                      │
┌───────────────────────────────────────────────────────────────────┴───────────────────────┐
│ Tầng 4: Hệ thống Lưu trữ & Mạng P2P                                                       │
│    • Lưu trữ lai: Bộ nhớ đệm SSD cục bộ (~/.depeft/storage) + Mạng phân tán IPFS Kubo     │
│    • Mạng P2P TCP bất đồng bộ với cơ chế lan truyền tin, chống trùng và chống DoS         │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Kiểm Thử & Đảm Bảo Chất Lượng

Toàn bộ hệ sinh thái DePEFT được kiểm tra nghiêm ngặt:

```bash
# Chạy toàn bộ 45 bài kiểm tra tự động (unit test & integration test)
cargo test

# Đảm bảo mã nguồn chuẩn xác tuyệt đối không có cảnh báo nào
cargo clippy --all-targets -- -D warnings
```

---

## 📄 Bản Quyền Mã Nguồn

- **Mã nguồn Blockchain & Node (`src/`, `Cargo.toml`)**: Giấy phép [GNU AGPL-3.0](./LICENSE).
- **Thư viện Python SDK (`sdk/python/`)**: Giấy phép kép [Apache-2.0](./sdk/python/LICENSE-APACHE) HOẶC [MIT](./sdk/python/LICENSE-MIT).
- **Hợp đồng thông minh Smart Contracts (`contracts/`)**: Giấy phép kép [Apache-2.0](./contracts/LICENSE-APACHE) HOẶC [MIT](./contracts/LICENSE-MIT).
