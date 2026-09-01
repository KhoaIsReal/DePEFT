# 🤝 Hướng dẫn Đóng góp cho DePEFT

Cảm ơn bạn đã quan tâm và muốn đóng góp cho **DePEFT** (Decentralized Parameter-Efficient Fine-Tuning)!

DePEFT là một app-chain AI phi tập trung mã nguồn mở, mang đến giải pháp tinh chỉnh mô hình đa vòng LoRA/QLoRA có thể mở rộng và xác thực được cho các mạng lưới phi tập trung, kết hợp với xác thực phần cứng TEE, autograd Hugging Face Candle, lưu trữ IPFS và cơ chế đồng thuận CometBFT.

---

## 🧭 Nguyên tắc Kiến trúc & Bất biến Bắt buộc

Khi đóng góp mã nguồn, bạn phải tuân thủ nghiêm ngặt các nguyên tắc kiến trúc cốt lõi sau:

1. **Đóng băng trọng số mô hình nền tảng**: Trọng số mô hình gốc $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ phải được đóng băng hoàn toàn trong suốt quá trình huấn luyện cục bộ của thợ đào. Chỉ các ma trận thứ hạng thấp $A$ và $B$ mới nhận cập nhật gradient.
2. **Cách ly môi trường Sandbox TEE**: Tập dữ liệu kiểm thử riêng tư (Private test sets) **tuyệt đối không** được truyền qua mạng hoặc ghi log dưới dạng văn bản thô. Chúng phải được bảo vệ và niêm phong tuyệt đối bên trong Enclave Sandbox TEE.
3. **Đồng thuận tương đối tất định**: Điểm loss từ validator có thể có độ trôi vi mô do phần cứng không đồng nhất (CUDA, ROCm, AVX-512). App-Chain **không bao giờ** được lấy trung bình cộng trực tiếp các giá trị loss dấu phẩy động; blockchain bắt buộc phải đánh giá thứ hạng tương đối bằng cách tổng hợp xếp hạng Borda Count.
4. **Cơ chế chống thông đồng Commit-Reveal**: Thợ đào bắt buộc phải gửi `commit_hash = SHA256(adapter_hash || salt)` trong Pha Commit trước khi tải tệp adapter `.safetensors` lên trong Pha Reveal.
5. **Biến đổi trạng thái qua xác thực mật mã**: Không giao dịch nào được phép thay đổi `AppChainState` nếu không có chữ ký Ed25519 hợp lệ khớp với khóa công khai của người gửi.
6. **Tính tất định BFT**: Các khối chỉ được ghi vào sổ cái bất biến khi thu thập đủ đa số tuyệt đối $> 2/3$ phiếu bầu mật mã `PRECOMMIT`.

---

## 🛠️ Thiết lập Môi trường Phát triển & Bộ công cụ

### Yêu cầu tiên quyết
- **Rust Toolchain**: 1.80+ (Bản Stable)
- **Cargo**: Đi kèm sẵn với Rust (`rustup default stable`)
- **IPFS Kubo Daemon** (Tùy chọn cho các bài test IPFS thực tế, mặc định sử dụng CAS đĩa cứng nội bộ)

### Clone & Build
```bash
git clone https://github.com/khoadepeft/DePEFT.git
cd DePEFT

# Biên dịch tất cả các module và file thực thi
cargo build

# Chạy toàn bộ bộ kiểm thử
cargo test
```

---

## 🧪 Quy chuẩn Kiểm thử (Testing Guidelines)

Tất cả các tính năng mới, bản vá lỗi và cải tiến giao thức đều phải có các bài kiểm thử đơn vị (unit tests) và kiểm thử tích hợp (integration tests) tương ứng.

### Chạy Tests
```bash
# Chạy tất cả tests
cargo test

# Chạy tests kèm backtrace chi tiết
RUST_BACKTRACE=1 cargo test

# Chạy một bài test cụ thể
cargo test test_candle_llm_transformer_relora_tournament -- --nocapture
```

### Các mục tiêu kiểm thử tích hợp chính:
- [`tests/integration_tests.rs`](file:///home/khoa/DePEFT/tests/integration_tests.rs):
  - `test_candle_lora_linear_forward_and_merge`: Kiểm tra toán học LoRA và hợp nhất SafeTensors.
  - `test_candle_llm_transformer_relora_tournament`: Vòng lặp huấn luyện đầy đủ 3 vòng của Candle LLM Transformer.
  - `test_hardware_tee_remote_attestation_and_on_chain_verification`: Xác thực quote phần cứng & cơ chế chống gian lận.
  - `test_hybrid_storage_and_ipfs_cas_caching`: Đồng bộ CAS đĩa cục bộ + IPFS Kubo trực tiếp.
  - `test_bft_consensus_2_phase_commit_and_equivocation_slashing`: Đồng thuận BFT 2-phase commit và phạt (slashing) validator bỏ phiếu hai lần.
  - `test_p2p_swarm_bidirectional_gossip_and_deduplication`: Đóng khung TCP length-delimited & lan truyền gossip.

---

## 📐 Chuẩn Mã nguồn & Phong cách Rust

1. **Định dạng code (Formatting)**: Đảm bảo code được format chuẩn bằng `rustfmt`:
   ```bash
   cargo fmt --check
   ```
2. **Clippy (Yêu cầu Zero Warnings)**: Chạy Clippy với cờ xử lý cảnh báo thành lỗi và đảm bảo không có lỗi:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
3. **Xử lý lỗi (Error Handling)**: Sử dụng `anyhow::Result` cùng các thông báo lỗi rõ ràng qua `ensure!` hoặc `bail!` trong logic nghiệp vụ. Tránh dùng `.unwrap()` trực tiếp trong các luồng code production.
4. **Tài liệu hóa (Documentation)**: Thêm chú thích Rustdoc (`///`) cho tất cả các struct, enum, trait và hàm công khai (public).

---

## 🚀 Quy trình Gửi Pull Request (PR)

1. **Fork repository** và tạo một nhánh tính năng (feature branch) từ `main`:
   ```bash
   git checkout -b feature/my-new-peft-optimizer
   ```
2. **Thực hiện thay đổi** tuân thủ theo các tiêu chuẩn code và nguyên tắc bất biến của kiến trúc.
3. **Đảm bảo toàn bộ tests đều vượt qua**:
   ```bash
   cargo test
   ```
4. **Commit với thông điệp theo chuẩn Conventional Commits**:
   - `feat(candle): add FlashAttention support to Transformer LM`
   - `fix(p2p): handle socket reconnection on dropped TCP stream`
   - `docs(agents): update REST API spec for task creation`
5. **Mở Pull Request** mô tả chi tiết các thay đổi, lý do thực hiện và phạm vi kiểm thử đã bổ sung.

---

## 🔒 Báo cáo Lỗ hổng Bảo mật

Vui lòng không báo cáo công khai các lỗ hổng bảo mật lên mục GitHub Issues. Nếu bạn phát hiện vấn đề liên quan đến chữ ký mật mã, xác thực TEE remote attestation hoặc an toàn đồng thuận BFT, vui lòng xem [SECURITY_VI.md](file:///home/khoa/DePEFT/SECURITY_VI.md) và liên hệ trực tiếp với đội ngũ duy trì bảo mật cốt lõi.
