# 🤝 Hướng dẫn Đóng góp cho DePEFT

Cảm ơn bạn đã quan tâm và muốn đóng góp cho dự án **DePEFT**!

DePEFT là dự án mã nguồn mở xây dựng blockchain chuyên dụng kết hợp huấn luyện mô hình AI phân tán, đảm bảo tính công bằng và bảo mật bằng phần cứng TEE, thư viện Hugging Face Candle và cơ chế đồng thuận CometBFT.

---

## 🧭 Các Quy tắc Cốt lõi Cần Nhớ

Khi viết mã nguồn đóng góp, bạn cần tuân thủ các nguyên tắc sau:

1. **Giữ nguyên mô hình gốc**: Trọng số mô hình ban đầu $W_0$ phải được giữ nguyên hoàn toàn khi thợ đào huấn luyện. Chỉ cập nhật hai ma trận nhỏ $A$ và $B$.
2. **Bảo mật dữ liệu kiểm thử**: Bộ dữ liệu dùng để chấm điểm mô hình **tuyệt đối không** được gửi ra ngoài mạng hoặc in ra màn hình. Dữ liệu này chỉ được mở bên trong vùng bảo mật TEE.
3. **Xếp hạng thay vì tính điểm trung bình**: Vì các loại card màn hình khác nhau có thể tạo ra sai số số thực siêu nhỏ, hệ thống bắt buộc phải dùng thứ hạng Borda Count để tính điểm cho thợ đào.
4. **Quy trình nộp bài 2 bước (Commit-Reveal)**: Thợ đào phải nộp mã băm `commit_hash` trước để khóa bài, sau đó mới tải file `.safetensors` lên mạng để tránh bị người khác nhìn trộm bài.
5. **Ký chữ ký điện tử cho mọi giao dịch**: Mọi thay đổi dữ liệu trên chuỗi đều phải có chữ ký Ed25519 hợp lệ từ ví người gửi.
6. **Chốt khối khi đủ 2/3 phiếu bầu**: Khối mới chỉ được lưu khi có hơn 2/3 số validator xác nhận hợp lệ.

---

## 🛠️ Cài đặt Môi trường Làm việc

### Yêu cầu
- **Rust**: Phiên bản 1.80 trở lên (bản Stable)
- **Cargo**: Tự động có sẵn khi cài Rust (`rustup default stable`)
- **IPFS Kubo**: (Tùy chọn nếu muốn thử kết nối mạng IPFS thật, mặc định hệ thống tự lưu trên ổ cứng)

### Tải mã nguồn và Biên dịch
```bash
git clone https://github.com/KhoaIsReal/DePEFT.git
cd DePEFT

# Biên dịch toàn bộ dự án
cargo build

# Chạy thử toàn bộ các bài kiểm tra
cargo test
```

---

## 🧪 Quy định về Viết và Chạy Kiểm thử (Test)

Mọi tính năng mới hoặc bản sửa lỗi đều cần có bài kiểm tra (test) đi kèm để bảo đảm hệ thống luôn hoạt động ổn định.

### Cách chạy Test
```bash
# Chạy tất cả các bài test
cargo test

# Chạy test và hiện chi tiết nếu có lỗi
RUST_BACKTRACE=1 cargo test

# Chạy riêng một bài test cụ thể
cargo test test_candle_llm_transformer_relora_tournament -- --nocapture
```

---

## 📐 Phong cách và Tiêu chuẩn Viết Code

1. **Định dạng code tự động**: Chạy lệnh kiểm tra định dạng chuẩn của Rust:
   ```bash
   cargo fmt --check
   ```
2. **Kiểm tra cảnh báo bằng Clippy**: Đảm bảo code sạch sẽ, không có bất kỳ cảnh báo nào:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
3. **Xử lý lỗi rõ ràng**: Sử dụng `anyhow::Result` kèm thông báo dễ hiểu. Hạn chế tối đa việc dùng `.unwrap()` trực tiếp để tránh làm chương trình bị dừng đột ngột.
4. **Viết chú thích (Comment)**: Thêm mô tả cho các hàm, struct và enum quan trọng.

---

## 🚀 Các bước Gửi Đóng góp (Pull Request)

1. **Fork dự án** về tài khoản GitHub của bạn và tạo một nhánh mới:
   ```bash
   git checkout -b feature/tinh-nang-moi
   ```
2. **Thực hiện chỉnh sửa** theo đúng quy chuẩn ở trên.
3. **Chạy lại kiểm tra** để đảm bảo mọi thứ hoạt động bình thường:
   ```bash
   cargo test
   ```
4. **Commit mã nguồn** với nội dung rõ ràng:
   - `feat: thêm thuật toán tối ưu hóa mới`
   - `fix: sửa lỗi mất kết nối mạng P2P`
   - `docs: cập nhật tài liệu hướng dẫn`
5. **Tạo Pull Request (PR)** trên GitHub mô tả chi tiết những gì bạn đã làm.

---

## 🔒 Báo cáo Lỗi Bảo mật

Nếu bạn phát hiện lỗi liên quan đến bảo mật (như lỗi mã hóa, lỗ hổng TEE hoặc lỗi đồng thuận), tuyệt đối KHÔNG đăng công khai lên mục GitHub Issues hay gửi email. Toàn bộ thông tin **BẮT BUỘC phải gửi độc quyền qua Tin nhắn Trực tiếp (DM) trên Discord** cho đội ngũ phát triển/core team và **phải được mã hóa bằng PGP Public Key chính thức** ([`depeft_security_pubkey.asc`](./depeft_security_pubkey.asc)). Vui lòng tham khảo tài liệu [SECURITY_VI.md](file:///home/khoa/DePEFT/DePEFT/SECURITY_VI.md) để biết thêm chi tiết.

