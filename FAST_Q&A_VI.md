# ⚡ DePEFT Fast Q&A : Cẩm Nang Cho Trader, Nhà Đầu Tư DePIN & Người Vận Hành Node

> [!WARNING]
> ### 🚨 CẢNH BÁO AN TOÀN / DỰ ÁN ĐANG CHẠY TESTNET
> **DEPEFT HIỆN TẠI ĐANG TRONG GIAI ĐOẠN TESTNET.** Chúng tôi **CHƯA** phát hành bất kỳ token, hợp đồng thông minh hay bể thanh khoản nào trên Mainnet (Ethereum, Base, Solana, BNB Chain hay bất kỳ sàn DEX/CEX nào). Mọi token tự xưng là $DEPEFT trên mainnet ở thời điểm này đều là lừa đảo giả mạo. Hãy luôn kiểm chứng thông tin trực tiếp với đội ngũ phát triển chính thức.

---

## 🧭 Tóm Tắt Dành Cho Nhà Đầu Tư & Trader Crypto
- **Lĩnh vực:** Trí tuệ Nhân tạo Phi tập trung (DeAI) / Hạ tầng Vật lý Phi tập trung (DePIN).
- **Giá trị Cốt lõi:** Cắt giảm 60% – 80% chi phí tinh chỉnh (Fine-Tuning) mô hình AI bằng cách huy động mạng lưới GPU cá nhân toàn cầu (QLoRA) và cơ chế tiến hóa trọng số đa vòng (ReLoRA), được xác thực bằng phần cứng TEE (Intel SGX / AMD SEV).
- **Kiến trúc Blockchain & Token:** Chuỗi khối App-Chain riêng biệt viết bằng Rust vận hành trên thuật toán đồng thuận 2 pha CometBFT, kết nối cầu nối (Bridge) sang chuẩn ERC-20 / SPL để giao dịch thanh khoản trên thị trường thứ cấp.

---

## 💰 1. Kinh Tế Token (Tokenomics), Nguồn Cung & Lực Cầu Giá Trị

### Q1: Có cơ chế đốt token (Burn) không? Làm thế nào $DEPEFT đạt được giảm phát?
**A:** Có. DePEFT tích hợp **Cơ chế Đốt Token Giảm Phát Động (Dynamic Deflationary Burn)** trực tiếp trong nhân blockchain:
$$\text{BurnRate} = \text{clamp}\left(0.005 + (N_{\text{tasks}} - 1) \times 0.015,\ 0.005,\ 0.10\right)$$
- Khi nhu cầu tinh chỉnh AI của các doanh nghiệp bùng nổ ($\ge 8$ nhiệm vụ chạy cùng lúc), **tối đa 10% toàn bộ quỹ tiền thưởng (bounty) sẽ bị tiêu hủy vĩnh viễn** khỏi tổng cung lưu thông.
- Khi nhu cầu khan hiếm (ví dụ chỉ có 1 nhiệm vụ), tỷ lệ đốt tự động hạ xuống $\approx 0.5\%$ để dồn tối đa $99.5\%$ tiền thưởng nuôi sống thợ đào và người duy trì mạng lưới.
- **Góc nhìn đầu tư:** Càng nhiều doanh nghiệp, bệnh viện, startup dùng DePEFT để huấn luyện AI thì tốc độ đốt token càng tăng nhanh, tạo áp lực khan hiếm đẩy giá trị token tăng trưởng.

---

### Q2: Tiền thưởng (Bounty) được phân chia như thế nào? Thợ đào có xả hết token kiếm được không?
**A:** DePEFT giải quyết triệt để "vòng xoáy tử thần" (death spiral) của các dự án DePIN bằng **Mô hình Đàn hồi Cung - Cầu 3 Tầng**:
1. **Tầng 1: Node Cổng Lưu trữ & Mạng IPFS (5% – 15%):** Đàn hồi tăng dần theo lưu lượng truyền tải và kích thước các file adapter `.safetensors`.
2. **Tầng 2: Hardware TEE Validators (15% – 45%):** Tự động tăng vọt khi số lượng máy chủ TEE bảo mật khan hiếm so với lượng thợ đào ($\frac{\text{Miners}}{\text{Validators}}$).
3. **Tầng 3: Thợ đào Cạnh tranh (40% – 80%):** Chỉ trao thưởng cho các mô hình có độ cải thiện loss xuất sắc nhất theo thuật toán xếp hạng Borda Count.

Ngoài ra, **các Validator bắt buộc phải stake (khóa) token** làm tài sản thế chấp bảo lãnh, rút bớt lượng lớn token ra khỏi áp lực bán trên thị trường.

---

### Q3: Ai là người bỏ tiền thật ra mua token? Nguồn cầu tự nhiên đến từ đâu?
**A:** Nhu cầu mua $DEPEFT đến từ 5 nhóm đối tượng kinh tế thực tế:
1. **Khách hàng Doanh nghiệp & AI Startups (Clients):** Mua $DEPEFT nạp vào quỹ ký quỹ (escrow) để thuê mạng lưới fine-tune mô hình.
2. **TEE Validators & Consensus Nodes:** Mua và stake $DEPEFT làm tài sản thế chấp để có quyền tham gia chấm điểm và nhận phí xác thực.
3. **Storage Gateways:** Stake token cam kết đảm bảo lưu trữ và giữ gìn các file trọng số AI an toàn trên IPFS.
4. **Người dùng Tải Mô hình (Inference Marketplace):** Trả phí bản quyền bằng $DEPEFT để tải các bộ trọng số AI chất lượng cao đã qua huấn luyện về sử dụng thương mại.
5. **Người nắm giữ Token (Delegators):** Ủy quyền (stake) token vào các Validator uy tín để cùng chia sẻ lợi nhuận thụ động.

---

## 🔒 2. Bảo Mật, Đồng Thuận & Cơ Chế Chống Gian Lận

### Q4: Làm thế nào chống thợ đào ăn cắp bài của nhau (Front-Running / Sao chép trọng số)?
**A:** DePEFT áp dụng **Quy trình 2 Pha Cam kết - Công bố (2-Phase Commit-Reveal)** nghiêm ngặt:
1. **Pha Commit:** Thợ đào chỉ được gửi một chuỗi mã băm ẩn duy nhất:  
   $$\text{CommitHash} = \text{SHA256}(\text{AdapterHash} \mathbin{\Vert} \text{salt})$$
2. **Pha Reveal:** Chỉ sau khi hạn chót commit đóng lại, thợ đào mới tải file `.safetensors` lên mạng và công bố chuỗi bí mật (salt).
3. **Cơ sở Dữ liệu Vector Nhúng:** Trước khi giải ngân tiền thưởng, Vector Database on-chain tính toán chữ ký tương đồng Cosine giữa tất cả các mô hình. Bất kỳ adapter nào bị phát hiện sao chép ($\text{Similarity} > 0.98$) sẽ bị loại bỏ và tước quyền nhận thưởng ngay lập tức.

---

### Q5: Ngăn chặn Validator thông đồng buff điểm cho thợ đào "sân sau" bằng cách nào?
**A:** Hệ thống được bảo vệ bằng nhiều lớp phòng thủ mật mã độc lập:
1. **Xác thực Phần cứng TEE Từ xa (Remote Attestation):** Quá trình đánh giá chạy bên trong "hộp đen phần cứng" (Intel SGX / AMD SEV-SNP). Chip xử lý ký số một `AttestationQuote` ràng buộc trực tiếp mã đo đạc MRENCLAVE với bảng xếp hạng (`report_data`).
2. **Đồng thuận Thứ hạng Tương đối (Borda Count):** Thay vì tin vào điểm loss số thực (vốn có sai số trôi dạt giữa các dòng GPU), chuỗi khối tổng hợp thứ tự xếp hạng tương đối của nhiều validator độc lập.
3. **Cơ chế Phạt Chém Stake (Slashing Engine):** Bất kỳ validator nào ký hai phiếu bầu xung đột hoặc gian lận chữ ký sẽ bị chém cọc và tước quyền biểu quyết ngay lập tức.

---

## ⚡ 3. Công Nghệ & Hạ Tầng Mạng Lưới

### Q6: Thợ đào cắm máy tại nhà bị kẹt sau CGNAT (không có IPv4 tĩnh, không mở cổng được) có đào được không?
**A:** **Hoàn toàn được!** DePEFT được thiết kế tối ưu cho cộng đồng thợ đào cá nhân:
- **Hỗ trợ Mạng kép Native IPv6:** Các máy có IPv6 tự động kết nối ngang hàng (P2P) trực tiếp không bị cản trở bởi NAT.
- **Tuyến P2P Circuit Relay (`RelayForward` / `RelayPayload`):** Đối với các máy bị kẹt sau IPv4 CGNAT đối xứng, dữ liệu được chuyển tiếp an toàn qua các Node Relay công khai thông qua kết nối Outbound TCP mà không cần mở cổng trên router.

---

### Q7: Mạng lưới sử dụng Engine Trí tuệ Nhân tạo nào?
**A:** DePEFT chạy trên nền tảng deep learning bằng Rust thuần túy với hiệu năng tối đa:
- Xây dựng trên nền **Hugging Face Candle** hỗ trợ lượng tử hóa khối QLoRA NF4 (4-bit) và INT4.
- Hỗ trợ tăng tốc phần cứng đa nền tảng: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal) và CPU AVX-512.
- **ReLoRA đa vòng:** Liên tục gộp trọng số adapter vào mô hình nền ($W_{N+1} = W_N + \Delta W^*$) mà không làm tràn bộ nhớ VRAM hay bùng nổ tham số.

---

## 🚀 4. Lộ Trình, Giao Dịch & Niêm Yết Sàn

### Q8: Khi nào lên Mainnet? Giao dịch và cung cấp thanh khoản ở đâu?
**A:** 
1. **Giai đoạn Hiện tại:** Chạy Testnet công khai với đầy đủ node daemon, vòi faucet, giao diện Web Dashboard và quy trình huấn luyện QLoRA thực tế.
2. **Lộ trình Khởi chạy Mainnet:**
   - Hoàn thành kiểm toán bảo mật độc lập và khởi tạo Genesis Block.
   - Triển khai hợp đồng Cầu nối (Bridge) chuẩn ERC-20 / SPL trên các mạng EVM lớn (Arbitrum/Base/Ethereum) hoặc Solana.
   - Tạo Bể thanh khoản ban đầu (Liquidity Pool) trên các sàn phi tập trung hàng đầu (Uniswap V3, Raydium) và khóa thanh khoản (Lock LP).
   - Nộp hồ sơ hiển thị trên CoinGecko, CoinMarketCap và niêm yết các sàn CEX tập trung (Tier-3/Tier-2).

---

### Q9: Các chuyên gia bảo mật và người dùng báo cáo lỗi ở đâu?
**A:** Mọi thông tin về lỗ hổng bảo mật, lỗi khai thác hay bất kỳ vấn đề an toàn nào **BẮT BUỘC phải gửi độc quyền qua Tin nhắn Trực tiếp (DM) trên Discord** cho đội ngũ phát triển chính thức. Tuyệt đối KHÔNG đăng công khai lên GitHub Issues hay gửi qua email. Vui lòng xem chi tiết tại [`SECURITY_VI.md`](./SECURITY_VI.md).
