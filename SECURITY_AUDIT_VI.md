# 🛡️ Sổ Cái Kiểm Toán Bảo Mật DePEFT

**Phạm vi:** Rust App-Chain Node, Đồng thuận CometBFT, Mạng P2P Gossip & Circuit Relay, Lưu trữ IPFS CAS, Xác thực Phần cứng TEE, Machine Learning Engine (Candle QLoRA), Python SDK và Hợp đồng Thông minh (Smart Contracts).  
**Nguyên tắc kiểm toán:** Các mục chỉ được gạch bỏ khi cả mã nguồn thực tế và bộ kiểm thử hồi quy (regression tests) chứng minh được biện pháp bảo vệ về mặt toán học.

---

## ✅ Các Biện Pháp Đã Được Kiểm Chứng & Khắc Phục (Chứng minh bởi 45 bài Test)

- ~~**Giả mạo Chữ ký Giao dịch (Transaction Signature Forgery):**~~ Bắt buộc xác thực chữ ký Ed25519 nghiêm ngặt trên mọi giao dịch.
- ~~**Không Khớp Địa chỉ Người gửi (Inner Sender Mismatch):**~~ Giao dịch bị từ chối nếu trường người gửi bên trong (`sender`/`client`/`miner`) không khớp với khóa công khai Ed25519 đã ký.
- ~~**Tấn công Phát lại qua Nonce (Replay Attack via Nonces):**~~ Cơ chế kiểm tra nonce tuần tự nghiêm ngặt cho từng tài khoản loại bỏ hoàn toàn các giao dịch phát lại.
- ~~**Nộp Trùng Lặp Commit và Reveal (Duplicate Commit/Reveal):**~~ Cơ chế máy trạng thái từ chối hành vi commit 2 lần hoặc gửi reveal trùng lặp của mỗi miner trong cùng một vòng đấu.
- ~~**Xếp hạng Đánh giá Sai Lệch (Evaluation Ranking Malformation):**~~ Xếp hạng của validator bị từ chối nếu chứa miner lạ, thiếu miner đã reveal hoặc chứa miner trùng lặp.
- ~~**Bằng chứng TEE Hết hạn hoặc ở Tương lai (Expired/Future TEE Quotes):**~~ Bằng chứng phần cứng TEE được xác thực với đồng hồ đồng thuận đơn điệu; từ chối quote có timestamp ở tương lai hoặc vượt ngưỡng trôi dạt tối đa.
- ~~**Giả mạo Chữ ký Bỏ phiếu BFT (Forged BFT Vote Signatures):**~~ Phiếu Prevote/Precommit của CometBFT bắt buộc phải được ký bởi khóa công khai hợp lệ trong tập hợp validator đã đăng ký.
- ~~**Validator Trùng Lặp trong Tập Hợp Đồng thuận (Duplicate Validators):**~~ Ngăn chặn triệt để hành vi trùng lặp validator và xác thực trọng số stake chính xác.
- ~~**Giới hạn Frame P2P & Chống Tấn công Slowloris:**~~ Giới hạn kích thước gói tin tối đa (64MB) và timeout trong bộ giải mã mạng ngăn ngừa cạn kiệt bộ nhớ.
- ~~**Lỗi Tiêu đề & Giới hạn Offset của SafeTensors:**~~ Trình phân tích SafeTensors zero-copy từ chối các offset byte vượt biên, kích thước âm, hoặc trọng số chứa giá trị NaN/Inf.
- ~~**Tấn công Điều hướng Đường dẫn Bộ nhớ (Storage Path Traversal):**~~ Hệ thống lưu trữ phân tán CAS xác thực định dạng multihash và từ chối các ký tự điều hướng (`..`, `/`, `\`).
- ~~**Bảo vệ Chiều Dữ liệu & NaN trong Vector DB:**~~ Bộ so khớp Cosine Similarity làm sạch dữ liệu đầu vào, từ chối vector sai chiều hoặc số thực không hợp lệ.
- ~~**Kích thước Khối Lượng tử hóa Bằng Không (Quantization Zero Block Size):**~~ Thuật toán lượng tử hóa NF4 và INT4 kiểm tra và chặn lỗi chia cho 0.
- ~~**Gốc Tin cậy TEE Chế độ Production Fail-Closed:**~~ Node chạy production từ chối các khóa giả lập; nếu thiếu cấu hình gốc phần cứng thực tế thì hệ thống sẽ tự động đóng và từ chối (fail-closed).
- ~~**Cô lập Endpoint Quản trị và Vòi Faucet (Endpoints Isolation):**~~ Các route nhạy cảm của người vận hành và faucet testnet bị tắt mặc định ở môi trường production; loại bỏ hoàn toàn cấu hình CORS lỏng lẻo.
- ~~**Lưu trữ Bền vững & Giao dịch WAL Nguyên tử (Durable Persistence):**~~ Cơ chế SQLite Write-Ahead Logging (WAL) lưu snapshot trạng thái cùng mã băm nguyên tử; tự động rollback nếu quá trình lưu trữ thất bại.
- ~~**Mạng Kép IPv6 Dual-Stack & Vượt Rào CGNAT (P2P Circuit Relay):**~~ Socket lắng nghe `[::]` và giao thức P2P Circuit Relay cho phép các thợ đào bị kẹt sau CGNAT của nhà mạng kết nối thông suốt.
- ~~**Bảo toàn Phân bổ Bounty 3 Tầng & Đốt Token Giảm Phát Động:**~~ Bảo toàn chính xác tuyệt đối về mặt toán học ($\sum \text{Balances} + \text{Burned} = \text{Total Bounty}$) theo nguyên lý đàn hồi cung - cầu.

---

## 🤖 Kết Quả Kiểm Toán Tự Động Bằng Bộ Công Cụ Mã Nguồn Mở (Zero-Đồng)

Mã nguồn DePEFT được quét và bảo vệ liên tục bằng các công cụ phân tích tĩnh bảo mật chuẩn công nghiệp:

| Công Cụ Kiểm Toán | Phân Hệ Mục Tiêu | Phạm Vi & Quy Chuẩn | Trạng Thái Kiểm Toán |
|---|---|---|---|
| **`cargo audit` (RustSec)** | Blockchain Node & Thư viện phụ thuộc | Quét **380 crate phụ thuộc** đối chiếu cơ sở dữ liệu RustSec Advisory Database. | ✅ **0 Lỗ Hổng (0 Vulnerabilities)** |
| **`cargo clippy`** | Toàn bộ mã nguồn Rust (`src/`, `tests/`) | Chạy cờ nghiêm ngặt nhất `-D warnings` kiểm soát an toàn bộ nhớ và chống deadlock. | ✅ **0 Cảnh Báo (0 Warnings)** |
| **`slither` (Trail of Bits)** | Hợp đồng Thông minh (`contracts/`) | Quét `DePeftToken.sol` & `DePeftEscrow.sol` với 102 bộ phát hiện lỗ hổng (Reentrancy, CEI, Gas). | ✅ **Sạch Lỗi Nghiêm Trọng (0 Critical/High/Medium)** |

Các biện pháp bảo vệ Hợp đồng thông minh đã được Slither xác thực:
- Áp dụng mẫu thiết kế **Checks-Effects-Interactions (CEI)** trên toàn bộ các hàm nạp tiền và hoàn tiền trong `DePeftEscrow.sol`, loại trừ triệt để nguy cơ tấn công tái vào (Reentrancy).
- Khai báo các biến trạng thái dưới dạng `constant` và `immutable` giúp tối ưu dung lượng bytecode EVM và tiết kiệm tối đa phí gas giao dịch.
- Khóa chặt trình biên dịch Solidity phiên bản `0.8.26`, ngăn ngừa các lỗi biên dịch ở tầng hạ tầng.

---

## 🔍 Lộ Trình Nâng Cấp & Hoàn Thiện Tiếp Theo

- [medium] **Gắn State Root của Khối BFT:** Đưa các giao dịch HTTP đi trực tiếp qua các khối đề xuất CometBFT đã finalize kèm theo Merkle state root.
- [medium] **Ràng buộc Mã băm Adapter khi Gộp Trọng số:** Kiểm tra tính toàn vẹn của file adapter tải từ IPFS so với mã băm SHA-256 đã commit on-chain trước khi gộp ReLoRA.
- [medium] **Kiểm tra Tính toàn vẹn Dataset & Model trước khi chạy:** Kiểm tra mã băm của mô hình nền và dữ liệu trước khi nạp vào bộ nhớ.
- [low] **Giới hạn Tần suất & Hạn ngạch trên Public RPC:** Áp dụng rate-limiting và giới hạn băng thông tải lên IPFS cho từng tài khoản.
- [low] **Mã hóa Giao tiếp P2P (Noise Protocol):** Nâng cấp kênh truyền TCP thô sang giao thức TLS/Noise có xác thực hai chiều.
- [low] **Tự động Phạt Cắt Stake khi Đạo nhái Mô hình:** Tự động trừ tiền phạt bảo lãnh (slashing) khi điểm tương đồng cosine giữa các adapter vượt ngưỡng sao chép (> 0.98).

---

## 📌 Chính Sách Báo Cáo Lỗ Hổng Bảo Mật

Toàn bộ thông tin về lỗ hổng bảo mật, lỗi khai thác hay bất kỳ vấn đề an toàn nào **BẮT BUỘC phải gửi độc quyền qua Tin nhắn Trực tiếp (DM) trên Discord** cho đội ngũ phát triển/core team và **phải được mã hóa bằng PGP Public Key chính thức** ([`depeft_security_pubkey.asc`](./depeft_security_pubkey.asc)). Vui lòng tham khảo chi tiết tại [`SECURITY_VI.md`](./SECURITY_VI.md). Tuyệt đối KHÔNG tạo GitHub Issue công khai hay gửi email.

