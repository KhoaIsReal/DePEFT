# Security audit ledger

Phạm vi: Rust node/P2P/storage/TEE/consensus, Python SDK, Solidity và deployment.
Mục chỉ được gạch khi source và regression test chứng minh được biện pháp bảo vệ.

## Đã kiểm chứng

- ~~Giả mạo transaction signature~~
- ~~Inner sender khác public key ký~~
- ~~Replay nonce~~
- ~~Duplicate commit và reveal~~
- ~~Ranking chứa miner lạ, thiếu hoặc trùng~~
- ~~Quote hết hạn hoặc có timestamp tương lai~~
- ~~Forged BFT vote signature~~
- ~~Duplicate validator trong validator set~~
- ~~P2P frame vượt giới hạn và slowloris read~~
- ~~SafeTensors offset vượt biên, NaN/Inf payload~~
- ~~CID traversal cơ bản~~
- ~~Vector DB NaN/sai dimension~~
- ~~Quantization block size bằng zero~~
- ~~Định tuyến P2P Circuit Relay và kết nối Dual-Stack IPv6 xuyên CGNAT~~
- ~~Bảo toàn toán học phân bổ bounty 3 tầng và cơ chế đốt token giảm phát động~~

## Cần vá / chưa chứng minh triệt để

- [critical] Trust root TEE giả lập hoặc private key nằm trong source.
- [critical] HTTP mutation không đi qua finalized BFT block/state root.
- [critical] State, escrow và finalized blocks không được persist atomically.
- [critical] Adapter bytes không được bind với `adapter_hash` khi evaluate/merge.
- [high] Model và dataset hash không được verify trước sử dụng.
- [high] Faucet Sybil, không có ngân sách toàn cục và không fail-closed production.
- [high] Upload CAS public: không auth/quota/rate-limit.
- [high] P2P connect API là SSRF/dial primitive.
- [high] CORS permissive trên API thay đổi state.
- [high] P2P plaintext/không authenticated, unbounded task/channel và peer-id takeover.
- [high] Phase/finalize thiếu consensus authority và height deadline runtime.
- [high] Python SDK không tương thích Ed25519/transaction envelope của Rust.
- [medium] Overflow/panic và allocation limit ở đường runtime.
- [medium] CAS symlink/TOCTOU và CID integrity không nhất quán với IPFS CID chuẩn.
- [medium] Solidity Top-K không check winner unique, round tuần tự/payout policy.
- [medium] ERC-20 approve race.
- [medium] Hardhat fallback private key công khai.
- [medium] Docker/IPFS RPC public, mutable `latest`, không TLS/mTLS.
- [design] Owner escrow tập trung; plagiarism không có economic enforcement; thiếu staking/slashing thật.

## Bằng chứng sau bản vá đầu tiên

- ~~Trust root TEE giả lập hoặc private key nằm trong source~~: verifier production không còn trust canonical key; không có root cấu hình sẽ fail closed.
- ~~Faucet và operator endpoint mặc định public~~: mặc định production không mount operator routes/faucet, CORS permissive đã bị bỏ. Endpoint chỉ được bật trong development hoặc sau mTLS reverse proxy.
- ~~CAS trả dữ liệu sai CID nội bộ~~: disk CAS từ chối object `bafy...` không hash đúng và không đọc non-regular file.
- ~~State chỉ nằm RAM, mất sau restart~~: SQLite WAL lưu snapshot canonical cùng state hash; HTTP, faucet và P2P rollback mutation nếu persist thất bại.

Các mục còn mở phải được giữ mở cho đến khi có test exploit/regression và implementation tương ứng.
