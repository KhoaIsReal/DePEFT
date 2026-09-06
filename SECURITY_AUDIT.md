# 🛡️ DePEFT Security Audit Ledger

**Scope:** Rust App-Chain Node, CometBFT Consensus, P2P Gossip & Circuit Relay, IPFS CAS Storage, Hardware TEE Verification, Machine Learning Engine (Candle QLoRA), Python SDK, and Smart Contracts.  
**Audit Principle:** Items are only struck through once both underlying source code and rigorous regression tests mathematically prove mitigation.

---

## ✅ Verified & Mitigated Controls (Proven by 45 Regression Tests)

- ~~**Transaction Signature Forgery:**~~ Ed25519 signature verification strictly enforced on every payload.
- ~~**Inner Sender Mismatch:**~~ Transactions rejected if inner `sender`/`client`/`miner` does not match the Ed25519 signing public key.
- ~~**Replay Attack via Nonces:**~~ Strict sequential nonce checking per account rejects replayed transactions.
- ~~**Duplicate Commit and Reveal:**~~ Double commitments and redundant reveals per miner/round rejected at state machine layer.
- ~~**Evaluation Ranking Malformation:**~~ Rankings rejected if containing foreign miners, missing revealed miners, or containing duplicates.
- ~~**Expired or Future-Dated TEE Quotes:**~~ Enclave attestation quotes verified against monotonic consensus clock; timestamps in the future or older than max drift threshold rejected.
- ~~**Forged BFT Vote Signatures:**~~ CometBFT prevotes/precommits strictly authenticated against registered validator set public keys.
- ~~**Duplicate Validators in Consensus Set:**~~ Validator set deduplication strictly enforced with stake weight validation.
- ~~**P2P Frame Limit & Slowloris Protection:**~~ Frame size caps (64MB) and timeout protections in codec prevent memory exhaustion.
- ~~**SafeTensors Malicious Header & Offset Bounds:**~~ Zero-copy safetensors parser rejects out-of-bounds byte offsets, negative sizes, and NaN/Inf weight values.
- ~~**Storage Path Traversal:**~~ Content-Addressable Storage (CAS) validates multihash format and rejects path delimiters (`..`, `/`, `\`).
- ~~**Vector DB Dimension & NaN Safety:**~~ Cosine similarity engine sanitizes vector inputs and rejects mismatched dimensions or non-finite floats.
- ~~**Quantization Zero Block Size:**~~ NF4 and INT4 quantization routines clamp and validate block parameters against zero-division panics.
- ~~**TEE Production Root of Trust Fail-Closed:**~~ Production nodes reject mock attestation keys; missing configured hardware roots fail-closed by default.
- ~~**Operator & Faucet Endpoints Isolation:**~~ Operator mutation routes and testnet faucet disabled by default in production mode; permissive CORS removed.
- ~~**Durable Persistence & WAL Atomicity:**~~ SQLite write-ahead logging (WAL) stores state snapshots and hashes atomically; mutations rollback if persistence fails.
- ~~**IPv6 Dual-Stack & NAT/CGNAT Traversal:**~~ Dual-stack `[::]` binding and authenticated P2P Circuit Relay prevent network isolation for residential miners behind symmetric NAT.
- ~~**Dynamic 3-Tier Bounty Split & Deflationary Burn:**~~ Exact mathematical conservation ($\sum \text{Balances} + \text{Burned} = \text{Total Bounty}$) verified under dynamic supply-demand elasticity.

---

## 🤖 Automated Zero-Cost Static Security Analysis & Verification

DePEFT codebase is continuously scanned by industry-standard static analysis engines:

| Audit Tool | Target Subsystem | Scope & Rules | Audit Status |
|---|---|---|---|
| **`cargo audit` (RustSec)** | Rust App-Chain & Dependencies | Scanned **380 crate dependencies** against the RustSec Advisory Database. | ✅ **0 Vulnerabilities** |
| **`cargo clippy`** | Rust Codebase (`src/`, `tests/`) | Strict `-D warnings` enforcement covering memory safety, deadlocks, and arithmetic panics. | ✅ **0 Warnings** |
| **`slither` (Trail of Bits)** | Smart Contracts (`contracts/`) | Scanned `DePeftToken.sol` & `DePeftEscrow.sol` with 102 detectors (Reentrancy, CEI, gas optimization). | ✅ **Clean (0 Critical / High / Medium)** |

Key Smart Contract hardening verified by Slither:
- **Checks-Effects-Interactions (CEI)** pattern applied across `DePeftEscrow.sol` deposits and refunds to neutralize reentrancy vectors.
- State variables marked as `constant` and `immutable` to minimize runtime bytecode size and eliminate EVM storage hijacking risks.
- Pinned to Solidity `0.8.26` compiler preventing known compiler-level optimizer bugs.

---

## 🔍 In-Progress Hardening & Engineering Roadmap

- [medium] **BFT Block State Root Inclusion:** Transition HTTP mutation directly into finalized CometBFT block proposals with Merkle state roots.
- [medium] **Adapter Hash Binding on Merge:** Ensure raw adapter bytes from IPFS are strictly checked against on-chain committed SHA-256 hash prior to ReLoRA weight fusion.
- [medium] **Pre-Execution Dataset & Model Hash Integrity:** Pre-fetch verification of base model weights against task specification hash.
- [low] **Rate-Limiting & Quota Throttling on Public RPC:** IPFS upload bandwidth limits and per-account rate limiting.
- [low] **Encrypted P2P Noise Protocol:** Upgrade raw TCP gossip channels to authenticated TLS/Noise protocol.
- [low] **Economic Slashing Enforcement for Plagiarism:** Automate token slash penalty deduction when cosine similarity between adapters exceeds theft threshold (> 0.98).

---

## 📌 Security Disclosure Policy

All security vulnerabilities, potential exploits, and bug disclosures **MUST be submitted exclusively via Direct Message (DM) on Discord** to the maintainers / core development team and **must be encrypted with our official PGP Public Key** ([`depeft_security_pubkey.asc`](./depeft_security_pubkey.asc)). For full instructions on responsible disclosure and key import, please refer to [`SECURITY.md`](./SECURITY.md). Do NOT open public issues or send emails.

