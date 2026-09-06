# ⚡ DePEFT Fast Q&A : The Crypto Trader, DePIN Investor & Node Operator Guide

> [!WARNING]
> ### 🚨 ANTI-SCAM NOTICE / MAINNET WARNING
> **DePEFT IS CURRENTLY IN TESTNET PHASE.** We have **NOT** launched any token, contract, or liquidity pool on Mainnet (Ethereum, Base, Solana, BNB Chain, or any DEX/CEX). Any token claiming to be $DEPEFT on mainnet is an unauthorized counterfeit scam. Always verify official updates directly with the core development team.

---

## 🧭 Executive Summary for Crypto Traders & Investors
- **Sector:** Decentralized AI (DeAI) / Decentralized Physical Infrastructure (DePIN).
- **Core Value Proposition:** Slashing AI fine-tuning costs by 60%–80% using consumer GPU swarms (QLoRA) and multi-round continuous weight evolution (ReLoRA), verified cryptographically inside hardware TEE enclaves (Intel SGX / AMD SEV).
- **Token Standard & Architecture:** Native Application-Specific Blockchain (Rust L1) powered by CometBFT 2-Phase Commit Consensus, bridged via ERC-20 / SPL utility tokens for secondary market liquidity.

---

## 💰 1. Tokenomics, Supply & Value Accrual

### Q1: Is there a token burn mechanism? How does $DEPEFT achieve deflation?
**A:** Yes. DePEFT implements an on-chain **Dynamic Deflationary Burn Mechanism**:
$$\text{BurnRate} = \text{clamp}\left(0.005 + (N_{\text{tasks}} - 1) \times 0.015,\ 0.005,\ 0.10\right)$$
- When client fine-tuning demand is high ($\ge 8$ active tasks), **up to 10% of every task bounty pool is permanently burned** from circulating supply.
- When demand is scarce (e.g. 1 active task), the burn rate drops down to $\approx 0.5\%$ to maximize yields for participating miners and node operators.
- **Investment takeaway:** As more AI startups, clinics, and enterprises fine-tune models on DePEFT, circulating token velocity accelerates into permanent token destruction.

---

### Q2: How does the bounty pool split? Won't miners just dump all their earnings?
**A:** DePEFT solves the classic DePIN "death spiral" using a **Three-Tier Elastic Supply-Demand Model**:
1. **Tier 1: Storage & Network Gateway Nodes (5%–15%):** Scales elastically with IPFS network throughput and relayed adapter traffic.
2. **Tier 2: Hardware TEE Validators (15%–45%):** Dynamically scales upward when high-security TEE server supply is scarce relative to miner supply ($\frac{\text{Miners}}{\text{Validators}}$).
3. **Tier 3: Competitive Miners (40%–80%):** Disbursed only to verified top performers via Top-K exponential decay or Winner-Takes-All policies.

Furthermore, **validators must stake collateral** to participate, locking substantial capital away from secondary market sell pressure.

---

### Q3: Who buys the token? Where does real organic demand come from?
**A:** Real utility demand is driven by 5 distinct economic actors:
1. **AI Enterprises & Startups (Clients):** Purchase $DEPEFT to fund training bounties in on-chain escrow.
2. **TEE Validators & Consensus Nodes:** Buy and stake $DEPEFT as collateral to secure consensus and earn evaluation fees.
3. **Storage Gateways:** Stake tokens to guarantee high availability and data persistence for model weight checkpoints.
4. **Model Consumers & Inference Callers:** Pay royalty tokens to download mature fine-tuned model checkpoints (e.g. specialized medical, financial, or legal LLMs).
5. **Delegators & Yield Farmers:** Stake tokens behind high-reputation validators to earn passive network rewards.

---

## 🔒 2. Security, Consensus & Anti-Cheat Guarantees

### Q4: How do you prevent miners from stealing each other's weights (Front-Running / Plagiarism)?
**A:** DePEFT enforces a strict **2-Phase Cryptographic Commit-Reveal Scheme**:
1. **Commit Phase:** Miners submit only a hidden cryptographic digest:  
   $$\text{commit\_hash} = \text{SHA256}(\text{adapter\_hash} \mathbin{\Vert} \text{salt})$$
2. **Reveal Phase:** Only after the deadline closes, miners upload their `.safetensors` files and reveal their secret salt.
3. **Embedded Vector Database:** Before any reward is disbursed, an on-chain Vector Database calculates cosine similarity signatures across all candidate models. Any plagiarized weights ($\text{similarity} > 0.98$) are detected and disqualified immediately.

---

### Q5: How do you stop malicious validators from favoring their own miner friends?
**A:** Multiple independent cryptographic and algorithmic safeguards exist:
1. **Hardware TEE Remote Attestation:** Evaluations run inside an isolated enclave (Intel SGX / AMD SEV-SNP). The processor's Quoting Enclave cryptographically signs an `AttestationQuote` binding the MRENCLAVE measurement and evaluation ranking hash (`report_data`).
2. **Relative Consensus (Borda Count Aggregation):** Rather than trusting raw floating-point loss (which varies across GPU architectures), the chain aggregates ordinal rankings across independent validators.
3. **Byzantine Slashing Engine:** Any validator committing equivocation (signing contradictory votes or quotes at the same height) is slashed on-chain immediately.

---

## ⚡ 3. Technology & Network Infrastructure

### Q6: Can residential miners mine from home if they are stuck behind CGNAT (No Public IPv4)?
**A:** **Yes!** DePEFT is designed specifically for decentralized grassroots participation:
- **Native Dual-Stack IPv6:** Residential nodes with IPv6 connect peer-to-peer without NAT restrictions.
- **P2P Circuit Relay (`RelayForward` / `RelayPayload`):** For miners stuck behind restrictive IPv4 CGNAT, traffic routes seamlessly through public Relay Nodes via outbound TCP connections without requiring manual router port forwarding.

---

### Q7: What machine learning framework powers the nodes?
**A:** DePEFT runs on a zero-overhead pure Rust deep learning engine:
- Built on **Hugging Face Candle** with native QLoRA (NF4 4-bit and INT4 block quantization).
- Native support for CUDA (NVIDIA), ROCm (AMD), Metal (Apple Silicon), and AVX-512 CPU fallbacks.
- **ReLoRA multi-round weight merge:** Continuously evolves base LLM checkpoints ($W_{N+1} = W_N + \Delta W^*$) without parameter explosion or VRAM leakage.

---

## 🚀 4. Roadmap, Trading & Exchange Listings

### Q8: When Mainnet? How can I trade or provide liquidity?
**A:** 
1. **Current Phase:** Active Testnet with public node daemon, faucet, CLI, and real QLoRA training runs.
2. **Mainnet Launch Protocol:**
   - Formal audit completion and Genesis Block initialization.
   - Deployment of audited ERC-20 / SPL Bridge contracts on major EVM / Solana chains.
   - Initial Liquidity Pool creation on top decentralized exchanges (e.g. Uniswap V3, Raydium) with locked LP tokens.
   - Applications for CoinGecko, CoinMarketCap, and Tier-3/Tier-2 centralized exchanges (CEX).

---

### Q9: Where should security researchers or users report bugs?
**A:** All vulnerability disclosures, security inquiries, and bug reports **MUST be sent exclusively via Direct Message (DM) on Discord** to the maintainers / core development team. Please do NOT post public issues on GitHub or send emails. See [`SECURITY.md`](./SECURITY.md) for full disclosure terms.
