# 🔒 Security Policy & Threat Model

DePEFT implements a defense-in-depth security architecture designed to defend decentralized AI training against adversarial participants, malicious validators, sybil attacks, and hardware drift.

---

## 🛡️ Threat Model & Cryptographic Mitigations

| Threat Vector | Attack Scenario | Protocol Mitigation |
|---|---|---|
| **Adapter Front-Running & Plagiarism** | A malicious miner snoops on the P2P network, steals another miner's adapter weights, and claims the reward. | **2-Phase Commit-Reveal Scheme**: Miners commit $\text{SHA256}(\text{hash} \mathbin{\Vert} \text{salt})$ during Commit Phase. Revealed weights are checked against the Embedded Vector DB for $> 0.98$ cosine similarity duplication. |
| **Test Set Memorization & Overfitting** | Miners overfit their adapters to the validation benchmark to score artificially high. | **TEE Enclave Sealing**: The Private Test Set is strictly sealed inside hardware TEE Enclaves (Intel SGX / AMD SEV-SNP) and is never accessible to miners. |
| **Validator Evaluation Fraud & Collusion** | A corrupt validator reports arbitrary loss rankings to favor a colluding miner. | **Cryptographic Remote Attestation**: The App-Chain enforces valid `MRENCLAVE` measurement, platform signature verification, and exact binding of `report_data = SHA512(task || round || ranking)`. |
| **Floating-Point Non-Determinism Drift** | Heterogeneous GPU architectures (CUDA vs ROCm vs CPU) compute slightly different IEEE 754 floats, causing consensus forks. | **Relative Consensus (Borda Count)**: The blockchain aggregates ordinal rankings ($A > B > C$) rather than raw floating-point loss averages. |
| **Byzantine Double Voting / Equivocation** | A malicious validator signs two conflicting blocks/votes at the same height to fork the chain. | **On-Chain Slashing Engine**: Cryptographic equivocation proofs immediately slash validator stake and revoke voting power. |
| **Sybil Network Flooding** | An adversary spins up hundreds of nodes to exhaust network resources. | **Escrow Requirements & Deduplication**: Tasks require locked token bounty escrows; P2P messages use SHA-256 LRU deduplication. |

---

## 🔍 On-Chain TEE Verification Flow

```mermaid
sequenceDiagram
    participant Validator as Validator (TEE)
    participant Chain as App-Chain State Machine
    
    Validator->>Validator: Run Evaluation in SGX/SEV Enclave
    Validator->>Validator: Compute report_data = SHA512(task || round || ranking)
    Validator->>Validator: Hardware QE signs (MRENCLAVE || MRSIGNER || report_data)
    Validator->>Chain: SubmitEvaluation(ranking, quote)
    
    Note over Chain: 1. Check report_data == computed binding
    Note over Chain: 2. Check MRENCLAVE in approved whitelist
    Note over Chain: 3. Verify Hardware Public Key Signature
    alt Valid Attestation
        Chain->>Chain: Admit Evaluation to Borda Count Aggregation
    else Invalid / Tampered Quote
        Chain->>Chain: REJECT Transaction & Slash Validator
    end
```

---

## 📢 Vulnerability Disclosure

If you discover a security vulnerability in DePEFT (such as a cryptographic bypass, TEE attestation flaw, or consensus safety bug), please report it responsibly:

- **Email**: `security@depeft.network` (or core maintainers)
- Please include detailed reproduction steps, proof-of-concept code, and affected module paths.
- We will acknowledge receipt within 24 hours and coordinate a coordinated disclosure timeline.
