use crate::blockchain::types::AccountId;
use crate::tee::enclave::HardwareTeeEnclave;
use crate::tee::types::AttestationQuote;
use anyhow::{bail, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::collections::HashSet;

/// On-Chain TEE Remote Attestation Verifier.
/// Ensures all submitted validator rankings were produced inside genuine, authorized hardware enclaves.
/// Proactively detects simulation quotes, debug-mode enclaves, and spoofed measurements.
#[derive(Debug, Clone)]
pub struct OnChainTeeVerifier {
    pub approved_mrenclaves: HashSet<[u8; 32]>,
    pub approved_mrsigners: HashSet<[u8; 32]>,
    pub approved_platform_keys: HashSet<[u8; 32]>,
    /// Fingerprints of known software simulation enclaves used for anti-spoofing detection
    pub known_simulator_mrenclaves: HashSet<[u8; 32]>,
    pub known_simulator_keys: HashSet<[u8; 32]>,
    pub enforce_attestation: bool,
    /// If false (default in production), software simulations and debug enclaves are rejected
    pub allow_simulation: bool,
}

impl Default for OnChainTeeVerifier {
    fn default() -> Self {
        Self {
            approved_mrenclaves: HashSet::new(),
            approved_mrsigners: HashSet::new(),
            approved_platform_keys: HashSet::new(),
            known_simulator_mrenclaves: HardwareTeeEnclave::known_simulator_measurements(),
            known_simulator_keys: HashSet::new(),
            enforce_attestation: true,
            allow_simulation: false, // Strict production default: simulation forbidden
        }
    }
}

impl OnChainTeeVerifier {
    pub fn new(enforce_attestation: bool) -> Self {
        let mut v = Self::default();
        v.enforce_attestation = enforce_attestation;
        v
    }

    /// Whitelist an approved enclave code measurement.
    pub fn register_mrenclave(&mut self, mrenclave: [u8; 32]) {
        self.approved_mrenclaves.insert(mrenclave);
    }

    /// Whitelist an approved enclave author signing key.
    pub fn register_mrsigner(&mut self, mrsigner: [u8; 32]) {
        self.approved_mrsigners.insert(mrsigner);
    }

    /// Whitelist an approved hardware platform root public key (Root of Trust).
    pub fn register_platform_key(&mut self, platform_key: [u8; 32]) {
        self.approved_platform_keys.insert(platform_key);
    }

    /// Explicitly trust a quote source for tests and local simulators.
    pub fn trust_quote_source(&mut self, mrenclave: [u8; 32], platform_key: [u8; 32]) {
        self.allow_simulation = true;
        self.register_mrenclave(mrenclave);
        self.register_platform_key(platform_key);
    }

    /// Whitelist and register an official testnet simulator source
    pub fn trust_simulator_source(&mut self, mrenclave: [u8; 32], platform_key: [u8; 32]) {
        self.allow_simulation = true;
        self.register_mrenclave(mrenclave);
        self.register_platform_key(platform_key);
        self.known_simulator_mrenclaves.insert(mrenclave);
        self.known_simulator_keys.insert(platform_key);
    }

    /// Set whether software simulation quotes are accepted (testnet = true, mainnet = false)
    pub fn set_allow_simulation(&mut self, allow: bool) {
        self.allow_simulation = allow;
    }

    /// Verify an Attestation Quote on-chain before admitting a validator evaluation.
    pub fn verify_quote(
        &self,
        quote: &AttestationQuote,
        task_id: u64,
        round: usize,
        ranking: &[AccountId],
    ) -> Result<()> {
        // 1. Verify Report Data binding (attestation must commit to exact task, round, ranking)
        if !quote.verify_report_data(task_id, round, ranking) {
            bail!(
                "TEE Attestation Quote report_data mismatch: quote was not generated for task #{} round {} ranking {:?}",
                task_id,
                round,
                ranking
            );
        }

        // Development simulations can opt out explicitly. Production uses the
        // default (`enforce_attestation = true`) and never reaches this branch.
        if !self.enforce_attestation
            && self.approved_mrenclaves.is_empty()
            && self.approved_mrsigners.is_empty()
            && self.approved_platform_keys.is_empty()
        {
            return Ok(());
        }

        // 2. Anti-Spoofing & Simulation Detection (enforced on production or when roots are configured)
        if !self.allow_simulation {
            if quote.security_flags.is_simulation {
                bail!(
                    "Simulation TEE quote rejected: node is running in strict production mode with simulation disallowed"
                );
            }
            if quote.security_flags.debug_mode {
                bail!(
                    "Insecure TEE enclave rejected: debug mode is strictly forbidden on production network"
                );
            }
            if quote.security_flags.hardware_level < 2 {
                bail!(
                    "Rejected TEE quote: insufficient hardware security level ({})",
                    quote.security_flags.hardware_level
                );
            }
            // Anti-spoofing check: quote claims to be genuine hardware, but its measurement matches a known simulation template!
            if self.known_simulator_mrenclaves.contains(&quote.measurement.mrenclave) {
                bail!(
                    "Spoofed TEE detected: quote presented known simulator measurement {} as genuine hardware",
                    quote.measurement.mrenclave_hex()
                );
            }
            if self.known_simulator_keys.contains(&quote.platform_public_key) {
                bail!(
                    "Spoofed TEE detected: quote signed with known simulation platform key 0x{}",
                    hex::encode(quote.platform_public_key)
                );
            }
        }

        // 3. Verify Enclave Measurement against on-chain whitelist
        if self.enforce_attestation && self.approved_mrenclaves.is_empty() {
            bail!(
                "TEE attestation is required but no enclave measurement trust root is configured"
            );
        }
        if !self
            .approved_mrenclaves
            .contains(&quote.measurement.mrenclave)
            && !self
                .approved_mrsigners
                .contains(&quote.measurement.mrsigner)
        {
            bail!(
                "Unauthorized MRENCLAVE measurement: {} is not in approved on-chain enclave registry",
                quote.measurement.mrenclave_hex()
            );
        }

        // 4. Verify Hardware Platform Public Key against Root-of-Trust whitelist
        if self.enforce_attestation && self.approved_platform_keys.is_empty() {
            bail!("TEE attestation is required but no platform-key trust root is configured");
        }
        if !self
            .approved_platform_keys
            .contains(&quote.platform_public_key)
        {
            bail!(
                "Unauthorized TEE Platform Public Key: 0x{} is not signed or whitelisted by Hardware Root of Trust",
                hex::encode(quote.platform_public_key)
            );
        }

        // 5. Verify Quote Timestamp Freshness and Future Drift
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        const MAX_QUOTE_AGE_SECONDS: u64 = 86400; // 24 hours max age for evaluation quote
        if now.saturating_sub(quote.timestamp) > MAX_QUOTE_AGE_SECONDS {
            bail!("TEE Attestation Quote is stale: quote timestamp exceeds maximum age of 24h");
        }
        if quote.timestamp > now + 60 {
            bail!("TEE Attestation Quote timestamp is in the future");
        }

        // 6. Cryptographically verify Hardware Platform Quote Signature
        let verifying_key = VerifyingKey::from_bytes(&quote.platform_public_key)
            .map_err(|e| anyhow::anyhow!("Invalid TEE platform public key: {}", e))?;

        if quote.quote_signature.len() != 64 {
            bail!("Invalid quote signature length: expected 64 bytes");
        }

        let sig_bytes: [u8; 64] = quote.quote_signature.as_slice().try_into()?;
        let signature = Signature::from_bytes(&sig_bytes);

        // Reconstruct signed payload using canonical helper
        let quote_payload = quote.payload();

        verifying_key
            .verify(&quote_payload, &signature)
            .map_err(|e| anyhow::anyhow!("Hardware TEE Attestation Quote cryptographic signature verification failed: {}", e))?;

        Ok(())
    }
}
