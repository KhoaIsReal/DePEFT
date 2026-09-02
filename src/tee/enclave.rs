use crate::blockchain::types::AccountId;
use crate::crypto::AccountKeypair;
use crate::tee::types::{AttestationQuote, EnclaveMeasurement, TeeType};
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hardware TEE Sandbox Enclave generating verifiable remote attestation quotes.
#[derive(Clone)]
pub struct HardwareTeeEnclave {
    pub tee_type: TeeType,
    pub enclave_id: String,
    pub measurement: EnclaveMeasurement,
    platform_keypair: AccountKeypair,
}

impl HardwareTeeEnclave {
    /// Initialize a new Hardware TEE Enclave instance.
    pub fn new(tee_type: TeeType, enclave_name: impl Into<String>) -> Self {
        let name = enclave_name.into();

        // Deterministic measurement based on enclave code hash / name
        let mut mrenclave_hasher = Sha256::new();
        mrenclave_hasher.update(b"depeft_tee_validator_evaluator_v1.0.0");
        mrenclave_hasher.update(name.as_bytes());
        let mrenclave: [u8; 32] = mrenclave_hasher.finalize().into();

        let mut mrsigner_hasher = Sha256::new();
        mrsigner_hasher.update(b"depeft_foundation_enclave_authority_root_key");
        let mrsigner: [u8; 32] = mrsigner_hasher.finalize().into();

        let measurement = EnclaveMeasurement {
            mrenclave,
            mrsigner,
            isv_prod_id: 1,
            isv_svn: 1,
        };

        let platform_keypair = AccountKeypair::generate();

        Self {
            tee_type,
            enclave_id: name,
            measurement,
            platform_keypair,
        }
    }

    /// Construct a local simulator enclave.
    ///
    /// This is deliberately *not* a production trust root: its signing key is
    /// generated in process and must be explicitly registered by a test. Real
    /// deployments must obtain quotes from a vendor-backed attestation service.
    pub fn official(tee_type: TeeType) -> Self {
        Self::new(tee_type, "official-validator-enclave")
    }

    /// Get platform public key bytes.
    pub fn platform_public_key(&self) -> [u8; 32] {
        self.platform_keypair.public_key_bytes()
    }

    /// Generate an Attestation Quote cryptographically bound to the evaluation ranking.
    pub fn generate_quote(
        &self,
        task_id: u64,
        round: usize,
        ranking: &[AccountId],
    ) -> Result<AttestationQuote> {
        let report_data = AttestationQuote::compute_report_data(task_id, round, ranking);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Sign payload: (tee_type || mrenclave || mrsigner || report_data || timestamp)
        let mut quote_payload = Vec::new();
        quote_payload.push(self.tee_type as u8);
        quote_payload.extend_from_slice(&self.measurement.mrenclave);
        quote_payload.extend_from_slice(&self.measurement.mrsigner);
        quote_payload.extend_from_slice(&report_data);
        quote_payload.extend_from_slice(&timestamp.to_be_bytes());

        let quote_signature = self.platform_keypair.sign_message(&quote_payload);

        Ok(AttestationQuote {
            tee_type: self.tee_type,
            measurement: self.measurement.clone(),
            report_data,
            platform_public_key: self.platform_keypair.public_key_bytes(),
            quote_signature,
            timestamp,
        })
    }
}
